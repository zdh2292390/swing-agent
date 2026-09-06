"""运行入口：拉最近 N 天的 bar -> 策略信号 -> (可选) Claude 复核 -> 风控 -> 调仓下单 -> 写日志。

用法：
  python run.py --once            # 立刻跑一次（市场未开盘会跳过，加 --force 忽略）
  python run.py                   # 常驻。INTERVAL=1h：交易日 10:31~15:31 每小时；INTERVAL=1d：交易日 9:35 一次

信号只用"已走完"的 bar（见 data.drop_incomplete_last_bar），和回测保持一致；
撮合价格用最新价（本地模拟盘取含未完成 bar 的最新收盘，Alpaca 取实时成交价）。
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tradebot.config import Settings, load_settings
from tradebot.data import bars_per_day, drop_incomplete_last_bar, fetch_bars, latest_prices
from tradebot.execution import Broker, make_broker
from tradebot.logging_utils import append_decision, setup_logging
from tradebot.portfolio import select_targets
from tradebot.iv import record_snapshots
from tradebot.risk import PortfolioState, RiskDecision, RiskLimits, RiskManager
from tradebot.strategies import as_portfolio, get_strategy

NY = ZoneInfo("America/New_York")


# ---------- 跨次运行的风控状态（当日起始净值、历史高点） ----------
def load_risk_state(path: Path, equity: float, today: str) -> dict:
    state = json.loads(path.read_text()) if path.exists() else {}
    if state.get("date") != today:
        state = {"date": today, "day_start_equity": equity, "peak_equity": max(equity, state.get("peak_equity", equity))}
    state["peak_equity"] = max(state.get("peak_equity", equity), equity)
    path.write_text(json.dumps(state, indent=2))
    return state


# ---------- 把目标权重换成订单 ----------
def rebalance(broker: Broker, approved: dict[str, float], equity: float, prices: dict[str, float], min_trade_value: float, log):
    positions = broker.positions()
    plan = []
    for sym, w in approved.items():
        px = prices[sym]
        target_value = w * equity
        current_value = positions[sym].market_value if sym in positions else 0.0
        delta = target_value - current_value
        if abs(delta) < min_trade_value:
            continue
        qty = math.floor(abs(delta) / px)
        if qty < 1:
            continue
        side = "buy" if delta > 0 else "sell"
        if side == "sell":
            qty = min(qty, math.floor(positions[sym].qty)) if sym in positions else 0
            if qty < 1:
                continue
        plan.append((side, sym, qty))

    for sym, pos in positions.items():  # 不在标的池里的持仓全部卖出
        if sym not in approved and math.floor(pos.qty) >= 1:
            plan.append(("sell", sym, math.floor(pos.qty)))
            log.warning("%s 已不在标的池，清仓", sym)

    plan.sort(key=lambda x: 0 if x[0] == "sell" else 1)  # 先卖后买
    results = []
    for side, sym, qty in plan:
        try:
            r = broker.submit_market_order(sym, qty, side)
            log.info("ORDER %s %s x%d @~%.2f -> %s", side.upper(), sym, qty, r.est_price, r.status)
            results.append(r.__dict__)
        except Exception as e:
            log.error("ORDER FAILED %s %s x%d: %s", side, sym, qty, e)
            results.append({"symbol": sym, "side": side, "qty": qty, "status": "error", "error": str(e)})
    return results


def run_once(settings: Settings, broker: Broker, strategy, risk: RiskManager, advisor, log, force: bool = False) -> dict:
    now = datetime.now(NY)
    if not force and not broker.is_market_open():
        log.info("市场未开盘，跳过（--force 可忽略）")
        return {"skipped": "market_closed", "time": now.isoformat()}

    symbols = settings.tradable_symbols  # 海外上市的只监控，不进交易
    interval = settings.interval
    pstrat = as_portfolio(strategy)  # 单标的策略自动包成等权组合
    # 拉够预热长度：日历天 ≈ bar 数 / 每天 bar 数 * 1.5（周末节假日），再留余量
    need_days = max(settings.lookback_days, math.ceil(pstrat.warmup_bars() / bars_per_day(interval) * 1.5) + 10)
    raw = fetch_bars(symbols, interval, need_days, cache_dir=None)
    data = {s: drop_incomplete_last_bar(df, interval) for s, df in raw.items()}

    stray = [s for s in broker.positions() if s not in symbols]  # 已从标的池移除但还持有的
    if settings.mode == "local":
        prices = latest_prices(raw)  # 含未完成 bar 的最新收盘，作为撮合价
        if stray:
            prices.update(latest_prices(fetch_bars(stray, interval, 10, cache_dir=None)))
        broker.set_prices(prices)
    else:
        prices = {s: broker.last_price(s) for s in symbols + stray}

    # 1. 策略目标权重：组合级策略一次看全部标的，取最后两根完整 bar
    weights_df = pstrat.target_weights(data)
    last = weights_df.iloc[-1].reindex(symbols).fillna(0.0)
    prev = (weights_df.iloc[-2] if len(weights_df) > 1 else last * 0).reindex(symbols).fillna(0.0)
    signals = {s: float(last[s]) for s in symbols}
    # fresh：空仓起步只跟新入场，不追策略的老仓位；align：直接对齐策略权重
    raw_targets, entry_notes = select_targets(last, prev, set(broker.positions()), settings.entry_mode)
    for n in entry_notes:
        log.info("ENTRY %s", n)

    # 2. Claude 复核（可选，只能否决）
    agent_report, agent_notes, targets = None, [], dict(raw_targets)
    if advisor is not None:
        try:
            agent_report = advisor.review(data, signals)
            targets, agent_notes = advisor.apply(agent_report, raw_targets)
            for n in agent_notes:
                log.info("AGENT %s", n)
        except Exception as e:
            log.error("AGENT 调用失败，沿用策略信号: %s", e)
            agent_notes = [f"agent_error: {e}"]

    # 3. 风控
    acct = broker.account()
    today = now.strftime("%Y-%m-%d")
    rs = load_risk_state(settings.log_dir / "risk_state.json", acct.equity, today)
    state = PortfolioState(
        equity=acct.equity,
        cash=acct.cash,
        positions={s: p.market_value for s, p in broker.positions().items()},
        day_start_equity=rs["day_start_equity"],
        peak_equity=rs["peak_equity"],
    )
    decision: RiskDecision = risk.approve(targets, state)
    for r in decision.reasons:
        log.warning("RISK %s", r)

    # 4. 下单
    orders = rebalance(broker, decision.approved, acct.equity, prices, settings.min_trade_value, log)

    record = {
        "time": now.isoformat(),
        "interval": interval,
        "last_bar": max(df.index[-1] for df in data.values()).isoformat(),
        "broker": broker.name,
        "strategy": strategy.describe(),
        "equity_before": acct.equity,
        "cash_before": acct.cash,
        "prices": prices,
        "signals": signals,
        "entry_mode": settings.entry_mode,
        "entry_notes": entry_notes,
        "raw_targets": raw_targets,
        "agent_report": agent_report.model_dump() if agent_report else None,
        "agent_notes": agent_notes,
        "risk_reasons": decision.reasons,
        "halted": decision.halted,
        "approved_targets": decision.approved,
        "orders": orders,
    }
    append_decision(settings.log_dir, record)

    # 5. 记一条今天的 IV30 / HV30（免费源没有 IV 历史，只能自己攒；失败不影响交易）
    try:
        hist = record_snapshots(symbols, settings.cache_dir, {s: data[s]["close"] for s in symbols if s in data})
        log.info("IV 快照已记录：%d 条历史，%d 只标的", len(hist), hist["symbol"].nunique() if len(hist) else 0)
    except Exception as e:
        log.warning("IV 快照记录失败: %s", e)
    log.info(
        "DONE interval=%s last_bar=%s equity=%.2f long=%s orders=%d",
        interval,
        record["last_bar"],
        acct.equity,
        [s for s, v in decision.approved.items() if v > 0],
        len(orders),
    )
    return record


def build(settings: Settings, log):
    broker = make_broker(settings)
    strategy = get_strategy(settings.strategy, **settings.strategy_params)
    risk = RiskManager(
        RiskLimits(
            max_position_pct=settings.max_position_pct,
            max_gross_exposure=settings.max_gross_exposure,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
        )
    )
    advisor = None
    if settings.agent_enabled:
        from tradebot.agent.advisor import ClaudeAdvisor

        advisor = ClaudeAdvisor(model=settings.agent_model)
        log.info("Claude 决策层已启用: %s", settings.agent_model)
    log.info("broker=%s interval=%s strategy=%s entry_mode=%s watchlist=%s", broker.name, settings.interval, strategy.describe(), settings.entry_mode, settings.watchlist.describe())
    if settings.monitor_only_symbols:
        log.info("仅监控、不交易的海外标的: %s", settings.monitor_only_symbols)
    return broker, strategy, risk, advisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只跑一次")
    ap.add_argument("--force", action="store_true", help="忽略开盘时间检查")
    args = ap.parse_args()

    settings = load_settings()
    log = setup_logging(settings.log_dir)
    if settings.mode == "alpaca" and not settings.alpaca_paper:
        log.warning("!!! 当前是 Alpaca 实盘模式 !!!")
    broker, strategy, risk, advisor = build(settings, log)

    if args.once:
        run_once(settings, broker, strategy, risk, advisor, log, force=args.force)
        return

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    def job():
        try:
            run_once(settings, broker, strategy, risk, advisor, log, force=args.force)
        except Exception:
            log.error("run_once 异常:\n%s", traceback.format_exc())

    sched = BlockingScheduler(timezone=NY)
    if settings.interval == "1h":
        # 小时 bar 在 :30 收盘，:31 运行拿到刚走完的 bar
        sched.add_job(job, CronTrigger(day_of_week="mon-fri", hour="10-15", minute=31, timezone=NY))
        log.info("调度器启动：周一至周五 10:31-15:31（纽约时间）每小时一次，Ctrl+C 退出")
    else:
        # 日线：开盘后 9:35 用昨天的完整日 bar 决策，对应回测"下一根开盘成交"
        sched.add_job(job, CronTrigger(day_of_week="mon-fri", hour=9, minute=35, timezone=NY))
        log.info("调度器启动：周一至周五 9:35（纽约时间）一次，Ctrl+C 退出")
    sched.start()


if __name__ == "__main__":
    main()
