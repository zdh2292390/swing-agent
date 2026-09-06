"""回测命令行。

  python backtest.py                                          # .env 里的周期、策略、标的
  python backtest.py --interval 1h --days 729                 # 小时线（最多 729 天）
  python backtest.py --symbols @mag7,BRK.B --params fast=10,slow=40
  python backtest.py --fill close                             # 按信号 bar 收盘成交的乐观假设做对比
  python backtest.py --refresh                                # 忽略缓存重新下载
"""
from __future__ import annotations

import argparse
from datetime import datetime

from tradebot.backtest import format_metrics, run_backtest, slice_result
from tradebot.config import DEFAULT_BACKTEST_DAYS, load_settings, parse_params
from tradebot.data import fetch_bars
from tradebot.strategies import BuyAndHold, get_strategy
from tradebot.universe import expand, is_us_listed


def main():
    s = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=s.strategy)
    ap.add_argument("--params", default="", help="例如 fast=20,slow=50")
    ap.add_argument("--symbols", default=None, help="逗号分隔，可用 @板块名（内置组或 watchlist 里的板块）；默认用当前标的池")
    ap.add_argument("--interval", default=s.interval, choices=["1d", "1h"])
    ap.add_argument("--days", type=int, default=None, help="回溯天数；默认日线 5000、小时线 729")
    ap.add_argument("--fill", default="next_open", choices=["next_open", "close"])
    ap.add_argument("--start", default=None, help="只统计这一天之后（YYYY-MM-DD），信号仍用全部历史算，用于样本外检验")
    ap.add_argument("--end", default=None, help="只统计这一天之前（YYYY-MM-DD）")
    ap.add_argument("--all", action="store_true", help="把海外上市（仅监控）的标的也放进回测；默认只回测美股")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    symbols = expand(args.symbols, s.watchlist.groups) if args.symbols else s.symbols
    if not args.all:
        skipped = [x for x in symbols if not is_us_listed(x)]
        symbols = [x for x in symbols if is_us_listed(x)]
        if skipped:
            print(f"跳过海外上市（仅监控）的标的: {', '.join(skipped)}（加 --all 可包含）")
    days = args.days or (s.backtest_days if args.interval == s.interval else DEFAULT_BACKTEST_DAYS[args.interval])
    params = {**s.strategy_params, **parse_params(args.params)} if args.strategy == s.strategy else parse_params(args.params)
    strategy = get_strategy(args.strategy, **params)

    print(f"下载/读取 {len(symbols)} 个标的 {args.interval} 最近 {days} 天 ...")
    data = fetch_bars(symbols, args.interval, days, cache_dir=s.cache_dir, max_cache_age_hours=0 if args.refresh else 12)
    for sym, df in data.items():
        print(f"  {sym:6s} {len(df):5d} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d %H:%M}")

    common = dict(initial_cash=s.initial_cash, commission_bps=s.commission_bps, slippage_bps=s.slippage_bps, fill=args.fill)
    res = run_backtest(data, strategy, max_position_pct=s.max_position_pct, **common)
    bench = run_backtest(data, BuyAndHold(), **common)
    if args.start or args.end:
        res, bench = slice_result(res, args.start, args.end, s.initial_cash), slice_result(bench, args.start, args.end, s.initial_cash)
        print(f"区间: {res.equity.index[0]:%Y-%m-%d} -> {res.equity.index[-1]:%Y-%m-%d}")

    print(f"\n周期: {args.interval}   成交假设: {args.fill}   成本: {s.commission_bps + s.slippage_bps:.1f} bps/边   单标的上限: {s.max_position_pct:.0%}\n")
    print(format_metrics({res.strategy: res.metrics, "buy_and_hold": bench.metrics}))

    ts = res.trade_stats
    if ts.get("n_trades"):
        print(
            f"\n交易: {ts['n_trades']} 笔已平仓, {ts.get('open_trades', 0)} 笔持有中 | 胜率 {ts['win_rate']:.1%} | "
            f"平均 {ts['avg_ret']:+.2%} 中位 {ts['median_ret']:+.2%} | 均赢 {ts['avg_win']:+.2%} 均亏 {ts['avg_loss']:+.2%} "
            f"盈亏比 {ts['payoff']:.2f} | 平均持有 {ts['avg_hold_bars']:.0f} bar | 最好 {ts['best']:+.1%} 最差 {ts['worst']:+.1%}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = s.log_dir / f"backtest_{args.strategy}_{args.interval}_{stamp}.csv"
    frame = res.equity.rename("strategy").to_frame().join(bench.equity.rename("buy_and_hold"))
    frame.to_csv(out_csv)
    print(f"\n净值曲线已保存: {out_csv}")
    if len(res.trades):
        trades_csv = out_csv.with_name(out_csv.stem + "_trades.csv")
        res.trades.to_csv(trades_csv, index=False)
        print(f"逐笔交易已保存: {trades_csv}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        frame.plot(ax=axes[0], title=f"{res.strategy} vs buy_and_hold ({args.interval}, {args.fill})")
        axes[0].set_ylabel("equity")
        res.exposure.plot(ax=axes[1], color="gray")
        axes[1].set_ylabel("exposure")
        out_png = out_csv.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        print(f"图已保存: {out_png}")


if __name__ == "__main__":
    main()
