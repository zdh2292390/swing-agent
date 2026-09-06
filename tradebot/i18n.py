"""面板的中英文切换与主题切换。

- 界面文字在代码里全部是中文，英文模式下用字典做“显示层翻译”：
  控件标签 / 页签 / 段标题 / 表格列名 / 表格里的状态字 / 图例，都在渲染前经过 tr()。
  没有收进字典的长段说明文字保持中文。
- 主题：Streamlit 没有页面内切主题的官方接口，这里用 st._config.set_option 改运行时配置再 rerun，
  对单人使用足够；预设写在 THEMES 里。
- 偏好存在项目根目录 ui_prefs.json，重启面板后仍然生效。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_LANG = "zh"

THEMES = {
    # 每套主题给主区和侧边栏各一组颜色；侧边栏在 Streamlit 里是独立的 [theme.sidebar] 配置，不跟主区联动，必须一起改
    "浅色": {"theme": {"base": "light", "primaryColor": "#2563eb", "backgroundColor": "#f8fafc", "secondaryBackgroundColor": "#ffffff", "textColor": "#0f172a",
                       "borderColor": "#e2e8f0", "dataframeBorderColor": "#e2e8f0", "dataframeHeaderBackgroundColor": "#f1f5f9", "linkColor": "#2563eb"},
             "sidebar": {"backgroundColor": "#ffffff", "secondaryBackgroundColor": "#f1f5f9", "textColor": "#0f172a", "borderColor": "#e2e8f0"}},
    "深色": {"theme": {"base": "dark", "primaryColor": "#60a5fa", "backgroundColor": "#0f1117", "secondaryBackgroundColor": "#1a1d27", "textColor": "#e5e7eb",
                       "borderColor": "#2a2f3d", "dataframeBorderColor": "#2a2f3d", "dataframeHeaderBackgroundColor": "#20242f", "linkColor": "#93c5fd"},
             "sidebar": {"backgroundColor": "#15181f", "secondaryBackgroundColor": "#1f2330", "textColor": "#e5e7eb", "borderColor": "#2a2f3d"}},
    "暗蓝": {"theme": {"base": "dark", "primaryColor": "#38bdf8", "backgroundColor": "#0b1220", "secondaryBackgroundColor": "#111a2e", "textColor": "#e2e8f0",
                       "borderColor": "#1f2a44", "dataframeBorderColor": "#1f2a44", "dataframeHeaderBackgroundColor": "#152038", "linkColor": "#7dd3fc"},
             "sidebar": {"backgroundColor": "#0e162a", "secondaryBackgroundColor": "#172342", "textColor": "#e2e8f0", "borderColor": "#1f2a44"}},
    "米色": {"theme": {"base": "light", "primaryColor": "#b45309", "backgroundColor": "#f6f1e7", "secondaryBackgroundColor": "#fffaf0", "textColor": "#292524",
                       "borderColor": "#e7dcc8", "dataframeBorderColor": "#e7dcc8", "dataframeHeaderBackgroundColor": "#efe6d4", "linkColor": "#9a3412"},
             "sidebar": {"backgroundColor": "#fbf6ec", "secondaryBackgroundColor": "#efe6d4", "textColor": "#292524", "borderColor": "#e7dcc8"}},
}


def apply_theme(st, name: str) -> None:
    """把主题写进运行时配置：主区 theme.* 与侧边栏 theme.sidebar.* 一起改。"""
    spec = THEMES[name]
    for k, v in spec["theme"].items():
        try:
            st._config.set_option(f"theme.{k}", v)
        except Exception:
            pass
    for k, v in spec["sidebar"].items():
        try:
            st._config.set_option(f"theme.sidebar.{k}", v)
        except Exception:
            pass


def is_dark(name: str) -> bool:
    return THEMES.get(name, THEMES["浅色"])["theme"]["base"] == "dark"


THEME_NAMES_EN = {"浅色": "Light", "深色": "Dark", "暗蓝": "Navy", "米色": "Sepia"}


def load_prefs(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_prefs(path: Path, prefs: dict) -> None:
    Path(path).write_text(json.dumps(prefs, ensure_ascii=False, indent=1), encoding="utf-8")


def set_lang(lang: str) -> None:
    global _LANG
    _LANG = "en" if lang == "en" else "zh"


def get_lang() -> str:
    return _LANG


# ---------------------------------------------------------------- 字典
T: dict[str, str] = {
    # 页签
    "信号": "Signals", "策略仓位": "Strategy Positions", "大盘宏观": "Market & Macro", "板块轮动": "Sector Rotation", "末日期权": "0DTE Options",
    "底部确认": "Bottom Check", "回测": "Backtest", "信号验证": "Signal Validation", "模拟盘": "Paper Account", "决策日志": "Decision Log",
    "行情": "Charts", "说明": "Help", "催化剂": "Catalysts", "基本面": "Fundamentals", "走势": "Chart", "新闻原文": "News",
    # 页头 / 侧边栏
    "美股 swing · 只做多": "US equities · swing · long only", "市场": "Market", "开盘中": "open", "休市": "closed", "纽约": "NY", "模式": "Mode",
    "周期": "Interval", "策略": "Strategy", "起步": "Entry mode", "只跟新入场": "fresh entries only", "对齐策略": "align to strategy",
    "对齐策略仓位": "align to strategy positions", "大盘过滤": "Regime filter", "Claude 复核": "Claude review", "开": "on", "关": "off",
    "刷新行情缓存": "Refresh market data", "自动刷新": "Auto refresh", "配置（来自 .env）": "Config (from .env)", "编辑标的池": "Edit watchlist",
    "已重新下载行情": "Market data refreshed", "语言": "Language", "主题": "Theme",
    "放行": "allowing", "拦截新仓": "blocking new entries", "实盘策略未启用": "disabled on live strategy",
    # 信号页
    "重新计算": "Recompute", "信号怎么读（完整规则见最后的“说明”页签）": "How to read the signals (full rules in the Help tab)",
    "今日入场信号": "Today's entry signals", "只用走完的 bar": "completed bars only", "截至": "as of", "收盘": "close",
    "最近几天每天的信号": "Signals day by day", "显示最近几根 bar": "Show last N bars", "含出场条件": "include exit conditions", "视图": "View",
    "按日期": "By date", "按标的": "By symbol", "板块": "Sector", "只看有信号": "Signals only", "显示出场条件": "Show exit conditions",
    "排序": "Sort by", "降序": "Descending", "更多列": "More columns", "点任意一行打开该标的的详情：催化剂、基本面、走势、新闻。": "Click any row to open the symbol's details: catalysts, fundamentals, chart, news.",
    "标的池是空的，先在侧边栏“编辑标的池”里添加。": "The watchlist is empty. Add symbols in the sidebar.",
    "只": "names", "多头": "bull", "有信号": "with signals", "无": "none",
    # 策略名
    "趋势回调": "Trend Pullback", "突破": "Breakout", "财报动量": "Earnings Momentum", "超跌反弹": "Oversold Bounce", "底部回升": "Bottom Recovery",
    "相对强度": "Relative Strength", "52周新高": "52w High", "主升浪": "Main Wave", "均线交叉": "MA Cross", "满仓": "Buy & Hold",
    "（实盘）": " (live)", "实盘策略": "live strategy",
    # 表格列
    "标的": "Symbol", "组": "Group", "现价": "Close", "最新价": "Last", "盘中%": "Intraday %", "涨跌1日": "1d %", "涨跌5日": "5d %", "涨跌20日": "20d %",
    "涨跌1小时": "1h %", "涨跌5小时": "5h %", "涨跌20小时": "20h %", "趋势": "Trend", "阶段": "Stage", "距55高": "vs 55-bar high", "ATR%": "ATR %",
    "2月最低": "2m low", "2月最高": "2m high", "5月最低": "5m low", "5月最高": "5m high", "财报": "Earnings", "距财报天": "Days to earnings", "共振": "Confluence",
    "权重": "Weight", "入场日": "Entry date", "持有bar": "Bars held", "入场价": "Entry px", "浮动%": "Open P&L %", "退出日": "Exit date", "区间%": "Return %", "分数": "Score",
    "名称": "Name", "代码": "Ticker", "类别": "Type", "最新": "Last", "距52周高": "vs 52w high", "日期": "Date", "事件": "Event", "影响": "Impact", "备注": "Note",
    "距今天": "Days away", "星期": "Weekday", "盘前/盘后": "Pre/Post", "分类": "Category", "象限": "Quadrant", "象限变化": "Quadrant change",
    "RS比率": "RS ratio", "RS动量": "RS momentum", "1周%": "1w %", "1月%": "1m %", "3月%": "3m %", "6月%": "6m %", "1周超额": "1w excess", "1月超额": "1m excess",
    "3月超额": "3m excess", "6月超额": "6m excess", "3月超额排名": "3m excess rank", "距52周高%": "vs 52w high %", "成员": "Members", ">SMA50%": "> SMA50 %",
    ">SMA200%": "> SMA200 %", "RSI>50%": "RSI > 50 %", "平均20日%": "Avg 20d %", "平均距52周高%": "Avg vs 52w high %",
    "排名": "Rank", "到期": "Expiry", "交易日": "Trading days", "日频到期": "Daily expiries", "平值": "ATM strike", "跨式": "Straddle", "隐含波动%": "Implied move %",
    "历史中位%": "Hist. median %", "隐含/历史": "Implied / hist.", "区间内概率": "P(inside range)", "突破概率": "P(breakout)", "平时区间概率": "P(inside), normal days",
    "样本": "Samples", "事件分布": "Event dist.", "事件日中位%": "Event-day median %", "事件样本": "Event samples", "ATM持仓量": "ATM OI", "价差%": "Spread %",
    "流动性": "Liquidity", "月数": "Months", "天数": "Days", "行权": "Strike", "权利金": "Premium", "杠杆": "Leverage", "RV%": "RV %", "IV/RV": "IV / RV",
    "时间价值%": "Time value %", "年化成本%": "Annual cost %", "盈亏平衡": "Breakeven", "需涨%": "Needed move %", "隐含胜率": "Implied P(win)", "隐含ITM": "Implied P(ITM)",
    "历史胜率": "Hist. P(win)", "历史上涨率": "Hist. P(up)", "历史中位涨%": "Hist. median %", "内在": "Intrinsic", "时间价值": "Time value", "买/卖": "Bid / Ask",
    "合约": "Contract", "首次成交": "First trade", "有成交天数": "Days traded", "起始价": "Start px", "区间涨跌%": "Change %", "区间最高": "High", "区间最低": "Low",
    "当前IV%": "IV now %", "IV中位%": "IV median %", "IV最低%": "IV low %", "IV最高%": "IV high %", "IV百分位": "IV percentile",
    "得分": "Score", "跌幅%": "Drawdown %", "低点日": "Low date", "低点价": "Low px", "前低价": "Prior low", "低点抬高": "Higher low", "颈线": "Neckline",
    "突破颈线": "Neckline break", "距颈线%": "vs neckline %", "收复SMA20": "Above SMA20", "量能": "Volume", "量比": "Vol ratio", "RSI背离": "RSI divergence",
    "距低点%": "vs low %", "低点距今": "Bars since low",
    "bar 数": "Bars", "年数": "Years", "总收益": "Total return", "年化收益": "CAGR", "年化波动": "Ann. vol", "最大回撤": "Max drawdown", "胜率(按 bar)": "Hit rate (bars)",
    "盈亏比": "Payoff", "平均仓位": "Avg exposure", "年换手": "Ann. turnover", "期末净值": "Final equity", "入场": "Entry", "出场": "Exit", "出场价": "Exit px",
    "收益": "Return", "持有 bar": "Bars held", "持有中": "Open",
    "信号数": "Signals", "信号平均%": "Signal avg %", "信号中位%": "Signal median %", "信号胜率%": "Signal win %", "基准平均%": "Baseline avg %",
    "基准胜率%": "Baseline win %", "超额%": "Excess %", "t值": "t-stat",
    "数量": "Qty", "成本": "Cost", "市值": "Value", "浮动盈亏": "Open P&L", "占净值": "% of equity", "时间": "Time", "净值": "Equity", "现金": "Cash",
    "持有": "Holdings", "订单数": "Orders", "停机": "Halted", "风控原因": "Risk notes", "参数": "Parameters", "当前值": "Value", "含义": "Meaning", "指标": "Metric", "值": "Value",
    # 表格状态字
    "▲ 今日": "▲ today", "▲ 新进": "▲ new", "● 在列": "● in list", "▼ 出场条件": "▼ exit cond.", "回踩": "pullback",
    "↑ 多头": "↑ Bull", "↓ 空头": "↓ Bear", "→ 震荡": "→ Range", "蓄势": "Basing", "启动": "Igniting", "主升": "Markup", "整理": "Consolidating", "过热": "Overheated", "衰竭": "Fading",
    "领先": "Leading", "走弱": "Weakening", "落后": "Lagging", "改善": "Improving", "美": "US", "韩": "KR", "台": "TW", "日": "JP", "仅监控": "watch-only",
    "韩 仅监控": "KR watch-only", "台 仅监控": "TW watch-only", "日 仅监控": "JP watch-only", "差": "poor", "盘后": "after close", "盘前": "pre-market", "待定": "TBD",
    "确认": "Confirmed", "确认后回踩": "Confirmed, pulling back", "确认（未抬高）": "Confirmed (no higher low)", "初步企稳": "Stabilizing", "未见底": "No bottom yet",
    "跌破低点": "Broke the low", "无明显下跌": "No decline", "数据不足": "Not enough data", "是": "yes", "空仓": "flat", "财报日": "earnings day", "宏观日": "macro day",
    "行业": "Sector ETF", "主题": "Theme ETF", "风格": "Style", "跨资产": "Cross-asset", "我的板块": "My groups", "指数": "Index", "波动": "Volatility", "利率": "Rates",
    "汇率": "FX", "商品": "Commodity", "加密": "Crypto",
    # 控件 / 段标题（信号页以外）
    "叠加策略信号": "Overlay strategy signals", "△ 满足条件": "△ conditions met", "▲▼ 回放进出": "▲▼ replay entries/exits", "▮ 持仓底色": "▮ holding shade",
    "显示区间": "Range", "显示天数": "Days shown", "详情": "Details", "打开详情": "Open details", "标的详情": "Symbol details", "下次财报": "Next earnings", "20 根 bar": "20 bars",
    "公司简介": "Company profile", "理论价": "Theoretical", "正股": "Underlying",
    "大盘快照": "Market snapshot", "指数用指数本身，不是 ETF": "indices, not ETFs", "宏观事件": "Macro events", "同步官方日程": "Sync official calendar",
    "保存事件日历": "Save calendar", "宏观简报": "Macro brief", "Claude 联网搜索未来两周大事件与近一周要闻": "Claude web-searches the next two weeks of events and last week's news",
    "生成 / 刷新宏观简报": "Generate / refresh brief",
    "这页怎么读": "How to read this page", "基准": "Benchmark", "范围": "Scope", "回溯（交易日）": "Look back (trading days)", "RRG 参数": "RRG settings",
    "RS 比率窗口（日）": "RS ratio window (days)", "RS 动量窗口（日）": "RS momentum window (days)", "轨迹点数（周）": "Trail points (weeks)", "显示轨迹": "Show trails",
    "刷新": "Refresh", "相对轮动图（RRG）": "Relative Rotation Graph (RRG)", "相对强度表": "Relative strength table", "按 3 月超额收益排序": "sorted by 3m excess return",
    "月度收益热力图": "Monthly return heatmap", "最后一列是本月至今": "last column is month-to-date",
    "排序口径": "Rank by", "卖方：区间内概率": "Seller: P(inside range)", "买方：突破概率": "Buyer: P(breakout)", "最长到期": "Max days to expiry",
    "只看流动性达标": "Liquid only", "刷新链": "Refresh chains", "LEAP Call 对比": "LEAP call comparison", "到期月数": "Months to expiry", "目标 Delta": "Target delta",
    "合约明细": "Contract details", "合约历史价格": "Contract price history", "Yahoo 最后成交价": "Yahoo last-trade prices", "虚线 = 用当前 IV 反推的理论价": "dashed = theoretical at current IV",
    "底部确认筛选": "Bottom confirmation screen", "回看 bar 数": "Lookback bars", "摆动窗口 k": "Swing window k", "最小跌幅 %": "Min decline %", "量能倍数": "Volume multiple", "重算": "Recompute",
    "运行回测": "Run backtest", "标的（逗号分隔，可用 @组名）": "Symbols (comma separated, @group allowed)", "包含海外上市（仅监控）标的": "Include foreign (watch-only) names",
    "回溯天数": "Lookback days", "成交假设": "Fill assumption", "单边成本 bps": "Cost per side (bps)", "单标的仓位上限": "Max weight per name",
    "各标的累计贡献": "Cumulative contribution by symbol", "最新目标权重": "Latest target weights", "逐笔交易": "Trades", "已平仓": "Closed", "胜率": "Win rate",
    "平均收益": "Avg return", "均赢 / 均亏": "Avg win / loss", "平均持有": "Avg hold", "下载逐笔交易 CSV": "Download trades CSV", "下载 CSV": "Download CSV",
    "去重bar": "Dedupe bars", "起": "From", "止": "To", "只统计大盘在 200 日线上方时的信号": "Only signals with SPY above its 200-day", "策略参数（默认用 .env / 默认值，可改后重算）": "Strategy parameters (defaults from .env; edit to recompute)",
    "持有 20 bar 平均": "Avg after 20 bars", "持有 20 bar 胜率": "Win rate after 20 bars", "t 值（20 bar）": "t-stat (20 bars)", "分布图持有期": "Holding period for distribution",
    "全部信号明细": "All signals", "按年份": "By year", "按标的": "By symbol",
    "立刻运行一次": "Run once now", "确认重置": "confirm reset", "重置模拟盘": "Reset paper account", "持仓数": "Positions", "距高点回撤": "Drawdown from peak", "当日熔断线": "Daily stop",
    "净值曲线": "Equity curve", "最近一次决策": "Latest decision", "本次无订单": "no orders", "查看某次的完整记录": "Inspect one record",
    "标的池未来 30 天财报": "Watchlist earnings, next 30 days", "事件日历": "Event calendar", "编辑全部事件": "Edit all events",
    "新建板块": "New group", "添加标的": "Add symbols", "启用该板块": "Enable group", "保存该板块": "Save group", "确认删除该板块": "confirm delete", "删除板块": "Delete group",
    "成员（取消勾选即删除）": "Members (uncheck to remove)", "已保存到 watchlist.json，.env 的 SYMBOLS 不再生效": "Saved to watchlist.json; SYMBOLS in .env no longer applies",
    "信号符号": "Signal symbols", "指标列": "Indicator columns", "策略规则": "Strategy rules", "共用机制": "Shared mechanics", "其他页面的口径": "Other pages' definitions",
    "参数：实盘策略用 .env 的值，其他用默认": "params: live strategy uses .env, others use defaults", "显示": "Shown", "含义": "Meaning", "列": "Column",
    "正股隐含波动率": "Underlying implied volatility", "IV30%": "IV30 %", "HV30%": "HV30 %", "历史天数": "History days", "区间低%": "Range low %", "区间高%": "Range high %",
    "近/远IV": "Near/far IV", "IV/VIX": "IV / VIX", "近月到期": "Near expiry", "远月到期": "Far expiry", "合约IV百分位": "Contract IV pct", "IV30 = 30 天平值隐含波动率，真实报价": "IV30 = 30-day ATM implied vol, real quotes",
    "主升浪体检": "Main-wave checklist", "模板": "template", "乖离 SMA50": "vs SMA50",
    # 图例短语（先于单字匹配）
    "满足入场条件": "entry conditions met", "模拟入场": "replay entry", "模拟退出": "replay exit", "模拟持仓期": "replay holding period", "成交价": "trade price",
    "策略回放中的模拟持仓期": "replay holding period", "绿色底 = 策略回放中的模拟持仓期": "shade = replay holding period", "成交量": "Volume", "20日均量": "20d avg vol",
    "个板块": " groups", "美股 · swing · 只做多": "US equities · swing · long only", "前低": "prior low", "最近低点": "latest low", "在列": "in list", "200日线": "200-day MA",
    "排队入场（按分数）": "queued (by score)", "排队": "queued", "同一天被多个策略选中": "picked by several strategies the same day",
    "趋势模板 ≥ 6 条且 ADX ≥ 20 为主升，4~5 条且转强为启动": "template ≥ 6 with ADX ≥ 20 = Markup; 4–5 and turning up = Igniting", "主升浪阶段": "Main-wave stage",
    "数据截至": "data as of", "回溯": "look back", "日）": "d)", "策略模拟仓位": "strategy replay positions", "模拟仓位": "replay positions", "最近 10 bar 退出": "exits in last 10 bars", "入场信号": "entry signal", "实际入场": "actual entry", "实际退出": "actual exit", "策略回撤": "strategy DD", "基准回撤": "benchmark DD",
    "仓位": "exposure", "信号后": "after signal", "基准（任意一天）": "baseline (any day)", "信号": "Signals", "基准": "Benchmark",
    "合约价格（右轴：正股）": "contract price (right axis: underlying)", "反推的隐含波动率 IV%": "back-solved implied volatility %", "期权价格": "option price", "期权收益率 %": "option return %",
    "到期时正股涨跌 %": "underlying move at expiry %", "到期时收益率 vs 正股涨跌": "return at expiry vs underlying move", "相对表现": "relative performance",
    "RS 比率（相对基准的强弱）": "RS ratio (strength vs benchmark)", "RS 动量": "RS momentum", "在变弱": "weakening", "在变强": "strengthening", "比基准弱": "weaker than benchmark", "比基准强": "stronger than benchmark",
    "信号后第几根 bar": "bars after signal", "平均累计收益 %": "avg cumulative return %", "收益 %": "return %", "占比": "share", "月度期权到期": "monthly opex",
    "比基准强，而且还在变强": "stronger than benchmark and strengthening", "比基准强，但势头在减弱": "stronger but losing momentum", "比基准弱，而且还在变弱": "weaker and weakening", "比基准弱，但势头在转强": "weaker but turning up",
    "顺势持有 / 回调找机会": "hold with trend / buy pullbacks", "别追高，注意兑现": "don't chase; consider taking profit", "回避，等它转到“改善”": "avoid until it turns to Improving", "最值得盯的候选：从弱变强的早期": "best candidates: early turn from weak to strong",
    "▲ 今日 = 今天收盘满足该策略的入场条件；▲ n天前 = 最近 5 天内出现过入场条件；▼ 出场条件 = 勾选后显示，表示今天满足出场规则。": "▲ today = entry conditions met at today's close; ▲ n days ago = met within the last 5 bars; ▼ = exit conditions met today (when enabled).",
    # 长段说明的整句翻译
    "共振 = 当天同时给出信号的策略个数（≥ 2 标绿）；阶段 = 主升浪阶段（蓄势 → 启动 → 主升 → 过热 → 衰竭，详情弹窗里有 8 条体检）。▲ 今日 = 今天收盘满足该策略的入场条件；▲ n天前 = 最近 5 天内出现过入场条件；▼ 出场条件 = 勾选后显示，表示今天满足出场规则。相对强度是状态型：▲ 新进 = 今天刚进入动量前 20%，● 在列 = 持续在前 20%（只在美股里排名，海外标的不占名额）。全部是规则计算，不看任何账户持仓。距55高 = 现价相对过去 55 根 bar 最高收盘；财报 = 下次财报日期（距今天数）。海外上市标的（如海力士）只出信号不交易，价格是当地货币。":
        "Confluence = number of strategies signalling the same day (≥ 2 in green). Stage = main-wave stage (Basing → Igniting → Markup → Overheated → Fading; the details dialog has the 8-point checklist). "
        "▲ today = entry conditions met at today's close; ▲ n days ago = met within the last 5 bars; ▼ exit = shown when enabled, exit rules met today. Relative Strength is state-based: ▲ new = just entered the top 20% by momentum, ● in list = still in the top 20% (ranked among US names only; foreign names take no slot). "
        "Everything is rule-based and ignores any account positions. vs 55-bar high = close relative to the highest close of the prior 55 bars; Earnings = next earnings date (days away). Foreign listings (e.g. SK Hynix) are watch-only, priced in local currency.",
    "每格是该板块当月的绝对收益；一行从左到右看它自己的节奏，一列从上到下看当月谁强谁弱，连续几个月的颜色带就是轮动。RS 比率 = 100 × (价格/基准) / 其过去 N 日均值；RS 动量 = 100 × RS 比率 / M 日前的 RS 比率。RRG 的常见近似，不是 JdK 原版算法。“我的板块”按美股成员每日等权合成，海外成员不计。全是历史统计，不预测方向。":
        "Each cell is the sector's absolute return for that month; read a row for its own rhythm, a column for who led that month, and multi-month colour bands for rotation. RS ratio = 100 × (price / benchmark) / its N-day mean; RS momentum = 100 × RS ratio / RS ratio M days ago. "
        "A common RRG approximation, not the proprietary JdK formula. 'My groups' are equal-weighted daily from US members only. All historical statistics, not forecasts.",
    "每格是当天收盘满足该策略入场条件的标的；勾“含出场条件”后 ▼ 后面是满足出场规则的标的。相对强度这类状态型策略只列当天“新进”前 20% 的标的，括号里是当天在列的数量。":
        "Each cell lists the symbols whose entry conditions were met at that day's close; with 'include exit conditions' ticked, ▼ lists symbols meeting exit rules. State-based strategies such as Relative Strength list only the names that newly entered the top 20% that day; the number in brackets is how many were in the list.",
}

_STATUS_RE = re.compile(r"^▲ (\d+)(日|小时)前$")
_KEYS_BY_LEN = sorted(T, key=len, reverse=True)


def tr(s):
    """精确匹配翻译；不在字典里原样返回。非字符串直接返回。"""
    if _LANG != "en" or not isinstance(s, str):
        return s
    if s in T:
        return T[s]
    m = _STATUS_RE.match(s)
    if m:
        return f"▲ {m.group(1)}{'d' if m.group(2) == '日' else 'h'} ago"
    return s


def tr_text(s):
    """给图例 / 段标题用：先整句匹配，不行就把句子里出现的字典词逐个替换（长词优先）。"""
    if _LANG != "en" or not isinstance(s, str):
        return s
    if s in T:
        return T[s]
    if len(s) > 80 and "<" not in s:  # 长段说明文字没有整句翻译就保持中文，不做零碎替换
        return s
    out = s
    for k in _KEYS_BY_LEN:
        if len(k) >= 2 and k in out:
            out = out.replace(k, T[k])
    out = re.sub(r"(\d+)\s*只(?![a-zA-Z一-鿿])", r"\1 names", out)  # “27 只” -> “27 names”，不碰“只用 / 只看”
    out = re.sub(r"(\d+)\s*个(?=[\s<·（(]|$)", r"\1", out)
    out = re.sub(r"(?<=[>：:·\s（(])无(?=[<\s）)]|$)", "none", out)
    out = re.sub(r"(?<=[>：:·\s（(])是(?=[<\s）)]|$)", "yes", out)
    return out


def translate_dataframe(df):
    """列名与字符串格子按字典翻译，返回新 DataFrame。"""
    import pandas as pd

    if _LANG != "en":
        return df
    new = df.copy()
    new.columns = [tr(c) if isinstance(c, str) else c for c in new.columns]
    for c in new.columns:
        col = new[c]
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            new[c] = col.map(lambda v: tr(v) if isinstance(v, str) else v)
    return new


def translate_styler(sty):
    """把 Styler 的样式按位置搬到翻译后的表上（列名 / 状态字翻译不会破坏样式）。"""
    import pandas as pd

    if _LANG != "en":
        return sty
    sty._compute()
    data = sty.data
    css = pd.DataFrame("", index=data.index, columns=data.columns, dtype=object)
    for (r, c), lst in sty.ctx.items():
        css.iat[r, c] = ";".join(f"{k}:{v}" for k, v in lst)
    new = translate_dataframe(data)
    css.columns = new.columns
    return new.style.apply(lambda _: css, axis=None)


def translate_fig(fig):
    """Plotly 图：图例名、坐标轴标题、子图标题、注释按 tr_text 翻译。"""
    if _LANG != "en":
        return fig
    try:
        for trace in fig.data:
            if getattr(trace, "name", None):
                trace.name = tr_text(trace.name)
        lay = fig.layout
        for ax in ("xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"):
            a = getattr(lay, ax, None)
            if a is not None and a.title is not None and a.title.text:
                a.title.text = tr_text(a.title.text)
        for ann in lay.annotations or ():
            if ann.text:
                ann.text = tr_text(ann.text)
        if lay.title is not None and lay.title.text:
            lay.title.text = tr_text(lay.title.text)
    except Exception:
        pass
    return fig


def install(st) -> None:
    """英文模式下给 Streamlit 的常用元素函数套一层翻译。只在每次脚本运行时调用一次。"""
    if _LANG != "en" or getattr(st, "_i18n_installed", False):
        return
    import pandas as pd
    from pandas.io.formats.style import Styler

    def wrap_label(fn, has_options=False):
        def inner(label=None, *args, **kwargs):
            if "help" in kwargs:
                kwargs["help"] = tr_text(kwargs["help"])
            if "placeholder" in kwargs:
                kwargs["placeholder"] = tr_text(kwargs["placeholder"])
            if has_options:
                ff = kwargs.get("format_func")
                kwargs["format_func"] = (lambda x, _f=ff: tr_text(_f(x))) if ff else (lambda x: tr(x))
            return fn(tr_text(label) if isinstance(label, str) else label, *args, **kwargs)
        return inner

    from streamlit.delta_generator import DeltaGenerator as DG

    def wrap_method(fn, has_options=False):
        def inner(self, label=None, *args, **kwargs):
            if "help" in kwargs:
                kwargs["help"] = tr_text(kwargs["help"])
            if "placeholder" in kwargs:
                kwargs["placeholder"] = tr_text(kwargs["placeholder"])
            if has_options:
                ff = kwargs.get("format_func")
                if ff:
                    kwargs["format_func"] = lambda x, _f=ff: str(tr_text(_f(x)))
                else:
                    kwargs["format_func"] = lambda x: tr(x) if isinstance(x, str) else str(x)  # 数字选项原样转成字符串
            return fn(self, tr_text(label) if isinstance(label, str) else label, *args, **kwargs)
        return inner

    for name in ("button", "checkbox", "slider", "number_input", "text_input", "date_input", "expander", "popover", "download_button", "subheader", "toast", "text_area", "segmented_control"):
        if hasattr(DG, name):
            setattr(DG, name, wrap_method(getattr(DG, name)))
    for name in ("selectbox", "multiselect", "radio", "select_slider"):
        setattr(DG, name, wrap_method(getattr(DG, name), has_options=True))

    def wrap_body(fn):
        def inner(self, body=None, *args, **kwargs):
            return fn(self, tr_text(body) if isinstance(body, str) else body, *args, **kwargs)
        return inner

    for name in ("markdown", "caption", "info", "warning", "success", "error"):
        setattr(DG, name, wrap_body(getattr(DG, name)))

    _tabs = DG.tabs
    DG.tabs = lambda self, labels, *a, **k: _tabs(self, [tr_text(x) for x in labels], *a, **k)

    _metric = DG.metric
    def metric(self, label, value, delta=None, *a, **k):
        if "help" in k:
            k["help"] = tr_text(k["help"])
        return _metric(self, tr_text(label), tr_text(value) if isinstance(value, str) else value, tr_text(delta) if isinstance(delta, str) else delta, *a, **k)
    DG.metric = metric

    _df = DG.dataframe
    def dataframe(self, data=None, *a, **k):
        if isinstance(data, Styler):
            data = translate_styler(data)
        elif isinstance(data, pd.DataFrame):
            data = translate_dataframe(data)
        if "column_config" in k and isinstance(k["column_config"], dict):
            k["column_config"] = {tr(kk): v for kk, v in k["column_config"].items()}
        return _df(self, data, *a, **k)
    DG.dataframe = dataframe

    _plot = DG.plotly_chart
    DG.plotly_chart = lambda self, fig, *a, **k: _plot(self, translate_fig(fig), *a, **k)

    # st.xxx 是在 import 时就绑定到主容器实例上的方法，类补丁管不到它们，这里再绑一次
    main = getattr(st, "_main", None)
    for name in ("button", "checkbox", "slider", "number_input", "text_input", "date_input", "expander", "popover", "download_button", "subheader", "toast", "text_area",
                 "segmented_control", "selectbox", "multiselect", "radio", "select_slider", "markdown", "caption", "info", "warning", "success", "error",
                 "tabs", "metric", "dataframe", "plotly_chart"):
        if hasattr(DG, name) and main is not None:
            try:
                setattr(st, name, getattr(main, name))
            except Exception:
                pass

    st._i18n_installed = True
