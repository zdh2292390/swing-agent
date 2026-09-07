"""标的池与符号映射。

统一符号用交易所写法（例如 BRK.B）。yfinance 用连字符 BRK-B，Alpaca 用点 BRK.B。
.env 的 SYMBOLS 里可以混写组名和单个标的：SYMBOLS=@mag7,@semis,BRK.B
"""
from __future__ import annotations

GROUPS: dict[str, list[str]] = {
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "semis": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "QCOM", "ARM", "AMAT", "LRCX", "KLAC", "INTC", "MRVL", "TXN"],
    "berkshire": ["BRK.B"],
    "ai_infra": ["CRWV", "TTMI"],  # AI 基础设施：CoreWeave（算力云）、TTM Technologies（数据中心 / 航天军工 PCB）
    "space": ["RKLB"],  # 航天：Rocket Lab
    # 存储/内存：美光、西数、闪迪，DRAM = Roundhill Memory ETF；SK 海力士是韩国上市，只监控不交易
    "storage": ["MU", "WDC", "SNDK", "DRAM", "000660.KS"],
    "benchmarks": ["SPY", "QQQ", "SMH"],
}

# 统一符号 -> yfinance 符号 的特例；一般规则见 to_yahoo
_YAHOO_OVERRIDES: dict[str, str] = {}
_YAHOO_REVERSE = {v: k for k, v in _YAHOO_OVERRIDES.items()}

# 交易所后缀 -> 市场名。带这些后缀的是海外上市，只能监控，不能通过 Alpaca 交易
EXCHANGE_SUFFIX = {"KS": "韩", "KQ": "韩", "T": "日", "HK": "港", "L": "英", "DE": "德", "PA": "法", "TW": "台", "SS": "沪", "SZ": "深"}
DISPLAY_NAMES = {"005930.KS": "三星电子", "000660.KS": "SK海力士", "DRAM": "Roundhill 内存ETF", "CRWV": "CoreWeave", "RKLB": "Rocket Lab",
                 "TTMI": "TTM Technologies"}


def _split_suffix(symbol: str) -> tuple[str, str]:
    """'005930.KS' -> ('005930', 'KS')；'7203.T' -> ('7203', 'T')；'BRK.B' -> ('BRK.B', '')。
    规则：点后面是已知交易所后缀、或两个及以上字母、或前面是纯数字代码，就是交易所后缀；
    否则（BRK.B、BF.B 这种单个字母）是股份类别。"""
    if "." in symbol:
        base, suf = symbol.rsplit(".", 1)
        if suf.isalpha() and (suf in EXCHANGE_SUFFIX or len(suf) >= 2 or base.isdigit()):
            return base, suf
    return symbol, ""


def to_yahoo(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol in _YAHOO_OVERRIDES:
        return _YAHOO_OVERRIDES[symbol]
    base, suf = _split_suffix(symbol)
    return symbol if suf else symbol.replace(".", "-")


def from_yahoo(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol in _YAHOO_REVERSE:
        return _YAHOO_REVERSE[symbol]
    base, suf = _split_suffix(symbol)
    return symbol if suf else symbol.replace("-", ".")


def to_alpaca(symbol: str) -> str:
    return symbol.upper()  # Alpaca 与统一写法一致（BRK.B）


def market_of(symbol: str) -> str:
    """'美' / '韩' / '日' ..."""
    _, suf = _split_suffix(symbol.upper())
    return EXCHANGE_SUFFIX.get(suf, "海外") if suf else "美"


def is_us_listed(symbol: str) -> bool:
    return market_of(symbol) == "美"


def display_name(symbol: str) -> str:
    return DISPLAY_NAMES.get(symbol.upper(), "")


def expand(tokens: list[str] | str, groups: dict[str, list[str]] | None = None) -> list[str]:
    """把 ['@mag7', '@semis', 'BRK.B'] 展开成去重后的统一符号列表，保持顺序。

    groups 可传入 watchlist 里用户自定义的板块，同名时覆盖内置组。
    """
    table = {**GROUPS, **{k.lower(): v for k, v in (groups or {}).items()}}
    if isinstance(tokens, str):
        tokens = tokens.split(",")
    out: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("@"):
            name = tok[1:].lower()
            if name not in table:
                raise KeyError(f"未知标的组 @{name}，可选: {', '.join('@' + g for g in table)}")
            items = table[name]
        else:
            items = [tok.upper()]
        for s in items:
            if s not in out:
                out.append(s)
    return out


def group_of(symbol: str) -> list[str]:
    symbol = symbol.upper()
    return [g for g, members in GROUPS.items() if symbol in members]
