from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .universe import expand, is_us_listed
from .watchlist import Watchlist

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = ROOT / "watchlist.json"
load_dotenv(ROOT / ".env")

INTERVALS = ("1d", "1h")
DEFAULT_LOOKBACK_DAYS = {"1h": 60, "1d": 750}      # 实盘算信号拉多少天；日线拉长一些让 swing 状态机重放更完整
DEFAULT_BACKTEST_DAYS = {"1h": 729, "1d": 5000}    # 回测默认回溯；小时线上限 729


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    if val is None:
        return default
    val = val.split("#", 1)[0].strip()  # 允许行内注释；去掉注释后为空也算未设置
    return val if val else default


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).lower() in {"1", "true", "yes", "on"}


def parse_params(text: str) -> dict:
    """把 'fast=20,slow=50' 解析成 {'fast': 20, 'slow': 50}，数值自动转 int/float。"""
    out: dict = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        v = v.strip()
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


@dataclass
class Settings:
    symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    symbols_raw: str = ""
    watchlist: Watchlist | None = None
    view_groups: list[str] = field(default_factory=list)  # 面板默认勾选的板块；空 = 全部

    @property
    def tradable_symbols(self) -> list[str]:
        """美股，才能进回测资金分配和实盘下单；海外上市的只监控。"""
        return [s for s in self.symbols if is_us_listed(s)]

    @property
    def monitor_only_symbols(self) -> list[str]:
        return [s for s in self.symbols if not is_us_listed(s)]
    interval: str = "1d"
    lookback_days: int = 400
    backtest_days: int = 5000
    strategy: str = "ma_cross"
    strategy_params: dict = field(default_factory=dict)

    mode: str = "local"  # local | alpaca
    entry_mode: str = "fresh"  # fresh = 只跟新入场信号；align = 直接对齐策略仓位
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    initial_cash: float = 100_000.0

    max_position_pct: float = 0.20
    max_gross_exposure: float = 1.0
    daily_loss_limit_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    min_trade_value: float = 200.0

    commission_bps: float = 0.0
    slippage_bps: float = 2.0

    agent_enabled: bool = False
    agent_model: str = "claude-opus-5"

    tz: str = "America/New_York"
    log_dir: Path = ROOT / "logs"
    cache_dir: Path = ROOT / "data_cache"


def load_settings() -> Settings:
    interval = _env("INTERVAL", "1d").lower()
    if interval not in INTERVALS:
        raise ValueError(f"INTERVAL={interval!r} 不支持，可选 {INTERVALS}")
    symbols_raw = _env("SYMBOLS", "@mag7,@semis,@berkshire")
    watchlist = Watchlist.load(WATCHLIST_PATH, symbols_raw)  # 有 watchlist.json 以它为准
    s = Settings(
        symbols=watchlist.symbols(),
        symbols_raw=symbols_raw,
        watchlist=watchlist,
        view_groups=[g.strip().lower() for g in _env("VIEW_GROUPS", "mag7,storage").split(",") if g.strip()],
        interval=interval,
        lookback_days=_env_int("LOOKBACK_DAYS", 0) or DEFAULT_LOOKBACK_DAYS[interval],
        backtest_days=_env_int("BACKTEST_DAYS", 0) or DEFAULT_BACKTEST_DAYS[interval],
        strategy=_env("STRATEGY", "ma_cross"),
        strategy_params=parse_params(_env("STRATEGY_PARAMS", "")),
        mode=_env("MODE", "local").lower(),
        entry_mode=_env("ENTRY_MODE", "fresh").lower(),
        alpaca_api_key=_env("ALPACA_API_KEY", ""),
        alpaca_secret_key=_env("ALPACA_SECRET_KEY", ""),
        alpaca_paper=_env_bool("ALPACA_PAPER", True),
        initial_cash=_env_float("INITIAL_CASH", 100_000.0),
        max_position_pct=_env_float("MAX_POSITION_PCT", 0.20),
        max_gross_exposure=_env_float("MAX_GROSS_EXPOSURE", 1.0),
        daily_loss_limit_pct=_env_float("DAILY_LOSS_LIMIT_PCT", 0.02),
        max_drawdown_pct=_env_float("MAX_DRAWDOWN_PCT", 0.10),
        min_trade_value=_env_float("MIN_TRADE_VALUE", 200.0),
        commission_bps=_env_float("COMMISSION_BPS", 0.0),
        slippage_bps=_env_float("SLIPPAGE_BPS", 2.0),
        agent_enabled=_env_bool("AGENT_ENABLED", False),
        agent_model=_env("AGENT_MODEL", "claude-opus-5"),
    )
    s.log_dir.mkdir(parents=True, exist_ok=True)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s


def update_env(values: dict[str, str], path: Path | None = None) -> Path:
    """把若干键写回 .env：已有的行原地替换（保留行内注释），没有的追加；然后刷新当前进程的环境变量。"""
    import re

    path = path or (ROOT / ".env")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done = set()
    out = []
    for line in lines:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m and m.group(1) in values:
            key = m.group(1)
            rest = m.group(2)
            comment = ""
            if "#" in rest:
                comment = "  #" + rest.split("#", 1)[1]
            out.append(f"{key}={values[key]}{comment}")
            done.add(key)
        else:
            out.append(line)
    for key, val in values.items():
        if key not in done:
            out.append(f"{key}={val}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    load_dotenv(path, override=True)
    for key, val in values.items():
        os.environ[key] = str(val)
    return path
