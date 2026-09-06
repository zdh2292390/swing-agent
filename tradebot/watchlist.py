"""可编辑的标的池：板块 -> 标的列表，存 watchlist.json。

没有文件时从 .env 的 SYMBOLS 初始化（@组名 变成同名板块，散的标的进"自选"）；
一旦在面板里保存过，就以文件为准，.env 的 SYMBOLS 不再生效。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .universe import GROUPS, from_yahoo

DEFAULT_GROUP = "自选"


def canonical(symbol: str) -> str:
    """统一写法：大写，BRK-B -> BRK.B。"""
    return from_yahoo(symbol.strip().upper())


@dataclass
class Watchlist:
    path: Path
    groups: dict[str, list[str]] = field(default_factory=dict)
    enabled: dict[str, bool] = field(default_factory=dict)
    source: str = "env"  # env | file

    # ---- 查询 ----
    def symbols(self) -> list[str]:
        out: list[str] = []
        for g, members in self.groups.items():
            if not self.enabled.get(g, True):
                continue
            for s in members:
                if s not in out:
                    out.append(s)
        return out

    def group_of(self, symbol: str) -> list[str]:
        symbol = canonical(symbol)
        return [g for g, m in self.groups.items() if symbol in m]

    def describe(self) -> str:
        n_on = sum(1 for g in self.groups if self.enabled.get(g, True))
        return f"{'watchlist.json' if self.source == 'file' else '.env'} · {n_on}/{len(self.groups)} 个板块 · {len(self.symbols())} 只"

    # ---- 编辑 ----
    def add_group(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("板块名不能为空")
        if name in self.groups:
            raise ValueError(f"板块 {name} 已存在")
        self.groups[name] = []
        self.enabled[name] = True

    def remove_group(self, name: str) -> None:
        self.groups.pop(name, None)
        self.enabled.pop(name, None)

    def rename_group(self, old: str, new: str) -> None:
        new = new.strip()
        if not new or new in self.groups:
            raise ValueError(f"无效的新名字 {new!r}")
        self.groups = {new if g == old else g: m for g, m in self.groups.items()}
        self.enabled[new] = self.enabled.pop(old, True)

    def set_members(self, group: str, members: list[str]) -> None:
        seen: list[str] = []
        for s in members:
            s = canonical(s)
            if s and s not in seen:
                seen.append(s)
        self.groups[group] = seen

    def add_symbols(self, group: str, symbols: list[str]) -> None:
        self.set_members(group, self.groups.get(group, []) + list(symbols))

    def remove_symbols(self, group: str, symbols: list[str]) -> None:
        drop = {canonical(s) for s in symbols}
        self.groups[group] = [s for s in self.groups.get(group, []) if s not in drop]

    def set_enabled(self, group: str, on: bool) -> None:
        self.enabled[group] = bool(on)

    # ---- 持久化 ----
    def save(self) -> None:
        self.path.write_text(json.dumps({"groups": self.groups, "enabled": self.enabled}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.source = "file"

    @classmethod
    def load(cls, path: Path, env_symbols: str = "") -> "Watchlist":
        path = Path(path)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            groups = {g: [canonical(s) for s in m] for g, m in raw.get("groups", {}).items()}
            enabled = {g: bool(raw.get("enabled", {}).get(g, True)) for g in groups}
            return cls(path, groups, enabled, "file")
        groups: dict[str, list[str]] = {}
        for tok in env_symbols.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.startswith("@"):
                name = tok[1:].lower()
                if name not in GROUPS:
                    raise KeyError(f"未知标的组 @{name}，可选: {', '.join('@' + g for g in GROUPS)}")
                groups[name] = list(GROUPS[name])
            else:
                groups.setdefault(DEFAULT_GROUP, [])
                s = canonical(tok)
                if s not in groups[DEFAULT_GROUP]:
                    groups[DEFAULT_GROUP].append(s)
        return cls(path, groups, {g: True for g in groups}, "env")
