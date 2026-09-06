import json

import pytest

from tradebot.universe import GROUPS, expand
from tradebot.watchlist import Watchlist, canonical


def test_seed_from_env_tokens(tmp_path):
    wl = Watchlist.load(tmp_path / "w.json", "@mag7,@berkshire,spy,BRK-B")
    assert wl.source == "env"
    assert list(wl.groups) == ["mag7", "berkshire", "自选"]
    assert wl.groups["自选"] == ["SPY", "BRK.B"]
    assert wl.symbols()[:7] == GROUPS["mag7"] and wl.symbols().count("BRK.B") == 1


def test_edit_save_load_round_trip(tmp_path):
    path = tmp_path / "w.json"
    wl = Watchlist.load(path, "@berkshire")
    wl.add_group("能源")
    wl.add_symbols("能源", ["xom", "CVX", "xom"])
    wl.remove_symbols("能源", ["CVX"])
    wl.set_enabled("berkshire", False)
    wl.save()
    assert json.loads(path.read_text(encoding="utf-8"))["groups"]["能源"] == ["XOM"]

    wl2 = Watchlist.load(path, "@mag7")  # 有文件时忽略 env
    assert wl2.source == "file"
    assert wl2.symbols() == ["XOM"]  # berkshire 被停用
    assert wl2.group_of("BRK.B") == ["berkshire"]
    with pytest.raises(ValueError):
        wl2.add_group("能源")
    wl2.remove_group("能源")
    assert wl2.symbols() == []


def test_canonical_and_expand_with_custom_groups():
    assert canonical(" brk-b ") == "BRK.B"
    assert canonical("005930.ks") == "005930.KS"
    syms = expand("@能源,@mag7", groups={"能源": ["XOM", "CVX"]})
    assert syms[:2] == ["XOM", "CVX"] and "AAPL" in syms
