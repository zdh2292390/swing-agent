import pandas as pd
import pytest

from tradebot.portfolio import select_targets


def test_fresh_mode_only_buys_new_entries_and_follows_held():
    last = pd.Series({"A": 0.2, "B": 0.2, "C": 0.0, "D": 0.2})
    prev = pd.Series({"A": 0.2, "B": 0.0, "C": 0.2, "D": 0.2})
    targets, notes = select_targets(last, prev, held={"C", "D"}, mode="fresh")
    assert targets == {"A": 0.0, "B": 0.2, "C": 0.0, "D": 0.2}  # A 老仓位不追；B 新入场买；C 账户有且策略退出 -> 卖；D 继续持有
    assert any("A" in n for n in notes) and len(notes) == 1


def test_align_mode_copies_strategy():
    last = pd.Series({"A": 0.2, "B": 0.0})
    prev = pd.Series({"A": 0.2, "B": 0.2})
    targets, notes = select_targets(last, prev, held=set(), mode="align")
    assert targets == {"A": 0.2, "B": 0.0} and notes == []


def test_bad_mode():
    with pytest.raises(ValueError):
        select_targets(pd.Series({"A": 0.1}), pd.Series({"A": 0.0}), set(), "nope")
