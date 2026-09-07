import pytest

from tradebot.universe import GROUPS, expand, from_yahoo, is_us_listed, market_of, to_alpaca, to_yahoo


def test_expand_groups_and_dedupe_keeps_order():
    syms = expand("@mag7,@semis,@berkshire")
    assert syms[:7] == GROUPS["mag7"]
    assert syms.count("NVDA") == 1  # NVDA 同时在 mag7 和 semis
    assert syms[-1] == "BRK.B"
    assert len(syms) == len(set(syms))


def test_expand_mixed_tokens_and_case():
    assert expand(" spy , @berkshire ,SPY") == ["SPY", "BRK.B"]


def test_unknown_group_raises():
    with pytest.raises(KeyError):
        expand("@nope")


def test_symbol_mapping_round_trip():
    assert to_yahoo("BRK.B") == "BRK-B"
    assert from_yahoo("BRK-B") == "BRK.B"
    assert to_alpaca("BRK.B") == "BRK.B"
    for s in ["AAPL", "TSM", "SMH"]:
        assert from_yahoo(to_yahoo(s)) == s


def test_exchange_suffix_is_kept_and_marked_foreign():
    assert to_yahoo("005930.KS") == "005930.KS" and from_yahoo("005930.KS") == "005930.KS"
    assert to_yahoo("000660.ks") == "000660.KS"
    assert to_yahoo("BRK.B") == "BRK-B"  # 一个字母仍是股份类别
    assert market_of("005930.KS") == "韩" and market_of("BRK.B") == "美" and market_of("7203.T") == "日"
    assert not is_us_listed("000660.KS") and is_us_listed("NVDA")
    assert "000660.KS" in GROUPS["storage"] and "DRAM" in GROUPS["storage"] and "005930.KS" not in GROUPS["storage"]
    assert "RMBS" not in GROUPS["storage"] and "2408.TW" not in GROUPS["storage"]
    assert market_of("2408.TW") == "台" and not is_us_listed("2344.TW")


def test_ai_infra_group():
    assert GROUPS["ai_infra"] == ["CRWV", "TTMI"] and set(expand("@ai_infra")) == {"CRWV", "TTMI"}


def test_space_group():
    assert GROUPS["space"] == ["RKLB"] and "RKLB" in expand("@space")
