from tradebot.risk import PortfolioState, RiskLimits, RiskManager


def state(equity=100_000, day_start=100_000, peak=100_000):
    return PortfolioState(equity=equity, cash=equity, positions={}, day_start_equity=day_start, peak_equity=peak)


def test_position_cap_and_gross_scaling():
    rm = RiskManager(RiskLimits(max_position_pct=0.2, max_gross_exposure=0.5))
    d = rm.approve({"A": 0.5, "B": 0.3, "C": -0.1}, state())
    assert not d.halted
    assert d.approved["C"] == 0.0
    # A、B 先被截到 0.2，总仓位 0.4 未超 0.5，不缩放
    assert abs(d.approved["A"] - 0.2) < 1e-12 and abs(d.approved["B"] - 0.2) < 1e-12

    d2 = rm.approve({"A": 0.2, "B": 0.2, "C": 0.2}, state())
    assert abs(sum(d2.approved.values()) - 0.5) < 1e-12


def test_daily_loss_circuit_breaker():
    rm = RiskManager(RiskLimits(daily_loss_limit_pct=0.02))
    d = rm.approve({"A": 0.2}, state(equity=97_900, day_start=100_000, peak=100_000))
    assert d.halted and all(v == 0 for v in d.approved.values())
    assert any("DAILY_STOP" in r for r in d.reasons)


def test_max_drawdown_kill_switch():
    rm = RiskManager(RiskLimits(max_drawdown_pct=0.10))
    d = rm.approve({"A": 0.2}, state(equity=89_000, day_start=89_000, peak=100_000))
    assert d.halted
    assert any("KILL_SWITCH" in r for r in d.reasons)


def test_normal_case_passes_through():
    rm = RiskManager(RiskLimits())
    d = rm.approve({"A": 0.15, "B": 0.0}, state(equity=99_000))
    assert not d.halted and d.approved == {"A": 0.15, "B": 0.0}
