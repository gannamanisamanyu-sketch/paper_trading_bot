import numpy as np
import pandas as pd
import pytest

from risk import metrics as M
from risk.limits import DrawdownMonitor, check_limits, gross_exposure_cap, stop_level, stop_triggered


def eq(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_max_drawdown_known_path():
    e = eq([100, 120, 90, 130, 104])
    assert M.max_drawdown(e) == pytest.approx(90 / 120 - 1)


def test_cagr_and_total_return():
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])
    e = pd.Series([100.0, 121.0], index=idx)
    assert M.total_return(e) == pytest.approx(0.21)
    assert M.cagr(e) == pytest.approx(0.1, abs=2e-3)


def test_sharpe_sortino_signs():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert M.sharpe_ratio(r) > 0
    assert M.sortino_ratio(r) > M.sharpe_ratio(r)  # downside dev < total dev for positive drift


def test_var_and_expected_shortfall():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))
    var = M.value_at_risk(r, 0.95)
    es = M.expected_shortfall(r, 0.95)
    assert var == pytest.approx(0.045, abs=1e-3)
    assert es >= var  # ES is always at least as bad as VaR


def test_beta_alpha_of_benchmark_itself():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0.0005, 0.01, 500))
    ba = M.beta_alpha(b, b)
    assert ba["beta"] == pytest.approx(1.0)
    assert ba["alpha"] == pytest.approx(0.0, abs=1e-12)


def test_round_trips_and_trade_stats():
    d = pd.to_datetime(["2024-01-02", "2024-01-10", "2024-02-01", "2024-02-05"])
    fills = pd.DataFrame({
        "date": d, "ticker": ["A", "A", "B", "B"], "side": ["BUY", "SELL", "BUY", "SELL"],
        "shares": [10, 10, 5, 5], "gross_value": [1000.0, 1200.0, 500.0, 400.0],
        "commission": [0.0] * 4, "reason": ["", "x", "", "y"]})
    trips = M.round_trips(fills)
    stats = M.trade_statistics(trips)
    assert stats["round_trips"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["profit_factor"] == pytest.approx(200 / 100)


def test_stop_levels():
    lvl = stop_level(100.0, "volatility", annual_vol=0.25, vol_multiple=2.0)
    assert lvl == pytest.approx(100 * (1 - 2 * 0.25 * np.sqrt(21 / 252)))
    assert stop_triggered(lvl - 0.01, lvl)
    assert not stop_triggered(lvl + 0.01, lvl)
    assert not stop_triggered(float("nan"), lvl)
    assert stop_level(100.0, "atr", atr=2.0, atr_multiple=3.0) == pytest.approx(94.0)


def test_drawdown_monitor_hysteresis():
    m = DrawdownMonitor(limit=0.15, reset=0.10)
    for v in [100, 90, 84]:
        m.update(v)
    assert m.active                 # -16% -> de-risk
    m.update(88)
    assert m.active                 # -12%: still above reset threshold
    m.update(91)
    assert not m.active             # -9%: recovered


def test_gross_cap_logic(cfg):
    assert gross_exposure_cap(cfg.risk, True, False) == 1.0
    assert gross_exposure_cap(cfg.risk, False, False) == cfg.risk.regime_gross_cap
    assert gross_exposure_cap(cfg.risk, True, True) == cfg.risk.drawdown_gross_cap


def test_limit_breach_detection(cfg):
    w = pd.Series({"A": 0.20, "B": 0.05})
    breaches = check_limits(w, {"A": "Tech", "B": "Tech"}, pd.Series({"A": 0.3, "B": 0.2}), cfg.portfolio)
    assert any("A: weight" in b for b in breaches)
    assert any("A: risk" in b for b in breaches)


def test_round_trips_ignore_sells_without_cost_basis():
    d = pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"])
    fills = pd.DataFrame({
        "date": d, "ticker": ["A", "A", "A"], "side": ["SELL", "BUY", "SELL"],
        "shares": [10, 5, 5], "gross_value": [1000.0, 500.0, 600.0],
        "commission": [0.0] * 3, "reason": ["pre-period", "", "x"]})
    trips = M.round_trips(fills)
    assert len(trips) == 1
    assert trips.iloc[0]["pnl"] == pytest.approx(100.0)
    assert trips.iloc[0]["return"] == pytest.approx(0.2)
