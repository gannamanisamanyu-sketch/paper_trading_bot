import numpy as np
import pandas as pd
import pytest

from portfolio.broker import SimulatedBroker
from portfolio.construction import apply_caps, base_weights, construct_portfolio
from portfolio.costs import CostModel


# ------------------------------------------------------------------ costs
def test_costs_always_hurt():
    cm = CostModel(half_spread_bps=2, slippage_bps=5)
    assert cm.fill_price(100.0, +10) == pytest.approx(100.07)
    assert cm.fill_price(100.0, -10) == pytest.approx(99.93)


def test_commission_minimum_and_scaling():
    cm = CostModel(commission_per_share=0.005, min_commission=1.0)
    assert cm.commission(10) == 1.0
    assert cm.commission(1000) == pytest.approx(5.0)
    assert cm.with_total_bps(50).price_impact == pytest.approx(0.005)


# ------------------------------------------------------------------ broker
def test_broker_cash_and_pnl_accounting():
    b = SimulatedBroker(10_000, CostModel(half_spread_bps=0, slippage_bps=0))
    d = pd.Timestamp("2024-01-02")
    b.execute(d, "AAA", 50, 100.0, "buy")
    assert b.cash == pytest.approx(5_000)
    assert b.positions["AAA"].avg_cost == pytest.approx(100.0)
    f = b.execute(d, "AAA", -20, 110.0, "trim")
    assert f.realized_pnl == pytest.approx(200.0)
    assert b.shares("AAA") == 30
    assert b.equity(pd.Series({"AAA": 110.0})) == pytest.approx(5_000 + 2_200 + 3_300)


def test_broker_no_leverage_no_shorting():
    b = SimulatedBroker(1_000, CostModel())
    d = pd.Timestamp("2024-01-02")
    b.execute(d, "AAA", 100, 50.0, "too big")       # wants $5,000 with $1,000 cash
    assert b.cash >= 0
    assert b.shares("AAA") == 19                     # floor(1000 / 50.035)
    b.execute(d, "AAA", -1_000, 50.0, "oversell")
    assert b.shares("AAA") == 0
    assert "AAA" not in b.positions


def test_broker_rejects_invalid_price():
    b = SimulatedBroker(1_000, CostModel())
    assert b.execute(pd.Timestamp("2024-01-02"), "AAA", 5, float("nan"), "x") is None


# ------------------------------------------------------------------ construction
def test_inverse_vol_weights():
    w = base_weights(["A", "B"], pd.Series({"A": 0.2, "B": 0.4}), "volatility")
    assert w["A"] == pytest.approx(2 / 3)
    assert w["B"] == pytest.approx(1 / 3)
    e = base_weights(["A", "B"], pd.Series({"A": 0.2, "B": 0.4}), "equal")
    assert (e == 0.5).all()


def test_caps_respected_and_excess_redistributed():
    w = pd.Series({"A": 0.5, "B": 0.2, "C": 0.2, "D": 0.1})
    caps = pd.Series(0.3, index=w.index)
    sectors = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Z"})
    out = apply_caps(w, caps, sectors, sector_cap=0.45)
    assert (out <= 0.3 + 1e-9).all()
    assert out.groupby(sectors).sum().max() <= 0.45 + 1e-9
    assert out.sum() == pytest.approx(1.0, abs=1e-6)


def test_construct_portfolio_limits(cfg):
    rng = np.random.default_rng(3)
    names = [f"T{i}" for i in range(8)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (126, 8)), columns=names)
    vols = pd.Series(np.linspace(0.15, 0.6, 8), index=names)
    sectors = pd.Series({n: "S1" if i < 5 else "S2" for i, n in enumerate(names)})
    res = construct_portfolio(names, vols, sectors, rets, cfg.portfolio)
    w = res.weights
    p = cfg.portfolio
    assert w.sum() <= 1 - p.min_cash_weight + 1e-9
    assert (w <= p.max_position_weight + 1e-9).all()
    assert ((w * vols) <= p.max_position_risk + 1e-9).all()
    assert w.groupby(sectors).sum().max() <= p.max_sector_weight + 1e-9
    assert res.est_volatility <= p.target_volatility + 1e-6


def test_gross_cap_applied(cfg):
    names = ["A", "B", "C"]
    rets = pd.DataFrame(np.random.default_rng(4).normal(0, 0.005, (126, 3)), columns=names)
    res = construct_portfolio(names, pd.Series(0.1, index=names), pd.Series({"A": "1", "B": "2", "C": "3"}),
                              rets, cfg.portfolio, gross_cap=0.5)
    assert res.weights.sum() <= 0.5 + 1e-9
