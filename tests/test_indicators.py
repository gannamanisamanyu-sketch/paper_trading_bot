import numpy as np
import pandas as pd
import pytest

from strategy import indicators as ind


def test_sma_requires_full_window():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = ind.sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[-1] == pytest.approx(4.0)


def test_momentum_skips_recent_month():
    s = pd.Series(np.arange(1.0, 31.0))  # 1..30
    m = ind.momentum(s, lookback=10, skip=2)
    # at the last index (29): price[27] / price[19] - 1 = 28/20 - 1
    assert m.iloc[-1] == pytest.approx(28 / 20 - 1)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1.0, 40.0))
    assert ind.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(40.0, 1.0, -1))
    assert ind.rsi(down, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(0)
    noisy = pd.Series(100 + rng.normal(0, 1, 300).cumsum())
    r = ind.rsi(noisy, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_atr_constant_range():
    n = 60
    close = pd.Series(np.full(n, 100.0))
    high, low = close + 1.0, close - 1.0
    a = ind.atr(high, low, close, 14)
    assert a.iloc[-1] == pytest.approx(2.0)


def test_realized_vol_matches_definition():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 300)
    px = pd.Series(100 * np.exp(np.cumsum(r)))
    v = ind.realized_volatility(px, 63).iloc[-1]
    expected = px.pct_change().iloc[-63:].std() * np.sqrt(252)
    assert v == pytest.approx(expected)


def test_beta_of_market_is_one():
    rng = np.random.default_rng(2)
    m = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    df = pd.DataFrame({"A": m, "B": m ** 2})  # B has ~2x the log-returns
    b = ind.rolling_beta(df, m, 252).iloc[-1]
    assert b["A"] == pytest.approx(1.0)
    assert b["B"] == pytest.approx(2.0, rel=0.05)
