"""Shared fixtures: a small synthetic market so tests run offline and deterministically."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_config  # noqa: E402
from data.fundamentals import FundamentalData  # noqa: E402
from data.market_data import build_market_data  # noqa: E402

TICKERS = [f"S{i:02d}" for i in range(12)]
SECTORS = {t: ["Tech", "Health", "Energy"][i % 3] for i, t in enumerate(TICKERS)}


def make_prices(n_days: int = 700, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    mkt = rng.normal(0.0004, 0.01, n_days)
    for k, t in enumerate(TICKERS + ["SPY", "^GSPC"]):
        if t in ("SPY", "^GSPC"):
            r = mkt
        else:
            drift = 0.0006 * ((k % 4) - 1.5) / 1.5
            r = drift + 0.9 * mkt + rng.normal(0, 0.012 + 0.002 * (k % 5), n_days)
        close = 50 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        div = np.where((np.arange(n_days) % 63 == 30) & (k % 2 == 0), 0.25, 0.0)
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t, "open": open_, "high": high, "low": low, "close": close,
            "adj_close": close, "volume": 2_000_000.0, "splits": 0.0, "dividends": div}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture(scope="session")
def md(prices):
    rf = pd.Series(0.02, index=pd.to_datetime(prices["date"].unique()))
    return build_market_data(prices, TICKERS, "SPY", "^GSPC", rf)


@pytest.fixture(scope="session")
def fundamentals(md):
    return FundamentalData.empty(md.tickers, md.dates)


@pytest.fixture(scope="session")
def cfg():
    return load_config().override({
        "periods.backtest_start": "2019-01-01",
        "strategy.min_avg_dollar_volume": 1_000_000,
        "strategy.min_history_days": 252,
        "portfolio.max_positions": 5,
    })


@pytest.fixture(scope="session")
def sectors():
    return SECTORS
