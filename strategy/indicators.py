"""Technical indicators.

Every function is *causal*: the value on date t uses only data from dates <= t
(rolling windows and ewm look backwards only). This is the first line of
defense against look-ahead bias, and ``tests/test_no_lookahead.py`` verifies it.
All functions work on Series or wide DataFrames (dates x tickers).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

Frame = Union[pd.Series, pd.DataFrame]


def sma(prices: Frame, window: int) -> Frame:
    """Simple moving average. Requires a full window (no partial averages)."""
    return prices.rolling(window, min_periods=window).mean()


def momentum(prices: Frame, lookback: int, skip: int = 0) -> Frame:
    """Total return from t-lookback to t-skip.

    The classic "12-1" momentum (lookback=252, skip=21) skips the most recent
    month, because one-month returns tend to *reverse* (short-term reversal
    effect), while 2-12 month returns tend to *continue* (Jegadeesh & Titman, 1993).
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def rsi(prices: Frame, period: int = 14) -> Frame:
    """Wilder's Relative Strength Index (0-100).

    RSI = 100 - 100 / (1 + RS), with RS = smoothed avg gain / smoothed avg loss,
    using Wilder's smoothing (an EMA with alpha = 1/period).
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # no losses in the window -> RSI = 100 by definition
    out = out.where(avg_loss != 0, 100.0)
    return out.where(avg_gain.notna())


def true_range(high: Frame, low: Frame, close: Frame) -> Frame:
    prev_close = close.shift(1)
    ranges = [high - low, (high - prev_close).abs(), (low - prev_close).abs()]
    if isinstance(high, pd.DataFrame):
        return pd.concat(ranges).groupby(level=0).max().reindex(high.index)
    return pd.concat(ranges, axis=1).max(axis=1)


def atr(high: Frame, low: Frame, close: Frame, period: int = 14) -> Frame:
    """Average True Range (Wilder): average daily price range incl. gaps, in $."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def realized_volatility(prices: Frame, window: int = 63, annualization: int = 252) -> Frame:
    """Annualized standard deviation of daily returns over ``window`` days."""
    rets = prices.pct_change(fill_method=None)
    return rets.rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(annualization)


def rolling_beta(prices: pd.DataFrame, market: pd.Series, window: int = 252) -> pd.DataFrame:
    """Rolling CAPM beta = Cov(r_i, r_m) / Var(r_m)."""
    rets = prices.pct_change(fill_method=None)
    mret = market.pct_change(fill_method=None).reindex(rets.index)
    minp = int(window * 0.8)
    cov = rets.rolling(window, min_periods=minp).cov(mret)
    var = mret.rolling(window, min_periods=minp).var()
    return cov.div(var, axis=0)


def macd(prices: Frame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Frame, Frame, Frame]:
    """MACD line, signal line and histogram.

    Computed for the dashboard only. It is NOT in the score: MACD is a
    difference of moving averages, so it mostly duplicates information already
    in the momentum and trend factors, and adding correlated factors would
    double-count the same signal.
    """
    ema_fast = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig
