"""Market data download, caching and cleaning (Yahoo Finance via yfinance).

Price conventions
-----------------
* ``close``: split-adjusted close, NOT dividend-adjusted (Yahoo's "Close").
* ``adj_close``: split- AND dividend-adjusted close (Yahoo's "Adj Close").
  Returns computed from it are total returns (price change + reinvested
  dividends), which is what an investor actually earns.
* ``adj_open/adj_high/adj_low``: raw OHLC multiplied by the same daily
  adjustment factor (adj_close / close), so all OHLC fields share one basis.

The backtest values positions with adjusted prices, so dividends show up as
return and splits never look like crashes. Share counts in the backtest are
therefore in "adjusted share" units. That is fine for measuring returns, and
the paper-trading engine re-prices everything at actual current prices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]
_YF_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Stock Splits": "splits",
    "Dividends": "dividends",
}


# ---------------------------------------------------------------- download
def download_prices(
    tickers: List[str],
    start: str,
    end: Optional[str],
    cache_file: Path,
    refresh: bool = False,
    batch_size: int = 40,
) -> pd.DataFrame:
    """Download daily OHLCV + split data and cache it as a long-format parquet.

    Returns a long DataFrame with columns
    ``date, ticker, open, high, low, close, adj_close, volume, splits``.
    """
    if cache_file.exists() and not refresh:
        cached = pd.read_parquet(cache_file)
        have = set(cached["ticker"].unique())
        missing = [t for t in tickers if t not in have]
        if "dividends" not in cached.columns:
            missing = missing or ["<dividends column>"]
        if not missing:
            logger.info("Loaded cached prices: %s (%d rows)", cache_file.name, len(cached))
            return cached
        logger.info("Cache is missing %d tickers; re-downloading all", len(missing))

    import yfinance as yf  # imported lazily so tests don't need network access

    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info("Downloading %d tickers (%s ... %s)", len(batch), batch[0], batch[-1])
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            continue
        long = raw.stack(level=1, future_stack=True).reset_index()
        long.columns = [str(c) for c in long.columns]
        long = long.rename(columns={"Date": "date", "Ticker": "ticker", "level_1": "ticker"})
        long = long.rename(columns=_YF_FIELD_MAP)
        keep = ["date", "ticker"] + [c for c in _YF_FIELD_MAP.values() if c in long.columns]
        frames.append(long[keep])

    if not frames:
        raise RuntimeError("No price data downloaded. Check your internet connection.")
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None).dt.normalize()
    for col in ("splits", "dividends"):
        if col not in data.columns:
            data[col] = 0.0
    data = data.dropna(subset=["close"]).sort_values(["ticker", "date"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_file, index=False)
    got = data["ticker"].nunique()
    failed = sorted(set(tickers) - set(data["ticker"].unique()))
    logger.info("Downloaded %d/%d tickers, %d rows -> %s", got, len(tickers), len(data), cache_file)
    if failed:
        logger.warning("No data for: %s", failed)
    return data


def download_risk_free(ticker: str, start: str, end: Optional[str], cache_file: Path,
                       refresh: bool = False) -> pd.Series:
    """Annualized 13-week T-bill yield (decimal), e.g. 0.05 = 5%."""
    if cache_file.exists() and not refresh:
        s = pd.read_parquet(cache_file)["rate"]
        return s
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        logger.warning("Risk-free series unavailable; assuming 0%%")
        return pd.Series(dtype=float, name="rate")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rate = (close / 100.0).rename("rate")
    rate.index = pd.to_datetime(rate.index).tz_localize(None).normalize()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    rate.to_frame().to_parquet(cache_file)
    return rate


# ---------------------------------------------------------------- panel container
@dataclass
class MarketData:
    """Wide (date x ticker) price panels aligned to one trading calendar."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame          # split-adjusted, not dividend-adjusted
    adj_close: pd.DataFrame      # split + dividend adjusted (total return)
    adj_open: pd.DataFrame
    adj_high: pd.DataFrame
    adj_low: pd.DataFrame
    volume: pd.DataFrame
    valid: pd.DataFrame          # True where the stock actually traded that day (not filled)
    split_factor_after: pd.DataFrame  # product of split ratios occurring AFTER each date
    benchmark: pd.Series         # benchmark adjusted close (SPY total return)
    index_level: pd.Series       # S&P 500 price index
    risk_free_daily: pd.Series   # daily risk-free return
    quality_flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    dividends: Optional[pd.DataFrame] = None   # cash dividend per share on the ex-date
    splits: Optional[pd.DataFrame] = None      # split ratio on the split date (1.0 = none)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    @property
    def tickers(self) -> List[str]:
        return list(self.adj_close.columns)

    def truncate(self, end: pd.Timestamp) -> "MarketData":
        """Return a copy containing only data up to and including ``end``.

        Used by the look-ahead-bias unit test: signals computed on truncated
        data must equal signals computed on the full history.
        """
        def cut(x):
            return x.loc[:end]

        return MarketData(
            open=cut(self.open), high=cut(self.high), low=cut(self.low), close=cut(self.close),
            adj_close=cut(self.adj_close), adj_open=cut(self.adj_open),
            adj_high=cut(self.adj_high), adj_low=cut(self.adj_low), volume=cut(self.volume),
            valid=cut(self.valid), split_factor_after=self.split_factor_after.loc[:end],
            benchmark=cut(self.benchmark), index_level=cut(self.index_level),
            risk_free_daily=cut(self.risk_free_daily), quality_flags=self.quality_flags,
            dividends=cut(self.dividends) if self.dividends is not None else None,
            splits=cut(self.splits) if self.splits is not None else None,
        )


def build_market_data(
    prices: pd.DataFrame,
    tickers: List[str],
    benchmark: str,
    index_ticker: str,
    risk_free: pd.Series,
    max_daily_move_flag: float = 0.25,
    max_ffill_days: int = 5,
) -> MarketData:
    """Clean the long price table and pivot it into aligned wide panels.

    Cleaning steps (each one is explainable in an interview):
    1. Trading calendar = the benchmark's trading days (SPY trades every
       NYSE session). Rows on other dates are dropped.
    2. Non-positive prices are treated as missing (bad ticks).
    3. Missing prices are forward-filled for at most ``max_ffill_days`` days,
       only so open positions can be marked to market. A separate ``valid``
       mask records the real observations, and a stock must have real data
       recently to be *eligible* for new trades.
    4. Daily moves larger than ``max_daily_move_flag`` are logged as
       data-quality flags. They are kept because they may be real (earnings,
       spin-offs), but they are reported.
    5. Adjusted OHLC is built from the adj_close/close ratio so splits and
       dividends never look like price crashes.
    """
    prices = prices.copy()
    for col in ["open", "high", "low", "close", "adj_close"]:
        prices.loc[prices[col] <= 0, col] = np.nan

    def pivot(col: str) -> pd.DataFrame:
        return prices.pivot_table(index="date", columns="ticker", values=col, aggfunc="last")

    bench_close = pivot("adj_close")[benchmark].dropna()
    calendar = bench_close.index
    if index_ticker in prices["ticker"].unique():
        index_level = pivot("close")[index_ticker].reindex(calendar).ffill()
    else:
        index_level = bench_close.copy()

    present = [t for t in tickers if t in prices["ticker"].unique()]
    sub = prices[prices["ticker"].isin(present)]

    def wide(col: str) -> pd.DataFrame:
        return (
            sub.pivot_table(index="date", columns="ticker", values=col, aggfunc="last")
            .reindex(index=calendar, columns=present)
        )

    close_raw = wide("close")
    valid = close_raw.notna()
    close = close_raw.ffill(limit=max_ffill_days)
    adj_close = wide("adj_close").ffill(limit=max_ffill_days)
    open_ = wide("open").ffill(limit=max_ffill_days)
    high = wide("high").ffill(limit=max_ffill_days)
    low = wide("low").ffill(limit=max_ffill_days)
    volume = wide("volume").fillna(0.0)

    factor = adj_close / close
    adj_open = open_ * factor
    adj_high = high * factor
    adj_low = low * factor

    # Split factor AFTER each date: used only to put reported EPS on today's share basis
    splits = wide("splits").fillna(0.0).replace(0.0, 1.0)
    # cumulative product of all split ratios strictly after date t
    rev_cum = splits.iloc[::-1].cumprod().iloc[::-1]
    split_factor_after = (rev_cum / splits).fillna(1.0)

    # data-quality flags
    rets = adj_close.pct_change(fill_method=None)
    big = rets.abs() > max_daily_move_flag
    flags = (
        rets[big].stack().rename("return").reset_index().rename(columns={"level_0": "date"})
        if big.any().any() else pd.DataFrame(columns=["date", "ticker", "return"])
    )
    if len(flags):
        logger.info("Data-quality flags: %d daily moves > %.0f%% (kept, see quality report)",
                    len(flags), max_daily_move_flag * 100)

    rf_annual = risk_free.reindex(calendar).ffill().bfill().fillna(0.0)
    rf_daily = rf_annual / 252.0

    dividends = wide("dividends").fillna(0.0) if "dividends" in sub else None

    return MarketData(
        open=open_, high=high, low=low, close=close, adj_close=adj_close,
        adj_open=adj_open, adj_high=adj_high, adj_low=adj_low, volume=volume,
        valid=valid, split_factor_after=split_factor_after,
        benchmark=bench_close, index_level=index_level, risk_free_daily=rf_daily,
        quality_flags=flags, dividends=dividends, splits=splits,
    )


def load_market_data(cfg, refresh: bool = False) -> "tuple[MarketData, object]":
    """Convenience: universe + download/cache + clean in one call."""
    from data.universe import load_universe

    universe = load_universe(cfg.path(cfg.data.universe_file))
    cache = cfg.path(cfg.data.cache_dir)
    all_tickers = universe.tradable + [cfg.data.benchmark, cfg.data.index_ticker]
    prices = download_prices(all_tickers, cfg.data.start_date, cfg.data.end_date,
                             cache / "prices.parquet", refresh=refresh)
    rf = download_risk_free(cfg.data.risk_free_ticker, cfg.data.start_date, cfg.data.end_date,
                            cache / "risk_free.parquet", refresh=refresh)
    md = build_market_data(prices, universe.tradable, cfg.data.benchmark, cfg.data.index_ticker,
                           rf, cfg.data.max_daily_move_flag)
    return md, universe
