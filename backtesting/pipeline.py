"""Glue code: load data once, then run any number of backtest variants on it."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from backtesting.engine import Backtester, BacktestResult
from data.fundamentals import FundamentalData, load_fundamentals
from data.market_data import MarketData, load_market_data
from data.universe import Universe
from portfolio.costs import CostModel
from risk.metrics import performance_summary, round_trips
from strategy.factors import FeatureSet

logger = logging.getLogger(__name__)

# strategy settings that change the precomputed indicator panels
_FEATURE_KEYS = ("momentum_lookback", "momentum_skip", "sector_momentum_lookback", "trend_sma",
                 "fast_sma", "rsi_period", "vol_lookback", "beta_lookback", "atr_period")


@dataclass
class DataContext:
    md: MarketData
    universe: Universe
    fundamentals: FundamentalData

    def __post_init__(self) -> None:
        self._feature_cache: Dict[Tuple, FeatureSet] = {}

    def features(self, scfg) -> FeatureSet:
        key = tuple(getattr(scfg, k) for k in _FEATURE_KEYS)
        if key not in self._feature_cache:
            self._feature_cache[key] = FeatureSet.build(self.md, self.fundamentals,
                                                        self.universe.sector_map, scfg)
        return self._feature_cache[key]


def load_context(cfg, refresh: bool = False) -> DataContext:
    md, universe = load_market_data(cfg, refresh=refresh)
    fundamentals = load_fundamentals(cfg, md, refresh=refresh)
    return DataContext(md, universe, fundamentals)


def run_backtest(cfg, ctx: DataContext, start: Optional[str] = None, end: Optional[str] = None,
                 cost_model: Optional[CostModel] = None, record_scores: bool = True) -> BacktestResult:
    bt = Backtester(cfg, ctx.md, ctx.features(cfg.strategy), ctx.universe.sector_map, cost_model)
    return bt.run(start=start, end=end, record_scores=record_scores)


def summarize(result: BacktestResult, ctx: DataContext, cfg,
              start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Dict]:
    """Strategy + SPY metrics, optionally restricted to a sub-period of the run."""
    eq = result.equity_curve
    bench = result.benchmark
    fills = result.fills
    s, e = eq.index[0], eq.index[-1]
    if start or end:
        s = pd.Timestamp(start) if start else eq.index[0]
        e = pd.Timestamp(end) if end else eq.index[-1]
        # include the prior close so the first day's return is counted
        prior = eq.index[eq.index < s]
        s0 = prior[-1] if len(prior) else s
        eq, bench = eq.loc[s0:e], bench.loc[s0:e]
    # Round trips are built from the COMPLETE fill history (so positions opened
    # before the period keep their true cost basis) and assigned to the period
    # in which they closed (or are still open at its end).
    hist = fills[fills["date"] <= e] if len(fills) else fills
    trips = round_trips(hist, ctx.md.adj_close.loc[eq.index[-1]], eq.index[-1])
    if len(trips):
        trips = trips[trips["open"] | (trips["exit_date"] >= s)]
    if len(fills):
        fills = fills[(fills["date"] >= s) & (fills["date"] <= e)]
    strat = performance_summary(eq, bench, ctx.md.risk_free_daily, cfg.risk.var_confidence,
                                fills=fills, trips=trips)
    strat["avg_gross_exposure"] = float(result.equity["gross_exposure"].loc[eq.index].mean())
    spy = performance_summary(bench, bench, ctx.md.risk_free_daily, cfg.risk.var_confidence)
    return {"strategy": strat, "spy": spy}


def period_bounds(cfg, md_last: pd.Timestamp) -> Dict[str, Tuple[str, str]]:
    p = cfg.periods
    val_start = (pd.Timestamp(p.train_end) + pd.Timedelta(days=1)).date().isoformat()
    test_start = (pd.Timestamp(p.validation_end) + pd.Timedelta(days=1)).date().isoformat()
    return {
        "train": (p.backtest_start, p.train_end),
        "validation": (val_start, p.validation_end),
        "test": (test_start, md_last.date().isoformat()),
        "full": (p.backtest_start, md_last.date().isoformat()),
    }
