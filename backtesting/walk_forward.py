"""Walk-forward (expanding-window) evaluation.

Why: a single backtest lets the researcher (consciously or not) pick the
configuration that happened to work best over the whole sample, and then
report that same sample as the result. Walk-forward testing simulates how the
strategy would actually have been run: at the start of each year, choose a
configuration using ONLY the years before it, then trade that choice for the
next year. The stitched sequence of those "next years" is a genuinely
out-of-sample track record.

Implementation:
* A small, pre-declared candidate set (4 configs in config.yaml), so there
  are very few "knobs" to overfit.
* Each candidate is backtested continuously from the start date.
* For each test year Y: pick the candidate with the highest Sharpe ratio over
  [start, end of Y-1], then take that candidate's daily returns during Y.
* Simplification (documented): the stitched series switches between
  continuously-run candidate portfolios, so it ignores the one-time cost of
  switching holdings when the chosen candidate changes.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from backtesting.pipeline import DataContext, run_backtest
from risk.metrics import performance_summary, sharpe_ratio

logger = logging.getLogger(__name__)


def run_walk_forward(cfg, ctx: DataContext) -> Dict[str, object]:
    wf = cfg.walk_forward
    curves: Dict[str, pd.Series] = {}
    for cand in wf.candidates:
        overrides = {f"strategy.momentum_lookback": cand["momentum_lookback"],
                     f"portfolio.sizing_method": cand["sizing_method"]}
        ccfg = cfg.override(overrides)
        logger.info("Walk-forward candidate %s", cand["name"])
        res = run_backtest(ccfg, ctx, record_scores=False)
        curves[cand["name"]] = res.equity_curve
        bench = res.benchmark
    rets = pd.DataFrame({k: v.pct_change() for k, v in curves.items()}).dropna()
    rf = ctx.md.risk_free_daily

    years = sorted(set(rets.index.year))
    choices: List[Dict[str, object]] = []
    stitched = []
    for y in years:
        if y < wf.first_test_year:
            continue
        hist = rets[rets.index.year < y]
        if len(hist) < 252:
            continue
        sharpes = {k: sharpe_ratio(hist[k], rf) for k in rets.columns}
        best = max(sharpes, key=lambda k: sharpes[k] if np.isfinite(sharpes[k]) else -np.inf)
        yr = rets.loc[rets.index.year == y, best]
        stitched.append(yr)
        choices.append({"year": y, "chosen": best, "in_sample_sharpe": sharpes[best],
                        "out_of_sample_return": float((1 + yr).prod() - 1),
                        **{f"sharpe_{k}": v for k, v in sharpes.items()}})

    oos = pd.concat(stitched)
    start_capital = cfg.portfolio.initial_capital
    oos_equity = start_capital * (1 + oos).cumprod()
    first = oos.index[0]
    prior = rets.index[rets.index < first][-1]
    oos_equity = pd.concat([pd.Series([start_capital], index=[prior]), oos_equity])
    b = bench.loc[prior:]
    bench_equity = start_capital * b / b.iloc[0]
    return {
        "choices": pd.DataFrame(choices),
        "oos_equity": oos_equity.rename("walk_forward"),
        "bench_equity": bench_equity.rename("SPY"),
        "summary": performance_summary(oos_equity, bench_equity, rf),
        "spy_summary": performance_summary(bench_equity, bench_equity, rf),
        "candidate_curves": pd.DataFrame(curves),
    }
