"""Portfolio construction: turn a list of selected stocks into target weights.

Pipeline (applied in this order):

1. **Base weights**
   * Equal weight: every name gets 1/N.
     + simple, no estimation error, gives every idea the same conviction
     - a 60%-vol stock contributes ~3x the risk of a 20%-vol stock
   * Volatility-adjusted (inverse volatility): w_i ~ 1 / sigma_i.
     + each name contributes roughly similar risk, so no single volatile stock dominates
     - depends on a volatility estimate (backward-looking, noisy), tilts toward
       defensive low-vol names, and can underweight the strongest growth stocks
2. **Gross scaling** so that cash >= ``min_cash_weight``.
3. **Position caps**: w_i <= min(max_position_weight, max_position_risk / sigma_i).
   The second term is the *position-level risk limit*: a 1-sigma annual move in
   one stock may cost at most ``max_position_risk`` (3%) of the portfolio.
4. **Sector caps**: sum of weights per sector <= ``max_sector_weight``.
   Steps 3-4 use iterative "water-filling": weight trimmed from a capped
   name/sector is redistributed to names that still have room. Anything that
   cannot be placed stays in cash.
5. **Volatility targeting**: estimate the portfolio's annualized volatility
   from the trailing covariance matrix. If it exceeds ``target_volatility``,
   scale every position down proportionally (the rest goes to cash).
6. **Gross-exposure caps** from the risk overlay (market-regime filter and
   drawdown control): gross exposure <= cap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConstructionResult:
    weights: pd.Series            # final target weights (fraction of equity)
    est_volatility: float         # estimated annualized vol of the final portfolio
    pre_target_volatility: float  # estimated vol before the vol-target scaling
    vol_scale: float              # multiplier applied by vol targeting (<= 1)
    gross_cap: float              # cap applied by the risk overlay
    notes: List[str]


def base_weights(tickers: List[str], vols: pd.Series, method: str) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    if method == "equal":
        w = pd.Series(1.0, index=tickers)
    elif method == "volatility":
        v = vols.reindex(tickers)
        v = v.fillna(v.median() if v.notna().any() else 0.25).clip(lower=0.05)
        w = 1.0 / v
    else:
        raise ValueError(f"Unknown sizing method: {method}")
    return w / w.sum()


def apply_caps(weights: pd.Series, name_caps: pd.Series, sectors: pd.Series,
               sector_cap: float, max_iter: int = 50) -> pd.Series:
    """Iteratively enforce per-name and per-sector caps, redistributing excess.

    The total weight is preserved where possible; weight that cannot be placed
    anywhere (all names/sectors capped) is dropped, i.e. left in cash.
    """
    target_total = weights.sum()
    w = weights.copy()
    for _ in range(max_iter):
        # 1) name caps
        w = np.minimum(w, name_caps.reindex(w.index))
        # 2) sector caps (scale the whole sector down proportionally)
        sec_sum = w.groupby(sectors.reindex(w.index)).transform("sum")
        over = sec_sum > sector_cap + 1e-12
        w[over] = w[over] * sector_cap / sec_sum[over]
        shortfall = target_total - w.sum()
        if shortfall <= 1e-9:
            break
        # 3) redistribute shortfall to names with room in both their name and sector cap
        sec_sum = w.groupby(sectors.reindex(w.index)).transform("sum")
        room = np.minimum(name_caps.reindex(w.index) - w, sector_cap - sec_sum).clip(lower=0)
        room = room[room > 1e-9]
        if room.empty:
            break
        add = (w[room.index] / w[room.index].sum()) * shortfall
        add = np.minimum(add, room)
        if add.sum() <= 1e-9:
            break
        w[room.index] += add
    return w


def estimate_portfolio_vol(weights: pd.Series, returns_window: pd.DataFrame) -> float:
    """Annualized portfolio volatility sqrt(w' Sigma w * 252) from trailing daily returns."""
    names = [t for t in weights.index if t in returns_window.columns and weights[t] != 0]
    if not names:
        return 0.0
    r = returns_window[names].dropna(how="all").fillna(0.0)
    if len(r) < 20:
        return float("nan")
    cov = r.cov().values * 252.0
    w = weights[names].values
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def construct_portfolio(
    tickers: List[str],
    vols: pd.Series,
    sectors: pd.Series,
    returns_window: pd.DataFrame,
    pcfg,
    gross_cap: float = 1.0,
) -> ConstructionResult:
    notes: List[str] = []
    if not tickers:
        return ConstructionResult(pd.Series(dtype=float), 0.0, 0.0, 1.0, gross_cap, ["no holdings"])

    w = base_weights(tickers, vols, pcfg.sizing_method)
    w = w * (1.0 - pcfg.min_cash_weight)

    v = vols.reindex(tickers).fillna(vols.median() if len(vols) else 0.25).clip(lower=0.05)
    name_caps = np.minimum(pcfg.max_position_weight, pcfg.max_position_risk / v)
    w = apply_caps(w, name_caps, sectors, pcfg.max_sector_weight)

    pre_vol = estimate_portfolio_vol(w, returns_window)
    scale = 1.0
    if pcfg.target_volatility and pre_vol and np.isfinite(pre_vol) and pre_vol > pcfg.target_volatility:
        scale = pcfg.target_volatility / pre_vol
        notes.append(f"vol target: est. vol {pre_vol:.1%} > {pcfg.target_volatility:.0%}, scaled x{scale:.2f}")
    w = w * scale

    if w.sum() > gross_cap:
        notes.append(f"gross exposure capped at {gross_cap:.0%} by risk overlay")
        w = w * gross_cap / w.sum()

    final_vol = estimate_portfolio_vol(w, returns_window)
    return ConstructionResult(w, final_vol, pre_vol, scale, gross_cap, notes)
