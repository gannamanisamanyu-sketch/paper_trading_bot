"""Risk overlay: stop-losses, drawdown control, market regime and limit checks.

These rules sit on top of the stock-selection model and can only REDUCE risk.
They never add exposure.

* **Trailing volatility stop** (default): exit if the close falls more than
  m x (monthly volatility) below the highest close since entry, where monthly
  vol = annual vol x sqrt(21/252). With m = 2 a 25%-vol stock is stopped ~14%
  below its peak, a 15%-vol utility ~9% below. The stop is scaled to the
  strategy's ~1-month holding horizon, so it reacts to a broken trend rather
  than to normal daily noise. A fixed -10% stop would fire constantly on volatile
  names and almost never on quiet ones.
  (An ATR-based stop, k x ATR(14), is available for comparison. In training it
  proved far too tight for a monthly strategy; see RESEARCH_LOG.md.)
* **Drawdown control**: if the portfolio falls more than ``drawdown_limit``
  from its peak, gross exposure is capped (e.g. at 50%) until the drawdown
  recovers below ``drawdown_reset``. The gap between the two levels
  (hysteresis) prevents flip-flopping around a single threshold.
* **Market regime filter**: if SPY is below its 200-day SMA, gross exposure is
  capped. Long-only momentum strategies historically suffer most in bear
  markets and sharp rebounds ("momentum crashes", Daniel & Moskowitz 2016).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


def stop_level(peak: float, method: str, annual_vol: float = np.nan, atr: float = np.nan,
               vol_multiple: float = 2.0, atr_multiple: float = 3.0,
               horizon_days: int = 21) -> float:
    """Price below which the trailing stop fires."""
    if method == "volatility":
        return peak * (1.0 - vol_multiple * annual_vol * np.sqrt(horizon_days / 252.0))
    return peak - atr_multiple * atr


def stop_triggered(price: float, level: float) -> bool:
    """True if the close is below the stop level."""
    if not (np.isfinite(price) and np.isfinite(level)):
        return False
    return price < level


@dataclass
class DrawdownMonitor:
    limit: float
    reset: float
    peak: float = 0.0
    active: bool = False  # True = de-risking mode

    def update(self, equity: float) -> float:
        """Update with today's equity; returns the current drawdown (<= 0)."""
        self.peak = max(self.peak, equity)
        dd = equity / self.peak - 1.0 if self.peak > 0 else 0.0
        if not self.active and dd <= -self.limit:
            self.active = True
        elif self.active and dd >= -self.reset:
            self.active = False
        return dd


def gross_exposure_cap(rcfg, market_risk_on: bool, drawdown_active: bool) -> float:
    cap = 1.0
    if rcfg.market_regime_filter and not market_risk_on:
        cap = min(cap, rcfg.regime_gross_cap)
    if drawdown_active:
        cap = min(cap, rcfg.drawdown_gross_cap)
    return cap


def check_limits(weights: pd.Series, sectors: Dict[str, str], vols: pd.Series, pcfg,
                 tolerance: float = 0.02) -> List[str]:
    """Return human-readable breaches of position / sector / cash / risk limits.

    Weights drift between rebalances as prices move, so a small tolerance is
    allowed before a breach is reported.
    """
    breaches: List[str] = []
    if weights.empty:
        return breaches
    for t, w in weights.items():
        if w > pcfg.max_position_weight + tolerance:
            breaches.append(f"{t}: weight {w:.1%} > max {pcfg.max_position_weight:.0%}")
        v = vols.get(t, np.nan)
        if np.isfinite(v) and w * v > pcfg.max_position_risk + tolerance / 2:
            breaches.append(f"{t}: risk {w * v:.1%} of equity > limit {pcfg.max_position_risk:.0%}")
    sec = weights.groupby(pd.Series({t: sectors.get(t, "Unknown") for t in weights.index})).sum()
    for s, w in sec.items():
        if w > pcfg.max_sector_weight + tolerance:
            breaches.append(f"sector {s}: {w:.1%} > max {pcfg.max_sector_weight:.0%}")
    cash = 1.0 - weights.sum()
    if cash < pcfg.min_cash_weight - tolerance:
        breaches.append(f"cash {cash:.1%} < minimum {pcfg.min_cash_weight:.0%}")
    return breaches
