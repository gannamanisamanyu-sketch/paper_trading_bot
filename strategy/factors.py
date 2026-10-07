"""Multi-factor scoring model.

Overall score (computed for every eligible stock on each rebalance date t,
using only information available at the close of t):

    Score = 0.30 * Momentum
          + 0.15 * Trend
          + 0.10 * Sector
          + 0.15 * Quality
          + 0.10 * Value
          + 0.10 * Growth
          - 0.10 * Risk

Each component is a cross-sectional z-score (how many standard deviations a
stock is above/below the average stock on that date), winsorized at +/-3 so
one outlier can't dominate. Missing data -> z = 0 (treated as average).

| Component | Definition | Financial rationale |
|-----------|------------|---------------------|
| Momentum  | 12-1 month total return | Winners keep winning over 3-12 months (behavioral under-reaction, slow information diffusion) |
| Trend     | avg of z(price/SMA200 - 1) and z(SMA50/SMA200 - 1) | Long-term trend persistence; avoids "catching falling knives" |
| Sector    | avg 6-month momentum of the stock's sector | Industry momentum (Moskowitz & Grinblatt, 1999) |
| Quality   | avg of z(ROE) and -z(liabilities/equity), within sector | Profitable, conservatively financed firms are more resilient |
| Value     | earnings yield E/P (= 1 / P/E), within sector | Avoid paying any price for momentum; E/P handles negative earnings gracefully, P/E doesn't |
| Growth    | avg of z(revenue growth) and z(earnings growth), within sector | Fundamental support for the price trend |
| Risk      | avg of z(63-day volatility) and z(252-day beta) | Penalize names that add a lot of risk per unit of expected return |

Fundamental factors are ranked *within sector* because "normal" levels differ
across industries (a bank's liabilities/equity of 10x is normal; for a software
company it would be alarming).

The weights are set a priori (momentum-led, roughly balanced otherwise) and
are NOT optimized on backtest results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy import indicators as ind

logger = logging.getLogger(__name__)

COMPONENTS = ["momentum", "trend", "sector", "quality", "value", "growth", "risk"]


def zscore(x: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorized at +/-clip, NaN -> 0 (neutral)."""
    x = x.astype(float)
    valid = x.dropna()
    if len(valid) < 3 or valid.std(ddof=0) == 0:
        return pd.Series(0.0, index=x.index)
    z = (x - valid.mean()) / valid.std(ddof=0)
    return z.clip(-clip, clip).fillna(0.0)


def sector_zscore(x: pd.Series, sectors: pd.Series, clip: float = 3.0,
                  min_group: int = 4) -> pd.Series:
    """Z-score within sector; small sectors fall back to the universe-wide z."""
    out = zscore(x, clip)
    for _, idx in sectors.groupby(sectors).groups.items():
        grp = x.reindex(idx)
        if grp.notna().sum() >= min_group:
            out.loc[idx] = zscore(grp, clip)
    return out


def _nanmean(*series: pd.Series) -> pd.Series:
    return pd.concat(series, axis=1).mean(axis=1)


@dataclass
class FeatureSet:
    """Precomputed, causal indicator panels (date x ticker).

    Computing these once for the whole history is just an optimization: every
    value on date t is a function of data up to t only, so slicing row t gives
    exactly what a trader would have seen on that day.
    """

    adj_close: pd.DataFrame
    momentum: pd.DataFrame
    sector_momentum_raw: pd.DataFrame
    trend_long: pd.DataFrame
    trend_cross: pd.DataFrame
    above_sma: pd.DataFrame
    rsi: pd.DataFrame
    volatility: pd.DataFrame
    beta: pd.DataFrame
    atr: pd.DataFrame
    dollar_volume: pd.DataFrame
    history_days: pd.DataFrame
    recent_valid: pd.DataFrame
    price: pd.DataFrame
    earnings_yield: pd.DataFrame
    roe: pd.DataFrame
    leverage: pd.DataFrame
    revenue_growth: pd.DataFrame
    earnings_growth: pd.DataFrame
    market_above_sma: pd.Series
    sectors: pd.Series

    @classmethod
    def build(cls, md, fundamentals, sector_map: Dict[str, str], scfg) -> "FeatureSet":
        px = md.adj_close
        sma_long = ind.sma(px, scfg.trend_sma)
        sma_fast = ind.sma(px, scfg.fast_sma)
        bench_sma = ind.sma(md.benchmark, scfg.trend_sma)
        eps = fundamentals.get("eps_diluted").reindex_like(px)
        return cls(
            adj_close=px,
            momentum=ind.momentum(px, scfg.momentum_lookback, scfg.momentum_skip),
            sector_momentum_raw=ind.momentum(px, scfg.sector_momentum_lookback, 0),
            trend_long=px / sma_long - 1.0,
            trend_cross=sma_fast / sma_long - 1.0,
            above_sma=px > sma_long,
            rsi=ind.rsi(px, scfg.rsi_period),
            volatility=ind.realized_volatility(px, scfg.vol_lookback),
            beta=ind.rolling_beta(px, md.benchmark, scfg.beta_lookback),
            atr=ind.atr(md.adj_high, md.adj_low, md.adj_close, scfg.atr_period),
            dollar_volume=(md.close * md.volume).rolling(63, min_periods=40).mean(),
            history_days=md.valid.cumsum(),
            recent_valid=md.valid.rolling(20, min_periods=20).sum() >= 20,
            # actual traded price at the time (undo later splits) for the $5 screen;
            # Yahoo's split-adjusted close would make a stock that LATER splits look cheap
            price=md.close * md.split_factor_after,
            earnings_yield=eps / md.close,
            roe=fundamentals.get("roe").reindex_like(px),
            leverage=fundamentals.get("leverage").reindex_like(px),
            revenue_growth=fundamentals.get("revenue_growth").reindex_like(px),
            earnings_growth=fundamentals.get("earnings_growth").reindex_like(px),
            market_above_sma=(md.benchmark > bench_sma).reindex(px.index).fillna(False),
            sectors=pd.Series({t: sector_map.get(t, "Unknown") for t in px.columns}),
        )


class FactorModel:
    """Turns a FeatureSet row (one date) into factor z-scores and a composite score."""

    def __init__(self, features: FeatureSet, scfg):
        self.f = features
        self.cfg = scfg
        self.w = scfg.weights

    def eligible(self, date: pd.Timestamp) -> pd.Series:
        """Liquidity / data-quality screen, using only data known at ``date``."""
        f, c = self.f, self.cfg
        row = lambda panel: panel.loc[date]
        return (
            (row(f.history_days) >= c.min_history_days)
            & row(f.recent_valid).fillna(False).astype(bool)
            & (row(f.price) >= c.min_price)
            & (row(f.dollar_volume) >= c.min_avg_dollar_volume)
            & row(f.momentum).notna()
            & row(f.volatility).notna()
        )

    def score(self, date: pd.Timestamp) -> pd.DataFrame:
        """Factor table for all eligible stocks on ``date``, sorted best first.

        Columns: raw inputs, component z-scores, composite ``score``, ``rank``.
        """
        f, clip = self.f, self.cfg.winsorize_z
        elig = self.eligible(date)
        names = elig[elig].index
        if len(names) == 0:
            return pd.DataFrame(columns=["score", "rank"])
        r = lambda panel: panel.loc[date, names]
        sectors = f.sectors.reindex(names)

        raw = pd.DataFrame({
            "momentum_12_1": r(f.momentum),
            "price_vs_sma200": r(f.trend_long),
            "sma50_vs_sma200": r(f.trend_cross),
            "rsi": r(f.rsi),
            "volatility": r(f.volatility),
            "beta": r(f.beta),
            "earnings_yield": r(f.earnings_yield),
            "roe": r(f.roe),
            "leverage": r(f.leverage),
            "revenue_growth": r(f.revenue_growth),
            "earnings_growth": r(f.earnings_growth),
            "atr": r(f.atr),
            "above_sma200": r(f.above_sma).astype(bool),
            "sector": sectors,
        })
        # Sector momentum: equal-weighted average 6m return of eligible names in each sector
        sec_mom = r(f.sector_momentum_raw).groupby(sectors).transform("mean")
        raw["sector_momentum"] = sec_mom

        comp = pd.DataFrame(index=names)
        comp["momentum"] = zscore(raw["momentum_12_1"], clip)
        comp["trend"] = _nanmean(zscore(raw["price_vs_sma200"], clip),
                                 zscore(raw["sma50_vs_sma200"], clip))
        comp["sector"] = zscore(raw["sector_momentum"], clip)
        comp["quality"] = _nanmean(sector_zscore(raw["roe"], sectors, clip),
                                   -sector_zscore(raw["leverage"], sectors, clip))
        comp["value"] = sector_zscore(raw["earnings_yield"], sectors, clip)
        comp["growth"] = _nanmean(sector_zscore(raw["revenue_growth"], sectors, clip),
                                  sector_zscore(raw["earnings_growth"], sectors, clip))
        comp["risk"] = _nanmean(zscore(raw["volatility"], clip), zscore(raw["beta"], clip))

        w = self.w
        score = (w.momentum * comp["momentum"] + w.trend * comp["trend"]
                 + w.sector * comp["sector"] + w.quality * comp["quality"]
                 + w.value * comp["value"] + w.growth * comp["growth"]
                 - w.risk_penalty * comp["risk"])
        out = raw.join(comp.add_prefix("z_"))
        out["score"] = score
        out = out.sort_values("score", ascending=False)
        out["rank"] = np.arange(1, len(out) + 1)
        out.index.name = "ticker"
        return out

    def market_risk_on(self, date: pd.Timestamp) -> bool:
        """S&P 500 trend filter: True if SPY is above its 200-day SMA."""
        return bool(self.f.market_above_sma.loc[date])
