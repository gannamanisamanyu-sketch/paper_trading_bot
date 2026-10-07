"""Stress testing: historical episodes and hypothetical shocks.

Historical: how did the strategy (as actually simulated) behave during
well-known market sell-offs compared with SPY?

Hypothetical: what would the CURRENT portfolio lose under instantaneous
shocks? Losses are estimated with each holding's beta (market shock) or
sector membership (sector shock). These are first-order linear estimates,
not predictions.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

HISTORICAL_EPISODES: List[Dict[str, str]] = [
    {"name": "Q4 2018 sell-off (Fed tightening)", "start": "2018-09-20", "end": "2018-12-24"},
    {"name": "COVID-19 crash", "start": "2020-02-19", "end": "2020-03-23"},
    {"name": "COVID-19 rebound", "start": "2020-03-23", "end": "2020-06-08"},
    {"name": "2022 bear market (inflation/rates)", "start": "2022-01-03", "end": "2022-10-12"},
    {"name": "2023 regional-bank stress", "start": "2023-03-08", "end": "2023-03-17"},
    {"name": "2025 tariff shock", "start": "2025-02-19", "end": "2025-04-08"},
]


def historical_stress(equity: pd.Series, bench: pd.Series) -> pd.DataFrame:
    rows = []
    for ep in HISTORICAL_EPISODES:
        s, e = pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"])
        if s < equity.index[0] or e > equity.index[-1]:
            continue
        eq = equity.loc[:e]
        start_val = eq.loc[:s].iloc[-1]
        window = equity.loc[s:e]
        b = bench.loc[:e]
        b_start = b.loc[:s].iloc[-1]
        rows.append({
            "episode": ep["name"], "start": ep["start"], "end": ep["end"],
            "strategy_return": float(window.iloc[-1] / start_val - 1),
            "spy_return": float(bench.loc[s:e].iloc[-1] / b_start - 1),
            "strategy_max_dd": float((window / window.cummax() - 1).min()),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["difference"] = df["strategy_return"] - df["spy_return"]
    return df


def hypothetical_stress(weights: pd.Series, betas: pd.Series, sectors: Dict[str, str],
                        equity: float) -> pd.DataFrame:
    """Instantaneous P&L of the current portfolio under simple shocks."""
    if weights.empty:
        return pd.DataFrame()
    b = betas.reindex(weights.index).fillna(1.0)
    port_beta = float((weights * b).sum())
    sec = pd.Series({t: sectors.get(t, "Unknown") for t in weights.index})
    rows = []
    for shock in [-0.05, -0.10, -0.20, -0.35]:
        loss = port_beta * shock
        rows.append({"scenario": f"S&P 500 {shock:.0%} (beta-scaled)", "portfolio_return": loss,
                     "dollar_pnl": loss * equity})
    by_sector = weights.groupby(sec).sum().sort_values(ascending=False)
    for s in by_sector.index[:3]:
        loss = -0.25 * by_sector[s]
        rows.append({"scenario": f"{s} sector -25%", "portfolio_return": loss,
                     "dollar_pnl": loss * equity})
    top = weights.idxmax()
    rows.append({"scenario": f"Largest position ({top}) -40% single-stock gap",
                 "portfolio_return": -0.40 * weights[top], "dollar_pnl": -0.40 * weights[top] * equity})
    return pd.DataFrame(rows)
