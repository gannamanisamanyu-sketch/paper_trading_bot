"""Strategy analysis: where did returns (and losses) come from?

* Factor information coefficients (IC): the Spearman rank correlation between
  each factor's z-score on a rebalance date and each stock's return over the
  following period. IC > 0 means the factor ranked stocks in the right
  order. Its t-stat (mean / std x sqrt(n)) says whether that is
  distinguishable from luck (|t| > 2).
* Quintile spread: next-period return of the top-20% minus bottom-20% of
  stocks by each factor.
* Contribution by stock and by sector: realized + unrealized P&L in dollars.
* Return decomposition: compare the strategy with a "beta-matched" SPY
  portfolio (beta x SPY + (1 - beta) x T-bills). If the strategy can't beat
  its own beta-matched benchmark, its returns came from market exposure, not
  from stock selection.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

FACTOR_COLS = ["z_momentum", "z_trend", "z_sector", "z_quality", "z_value", "z_growth",
               "z_risk", "score"]


def factor_ic(scores: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per-rebalance IC for every factor (rows = rebalance dates)."""
    dates = sorted(scores["date"].unique())
    rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        s = scores[scores["date"] == d0].set_index("ticker")
        fwd = (prices.loc[d1] / prices.loc[d0] - 1).reindex(s.index)
        row = {"date": d0}
        for c in FACTOR_COLS:
            if c in s:
                row[c] = s[c].corr(fwd, method="spearman")
        q = pd.qcut(s["score"].rank(method="first"), 5, labels=False)
        row["top_minus_bottom_quintile"] = fwd[q == 4].mean() - fwd[q == 0].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def ic_summary(ic: pd.DataFrame, bounds: Dict[str, tuple]) -> pd.DataFrame:
    out = []
    for name, (s, e) in bounds.items():
        sub = ic.loc[s:e]
        if len(sub) < 3:
            continue
        for c in FACTOR_COLS:
            if c not in sub:
                continue
            x = sub[c].dropna()
            out.append({"period": name, "factor": c.replace("z_", ""), "mean_ic": x.mean(),
                        "ic_t_stat": x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else np.nan,
                        "hit_rate": (x > 0).mean(), "n": len(x)})
    return pd.DataFrame(out)


def contribution_by_stock(fills: pd.DataFrame, positions_last: pd.Series,
                          names: Dict[str, str], sectors: Dict[str, str]) -> pd.DataFrame:
    """Total $ P&L per ticker = sell proceeds + current value - buy costs."""
    if fills.empty:
        return pd.DataFrame()
    signed = np.where(fills["side"] == "SELL", 1.0, -1.0)
    cash_flow = signed * fills["gross_value"] - fills["commission"]
    pnl = cash_flow.groupby(fills["ticker"]).sum()
    pnl = pnl.add(positions_last.reindex(pnl.index).fillna(0.0), fill_value=0.0)
    df = pnl.rename("pnl").to_frame()
    df["name"] = df.index.map(names)
    df["sector"] = df.index.map(sectors)
    df["trades"] = fills.groupby("ticker").size().reindex(df.index)
    return df.sort_values("pnl", ascending=False)


def contribution_by_sector(stock_pnl: pd.DataFrame) -> pd.DataFrame:
    if stock_pnl.empty:
        return pd.DataFrame()
    g = stock_pnl.groupby("sector")["pnl"].agg(["sum", "count"]).rename(
        columns={"sum": "pnl", "count": "stocks_traded"})
    return g.sort_values("pnl", ascending=False)


def beta_matched_comparison(equity: pd.Series, bench: pd.Series, rf_daily: pd.Series,
                            beta: float) -> Dict[str, float]:
    r = equity.pct_change().dropna()
    br = bench.pct_change().reindex(r.index).fillna(0.0)
    rf = rf_daily.reindex(r.index).fillna(0.0)
    matched = beta * br + (1 - beta) * rf
    years = (r.index[-1] - r.index[0]).days / 365.25
    ann = lambda x: (1 + x).prod() ** (1 / years) - 1
    return {"strategy_cagr": ann(r), "beta_matched_cagr": ann(matched),
            "spy_cagr": ann(br), "beta": beta,
            "selection_and_timing_effect": ann(r) - ann(matched)}


def worst_periods(equity: pd.Series, n: int = 5) -> pd.DataFrame:
    m = equity.resample("ME").last().pct_change().dropna()
    return m.nsmallest(n).rename("monthly_return").to_frame()
