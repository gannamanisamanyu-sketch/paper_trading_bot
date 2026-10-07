"""Performance and risk metrics.

| Metric | What it tells you |
|--------|-------------------|
| Total return / CAGR | How much money was made, per year compounded |
| Annualized volatility | How bumpy the ride was (std of daily returns x sqrt(252)) |
| Sharpe ratio | Excess return (over T-bills) per unit of TOTAL volatility: the standard risk-adjusted return measure |
| Sortino ratio | Like Sharpe, but only penalizes DOWNSIDE volatility. Upside volatility is not "risk" to an investor |
| Maximum drawdown | Worst peak-to-trough loss: what an investor actually lives through and may capitulate at |
| VaR (95%, 1-day) | A loss that is exceeded on only 5% of days |
| Expected Shortfall (CVaR) | The AVERAGE loss on those worst 5% of days. Unlike VaR, it describes how bad the tail is |
| Beta | Sensitivity to the market: beta 0.8 means ~0.8% move per 1% S&P move |
| Alpha (CAPM) | Return NOT explained by market exposure: the intercept of excess returns regressed on market excess returns |
| Information ratio | Active return vs. benchmark per unit of tracking error |
| Win rate / profit factor | Share of winning trades; gross profits / gross losses (> 1 means winners outweigh losers) |

All ratios use daily data annualized with 252 trading days.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf_daily: Optional[pd.Series] = None) -> float:
    excess = returns - (rf_daily.reindex(returns.index).fillna(0.0) if rf_daily is not None else 0.0)
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def sortino_ratio(returns: pd.Series, rf_daily: Optional[pd.Series] = None) -> float:
    excess = returns - (rf_daily.reindex(returns.index).fillna(0.0) if rf_daily is not None else 0.0)
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def max_drawdown_details(equity: pd.Series) -> Dict[str, object]:
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    rec = equity.loc[trough:]
    recovered = rec[rec >= equity.loc[peak]]
    return {"peak": peak, "trough": trough,
            "recovery": recovered.index[0] if len(recovered) else None,
            "depth": float(dd.min())}


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical 1-day VaR, reported as a positive loss fraction."""
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian 1-day VaR = -(mu + z * sigma). Understates fat tails; shown for comparison."""
    z = stats.norm.ppf(1 - confidence)
    return float(-(returns.mean() + z * returns.std(ddof=1)))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical 1-day Expected Shortfall (CVaR): mean loss beyond VaR, positive number."""
    r = returns.dropna()
    cutoff = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


def beta_alpha(returns: pd.Series, bench_returns: pd.Series,
               rf_daily: Optional[pd.Series] = None) -> Dict[str, float]:
    """CAPM regression of daily excess returns: r_p - rf = alpha + beta (r_m - rf) + e.

    Alpha is annualized (daily intercept x 252). The t-stat tells whether alpha
    is statistically distinguishable from zero (|t| > ~2).
    """
    df = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
    if len(df) < 30:
        return {"beta": float("nan"), "alpha": float("nan"), "alpha_t": float("nan"), "r2": float("nan")}
    rf = rf_daily.reindex(df.index).fillna(0.0) if rf_daily is not None else 0.0
    y = df.iloc[:, 0] - rf
    x = df.iloc[:, 1] - rf
    res = stats.linregress(x, y)
    n = len(df)
    resid = y - (res.intercept + res.slope * x)
    se_int = np.sqrt((resid ** 2).sum() / (n - 2)) * np.sqrt(1 / n + x.mean() ** 2 / ((x - x.mean()) ** 2).sum())
    return {"beta": float(res.slope), "alpha": float(res.intercept * TRADING_DAYS),
            "alpha_t": float(res.intercept / se_int) if se_int > 0 else float("nan"),
            "r2": float(res.rvalue ** 2)}


def information_ratio(returns: pd.Series, bench_returns: pd.Series) -> float:
    active = (returns - bench_returns.reindex(returns.index)).dropna()
    te = active.std(ddof=1)
    return float(active.mean() / te * np.sqrt(TRADING_DAYS)) if te > 0 else float("nan")


# ------------------------------------------------------------------ trade statistics
def round_trips(fills: pd.DataFrame, last_prices: Optional[pd.Series] = None,
                end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Group fills into position "episodes" (from 0 shares back to 0 shares).

    Partial trims during rebalancing belong to the same episode. P&L = cash
    received from sells - cash paid for buys (incl. costs). Positions still
    open at the end are marked to market and flagged ``open=True``. Pass the
    COMPLETE fill history: sells without a preceding buy are ignored.
    """
    if fills is None or fills.empty:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "pnl", "return",
                                     "holding_days", "open", "exit_reason"])
    trips = []
    for ticker, g in fills.sort_values("date", kind="stable").groupby("ticker", sort=False):
        shares, invested, proceeds, entry = 0.0, 0.0, 0.0, None
        for _, f in g.iterrows():
            if f["side"] == "BUY":
                if shares <= 1e-9:
                    entry, invested, proceeds = f["date"], 0.0, 0.0
                shares += f["shares"]
                invested += f["gross_value"] + f["commission"]
            else:
                if entry is None:
                    continue  # sell of a position opened before these fills begin: no cost basis
                shares -= f["shares"]
                proceeds += f["gross_value"] - f["commission"]
                if shares <= 1e-9 and entry is not None:
                    trips.append({"ticker": ticker, "entry_date": entry, "exit_date": f["date"],
                                  "pnl": proceeds - invested, "return": proceeds / invested - 1,
                                  "holding_days": (f["date"] - entry).days, "open": False,
                                  "exit_reason": f["reason"]})
                    entry, shares = None, 0.0
        if shares > 1e-9 and entry is not None and last_prices is not None:
            mv = shares * float(last_prices.get(ticker, np.nan))
            trips.append({"ticker": ticker, "entry_date": entry, "exit_date": end_date,
                          "pnl": proceeds + mv - invested, "return": (proceeds + mv) / invested - 1,
                          "holding_days": (end_date - entry).days if end_date is not None else np.nan,
                          "open": True, "exit_reason": "still open"})
    return pd.DataFrame(trips)


def trade_statistics(trips: pd.DataFrame) -> Dict[str, float]:
    closed = trips[~trips["open"]] if len(trips) else trips
    if closed is None or len(closed) == 0:
        return {"round_trips": 0, "win_rate": float("nan"), "profit_factor": float("nan"),
                "avg_win": float("nan"), "avg_loss": float("nan"),
                "avg_win_pct": float("nan"), "avg_loss_pct": float("nan"),
                "avg_holding_days": float("nan")}
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    gross_loss = -losses["pnl"].sum()
    return {
        "round_trips": int(len(closed)),
        "win_rate": float(len(wins) / len(closed)),
        "profit_factor": float(wins["pnl"].sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "avg_win_pct": float(wins["return"].mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(losses["return"].mean()) if len(losses) else 0.0,
        "avg_holding_days": float(closed["holding_days"].mean()),
    }


def performance_summary(equity: pd.Series, bench_equity: Optional[pd.Series] = None,
                        rf_daily: Optional[pd.Series] = None, confidence: float = 0.95,
                        fills: Optional[pd.DataFrame] = None,
                        trips: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """All headline metrics for one equity curve (optionally vs. a benchmark)."""
    r = returns_from_equity(equity)
    out: Dict[str, float] = {
        "start": equity.index[0].date().isoformat(),
        "end": equity.index[-1].date().isoformat(),
        "initial_capital": float(equity.iloc[0]),
        "final_value": float(equity.iloc[-1]),
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r, rf_daily),
        "sortino": sortino_ratio(r, rf_daily),
        "max_drawdown": max_drawdown(equity),
        "var_95": value_at_risk(r, confidence),
        "es_95": expected_shortfall(r, confidence),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else float("nan")
    if bench_equity is not None:
        br = returns_from_equity(bench_equity.reindex(equity.index).ffill())
        ba = beta_alpha(r, br, rf_daily)
        out.update({"beta": ba["beta"], "alpha": ba["alpha"], "alpha_t": ba["alpha_t"],
                    "r2": ba["r2"], "information_ratio": information_ratio(r, br),
                    "correlation": float(r.corr(br))})
    if fills is not None:
        out["num_trades"] = int(len(fills))
        out["total_costs"] = float(fills["cost"].sum()) if len(fills) else 0.0
    if trips is not None:
        out.update(trade_statistics(trips))
    return out


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month table of returns (plus a full-year column)."""
    m = equity.resample("ME").last().pct_change()
    first = equity.resample("ME").last().iloc[0] / equity.iloc[0] - 1
    m.iloc[0] = first
    table = m.to_frame("r")
    table["year"], table["month"] = table.index.year, table.index.month
    pivot = table.pivot(index="year", columns="month", values="r")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                     "Nov", "Dec"][: len(pivot.columns)] if len(pivot.columns) == 12 else \
        [pd.Timestamp(2000, c, 1).strftime("%b") for c in pivot.columns]
    yearly = equity.resample("YE").last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / equity.iloc[0] - 1
    pivot["Year"] = yr.values
    return pivot
