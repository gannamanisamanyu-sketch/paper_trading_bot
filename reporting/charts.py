"""Static report charts (matplotlib -> PNG). The dashboard draws its own interactive versions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

STRAT = "#2563eb"
BENCH = "#6b7280"
NEG = "#dc2626"
POS = "#16a34a"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _shade_periods(ax, bounds: Optional[Dict[str, tuple]]) -> None:
    if not bounds:
        return
    colors = {"train": "#f3f4f6", "validation": "#fef9c3", "test": "#dcfce7"}
    for name in ("train", "validation", "test"):
        if name in bounds:
            s, e = bounds[name]
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=colors[name], alpha=0.6, lw=0)
            ax.text(pd.Timestamp(s), 1.0, f" {name}", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=8, color="#374151")


def equity_vs_benchmark(equity: pd.Series, bench: pd.Series, path: Path,
                        bounds: Optional[Dict[str, tuple]] = None) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _shade_periods(ax, bounds)
    ax.plot(equity.index, equity.values, color=STRAT, lw=1.6, label="Strategy")
    ax.plot(bench.index, bench.values, color=BENCH, lw=1.3, label="SPY buy & hold")
    ax.set_title("Portfolio value: strategy vs. S&P 500 (SPY, total return)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.legend(loc="upper left")
    return _save(fig, path)


def drawdown_chart(equity: pd.Series, bench: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    dd = equity / equity.cummax() - 1
    bdd = bench / bench.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color=STRAT, alpha=0.35, label="Strategy")
    ax.plot(bdd.index, bdd.values, color=BENCH, lw=1, label="SPY")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Drawdown from previous peak")
    ax.legend(loc="lower left")
    return _save(fig, path)


def monthly_heatmap(table: pd.DataFrame, path: Path) -> Path:
    data = table.drop(columns=["Year"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(data) + 1.2))
    v = np.nanmax(np.abs(data.values)) if data.size else 0.1
    im = ax.imshow(data.values, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(data.columns)), data.columns)
    ax.set_yticks(range(len(data.index)), data.index)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = data.values[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:.1%}", ha="center", va="center", fontsize=7.5)
    if "Year" in table:
        for i, y in enumerate(table["Year"].values):
            ax.text(len(data.columns) - 0.3, i, f"  {y:+.1%}", va="center", fontsize=8,
                    fontweight="bold", color=POS if y >= 0 else NEG)
    ax.set_title("Monthly returns (right: calendar-year return)")
    return _save(fig, path)


def sector_allocation(positions: pd.DataFrame, equity: pd.Series, sectors: Dict[str, str],
                      path: Path) -> Path:
    w = positions.div(equity, axis=0)
    by_sec = w.T.groupby(lambda t: sectors.get(t, "Unknown")).sum().T
    by_sec = by_sec.resample("ME").last()
    by_sec = by_sec.loc[:, by_sec.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.stackplot(by_sec.index, by_sec.T.values, labels=by_sec.columns,
                 colors=plt.cm.tab20.colors[: len(by_sec.columns)], alpha=0.9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_title("Sector allocation over time (white space = cash)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    return _save(fig, path)


def position_allocation(weights: pd.Series, path: Path, title: str) -> Path:
    w = weights.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(w) + 1.2))
    ax.barh(w.index, w.values, color=STRAT)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(title)
    return _save(fig, path)


def risk_return_scatter(points: pd.DataFrame, path: Path) -> Path:
    """points: index = label, columns = volatility, cagr."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, r in points.iterrows():
        color = STRAT if "Strategy" in label else (BENCH if "SPY" in label else "#a78bfa")
        ax.scatter(r["volatility"], r["cagr"], s=60 if color != "#a78bfa" else 25, color=color, zorder=3)
        if color != "#a78bfa":
            ax.annotate(label, (r["volatility"], r["cagr"]), xytext=(6, 4), textcoords="offset points")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("CAGR")
    ax.set_title("Risk vs. return (purple = sensitivity variants)")
    return _save(fig, path)


def factor_ic_chart(ic_table: pd.DataFrame, path: Path) -> Path:
    piv = ic_table.pivot(index="factor", columns="period", values="mean_ic")
    piv = piv[[c for c in ["train", "validation", "test", "full"] if c in piv]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    piv.plot.bar(ax=ax, color=["#9ca3af", "#facc15", "#22c55e", STRAT][: len(piv.columns)], width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Mean monthly rank IC by factor (higher = better stock ranking)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, path)


def rolling_metrics(equity: pd.Series, bench: pd.Series, path: Path, window: int = 252) -> Path:
    r = equity.pct_change()
    br = bench.pct_change()
    roll_beta = r.rolling(window).cov(br) / br.rolling(window).var()
    roll_excess = (1 + r).rolling(window).apply(np.prod, raw=True) - (1 + br).rolling(window).apply(np.prod, raw=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    a1.plot(roll_beta, color=STRAT)
    a1.axhline(1, color=BENCH, lw=0.8, ls="--")
    a1.set_title("Rolling 1-year beta to SPY")
    a2.fill_between(roll_excess.index, roll_excess.values, 0,
                    where=roll_excess.values >= 0, color=POS, alpha=0.5)
    a2.fill_between(roll_excess.index, roll_excess.values, 0,
                    where=roll_excess.values < 0, color=NEG, alpha=0.5)
    a2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    a2.set_title("Rolling 1-year return minus SPY")
    return _save(fig, path)


def cost_sensitivity_chart(sens: pd.DataFrame, path: Path) -> Path:
    c = sens[sens["parameter"].str.startswith("Transaction costs")].copy()
    c["bps"] = c["value"].astype(float)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(c["bps"], c["cagr"], marker="o", color=STRAT, label="Strategy CAGR")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("One-way transaction cost (bps)")
    ax.set_title("Performance vs. transaction-cost assumption")
    ax2 = ax.twinx()
    ax2.plot(c["bps"], c["sharpe"], marker="s", color="#f59e0b", label="Sharpe")
    ax2.set_ylabel("Sharpe")
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left")
    return _save(fig, path)
