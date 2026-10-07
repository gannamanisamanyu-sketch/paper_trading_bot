"""Today's model portfolio: what the strategy would hold if started fresh today.

Written to ``results/live/`` by ``python main.py signals`` (run daily by the
GitHub Action) and shown on the website's "Today's Signals" page. This is a
stateless snapshot. The persistent paper portfolio lives in ``data/paper/``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from backtesting.engine import TradingSystem
from backtesting.pipeline import DataContext
from paper_trading.engine import NY, _market_open_now
from portfolio.broker import SimulatedBroker
from portfolio.costs import CostModel

logger = logging.getLogger(__name__)


def last_complete_date(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """Latest date with a COMPLETED daily bar (drops today's bar while the market is open)."""
    if _market_open_now() and dates[-1].date() == datetime.now(NY).date():
        return dates[-2]
    return dates[-1]


def compute_live_signals(cfg, ctx: DataContext, out_dir: Path) -> Dict:
    md = ctx.md
    date = last_complete_date(md.dates)
    capital = float(cfg.paper_trading.starting_capital)
    system = TradingSystem(cfg, ctx.features(cfg.strategy), ctx.universe.sector_map)
    broker = SimulatedBroker(capital, CostModel.from_config(cfg.costs))
    close = md.close.loc[date]
    orders, signals, scores, info = system.rebalance(date, broker, close, False, set())

    names, sectors = ctx.universe.name_map, ctx.universe.sector_map
    w = pd.Series({o.ticker: o.target_weight for o in orders if o.kind == "target"}, dtype=float)
    targets = pd.DataFrame({"ticker": w.index, "weight": w.values})
    if len(targets):
        targets["name"] = targets["ticker"].map(names)
        targets["sector"] = targets["ticker"].map(sectors)
        targets["price"] = targets["ticker"].map(close)
        targets["dollars"] = targets["weight"] * capital
        targets["shares"] = np.floor(targets["dollars"] / targets["price"])
        targets = targets.merge(signals[["ticker", "score", "rank", "reason"]], on="ticker", how="left")
        targets = targets.sort_values("weight", ascending=False)

    keep = ["sector", "score", "rank", "momentum_12_1", "price_vs_sma200", "rsi", "volatility", "beta",
            "earnings_yield", "roe", "revenue_growth", "z_momentum", "z_trend", "z_sector", "z_quality",
            "z_value", "z_growth", "z_risk", "above_sma200"]
    sc = scores[[c for c in keep if c in scores]].copy()
    sc.insert(0, "name", sc.index.map(names))
    sc["in_model_portfolio"] = sc.index.isin(w.index)

    meta = {"as_of": str(date.date()), "generated": datetime.now().isoformat(timespec="seconds"),
            "capital": capital, "market_risk_on": bool(info["risk_on"]),
            "gross_cap": float(info["gross_cap"]), "est_volatility": float(info["est_vol"]),
            "pre_target_volatility": float(info["pre_target_vol"]), "invested": float(w.sum()),
            "n_positions": int(len(w)), "eligible_stocks": int(len(scores)), "notes": info["notes"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(out_dir / "model_portfolio.csv", index=False)
    sc.to_csv(out_dir / "scores.csv")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Live signals for %s: %d positions, %.0f%% invested -> %s",
                date.date(), len(w), 100 * w.sum(), out_dir)
    return {"meta": meta, "targets": targets, "scores": sc}
