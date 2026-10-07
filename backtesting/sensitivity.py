"""One-at-a-time sensitivity analysis.

Purpose: robustness, NOT optimization. Each parameter is moved away from its
base value while everything else is held fixed. A robust strategy shows
results that change gradually. If performance only looks good at one exact
setting, the "edge" is probably an artifact of that setting (overfitting).

The table reports full-period metrics plus the out-of-sample (test) Sharpe
ratio for every variant. None of these results is used to change the base
configuration (see RESEARCH_LOG.md).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from backtesting.pipeline import DataContext, period_bounds, run_backtest, summarize
from portfolio.costs import CostModel

logger = logging.getLogger(__name__)


def sensitivity_grid(cfg) -> List[Tuple[str, str, Dict, float]]:
    """(parameter, label, config overrides, total cost bps or None)."""
    base_bps = cfg.costs.half_spread_bps + cfg.costs.slippage_bps
    g: List[Tuple[str, str, Dict, float]] = []
    for f in ["weekly", "biweekly", "monthly", "quarterly"]:
        g.append(("Rebalancing frequency", f, {"strategy.rebalance_frequency": f}, None))
    for bps in [0, base_bps, 15, 25, 50]:
        g.append(("Transaction costs (one-way bps)", f"{bps:g}", {}, bps))
    for w in [0.05, 0.10, 0.15]:
        g.append(("Max position weight", f"{w:.0%}", {"portfolio.max_position_weight": w}, None))
    for n in [10, 15, 20, 30]:
        g.append(("Number of holdings", str(n), {"portfolio.max_positions": n}, None))
    for lb in [126, 189, 252]:
        g.append(("Momentum lookback (days)", str(lb), {"strategy.momentum_lookback": lb}, None))
    for tv in [0.12, 0.16, 0.20, None]:
        g.append(("Volatility target", "none" if tv is None else f"{tv:.0%}",
                  {"portfolio.target_volatility": tv}, None))
    g.append(("Stop-loss", "none", {"risk.use_stop_loss": False}, None))
    for m in [1.5, 2.0, 3.0]:
        g.append(("Stop-loss", f"{m:g}x monthly vol",
                  {"risk.use_stop_loss": True, "risk.stop_method": "volatility",
                   "risk.stop_loss_vol_multiple": m}, None))
    g.append(("Stop-loss", "3x ATR(14) (original)",
              {"risk.use_stop_loss": True, "risk.stop_method": "atr"}, None))
    for sm in ["equal", "volatility"]:
        g.append(("Position sizing", sm, {"portfolio.sizing_method": sm}, None))
    g.append(("Risk overlay", "regime filter off", {"risk.market_regime_filter": False}, None))
    g.append(("Risk overlay", "drawdown control off", {"risk.drawdown_limit": 1.0}, None))
    return g


def is_base(cfg, overrides: Dict, bps) -> bool:
    if bps is not None:
        return abs(bps - (cfg.costs.half_spread_bps + cfg.costs.slippage_bps)) < 1e-9
    raw = cfg.to_dict()
    for k, v in overrides.items():
        sec, key = k.split(".")
        if raw[sec][key] != v:
            return False
    return True


def run_sensitivity(cfg, ctx: DataContext) -> pd.DataFrame:
    bounds = period_bounds(cfg, ctx.md.dates[-1])
    base_cost = CostModel.from_config(cfg.costs)
    rows = []
    for param, label, overrides, bps in sensitivity_grid(cfg):
        vcfg = cfg.override(overrides) if overrides else cfg
        cost = base_cost.with_total_bps(bps) if bps is not None else None
        logger.info("Sensitivity: %s = %s", param, label)
        res = run_backtest(vcfg, ctx, cost_model=cost, record_scores=False)
        full = summarize(res, ctx, vcfg)["strategy"]
        test = summarize(res, ctx, vcfg, *bounds["test"])["strategy"]
        rows.append({
            "parameter": param, "value": label, "base": is_base(cfg, overrides, bps),
            "cagr": full["cagr"], "volatility": full["volatility"], "sharpe": full["sharpe"],
            "sortino": full["sortino"], "max_drawdown": full["max_drawdown"], "beta": full["beta"],
            "alpha": full["alpha"], "trades": full["num_trades"], "costs_paid": full["total_costs"],
            "avg_gross": full["avg_gross_exposure"], "test_sharpe": test["sharpe"],
            "test_cagr": test["cagr"],
        })
    return pd.DataFrame(rows)
