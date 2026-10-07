"""Event-driven daily backtesting engine.

Timeline for each trading day t (this ordering is what prevents look-ahead bias):

    09:30  OPEN of day t   -> execute orders created after the close of day t-1
                              (fill = open price +/- spread & slippage)
    16:00  CLOSE of day t  -> mark portfolio to market, accrue cash interest
                           -> update drawdown monitor, check trailing stops
                           -> if t is a rebalance date: score stocks with data
                              up to and including t, generate signals and
                              target weights
                           -> new orders are queued for the OPEN of day t+1

A decision made with the close of day t can therefore never trade at the
close of day t (impossible in reality) or use any later price.

The same ``TradingSystem`` decision logic is reused by the paper-trading
engine, so paper trading runs the exact rules that were backtested.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from portfolio.broker import SimulatedBroker
from portfolio.construction import ConstructionResult, construct_portfolio
from portfolio.costs import CostModel
from risk.limits import DrawdownMonitor, gross_exposure_cap, stop_level, stop_triggered
from strategy.factors import FactorModel, FeatureSet
from strategy.signals import BUY, HOLD, SELL, generate_signals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- orders
@dataclass
class Order:
    ticker: str
    kind: str              # "target" (rebalance to weight) or "exit" (sell everything)
    target_weight: float
    reason: str
    created: pd.Timestamp


def rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Last trading day of each period (the decision is made at that close)."""
    s = pd.Series(dates, index=dates)
    if frequency == "monthly":
        grp = [dates.year, dates.month]
    elif frequency == "quarterly":
        grp = [dates.year, dates.quarter]
    elif frequency in ("weekly", "biweekly"):
        iso = dates.isocalendar()
        grp = [iso.year.values, iso.week.values]
    else:
        raise ValueError(frequency)
    last = s.groupby(grp).max().sort_values()
    out = pd.DatetimeIndex(last.values)
    if frequency == "biweekly":
        out = out[::2]
    # the final period may be incomplete (e.g. today is mid-month): drop it
    if len(out) and out[-1] == dates[-1]:
        period_end_ok = {
            "monthly": (dates[-1] + pd.offsets.BDay(1)).month != dates[-1].month,
        }.get(frequency, True)
        if not period_end_ok:
            out = out[:-1]
    return out


# ---------------------------------------------------------------- decision logic
class TradingSystem:
    """Stateless-per-call decision logic shared by backtesting and paper trading."""

    def __init__(self, cfg, features: FeatureSet, sector_map: Dict[str, str]):
        self.cfg = cfg
        self.features = features
        self.model = FactorModel(features, cfg.strategy)
        self.sectors = pd.Series(sector_map)

    def rebalance(self, date: pd.Timestamp, broker: SimulatedBroker, close: pd.Series,
                  drawdown_active: bool, stopped_out: Set[str], exiting: Set[str] = frozenset()):
        """Score -> signals -> target weights -> orders, all as of the close of ``date``.

        ``exiting`` = names with a stop-loss exit already queued for the next open.
        They are treated as already sold: they don't occupy a slot or receive a
        target weight, so their capital is reallocated to the next-best names.
        """
        pcfg = self.cfg.portfolio
        scores = self.model.score(date)
        holdings = [t for t in broker.positions if t not in exiting]
        signals = generate_signals(scores, holdings, self.cfg.strategy,
                                   pcfg.max_positions, set(stopped_out) | set(exiting))
        selected = signals.loc[signals["signal"].isin([BUY, HOLD]), "ticker"].tolist()

        risk_on = self.model.market_risk_on(date)
        cap = gross_exposure_cap(self.cfg.risk, risk_on, drawdown_active)
        vols = self.features.volatility.loc[date]
        window = self.features.adj_close.loc[:date].iloc[-127:].pct_change(fill_method=None).iloc[1:]
        construction = construct_portfolio(selected, vols, self.sectors, window, pcfg, cap)

        orders: List[Order] = []
        reasons = dict(zip(signals["ticker"], signals["reason"]))
        for t in broker.positions:
            if t in exiting:
                continue  # exit order already queued by the stop-loss
            if t not in construction.weights.index:
                orders.append(Order(t, "exit", 0.0, "SELL: " + reasons.get(t, "not selected"), date))
        for t, w in construction.weights.items():
            sig = "HOLD/resize" if t in broker.positions else "BUY"
            orders.append(Order(t, "target", float(w), f"{sig}: {reasons.get(t, '')}", date))

        signals = signals.assign(date=date, target_weight=signals["ticker"].map(construction.weights).fillna(0.0))
        context = {"date": date, "risk_on": risk_on, "gross_cap": cap,
                   "drawdown_mode": drawdown_active, "est_vol": construction.est_volatility,
                   "pre_target_vol": construction.pre_target_volatility,
                   "vol_scale": construction.vol_scale, "n_selected": len(selected),
                   "notes": "; ".join(construction.notes)}
        return orders, signals, scores, context

    def stop_orders(self, date: pd.Timestamp, broker: SimulatedBroker, close: pd.Series) -> List[Order]:
        rcfg = self.cfg.risk
        if not rcfg.use_stop_loss:
            return []
        atr = self.features.atr.loc[date]
        vol = self.features.volatility.loc[date]
        out = []
        for t, p in broker.positions.items():
            px = close.get(t, np.nan)
            level = stop_level(p.peak_price, rcfg.stop_method, vol.get(t, np.nan), atr.get(t, np.nan),
                               rcfg.stop_loss_vol_multiple, rcfg.stop_loss_atr_multiple)
            if stop_triggered(px, level):
                out.append(Order(t, "exit", 0.0,
                                 f"STOP: close {px:.2f} < stop {level:.2f} "
                                 f"(peak {p.peak_price:.2f}, {rcfg.stop_method} stop)", date))
        return out


def execute_orders(date: pd.Timestamp, broker: SimulatedBroker, orders: List[Order],
                   open_px: pd.Series, fallback_px: pd.Series, rebalance_threshold: float) -> None:
    """Fill queued orders at today's open: exits and reductions first (to raise cash), then buys."""
    if not orders:
        return
    px = open_px.where(open_px.notna() & (open_px > 0), fallback_px)
    equity = broker.equity(px)
    exits = [o for o in orders if o.kind == "exit"]
    targets = {o.ticker: o for o in orders if o.kind == "target"}
    exit_names = {o.ticker for o in exits}

    for o in exits:
        if broker.shares(o.ticker) > 0:
            broker.execute(date, o.ticker, -broker.shares(o.ticker), px.get(o.ticker, np.nan), o.reason)

    buys = []
    for t, o in targets.items():
        if t in exit_names:
            continue  # a stop-loss exit overrides a same-day rebalance target
        price = px.get(t, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        current = broker.shares(t)
        target_shares = o.target_weight * equity / price
        delta = target_shares - current
        is_resize = current > 0 and o.target_weight > 0
        if is_resize and abs(delta * price) < rebalance_threshold * equity:
            continue  # tiny adjustment: not worth the trading cost
        if delta < 0:
            broker.execute(date, t, delta, price, o.reason)
        elif delta > 0:
            buys.append((t, delta, price, o.reason))
    for t, delta, price, reason in buys:
        broker.execute(date, t, delta, price, reason)


# ---------------------------------------------------------------- results
@dataclass
class BacktestResult:
    equity: pd.DataFrame          # daily: equity, cash, invested, gross, n_positions, drawdown, ...
    fills: pd.DataFrame           # every simulated execution
    signals: pd.DataFrame         # every rebalance decision with reason
    scores: pd.DataFrame          # full factor table at every rebalance (for attribution)
    positions: pd.DataFrame       # daily market value per ticker
    rebalances: pd.DataFrame      # risk-overlay context at each rebalance
    benchmark: pd.Series          # SPY buy-and-hold equity on the same dates
    config_overrides: Dict = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return self.equity["equity"]


# ---------------------------------------------------------------- engine
class Backtester:
    def __init__(self, cfg, md, features: FeatureSet, sector_map: Dict[str, str],
                 cost_model: Optional[CostModel] = None):
        self.cfg = cfg
        self.md = md
        self.system = TradingSystem(cfg, features, sector_map)
        self.cost_model = cost_model or CostModel.from_config(cfg.costs)

    def run(self, start: Optional[str] = None, end: Optional[str] = None,
            record_scores: bool = True) -> BacktestResult:
        cfg, md = self.cfg, self.md
        start = pd.Timestamp(start or cfg.periods.backtest_start)
        dates = md.dates[(md.dates >= start) & (md.dates <= pd.Timestamp(end or md.dates[-1]))]
        if len(dates) < 2:
            raise ValueError("Backtest window too short")
        rebal = set(rebalance_dates(md.dates, cfg.strategy.rebalance_frequency))
        rebal.add(dates[0])  # form the initial portfolio at the first close

        broker = SimulatedBroker(cfg.portfolio.initial_capital, self.cost_model)
        ddm = DrawdownMonitor(cfg.risk.drawdown_limit, cfg.risk.drawdown_reset)
        pending: List[Order] = []
        stopped: Set[str] = set()
        rows, sig_frames, score_frames, ctx_rows, pos_rows = [], [], [], [], {}
        prev_close = None

        for i, date in enumerate(dates):
            close = md.adj_close.loc[date]
            # 1) market open: fill yesterday's orders
            if pending:
                execute_orders(date, broker, pending, md.adj_open.loc[date],
                               prev_close if prev_close is not None else close,
                               cfg.portfolio.rebalance_threshold)
                pending = []
            # 2) close: interest, mark-to-market, risk monitoring
            if cfg.portfolio.cash_earns_risk_free and i > 0:
                broker.accrue_interest(float(md.risk_free_daily.loc[dates[i - 1]]))
            broker.update_peaks(close)
            equity = broker.equity(close)
            dd = ddm.update(equity)
            mv = {t: p.shares * close.get(t, np.nan) for t, p in broker.positions.items()}
            pos_rows[date] = mv
            invested = float(np.nansum(list(mv.values())))
            rows.append({"date": date, "equity": equity, "cash": broker.cash, "invested": invested,
                         "gross_exposure": invested / equity if equity > 0 else 0.0,
                         "n_positions": len(broker.positions), "drawdown": dd,
                         "drawdown_mode": ddm.active})
            if i == len(dates) - 1:
                break  # no orders after the last bar (they could never be filled)

            stop = self.system.stop_orders(date, broker, close)
            for o in stop:
                stopped.add(o.ticker)
            pending.extend(stop)

            if date in rebal:
                stop_names = {o.ticker for o in stop}
                orders, signals, scores, ctx = self.system.rebalance(date, broker, close, ddm.active,
                                                                     stopped, exiting=stop_names)
                pending.extend(orders)
                sig_frames.append(signals)
                if record_scores:
                    score_frames.append(scores.assign(date=date))
                ctx_rows.append(ctx)
                stopped = set()
            prev_close = close

        equity_df = pd.DataFrame(rows).set_index("date")
        positions = pd.DataFrame.from_dict(pos_rows, orient="index").reindex(equity_df.index).fillna(0.0)
        bench = self.benchmark_equity(dates)
        return BacktestResult(
            equity=equity_df,
            fills=broker.fills_frame(),
            signals=pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame(),
            scores=(pd.concat(score_frames).reset_index() if score_frames else pd.DataFrame()),
            positions=positions,
            rebalances=pd.DataFrame(ctx_rows),
            benchmark=bench,
        )

    def benchmark_equity(self, dates: pd.DatetimeIndex) -> pd.Series:
        """SPY buy-and-hold: invest all capital at the first close, pay one-way costs."""
        b = self.md.benchmark.reindex(dates).ffill()
        cap = self.cfg.portfolio.initial_capital * (1 - self.cost_model.price_impact)
        return (cap * b / b.iloc[0]).rename("SPY")
