"""Paper-trading engine: run the backtested rules forward in time, locally.

    *** PAPER TRADING ONLY ***
    There is no brokerage connection anywhere in this project. "Orders" are rows
    in a CSV file, and "fills" are simulated with the same cost model as the
    backtest.

Daily cycle (``python main.py paper run``, ideally after the 4pm ET close):

1. Download the latest data. If the market is still open, today's incomplete
   bar is dropped so signals only use completed closes.
2. For every trading day since the last run, in order:
   a. credit cash dividends (splits since the last run are applied once,
      up front, because Yahoo restates all past prices for splits)
   b. fill pending orders at that day's OPEN (+ spread & slippage)
   c. mark to market at the CLOSE, accrue T-bill interest on cash
   d. update the drawdown monitor and check trailing stops
   e. if it is a rebalance date, generate signals and new target orders
3. Save state, logs and recommended trades, then fire alerts.

Fill realism: orders generated from day t's close are filled at day t+1's
open, the same convention as the backtest. If you run the bot on day t+1
after the close, the order is filled at that day's real opening price.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.engine import Order, TradingSystem, execute_orders
from backtesting.pipeline import DataContext, load_context
from paper_trading.alerts import (BUY_SIGNAL, DRAWDOWN, REBALANCE, RISK_LIMIT, SELL_SIGNAL,
                                  STOP_LOSS, AlertManager)
from paper_trading.store import PaperStore
from portfolio.broker import SimulatedBroker
from portfolio.costs import CostModel
from risk.limits import DrawdownMonitor, check_limits

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    NY = None


def _market_open_now() -> bool:
    if NY is None:
        return False
    now = datetime.now(NY)
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 15)


def is_month_end(date: pd.Timestamp, next_date: Optional[pd.Timestamp]) -> bool:
    """True if ``date`` is the last trading day of its month.

    For historical days the next trading day is known from the data. For the
    latest day we use the next weekday (holidays are handled by the catch-up
    rule in ``PaperTrader``).
    """
    nxt = next_date if next_date is not None else date + pd.offsets.BDay(1)
    return nxt.month != date.month


class PaperTrader:
    def __init__(self, cfg, ctx: Optional[DataContext] = None):
        self.cfg = cfg
        self.store = PaperStore(cfg.path(cfg.paper_trading.state_dir))
        self.ctx = ctx
        self.alerts = AlertManager(cfg.alerts, self.store.alerts_file)
        self.cost_model = CostModel.from_config(cfg.costs)

    # ------------------------------------------------------------------ setup
    def init(self, force: bool = False) -> None:
        if self.store.exists() and not force:
            raise SystemExit("A paper portfolio already exists. Use `paper init --force` to reset it.")
        if force and self.store.root.exists():
            for f in self.store.root.glob("*"):
                f.unlink()
        capital = float(self.cfg.paper_trading.starting_capital)
        self.store.save_state({
            "created": datetime.now().isoformat(timespec="seconds"),
            "starting_capital": capital, "cash": capital, "positions": {},
            "last_processed_date": None, "last_rebalance_period": None,
            "peak_equity": capital, "drawdown_mode": False, "stopped_since_rebalance": [],
            "mode": "PAPER_TRADING_ONLY",
        })
        self.store.save_orders([])
        print(f"Paper portfolio created with ${capital:,.0f} in {self.store.root}")
        print("Next: run `python main.py paper run` after the market close.")

    # ------------------------------------------------------------------ helpers
    def _context(self, refresh: bool) -> DataContext:
        if self.ctx is None:
            self.ctx = load_context(self.cfg, refresh=refresh)
        return self.ctx

    def _broker_from_state(self, state: Dict) -> SimulatedBroker:
        b = SimulatedBroker(state["cash"], self.cost_model)
        b.positions = self.store.positions_from_json(state["positions"])
        return b

    def _trade_rows(self, fills, broker: SimulatedBroker, prices: pd.Series) -> pd.DataFrame:
        if not fills:
            return pd.DataFrame()
        equity = broker.equity(prices)
        rows = []
        for f in fills:
            is_sell = f.side == "SELL"
            entry = (f.fill_price - (f.realized_pnl + f.commission) / f.shares) if is_sell else f.fill_price
            rows.append({"date": f.date.date(), "ticker": f.ticker, "signal": f.side, "shares": f.shares,
                         "entry_price": round(entry, 4), "exit_price": round(f.fill_price, 4) if is_sell else np.nan,
                         "fill_price": round(f.fill_price, 4), "position_value": round(f.gross_value, 2),
                         "pnl": round(f.realized_pnl, 2) if is_sell else 0.0,
                         "commission": round(f.commission, 2), "trading_cost": round(f.cost, 2),
                         "portfolio_value": round(equity, 2), "cash": round(f.cash_after, 2),
                         "reason": f.reason})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ main cycle
    def run(self, refresh: bool = True) -> Dict:
        if not self.store.exists():
            raise SystemExit("No paper portfolio found. Run `python main.py paper init` first.")
        cfg = self.cfg
        ctx = self._context(refresh)
        md = ctx.md
        dates = md.dates
        if _market_open_now() and dates[-1].date() == datetime.now(NY).date():
            logger.info("Market is open: ignoring today's incomplete bar")
            dates = dates[:-1]

        state = self.store.load_state()
        broker = self._broker_from_state(state)
        system = TradingSystem(cfg, ctx.features(cfg.strategy), ctx.universe.sector_map)
        ddm = DrawdownMonitor(cfg.risk.drawdown_limit, cfg.risk.drawdown_reset,
                              peak=state["peak_equity"], active=state["drawdown_mode"])
        pending: List[Order] = self.store.load_orders()
        stopped = set(state.get("stopped_since_rebalance", []))
        last = pd.Timestamp(state["last_processed_date"]) if state["last_processed_date"] else None
        new_days = dates[dates > last] if last is not None else dates[-1:]
        if last is not None:
            self._apply_splits_since(broker, md, last)
        if len(new_days) == 0:
            print(f"Already up to date (last processed {last.date()}). Nothing to do.")
            self.print_status()
            return {}

        prev_equity = None
        eq_hist = self.store.read("equity")
        if len(eq_hist):
            prev_equity = float(eq_hist["equity"].iloc[-1])
        start_cap = state["starting_capital"]
        equity_rows, all_trades, signal_frames = [], [], []
        new_orders: List[Order] = []

        for k, d in enumerate(new_days):
            close = md.close.loc[d]
            pos_in_dates = dates.get_loc(d)
            next_d = dates[pos_in_dates + 1] if pos_in_dates + 1 < len(dates) else None

            # a) dividends (splits were already applied before the loop)
            div_cash = 0.0
            for t, p in list(broker.positions.items()):
                if md.dividends is not None and t in md.dividends:
                    dps = float(md.dividends.at[d, t])
                    if dps > 0:
                        div_cash += dps * p.shares
            broker.cash += div_cash

            # b) fill pending orders at the open
            n_before = len(broker.fills)
            if pending:
                execute_orders(d, broker, pending, md.open.loc[d],
                               md.close.loc[dates[pos_in_dates - 1]] if pos_in_dates > 0 else close,
                               cfg.portfolio.rebalance_threshold)
                pending = []
            fills_today = broker.fills[n_before:]

            # c) mark to market + interest
            interest = 0.0
            # no interest on the day the portfolio is created (cash wasn't held overnight yet)
            if cfg.portfolio.cash_earns_risk_free and pos_in_dates > 0 and (last is not None or k > 0):
                before = broker.cash
                broker.accrue_interest(float(md.risk_free_daily.loc[dates[pos_in_dates - 1]]))
                interest = broker.cash - before
            broker.update_peaks(close)
            equity = broker.equity(close)
            all_trades.append(self._trade_rows(fills_today, broker, close))
            dd = ddm.update(equity)
            daily_pnl = equity - (prev_equity if prev_equity is not None else start_cap)
            equity_rows.append({
                "date": d.date(), "equity": round(equity, 2), "cash": round(broker.cash, 2),
                "invested": round(equity - broker.cash, 2), "n_positions": len(broker.positions),
                "daily_pnl": round(daily_pnl, 2),
                "daily_return": daily_pnl / prev_equity if prev_equity else daily_pnl / start_cap,
                "total_return": equity / start_cap - 1, "drawdown": dd,
                "dividends": round(div_cash, 2), "interest": round(interest, 2)})
            prev_equity = equity

            # d) risk checks -> stop orders
            stops = system.stop_orders(d, broker, close)
            for o in stops:
                stopped.add(o.ticker)
                self.alerts.add(d, STOP_LOSS, f"{o.ticker}: {o.reason}")
            day_orders = list(stops)

            # e) rebalance?
            period = f"{d.year}-{d.month:02d}"
            first_ever = state["last_rebalance_period"] is None and k == len(new_days) - 1
            month_end = is_month_end(d, next_d)
            if (month_end and state["last_rebalance_period"] != period) or first_ever:
                day_orders += self._rebalance(system, d, broker, close, ddm, stopped, signal_frames,
                                              exiting={o.ticker for o in stops})
                state["last_rebalance_period"] = period
                stopped = set()

            if next_d is not None and next_d in new_days:
                pending = day_orders     # will be filled at the next processed open
            else:
                new_orders = day_orders  # waits for tomorrow's open

        # catch-up: previous month-end missed (holiday edge case or bot not run)
        latest = new_days[-1]
        prev_month = (latest.to_period("M") - 1).strftime("%Y-%m")
        lrp = state["last_rebalance_period"]
        if lrp is not None and lrp < prev_month:
            logger.info("Missed month-end rebalance for %s: rebalancing on %s", prev_month, latest.date())
            new_orders += self._rebalance(system, latest, broker, md.close.loc[latest], ddm, stopped,
                                          signal_frames,
                                          exiting={o.ticker for o in new_orders if o.kind == "exit"})
            state["last_rebalance_period"] = prev_month
            stopped = set()

        # ---- alerts on the final state
        close = md.close.loc[latest]
        equity = broker.equity(close)
        if ddm.peak > 0 and (equity / ddm.peak - 1) <= -cfg.alerts.drawdown_alert:
            self.alerts.add(latest, DRAWDOWN, f"Portfolio drawdown {equity / ddm.peak - 1:.1%} exceeds "
                                              f"alert threshold {cfg.alerts.drawdown_alert:.0%}")
        vols = ctx.features(cfg.strategy).volatility.loc[latest]
        for breach in check_limits(broker.weights(close), ctx.universe.sector_map, vols, cfg.portfolio):
            self.alerts.add(latest, RISK_LIMIT, breach)

        # ---- persist
        state.update({"cash": broker.cash, "positions": self.store.positions_to_json(broker.positions),
                      "last_processed_date": str(latest.date()), "peak_equity": ddm.peak,
                      "drawdown_mode": ddm.active, "stopped_since_rebalance": sorted(stopped),
                      "last_run": datetime.now().isoformat(timespec="seconds")})
        self.store.save_state(state)
        self.store.save_orders(new_orders)
        trades = pd.concat([t for t in all_trades if len(t)], ignore_index=True) if any(len(t) for t in all_trades) else pd.DataFrame()
        self.store.append_trades(trades)
        self.store.append_equity(pd.DataFrame(equity_rows))
        if signal_frames:
            self.store.append_signals(pd.concat(signal_frames, ignore_index=True))
        sent = self.alerts.dispatch()
        self._print_run_summary(new_days, trades, new_orders, broker, close, sent)
        return {"days": len(new_days), "trades": trades, "orders": new_orders, "alerts": sent}

    @staticmethod
    def _apply_splits_since(broker: SimulatedBroker, md, last: pd.Timestamp) -> None:
        """Restate stored positions onto the split basis of the freshly downloaded data.

        Yahoo back-adjusts ALL historical prices (and dividends) for splits, so every
        price in ``md`` is on today's share basis. Positions saved before a split
        are converted once here: shares x ratio, per-share cost and stop peak / ratio.
        Doing it before the day loop (not on the split date) keeps catch-up runs that
        span a split consistent. Value and P&L are unchanged by construction.
        """
        if md.splits is None:
            return
        after = md.splits.loc[md.splits.index > last]
        for t, p in broker.positions.items():
            if t not in after:
                continue
            ratio = float(after[t].prod())
            if abs(ratio - 1.0) > 1e-12:
                p.shares *= ratio
                p.avg_cost /= ratio
                p.peak_price /= ratio
                logger.info("Applied %.4g:1 split(s) since %s to %s", ratio, last.date(), t)

    def _rebalance(self, system, d, broker, close, ddm, stopped, signal_frames,
                   exiting=frozenset()) -> List[Order]:
        orders, signals, scores, ctx = system.rebalance(d, broker, close, ddm.active, stopped, exiting)
        signals = signals.assign(date=d.date(), risk_on=ctx["risk_on"], gross_cap=ctx["gross_cap"])
        signal_frames.append(signals)
        buys = signals[signals["signal"] == "BUY"]
        sells = signals[signals["signal"] == "SELL"]
        self.alerts.add(d, REBALANCE, f"Monthly rebalance: {len(buys)} buy, {len(sells)} sell, "
                                      f"{(signals['signal'] == 'HOLD').sum()} hold; gross cap {ctx['gross_cap']:.0%}"
                                      + (f"; {ctx['notes']}" if ctx["notes"] else ""))
        for r in buys.itertuples():
            self.alerts.add(d, BUY_SIGNAL, f"BUY {r.ticker} (target {r.target_weight:.1%}): {r.reason}")
        for r in sells.itertuples():
            self.alerts.add(d, SELL_SIGNAL, f"SELL {r.ticker}: {r.reason}")
        return orders

    # ------------------------------------------------------------------ display
    def _print_run_summary(self, days, trades, orders, broker, close, alerts) -> None:
        eq = broker.equity(close)
        print("\n" + "=" * 72)
        print(f" PAPER TRADING RUN: processed {len(days)} day(s), through {days[-1].date()}")
        print("  (simulation only, no real orders were placed)")
        print("=" * 72)
        if len(trades):
            print("\nSimulated fills:")
            print(trades[["date", "ticker", "signal", "shares", "fill_price", "pnl", "reason"]].to_string(index=False))
        if orders:
            print("\nRecommended trades for the NEXT market open (will be simulated on the next run):")
            rows = []
            for o in orders:
                px = close.get(o.ticker, np.nan)
                cur = broker.shares(o.ticker)
                tgt = 0.0 if o.kind == "exit" else o.target_weight * eq / px
                rows.append({"ticker": o.ticker, "action": "SELL" if tgt < cur else "BUY",
                             "current_shares": cur, "target_shares": int(tgt),
                             "est_trade_$": round((int(tgt) - cur) * px, 0),
                             "target_weight": f"{o.target_weight:.1%}", "reason": o.reason[:60]})
            df = pd.DataFrame(rows)
            df = df[df["est_trade_$"].abs() >= self.cfg.portfolio.rebalance_threshold * eq]
            print(df.to_string(index=False) if len(df) else "  (all adjustments below the rebalance threshold)")
        else:
            print("\nNo new orders.")
        self.print_status(broker=broker, close=close)

    def print_status(self, broker: Optional[SimulatedBroker] = None, close: Optional[pd.Series] = None) -> None:
        state = self.store.load_state()
        if broker is None:
            broker = self._broker_from_state(state)
        eqdf = self.store.read("equity")
        if close is None:
            # status without downloading: value positions at last recorded prices
            if len(eqdf):
                equity = float(eqdf["equity"].iloc[-1])
            else:
                equity = broker.cash
            print(f"\nPortfolio value (last run {state.get('last_processed_date')}): ${equity:,.2f}")
            print(f"Cash: ${broker.cash:,.2f}  |  Positions: {len(broker.positions)}")
            for t, p in sorted(broker.positions.items()):
                print(f"  {t:6s} {p.shares:>8.0f} sh  avg cost ${p.avg_cost:,.2f}  since {p.entry_date.date()}")
            return
        equity = broker.equity(close)
        start = state["starting_capital"]
        print(f"\nPortfolio value: ${equity:,.2f}  (total return {equity / start - 1:+.2%})")
        print(f"Cash: ${broker.cash:,.2f} ({broker.cash / equity:.1%})")
        if broker.positions:
            rows = []
            for t, p in sorted(broker.positions.items()):
                px = close.get(t, np.nan)
                rows.append({"ticker": t, "shares": p.shares, "avg_cost": round(p.avg_cost, 2),
                             "price": round(px, 2), "value": round(p.shares * px, 0),
                             "weight": f"{p.shares * px / equity:.1%}",
                             "unrealized_pnl": round((px - p.avg_cost) * p.shares, 0),
                             "since": p.entry_date.date()})
            print(pd.DataFrame(rows).to_string(index=False))
