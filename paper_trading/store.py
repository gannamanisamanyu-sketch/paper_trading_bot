"""Persistent CSV/JSON storage for the paper portfolio.

data/paper/
    state.json           cash, positions, risk state, last processed date
    pending_orders.csv   orders waiting for the next market open
    trades.csv           every simulated fill (the trade ledger)
    daily_equity.csv     one row per trading day: value, cash, P&L
    signals.csv          every signal generated (BUY/SELL/HOLD with reasons)
    alerts.csv           alert history

Plain files keep the system transparent: you can open any of them in Excel.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtesting.engine import Order
from portfolio.broker import Position

TRADE_COLUMNS = ["date", "ticker", "signal", "shares", "entry_price", "exit_price", "fill_price",
                 "position_value", "pnl", "commission", "trading_cost", "portfolio_value", "cash", "reason"]
EQUITY_COLUMNS = ["date", "equity", "cash", "invested", "n_positions", "daily_pnl", "daily_return",
                  "total_return", "drawdown", "dividends", "interest"]


class PaperStore:
    def __init__(self, root: Path):
        self.root = root
        self.state_file = root / "state.json"
        self.orders_file = root / "pending_orders.csv"
        self.trades_file = root / "trades.csv"
        self.equity_file = root / "daily_equity.csv"
        self.signals_file = root / "signals.csv"
        self.alerts_file = root / "alerts.csv"

    def exists(self) -> bool:
        return self.state_file.exists()

    # ------------------------------------------------------------------ state
    def save_state(self, state: Dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(self.state_file)  # atomic: a crash can't leave a half-written state

    def load_state(self) -> Dict:
        return json.loads(self.state_file.read_text())

    @staticmethod
    def positions_to_json(positions: Dict[str, Position]) -> Dict:
        return {t: {**asdict(p), "entry_date": str(pd.Timestamp(p.entry_date).date())}
                for t, p in positions.items()}

    @staticmethod
    def positions_from_json(raw: Dict) -> Dict[str, Position]:
        return {t: Position(t, float(p["shares"]), float(p["avg_cost"]), pd.Timestamp(p["entry_date"]),
                            float(p["peak_price"])) for t, p in raw.items()}

    # ------------------------------------------------------------------ orders
    def save_orders(self, orders: List[Order]) -> None:
        df = pd.DataFrame([{"ticker": o.ticker, "kind": o.kind, "target_weight": o.target_weight,
                            "reason": o.reason, "created": str(pd.Timestamp(o.created).date())}
                           for o in orders], columns=["ticker", "kind", "target_weight", "reason", "created"])
        df.to_csv(self.orders_file, index=False)

    def load_orders(self) -> List[Order]:
        if not self.orders_file.exists():
            return []
        df = pd.read_csv(self.orders_file)
        return [Order(r.ticker, r.kind, float(r.target_weight), str(r.reason), pd.Timestamp(r.created))
                for r in df.itertuples()]

    # ------------------------------------------------------------------ logs
    def _append(self, path: Path, rows: pd.DataFrame) -> None:
        if rows is None or rows.empty:
            return
        header = not path.exists()
        rows.to_csv(path, mode="a", header=header, index=False)

    def append_trades(self, rows: pd.DataFrame) -> None:
        self._append(self.trades_file, rows[TRADE_COLUMNS] if len(rows) else rows)

    def append_equity(self, rows: pd.DataFrame) -> None:
        self._append(self.equity_file, rows[EQUITY_COLUMNS] if len(rows) else rows)

    def append_signals(self, rows: pd.DataFrame) -> None:
        self._append(self.signals_file, rows)

    def read(self, which: str) -> pd.DataFrame:
        path = {"trades": self.trades_file, "equity": self.equity_file, "signals": self.signals_file,
                "alerts": self.alerts_file, "orders": self.orders_file}[which]
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        for c in ("date", "created"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
        return df
