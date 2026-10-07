"""Simulated broker: cash, positions and an append-only trade ledger.

This class NEVER talks to a real brokerage. It is used identically by the
backtester and the paper-trading engine, so the paper-trading results are
produced by exactly the same accounting code that was backtested.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from portfolio.costs import CostModel

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float          # average cost per share incl. trading costs
    entry_date: pd.Timestamp
    peak_price: float        # highest close since entry (for the trailing stop)


@dataclass
class Fill:
    date: pd.Timestamp
    ticker: str
    side: str                # BUY / SELL
    shares: float
    reference_price: float   # price before costs (e.g. the open)
    fill_price: float        # price after spread + slippage
    commission: float
    cost: float              # total trading cost in $ vs. reference price
    gross_value: float       # shares * fill_price
    realized_pnl: float      # for sells: (fill - avg_cost) * shares - commission
    cash_after: float
    reason: str


class SimulatedBroker:
    def __init__(self, cash: float, cost_model: CostModel, allow_fractional: bool = False):
        self.cash = float(cash)
        self.cost_model = cost_model
        self.allow_fractional = allow_fractional
        self.positions: Dict[str, Position] = {}
        self.fills: List[Fill] = []

    # ------------------------------------------------------------------ queries
    def shares(self, ticker: str) -> float:
        p = self.positions.get(ticker)
        return p.shares if p else 0.0

    def market_value(self, prices: pd.Series) -> float:
        return float(sum(p.shares * prices.get(t, np.nan) for t, p in self.positions.items()
                         if np.isfinite(prices.get(t, np.nan))))

    def equity(self, prices: pd.Series) -> float:
        return self.cash + self.market_value(prices)

    def weights(self, prices: pd.Series) -> pd.Series:
        eq = self.equity(prices)
        if eq <= 0:
            return pd.Series(dtype=float)
        return pd.Series({t: p.shares * prices[t] / eq for t, p in self.positions.items()})

    # ------------------------------------------------------------------ trading
    def _round(self, shares: float) -> float:
        return shares if self.allow_fractional else float(np.floor(abs(shares)) * np.sign(shares))

    def execute(self, date: pd.Timestamp, ticker: str, shares: float, reference_price: float,
                reason: str) -> Optional[Fill]:
        """Execute a market order for ``shares`` (positive = buy, negative = sell).

        Buys are reduced if cash is insufficient (no margin/leverage).
        Sells are capped at the shares held (no short selling).
        """
        if not np.isfinite(reference_price) or reference_price <= 0:
            logger.warning("%s %s: no valid price, order skipped", date.date(), ticker)
            return None
        shares = self._round(shares)
        if shares == 0:
            return None
        cm = self.cost_model
        if shares < 0:
            shares = -min(abs(shares), self.shares(ticker))
            if shares == 0:
                return None
        fill_price = cm.fill_price(reference_price, shares)
        commission = cm.commission(shares)
        if shares > 0:
            max_affordable = (self.cash - commission) / fill_price
            if shares > max_affordable:
                shares = self._round(max_affordable)
                if shares <= 0:
                    return None
                commission = cm.commission(shares)
        gross = shares * fill_price
        self.cash -= gross + commission
        cost = abs(shares) * abs(fill_price - reference_price) + commission

        realized = 0.0
        pos = self.positions.get(ticker)
        if shares > 0:
            if pos is None:
                self.positions[ticker] = Position(ticker, shares, (gross + commission) / shares,
                                                  date, reference_price)
            else:
                total_cost = pos.avg_cost * pos.shares + gross + commission
                pos.shares += shares
                pos.avg_cost = total_cost / pos.shares
        else:
            realized = (fill_price - pos.avg_cost) * abs(shares) - commission
            pos.shares += shares
            if pos.shares <= 1e-9:
                del self.positions[ticker]

        fill = Fill(date, ticker, "BUY" if shares > 0 else "SELL", abs(shares), reference_price,
                    fill_price, commission, cost, abs(gross), realized, self.cash, reason)
        self.fills.append(fill)
        return fill

    def update_peaks(self, prices: pd.Series) -> None:
        for t, p in self.positions.items():
            px = prices.get(t, np.nan)
            if np.isfinite(px):
                p.peak_price = max(p.peak_price, px)

    def accrue_interest(self, daily_rate: float) -> None:
        if self.cash > 0 and np.isfinite(daily_rate):
            self.cash *= 1.0 + daily_rate

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=list(Fill.__dataclass_fields__))
        return pd.DataFrame([f.__dict__ for f in self.fills])
