"""Transaction cost model.

Three separate costs are modeled:

* **Commission**: per-share fee with an optional minimum ($0 by default,
  since most U.S. retail brokers are commission-free).
* **Bid/ask spread**: you buy at the ask and sell at the bid, so each trade
  pays half the spread relative to the mid price. Mega-cap spreads are
  roughly 1-3 bps, so 2 bps is the default.
* **Slippage / market impact**: the price moves between the decision and the
  fill. 5 bps is a conservative default for $100k orders in liquid large caps.

Spread and slippage move the FILL PRICE (worse for us in both directions);
commission is charged separately in cash.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0

    @classmethod
    def from_config(cls, ccfg) -> "CostModel":
        return cls(ccfg.commission_per_share, ccfg.min_commission,
                   ccfg.half_spread_bps, ccfg.slippage_bps)

    @property
    def price_impact(self) -> float:
        """One-way fractional price penalty (spread + slippage)."""
        return (self.half_spread_bps + self.slippage_bps) / 10_000.0

    def fill_price(self, reference_price: float, shares: float) -> float:
        """Buy fills above the reference price, sell fills below it."""
        direction = 1.0 if shares > 0 else -1.0
        return reference_price * (1.0 + direction * self.price_impact)

    def commission(self, shares: float) -> float:
        if shares == 0:
            return 0.0
        return max(self.min_commission, abs(shares) * self.commission_per_share)

    def with_total_bps(self, total_bps: float) -> "CostModel":
        """Same model with a different total one-way cost (for sensitivity tests).

        The total is split in the same spread:slippage proportion as the original.
        """
        base = self.half_spread_bps + self.slippage_bps
        share = self.half_spread_bps / base if base > 0 else 0.3
        return CostModel(self.commission_per_share, self.min_commission,
                         total_bps * share, total_bps * (1 - share))
