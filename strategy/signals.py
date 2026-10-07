"""Systematic BUY / SELL / HOLD rules.

On every rebalance date (after the close):

SELL a held stock if ANY of:
  1. it is no longer eligible (liquidity/data screen failed)
  2. its score < ``sell_score`` (clearly below-average now)
  3. its rank > ``hold_rank_buffer * max_positions`` (e.g. outside the top 30)

HOLD a held stock otherwise. The rank buffer (hysteresis) means a stock is
bought in the top 15 but only sold once it drops out of the top 30. This
avoids selling a name that slips from #15 to #16 and buying it back next month,
which would only generate transaction costs.

BUY new stocks, best score first, while open slots remain, if ALL of:
  1. eligible
  2. score > ``min_buy_score``
  3. price above its 200-day SMA (no new longs in downtrends)
  4. RSI(14) < ``rsi_max_entry`` (don't chase extremely overbought names)
  5. not stopped out since the previous rebalance (no immediate re-entry)

Between rebalances, the only trading is risk-driven: trailing stop-losses
(see risk/limits.py). No discretionary decisions are ever required.
"""
from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


def generate_signals(scores: pd.DataFrame, holdings: Iterable[str], scfg, max_positions: int,
                     stopped_out: Set[str] = frozenset()) -> pd.DataFrame:
    """Return one row per ticker that is held or bought: signal, score, rank, reason."""
    holdings = set(holdings)
    rows = []
    hold_rank_limit = int(round(scfg.hold_rank_buffer * max_positions))

    keep = []
    for t in sorted(holdings):
        if t not in scores.index:
            rows.append({"ticker": t, "signal": SELL, "score": float("nan"), "rank": float("nan"),
                         "reason": "no longer eligible (liquidity/data screen)"})
            continue
        s = scores.loc[t]
        if s["score"] < scfg.sell_score:
            rows.append({"ticker": t, "signal": SELL, "score": s["score"], "rank": s["rank"],
                         "reason": f"score {s['score']:.2f} < sell threshold {scfg.sell_score:.2f}"})
        elif s["rank"] > hold_rank_limit:
            rows.append({"ticker": t, "signal": SELL, "score": s["score"], "rank": s["rank"],
                         "reason": f"rank {int(s['rank'])} outside top {hold_rank_limit}"})
        else:
            keep.append(t)
            rows.append({"ticker": t, "signal": HOLD, "score": s["score"], "rank": s["rank"],
                         "reason": f"rank {int(s['rank'])} within top {hold_rank_limit}"})

    slots = max_positions - len(keep)
    for t, s in scores.iterrows():
        if slots <= 0:
            break
        if t in holdings or t in stopped_out:
            continue
        if s["score"] <= scfg.min_buy_score:
            break  # sorted by score: nothing further qualifies
        if not bool(s["above_sma200"]):
            continue
        if pd.notna(s["rsi"]) and s["rsi"] >= scfg.rsi_max_entry:
            continue
        rows.append({"ticker": t, "signal": BUY, "score": s["score"], "rank": s["rank"],
                     "reason": f"rank {int(s['rank'])}, score {s['score']:.2f}, above 200d SMA"})
        slots -= 1

    out = pd.DataFrame(rows, columns=["ticker", "signal", "score", "rank", "reason"])
    return out
