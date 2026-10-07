"""Stock universe definition.

The universe is a fixed, hand-curated list of large, liquid U.S. companies that
were established S&P 500 members by the end of 2015 (the year before the
backtest starts), spread across all 11 GICS sectors. It was chosen from what
was knowable at that time, NOT from later performance. For example, NVDA and
TSLA are deliberately excluded: in 2015 Nvidia was a ~$18B company and Tesla
was not in the S&P 500 until December 2020.

Companies that were later acquired or delisted are kept in the file with
``status=delisted`` (or ``no_data`` when the free source lacks usable history). Free data sources (Yahoo Finance) no longer carry their
price history, so they cannot be traded in the backtest. They are listed
explicitly so the survivorship bias is documented and measurable rather than
hidden.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Universe:
    table: pd.DataFrame  # columns: ticker, name, sector, status, note

    @property
    def tradable(self) -> List[str]:
        """Tickers with available price history (status == active)."""
        return self.table.loc[self.table["status"] == "active", "ticker"].tolist()

    @property
    def delisted(self) -> pd.DataFrame:
        return self.table.loc[self.table["status"] != "active"]

    @property
    def sector_map(self) -> Dict[str, str]:
        return dict(zip(self.table["ticker"], self.table["sector"]))

    @property
    def name_map(self) -> Dict[str, str]:
        return dict(zip(self.table["ticker"], self.table["name"]))


def load_universe(path: Path) -> Universe:
    """Load and validate the universe CSV."""
    table = pd.read_csv(path, dtype=str).fillna("")
    required = {"ticker", "name", "sector", "status"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Universe file missing columns: {missing}")
    if table["ticker"].duplicated().any():
        dupes = table.loc[table["ticker"].duplicated(), "ticker"].tolist()
        raise ValueError(f"Duplicate tickers in universe: {dupes}")
    universe = Universe(table=table)
    logger.info(
        "Universe: %d active tickers across %d sectors; %d delisted names excluded "
        "(survivorship bias documented)",
        len(universe.tradable),
        table.loc[table["status"] == "active", "sector"].nunique(),
        len(universe.delisted),
    )
    return universe
