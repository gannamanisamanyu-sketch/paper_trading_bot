"""Point-in-time fundamental data from SEC EDGAR (XBRL "companyfacts" API).

Why not use yfinance fundamentals? ``yf.Ticker(t).info`` only gives *today's*
P/E, ROE, etc. Using today's fundamentals in a 2017 backtest would leak
future information (look-ahead bias). SEC EDGAR instead records every
reported number together with the date it was FILED. A value is only used
after its filing date, which is exactly when the market could have known it.

Design choices
--------------
* Annual (10-K) figures only. Quarterly XBRL facts mix 3-month and
  year-to-date durations and are much harder to make reliable. Annual data is
  slower to update but clean and auditable.
* "As first reported": for each fiscal year we use the value from the FIRST
  filing that contained it. Later restatements would not have been known at
  the time. Every metric carries its OWN availability date (e.g. EPS can be
  tagged in a different filing than revenue).
* Growth rates use the current and prior year as shown side by side in the
  SAME 10-K, so both numbers were public on the same day and share one definition.
* One trading-day lag after the filing date, because filings often arrive
  after the market close.
* Several XBRL tags are tried per concept, because companies (and especially
  banks) report revenue under different tags. If nothing is found the factor
  is NaN and is treated as neutral (z-score 0) by the strategy.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Concept -> candidate us-gaap tags, in priority order.
CONCEPT_TAGS: Dict[str, List[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
        "InterestAndDividendIncomeOperating",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "EarningsPerShareBasic",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "liabilities": ["Liabilities"],
    "liabilities_and_equity": ["LiabilitiesAndStockholdersEquity"],
}
DURATION_CONCEPTS = {"revenue", "net_income", "eps_diluted"}

# Yahoo ticker -> SEC ticker spelling differences
SEC_TICKER_ALIASES = {"BRK-B": "BRK-B", "BF-B": "BF-B"}

# Companies that re-organized under a NEW SEC registrant (new holding company,
# re-domiciliation, merger structure). The SEC ticker list maps the ticker to
# the current CIK only, so the predecessor's filings are added here so the
# fundamental history is continuous.
PREDECESSOR_CIKS: Dict[str, List[int]] = {
    "GOOGL": [1288776],          # Google Inc. -> Alphabet Inc. (2015)
    "AVGO": [1649338, 1441634],  # Avago -> Broadcom Ltd (2016) -> Broadcom Inc. (2018)
    "DIS": [1001039],            # old Walt Disney Co. (pre-2019 Fox deal)
    "BLK": [1364742],            # BlackRock Inc. (pre-2024 holding-company change)
    "XOM": [34088],              # Exxon Mobil Corp (pre holding-company reorganization)
}


def _get(url: str, user_agent: str, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 404:
            return resp
        time.sleep(1.5 * (attempt + 1))
    resp.raise_for_status()
    return resp


def load_cik_map(cache_dir: Path, user_agent: str, refresh: bool = False) -> Dict[str, int]:
    path = cache_dir / "sec_company_tickers.json"
    if not path.exists() or refresh:
        resp = _get(SEC_TICKER_URL, user_agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    raw = json.loads(path.read_text())
    return {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}


def iter_company_facts(tickers: List[str], cache_dir: Path, user_agent: str,
                       refresh: bool = False):
    """Yield (ticker, facts-or-None) one company at a time.

    A generator keeps memory low: each companyfacts JSON can be several MB, so
    only one company's raw filings are held in memory at once.
    """
    facts_dir = cache_dir / "sec_facts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    cik_map = load_cik_map(cache_dir, user_agent, refresh)
    for t in tickers:
        sec_ticker = SEC_TICKER_ALIASES.get(t, t).upper()
        cik = cik_map.get(sec_ticker) or cik_map.get(sec_ticker.replace("-", "."))
        ciks = ([cik] if cik is not None else []) + PREDECESSOR_CIKS.get(t, [])
        if not ciks:
            logger.warning("No SEC CIK for %s; fundamentals unavailable", t)
            yield t, None
            continue
        docs = [d for d in (_fetch_facts(c, facts_dir, user_agent, refresh) for c in ciks) if d]
        if not docs:
            logger.warning("SEC facts unavailable for %s", t)
        yield t, (_merge_facts(docs) if docs else None)


def _fetch_facts(cik: int, facts_dir: Path, user_agent: str, refresh: bool) -> Optional[dict]:
    path = facts_dir / f"CIK{cik:010d}.json"
    if not path.exists() or refresh:
        resp = _get(SEC_FACTS_URL.format(cik=cik), user_agent)
        if resp.status_code != 200:
            return None
        path.write_bytes(resp.content)
        time.sleep(0.12)  # SEC fair-access policy: <= 10 requests/second
    return json.loads(path.read_text())


def _merge_facts(docs: List[dict]) -> dict:
    """Merge us-gaap facts from several registrants (current + predecessors)."""
    if len(docs) == 1:
        return docs[0]
    merged: Dict[str, dict] = {}
    for doc in docs:
        for tag, node in doc.get("facts", {}).get("us-gaap", {}).items():
            tgt = merged.setdefault(tag, {"units": {}})
            for unit, rows in node.get("units", {}).items():
                tgt["units"].setdefault(unit, []).extend(rows)
    return {"facts": {"us-gaap": merged}}


def _concept_rows(facts: dict, tags: List[str], duration: bool) -> pd.DataFrame:
    """All annual (10-K) observations of a concept across candidate tags.

    Columns: end, val, filed, priority (index of the tag in ``tags``; lower = preferred).
    Duration items (revenue, net income, EPS) are restricted to ~1-year periods.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    pieces = []
    for priority, tag in enumerate(tags):
        node = gaap.get(tag)
        if not node:
            continue
        for unit, rows in node.get("units", {}).items():
            if unit not in ("USD", "USD/shares"):
                continue
            df = pd.DataFrame(rows)
            if df.empty or "form" not in df:
                continue
            df = df[df["form"].isin(["10-K", "10-K/A", "10-KT"])]
            if df.empty:
                continue
            df = df.assign(end=pd.to_datetime(df["end"]), filed=pd.to_datetime(df["filed"]))
            if duration:
                if "start" not in df:
                    continue
                days = (df["end"] - pd.to_datetime(df["start"])).dt.days
                df = df[(days >= 340) & (days <= 390)]
            pieces.append(df[["end", "val", "filed"]].assign(priority=priority))
    if not pieces:
        return pd.DataFrame(columns=["end", "val", "filed", "priority"])
    return pd.concat(pieces, ignore_index=True)


def first_reported(rows: pd.DataFrame) -> pd.DataFrame:
    """For each period end: the value from the EARLIEST filing that reported it.

    Tag priority only breaks ties within that same filing. (Choosing the
    preferred tag first would silently use a later 10-K's number on the
    earlier filing's date, which is a subtle look-ahead.)
    Returns index = period end, columns: val, filed, priority.
    """
    if rows.empty:
        return pd.DataFrame(columns=["val", "filed", "priority"])
    r = rows.sort_values(["end", "filed", "priority"])
    return r.groupby("end")[["val", "filed", "priority"]].first()


def prior_in_same_filing(rows: pd.DataFrame, first: pd.DataFrame) -> pd.Series:
    """Prior-year comparative value reported in the SAME filing and tag as ``first``.

    A 10-K shows the current year next to the prior year on the same basis, so
    growth computed from one filing is both point-in-time and consistently defined
    (no mixing of tags or later restatements).
    """
    out = pd.Series(np.nan, index=first.index)
    if rows.empty:
        return out
    for end, f in first.iterrows():
        cand = rows[(rows["filed"] == f["filed"]) & (rows["priority"] == f["priority"])]
        gap = (end - cand["end"]).dt.days
        cand = cand[(gap >= 330) & (gap <= 400)]
        if len(cand):
            out[end] = float(cand.sort_values("end").iloc[-1]["val"])
    return out


def build_fundamental_table(facts: Optional[dict]) -> pd.DataFrame:
    """One row per fiscal year with values AND the date each value became public.

    Columns (``*_filed`` = first filing date of that number):
      net_income, net_income_prev, net_income_filed
      revenue, revenue_prev, revenue_filed
      equity, equity_prev, equity_filed
      liabilities, liabilities_filed
      eps_diluted, eps_diluted_filed
    """
    if not facts:
        return pd.DataFrame()
    cols: Dict[str, pd.Series] = {}
    rows_by: Dict[str, pd.DataFrame] = {}
    firsts: Dict[str, pd.DataFrame] = {}
    for concept, tags in CONCEPT_TAGS.items():
        rows = _concept_rows(facts, tags, concept in DURATION_CONCEPTS)
        rows_by[concept], firsts[concept] = rows, first_reported(rows)
    ni = firsts["net_income"]
    if ni.empty:
        return pd.DataFrame()
    index = ni.index  # fiscal years = periods with a reported annual net income
    for concept in ("net_income", "revenue", "equity", "eps_diluted"):
        f = firsts[concept].reindex(index)
        cols[concept] = f["val"].astype(float)
        cols[f"{concept}_filed"] = f["filed"]
        if concept in ("net_income", "revenue", "equity"):
            cols[f"{concept}_prev"] = prior_in_same_filing(rows_by[concept], firsts[concept]).reindex(index)
    liab = firsts["liabilities"].reindex(index)
    le = firsts["liabilities_and_equity"].reindex(index)
    eq = firsts["equity"].reindex(index)
    derived = le["val"].astype(float) - eq["val"].astype(float)
    derived_filed = pd.concat([le["filed"], eq["filed"]], axis=1).max(axis=1)
    cols["liabilities"] = liab["val"].astype(float).fillna(derived)
    cols["liabilities_filed"] = liab["filed"].where(liab["val"].notna(), derived_filed)
    table = pd.DataFrame(cols, index=index).sort_index()
    table.index.name = "period_end"
    return table


def compute_annual_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Derive factor inputs, each with its own availability date (``avail_<metric>``).

    * ROE = net income / average(beginning, ending) equity, both from the 10-K
      balance sheet; available once NI and equity have both been filed
    * Leverage = total liabilities / equity (debt-to-equity proxy reported by
      every company; banks are handled by ranking fundamentals *within* sector)
    * Revenue growth = revenue_t / revenue_(t-1) - 1, both years from the same filing
    * Earnings growth = (NI_t - NI_(t-1)) / |NI_(t-1)|, same filing, clipped to
      [-1, +1] so a swing from a tiny loss to a profit doesn't create a +5000% outlier
    * EPS (diluted), available on its own filing date
    """
    if table.empty:
        return table
    t = table.copy()
    later = lambda *c: pd.concat([t[x] for x in c], axis=1).max(axis=1)
    eq, eq_prev = t["equity"], t["equity_prev"]
    avg_eq = ((eq + eq_prev) / 2).fillna(eq)
    t["roe"] = np.where(avg_eq > 0, t["net_income"] / avg_eq, np.nan)
    t["avail_roe"] = later("net_income_filed", "equity_filed")
    t["leverage"] = np.where(eq > 0, t["liabilities"] / eq, np.nan)
    t["avail_leverage"] = later("liabilities_filed", "equity_filed")
    prev_rev = t["revenue_prev"]
    t["revenue_growth"] = np.where(prev_rev > 0, t["revenue"] / prev_rev - 1, np.nan)
    t["avail_revenue_growth"] = t["revenue_filed"]
    prev_ni = t["net_income_prev"]
    t["earnings_growth"] = np.clip(
        np.where(prev_ni.abs() > 0, (t["net_income"] - prev_ni) / prev_ni.abs(), np.nan), -1, 1)
    t["avail_earnings_growth"] = t["net_income_filed"]
    t["avail_eps_diluted"] = t["eps_diluted_filed"]
    return t


def point_in_time_panel(
    metrics_by_ticker: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    column: str,
    lag_days: int = 1,
    max_age_days: int = 550,
    divisor: Optional[pd.DataFrame] = None,
    avail_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build a (date x ticker) panel where each cell holds the latest value
    that was PUBLIC on that date (filing date + ``lag_days`` trading days).

    Values older than ``max_age_days`` (e.g. a company stopped filing) are
    dropped instead of being carried forward forever.

    ``divisor`` (date x ticker), if given, divides each value on the date it
    becomes effective. It is used to restate per-share numbers (EPS) onto the
    same split basis as Yahoo's split-adjusted prices.
    """
    avail_col = avail_col or f"avail_{column}"
    out = pd.DataFrame(index=dates, columns=list(metrics_by_ticker), dtype=float)
    for ticker, m in metrics_by_ticker.items():
        if m is None or m.empty or column not in m:
            continue
        if avail_col not in m:
            continue
        s = m[[column, avail_col]].dropna().rename(columns={avail_col: "available"})
        if s.empty:
            continue
        s = s.reset_index().sort_values(["available", "period_end"])
        # never let an OLDER fiscal year replace a newer one that was already public
        s = s[s["period_end"] >= s["period_end"].cummax()]
        # if the same filing date reports several years, the latest period wins
        s = s.groupby("available").last()
        pos = dates.searchsorted(s.index, side="right")  # first trading day AFTER filing
        pos = pos + (lag_days - 1)
        valid = pos < len(dates)
        eff_dates = dates[pos[valid]]
        vals = pd.Series(s[column].values[valid], index=eff_dates)
        if divisor is not None and ticker in divisor:
            vals = vals / divisor[ticker].reindex(eff_dates).fillna(1.0).values
        vals = vals[~vals.index.duplicated(keep="last")]
        period_end = pd.Series(pd.to_datetime(s["period_end"].values[valid]), index=eff_dates)
        period_end = period_end[~period_end.index.duplicated(keep="last")]
        col = vals.reindex(dates).ffill()
        pe = period_end.reindex(dates).ffill()
        age = (dates.to_series() - pe).dt.days
        col[age > max_age_days] = np.nan
        out[ticker] = col.values
    return out


class FundamentalData:
    """Point-in-time fundamental panels aligned to the market-data calendar."""

    FIELDS = ["roe", "leverage", "revenue_growth", "earnings_growth", "eps_diluted"]

    def __init__(self, panels: Dict[str, pd.DataFrame], coverage: pd.DataFrame):
        self.panels = panels
        self.coverage = coverage

    def get(self, field: str) -> pd.DataFrame:
        return self.panels[field]

    @classmethod
    def build(cls, tickers: List[str], dates: pd.DatetimeIndex, cache_dir: Path,
              user_agent: str, refresh: bool = False,
              split_factor_after: Optional[pd.DataFrame] = None,
              snapshot_file: Optional[Path] = None) -> "FundamentalData":
        metrics = {t: compute_annual_metrics(build_fundamental_table(f))
                   for t, f in iter_company_facts(tickers, cache_dir, user_agent, refresh)}
        if snapshot_file is not None:
            save_snapshot(metrics, snapshot_file)
        return cls.from_metrics(metrics, dates, split_factor_after)

    @classmethod
    def from_metrics(cls, metrics: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                     split_factor_after: Optional[pd.DataFrame] = None) -> "FundamentalData":
        panels = {}
        for f in cls.FIELDS:
            # EPS is per-share: convert to today's split basis at the filing date
            div = split_factor_after if f == "eps_diluted" else None
            panels[f] = point_in_time_panel(metrics, dates, f, divisor=div)
        coverage = pd.DataFrame({f: panels[f].notna().mean() for f in cls.FIELDS})
        logger.info("Fundamental coverage (share of days with data):\n%s",
                    coverage.mean().round(2).to_string())
        return cls(panels, coverage)

    @classmethod
    def empty(cls, tickers: List[str], dates: pd.DatetimeIndex) -> "FundamentalData":
        """Fallback when SEC data is unavailable: all-NaN panels (neutral factors)."""
        nan = pd.DataFrame(np.nan, index=dates, columns=tickers)
        panels = {f: nan.copy() for f in cls.FIELDS}
        return cls(panels, pd.DataFrame({f: nan.notna().mean() for f in cls.FIELDS}))


def save_snapshot(metrics: Dict[str, pd.DataFrame], path: Path) -> None:
    """Save the derived annual metrics (with filing dates) as one small parquet file.

    SEC EDGAR data is public domain, so this snapshot can be committed to the
    repository. A fresh clone or a cloud deployment can then run without
    downloading ~300 MB of raw XBRL filings.
    """
    frames = [m.reset_index().assign(ticker=t).dropna(axis=1, how="all")
              for t, m in metrics.items() if m is not None and len(m)]
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    logger.info("Saved fundamentals snapshot -> %s", path)


def load_snapshot(path: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    return {t: g.drop(columns="ticker").set_index("period_end").sort_index()
            for t, g in df.groupby("ticker")}


def load_fundamentals(cfg, md, refresh: bool = False) -> FundamentalData:
    """Build point-in-time fundamentals.

    Source order: (1) raw SEC filings if they are cached locally or a refresh
    is requested; (2) the committed snapshot (``data/snapshot/sec_metrics.parquet``);
    (3) a fresh SEC download. If everything fails, factors become neutral (z = 0).
    """
    import os

    cache = cfg.path(cfg.data.cache_dir)
    snapshot = cfg.path(cfg.data.snapshot_file)
    user_agent = os.environ.get("SEC_USER_AGENT") or cfg.data.sec_user_agent
    have_raw = (cache / "sec_facts").exists() and any((cache / "sec_facts").glob("*.json"))
    if snapshot.exists() and not refresh and not have_raw:
        logger.info("Using committed fundamentals snapshot %s", snapshot.name)
        return FundamentalData.from_metrics(load_snapshot(snapshot), md.dates, md.split_factor_after)
    try:
        return FundamentalData.build(md.tickers, md.dates, cache, user_agent, refresh,
                                     split_factor_after=md.split_factor_after, snapshot_file=snapshot)
    except (requests.RequestException, OSError) as exc:
        if snapshot.exists():
            logger.warning("SEC EDGAR unavailable (%s); using snapshot", exc)
            return FundamentalData.from_metrics(load_snapshot(snapshot), md.dates, md.split_factor_after)
        logger.error("SEC EDGAR unavailable (%s). Fundamental factors set to neutral.", exc)
        return FundamentalData.empty(md.tickers, md.dates)
