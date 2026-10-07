"""Shared helpers for the web app: file loaders, formatting, chart styling."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_config  # noqa: E402

BT = ROOT / "results" / "backtest"
LIVE = ROOT / "results" / "live"
REPORTS = ROOT / "reports"
BLUE, GREY, RED, GREEN, PURPLE = "#2563eb", "#9ca3af", "#dc2626", "#16a34a", "#8b5cf6"

# Set after the repository is created (shown on the Overview page)
GITHUB_URL = "https://github.com/gannamanisamanyu-sketch/paper_trading_bot"


@st.cache_resource
def get_config():
    return load_config()


@st.cache_data(ttl=600)
def load_csv(path: str, **kw) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_json(path: str) -> Optional[Dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(ttl=600)
def universe() -> pd.DataFrame:
    return pd.read_csv(ROOT / "config" / "universe.csv")


def sector_map() -> Dict[str, str]:
    u = universe()
    return dict(zip(u["ticker"], u["sector"]))


def backtest_equity() -> pd.DataFrame:
    df = load_csv(str(BT / "equity.csv"), parse_dates=["date"])
    return df.set_index("date") if len(df) else df


def metrics() -> Optional[Dict]:
    return load_json(str(BT / "metrics.json"))


def pct(x, d: int = 1) -> str:
    try:
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}%}"
    except TypeError:
        return "n/a"


def money(x) -> str:
    try:
        return "n/a" if x is None or not np.isfinite(x) else f"${x:,.0f}"
    except TypeError:
        return "n/a"


def style_fig(fig, h: int = 360, title: Optional[str] = None):
    has_title = bool(title or fig.layout.title.text)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=78 if has_title else 30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), hovermode="x unified",
                      title=dict(y=0.98, yanchor="top"))
    if title:
        fig.update_layout(title=title)
    return fig


def paper_banner() -> None:
    st.caption("⚠️ **Paper trading only.** Simulated results for education. No brokerage connection, "
               "no real money, not investment advice.")


def missing(msg: str) -> None:
    st.info(msg, icon="ℹ️")


CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px}
[data-testid="stMetricValue"] {font-size: 1.6rem}
.verdict {border-left: 4px solid #2563eb; padding: 10px 16px; border-radius: 6px;
          background: rgba(37, 99, 235, 0.07); margin: 6px 0 14px 0}
</style>
"""
