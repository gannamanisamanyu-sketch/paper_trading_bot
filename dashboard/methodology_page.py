"""Methodology & Reports: the investment-committee memo, research log and design notes."""
from __future__ import annotations

import streamlit as st

from dashboard.common import GITHUB_URL, REPORTS, ROOT, paper_banner


def _md(path):
    return path.read_text() if path.exists() else "_Not generated yet._"


def page() -> None:
    st.title("📄 Methodology & Reports")
    paper_banner()
    if GITHUB_URL:
        st.markdown(f"Full source code, README and tests: [{GITHUB_URL}]({GITHUB_URL})")
    tabs = st.tabs(["Investment committee memo", "How bias is prevented", "Research log", "Full performance report"])
    with tabs[0]:
        st.markdown(_md(REPORTS / "investment_committee_report.md"))
    with tabs[1]:
        st.markdown("""
### Look-ahead bias (using information you couldn't have had)
1. **Causal indicators.** Every moving average, momentum, RSI, volatility and beta value on day *t* uses only data up to *t*.
2. **Next-open execution.** Decisions use day *t*'s close and can only trade at day *t+1*'s open, with spread and slippage.
3. **Point-in-time fundamentals.** ROE, leverage, growth and EPS come from SEC 10-K filings and are used only from the day
   *after* each number was first filed. Growth compares two years as shown side by side in the same filing.
4. **Automated proof.** A unit test reruns the backtest on data cut off at a date and checks that every result up
   to that date is identical. Any peek into the future would break it.

### Survivorship bias (only testing companies that survived)
The stock list is fixed as of end-2015 using only what was known then (no Nvidia or Tesla, which became large later).
Eleven companies on the original list were later acquired or delisted (or lack usable free data), so they're
documented as excluded. This bias likely flatters results a little.

### Overfitting (fitting noise in one sample)
* Factor weights and limits set **before** any backtest; train / validation / out-of-sample split; settings
  frozen before the test period ran.
* Walk-forward test (choose from 4 configurations using prior years only).
* 30+ one-at-a-time sensitivity runs; **no variant beat SPY's Sharpe**, so the conclusion doesn't hinge on a setting.
* Every change made after seeing results is in the research log.

### Costs
2 bps half-spread + 5 bps slippage per trade (7 bps one-way), $0 commission, tested from 0 to 50 bps.
""")
    with tabs[2]:
        st.markdown(_md(ROOT / "RESEARCH_LOG.md"))
    with tabs[3]:
        text = _md(REPORTS / "performance_report.md")
        # charts are shown elsewhere in the app; drop image links from the markdown view
        st.markdown("\n".join(l for l in text.splitlines() if not l.startswith("![")))
