"""Web app entry point (Streamlit multi-page app).

Local:   python main.py dashboard        (or: streamlit run streamlit_app.py)
Cloud:   deploy streamlit_app.py on Streamlit Community Cloud

The app only reads files produced by the CLI (results/, data/paper/), except the
Backtest Lab page, which downloads market data and runs backtests on demand.
It never trades and has no brokerage connection.
"""
from __future__ import annotations

import streamlit as st

from dashboard import lab_page, methodology_page, overview, paper_page, results_pages, signals_page
from dashboard.common import CSS


def main() -> None:
    st.set_page_config(page_title="Equity Paper Trading & Risk System", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    pages = {
        "Live": [
            st.Page(overview.page, title="Overview", icon="🏠", default=True),
            st.Page(signals_page.page, title="Today's Signals", icon="📡", url_path="signals"),
            st.Page(paper_page.page, title="Paper Portfolio", icon="📒", url_path="paper"),
        ],
        "Research": [
            st.Page(results_pages.backtest_page, title="Backtest Results", icon="📈", url_path="backtest"),
            st.Page(results_pages.risk_page, title="Risk", icon="🛡️", url_path="risk"),
            st.Page(results_pages.robustness_page, title="Robustness", icon="🔬", url_path="robustness"),
            st.Page(lab_page.page, title="Backtest Lab", icon="🧪", url_path="lab"),
            st.Page(methodology_page.page, title="Methodology & Reports", icon="📄", url_path="methodology"),
        ],
    }
    nav = st.navigation(pages)
    with st.sidebar:
        st.caption("Educational paper-trading project. Not investment advice.")
    nav.run()


if __name__ == "__main__":
    main()
