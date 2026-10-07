"""Overview: what the project is, headline results, and the honest verdict."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, BT, GITHUB_URL, GREY, LIVE, backtest_equity, load_csv, load_json,
                              metrics, missing, money, paper_banner, pct, style_fig)


def page() -> None:
    st.title("Algorithmic U.S. Equity Paper Trading & Risk Management System")
    st.markdown(
        "A Python system that ranks ~110 large-cap U.S. stocks on **momentum, trend, quality, value, growth "
        "and risk**, builds a risk-controlled 15-stock portfolio, backtests it with no look-ahead bias, and "
        "runs it forward as a **$100,000 paper portfolio** that updates automatically every trading day.")
    paper_banner()
    if GITHUB_URL:
        st.markdown(f"[📂 Source code on GitHub]({GITHUB_URL})")

    m = metrics()
    eq = backtest_equity()
    if m is None or eq.empty:
        missing("Backtest results not found. Run `python main.py report`.")
        return
    full, spy = m["periods"]["full"]["strategy"], m["periods"]["full"]["spy"]
    test, spy_test = m["periods"]["test"]["strategy"], m["periods"]["test"]["spy"]
    b = m["bounds"]

    st.subheader(f"Backtest {b['full'][0]} → {b['full'][1]}, starting with $100,000")
    c = st.columns(4)
    c[0].metric("Strategy final value", money(full["final_value"]), f"SPY {money(spy['final_value'])}",
                delta_color="off", border=True)
    c[1].metric("Annual return (CAGR)", pct(full["cagr"], 2), f"SPY {pct(spy['cagr'], 2)}", delta_color="off", border=True)
    c[2].metric("Sharpe ratio", f"{full['sharpe']:.2f}", f"SPY {spy['sharpe']:.2f}", delta_color="off", border=True)
    c[3].metric("Max drawdown", pct(full["max_drawdown"]), f"SPY {pct(spy['max_drawdown'])}", delta_color="off", border=True)

    fig = go.Figure()
    for name, (a0, a1), col in [("train", b["train"], "rgba(156,163,175,0.12)"),
                                ("validation", b["validation"], "rgba(250,204,21,0.12)"),
                                ("out-of-sample test", b["test"], "rgba(34,197,94,0.12)")]:
        fig.add_vrect(x0=a0, x1=a1, fillcolor=col, line_width=0, annotation_text=name,
                      annotation_position="top left")
    fig.add_scatter(x=eq.index, y=eq["equity"], name="Strategy", line=dict(color=BLUE, width=2.2))
    fig.add_scatter(x=eq.index, y=eq["spy"], name="SPY buy & hold", line=dict(color=GREY))
    fig.update_layout(yaxis_tickprefix="$")
    st.plotly_chart(style_fig(fig, 430, "Portfolio value: strategy vs. S&P 500 (SPY, dividends reinvested)"))

    beat = full["sharpe"] > spy["sharpe"] and full["alpha"] > 0
    st.markdown(
        f"""<div class="verdict"><b>Honest verdict.</b> The strategy {'<b>did</b>' if beat else '<b>did not</b>'}
        beat SPY on a risk-adjusted basis (Sharpe {full['sharpe']:.2f} vs. {spy['sharpe']:.2f}; alpha
        {pct(full['alpha'], 2)}/yr, t-stat {full['alpha_t']:.2f}). It behaves like a <b>low-beta defensive
        portfolio</b> (beta {full['beta']:.2f}): much smaller drawdowns, but it gives up a lot of upside in bull markets.
        On the untouched out-of-sample period ({b['test'][0]} → {b['test'][1]}) it returned
        {pct(test['cagr'], 1)}/yr vs. {pct(spy_test['cagr'], 1)} for SPY.</div>""",
        unsafe_allow_html=True)

    hs = load_csv(str(BT / "stress_historical.csv"))
    if len(hs):
        st.subheader("Where the risk management helped, and where it hurt")
        fig = go.Figure()
        fig.add_bar(x=hs["episode"], y=hs["strategy_return"], name="Strategy", marker_color=BLUE,
                    text=[pct(v) for v in hs["strategy_return"]], textposition="outside")
        fig.add_bar(x=hs["episode"], y=hs["spy_return"], name="SPY", marker_color=GREY,
                    text=[pct(v) for v in hs["spy_return"]], textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", barmode="group")
        st.plotly_chart(style_fig(fig, 380))

    st.subheader("How it works")
    cols = st.columns(4)
    steps = [
        ("1 · Data", "Daily prices (Yahoo Finance) and fundamentals from SEC 10-K filings, each used only "
                     "after the date it became public."),
        ("2 · Score", "Every month, each stock gets a weighted score from 7 factors (momentum, trend, sector, "
                      "quality, value, growth, minus a risk penalty)."),
        ("3 · Build", "Buy the top 15 above their 200-day average. Volatility-adjusted sizing, 10% stock cap, "
                      "30% sector cap, 5% cash floor, 16% volatility target."),
        ("4 · Protect", "Trailing stops, a market-regime filter (SPY below its 200-day average → max 50% invested) "
                        "and drawdown control. Fills at the next open, with trading costs."),
    ]
    for col, (h, t) in zip(cols, steps):
        with col.container(border=True):
            st.markdown(f"**{h}**")
            st.caption(t)

    live = load_json(str(LIVE / "meta.json"))
    if live:
        st.info(f"Today's model portfolio (as of {live['as_of']}) holds {live['n_positions']} stocks, "
                f"{live['invested']:.0%} invested. Market regime: "
                f"{'risk-on ✅' if live['market_risk_on'] else 'risk-off ⚠️'}. See **Today's Signals**.", icon="📡")
