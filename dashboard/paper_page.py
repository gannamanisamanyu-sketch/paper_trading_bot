"""Paper Portfolio: the persistent $100k paper account, updated daily by the GitHub Action."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, GREY, ROOT, get_config, load_csv, missing, money, paper_banner,
                              pct, sector_map, style_fig)
from risk import metrics as M


def page() -> None:
    cfg = get_config()
    st.title("📒 Paper Portfolio")
    paper_banner()
    root = ROOT / cfg.paper_trading.state_dir
    state_p = root / "state.json"
    if not state_p.exists():
        missing("No paper portfolio yet. Run `python main.py paper init`, then `python main.py paper run`.")
        return
    state = json.loads(state_p.read_text())
    eq = load_csv(str(root / "daily_equity.csv"), parse_dates=["date"])
    trades = load_csv(str(root / "trades.csv"), parse_dates=["date"])
    orders = load_csv(str(root / "pending_orders.csv"))
    signals = load_csv(str(root / "signals.csv"), parse_dates=["date"])
    alerts = load_csv(str(root / "alerts.csv"))
    sectors = sector_map()

    start = float(state["starting_capital"])
    pos = pd.DataFrame(state["positions"]).T
    equity = float(eq["equity"].iloc[-1]) if len(eq) else start
    daily_pnl = float(eq["daily_pnl"].iloc[-1]) if len(eq) else 0.0

    st.caption(f"Started {state.get('created', '')[:10]} with {money(start)} · last processed trading day "
               f"**{state.get('last_processed_date')}** · updated automatically after each U.S. market close.")
    c = st.columns(4)
    c[0].metric("Portfolio value", money(equity), pct(equity / start - 1, 2) + " since start", border=True)
    c[1].metric("Daily P&L", money(daily_pnl), pct(daily_pnl / (equity - daily_pnl), 2) if equity else None, border=True)
    c[2].metric("Cash", money(state["cash"]), pct(state["cash"] / equity) + " of portfolio", delta_color="off", border=True)
    c[3].metric("Positions", len(pos), border=True)

    if len(eq) > 1:
        e = eq.set_index("date")["equity"]
        r = e.pct_change().dropna()
        c = st.columns(3)
        c[0].metric("Volatility (ann.)", pct(M.annualized_volatility(r)) if len(r) > 10 else "needs 10+ days", border=True)
        c[1].metric("Sharpe (since start)", f"{M.sharpe_ratio(r):.2f}" if len(r) > 20 else "needs 20+ days", border=True)
        c[2].metric("Max drawdown", pct(M.max_drawdown(e)), border=True)
        fig = go.Figure()
        fig.add_scatter(x=e.index, y=e, name="Paper portfolio", line=dict(color=BLUE, width=2.4))
        fig.update_layout(yaxis_tickprefix="$")
        st.plotly_chart(style_fig(fig, 340, "Paper portfolio value"))

    if len(pos):
        pos["shares"] = pos["shares"].astype(float)
        pos["avg_cost"] = pos["avg_cost"].astype(float)
        pos["sector"] = pos.index.map(sectors)
        pos["cost_basis"] = pos["shares"] * pos["avg_cost"]
        a, b = st.columns(2)
        sec = pos.groupby("sector")["cost_basis"].sum().reset_index()
        sec.loc[len(sec)] = ["Cash", state["cash"]]
        a.plotly_chart(style_fig(px.pie(sec, names="sector", values="cost_basis", hole=0.5), 360,
                                 "Allocation by sector (at cost)"))
        b.plotly_chart(style_fig(px.pie(pos.reset_index(), names="index", values="cost_basis", hole=0.5), 360,
                                 "Allocation by stock (at cost)"))
        st.subheader("Current positions")
        st.dataframe(pos[["sector", "shares", "avg_cost", "cost_basis", "entry_date", "peak_price"]],
                     column_config={"avg_cost": st.column_config.NumberColumn("avg cost", format="$%.2f"),
                                    "cost_basis": st.column_config.NumberColumn("cost basis", format="$%.0f"),
                                    "peak_price": st.column_config.NumberColumn("stop peak", format="$%.2f"),
                                    "shares": st.column_config.NumberColumn(format="%d")})
    else:
        st.info("No positions yet. Orders created after a close are filled at the next market open.")

    a, b = st.columns(2)
    with a:
        st.subheader("Orders for the next open")
        if len(orders):
            st.dataframe(orders, hide_index=True,
                         column_config={"target_weight": st.column_config.NumberColumn("target", format="percent")})
        else:
            st.caption("No pending orders.")
    with b:
        st.subheader("Latest rebalance signals")
        if len(signals):
            latest = signals[signals["date"] == signals["date"].max()]
            st.dataframe(latest[["ticker", "signal", "score", "rank", "target_weight", "reason"]], hide_index=True,
                         column_config={"target_weight": st.column_config.NumberColumn("target", format="percent"),
                                        "score": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.caption("No signals yet.")

    st.subheader("Trade history")
    if len(trades):
        st.dataframe(trades.sort_values("date", ascending=False), hide_index=True, height=320)
        closed = trades[trades["signal"] == "SELL"]
        if len(closed):
            st.caption(f"Realized P&L to date: {money(closed['pnl'].sum())} across {len(closed)} sells.")
    else:
        st.caption("No simulated fills yet.")
    if len(alerts):
        st.subheader("Alerts")
        st.dataframe(alerts.tail(20).iloc[::-1], hide_index=True)
