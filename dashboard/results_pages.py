"""Backtest Results, Risk and Robustness pages (read the files written by `main.py report`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, BT, GREEN, GREY, PURPLE, RED, backtest_equity, get_config, load_csv,
                              metrics, missing, money, paper_banner, pct, sector_map, style_fig)
from risk import metrics as M

KEYS = [("total_return", "Total return", "pct"), ("cagr", "CAGR", "pct"), ("volatility", "Volatility", "pct"),
        ("sharpe", "Sharpe", "num"), ("sortino", "Sortino", "num"), ("max_drawdown", "Max drawdown", "pct"),
        ("var_95", "VaR 95% (1d)", "pct"), ("es_95", "Exp. shortfall 95%", "pct"), ("beta", "Beta", "num"),
        ("alpha", "Alpha (ann.)", "pct"), ("alpha_t", "Alpha t-stat", "num"),
        ("information_ratio", "Information ratio", "num"), ("num_trades", "Trades", "int"),
        ("win_rate", "Win rate", "pct"), ("profit_factor", "Profit factor", "num"),
        ("avg_holding_days", "Avg holding (days)", "int")]


def _need():
    m, eq = metrics(), backtest_equity()
    if m is None or eq.empty:
        missing("No backtest results found. Run `python main.py report`.")
        return None, None
    return m, eq


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return {"pct": lambda x: f"{x:.2%}", "num": lambda x: f"{x:.2f}", "int": lambda x: f"{x:,.0f}"}[kind](v)


# ------------------------------------------------------------------ backtest
def backtest_page() -> None:
    st.title("📈 Backtest Results")
    paper_banner()
    m, eq = _need()
    if m is None:
        return
    st.caption("Train 2016–2019 (design) · Validation 2020–2021 (robustness) · Test 2022→ (run once with frozen "
               "settings). Signals at the close, fills at the next open, 7 bps one-way costs.")
    rows = {}
    for p in ["train", "validation", "test", "full"]:
        rows[f"Strategy · {p}"] = m["periods"][p]["strategy"]
        rows[f"SPY · {p}"] = m["periods"][p]["spy"]
    tab = pd.DataFrame({lbl: [_fmt(r.get(k), kind) for k, _, kind in KEYS] for lbl, r in rows.items()},
                       index=[n for _, n, _ in KEYS])
    st.dataframe(tab, height=600)

    e, s = eq["equity"], eq["spy"]
    a, b = st.columns(2)
    dd, sdd = e / e.cummax() - 1, s / s.cummax() - 1
    fig = go.Figure()
    fig.add_scatter(x=dd.index, y=dd, fill="tozeroy", name="Strategy", line=dict(color=BLUE))
    fig.add_scatter(x=sdd.index, y=sdd, name="SPY", line=dict(color=GREY, width=1))
    fig.update_layout(yaxis_tickformat=".0%")
    a.plotly_chart(style_fig(fig, 360, "Drawdown from previous peak"))
    r, br = e.pct_change(), s.pct_change()
    excess = (1 + r).rolling(252).apply(np.prod, raw=True) - (1 + br).rolling(252).apply(np.prod, raw=True)
    fig = go.Figure()
    fig.add_scatter(x=excess.index, y=excess, fill="tozeroy", name="12m return − SPY", line=dict(color=PURPLE))
    fig.update_layout(yaxis_tickformat=".0%")
    b.plotly_chart(style_fig(fig, 360, "Rolling 12-month return minus SPY"))

    mt = M.monthly_returns_table(e)
    z = mt.drop(columns="Year")
    fig = px.imshow(z, text_auto=".1%", color_continuous_scale="RdYlGn", zmin=-0.08, zmax=0.08, aspect="auto")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style_fig(fig, 460, "Monthly returns"))

    st.subheader("Where did returns come from?")
    dec = m.get("decomposition", {})
    if dec:
        st.markdown(
            f"With an average beta of **{dec['beta']:.2f}**, a passive mix of {dec['beta']:.2f}× SPY + "
            f"{1 - dec['beta']:.2f}× T-bills compounded at **{pct(dec['beta_matched_cagr'], 2)}**/yr vs. the strategy's "
            f"**{pct(dec['strategy_cagr'], 2)}**. Stock selection + timing: **{dec['selection_and_timing_effect']:+.2%}/yr**.")
    a, b = st.columns(2)
    pnl = load_csv(str(BT / "pnl_by_stock.csv"))
    if len(pnl):
        top = pd.concat([pnl.head(10), pnl.tail(10)])
        fig = px.bar(top, x="pnl", y="ticker", orientation="h", color=top["pnl"] > 0,
                     color_discrete_map={True: GREEN, False: RED})
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, xaxis_tickprefix="$")
        a.plotly_chart(style_fig(fig, 520, "Best & worst contributors ($ P&L)"))
    sec = load_csv(str(BT / "pnl_by_sector.csv"))
    if len(sec):
        fig = px.bar(sec, x="pnl", y="sector", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickprefix="$")
        b.plotly_chart(style_fig(fig, 520, "P&L by sector"))

    st.subheader("Trade ledger")
    trades = load_csv(str(BT / "trades.csv"), parse_dates=["date"])
    if len(trades):
        tk = st.multiselect("Filter tickers", sorted(trades["ticker"].unique()))
        t = trades[trades["ticker"].isin(tk)] if tk else trades
        st.dataframe(t.sort_values("date", ascending=False), hide_index=True, height=380)


# ------------------------------------------------------------------ risk
def risk_page() -> None:
    cfg = get_config()
    st.title("🛡️ Risk")
    paper_banner()
    m, eq = _need()
    if m is None:
        return
    e, s = eq["equity"], eq["spy"]
    r, br = e.pct_change().dropna(), s.pct_change().dropna()
    c = st.columns(4)
    c[0].metric("VaR 95% (1 day)", pct(M.value_at_risk(r), 2), f"SPY {pct(M.value_at_risk(br), 2)}", delta_color="off", border=True)
    c[1].metric("Expected shortfall 95%", pct(M.expected_shortfall(r), 2), f"SPY {pct(M.expected_shortfall(br), 2)}",
                delta_color="off", border=True)
    c[2].metric("Average invested", pct(eq["gross_exposure"].mean(), 0), border=True)
    c[3].metric("Days in drawdown-control mode", f"{int(eq['drawdown_mode'].sum()):,}", border=True)

    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=r.index, y=r.rolling(252).std() * np.sqrt(252), name="Strategy", line=dict(color=BLUE))
    fig.add_scatter(x=br.index, y=br.rolling(252).std() * np.sqrt(252), name="SPY", line=dict(color=GREY))
    if cfg.portfolio.target_volatility:
        fig.add_hline(y=cfg.portfolio.target_volatility, line_dash="dash", annotation_text="vol target")
    fig.update_layout(yaxis_tickformat=".0%")
    a.plotly_chart(style_fig(fig, 340, "Rolling 1-year volatility"))
    fig = px.area(eq, y="gross_exposure", labels={"gross_exposure": "invested", "date": ""})
    fig.update_layout(yaxis_tickformat=".0%")
    b.plotly_chart(style_fig(fig, 340, "Share of capital invested (rest is cash)"))

    pos = load_csv(str(BT / "positions.csv"), index_col=0, parse_dates=True)
    if len(pos):
        smap = sector_map()
        w = pos.div(eq["equity"], axis=0)
        sec = w.T.groupby(lambda t: smap.get(t, "Unknown")).sum().T.resample("ME").last()
        fig = px.area(sec, labels={"value": "weight", "variable": "sector", "index": ""})
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 400, "Sector allocation over time (gap to 100% = cash)"))

    a, b = st.columns(2)
    hs = load_csv(str(BT / "stress_historical.csv"))
    if len(hs):
        with a:
            st.subheader("Historical stress episodes")
            st.dataframe(hs[["episode", "start", "end", "strategy_return", "spy_return", "strategy_max_dd"]],
                         hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format="percent")
                                        for c in ["strategy_return", "spy_return", "strategy_max_dd"]})
    hy = load_csv(str(BT / "stress_hypothetical.csv"))
    if len(hy):
        with b:
            st.subheader("Hypothetical shocks (end-of-backtest portfolio)")
            st.dataframe(hy, hide_index=True,
                         column_config={"portfolio_return": st.column_config.NumberColumn("return", format="percent"),
                                        "dollar_pnl": st.column_config.NumberColumn("$ P&L", format="$%.0f")})
    st.subheader("Risk rules")
    p, rk = cfg.portfolio, cfg.risk
    st.markdown(f"""
| Layer | Rule |
|---|---|
| Position size | ≤ {p.max_position_weight:.0%} of equity, and weight × volatility ≤ {p.max_position_risk:.0%} |
| Sector | ≤ {p.max_sector_weight:.0%} of equity |
| Cash | ≥ {p.min_cash_weight:.0%} |
| Holdings | ≤ {p.max_positions} stocks, {p.sizing_method} sizing |
| Volatility target | scale down if estimated volatility > {p.target_volatility:.0%} |
| Trailing stop | exit if price < peak × (1 − {rk.stop_loss_vol_multiple:g} × monthly volatility) |
| Drawdown control | drawdown > {rk.drawdown_limit:.0%} → ≤ {rk.drawdown_gross_cap:.0%} invested until < {rk.drawdown_reset:.0%} |
| Market regime | SPY below 200-day average → ≤ {rk.regime_gross_cap:.0%} invested |
""")


# ------------------------------------------------------------------ robustness
def robustness_page() -> None:
    st.title("🔬 Robustness")
    paper_banner()
    m, eq = _need()
    if m is None:
        return
    st.markdown("Is the result real or a lucky parameter choice? Each parameter is changed **one at a time**. "
                "A robust strategy changes gradually; a fragile one only works at one exact setting.")
    sens = load_csv(str(BT / "sensitivity.csv"))
    full, spy = m["periods"]["full"]["strategy"], m["periods"]["full"]["spy"]
    pts = pd.DataFrame([{"label": "Strategy (base)", "volatility": full["volatility"], "cagr": full["cagr"], "kind": "base"},
                        {"label": "SPY", "volatility": spy["volatility"], "cagr": spy["cagr"], "kind": "SPY"}])
    if len(sens):
        v = sens[~sens["base"]].assign(label=lambda d: d["parameter"] + ": " + d["value"], kind="variant")
        pts = pd.concat([pts, v[["label", "volatility", "cagr", "kind"]]])
    fig = px.scatter(pts, x="volatility", y="cagr", color="kind", hover_name="label",
                     color_discrete_map={"base": BLUE, "SPY": GREY, "variant": PURPLE})
    fig.update_traces(marker=dict(size=11))
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 420, "Risk vs. return: base case, SPY and every variant"))
    if len(sens):
        st.dataframe(sens.drop(columns=["base"]), hide_index=True, height=520,
                     column_config={**{c: st.column_config.NumberColumn(format="percent")
                                       for c in ["cagr", "volatility", "max_drawdown", "alpha", "avg_gross", "test_cagr"]},
                                    **{c: st.column_config.NumberColumn(format="%.2f")
                                       for c in ["sharpe", "sortino", "beta", "test_sharpe"]},
                                    "costs_paid": st.column_config.NumberColumn(format="$%.0f")})
    a, b = st.columns(2)
    ic = load_csv(str(BT / "factor_ic_summary.csv"))
    if len(ic):
        fig = px.bar(ic, x="factor", y="mean_ic", color="period", barmode="group")
        a.plotly_chart(style_fig(fig, 380, "Factor information coefficient (does the factor rank stocks correctly?)"))
    wf = load_csv(str(BT / "walk_forward_equity.csv"), index_col=0, parse_dates=True)
    if len(wf):
        fig = go.Figure()
        fig.add_scatter(x=wf.index, y=wf["walk_forward"], name="Walk-forward (out-of-sample)", line=dict(color=BLUE))
        fig.add_scatter(x=wf.index, y=wf["SPY"], name="SPY", line=dict(color=GREY))
        fig.update_layout(yaxis_tickprefix="$")
        b.plotly_chart(style_fig(fig, 380, "Walk-forward: config chosen each year from prior years only"))
    ch = load_csv(str(BT / "walk_forward_choices.csv"))
    if len(ch):
        st.dataframe(ch[["year", "chosen", "in_sample_sharpe", "out_of_sample_return"]], hide_index=True,
                     column_config={"in_sample_sharpe": st.column_config.NumberColumn(format="%.2f"),
                                    "out_of_sample_return": st.column_config.NumberColumn(format="percent")})
