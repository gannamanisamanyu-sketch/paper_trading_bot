"""Backtest Lab: change the strategy's settings and re-run the full backtest live."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import BLUE, GREY, PURPLE, backtest_equity, get_config, money, paper_banner, pct, style_fig


@st.cache_resource(ttl=6 * 3600, show_spinner="Loading market data (first load downloads ~10 years of prices, ~30-60 s)…")
def get_context():
    from backtesting.pipeline import load_context

    return load_context(get_config())


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=64)
def run_variant(overrides: tuple, cost_bps: float):
    from backtesting.pipeline import period_bounds, run_backtest, summarize
    from portfolio.costs import CostModel

    cfg = get_config()
    # overrides arrive as a hashable tuple (for caching); nested dicts were stored as tuples of pairs
    vcfg = cfg.override({k: dict(v) if isinstance(v, tuple) else v for k, v in overrides})
    ctx = get_context()
    cost = CostModel.from_config(cfg.costs).with_total_bps(cost_bps)
    res = run_backtest(vcfg, ctx, cost_model=cost, record_scores=False)
    bounds = period_bounds(vcfg, ctx.md.dates[-1])
    summ = {p: summarize(res, ctx, vcfg, *bounds[p]) for p in ("full", "test")}
    return res.equity_curve, res.benchmark, res.equity["gross_exposure"], summ, len(res.fills)


def page() -> None:
    cfg = get_config()
    st.title("🧪 Backtest Lab")
    paper_banner()
    st.markdown("Change the rules and re-run the **entire 2016 → today backtest** (takes ~5–10 seconds). "
                "Compare the result with the frozen base case and with SPY.")
    st.warning("**Overfitting warning.** Picking whichever settings look best on this same history is exactly how "
               "backtests get overfit. The base case was frozen before the out-of-sample test. Use this page to "
               "build intuition, not to find a 'better' strategy.", icon="⚠️")

    s, p, r = cfg.strategy, cfg.portfolio, cfg.risk
    with st.form("lab"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Portfolio**")
            max_pos = st.slider("Max number of stocks", 5, 30, p.max_positions)
            sizing = st.radio("Position sizing", ["volatility", "equal"],
                              index=0 if p.sizing_method == "volatility" else 1, horizontal=True,
                              format_func=lambda x: "Inverse volatility" if x == "volatility" else "Equal weight")
            max_w = st.slider("Max weight per stock", 0.05, 0.25, p.max_position_weight, 0.01, format="%.2f")
            max_sec = st.slider("Max weight per sector", 0.15, 1.0, p.max_sector_weight, 0.05, format="%.2f")
            freq = st.selectbox("Rebalance", ["weekly", "biweekly", "monthly", "quarterly"],
                                index=["weekly", "biweekly", "monthly", "quarterly"].index(s.rebalance_frequency))
        with b:
            st.markdown("**Risk management**")
            vt = st.select_slider("Volatility target", ["none", "10%", "12%", "16%", "20%", "25%"], value="16%")
            stop = st.select_slider("Trailing stop (× monthly volatility)", ["off", "1.5", "2", "3", "4"], value="2")
            regime = st.toggle("Market-regime filter (SPY < 200-day avg → max 50% invested)", r.market_regime_filter)
            ddc = st.toggle("Drawdown control (−15% → max 50% invested)", r.drawdown_limit < 1)
            costs = st.slider("Trading cost, one-way (bps)", 0, 50,
                              int(round(cfg.costs.half_spread_bps + cfg.costs.slippage_bps)))
        with c:
            st.markdown("**Factor weights**")
            mom_lb = st.select_slider("Momentum lookback (trading days)", [63, 126, 189, 252], value=s.momentum_lookback)
            w = s.weights
            wm = st.slider("Momentum", 0.0, 1.0, w.momentum, 0.05)
            wt = st.slider("Trend", 0.0, 1.0, w.trend, 0.05)
            ws = st.slider("Sector", 0.0, 1.0, w.sector, 0.05)
            wq = st.slider("Quality", 0.0, 1.0, w.quality, 0.05)
            wv = st.slider("Value", 0.0, 1.0, w.value, 0.05)
            wg = st.slider("Growth", 0.0, 1.0, w.growth, 0.05)
            wr = st.slider("Risk penalty", 0.0, 1.0, w.risk_penalty, 0.05)
        submitted = st.form_submit_button("▶ Run backtest", type="primary")

    if not submitted:
        st.caption("Set the parameters and press **Run backtest**.")
        return

    overrides = {
        "portfolio.max_positions": int(max_pos), "portfolio.sizing_method": sizing,
        "portfolio.max_position_weight": float(max_w), "portfolio.max_sector_weight": float(max_sec),
        "strategy.rebalance_frequency": freq,
        "portfolio.target_volatility": None if vt == "none" else float(vt.strip("%")) / 100,
        "risk.use_stop_loss": stop != "off", "risk.stop_method": "volatility",
        "risk.stop_loss_vol_multiple": 2.0 if stop == "off" else float(stop),
        "risk.market_regime_filter": bool(regime), "risk.drawdown_limit": 0.15 if ddc else 1.0,
        "strategy.momentum_lookback": int(mom_lb),
        "strategy.weights": {"momentum": wm, "trend": wt, "sector": ws, "quality": wq, "value": wv,
                             "growth": wg, "risk_penalty": wr},
    }
    key = tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v) for k, v in overrides.items()))
    try:
        get_context()
    except Exception as exc:  # network / data-source failure on the server
        st.error(f"Could not load market data on this server ({exc}). Try again in a minute.")
        return
    t0 = time.time()
    with st.spinner("Running backtest…"):
        try:
            eq, bench, gross, summ, n_fills = run_variant(key, float(costs))
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            return
    st.success(f"Done in {time.time() - t0:.1f} s · {n_fills:,} simulated trades")

    full, spy, test = summ["full"]["strategy"], summ["full"]["spy"], summ["test"]["strategy"]
    base = backtest_equity()
    c = st.columns(5)
    c[0].metric("Final value", money(full["final_value"]), f"SPY {money(spy['final_value'])}", delta_color="off", border=True)
    c[1].metric("CAGR", pct(full["cagr"], 2), f"SPY {pct(spy['cagr'], 2)}", delta_color="off", border=True)
    c[2].metric("Sharpe", f"{full['sharpe']:.2f}", f"SPY {spy['sharpe']:.2f}", delta_color="off", border=True)
    c[3].metric("Max drawdown", pct(full["max_drawdown"]), f"SPY {pct(spy['max_drawdown'])}", delta_color="off", border=True)
    c[4].metric("Out-of-sample Sharpe", f"{test['sharpe']:.2f}", "2022 → today", delta_color="off", border=True)

    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq, name="Your settings", line=dict(color=PURPLE, width=2.4))
    if len(base):
        fig.add_scatter(x=base.index, y=base["equity"], name="Base case (frozen)", line=dict(color=BLUE, width=1.5))
    fig.add_scatter(x=bench.index, y=bench, name="SPY", line=dict(color=GREY))
    fig.update_layout(yaxis_tickprefix="$")
    st.plotly_chart(style_fig(fig, 420, "Portfolio value"))
    a, b = st.columns(2)
    dd = eq / eq.cummax() - 1
    fig = go.Figure()
    fig.add_scatter(x=dd.index, y=dd, fill="tozeroy", name="Your settings", line=dict(color=PURPLE))
    fig.update_layout(yaxis_tickformat=".0%")
    a.plotly_chart(style_fig(fig, 320, "Drawdown"))
    fig = go.Figure()
    fig.add_scatter(x=gross.index, y=gross, fill="tozeroy", name="Invested", line=dict(color=PURPLE))
    fig.update_layout(yaxis_tickformat=".0%")
    b.plotly_chart(style_fig(fig, 320, "Share of capital invested"))
    rows = ["cagr", "volatility", "sharpe", "sortino", "max_drawdown", "beta", "alpha", "alpha_t", "var_95",
            "es_95", "win_rate", "profit_factor", "avg_holding_days", "total_costs"]
    tab = pd.DataFrame({"Your settings": [full.get(k) for k in rows], "SPY": [spy.get(k) for k in rows]}, index=rows)
    st.dataframe(tab.T.style.format(lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v))
                                    else f"{v:,.3f}"))
