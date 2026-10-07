"""Run the complete analysis and write the performance report.

Outputs
-------
results/                     machine-readable CSV/JSON (read by the dashboard)
reports/performance_report.md   GitHub-readable report with charts
reports/performance_report.html self-contained HTML version (charts embedded)
reports/figures/*.png        charts
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtesting.pipeline import DataContext, period_bounds, run_backtest, summarize
from backtesting.sensitivity import run_sensitivity
from backtesting.walk_forward import run_walk_forward
from reporting import attribution as attr
from reporting import charts
from risk.metrics import monthly_returns_table, performance_summary
from risk.stress import historical_stress, hypothetical_stress

logger = logging.getLogger(__name__)

PCT = {"total_return", "cagr", "volatility", "max_drawdown", "var_95", "es_95", "alpha",
       "win_rate", "avg_win_pct", "avg_loss_pct", "best_day", "worst_day", "avg_gross_exposure",
       "information_ratio_unused"}
MONEY = {"initial_capital", "final_value", "avg_win", "avg_loss", "total_costs"}
LABELS = {
    "initial_capital": "Initial capital", "final_value": "Final portfolio value",
    "total_return": "Total return", "cagr": "CAGR", "volatility": "Annualized volatility",
    "sharpe": "Sharpe ratio", "sortino": "Sortino ratio", "max_drawdown": "Maximum drawdown",
    "calmar": "Calmar ratio (CAGR / |MaxDD|)", "var_95": "VaR 95% (1-day, historical)",
    "es_95": "Expected Shortfall 95% (1-day)", "beta": "Beta vs. SPY",
    "alpha": "Alpha vs. SPY (CAPM, annualized)", "alpha_t": "Alpha t-statistic",
    "information_ratio": "Information ratio", "correlation": "Correlation with SPY",
    "num_trades": "Number of trades (fills)", "round_trips": "Closed positions (round trips)",
    "win_rate": "Win rate (closed positions)", "profit_factor": "Profit factor",
    "avg_win": "Average winning position ($)", "avg_loss": "Average losing position ($)",
    "avg_win_pct": "Average winning position (%)", "avg_loss_pct": "Average losing position (%)",
    "avg_holding_days": "Average holding period (days)", "total_costs": "Total trading costs paid",
    "avg_gross_exposure": "Average gross exposure",
}
REPORT_ORDER = ["initial_capital", "final_value", "total_return", "cagr", "volatility", "sharpe",
                "sortino", "max_drawdown", "calmar", "var_95", "es_95", "beta", "alpha", "alpha_t",
                "information_ratio", "num_trades", "round_trips", "win_rate", "profit_factor",
                "avg_win", "avg_loss", "avg_win_pct", "avg_loss_pct", "avg_holding_days",
                "total_costs", "avg_gross_exposure"]


def fmt(key: str, v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if key in PCT:
        return f"{v:.2%}"
    if key in MONEY:
        return f"${v:,.0f}"
    if key in {"num_trades", "round_trips"}:
        return f"{int(v):,}"
    if key == "avg_holding_days":
        return f"{v:.0f}"
    return f"{v:.2f}"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


# ------------------------------------------------------------------ analysis
def build_analysis(cfg, ctx: DataContext, walk_forward: bool = True,
                   sensitivity: bool = True) -> Dict[str, object]:
    md = ctx.md
    results_dir = cfg.path(cfg.output.results_dir) / "backtest"
    fig_dir = cfg.path(cfg.output.reports_dir) / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    bounds = period_bounds(cfg, md.dates[-1])

    logger.info("Running base backtest %s -> %s", *bounds["full"])
    res = run_backtest(cfg, ctx)
    eq, bench = res.equity_curve, res.benchmark

    periods = {name: summarize(res, ctx, cfg, s, e) for name, (s, e) in bounds.items()}
    idx = md.index_level.reindex(eq.index).ffill()
    idx_eq = cfg.portfolio.initial_capital * idx / idx.iloc[0]
    sp500_price = performance_summary(idx_eq, bench, md.risk_free_daily)

    # ---- analysis
    ic = attr.factor_ic(res.scores, md.adj_close)
    ic_tab = attr.ic_summary(ic, bounds)
    last_mv = res.positions.iloc[-1]
    stock_pnl = attr.contribution_by_stock(res.fills, last_mv, ctx.universe.name_map,
                                           ctx.universe.sector_map)
    sector_pnl = attr.contribution_by_sector(stock_pnl)
    beta = periods["full"]["strategy"]["beta"]
    decomposition = attr.beta_matched_comparison(eq, bench, md.risk_free_daily, beta)
    hist_stress = historical_stress(eq, bench)
    weights_now = (last_mv[last_mv > 0] / eq.iloc[-1])
    from strategy.factors import FeatureSet  # noqa: F401  (features cached in ctx)
    betas_now = ctx.features(cfg.strategy).beta.iloc[-1]
    hypo_stress = hypothetical_stress(weights_now, betas_now, ctx.universe.sector_map, eq.iloc[-1])
    monthly = monthly_returns_table(eq)
    worst = attr.worst_periods(eq)

    wf = run_walk_forward(cfg, ctx) if walk_forward else None
    sens = run_sensitivity(cfg, ctx) if sensitivity else None

    # ---- save machine-readable results
    res.equity.assign(spy=bench).to_csv(results_dir / "equity.csv")
    res.fills.to_csv(results_dir / "trades.csv", index=False)
    res.signals.to_csv(results_dir / "signals.csv", index=False)
    res.positions.to_csv(results_dir / "positions.csv")
    res.rebalances.to_csv(results_dir / "rebalances.csv", index=False)
    ic.to_csv(results_dir / "factor_ic.csv")
    ic_tab.to_csv(results_dir / "factor_ic_summary.csv", index=False)
    stock_pnl.to_csv(results_dir / "pnl_by_stock.csv")
    sector_pnl.to_csv(results_dir / "pnl_by_sector.csv")
    hist_stress.to_csv(results_dir / "stress_historical.csv", index=False)
    hypo_stress.to_csv(results_dir / "stress_hypothetical.csv", index=False)
    monthly.to_csv(results_dir / "monthly_returns.csv")
    md.quality_flags.to_csv(results_dir / "data_quality_flags.csv", index=False)
    if sens is not None:
        sens.to_csv(results_dir / "sensitivity.csv", index=False)
    if wf is not None:
        wf["choices"].to_csv(results_dir / "walk_forward_choices.csv", index=False)
        pd.concat([wf["oos_equity"], wf["bench_equity"]], axis=1).to_csv(results_dir / "walk_forward_equity.csv")
    metrics = {"generated": datetime.now().isoformat(timespec="seconds"),
               "periods": {k: v for k, v in periods.items()}, "sp500_price_index": sp500_price,
               "bounds": bounds, "decomposition": decomposition,
               "walk_forward": ({"summary": wf["summary"], "spy": wf["spy_summary"]} if wf else None),
               "universe": {"active": len(ctx.universe.tradable),
                            "excluded": ctx.universe.delisted[["ticker", "name", "status", "note"]].to_dict("records")},
               "config": cfg.to_dict()}
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))

    # ---- charts
    figs = {
        "equity": charts.equity_vs_benchmark(eq, bench, fig_dir / "equity_vs_spy.png", bounds),
        "drawdown": charts.drawdown_chart(eq, bench, fig_dir / "drawdown.png"),
        "monthly": charts.monthly_heatmap(monthly, fig_dir / "monthly_returns.png"),
        "sectors": charts.sector_allocation(res.positions, eq, ctx.universe.sector_map,
                                            fig_dir / "sector_allocation.png"),
        "positions": charts.position_allocation(weights_now, fig_dir / "current_positions.png",
                                                f"Positions at end of backtest ({eq.index[-1].date()})"),
        "ic": charts.factor_ic_chart(ic_tab, fig_dir / "factor_ic.png"),
        "rolling": charts.rolling_metrics(eq, bench, fig_dir / "rolling_beta_excess.png"),
    }
    pts = pd.DataFrame({"Strategy": periods["full"]["strategy"], "SPY": periods["full"]["spy"]}).T[["volatility", "cagr"]]
    if sens is not None:
        variants = sens[~sens["base"]].set_index(sens[~sens["base"]]["parameter"] + ": " + sens[~sens["base"]]["value"])[["volatility", "cagr"]]
        pts = pd.concat([pts, variants])
        figs["costs"] = charts.cost_sensitivity_chart(sens, fig_dir / "cost_sensitivity.png")
    figs["risk_return"] = charts.risk_return_scatter(pts, fig_dir / "risk_return.png")
    if wf is not None:
        figs["walk_forward"] = charts.equity_vs_benchmark(wf["oos_equity"], wf["bench_equity"],
                                                          fig_dir / "walk_forward.png")

    return {"cfg": cfg, "result": res, "periods": periods, "sp500_price": sp500_price,
            "ic": ic, "ic_table": ic_tab, "stock_pnl": stock_pnl, "sector_pnl": sector_pnl,
            "decomposition": decomposition, "hist_stress": hist_stress, "hypo_stress": hypo_stress,
            "monthly": monthly, "worst": worst, "walk_forward": wf, "sensitivity": sens,
            "figures": figs, "bounds": bounds, "ctx": ctx, "weights_now": weights_now}


# ------------------------------------------------------------------ markdown
def _metrics_table(periods: Dict, sp500_price: Dict) -> str:
    cols = [("full", "strategy", "Strategy (full)"), ("full", "spy", "SPY (full)"),
            ("train", "strategy", "Strategy train"), ("validation", "strategy", "Strategy valid."),
            ("test", "strategy", "Strategy TEST"), ("test", "spy", "SPY test")]
    head = "| Metric | " + " | ".join(c[2] for c in cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    lines = []
    for k in REPORT_ORDER:
        vals = []
        for p, who, _ in cols:
            v = periods[p][who].get(k)
            vals.append(fmt(k, v) if v is not None else "")
        lines.append(f"| {LABELS.get(k, k)} | " + " | ".join(vals) + " |")
    return head + "\n".join(lines)


def _df_md(df: pd.DataFrame, pct_cols=(), money_cols=(), num_cols=(), index=True) -> str:
    d = df.copy()
    for c in pct_cols:
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")
    for c in money_cols:
        if c in d:
            d[c] = d[c].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "n/a")
    for c in num_cols:
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    return d.to_markdown(index=index)


def write_markdown(an: Dict, path: Path) -> Path:
    cfg, P, b = an["cfg"], an["periods"], an["bounds"]
    full, spy = P["full"]["strategy"], P["full"]["spy"]
    test, spy_test = P["test"]["strategy"], P["test"]["spy"]
    dec = an["decomposition"]
    rel = lambda p: f"figures/{Path(p).name}"
    F = an["figures"]
    beat_sharpe = full["sharpe"] > spy["sharpe"]
    out = []
    out.append("# Performance Report: Algorithmic U.S. Equity Paper Trading & Risk Management System\n")
    out.append(f"*Generated {datetime.now():%Y-%m-%d %H:%M} from `config/config.yaml`. "
               "Simulated results only. Educational project, not investment advice.*\n")
    out.append("## Headline\n")
    out.append(
        f"- **Full period ({b['full'][0]} → {b['full'][1]})**: ${full['initial_capital']:,.0f} → "
        f"**${full['final_value']:,.0f}** ({full['total_return']:.1%} total, {full['cagr']:.2%} CAGR) vs. SPY "
        f"${spy['final_value']:,.0f} ({spy['cagr']:.2%} CAGR).\n"
        f"- Risk-adjusted: Sharpe **{full['sharpe']:.2f}** vs. SPY {spy['sharpe']:.2f}; max drawdown "
        f"**{full['max_drawdown']:.1%}** vs. {spy['max_drawdown']:.1%}; beta {full['beta']:.2f}.\n"
        f"- Alpha vs. SPY: **{full['alpha']:.2%} per year** (t-stat {full['alpha_t']:.2f}; "
        f"{'statistically significant' if abs(full['alpha_t']) > 2 else 'not statistically different from zero'}).\n"
        f"- **Out-of-sample test ({b['test'][0]} → {b['test'][1]})**: CAGR {test['cagr']:.2%} vs. SPY "
        f"{spy_test['cagr']:.2%}; Sharpe {test['sharpe']:.2f} vs. {spy_test['sharpe']:.2f}.\n"
        f"- Verdict: the strategy **{'did' if beat_sharpe and full['alpha'] > 0 else 'did NOT'} generate excess "
        f"risk-adjusted returns** relative to SPY over the full sample.\n")
    out.append(f"![Equity]({rel(F['equity'])})\n")
    out.append("## Performance metrics\n")
    out.append(_metrics_table(P, an["sp500_price"]) + "\n")
    sp = an["sp500_price"]
    out.append(f"\nS&P 500 *price* index (^GSPC, excludes dividends) over the full period: "
               f"CAGR {sp['cagr']:.2%}, Sharpe {sp['sharpe']:.2f}, max drawdown {sp['max_drawdown']:.1%}. "
               "SPY (total return) is the fairer benchmark because the strategy's returns include dividends.\n")
    out.append(f"![Drawdown]({rel(F['drawdown'])})\n")
    out.append(f"![Rolling]({rel(F['rolling'])})\n")
    out.append(f"![Monthly]({rel(F['monthly'])})\n")

    out.append("## Where did returns come from?\n")
    out.append(
        f"**Beta-matched comparison.** With an average beta of {dec['beta']:.2f}, a passive portfolio of "
        f"{dec['beta']:.2f}× SPY + {1 - dec['beta']:.2f}× T-bills would have compounded at "
        f"{dec['beta_matched_cagr']:.2%}/yr. The strategy compounded at {dec['strategy_cagr']:.2%}/yr, so stock "
        f"selection + timing contributed **{dec['selection_and_timing_effect']:+.2%}/yr** beyond its market exposure.\n")
    out.append("**Factor information coefficients** (monthly Spearman rank correlation of factor score vs. "
               "next-month return; |t| > 2 ≈ statistically meaningful):\n")
    ict = an["ic_table"].copy()
    ict = ict.pivot(index="factor", columns="period", values="mean_ic").join(
        an["ic_table"].pivot(index="factor", columns="period", values="ic_t_stat"), rsuffix="_t")
    order = [c for c in ["train", "train_t", "validation", "validation_t", "test", "test_t", "full", "full_t"] if c in ict]
    out.append(ict[order].round(3).to_markdown() + "\n")
    out.append(f"![IC]({rel(F['ic'])})\n")
    sp_ = an["stock_pnl"]
    out.append("**Top 10 contributors ($ P&L, realized + unrealized):**\n")
    out.append(_df_md(sp_.head(10)[["name", "sector", "pnl", "trades"]], money_cols=["pnl"]) + "\n")
    out.append("**Bottom 10 contributors:**\n")
    out.append(_df_md(sp_.tail(10).iloc[::-1][["name", "sector", "pnl", "trades"]], money_cols=["pnl"]) + "\n")
    out.append("**P&L by sector:**\n")
    out.append(_df_md(an["sector_pnl"], money_cols=["pnl"]) + "\n")
    out.append(f"![Sectors]({rel(F['sectors'])})\n")
    out.append("**Worst months:**\n")
    out.append(_df_md(an["worst"].rename(index=lambda d: d.strftime("%Y-%m")), pct_cols=["monthly_return"]) + "\n")

    out.append("## Stress testing\n")
    out.append("**Historical episodes** (simulated strategy vs. SPY):\n")
    out.append(_df_md(an["hist_stress"], pct_cols=["strategy_return", "spy_return", "strategy_max_dd", "difference"], index=False) + "\n")
    out.append("**Hypothetical instantaneous shocks to the current portfolio:**\n")
    out.append(_df_md(an["hypo_stress"], pct_cols=["portfolio_return"], money_cols=["dollar_pnl"], index=False) + "\n")

    wf = an["walk_forward"]
    if wf is not None:
        out.append("## Walk-forward (expanding window) out-of-sample test\n")
        out.append("Each January the configuration with the best Sharpe ratio over *all prior years* is selected "
                   "from 4 pre-declared candidates and traded for that year.\n")
        ch = wf["choices"][["year", "chosen", "in_sample_sharpe", "out_of_sample_return"]]
        out.append(_df_md(ch, pct_cols=["out_of_sample_return"], num_cols=["in_sample_sharpe"], index=False) + "\n")
        s, bs = wf["summary"], wf["spy_summary"]
        out.append(f"\nStitched walk-forward result: CAGR **{s['cagr']:.2%}**, Sharpe **{s['sharpe']:.2f}**, max drawdown "
                   f"{s['max_drawdown']:.1%} vs. SPY CAGR {bs['cagr']:.2%}, Sharpe {bs['sharpe']:.2f}.\n")
        out.append(f"![WF]({rel(F['walk_forward'])})\n")

    sens = an["sensitivity"]
    if sens is not None:
        out.append("## Sensitivity analysis (one parameter at a time; ★ = base case)\n")
        t = sens.copy()
        t["value"] = np.where(t["base"], "★ " + t["value"], t["value"])
        t = t[["parameter", "value", "cagr", "volatility", "sharpe", "max_drawdown", "alpha", "trades",
               "avg_gross", "test_sharpe"]]
        out.append(_df_md(t, pct_cols=["cagr", "volatility", "max_drawdown", "alpha", "avg_gross"],
                          num_cols=["sharpe", "test_sharpe"], index=False) + "\n")
        out.append(f"![Costs]({rel(F['costs'])})\n")
        out.append(f"![RiskReturn]({rel(F['risk_return'])})\n")

    out.append("## Current portfolio (end of backtest)\n")
    out.append(f"![Positions]({rel(F['positions'])})\n")
    out.append("## Methodology notes & limitations\n")
    excl = an["ctx"].universe.delisted
    out.append(
        f"- Universe: {len(an['ctx'].universe.tradable)} large caps fixed as of end-2015. {len(excl)} constituents "
        f"({', '.join(excl['ticker'])}) were later acquired/delisted or lack free history and are excluded, a residual "
        "**survivorship bias** that likely flatters results.\n"
        "- Signals use the close of day t; trades fill at the open of t+1 with spread + slippage.\n"
        "- Fundamentals: SEC EDGAR 10-K data, used only after the filing date (point-in-time).\n"
        "- Sector labels are current GICS sectors (a small, documented look-ahead).\n"
        "- Adjusted prices: share counts are in adjusted units; dividends are reinvested via the adjusted series.\n"
        "- See `RESEARCH_LOG.md` for every design change made after looking at results.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    return path


def write_html(md_path: Path, html_path: Path) -> Path:
    """Very small markdown->HTML conversion with embedded images (self-contained file)."""
    import re

    text = md_path.read_text()
    fig_dir = md_path.parent

    def img(m):
        p = fig_dir / m.group(2)
        if not p.exists():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img alt="{m.group(1)}" src="data:image/png;base64,{b64}"/>'

    html_lines, in_table, rows = [], False, []

    def flush_table():
        nonlocal rows
        if not rows:
            return ""
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        head, body = cells[0], [r for r in cells[2:]]
        h = "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>"
        h += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
        rows = []
        return h + "</tbody></table>"

    def inline(s: str) -> str:
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    for line in text.splitlines():
        if line.startswith("|"):
            rows.append(line)
            continue
        if rows:
            html_lines.append(flush_table())
        if line.startswith("!["):
            html_lines.append(re.sub(r"!\[(.*?)\]\((.*?)\)", img, line))
        elif line.startswith("# "):
            html_lines.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip():
            html_lines.append(f"<p>{inline(line)}</p>")
    if rows:
        html_lines.append(flush_table())
    css = """body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#111827;line-height:1.5}
h1{font-size:24px}h2{font-size:18px;margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}
table{border-collapse:collapse;font-size:12.5px;margin:10px 0;display:block;overflow-x:auto}
th,td{border:1px solid #e5e7eb;padding:4px 8px;text-align:right}th{background:#f9fafb}td:first-child,th:first-child{text-align:left}
img{max-width:100%;margin:10px 0}code{background:#f3f4f6;padding:1px 4px;border-radius:3px}"""
    html = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Strategy Performance Report</title><style>{css}</style></head><body>"
            + "\n".join(html_lines) + "</body></html>")
    html_path.write_text(html)
    return html_path


def generate_report(cfg, ctx: DataContext, walk_forward: bool = True, sensitivity: bool = True) -> Dict:
    an = build_analysis(cfg, ctx, walk_forward, sensitivity)
    rep = cfg.path(cfg.output.reports_dir)
    md_path = write_markdown(an, rep / "performance_report.md")
    write_html(md_path, rep / "performance_report.html")
    logger.info("Report written to %s", md_path)
    return an
