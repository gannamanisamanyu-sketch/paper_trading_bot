"""Command-line entry point.

    python main.py download            # download/refresh price + SEC fundamental data
    python main.py backtest            # run the base backtest, print summary by period
    python main.py report              # full analysis: backtest + walk-forward + sensitivity + report
    python main.py paper init          # create a new $100k paper portfolio
    python main.py paper run           # daily paper-trading cycle (fill orders, mark, signals)
    python main.py paper status        # show positions, P&L, pending orders
    python main.py signals             # today's model portfolio -> results/live/ (for the website)
    python main.py dashboard           # launch the Streamlit dashboard

PAPER TRADING ONLY: nothing in this project connects to a brokerage.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config.settings import PROJECT_ROOT, load_config


def setup_logging(verbose: bool) -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "ptb.log", encoding="utf-8")],
    )
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_download(cfg, args) -> None:
    from backtesting.pipeline import load_context

    ctx = load_context(cfg, refresh=True)
    print(f"Prices: {ctx.md.adj_close.shape[1]} tickers, {ctx.md.dates[0].date()} -> {ctx.md.dates[-1].date()}")
    print("Fundamental coverage:", ctx.fundamentals.coverage.mean().round(2).to_dict())


def cmd_backtest(cfg, args) -> None:
    from backtesting.pipeline import load_context, period_bounds, run_backtest, summarize

    ctx = load_context(cfg, refresh=args.refresh)
    res = run_backtest(cfg, ctx)
    keys = ["final_value", "total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown",
            "beta", "alpha", "num_trades", "win_rate"]
    for name, (s, e) in period_bounds(cfg, ctx.md.dates[-1]).items():
        sm = summarize(res, ctx, cfg, s, e)
        print(f"\n=== {name.upper()} {s} -> {e}")
        for k in keys:
            sv, bv = sm["strategy"].get(k), sm["spy"].get(k)
            print(f"  {k:14s} strategy {sv:>12.4f}   SPY {bv if bv is not None else float('nan'):>12.4f}")


def cmd_report(cfg, args) -> None:
    from backtesting.pipeline import load_context
    from reporting.report import generate_report

    ctx = load_context(cfg, refresh=args.refresh)
    generate_report(cfg, ctx, walk_forward=not args.quick, sensitivity=not args.quick)
    print(f"Report: {cfg.path(cfg.output.reports_dir) / 'performance_report.md'}")


def cmd_paper(cfg, args) -> None:
    from paper_trading.engine import PaperTrader

    trader = PaperTrader(cfg)
    if args.action == "init":
        trader.init(force=args.force)
    elif args.action == "run":
        trader.run(refresh=not args.no_refresh)
    elif args.action == "status":
        trader.print_status()


def cmd_signals(cfg, args) -> None:
    from backtesting.pipeline import load_context
    from paper_trading.live import compute_live_signals

    ctx = load_context(cfg, refresh=args.refresh)
    out = compute_live_signals(cfg, ctx, cfg.path(cfg.output.results_dir) / "live")
    m = out["meta"]
    print(f"Model portfolio as of {m['as_of']}: {m['n_positions']} positions, "
          f"{m['invested']:.0%} invested, market {'risk-on' if m['market_risk_on'] else 'RISK-OFF'}")
    if len(out["targets"]):
        print(out["targets"][["ticker", "name", "sector", "weight", "shares", "price"]].to_string(index=False))


def cmd_dashboard(cfg, args) -> None:
    app = PROJECT_ROOT / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="U.S. equity paper-trading & risk system (PAPER ONLY)")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download/refresh all data")
    p = sub.add_parser("backtest", help="run base backtest and print summary")
    p.add_argument("--refresh", action="store_true", help="re-download data first")
    p = sub.add_parser("report", help="full analysis + performance report")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--quick", action="store_true", help="skip walk-forward and sensitivity")
    p = sub.add_parser("paper", help="paper trading")
    p.add_argument("action", choices=["init", "run", "status"])
    p.add_argument("--force", action="store_true", help="(init) overwrite an existing paper portfolio")
    p.add_argument("--no-refresh", action="store_true", help="(run) use cached data")
    p = sub.add_parser("signals", help="today's model portfolio (writes results/live/)")
    p.add_argument("--refresh", action="store_true", help="re-download data first")
    sub.add_parser("dashboard", help="launch the web app locally")

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    {"download": cmd_download, "backtest": cmd_backtest, "report": cmd_report,
     "paper": cmd_paper, "signals": cmd_signals, "dashboard": cmd_dashboard}[args.command](cfg, args)


if __name__ == "__main__":
    main()
