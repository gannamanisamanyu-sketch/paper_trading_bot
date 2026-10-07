"""The most important test in the project: no decision may depend on future data.

Method: compute factor scores / backtest decisions on the FULL history, then
recompute them on data TRUNCATED at date t (as if today were t). If any
calculation peeked ahead, the two answers would differ.
"""
import numpy as np
import pandas as pd
import pytest

from backtesting.engine import Backtester
from strategy.factors import FactorModel, FeatureSet


@pytest.mark.parametrize("cut_index", [320, 450, 600])
def test_scores_identical_on_truncated_history(md, fundamentals, cfg, sectors, cut_index):
    date = md.dates[cut_index]
    full = FactorModel(FeatureSet.build(md, fundamentals, sectors, cfg.strategy), cfg.strategy).score(date)
    md_cut = md.truncate(date)
    fd_cut = type(fundamentals)({k: v.loc[:date] for k, v in fundamentals.panels.items()}, fundamentals.coverage)
    cut = FactorModel(FeatureSet.build(md_cut, fd_cut, sectors, cfg.strategy), cfg.strategy).score(date)
    assert len(full) > 0
    pd.testing.assert_frame_equal(full.sort_index(), cut.sort_index(), check_like=True)


def test_backtest_path_identical_on_truncated_history(md, fundamentals, cfg, sectors):
    """Equity up to date t must not change when data after t is appended."""
    cut_date = md.dates[560]
    feats_full = FeatureSet.build(md, fundamentals, sectors, cfg.strategy)
    res_full = Backtester(cfg, md, feats_full, sectors).run()
    md_cut = md.truncate(cut_date)
    fd_cut = type(fundamentals)({k: v.loc[:cut_date] for k, v in fundamentals.panels.items()},
                                fundamentals.coverage)
    feats_cut = FeatureSet.build(md_cut, fd_cut, sectors, cfg.strategy)
    res_cut = Backtester(cfg, md_cut, feats_cut, sectors).run()
    a = res_full.equity["equity"].loc[:cut_date]
    b = res_cut.equity["equity"].loc[:cut_date]
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)
    fa = res_full.fills[res_full.fills["date"] <= cut_date].reset_index(drop=True)
    fb = res_cut.fills[res_cut.fills["date"] <= cut_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(fa[["date", "ticker", "side", "shares"]], fb[["date", "ticker", "side", "shares"]])


def test_orders_fill_at_next_open_not_same_close(md, fundamentals, cfg, sectors):
    feats = FeatureSet.build(md, fundamentals, sectors, cfg.strategy)
    res = Backtester(cfg, md, feats, sectors).run()
    first_signal = res.signals["date"].min()
    first_fill = res.fills["date"].min()
    assert first_fill > first_signal
    # every fill's reference price is that day's open, never that day's close
    f = res.fills.iloc[0]
    assert f["reference_price"] == pytest.approx(md.adj_open.loc[f["date"], f["ticker"]])


def test_stopped_name_frees_its_slot_on_rebalance_day(md, fundamentals, cfg, sectors):
    """A stock stopped out on a rebalance day must not keep a slot or a target weight."""
    from backtesting.engine import TradingSystem
    from portfolio.broker import Position, SimulatedBroker
    from portfolio.costs import CostModel

    feats = FeatureSet.build(md, fundamentals, sectors, cfg.strategy)
    system = TradingSystem(cfg, feats, sectors)
    date = md.dates[600]
    scores = system.model.score(date)
    top = scores.index[0]
    broker = SimulatedBroker(50_000, CostModel())
    broker.positions[top] = Position(top, 100, 50.0, date, 60.0)
    orders, signals, _, _ = system.rebalance(date, broker, md.adj_close.loc[date], False, set(),
                                             exiting={top})
    assert top not in set(o.ticker for o in orders)          # no duplicate exit, no target
    assert top not in set(signals.loc[signals.signal.isin(["BUY", "HOLD"]), "ticker"])
