"""Paper-trading engine: persistence, idempotency and next-open fills (offline, synthetic data)."""
import pandas as pd
import pytest

from backtesting.pipeline import DataContext
from data.universe import Universe
from paper_trading.engine import PaperTrader, is_month_end
from tests.conftest import SECTORS, TICKERS


@pytest.fixture()
def trader_factory(tmp_path, cfg, md, fundamentals):
    universe = Universe(pd.DataFrame({"ticker": TICKERS, "name": TICKERS,
                                      "sector": [SECTORS[t] for t in TICKERS],
                                      "status": "active", "note": ""}))
    pcfg = cfg.override({"paper_trading.state_dir": str(tmp_path / "paper"),
                         "alerts.macos_notifications": False})
    # state_dir is resolved relative to the project root; an absolute path overrides that
    def make(cut):
        ctx = DataContext(md.truncate(pd.Timestamp(cut)), universe,
                          type(fundamentals)({k: v.loc[:cut] for k, v in fundamentals.panels.items()},
                                             fundamentals.coverage))
        return PaperTrader(pcfg, ctx)
    return make


def test_month_end_detection():
    assert is_month_end(pd.Timestamp("2024-05-31"), pd.Timestamp("2024-06-03"))
    assert not is_month_end(pd.Timestamp("2024-05-30"), pd.Timestamp("2024-05-31"))


def test_paper_cycle(trader_factory, md):
    d0 = md.dates[400]
    t = trader_factory(d0)
    t.init()
    out = t.run(refresh=False)
    assert out["days"] == 1 and len(out["orders"]) > 0     # initial portfolio -> orders for next open
    assert out["trades"].empty                              # nothing filled yet

    # running again on the same data does nothing (idempotent)
    assert trader_factory(d0).run(refresh=False) == {}

    # next day: the orders fill at that day's OPEN
    d1 = md.dates[401]
    out2 = trader_factory(d1).run(refresh=False)
    fills = out2["trades"]
    assert len(fills) > 0
    assert (pd.to_datetime(fills["date"]) == d1).all()
    tk = fills.iloc[0]["ticker"]
    impact = 7 / 10_000
    assert fills.iloc[0]["fill_price"] == pytest.approx(md.open.loc[d1, tk] * (1 + impact), rel=1e-4)

    # later run processes all missed days and keeps an unbroken equity log
    d2 = md.dates[460]
    trader_factory(d2).run(refresh=False)
    eq = trader_factory(d2).store.read("equity")
    assert len(eq) == 461 - 400
    assert eq["date"].is_monotonic_increasing
    state = trader_factory(d2).store.load_state()
    assert state["mode"] == "PAPER_TRADING_ONLY"
    assert state["cash"] >= 0


def test_init_refuses_to_overwrite(trader_factory, md):
    t = trader_factory(md.dates[400])
    t.init()
    with pytest.raises(SystemExit):
        t.init()


def test_split_restatement_preserves_value():
    from paper_trading.engine import PaperTrader
    from portfolio.broker import Position, SimulatedBroker
    from portfolio.costs import CostModel

    dates = pd.bdate_range("2024-06-03", periods=5)
    splits = pd.DataFrame(1.0, index=dates, columns=["AAA"])
    splits.loc[dates[3], "AAA"] = 4.0  # 4:1 split on day 4

    class _MD:
        pass
    md = _MD()
    md.splits = splits
    b = SimulatedBroker(0.0, CostModel())
    b.positions["AAA"] = Position("AAA", 10, 400.0, dates[0], 420.0)
    PaperTrader._apply_splits_since(b, md, dates[1])
    p = b.positions["AAA"]
    assert p.shares == 40 and p.avg_cost == pytest.approx(100.0) and p.peak_price == pytest.approx(105.0)
    PaperTrader._apply_splits_since(b, md, dates[4])   # no new splits -> unchanged
    assert b.positions["AAA"].shares == 40
