import numpy as np
import pandas as pd
import pytest

from data.fundamentals import compute_annual_metrics, point_in_time_panel
from strategy.factors import sector_zscore, zscore
from strategy.signals import BUY, HOLD, SELL, generate_signals


def score_table(n=40):
    t = [f"T{i}" for i in range(n)]
    return pd.DataFrame({"score": np.linspace(1.5, -1.5, n), "rank": np.arange(1, n + 1),
                         "above_sma200": True, "rsi": 50.0}, index=t)


def test_buy_top_names_and_respect_slots(cfg):
    s = score_table()
    out = generate_signals(s, [], cfg.strategy, max_positions=5)
    assert list(out.loc[out.signal == BUY, "ticker"]) == ["T0", "T1", "T2", "T3", "T4"]


def test_hysteresis_buffer(cfg):
    s = score_table()
    # T7 is rank 8: outside the top 5 but inside the top 10 buffer -> HOLD
    out = generate_signals(s, ["T7", "T25"], cfg.strategy, max_positions=5).set_index("ticker")
    assert out.loc["T7", "signal"] == HOLD
    assert out.loc["T25", "signal"] == SELL   # rank 26 > 10 and score negative


def test_entry_filters(cfg):
    s = score_table()
    s.loc["T0", "above_sma200"] = False       # downtrend
    s.loc["T1", "rsi"] = 85.0                 # overbought
    out = generate_signals(s, [], cfg.strategy, max_positions=3, stopped_out={"T2"})
    assert list(out.loc[out.signal == BUY, "ticker"]) == ["T3", "T4", "T5"]


def test_no_buys_below_min_score(cfg):
    s = score_table()
    s["score"] = -0.5
    out = generate_signals(s, [], cfg.strategy, max_positions=5)
    assert (out.signal == BUY).sum() == 0


def test_zscores_winsorized_and_neutral_nan():
    x = pd.Series([1.0, 2.0, 3.0, 100.0, np.nan])
    z = zscore(x, clip=1.5)
    assert z.max() <= 1.5 and z.iloc[-1] == 0.0
    sectors = pd.Series(["A", "A", "A", "A", "B"], index=x.index)
    zs = sector_zscore(x, sectors, clip=3.0, min_group=4)
    assert zs.iloc[-1] == 0.0


def test_point_in_time_uses_filing_date_not_period_end():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    m = pd.DataFrame({"roe": [0.10, 0.20],
                      "available": pd.to_datetime(["2020-02-14", "2021-02-12"])},
                     index=pd.to_datetime(["2019-12-31", "2020-12-31"]))
    m.index.name = "period_end"
    p = point_in_time_panel({"X": m}, dates, "roe", lag_days=1, avail_col="available")["X"]
    assert np.isnan(p.loc["2020-02-14"])                 # filing day itself: not yet usable
    assert p.loc["2020-02-17"] == pytest.approx(0.10)   # first trading day after filing
    assert p.loc["2021-02-12"] == pytest.approx(0.10)   # FY2020 not yet public
    assert p.loc["2021-02-16"] == pytest.approx(0.20)


def test_eps_split_restatement():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    m = pd.DataFrame({"eps_diluted": [8.0], "available": pd.to_datetime(["2020-02-03"])},
                     index=pd.to_datetime(["2019-12-31"]))
    m.index.name = "period_end"
    # a 4:1 split happens later -> reported EPS must be divided by 4 to match split-adjusted prices
    divisor = pd.DataFrame({"X": np.where(dates < "2020-08-31", 4.0, 1.0)}, index=dates)
    p = point_in_time_panel({"X": m}, dates, "eps_diluted", divisor=divisor, avail_col="available")["X"]
    assert p.loc["2020-12-30"] == pytest.approx(2.0)


def test_annual_metrics_and_per_field_availability():
    idx = pd.to_datetime(["2019-12-31", "2020-12-31"])
    f19, f20, late = pd.Timestamp("2020-02-10"), pd.Timestamp("2021-02-10"), pd.Timestamp("2022-02-10")
    t = pd.DataFrame({
        "net_income": [10.0, -5.0], "net_income_prev": [8.0, 10.0], "net_income_filed": [f19, f20],
        "revenue": [100.0, 120.0], "revenue_prev": [90.0, 100.0], "revenue_filed": [f19, f20],
        "equity": [50.0, 70.0], "equity_prev": [40.0, 50.0], "equity_filed": [f19, f20],
        "liabilities": [100.0, 140.0], "liabilities_filed": [f19, f20],
        "eps_diluted": [1.0, -0.5], "eps_diluted_filed": [f19, late]}, index=idx)
    t.index.name = "period_end"
    m = compute_annual_metrics(t)
    assert m["revenue_growth"].iloc[-1] == pytest.approx(0.2)
    assert m["earnings_growth"].iloc[-1] == pytest.approx(-1.0)   # clipped
    assert m["roe"].iloc[-1] == pytest.approx(-5 / 60)
    assert m["leverage"].iloc[-1] == pytest.approx(2.0)
    # EPS first appeared in a LATER filing -> it must carry that later date
    assert m["avail_eps_diluted"].iloc[-1] == late
    assert m["avail_roe"].iloc[-1] == f20


def test_first_reported_prefers_earliest_filing_over_tag_priority():
    from data.fundamentals import first_reported, prior_in_same_filing
    rows = pd.DataFrame({
        "end": pd.to_datetime(["2020-12-31", "2020-12-31", "2019-12-31"]),
        "val": [110.0, 111.0, 100.0],
        "filed": pd.to_datetime(["2021-02-01", "2022-02-01", "2021-02-01"]),
        "priority": [1, 0, 1]})  # preferred tag (0) only appears in the LATER filing
    f = first_reported(rows)
    assert f.loc["2020-12-31", "val"] == 110.0
    assert f.loc["2020-12-31", "filed"] == pd.Timestamp("2021-02-01")
    prev = prior_in_same_filing(rows, f)
    assert prev.loc["2020-12-31"] == 100.0
