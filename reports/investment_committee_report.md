# Investment Committee Memo
**Strategy:** Systematic Multi-Factor U.S. Large-Cap Equity (Long-Only), Paper Portfolio
**Prepared by:** [Your name] · **Date:** October 6, 2026
**Request:** Approval to continue running the strategy as a **paper portfolio**. No capital is requested.

> Simulated results. Educational project; not investment advice.

---

## 1. Executive Summary
We tested a rules-based strategy that ranks ~110 large-cap U.S. stocks monthly on momentum, trend,
quality, value, growth and risk, and holds the top 15 under strict risk controls. Over **January 2016 to
October 2026**, $100,000 grew to **$200,524 (6.7% CAGR)** versus **$460,536 (15.3% CAGR)** for SPY.
Risk was meaningfully lower (volatility 12.2% vs. 17.7%; max drawdown **−18.4% vs. −33.7%**; beta 0.49),
but **risk-adjusted performance was also lower** (Sharpe 0.41 vs. 0.77). Alpha was −1.7% per year with a
t-statistic of −0.64, statistically indistinguishable from zero. On the untouched out-of-sample period
(2022 to 2026) the strategy returned 3.8%/yr (Sharpe 0.04) vs. 12.4% (Sharpe 0.54) for SPY.

**Recommendation:** do **not** allocate capital. Continue paper trading to observe live behavior, and
treat the strategy as a *defensive equity sleeve* rather than an alpha source. Its demonstrated value is
drawdown control, not stock selection.

## 2. Strategy
* **Universe:** 109 tradable large caps across 11 sectors, fixed as of end-2015 to avoid hindsight bias.
* **Score:** 0.30 Momentum (12-1 month) + 0.15 Trend (price vs. 200-day SMA) + 0.10 Sector momentum +
  0.15 Quality (ROE, leverage) + 0.10 Value (earnings yield) + 0.10 Growth (revenue & earnings) −
  0.10 Risk (volatility, beta). Fundamental factors are ranked within sector and sourced from SEC
  10-K filings only after each number's first filing date. Weights were fixed a priori.
* **Rules:** buy the highest-scoring names (score > 0, above 200-day SMA, RSI < 80); hold while in the
  top 30; sell below rank 30 or score −0.25. Monthly rebalance; signals at the close, fills at the next open.

## 3. Portfolio Construction
Up to 15 positions, **inverse-volatility weighted**, with ≤ 10% per name, ≤ 30% per sector, ≥ 5% cash,
and a per-position risk limit (weight × volatility ≤ 3% of equity). The portfolio is scaled down if its
estimated volatility exceeds 16%. Equal weighting performed slightly better in this sample (Sharpe 0.44 vs.
0.41, similar drawdown), so the theoretical benefit of inverse-volatility weighting did not show up here.

## 4. Risk Management
| Control | Rule | Observed effect |
|---|---|---|
| Market regime | SPY < 200-day SMA → ≤ 50% invested | Most useful overlay: without it Sharpe falls to 0.38 and max DD widens to −23.8% |
| Trailing stop | exit at 2 × monthly σ below peak | **Hurt returns**: no stop gives Sharpe 0.60; kept because changing it now would be curve-fitting |
| Drawdown control | DD > 15% → ≤ 50% invested | Active 268 of 2,705 days (10%); slightly negative (off: Sharpe 0.45) |
| Limits | 10% name / 30% sector / 5% cash | Enforced at rebalance; between rebalances drift took one sector to 32.0% and cash to 1.0% |

Average gross exposure was 76%. 1-day 95% VaR was 1.21% (SPY 1.66%); Expected Shortfall 1.92% (SPY 2.70%).

## 5. Backtest Methodology
* Daily event-driven simulation from $100,000; orders from day *t*'s close fill at day *t+1*'s open with
  2 bps half-spread + 5 bps slippage; idle cash earns the T-bill rate.
* **Look-ahead controls:** causal indicators, next-open execution, point-in-time fundamentals (per-metric
  filing dates), and an automated test proving that results up to any date are unchanged when future data is removed.
* **Overfitting controls:** train (2016–19) / validation (2020–21) / test (2022–26) split, settings frozen
  before the test was run, a 4-candidate walk-forward, and a 30+-variant sensitivity analysis.
* **Post-hoc changes, fully logged:** one design change (stop-loss, training data only), plus correctness
  fixes found in an independent code review after the test was first run. The fixes lowered the
  out-of-sample Sharpe from 0.34 to 0.04. No parameters were changed.

## 6. Performance
| | Train 2016–19 | Validation 2020–21 | **Test 2022–26** | Full | SPY Full |
|---|---:|---:|---:|---:|---:|
| CAGR | 10.3% | 6.5% | **3.8%** | 6.7% | 15.3% |
| Volatility | 11.6% | 15.3% | 11.2% | 12.2% | 17.7% |
| Sharpe | 0.79 | 0.48 | **0.04** | 0.41 | 0.77 |
| Max drawdown | −16.9% | −18.4% | −16.2% | −18.4% | −33.7% |
| Beta | 0.68 | 0.47 | 0.42 | 0.49 | 1.00 |
| Alpha (ann.) | +0.1% | −4.0% | −3.4% | −1.7% | n/a |

Performance **decayed from train to test**, the classic signature of a strategy without a durable edge.
1,334 fills, 358 closed positions, 47% win rate, profit factor 1.52, average holding period 153 days,
$5,417 total trading costs. **Walk-forward** (2019–26, chosen from prior years only): 5.2% CAGR,
Sharpe 0.25 vs. SPY 17.5%, 0.79.

**Attribution.** Monthly rank information coefficients were near zero for every factor in every period
(composite IC ≈ −0.01 over the full sample). The factors did not reliably rank stocks within this mega-cap
universe. A passive 0.49 × SPY + 0.51 × T-bill mix would have compounded at 8.9%/yr vs. 6.7% for the
strategy, so the active decisions *subtracted* ~2.2%/yr. The largest gains came from Information
Technology (+$15k), Energy (+$14k) and Communication Services (+$14k); top contributors were Phillips 66,
GE and Boeing.

## 7. Benchmark Comparison & Stress Testing
| Episode | Strategy | SPY |
|---|---:|---:|
| Q4 2018 sell-off | −15.3% | −19.4% |
| COVID crash (Feb–Mar 2020) | **−18.3%** | −33.7% |
| COVID rebound (Mar–Jun 2020) | +12.6% | **+45.0%** |
| 2022 bear market | **−9.0%** | −24.5% |
| 2023 regional-bank stress | −4.3% | −1.9% |
| 2025 tariff shock | **−10.8%** | −18.8% |

The strategy protects well in *gradual* bear markets (2022), where the regime filter and stops have time to
act, and roughly halves losses in sharp crashes. It badly lags V-shaped recoveries because it re-enters only
after trends turn up again. A hypothetical instantaneous −20% market shock would cost the current portfolio
≈ 10.8% ($21.6k).

**Sensitivity.** Across 30+ one-at-a-time variants the full-period Sharpe ranged from 0.30 to 0.63, and
**no variant matched SPY's 0.77**. Test-period Sharpes ranged from 0.02 to 0.54: the result depends heavily on
arbitrary choices, which points to noise rather than signal. Performance degrades with costs (Sharpe 0.43 at
0 bps → 0.30 at 50 bps).

## 8. Limitations
1. **Survivorship bias:** 11 of 120 original names lack free data (mostly acquired companies), which likely flatters results.
2. **Weak, unstable signal:** near-zero ICs; small changes produce large swings in out-of-sample results.
3. **Annual fundamentals** update slowly; current GICS sectors are applied historically.
4. **Simplified execution:** fixed-bps costs, whole shares, no capacity, tax or market-impact modeling.
5. **One sample:** ~11 years with a dominant mega-cap bull market; results may not generalize.

## 9. Conclusion
The system is technically sound: reproducible, free of look-ahead bias, cost-aware, independently
reviewed, and honestly validated. But **the strategy does not earn excess risk-adjusted returns**, and its
out-of-sample performance was weak. It is a lower-risk, lower-return equity exposure whose main
contribution is cutting drawdowns in 2020, 2022 and 2025. The committee should treat it as a research
platform and a defensive-overlay prototype. Next steps: point-in-time universe data, a broader universe
with more factor dispersion, and regime-aware momentum. Re-evaluate after 6–12 months of live paper trading.
