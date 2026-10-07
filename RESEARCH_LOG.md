# Research Log

Every design change made after looking at backtest results is recorded here, along with
the period it was based on and the reasoning. This makes the research process auditable
and protects against overfitting: if a change can't be justified *without* pointing to the
backtest number it improved, it doesn't get made.

**Period discipline**
| Period | Dates | Used for |
|---|---|---|
| Train | 2016-01-01 → 2019-12-31 | Debugging and design review |
| Validation | 2020-01-01 → 2021-12-31 | Robustness checks and walk-forward candidate selection, no redesign |
| Test (out-of-sample) | 2022-01-01 → latest | Run once with frozen settings, reported as-is |

---

## Entry 0: Initial design (before any backtest)
Factor weights, universe, position limits, volatility target (16%), cost assumptions
(2 bps half-spread + 5 bps slippage) and all other parameters in `config/config.yaml`
were set from first principles before the first backtest was run.

## Entry 1: Stop-loss redesign (based on TRAIN period only)
**Observation.** With the initial 3 × ATR(14) trailing stop, 296 of 331 position entries
(89%) were closed by the stop. Average invested capital was only ~70%, and turnover was
very high (813 fills in 4 years).

**Diagnosis.** A logic mismatch, not a tuning issue. 3 × ATR(14) is about 5% below the
peak for a typical large cap. That is a swing-trading stop designed for multi-day holds.
This strategy is rebalanced monthly from a 12-month signal, so a normal 5% pullback was
forcing exits that the selection model itself would not have made. The stop was acting
as a noise trigger, not a risk control.

**Change.** The stop distance is now scaled to the strategy's holding horizon:
exit if close < peak × (1 − 2 × σ_annual × √(21/252)), i.e. a ~2-sigma *monthly* decline.
For a 25%-vol stock that is ~14% below the peak.
The multiple (2) was chosen from that statistical reasoning, **not** by searching for the
best backtest value. The ATR stop is kept as a sensitivity-test alternative.

**Effect on train period (reported for transparency):**
stop exits 296 → 61; average gross exposure 70% → 80%; CAGR 3.5% → 7.1%;
Sharpe 0.28 → 0.53. SPY over the same period: CAGR 14.7%, Sharpe 1.03.
**The strategy still underperformed SPY in training.**

## Entry 2: Finding left unchanged (TRAIN period)
Monthly rank information coefficients (IC: correlation between each factor's z-score and
next-month return across the universe) were all near zero and statistically insignificant
in training (composite score IC ≈ −0.01). **The factor weights were deliberately NOT
changed in response**, because re-weighting factors to fit four years of data is textbook
overfitting. This is reported as a key finding: within a universe of ~110 mega/large caps,
these well-known factors showed little cross-sectional predictive power in 2016-2019.

## Freeze
After Entry 1, **all strategy settings are frozen.** Validation and test results are
generated with this exact configuration. Any later change must be logged here as a new
entry and the test-period result labeled as "post-hoc".

## Entry 3: Correctness fixes after the test period was run (post-hoc, bug fixes only)
An independent code review found three bugs. Fixing *bugs* after seeing test results is legitimate;
changing *parameters* would not be. No parameter was changed. All results in the repository were regenerated.

1. **Stop-loss slot bug** (`backtesting/engine.py`). A stock stopped out on a rebalance day still occupied
   a portfolio slot and a target weight, so its capital sat in cash until the next rebalance.
   Fixed: stopped names are treated as already sold when the rebalance is computed.
2. **Fundamentals were not fully point-in-time** (`data/fundamentals.py`). The preferred XBRL tag was chosen
   before the earliest filing, so some values (e.g. Alphabet's FY2016 EPS, first tagged in a 2019 filing)
   were used before they were public. Fixed: earliest filing first; every metric has its own availability
   date; growth uses the current and prior year from the same 10-K. Coverage rose from ~87% to ~96%.
3. **Price screen used future splits** (`strategy/factors.py`). The $5 screen used split-adjusted prices.
   No effect on this universe, but fixed to use the actual traded price.

Two reporting bugs were also fixed: sub-period trade statistics mishandled positions opened before the
period, and paper-trading catch-up runs could mis-handle a split that happened between runs.

**Effect (before → after):**
| | Full CAGR | Full Sharpe | Max DD | Test CAGR | Test Sharpe |
|---|---:|---:|---:|---:|---:|
| Before fixes | 7.25% | 0.45 | −20.4% | 7.40% | 0.34 |
| **After fixes** | **6.68%** | **0.41** | **−18.4%** | **3.82%** | **0.04** |

**Observations, deliberately NOT acted on:**
* Out-of-sample performance moved a lot (Sharpe 0.34 → 0.04) from small correctness changes. That is
  further evidence the strategy has no robust edge: its results are dominated by noise.
* After the fixes, the original 3 × ATR stop (full Sharpe 0.57) and no stop at all (0.60) both score
  better than the volatility stop adopted in Entry 1 (0.41). Entry 1 was based on training data from the
  buggy version. Switching back now would be choosing a rule by its test-period result, so the frozen
  configuration stands, and this is reported as a finding.
* Equal weighting (Sharpe 0.44) now edges out inverse-volatility weighting (0.41).
