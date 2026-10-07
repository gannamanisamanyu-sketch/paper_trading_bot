# Resume Bullets & Interview Preparation

Every number below comes from the completed backtest (`reports/performance_report.md`, data through
2026-10-06, after the bug fixes logged in `RESEARCH_LOG.md`). If you re-run the backtest later,
update the numbers before using them.

---

## Resume bullets (pick ONE version)

**Project line:** *Algorithmic U.S. Equity Paper Trading & Risk Management System (Python)* · GitHub · live web app

**Version A: technical / quant emphasis**
- Built a Python paper-trading system that ranks 109 large-cap U.S. stocks monthly on 7 technical, fundamental and market factors, using point-in-time SEC EDGAR financials to avoid look-ahead bias
- Implemented volatility-adjusted position sizing, a 16% volatility target, trailing stops and position/sector/cash limits; backtested 2016–2026 net of spread and slippage against the S&P 500 (SPY)
- Validated with train/validation/out-of-sample splits, walk-forward testing and 30+ sensitivity runs; found that the strategy cut max drawdown to 18.4% vs. 33.7% for SPY but generated no statistically significant alpha

**Version B: risk-management emphasis (good for AM / risk / IB)**
- Developed a Python multi-factor equity strategy and event-driven backtester with realistic execution (next-open fills, transaction costs, cash interest) and an automated test proving no look-ahead bias
- Designed a layered risk framework (volatility targeting, market-regime and drawdown overlays, trailing stops, 10% position / 30% sector limits) and measured Sharpe, Sortino, VaR, Expected Shortfall and CAPM alpha/beta
- Strategy (beta 0.49) lost 9.0% vs. SPY's 24.5% in the 2022 bear market but lagged SPY over 2016–2026 (6.7% vs. 15.3% CAGR); presented findings in an investment-committee memo

**Version C: concise / full-stack emphasis**
- Built and deployed a Python U.S. equity paper-trading and risk system (multi-factor ranking of 109 stocks, volatility-adjusted sizing, stop-losses, sector limits) with a Streamlit web app, daily GitHub Actions automation and 45 unit tests
- Backtested 2016–2026 vs. the S&P 500 with out-of-sample and walk-forward tests: max drawdown −18.4% vs. −33.7%, Sharpe 0.41 vs. 0.77, no statistically significant alpha

> Why the bullets mention underperformance: an interviewer *will* ask "did it beat the market?"
> Honest, quantified risk results plus rigorous validation are far more credible than a suspiciously
> high backtest return. Never write "profitable strategy" or "outperformed": both would be false on a
> risk-adjusted basis.

**30-second pitch:**
"I built a Python system that ranks about 110 large-cap stocks on momentum, quality, value, growth and
risk, sizes positions by volatility with sector and drawdown limits, and backtests it with no look-ahead
bias using SEC filing dates for fundamentals. It's live as a web app, and a GitHub Action paper-trades it
every day. The honest result: it halved drawdowns in crashes but didn't beat the S&P 500, even
risk-adjusted, and out of sample it was nearly flat on a risk-adjusted basis. The most valuable part was
learning to validate honestly. An independent code review found bugs after my first test run; I fixed
them, logged the before-and-after, and the out-of-sample result got worse, which I reported as-is."

---

## 20 interview questions with answers

**1. Walk me through the project.**
It's a full pipeline: data → factor scores → buy/sell/hold signals → portfolio construction with risk limits
→ simulated execution → performance analysis. I backtested 2016 to 2026 starting with $100k, split into
training, validation and an out-of-sample test, then built a paper-trading engine that runs the same code
daily, automated with GitHub Actions, plus a web app. Result: $100k → $201k vs. $461k for SPY, with an
18% max drawdown vs. 34%.

**2. Why did you choose this strategy?**
Momentum and quality are among the most documented equity factors, and they're intuitive to explain:
trends persist because investors under-react to news, and profitable, low-leverage firms are more
resilient. I added value and growth so I wasn't buying expensive momentum blindly, plus a risk penalty. I
chose something explainable over something exotic, because I wanted to defend every line.

**3. Why these factors, and how are they weighted?**
Momentum 30%, Trend 15%, Sector 10%, Quality 15%, Value 10%, Growth 10%, minus a 10% risk penalty.
Momentum leads because it's the core idea; the others are diversifying tilts. The key point is that the
weights were fixed *before* backtesting and never optimized. Fundamentals are ranked within sector,
because a bank's 10× liabilities/equity is normal while a software company's would be alarming.

**4. How did you prevent look-ahead bias?**
Four ways. (1) All indicators are backward-looking rolling calculations. (2) Signals use day *t*'s close
and can only trade at day *t+1*'s open. (3) Fundamentals come from SEC filings and are used only from the
day after each number was *first* filed. yfinance only gives today's P/E, which would leak the future.
(4) A unit test reruns the backtest on data cut off at a date and checks that every result up to that date
is identical. Interesting detail: a code review still found a subtle leak, where a preferred data tag was
picked from a *later* filing. I fixed it and gave every metric its own filing date.

**5. What is overfitting?**
Fitting a model to the noise of one historical sample rather than a real, repeatable pattern. Try enough
parameter combinations and any strategy looks great in a backtest, then fails on new data. The signs are
great in-sample results that collapse out of sample, or results that only work at one exact setting.
Mine went from Sharpe 0.79 in training to 0.04 out of sample. Since I didn't optimize anything, that
decay says the factors simply had no durable edge in this universe.

**6. How did you prevent overfitting?**
Few free parameters, set a priori. A train/validation/test split with settings frozen before the test
ran. A research log documenting every post-result change. A walk-forward test that picks among only 4
pre-declared configs using prior years. And 30+ sensitivity runs. I also *didn't* adopt variants that
looked better, like removing the stop-loss (Sharpe 0.60 vs. 0.41), because choosing them after seeing
test results would be overfitting.

**7. Why use the Sharpe ratio?**
It measures excess return over the risk-free rate per unit of volatility, so you can compare strategies
with different risk levels. Mine had lower returns and lower volatility, so Sharpe is the fair comparison:
0.41 vs. SPY's 0.77. SPY was better even after adjusting for risk. I used the actual T-bill rate, not
zero, because rates were 4–5% for part of the period.

**8. Why use the Sortino ratio?**
Sharpe penalizes upside volatility too, but investors don't mind upside surprises. Sortino divides excess
return by *downside* deviation only. Mine was 0.55 vs. SPY's 1.08, the same conclusion.

**9. What is alpha?**
The return not explained by market exposure. I regress the strategy's daily excess returns on SPY's: the
slope is beta and the annualized intercept is alpha. Mine was −1.7% per year with a t-stat of −0.64,
statistically zero. After accounting for beta, the strategy didn't add value.

**10. What is beta?**
Sensitivity to the market. Mine was 0.49: when SPY moves 1%, the strategy moves about 0.49%. It's low
because the strategy held cash at times (76% average invested), used a regime filter, and the risk
penalty tilted toward lower-beta stocks. Low beta explains most of both the protection in crashes and the
lag in bull markets.

**11. What is maximum drawdown and why does it matter?**
The largest peak-to-trough loss. Mine was −18.4%, in the March 2020 COVID crash, vs. −33.7% for SPY. It
matters because it's what investors actually live through. A 50% loss needs a 100% gain to recover, and
many investors sell at the bottom. It's the metric where this strategy genuinely did well.

**12. What is VaR, and how is it different from Expected Shortfall?**
95% one-day VaR is the loss you'd exceed on only 5% of days. Mine was 1.21% vs. 1.66% for SPY. VaR
doesn't say *how bad* those days are. Expected Shortfall is the average loss on them: 1.92% vs. 2.70% for
SPY. Regulators moved toward ES (Basel FRTB) because it captures tail severity. I also show parametric
VaR, which assumes normal returns and understates fat tails.

**13. Why use volatility-adjusted position sizing?**
With equal weights, a 60%-vol stock carries three times the risk of a 20%-vol stock, so a few volatile
names dominate risk. Inverse-vol weighting gives each position roughly equal risk. In my test it
*didn't* help: equal weighting had a Sharpe of 0.44 vs. 0.41 for inverse-vol, with nearly identical
drawdown. A good answer is that the theory is sound, but inverse-vol tilts toward defensive names, and in
a bull market led by volatile mega-cap tech that tilt cost return.

**14. How did you account for transaction costs?**
Each trade pays a 2 bps half-spread (buy at the ask, sell at the bid) plus 5 bps slippage, applied to the
fill price, with commission modeled separately ($0 by default). Fills happen at the next day's open. Total
costs were about $5,400 over 10 years. I stress-tested 0 to 50 bps: Sharpe falls from 0.43 to 0.30. A
rank buffer and a 1% minimum trade size limit turnover.

**15. Why did the strategy underperform SPY?**
Three reasons. First, low beta in a strong bull market: SPY compounded 15%/yr, and holding about half the
market's risk mechanically gives up a lot. Second, the factors had no predictive power here: monthly
information coefficients were near zero. Among 110 mega-caps there's little cross-sectional dispersion,
and the decade was dominated by a few mega-cap winners. Third, the risk overlays re-entered late after
crashes. From the March 2020 low the strategy gained 13% vs. 45% for SPY. A beta-matched passive
portfolio beat it by ~2.2%/yr.

**16. What was the biggest weakness of the model?**
The signal itself: near-zero IC means the ranking didn't predict returns, so the risk management was
protecting a strategy without an edge. That's why the results are fragile: small bug fixes moved the
out-of-sample Sharpe from 0.34 to 0.04. Second, survivorship bias: I could only get free data for
companies that still exist, so results are, if anything, flattered.

**17. What would you improve next?**
(1) Point-in-time index membership data to remove survivorship bias. (2) A broader universe like the full
S&P 500 or mid-caps, where factor dispersion is larger. (3) Quarterly TTM fundamentals and
earnings-revision factors. (4) Momentum-crash protection after market rebounds. (5) A risk model and
optimizer instead of rank-and-cap. Then I'd let the paper portfolio run 6–12 months before trusting anything.

**18. What is survivorship bias and how did you handle it?**
Testing only on companies that survived to today ignores the ones that failed or were acquired, which
inflates results. I fixed the universe as of 2015 using only what was known then. That's why NVDA and
TSLA aren't in it. I kept the 11 names without free data in the universe file, flagged, so the bias is
documented. Free data can't fully fix it.

**19. What's an information coefficient and what did yours show?**
It's the rank correlation between a factor's score today and each stock's next-month return, so it tells
you whether the factor ranks stocks in the right order. Useful factors in practice have ICs of around
0.02–0.05. Mine were near zero, with the composite around −0.01 over the full sample. Growth was the
closest to meaningful in the test period (t ≈ 1.9). That's the clearest evidence the stock selection had no edge.

**20. A code review found bugs after you ran the out-of-sample test. What did you do?**
I fixed them: a stop-loss accounting issue, a subtle fundamentals timing leak, and a price-screen issue.
Then I re-ran everything and logged before-and-after numbers. Out-of-sample Sharpe dropped from 0.34 to
0.04. Fixing bugs after a test is legitimate; changing parameters isn't. So I didn't re-tune anything,
even though the fixes made some variants (like the original stop) look better. Reporting a worse
corrected number is the point of a research process.

**Bonus: why skip the most recent month in momentum?**
One-month returns tend to *reverse* (the short-term reversal effect, partly from liquidity and bid/ask
bounce), while 2–12-month returns tend to *continue*. Skipping the last month isolates the persistent part.

**Bonus: how is paper trading different from the backtest?**
Same code: both use the same `TradingSystem` and simulated broker classes. The paper engine runs daily on
live data via GitHub Actions, fills yesterday's orders at today's real opening price, credits dividends,
handles splits, and saves everything to the repo. It never connects to a broker. It's a true forward test,
because those future days didn't exist when I designed the strategy.
