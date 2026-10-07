# data/

Code for loading data lives here (`universe.py`, `market_data.py`, `fundamentals.py`).
Downloaded files go to `data/cache/` (git-ignored) and are re-created with:

```bash
python main.py download
```

| File | Source | What it is |
|---|---|---|
| `cache/prices.parquet` | Yahoo Finance via `yfinance` (free) | Daily OHLC, Adj Close, volume, splits, dividends for the universe + SPY + ^GSPC, 2015 → today |
| `cache/risk_free.parquet` | Yahoo Finance `^IRX` | 13-week T-bill yield (risk-free rate) |
| `cache/sec_company_tickers.json` | SEC EDGAR | Ticker → CIK mapping |
| `cache/sec_facts/CIK##########.json` | SEC EDGAR XBRL `companyfacts` API (free, no key) | Every reported financial fact with its filing date |
| `paper/` | created by `python main.py paper init` | Your persistent paper portfolio (state, trades, equity, signals, alerts) |

No API keys are needed. The SEC asks automated clients to identify themselves in the
`User-Agent` header: set `data.sec_user_agent` in `config/config.yaml` to your
name and email before heavy use.

**No internet?** Without a connection, put a `prices.parquet` with the columns
`date, ticker, open, high, low, close, adj_close, volume, splits, dividends` in `data/cache/`.
Fundamental factors then fall back to neutral (z = 0), and the system still runs on
technical/market factors alone.
