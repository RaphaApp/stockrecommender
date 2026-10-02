# Alpha Quant Engine — changes in this build (config schema v21)

Deploy app.py, config.py and indicators.py together (app.py expects config v21).
portfolio.py and sec_research.py are unchanged.

## Fixes
- Sidebar multiselect chips: first letter cut off. Cause: chips were allowed to grow
  wider than their overflow-hidden row, which the browser then scrolled. Chips are now
  capped at the row width and long labels wrap. Also styled for Streamlit's newer
  react-aria multiselect, where the old chip selectors no longer match.
- Prices and market caps show in their own currency (¥, €, HK$…) instead of "$".
- Postgres: read_sql_query calls with parameters failed silently (placeholder bug).
- News buzz: RTX and SLB never matched any headline; Visa / Recruit / Canon matched
  unrelated stories. See NEWS_QUERY_NAMES in config.py.
- Duplicate function definitions in indicators.py; duplicated Japanese table label.
- requirements.txt now lists what this app imports (yfinance, curl_cffi, plotly,
  psycopg2-binary).

## Model (PRODUCTION_MODEL_VERSION = composite-v3-abnormal-buzz)
- Hype buzz kicker scores mentions against the stock's own 30-day normal
  (buzz_history table), or the level expected for its market cap until 5 days of
  history exist. Set HYPE_BUZZ_MODE = "legacy" in app.py to restore v2 exactly.
- Raw mentions, buzz ratio, legacy bonus, market regime and 63-day volatility are
  recorded as Model Lab candidate signals.

## UI
- Top Selections: diversified podium (one per theme), diverging factor bars, "why"
  line, factor agreement, entry/target/stop with reward:risk, risk flags
  (earnings ≤14d, downtrend home market, RSI ≥72, payout >100%, vol ≥50%, buzz spike),
  and a "Next in line" table that opens Chart Deep Dive.
- Status strip gains a Trend cell (each home index vs its 200-day average).
- Category > Growth ranks by market-relative momentum instead of raw 1-month return.
