# Alpha Quant v21.2 Hype UI patch (config v24)

## Install
Back up the existing app files and database. Replace app.py and config.py together.
The ZIP also includes the supplied indicators.py and sentiment_sources.py unchanged,
so all four files can be copied together. Keep your existing portfolio.py,
sec_research.py, requirements.txt and .streamlit folder. This is an update package,
not a complete standalone deployment. No database migration or credentials included.
Expected badge: config v24 / app expects v24.

## UI location
Chart Deep Dive: immediately after the factor contribution explanation and before
raw evidence and the price chart. A Hype summary leads into two bordered cards:
Yahoo! board "everyone's evaluation" and Stocktwits. Both headings always appear.
Disabled, unavailable, not collected, credentials missing and not applicable states
remain visible. Raw categories, provider window, timestamps, parser and failure
reason are in source-detail expanders. English and Japanese translations included.
Existing mobile CSS stacks columns; tested at 390px without horizontal overflow.
No source fetch occurs during panel rendering. Data remains the scan-time snapshot.
Stocktwits score is /100, never buy percentage; missing values never become 0 or 50.
Provider labels are rendered as literal text, not unsafe HTML.

## Additional correctness fixes
The first-run quick-scan button previously forced use_jp_forum=True despite the
experimental toggle defaulting off. It now respects the selected toggle and sources.
run_engine defaults to JP_FORUM_DEFAULT_ON. The Sell Scanner also respects the
experimental toggle rather than fetching/scoring the Japan poll unconditionally.
Each main-scan result stores jp_forum_enabled for honest restored-snapshot status.
No weights, thresholds, scoring formulas or production model version were changed.
Opt-in enforcement intentionally changes paths that previously bypassed the setting.

## Independent checks performed
- All four Python modules compile.
- 31 targeted pytest checks pass, including 12 real Streamlit panel cases across
  English/Japanese, full Deep Dive placement, provider-label safety, prior poll
  boundaries, 7,776 deterministic poll combinations, invalid volume and empty Audit.
- Real app startup succeeds and shows config v24 (yfinance absent in this environment).
- Real Chromium desktop and mobile checks pass using clearly synthetic UI fixtures.
- indicators.py and sentiment_sources.py are byte-identical to the supplied files.
- analyze_ticker, score_with_weights, record_observations and render_engine_audit
  are AST-identical to the supplied functions.

The 50.5 + 50.5 decimal fixture now correctly fails the parser's displayed-precision
feasibility check; an integer-rounded 50 + 51 fixture remains accepted and scorable.
The all-neutral poll scores 50; 100 + 2 is rejected; a later unrelated HTML heading
cannot supply the poll; an empty recommendations table still calls Model Lab.

## Limits
This is targeted verification, not reproduction of Claude's full release suite.
No live Yahoo Japan page, authenticated Stocktwits API, Postgres/Supabase or live
market-data scan was verified. The Japan parser remains structural/experimental.
Browser appearance was checked for this new panel, not every app page.
No claims about investment returns are made. Synthetic fixtures are tests only and
are not included in production data or a live database.

## Reproduce targeted tests
Place the package files together and run:
    pytest -q review/test_hype_review.py
Requires your existing app dependencies plus pytest. Tests use fixtures, not live APIs.

## Rollback
Restore the previous app.py and config.py together. No database rollback is needed.
