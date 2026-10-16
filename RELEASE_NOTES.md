# Alpha Quant Engine — release v21.3 (Hype panel + opt-in enforced on every path)

Config schema 24. Production model `composite-v6-jp-opt-in-enforced-everywhere`.
Early Setup model and Japan parser unchanged (`early-setup-v6-…`, `jp-board-v3-…`).

Source: an externally supplied patch (a Deep Dive Hype panel plus opt-in fixes) built on
the v21.2 files. I verified its claims, kept what held up, and added fixes of my own. The
patch's own test file (`review/test_hype_review.py`) was listed in its manifest but not
attached, so its "31 tests" and its Chromium checks were not reproduced from the package;
I wrote and ran my own (below).

## Deploy

Replace **`app.py` and `config.py`** (a 2-file update). `indicators.py`,
`sentiment_sources.py`, `portfolio.py`, `sec_research.py`, `requirements.txt` and
`.streamlit/` are unchanged from v21.2 (`config.py`, `indicators.py` and
`sentiment_sources.py` are byte-identical to the files supplied with the patch).
**No database migration** (schema statements identical to v21.2; a v21.2 database was
opened with this build and every row preserved). Expect the sidebar badge
`config v24 · app expects v24`. Rollback: restore the v21.2 `app.py` and `config.py`.

## The important finding: the Japan opt-in was not actually enforced (my v21.2 error)

v21.2 made the poll "off by default" by changing the toggle's default, and its only test
asserted that constant. Reproduced on the published v21.2 code, with the toggle OFF:

| Path | v21.2 | v21.3 |
|---|---|---|
| First-run quick-scan button (real button, real Streamlit, network blocked and recorded) | 10 Yahoo! Japan board requests | 0 |
| `run_engine()` called with no `use_jp_forum` argument | fetched and blended the poll | 0 requests |
| Sell Scanner on a `.T` name | fetched the board and scored a **100 "euphoric — possible retail top"** signal from the unverified parser | no request, no signal |

All three now respect the toggle, and the toggle works in the other direction too (on ->
the poll is fetched and used). The Sell Scanner reads the sidebar state through
`st.session_state["_jp_forum_on"]`; a test pins that the sidebar actually publishes it.

## Intentional behaviour changes (disclosed)

| Change | Effect |
|---|---|
| Opt-in enforced on every path | Japanese composites from the first-run quick scan no longer include the board unless you switch it on; the Sell Scanner's Japan "forum" signal is skipped while it is off. |
| Production model v5 -> v6 | The supplied patch left the version unchanged while acknowledging that these paths change. They change composites, so v5 observations (which can carry a poll although v5 says opt-in) are kept apart from v6. Old rows are kept and excluded from current-model statistics, as before. |

Unchanged: thresholds, weights, every scoring formula, frozen legacy learning, Stocktwits
measurement-only, Early Setup.

## What the patch adds (verified)

* **Hype panel in the Chart Deep Dive**, directly after "Why this pick" and before the
  numbers and the price chart (checked on a real rendered page). A summary (Hype score,
  mentions, volume breakout days) and two bordered cards, Yahoo! 掲示板「みんなの評価」 and
  Stocktwits, each always visible with an explicit state: available, disabled, not
  collected, not applicable, credentials missing, rate limited, and so on. Raw categories,
  provider window, timestamps, parser version and failure reason sit in a details
  expander. Stocktwits is shown as a score out of 100, never as a percentage; missing data
  never becomes 0 or 50; the provider label is rendered as literal text. Timestamps keep
  their actual offset instead of being labelled UTC. 37 strings in English and Japanese.
* Rendering reads the scan-time snapshot and fetches nothing (every panel test runs with
  all network and data-source functions booby-trapped).
* Each result records `jp_forum_enabled`, so a restored snapshot can say honestly whether
  the poll was off or simply not collected.

## What I changed on top of the patch

1. **Restored the Deep Dive's mention context.** The panel replaced a block that also
   said what the mention count is and which feeds produced it (Reddit/GDELT status and
   last-updated). Restored with existing strings, so `config.py` stays exactly as supplied.
2. **Fixed a wrong kanji in two user-visible Sell Scanner strings:** 「掛示板」 -> 「掲示板」.
3. **Model version bump** (above).
4. **Tests that exercise behaviour, not constants** (47 new).

## Validation (executed for v21.3)

    python -m py_compile app.py config.py indicators.py portfolio.py sec_research.py sentiment_sources.py   # OK
    python -m pytest -q                                                                                      # 362 passed, 0 failed

Per module: test_diagnostics 25, test_early_setup 33, test_feedback 19, test_indicators 47,
test_integrity 31, test_p0_fixes 26, test_p1_integrity 30, test_portfolio 24,
test_review_examples 18, test_v211_fixes 26, test_v212_fixes 36, test_v213_hype_panel 47
(new). v21.2 had 315; +47.

Real-Streamlit tests (the panel in both languages across every state, and the real hero
button clicked with the toggle off and on) run in a clean subprocess, because the other
test modules replace `sys.modules["streamlit"]` with a stub that would otherwise break them.

Mutation checks: fourteen defects were reintroduced one at a time, each made the suite
fail: hero button hard-coded on; `run_engine` default on; Sell Scanner ignoring the
toggle; toggle default on; sidebar not publishing its state (this one was NOT caught at
first, and a test was added); Stocktwits score shown as a percentage; label rendered as
markdown; missing score becoming 50; the panel fetching while rendering; a poll claimed as
"used" without being used; the Stocktwits card hidden when not applicable; the restored
context removed; the wrong kanji; the version not bumped.

Real Chromium (installed in the test environment), against the real Deep Dive page with
the app's CSS applied, at 390px and 1280px: no horizontal overflow, no element past the
right edge, no clipped text; on desktop the three summary metrics sit side by side with the
two cards beside each other; on mobile everything stacks and reads cleanly (screenshots
inspected). Only this panel was checked, not every page.

## Still NOT verified

Yahoo! Japan's live page (the parser stays experimental), live Stocktwits access,
Postgres/Supabase, other pages in a browser. The patch's own test file and Chromium
checks could not be reproduced from the package. Unit-test success says nothing about
investment returns.


---

# v21.2 (original notes, with a correction marked [v21.3])

Config schema 23. Production model `composite-v5-jp-poll-shared-validity-optin`.
Early Setup model unchanged (`early-setup-v6-price-aligned-volume-confirmation`).
Japan board parser `jp-board-v3-dom-scoped-shared-validity`.

Source: a second independent review of v21.1. All three gaps were reproduced on the
published v21.1 code first. Two v21.1 release-note statements were inaccurate and are
marked [v21.2] below.

## Deploy

Replace the same four files as v21.1: `app.py`, `config.py`, `indicators.py`,
`sentiment_sources.py`. **No database migration** (schema statements identical to v21.1).
Expect the sidebar badge `config v23 · app expects v23`.

## Intentional behaviour changes (disclosed)

| Change | Effect |
|---|---|
| **Japan poll now OFF by default** (sidebar toggle labelled *experimental*) | Following the reviewer's recommendation: the page parser is structural and unverified against Yahoo's live page, and it feeds the production hype score. Japanese hype is volume-only unless you switch it on. |
| [v21.3] **The v21.2 statement above was WRONG for three paths.** Only the sidebar toggle's default had changed. The first-run quick-scan button hard-coded the poll on, `run_engine`'s own default was still on, and the Sell Scanner fetched the board unconditionally and scored it. Fixed in v21.3. | |
| Production model v4 -> v5 | With the poll on: an all-様子見 poll now scores neutral 50 (it was dropped); impossible displays such as 100%+2% are rejected (they were shown, then dropped); scopes end at HTML structure. Model Lab excludes v4 rows (kept, counted). |

Unchanged: thresholds, weights, frozen legacy learning, Stocktwits measurement-only, every
non-Japan score. No change to Early Setup.

## Fixes

1. **"Accepted ⇒ scorable" now holds by construction.** One rule,
   `indicators.jp_poll_problem`, is called by the parser before it returns `ok` AND by both
   scorers. A five-category poll that is entirely 様子見 is a real, neutral poll (score 50);
   only 0%/0% with nothing else shown is `no_votes` (status unavailable, labelled, not
   scored). The neutral share now travels with the poll as a third tuple element so the
   scorers can tell those apart. The parser also rejects displays that cannot come from real
   shares summing to 100 even after rounding (100% + 2% -> `impossible_display`).
   Deterministic tests: 12 named boundary polls and a fixed 7,776-poll grid of corner values
   (0/1/2/50/99/100); an ad-hoc boundary grid of 35,170 accepted polls found no
   counterexample.
2. **Scopes are cut at HTML structure.** The scope from a 「みんなの評価」 heading now ends at
   the next heading, section, article, nav, footer or aside on the RAW html (headings were
   previously stripped before cutting). `<head>`, `<script>` and `<style>` are removed first.
   The reviewer's structure (empty poll section followed by a 掲示板投稿 section with
   numbers) now returns `unavailable`. **Limitation:** a page built only from plain `<div>`s
   with no headings or sections could still let a scope run on — this is not DOM-verified
   isolation, which is why the poll is now off by default.
3. **Model Lab hidden with no recommendations.** Shown on that path too (live only).

### Found while testing the fixes

4. **Demo mode with the demo file missing crashed the Audit page** ("no such table").
   It now drops back to the live view.

## Validation (executed for v21.2)

    python -m py_compile app.py config.py indicators.py portfolio.py sec_research.py sentiment_sources.py   # OK
    python -m pytest -q                                                                                      # 315 passed, 0 failed

Per module: test_diagnostics 25, test_early_setup 33, test_feedback 19, test_indicators 47,
test_integrity 31, test_p0_fixes 26, test_p1_integrity 30, test_portfolio 24,
test_review_examples 18, test_v211_fixes 26, test_v212_fixes 36 (new). v21.1 had 279; +36.

Mutation checks — seven fixes reverted one at a time, each made the suite fail (1, 3, 1,
2, 2, 1 and 1 failing tests), then restored.

Real Streamlit (AppTest): startup with no exceptions, badge `config v23`; the real sidebar
toggle reads "Yahoo!掲示板 sentiment (Japan, experimental)" and defaults to off; the Audit
page with zero recommendations shows its message and the Model Lab, no exceptions.

Note for reviewers: the test counts above can be reproduced from
`alpha_quant_v21_2_TESTS_AND_REVIEW.zip` placed next to the runtime files.

## Still NOT verified

Yahoo! Japan's live page (no permitted fixture), live Stocktwits access, Postgres/Supabase,
browser appearance. Unit-test success says nothing about investment returns.


---

# v21.1 (original notes, with corrections marked [v21.2])

Config schema 22. Production model `composite-v4-jp-poll-scope-and-tolerance`.
Early Setup model unchanged (`early-setup-v6-price-aligned-volume-confirmation`).

Source: an independent review of the v21 build. Every finding was reproduced on the
published v21 code before anything was changed (all six reproduced), then fixed with a
test that fails on the old behaviour.

## Deploy

Replace these four files: `app.py`, `config.py`, `indicators.py`, `sentiment_sources.py`.
`portfolio.py`, `sec_research.py`, `requirements.txt` and `.streamlit/` are unchanged from
v21. **No database migration**: v21.1 adds no tables or columns (verified by diffing the
schema statements, and by opening a database written by the published v21 build).
Keep the four files together; mixing them with v21 copies will show the stale-config
warning or import errors.

Rollback: restore the v21 files. Nothing in the database needs reverting.

## Intentional behaviour changes (disclosed)

| Change | Effect |
|---|---|
| Production model v3 -> v4 | The hype input changes for some Japanese names: (a) polls whose shown percentages total 100-102.5% are now SCORED (they were displayed but silently skipped); (b) a page with no みんなの評価 heading no longer contributes a number. Model Lab excludes v3 rows (kept, counted). |
| Audit headline = verified outcomes only | "Accuracy" now counts only outcomes judged against a benchmark. Outcomes written before the basis was recorded are listed separately. **For an existing database the verified row starts small or empty and grows as new outcomes mature** — that is correct, not a fault. |

Unchanged: BUY >= 65, SELL < 45, factor weights, ENABLE_LEGACY_WEIGHT_LEARNING = False,
relative momentum, movers, expectations wiring, demo isolation, Stocktwits measurement-only.
Early Setup scores, states, components and confirmation are identical to v21: 600 random
price/volume series (including stale, gappy, zero and absent volume) were compared
between the published v21 detector and v21.1, with zero differences.

## Fixes

1. **Parser accepted polls the scorers skipped.** The parser tolerated up to 102.5% total;
   both scorers rejected anything over 100%, so a poll could be shown on screen and then
   contribute nothing. One shared constant, `indicators.JP_POLL_ROUNDING_TOLERANCE`, now
   governs parser and both scorers. Raw values are shown as received (not renormalised);
   scoring uses the net bull-minus-bear spread. [v21.2: the claim below was too strong —
   the property test excluded 0/0 and its random values never reached 100+2, so an
   all-様子見 poll and an impossible 100%+2% display were still accepted but unscorable.
   Fixed in v21.2.] A property check found zero
   accepted-but-unscorable polls (20,000 random polls in an ad-hoc run; 5,000, seeded, in
   the permanent test).
2. **Japan adapter did not isolate the poll.** Parsing is now confined to the text after
   a 「みんなの評価」 heading, cut at the next heading. [v21.2: INACCURATE — it was cut only
   at the next 「みんなの評価」 or 500 characters, because HTML headings are stripped before
   parsing; a following section's numbers could be read. Fixed in v21.2.] No heading -> `unavailable`
   (reason `poll_heading_not_found`, shown in the Deep Dive). A valid component next to a
   malformed one -> `ambiguous`. Look-alike labels outside the scope are ignored.
   **Still unverified against Yahoo's live page**: if the real page does not match this
   structure, every Japanese poll will report `unavailable` with that reason (fail-safe,
   not wrong numbers). The UI wording "now read correctly" was removed.
3. **Duplicate volume dates crashed Early Setup**, and the confirmation-volume flag only
   checked for missing values, so zero volume still reported "available". Volume validity
   is now decided once (`indicators.aligned_volume_window`) and reused for confirmation
   and the quality flag. Duplicated dates -> not usable, no crash.
4. **Stocktwits candidates were labelled "production" in the Model Lab.** Kind is now
   derived (production = the six factors; everything else = candidate) and names come
   from one table, with a test that runs the real candidate attachment and checks every
   key it writes is classified and translated in English and Japanese.
5. **Accuracy included outcomes of unknown provenance.** Split into verified
   (benchmark-relative), absolute-return, and legacy/unknown. The legacy group may include
   simulated rows an older demo seed wrote into the live table; nothing is deleted.
6. **Demo initialisation touched the live database.** `_init_schema(demo=True)` checked
   and wrote the LIVE weights table; on a fresh install it created a live database file
   and then failed. Now fully self-contained.

### Found while verifying the fixes (not in the review)

7. **The Model Lab was hidden whenever the mock portfolio was empty.** It reads only the
   observations ledger but sat below the mock-portfolio section's early `return`, so a
   fresh database (or no picks yet) never showed it. It is now its own function, shown on
   both paths, and appears exactly once when the portfolio has rows.

## Validation (executed for v21.1)

    python -m py_compile app.py config.py indicators.py portfolio.py sec_research.py sentiment_sources.py   # OK
    python -m pytest -q                                                                                      # 279 passed, 0 failed

Per module: test_diagnostics 25, test_early_setup 33, test_feedback 19, test_indicators 47,
test_integrity 31, test_p0_fixes 26, test_p1_integrity 30, test_portfolio 24,
test_review_examples 18, test_v211_fixes 26 (new). v21 had 253; +26.

Mutation checks: eight mutations (finding 2 has two; the hidden Model Lab is the eighth)
were reintroduced one at a time into the fixed code, each confirmed to make the suite
fail (3, 1, 1, 3, 3, 5, 2 and 1 failing tests respectively), then reverted.

Real Streamlit (AppTest, yfinance installed): app startup with no exceptions, badge
`config v22 · app expects v22`. The Audit page and Model Lab were rendered against a
seeded database: verified accuracy 75% from 4 outcomes with 40 legacy rows (all wins,
as the old seeder produced) listed separately and not counted; Stocktwits rows shown as
"candidate" with readable names; the Model Lab shown once with and without mock picks.

Upgrade: a database written by the published v21 build (5 observations, 3 recommendations
with mixed provenance) opened with v21.1; row counts identical; v3 observations kept and
excluded from v4 statistics.

## Still NOT verified

- **Postgres/Supabase** (no server available), **Yahoo! Japan's live page**, **live
  Stocktwits access**, and how pages look in a real browser (AppTest is not a browser).
- Unit-test success says nothing about investment returns.


---

# v21 (original release notes, with corrections marked [v21.1])

Baseline: the six files hashed in BASELINE_SHA256.json (config v20,
composite-v2-relative-momentum, early-setup-v5). All six matched before any edit.

## Deploy

Deploy **all** runtime files together (the import graph now needs six modules):

    app.py  config.py  indicators.py  portfolio.py  sec_research.py  sentiment_sources.py
    requirements.txt  .streamlit/config.toml

`sentiment_sources.py` is NEW. If it is missing the app still starts, but Japan board
parsing and Stocktwits report "unavailable".

Back up `stock_engine.db` (or your Postgres database) before the first run. Migrations
are additive and run automatically on startup; nothing is deleted.

Stocktwits: copy `.streamlit/secrets.toml.example` and add APPROVED API credentials.
Without them no request is made.

## Rollback

Restore the previous six files and the database backup. The added columns and tables
are ignored by the old version, so even without a database restore the old app runs —
but rows written by v21 would then carry v21 fields the old app does not read.

## Intentional model / behaviour changes (disclosed)

| Change | Effect |
|---|---|
| Production model `composite-v2-relative-momentum` -> `composite-v3-jp-board-parser-fix` | Japanese names whose board poll was previously mis-read now feed the CORRECT poll into the hype factor (formula unchanged: 50/50 with volume hype). Composites for those names change. Model Lab excludes v2 rows (kept, counted). |
| Early Setup `v5` -> `early-setup-v6-price-aligned-volume-confirmation` | Stale or gappy volume can no longer confirm a setup. Old rows excluded from current-version stats, not deleted. |
| Unmapped listings no longer benchmark against SPY | 67 names on .AS .CO .HE .MC .MI .OL .ST .SW now have NO benchmark: relative momentum is neutral for them, forward excess returns stay NULL, legacy outcomes are labelled `absolute`. Previously they were measured against a USD ETF. |
| Demo data isolated | The "Seed Simulated Audit Dataset" button writes ONLY to `stock_engine_demo.db`. |

Unchanged: BUY >= 65, SELL < 45, factor weights, ENABLE_LEGACY_WEIGHT_LEARNING = False,
relative momentum, movers, expectations wiring, legacy weekend compatibility.

## Fixes

### P0
1. **Japan board parser** (new `sentiment_sources.parse_jp_board`). Reproduced before
   fixing: `31%買 売67%` -> (67, 67); decimals 33.33/0/0/0/66.67 -> (0, 0); and a
   fourth case, five categories 10/21/2/30/37 (67% sell) -> (10, 30). Now: isolated
   component, five categories with decimals, buy = strong + buy, sell = sell + strong,
   duplicates must agree, mixed layouts rejected, status instead of a guessed number.
   **Not verified against a live Yahoo DOM** (no permitted fixture was available).
   [v21.1] The v21 wording "isolated component" overstated the implementation: it did
   not require the poll heading and ignored a malformed duplicate. Fixed in v21.1.
2. **Stale volume** could set volume_confirmed=True while volume_available=False.
   Confirmation now reads volume on the price calendar (25 sessions, all valid).
3. **Demo data** was written into live tables and called `save_weights()` five times on
   the live database. Now a separate file, explicit `demo=True` routing, DEMO banner,
   exit/clear actions. Older contamination is DETECTED (weights exactly, by their note;
   recommendations by the old seeder's fingerprint) and reported, never deleted; an
   explicit button resets weights to defaults while keeping history.

### P1
4. Benchmarks: unmapped suffix -> unavailable (above). Five call sites would have
   crashed (`sorted()` over a set mixing None and str) and were fixed.
5. Explanations use the weights that PRODUCED the score (`weights_snapshot` on every
   result); legacy results are labelled approximate.
6. Level refresh no longer overwrites scan-time price/levels; refreshed levels are
   stored separately with their own as-of and fetch time.
7. Mock picks are dated by `price_date` (legacy timestamp fallback) in the learner and
   in the Audit join, which previously dropped levels for weekend / pre-open picks.
8. Model Lab quintiles are formed within (scan date x region) cohorts, equal-weighted;
   thin and tied cohorts are excluded and counted. Pooled quintiles let market regime
   pose as stock selection.
9. Time provenance: `scored_at` (UTC) on every observation; retrieval times on
   expectations, Japan board and Stocktwits; new append-only `observation_snapshots`
   log (observations remains the daily convenience view used by movers / NEW flags).

### Stocktwits (measurement only)
Approved API only, credentials from secrets, no request without them, 30-minute cache
that never stores transient failures, circuit breaker on 401/403/429, per-scan budget,
listing-based routing (SHEL.L is never mapped to SHEL), score stored as the provider's
0-100 value + label. Never affects composite, call, weights or Early Setup.

### Tests
Two date-dependent fixtures made deterministic (`test_feedback.py`). One failed whenever
"today - 100 days" fell on a weekend; the other passed vacuously on those days. Two
scan tests patched `effective_universe` (a no-op) instead of `TICKER_UNIVERSE`.

## Migrations (additive, idempotent)

- `observations`: + `retail_sentiment` TEXT, + `scored_at` TEXT
- `recommendations`: + `outcome_basis` TEXT
- new table `observation_snapshots` (+ index `idx_obssnap_ticker`)
- existing `scan_snapshots` table untouched (name collision found and avoided)

Upgrade test: a database produced by the v20 code, including the OLD demo seeder's
contamination (43 recommendations, demo weights active), upgraded twice. Row counts and
sampled values identical; new columns/tables present; contamination reported as
5 weight rows / active weights demo / 40 suspect recommendations (the 3 real ones not
flagged).

## Validation (executed in this release)

    python -m py_compile app.py config.py indicators.py portfolio.py sec_research.py sentiment_sources.py   # OK
    python -m pytest -q                                                                                      # 253 passed, 0 failed, 0 skipped

Per module: test_diagnostics 25, test_early_setup 33, test_feedback 19,
test_indicators 47, test_integrity 31, test_p0_fixes 26 (new), test_p1_integrity 30 (new),
test_portfolio 24, test_review_examples 18.

Baseline before edits: 197 collected, 196 passed, 1 failed (the date-dependent fixture).

Mutation checks — each bug restored, its test confirmed to FAIL, then reverted:
old JP parser (4 fail), raw stale-volume tail (1), demo writes live weights (1),
unmapped suffix -> SPY (2), explanation uses current weights (1), level refresh mutates
price (1), Stocktwits score -> buy % (2), missing Stocktwits -> 50 (1),
pooled quintiles (2).

Real Streamlit (streamlit.testing AppTest, yfinance installed): startup with no
exceptions, badge `config v21 · app expects v21`; clicking the real seed button wrote
40 + 6 rows to the demo database and left the live database byte-identical in counts
and active weights.

## NOT verified / limitations

- **Postgres**: not exercised (no server available). SQL is written to the existing
  dialect rules; treat the Postgres path as untested for this release.
- **Live Yahoo! Japan DOM**: no permitted fixture; the adapter is structural.
- **Live Stocktwits**: no approved credentials; authentication, entitlement, response
  completeness and the 24h window's availability are unverified.
- **AppTest is not a browser**: rendering in a real browser was not visually checked.
- Unit-test success says nothing about investment returns.

## Handoff items deliberately NOT done in this release

- Local benchmarks for the eight unmapped European suffixes (symbols unverifiable here).
- Exchange-local timezone normalisation before joins; intraday vs completed-bar
  separation (P1-5 second half).
- Executable-entry backtest after `scored_at` (P1-6); `evaluate_outcomes_only()` still
  uses calendar-day horizons and is labelled legacy.
- Deep Scan input consistency (P12), most UI items (P13: currency-aware `fmt_money`,
  Top Selections BUY filter, "Weekly Growth" rename, HTML escaping audit, per-stock feed
  health), research items (P14) and hygiene items (P15: duplicate definitions, MACD-zero
  policy, variable-window long momentum, trade-level positivity, insider date filter,
  SQLite connection closing).
- Percent-before-label FIVE-category layout (needs a verified fixture).
