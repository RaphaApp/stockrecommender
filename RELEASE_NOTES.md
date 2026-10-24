# Alpha Quant Engine — v21.6 (links to the sources' public pages)

Config schema 27. Production model and Japan parser unchanged
(`composite-v8-jp-visible-detail`, `jp-board-v5-visible-detail-safe-summary`).

## What changed

In the Chart Deep Dive's Hype panel, each source card now has a button that opens that
source's public page in your browser, directly under the card's status line:

* **US stocks:** "Open on Stocktwits ↗" (`stocktwits.com/symbol/AAPL`). Share classes use the
  existing mapping (BRK-B -> BRK.B, BF-B -> BF.B); a US-listed ADR such as BABA links to its own
  page; a foreign listing such as SHEL.L is never sent to its ADR's page.
* **Tokyo stocks:** "Open Yahoo!掲示板 ↗" (`finance.yahoo.co.jp/quote/7203.T/forum`), shown even
  when the app's own fetch was blocked, so you can still read the page yourself.
* Anything else (Europe, Hong Kong, indices, malformed input) gets no button.

They are **links only**: building and rendering them makes no request, and nothing on those
pages enters the app, the ledger or any score. Each button carries that sentence as its
tooltip. Links open in a new tab with `rel=noreferrer`. English and Japanese labels.

Found while testing: the existing "unsuffixed means US listing" rule would have produced
Stocktwits links for strings such as `7203` or `AA PL`. The link therefore also requires a
real US ticker shape (1-5 letters, optional class letter); none of the 216 tickers in your US
universes loses its link. Measurement routing is unchanged.

## Deploy

Replace **`app.py` and `config.py`** (a two-file update over v21.5-reviewed). No database
migration. Badge: `config v27 · app expects v27`. Rollback: the v21.5-reviewed `app.py` and
`config.py` (or your original five working files).

## Validation

    python -m pytest -q          # 575 passed, 0 failed  (test_v216_links.py: 31 new)

Five mutations (buttons removed, shape check removed, raw ticker used, any ".T" accepted,
tooltip dropped) each made the suite fail. Real Streamlit: the panel in English and Japanese
for US, share-class, Tokyo, blocked-Tokyo and UK listings, with all network functions
booby-trapped. Real Chromium at 390 px: correct URLs, new tab, `noreferrer`, no horizontal
overflow (screenshots inspected). The link format was checked against the URL the code
builds, not by loading Stocktwits or Yahoo! (no network here).


---

# v21.5 Japan Fetch patch, REVIEWED (previous notes)

Config schema 26 · production model `composite-v8-jp-visible-detail` · Japan parser
`jp-board-v5-visible-detail-safe-summary`. **All three identifiers are unchanged from the patch
you ran**, and `config.py` is byte-identical to the one supplied.

## Verdict

The working implementation passes review and is **kept**. Nothing was redesigned. The request
it makes is unchanged (same URL, headers, verified TLS, timeout). Your report that fetching
works is treated as **user-confirmed in your environment**; I could not reach Yahoo! from
here, so I did not and cannot reproduce it. What I changed is limited to defects the review
found, each with a test that fails on the supplied code.

## 1. Your working files, recorded before any edit

SHA-256 of the five files exactly as supplied (also in `WORKING_FILES_SHA256.txt`):

    app_d95c3e.py                 2907016f4c11a5c3ee7dd8ed5eb3d61f678297497aca66d6e1ea6fa2e6ace9e7
    config_012956.py              38542698a08e28b4c5d75cdfaf7f1922347286060388bb6232e4cee73b0e0be9
    sentiment_sources_9963bb.py   843b0e855fe390c0c0731b6cb2250d02a06f92ecd857b9a676c08b1722ae80a1
    jp_board_diagnose.py          2040507c43c1d240b1720a4dcfb154d13eac172146c288ae9168a35d2c26b43b
    jp_board_transport.py         36dd35739e7b2188c43c9c41cc73c3825d6a87acdee780d981df2c8ebe4d5682

**Keep those five files**: they are your known-working set and the rollback target.

**What I could NOT record, because only you saw it:** the ticker, the displayed percentages,
the parser status and the retrieval timestamp of your successful run, and a raw-page fixture.
I will not invent them. Capture them now (about a minute), from the app folder:

    python jp_board_diagnose.py 7203.T --record run.json --poll-html poll.html

`run.json` holds ticker, UTC retrieval time, HTTP status, size, parser version, status, the
percentages and which source produced them. `poll.html` holds ONLY the poll block (not the
page), written with a round-trip check ("the fixture reproduces the original result") and a
warning if it looks like it contains a secret or an e-mail address. It carries no cookies (the
request sends none). Read both files before sharing them; with them I can turn your real,
working layout into a permanent regression fixture.

## 2. Findings, in order of importance

**F1 — quadratic slowdown (measured, fixed).** The accessibility adapter copied everything
before every tag to see whether that tag had an `aria-label`. A 1.3 MB page (60,000 tags)
took **21.9 s to parse** (0.03 s before the patch); it grows with the square of the page size
(minutes at the 5 MB cap), per ticker, 16 Japanese tickers per scan. Fixed with a linear
version whose output is **identical**: differential-tested against the supplied
implementation on 6,000 random pages (zero differences; the adapter fired on ~58% of them),
and a 1,500-page version of that comparison is a permanent test. Now 0.07 s.

**F2 — pathological markup could hang the parser (found in review; predates the patch, in
my earlier code).** 0.4 MB of unclosed `<script>`/`<style>` tags took >20 s; 100 KB of `<`
took 4 s; there is no timeout around parsing, and the new saved-page upload makes a truncated
file easy to feed in. A linear guard (`markup_problem`) now refuses grossly unbalanced markup
instantly with a visible reason (`malformed_html_unclosed_blocks` / `malformed_html_angle_flood`).
Ordinary pages never trip it (a script-heavy page with 300 scripts and 8,000 `<` comparisons
is a test).

**F3 — failure reporting and blocking.** Transport failures reached the UI as a bare class
name ("network_error (ValueError)"), because the patch also stopped logging the message.
They now carry a stable reason: `tls_verification_failed`, `timeout`, `unexpected_redirect`,
`page_too_large`, `ca_bundle_unreadable`, `jp_board_transport_missing`, `http_403`, ... And a
**403/429 was re-requested for every Japanese stock in a scan**, contrary to the app's own
"stop on blocking" rule: after a 403 or 429 the app now stops requesting for 30 minutes (per
browser session; a proposed policy, not a Yahoo! rule), reporting `circuit_open_http_403`
instead, and never tries to get around it. A TLS/network failure now also shows the
"run jp_board_diagnose.py" hint (it prints the CA-bundle advice).

**F4 — a limitation I did NOT change: nested visible vs accessible disagreement.** If the
visible numbers sit *inside* the aria-labelled element, the adapter's injected heading ends
the visible scope, so the scorer reads the accessible label and never sees different visible
numbers (the sibling layout correctly refuses them as ambiguous). I left scoring alone on
purpose: your live success may depend on this exact scope behaviour and I cannot see your
page. The diagnostic now exposes it: it reports the poll source (visible / accessible /
both) and prints a WARNING if the visible and accessible readings differ. Pinned by tests.

**F5 — behaviour differences from my v21.4 (deliberate, fail-visible).** A one-sided summary
next to an ordered breakdown is now refused (`unresolved_numeric_poll_labels`); a summary
given sell-first is not a recognised layout and fails visibly. Where arithmetic fits two
different readings, the result is `ambiguous_category_summary`. Correction to the handoff: my
v21.4 did not "choose the first occurrence" in general; it preferred the more precisely
displayed number and used position only for equal precision. The new rule is stricter and
easier to defend. In practice an ordered five-label run (your layout) never reaches the
arithmetic fallback, so the ambiguity rule applies only to non-ordered layouts.

**F6 — security behaviour change to know about.** The Japan transport ignores
`STOCKREC_INSECURE_SSL` (verification always on). If you ever relied on that switch for the
Japan board behind a corporate proxy, set `STOCKREC_CA_BUNDLE` to your trusted CA file
instead. Verified by a test that sets the switch and checks the context still requires and
checks certificates.

**F7 — diagnostic double counting fixed** (one occurrence inside a script counted twice
after the decoded-JSON text was appended).

**F8 — saved-page upload verified, not changed.** It parses markup only, writes nothing to the
ledger, changes no score and no weight, does not touch the scan result, is a separate widget
per ticker, refuses files over 5 MB, and reports malformed pages instead of hanging (tests,
including a deep-equality check of the result and row counts in five ledger tables). Streamlit
itself labels the uploader "200MB per file"; the app's own 5 MB limit applies after upload.

## 3. The handoff's verification checklist

| Item | Result |
|---|---|
| Fetch -> parser -> shared validity -> Hype -> ledger | Run end to end through the **real transport** with a fake HTTP layer (`test_the_whole_chain_...`). Not live. |
| Precise values and summary provenance survive persistence | Yes, in `observations` and in the append-only `observation_snapshots`, with source URL, HTTP status, retrieval time, transport and page size. |
| Toggle off by default, controls every scan path | Yes: `run_engine` default, hero button, Sell Scanner (earlier tests) and a new combined test. |
| Deep Scan does not request the board | Pinned by a test. |
| Same-bar rescans keep movers and NEW | Existing `test_integrity` tests, all passing. |
| Missing/blocked never zero or neutral | A real scan with a 403: poll NaN, hype and composite equal the poll-off run. |
| Saved-page imports cannot mutate scores or the ledger | Yes (F8). |
| Existing observations stored; model filtering correct | v7 rows kept, excluded from v8 statistics (test, plus a v21.4 database upgrade). |
| Stocktwits unchanged | AST-identical code; all earlier Stocktwits and credential-leak tests pass. |
| Weights, thresholds, relative momentum, legacy freeze | Unchanged and pinned by value; `indicators.py` is byte-identical. |

Regression cases requested, all present: summary before/after the breakdown, all three
headline layouts (and five separators), contradictory and duplicated summaries, equal-valued
and different-valued ambiguous readings, visible + accessible duplicates (agreeing, disagreeing,
nested), unrelated percentages in adjacent sections, HTTP 403/429/500/503, TLS failure,
timeout, redirect, oversize, bad ticker, bad charset, restored snapshots and both UI languages.

**Wording to use:** weights and thresholds are unchanged; **a recommendation can change when
an enabled Japan poll newly parses and moves Hype.** It is not "recommendations unchanged".

## 4. Deploy and rollback

Replace `app.py`, `sentiment_sources.py`, `jp_board_transport.py`, `jp_board_diagnose.py`
(`config.py` is unchanged, so it needs no replacing). `indicators.py`, `portfolio.py`,
`sec_research.py`, `requirements.txt` and `.streamlit/` are unchanged. **No database migration**
(schema statements identical; a v21.4 database opened and every row preserved). Badge:
`config v26 · app expects v26`. **Rollback:** put back your five supplied files; nothing in the
database needs reverting.

## 5. Validation executed

    python -m py_compile app.py config.py indicators.py portfolio.py sec_research.py sentiment_sources.py jp_board_transport.py jp_board_diagnose.py   # OK
    python -m pytest -q                                                                                                                               # 544 passed, 0 failed

Per module: test_diagnostics 25, test_early_setup 33, test_feedback 19, test_indicators 47,
test_integrity 31, test_p0_fixes 26, test_p1_integrity 30, test_portfolio 24,
test_review_examples 18, test_v211_fixes 26, test_v212_fixes 36, test_v213_hype_panel 47,
test_v214_jp_summary 64, **test_v215_review 118 (new)**. v21.4 had 426.

**Baseline failures when your five files were first laid over the full suite: 11 of 426.**
Seven tests mocked the old fetch function the patch replaced (they reached the real network
and failed; now they run the real transport through a fake HTTP layer); two pinned the old
model/parser versions; one pinned the old one-sided-summary behaviour; one was a stale
diagnostic test that passed by accident. None was a regression in the patch's scoring.

Mutation checks: 18 defects reintroduced one at a time, each made the suite fail (including
the supplied quadratic adapter: the large-page test fails with `21.5 < 2.0`; and a mutation
that sums script counts, a redirect not refused, insecure SSL honoured, no size cap, TLS
reported as a generic error, no circuit breaker, class-name-only reasons, no hint on network
errors, position-based ambiguity, a saved-page import attached to the result, no oversize
guard, a fixture holding the whole page, an over-eager malformed-markup guard, a model
version not bumped, and a disagreement flag that never fires). One first attempt was a bad
mutation (a NameError, not the speed test) and was redone.

Real Streamlit: app starts with no exceptions, badge `config v26`, the Japan toggle off by
default; the Deep Dive panel is rendered in English and Japanese for every status, including
restored snapshots from older builds. Real Chromium at 390 px and 1280 px (screenshots
inspected): the failure state, the diagnostic hint and the saved-page control render without
horizontal overflow.

## 6. NOT verified

* Live Yahoo! access beyond what you confirmed: other tickers, other days, other networks or
  hosts (a cloud host may be blocked; Copilot's own request received HTTP 403).
* Which layout your live page uses (visible, accessible label, or both) until you run the
  diagnostic above.
* Stocktwits provider access (pending), Postgres/Supabase, and how other pages look in a
  browser. No RapidAPI endpoint exists anywhere in the code and none was added.
* Unit-test success says nothing about investment returns.


---

# v21.4 (earlier release notes)

Config schema 25. Production model `composite-v7-jp-summary-crosscheck`.
Japan parser `jp-board-v4-summary-crosscheck`. Early Setup model unchanged.

## Baseline (verified before editing)

The files attached with the handoff were the *same Copilot patch you sent earlier*,
byte-for-byte (`app_1.py` == the earlier `app.py`, and so on). Its `app.py` is older than
my v21.3 (it still has model v5, the 掛示板 typo and no restored Deep Dive context);
`config.py`, `indicators.py` and `sentiment_sources.py` equal my v21.3. So v21.3 was used as
the baseline: it already contains everything in that patch. The handoff's section 3 (opt-in
fixes) is present and tested: first-run button, `run_engine` default, Sell Scanner, and the
stored `jp_forum_enabled`.

## Deploy

Replace **`app.py`, `config.py`, `sentiment_sources.py`** and add **`jp_board_diagnose.py`**
(a stand-alone script, not imported by the app). `indicators.py`, `portfolio.py`,
`sec_research.py`, `requirements.txt`, `.streamlit/` are unchanged from v21.3. No database
migration (schema statements identical; v21.3 database opened and preserved). Badge:
`config v25 · app expects v25`. Rollback: restore the v21.3 files.

## The reported bug: a rounded headline summary broke the parser

Reproduced with the handoff's numbers on v21.3: strong buy 62.96 / buy 9.26 / watch 7.41 /
sell 3.70 / strong sell 16.67, plus a summary "買いたい 72% / 売りたい 20%". The breakdown
alone read fine; with the summary, in either order, the result was
`parse_error: duplicate_label`, because the summary reuses the plain buy/sell labels.

Fix: strong-buy, watch and strong-sell must each appear exactly once; the plain buy and sell
labels may appear once (breakdown only) or twice (breakdown plus summary). Which of the two
numbers is the category is decided by **arithmetic, not position**: the summary must equal
strong + category within display rounding (62.96 + 9.26 = 72.22 -> "72"). The score uses the
**precise category sums** (72.22 / 20.37), never the rounded summary; nothing is counted
twice; the summary is kept as provenance (`summary_buy_pct`, `summary_sell_pct`) in the
result and the ledger. A summary that contradicts the breakdown is rejected
(`summary_conflicts_with_categories`), never averaged. A whole component repeated in one
scope, or a strong-buy/watch/strong-sell shown twice, is still rejected as `duplicate_label`.
The allowed gap is half-unit rounding plus 0.5 (a summary built from separately rounded
parts): 1.01 against 72.22, so 73 is accepted and 71 is not. This tolerance is my
engineering choice, not a documented Yahoo! rule.

## What this does NOT establish (important)

**Nothing here proves the live page is readable.** Every test uses fixtures built from the
handoff's description, and this environment cannot reach Yahoo!. The summary format is the
reported one; other layouts (for example a percent shown before its label) are not handled
and fail visibly. There is a second plausible reason the poll "returns no data" that a fixture
cannot settle: Yahoo! Japan Finance is largely script-rendered, and this parser
deliberately does not read `<script>` data. If the poll numbers exist only as embedded JSON,
no summary fix will help and a different, verified adapter would be needed.

## New: jp_board_diagnose.py — find out which case you are in

Run from the app folder on your PC (where the page is reachable). It uses the app's own
request settings, including `STOCKREC_CA_BUNDLE`:

    python jp_board_diagnose.py 7203.T --save page.html     # fetch, explain, keep the HTML
    python jp_board_diagnose.py --file page.html            # analyse a saved page

It prints a plain verdict, then the details: READABLE (with the numbers and any summary
found); heading missing (blocking, a login/consent page, or a layout change); poll wording
present **only inside script data**; or heading found but the poll did not validate (with
the reason and the text the parser saw). It never guesses a number, and a fetch failure is
reported as a failure. Personal diagnostic use; check the saved page before sharing it.

The Deep Dive now also tells you when a poll can't be read (but not when nobody has voted):
*"run jp_board_diagnose.py … to see why"*, and its source-details expander shows the rounded
summary when the page had one.

## For your reply: what would settle it fastest

Switch the Japan toggle on, run a **new main scan**, open a Japanese stock in Deep Dive and
read the **Reason** line in the Yahoo! card's details — or run the diagnostic on `7203.T`.
Either tells us whether it is the summary case (now fixed), a missing heading, script-only
data, or something else.

## Do users understand when the Japan poll applies? (handoff section 4)

Now stated in the toggle's help text in both languages: it applies to **main scans and the
Sell Scanner, not Deep Scan**, and takes effect on the **next main scan** after you switch it
on. The Deep Dive card already says "enable the toggle … and run a new scan" when off, and "a
new main scan is needed" when nothing was collected. Deep Scan not fetching the poll is
pinned by a test (it makes no board request), so a future change has to update the help text.

## Stocktwits and security (handoff sections 5 and 6) — no code change, by design

* No RapidAPI integration was written and none is referenced anywhere (a test fails if the
  word appears in the app code). The only supported route stays the documented approved API
  with `STOCKTWITS_API_USERNAME` / `STOCKTWITS_API_PASSWORD`. A RapidAPI key must not be put
  in those fields.
* Open until the provider replies: the endpoint and host, the exact authentication (the form
  also asks for an OAuth client id/secret and access token), whether it returns normalised
  sentiment or only messages, the schema, windows, limits and permitted use. If only tagged
  messages are available, any derived figure must be labelled "bullish share of sampled
  tagged posts", not Stocktwits' own score (untagged posts stay unclassified). Not
  implemented — there is nothing to build against yet.
* New tests prove the existing client never returns or logs the username, password or the
  Basic-auth token on any failure path (HTTP 401/429, network, timeout, malformed body), nor
  does the app path that wraps it. They use obviously fake sentinel strings.
* A key that was pasted into a chat must be treated as compromised. I cannot see your
  RapidAPI account, so I cannot confirm that it was rotated or the old key deleted: please
  confirm that yourself, and never paste keys into chats, code, notes or test files.

## Intentional behaviour change (disclosed)

Production model v6 -> v7. Japanese pages that carry a rounded summary used to contribute no
poll and now do (when the toggle is on), which changes the hype input for those names.
v6 observations are kept but excluded from current-model statistics. With the toggle off
(the default) nothing changes. Weights, thresholds and recommendations are untouched.

## Validation (executed for v21.4)

    python -m py_compile app.py config.py indicators.py portfolio.py sec_research.py sentiment_sources.py jp_board_diagnose.py   # OK
    python -m pytest -q                                                                                                          # 426 passed, 0 failed

Per module: test_diagnostics 25, test_early_setup 33, test_feedback 19, test_indicators 47,
test_integrity 31, test_p0_fixes 26, test_p1_integrity 30, test_portfolio 24,
test_review_examples 18, test_v211_fixes 26, test_v212_fixes 36, test_v213_hype_panel 47,
test_v214_jp_summary 64 (new). v21.3 had 362; +64.

Mutation checks: fourteen defects reintroduced one at a time, each made the suite fail
(repeated plain label rejected again; score from the rounded summary; no cross-check;
precision tie-break flipped; repeated strong/watch ignored; diagnostic ignoring script data;
summary not carried to the ledger; hint shown for an empty poll; version not bumped; Deep
Scan fetching the board; the password logged; Deep Scan missing from the help; summary line
removed from the panel; a RapidAPI reference added).

Covered end to end: a page with a summary through `run_engine`, into the observations ledger
(`retail_sentiment.jp_board` carries the summary, parser version and categories; model
`composite-v7…`), with the poll visible in the hype score through the unchanged 50/50 blend;
a conflicting summary recorded as a failure with no number; the toggle still gating
everything; the Deep Dive card rendered in real Streamlit in English and Japanese.

Real Streamlit start-up: no exceptions, badge `config v25`, toggle defaults to off.
Not re-run: the real-Chromium layout check from v21.3 (the panel gained one text line and
one caption; those are covered by real-Streamlit rendering tests, not by a new browser check).

## Still NOT verified

Yahoo! Japan's live page (see above), live Stocktwits access and the RapidAPI question,
Postgres/Supabase, browsers beyond the v21.3 panel check. Unit-test success says nothing about
investment returns.


---

# v21.3 (previous release notes)

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
