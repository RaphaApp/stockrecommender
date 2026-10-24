"""External crowd-sentiment sources for the Alpha Quant Engine.

Pure parsing/normalisation plus one approved-API network client. No Streamlit
imports, so every function here is testable offline. Two sources, deliberately
kept as DIFFERENT metrics that must never be interchanged:

  * Yahoo! Japan board "みんなの評価" — a posting-sentiment DISTRIBUTION across five
    categories (strong buy / buy / watch / sell / strong sell), in percent.
  * Stocktwits — a provider-NORMALISED 0-100 sentiment score with a provider label.
    A score of 34 is NOT 34% buy, not a probability and not a recommendation.

Adapted from review_examples.py (tested reference helpers supplied with the
handoff). The Japan page adapter is STRUCTURAL and UNVERIFIED against the live Yahoo
markup: no permitted page fixture was available. It reads only text inside a scope
that begins at the 「みんなの評価」 heading, so if Yahoo's page does not match that
assumption the result is status="unavailable" with a specific reason — never a guessed
number. See parse_jp_board().
"""
from __future__ import annotations

import base64
import html as _html
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

# The SAME tolerance the scorers use (indicators.forum_*_score), so a poll accepted
# here can never be silently skipped there.
from indicators import JP_POLL_ROUNDING_TOLERANCE as _TOTAL_TOLERANCE
from indicators import jp_poll_problem

# --------------------------------------------------------------------------
# Yahoo! Japan board poll
# --------------------------------------------------------------------------
JP_PARSER_VERSION = "jp-board-v5-visible-detail-safe-summary"
JP_LABELS = (
    "\u5f37\u304f\u8cb7\u3044\u305f\u3044",   # 強く買いたい  strong buy
    "\u8cb7\u3044\u305f\u3044",               # 買いたい      buy
    "\u69d8\u5b50\u898b",                     # 様子見        watch / neutral
    "\u58f2\u308a\u305f\u3044",               # 売りたい      sell
    "\u5f37\u304f\u58f2\u308a\u305f\u3044",   # 強く売りたい  strong sell
)
_STRONG = (JP_LABELS[0], JP_LABELS[4])
_BUY, _SELL = JP_LABELS[1], JP_LABELS[3]
# The poll's heading ("みんなの評価" = "everyone's evaluation"). Parsing is confined to
# the text that follows it; without the heading there is no poll to read.
JP_POLL_HEADINGS = ("\u307f\u3093\u306a\u306e\u8a55\u4fa1",)
# A poll widget is compact: its categories must all fall within this many characters of
# its heading (or the next heading, whichever is first).
_SCOPE_CHARS = 500
_AGREE_TOLERANCE = 0.51          # duplicate components must agree to rounding


# A page whose markup is grossly unbalanced makes the lazy tag/block regexes below
# quadratic: 0.4 MB of unclosed <script> tags took >20 s, and 100 KB of "<" took 4 s, with no
# timeout around parsing. These checks are linear (str.count) and ordinary pages never
# trip them: real pages have balanced script/style/head blocks and near-equal "<" and ">".
_UNBALANCED_BLOCKS_SLACK = 16
_ANGLE_SURPLUS_LIMIT = 50_000


def markup_problem(raw: str) -> str | None:
    """Reason a page is too malformed to parse safely, or None. Fail-visible, never a guess."""
    if not raw:
        return None
    low = raw.lower()
    for name in ("script", "style", "head"):
        if low.count("<" + name) > low.count("</" + name) + _UNBALANCED_BLOCKS_SLACK:
            return "malformed_html_unclosed_blocks"
    if raw.count("<") - raw.count(">") > _ANGLE_SURPLUS_LIMIT:
        return "malformed_html_angle_flood"
    return None


def html_to_text(raw: str) -> str:
    """Visible text only: drop script/style, strip tags, unescape, collapse spaces."""
    if not raw:
        return ""
    t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def _half_unit(raw: str) -> float:
    """Half of the last displayed digit: '37' -> 0.5, '33.3' -> 0.05, '33.33' -> 0.005."""
    return 0.5 * 10 ** (-len(raw.split(".", 1)[1]) if "." in raw else 0)


# A headline summary ("買いたい 72% / 売りたい 20%") may be shown beside the five-category
# breakdown. It is the rounded sum of strong + plain on each side, so it reuses the plain
# labels. This slack is added on top of the displayed rounding half-units when checking it
# against the categories; it covers a summary built from separately-rounded parts.
_SUMMARY_SLACK = 0.5


class _Hit:
    """One 'LABEL value%' occurrence: the number, its text as displayed, and where."""
    __slots__ = ("value", "raw", "pos")

    def __init__(self, value: float, raw: str, pos: int):
        self.value, self.raw, self.pos = value, raw, pos

    @property
    def decimals(self) -> int:
        return len(self.raw.split(".", 1)[1]) if "." in self.raw else 0


def _resolve_side(strong: "_Hit", hits: list) -> tuple[Any, Any, str]:
    """Split one side's plain-label occurrences into (category, summary, reason).

    One occurrence: it is the category and no summary is shown. Two: one is the category
    and the other the rounded headline summary. Which is which is decided by arithmetic,
    not by position (the summary may come before or after the breakdown): the summary
    must equal strong + category within display rounding. Exactly one reading fitting ->
    use it. Both fitting is possible only when the strong share is ~0 and the two numbers
    nearly coincide; the aggregate is then the same within rounding, and the more
    precisely displayed number is taken as the category. Neither fitting -> the numbers
    contradict each other, which is reported, never averaged away. Three or more -> the
    component is duplicated or malformed.
    """
    if len(hits) == 1:
        return hits[0], None, ""
    if len(hits) != 2:
        return None, None, "duplicate_label"
    fits = []
    for cat, summ in ((hits[0], hits[1]), (hits[1], hits[0])):
        bound = (0.5 * 10 ** -summ.decimals + 0.5 * 10 ** -strong.decimals
                 + 0.5 * 10 ** -cat.decimals + _SUMMARY_SLACK + 1e-9)
        if abs(summ.value - (strong.value + cat.value)) <= bound:
            fits.append((cat, summ))
    if not fits:
        return None, None, "summary_conflicts_with_categories"
    if len(fits) == 1:
        return fits[0][0], fits[0][1], ""
    # Arithmetic cannot identify different readings just because one occurs first.
    if abs(fits[0][0].value - fits[1][0].value) > 1e-9:
        return None, None, "ambiguous_category_summary"
    return fits[0][0], fits[0][1], ""


def _five_category_detail_legacy(scoped_text: str) -> tuple[dict | None, str]:
    """(result, reason) for LABEL-THEN-PERCENT text from ONE isolated component.

    Strong-buy, watch and strong-sell must each appear exactly once. Buy and sell may
    appear once (breakdown only) or twice (breakdown plus a rounded headline summary);
    the summary is cross-checked against the categories and kept as provenance, but the
    SCORE uses the precise category sums, never the rounded summary, and nothing is
    counted twice. Any other repetition means two components or a malformed one, and is
    rejected rather than ignored.
    """
    alternatives = "|".join(re.escape(x) for x in sorted(JP_LABELS, key=len, reverse=True))
    pattern = (rf"(?P<label>{alternatives})\s*[:\uff1a]?\s*"
               rf"(?P<value>\d+(?:\.\d+)?)\s*[%\uff05]")
    hits: dict[str, list] = {lab: [] for lab in JP_LABELS}
    for match in re.finditer(pattern, scoped_text):
        value = float(match["value"])
        if not 0 <= value <= 100:
            return None, "value_out_of_range"
        hits[match["label"]].append(_Hit(value, match["value"], match.start()))
    if any(not h for h in hits.values()):
        return None, "missing_category"
    if any(len(hits[lab]) > 1 for lab in (JP_LABELS[0], JP_LABELS[2], JP_LABELS[4])):
        return None, "duplicate_label"

    buy_cat, buy_sum, why = _resolve_side(hits[JP_LABELS[0]][0], hits[JP_LABELS[1]])
    if why:
        return None, why
    sell_cat, sell_sum, why = _resolve_side(hits[JP_LABELS[4]][0], hits[JP_LABELS[3]])
    if why:
        return None, why

    chosen = {JP_LABELS[0]: hits[JP_LABELS[0]][0], JP_LABELS[1]: buy_cat,
              JP_LABELS[2]: hits[JP_LABELS[2]][0], JP_LABELS[3]: sell_cat,
              JP_LABELS[4]: hits[JP_LABELS[4]][0]}
    values = {k: h.value for k, h in chosen.items()}
    half = {k: 0.5 * 10 ** -h.decimals for k, h in chosen.items()}
    if abs(sum(values.values()) - 100.0) > _TOTAL_TOLERANCE:
        return None, "total_not_100"
    # Could these displayed numbers come from real shares that sum to 100? Even at the
    # lowest value each could round from, 100% + 2% exceeds 100 — not a real poll.
    if sum(max(v - half[k], 0.0) for k, v in values.items()) > 100.0 + 1e-9:
        return None, "impossible_display"
    buy = values[JP_LABELS[0]] + values[JP_LABELS[1]]
    sell = values[JP_LABELS[3]] + values[JP_LABELS[4]]
    neutral = values[JP_LABELS[2]]
    why = jp_poll_problem(buy, sell, neutral)        # the scorers' own rule
    if why is not None:
        return None, why
    return {"buy_pct": buy, "neutral_pct": neutral, "sell_pct": sell, "categories": values,
            "summary_buy_pct": None if buy_sum is None else buy_sum.value,
            "summary_sell_pct": None if sell_sum is None else sell_sum.value}, ""



def _five_category_detail(scoped_text: str) -> tuple[dict | None, str]:
    """Prefer an explicitly ordered five-category run, never infer it by position
    among unrelated duplicate labels. This supports the public page's mixed-order
    headline (75% buy / sell 19%) independently of the precise breakdown.
    No embedded script data is used for scoring.
    """
    num = r"(\d+(?:\.\d+)?)\s*[%\uff05]"
    sep = r"[\s,\u3001;/|\uff0f]*"
    pieces = [re.escape(lab) + r"\s*[:\uff1a]?\s*" + num for lab in JP_LABELS]
    matches = list(re.finditer(sep.join(pieces), scoped_text))
    if not matches:
        return _five_category_detail_legacy(scoped_text)
    if len(matches) != 1:
        return None, "duplicate_label"
    m = matches[0]
    result, why = _five_category_detail_legacy(m.group(0))
    if result is None:
        return None, why
    remaining = scoped_text[:m.start()] + " " + scoped_text[m.end():]
    # Repeated strong/watch labels WITH values indicate another partial component.
    for lab in (JP_LABELS[0], JP_LABELS[2], JP_LABELS[4]):
        if re.search(re.escape(lab) + r"\s*[:\uff1a]?\s*" + num, remaining):
            return None, "duplicate_label"
    # Recognise each summary as one complete pair, not independent overlapping
    # label/value searches (which can bind the same percentage twice).
    b, se = re.escape(_BUY), re.escape(_SELL)
    pair_patterns = [
        b + r"\s*[:\uff1a]?\s*" + num + sep + se + r"\s*[:\uff1a]?\s*" + num,
        num + r"\s*" + b + sep + se + r"\s*[:\uff1a]?\s*" + num,
        num + r"\s*" + b + sep + num + r"\s*" + se,
    ]
    pairs = []
    for pattern in pair_patterns:
        for hit in re.finditer(pattern, remaining):
            item = (hit.start(), hit.end(), hit.group(1), hit.group(2))
            if item not in pairs:
                pairs.append(item)
    if len(pairs) > 1:
        return None, "duplicate_or_ambiguous_summary"
    if pairs:
        start, end, buy_raw, sell_raw = pairs[0]
        for key, raw, labels in (("buy_pct", buy_raw, JP_LABELS[:2]),
                                 ("sell_pct", sell_raw, (JP_LABELS[3], JP_LABELS[4]))):
            val = float(raw)
            if not 0 <= val <= 100:
                return None, "value_out_of_range"
            # Display precision bound plus the existing explicitly disclosed slack.
            bound = _half_unit(raw) + sum(_half_unit(m.group(JP_LABELS.index(l)+1))
                                         for l in labels) + _SUMMARY_SLACK + 1e-9
            if abs(val - result[key]) > bound:
                return None, "summary_conflicts_with_categories"
            result["summary_" + key] = val
        remaining = remaining[:start] + " " + remaining[end:]
    # Do not silently ignore any extra numeric poll label we did not recognise.
    labels = "|".join(re.escape(x) for x in sorted(JP_LABELS, key=len, reverse=True))
    if re.search(r"(?:" + labels + r")\s*[:\uff1a]?\s*" + num, remaining) or re.search(
            num + r"\s*(?:" + labels + r")", remaining):
        return None, "unresolved_numeric_poll_labels"
    return result, ""


_ARIA_LABEL_ATTR = re.compile(r"\baria-label\s*=", re.I)


def _accessible_poll_text(raw: str) -> str:
    """Expose a complete five-category aria-label to the scoped parser.
    Accessible labels are page markup, not scripts. Only full ordered category
    descriptions qualify. They are inserted in-place, never collected page-wide.
    If visible and accessible copies both exist, agreement is checked separately.

    Linear time. An earlier version copied everything before EVERY tag to look for the
    nearest poll heading, which is quadratic: a 1.3 MB page took ~22 s (v21.4 parsing took
    0.03 s), and the 5 MB size cap would have meant minutes per ticker. The result is
    identical — same tags qualify, same text is inserted — which a differential test
    against the previous implementation pins. Only tags that actually carry an
    aria-label are examined; the nearest preceding heading comes from bisecting the
    heading positions, and the first structural boundary after each heading is computed
    once.
    """
    import bisect
    heads = {h: [m.start() for m in re.finditer(re.escape(h), raw)] for h in JP_POLL_HEADINGS}
    if not any(heads.values()):
        return raw                                   # no poll heading: nothing can qualify
    first_boundary: dict[int, int] = {}              # heading start -> first boundary start

    def in_poll_scope(pos: int) -> bool:
        best = -1
        for h, starts in heads.items():
            i = bisect.bisect_right(starts, pos - len(h)) - 1     # last occurrence ending <= pos
            if i >= 0:
                best = max(best, starts[i])
        if best < 0:
            return False
        if best not in first_boundary:
            m = _BOUNDARY.search(raw, best)
            first_boundary[best] = m.start() if m else len(raw) + 1
        return not first_boundary[best] < pos        # a boundary BEFORE the tag ends the scope

    def replace(match):
        tag = match.group(0)
        if not _ARIA_LABEL_ATTR.search(tag) or not in_poll_scope(match.start()):
            return tag
        attr = re.search(r'\baria-label\s*=\s*([\x22\x27])(.*?)\1', tag, re.I | re.S)
        if not attr:
            return tag
        text = _html.unescape(attr.group(2))
        if all(lab in text for lab in JP_LABELS) and len(re.findall(r'\d+(?:\.\d+)?\s*[%\uff05]', text)) == 5:
            # Put it in its own heading-delimited scope at the original position.
            return tag + '<h6>みんなの評価</h6>' + _html.escape(text) + '<h6>poll detail end</h6>'
        return tag
    return re.sub(r'<[^>]+>', replace, raw)


def _decoded_script_strings(raw: str) -> str:
    """Diagnostic only: decode JSON string literals, never execute JavaScript.
    Bounded input/output; decoded values are never passed into the scoring parser.
    """
    chunks = []
    for block in re.finditer(r'(?is)<script\b[^>]*>(.*?)</script\s*>', (raw or '')[:5_000_000]):
        body = block.group(1)
        for token in re.finditer(r'"(?:[^"\\]|\\.)*"', body):
            try:
                value = json.loads(token.group(0))
            except (ValueError, TypeError):
                continue
            if isinstance(value, str):
                chunks.append(value[:20000])
            if len(chunks) >= 20000:
                return ' '.join(chunks)
    return ' '.join(chunks)

def parse_jp_five_categories(scoped_text: str) -> dict[str, Any] | None:
    """Parse ONE isolated poll component into aggregates, or None. Never pass a whole
    page. Accepts decimals; rejects duplicate labels, missing categories, implausible
    totals, impossible displays, and anything the scorers would refuse."""
    return _five_category_detail(scoped_text)[0]


def _num(v: str) -> float | None:
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) and 0.0 <= x <= 100.0 else None


def _two_category(text: str) -> tuple[dict | None, str | None, str]:
    """Strict two-category adapters, used ONLY when no strong-buy/strong-sell label
    exists anywhere (otherwise '買いたい' would be found inside '強く買いたい').
    Returns (result, layout, failure_reason). Each label must occur exactly once.
    Label-first and percent-first are separate adapters; if both match the page
    is ambiguous, and a page mixing the two layouts is rejected — combining
    layouts label by label is how the previous parser bound one percent twice."""
    if text.count(_BUY) != 1 or text.count(_SELL) != 1:
        return None, None, "labels_not_unique"
    lbl = rf"(?:{re.escape(_BUY)}|{re.escape(_SELL)})"
    num = r"(\d+(?:\.\d+)?)\s*[%\uff05]"
    label_first = {m.group(1): _num(m.group(2)) for m in re.finditer(
        rf"({lbl})\s*[:\uff1a]?\s*{num}", text)}
    pct_first = {m.group(2): _num(m.group(1)) for m in re.finditer(
        rf"{num}\s{{0,3}}({lbl})", text)}
    found = []
    for layout, d in (("two_category_label_first", label_first),
                      ("two_category_percent_first", pct_first)):
        if set(d) == {_BUY, _SELL} and None not in d.values():
            found.append((layout, d))
    if not found:
        return None, None, "no_complete_layout"
    if len(found) == 2 and found[0][1] != found[1][1]:
        return None, None, "ambiguous_layouts"
    layout, d = found[0]
    buy, sell = d[_BUY], d[_SELL]
    why = jp_poll_problem(buy, sell, None)           # the scorers' own rule
    if why is not None:
        return None, None, why
    return ({"buy_pct": buy, "neutral_pct": None, "sell_pct": sell,
             "categories": {_BUY: buy, _SELL: sell}}, layout, "")


# Structural boundaries that end a poll scope: the next heading or section-level element.
# Headings are stripped when HTML becomes text, so the scope must be cut on the RAW html;
# cutting only at the next 「みんなの評価」 let a scope run into the following section.
_BOUNDARY = re.compile(r"(?i)<(?:h[1-6]|section|article|nav|footer|aside)\b"
                       r"|</(?:section|article)\s*>")
_RAW_SCOPE_CHARS = 20000


def _poll_scopes(raw_html: str) -> list[str]:
    """Text of each poll scope: from a 「みんなの評価」 heading to the first structural
    boundary after it (next heading / section / article / nav / footer / aside), the
    next poll heading, or a size cap — whichever comes first. <head>, <script> and
    <style> are removed first so page metadata and embedded data cannot open a scope.

    Plain text (no tags) has no structure to cut on; there the scope runs to the next
    poll heading or the character cap. This is still not DOM-verified isolation: a page
    built from plain <div>s without headings or sections could let a scope run on.
    """
    raw = re.sub(r"(?is)<(head|script|style)\b.*?</\1\s*>", " ", raw_html or "")
    raw = _accessible_poll_text(raw)
    starts = sorted(m.start() for h in JP_POLL_HEADINGS for m in re.finditer(re.escape(h), raw))
    out = []
    for i, st in enumerate(starts):
        end = min(starts[i + 1] if i + 1 < len(starts) else len(raw), st + _RAW_SCOPE_CHARS)
        cut = _BOUNDARY.search(raw, st, end)
        if cut is not None:
            end = cut.start()
        out.append(html_to_text(raw[st:end])[:_SCOPE_CHARS])
    return out


def poll_scope_html(raw_html: str, max_chars: int = 60_000) -> list[str]:
    """The raw HTML of each poll scope of the ORIGINAL page: from a 「みんなの評価」 heading to
    the next heading or structural boundary (capped). Used to save a small, poll-only
    fixture instead of a whole page: the rest of a saved page can hold personal or
    unrelated content, the poll block does not need it. Pair with a round-trip check
    (parse the fixture, compare) because slicing is only worth trusting if it reproduces
    the same result."""
    raw = re.sub(r"(?is)<(head|script|style)\b.*?</\1\s*>", " ", raw_html or "")
    starts = sorted(m.start() for h in JP_POLL_HEADINGS for m in re.finditer(re.escape(h), raw))
    out = []
    for i, st in enumerate(starts):
        end = min(starts[i + 1] if i + 1 < len(starts) else len(raw), st + max_chars)
        cut = _BOUNDARY.search(raw, st, end)
        if cut is not None:
            end = cut.start()
        out.append(raw[st:end])
    return out


def _parse_scope(scope: str) -> tuple[str, dict | None, str | None, str]:
    """One scope -> (kind, result, layout, reason); kind in ok | none | invalid.
    'none' = the scope holds no poll vocabulary at all (a menu link, say);
    'invalid' = it has poll labels but they do not form a valid poll."""
    has_strong = any(x in scope for x in _STRONG)
    if not has_strong and _BUY not in scope and _SELL not in scope:
        return "none", None, None, "no_poll_labels"
    if has_strong:
        # five-category vocabulary: parse ONLY as five categories. A two-category read
        # would bind 買いたい inside 強く買いたい.
        got, why = _five_category_detail(scope)
        if got is None:
            return "invalid", None, None, why
        return "ok", got, "five_category", ""
    res, layout, why = _two_category(scope)
    if res is None:
        # buy 0% / sell 0% with nothing else shown = nothing recorded: a labelled,
        # non-scoring state rather than an error
        return ("empty" if why == "no_votes" else "invalid"), None, None, why
    return "ok", res, layout, ""


def parse_jp_board(raw_html: str) -> dict[str, Any]:
    """Page adapter for the みんなの評価 poll. STRUCTURAL and UNVERIFIED against a live
    DOM (no permitted fixture was available).

    Isolation: only the text following a 「みんなの評価」 heading is read, so unrelated
    percentages and look-alike labels elsewhere on the page cannot be picked up. With
    no heading the result is unavailable (reason poll_heading_not_found) — the usual
    sign that Yahoo's markup differs from this assumption.

    Duplicates (desktop + mobile copies): all valid scopes must agree to rounding. A
    valid scope alongside a malformed one is AMBIGUOUS, not 'the good one wins' — a
    page that disagrees with itself is not trusted.

    status: ok | unavailable | parse_error | ambiguous. Never invents 50 or 0.
    """
    out: dict[str, Any] = {
        "status": "unavailable", "buy_pct": None, "neutral_pct": None, "sell_pct": None,
        "categories": {}, "layout": None, "parser_version": JP_PARSER_VERSION,
        "metric_type": "posting_sentiment_distribution", "source": "yahoo_jp_board",
        "vote_count": None, "window": None, "reason": "",
        # the rounded headline summary, when the page shows one next to the breakdown
        "summary_buy_pct": None, "summary_sell_pct": None,
    }
    problem = markup_problem(raw_html or "")
    if problem:
        out.update(status="parse_error", reason=problem)
        return out
    if not html_to_text(raw_html or ""):
        out["reason"] = "empty_page"
        return out
    scopes = _poll_scopes(raw_html)
    if not scopes:
        out["reason"] = "poll_heading_not_found"
        return out

    parsed = [_parse_scope(sc) for sc in scopes]
    oks = [p for p in parsed if p[0] == "ok"]
    invalid = [p for p in parsed if p[0] == "invalid"]
    if not oks:
        if invalid:
            out.update(status="parse_error", reason=invalid[0][3])
        elif any(p[0] == "empty" for p in parsed):
            out["reason"] = "no_votes"
        else:
            out["reason"] = "no_poll_labels"
        return out
    if invalid:
        out.update(status="ambiguous", reason="valid_and_malformed_components")
        return out
    first_kind, first, layout, _ = oks[0]
    for _k, other, olayout, _r in oks[1:]:
        if olayout != layout or any(
                abs(other["categories"].get(k, -1e9) - v) > _AGREE_TOLERANCE
                for k, v in first["categories"].items()):
            out.update(status="ambiguous", reason="duplicate_components_disagree")
            return out
    out.update(status="ok", layout=layout, buy_pct=first["buy_pct"],
               neutral_pct=first["neutral_pct"], sell_pct=first["sell_pct"],
               categories=first["categories"],
               summary_buy_pct=first.get("summary_buy_pct"),
               summary_sell_pct=first.get("summary_sell_pct"),
               reason=f"{len(oks)} agreeing component(s)")
    return out


def describe_poll_page(raw_html: str, context: int = 60, max_snippets: int = 24) -> dict[str, Any]:
    """Diagnostic view of what the Japan-board parser sees on a page. Pure: no network.

    Built to answer "why is there no poll?" on a REAL page, which cannot be verified from
    a test environment. It reports the parse outcome and, separately:
      * how often the 「みんなの評価」 heading appears in VISIBLE text;
      * how often the poll labels appear in visible text versus only inside <script>/<style>
        blocks (also counting JSON \\uXXXX-escaped strings, decoded for DETECTION only, never
        for scoring). Labels present only in script data mean the numbers are embedded JSON
        that this parser deliberately does not read;
      * which source produced the poll: visible text, an accessible (aria-label) copy, or both,
        and whether the two DISAGREE. Scoring is unchanged by this; it only exposes a case the
        scorer cannot — an accessible label overriding different visible numbers inside the
        same element;
      * the text of each scope the parser would read, and short snippets around each label.
    """
    raw = raw_html or ""
    problem = markup_problem(raw)
    res = parse_jp_board(raw)
    result = {k: res.get(k) for k in ("status", "reason", "layout", "buy_pct", "neutral_pct",
                                      "sell_pct", "summary_buy_pct", "summary_sell_pct",
                                      "parser_version")}
    if problem:                                   # nothing below is safe to run on such markup
        return {"result": result, "malformed_html": problem, "page_chars": len(raw),
                "visible_text_chars": 0, "heading_in_visible_text": 0, "heading_in_script_data": 0,
                "label_counts_visible": {l: 0 for l in JP_LABELS},
                "label_counts_script_data": {l: 0 for l in JP_LABELS},
                "scope_count": 0, "scopes": [], "snippets": [], "poll_source": "none",
                "visible_only": None, "source_disagreement": False}
    visible = html_to_text(raw)
    script_blocks = " ".join(m.group(0) for m in re.finditer(r"(?is)<(script|style)\b.*?</\1\s*>", raw))
    decoded = _decoded_script_strings(raw)
    labels = list(JP_LABELS) + list(JP_POLL_HEADINGS)

    def count(text: str, needle: str) -> int:
        return len(re.findall(re.escape(needle), text))

    def script_count(needle: str) -> int:
        # max, not sum: a plain (unescaped) occurrence appears in BOTH the raw script text
        # and its decoded strings, and must count once
        return max(count(script_blocks, needle), count(decoded, needle))
    snippets = []
    for lab in labels:
        for m in re.finditer(re.escape(lab), visible):
            snippets.append({"label": lab, "where": "visible",
                             "text": visible[max(0, m.start() - context): m.end() + context]})
    scopes = _poll_scopes(raw)
    # source: compare the page as parsed (accessible copies exposed) with the same page
    # with them removed. Different headings in the two views = an accessible copy exists.
    injected = _accessible_poll_text(re.sub(r"(?is)<(head|script|style)\b.*?</\1\s*>", " ", raw))
    n_accessible = injected.count("poll detail end")
    visible_only = None
    disagree = False
    if n_accessible:
        stripped = _ARIA_LABEL_ATTR.sub("data-x=", raw)          # neutralise aria-labels
        vo = parse_jp_board(stripped)
        visible_only = {k: vo.get(k) for k in ("status", "reason", "buy_pct", "neutral_pct", "sell_pct")}
        if vo.get("status") == "ok" and res.get("status") == "ok":
            disagree = (abs(vo["buy_pct"] - res["buy_pct"]) > _AGREE_TOLERANCE
                        or abs(vo["sell_pct"] - res["sell_pct"]) > _AGREE_TOLERANCE)
    source = ("none" if res.get("status") != "ok" else
              "accessible_label" if n_accessible and (visible_only or {}).get("status") != "ok" else
              "visible_and_accessible" if n_accessible else "visible_text")
    return {
        "result": result, "malformed_html": None,
        "page_chars": len(raw), "visible_text_chars": len(visible),
        "heading_in_visible_text": sum(count(visible, h) for h in JP_POLL_HEADINGS),
        "heading_in_script_data": sum(script_count(h) for h in JP_POLL_HEADINGS),
        "label_counts_visible": {lab: count(visible, lab) for lab in JP_LABELS},
        "label_counts_script_data": {lab: script_count(lab) for lab in JP_LABELS},
        "scope_count": len(scopes),
        "scopes": [sc[:500] for sc in scopes],
        "snippets": snippets[:max_snippets],
        "poll_source": source, "accessible_scopes": n_accessible,
        "visible_only": visible_only, "source_disagreement": bool(disagree),
    }


# --------------------------------------------------------------------------
# Stocktwits — APPROVED API ONLY (no scraping of web pages or private endpoints)
# --------------------------------------------------------------------------
STOCKTWITS_PARSER_VERSION = "stocktwits-detail-v1"
STOCKTWITS_ENDPOINT = ("https://api-gw-prd.stocktwits.com/api-middleware/external/"
                       "sentiment/v2/{symbol}/detail")
STOCKTWITS_STATUSES = ("ok", "unavailable", "not_configured", "invalid_symbol",
                       "unauthorized", "forbidden", "rate_limited", "network_error",
                       "http_error", "invalid_response", "circuit_open")


def _score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 100 else None


def normalize_stocktwits(payload: dict, symbol: str, window: str) -> dict:
    """Map the documented response onto a stable record. Uses valueNormalized /
    labelNormalized exactly as supplied; never derives buy/sell percentages, never
    substitutes another window, treats a legitimate 0 as a value, and leaves
    provider_asof null when the provider gives none."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = {
        "source": "stocktwits", "symbol": symbol, "window": window,
        "metric_type": "provider_normalized_sentiment", "status": "unavailable",
        "score": None, "label": None, "message_volume_score": None,
        "buy_pct": None, "sell_pct": None,
        "fetched_at": now, "provider_asof": None,
        "parser_version": STOCKTWITS_PARSER_VERSION,
    }
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return out
    sentiment = data.get("sentiment")
    item = sentiment.get(window) if isinstance(sentiment, dict) else None
    if not isinstance(item, dict):
        return out
    score = _score(item.get("valueNormalized"))
    if score is None:
        return out
    label = item.get("labelNormalized")
    out.update(status="ok", score=score, label=label if isinstance(label, str) else None)
    volume = data.get("messageVolume")
    vol_item = volume.get(window) if isinstance(volume, dict) else None
    if isinstance(vol_item, dict):
        # a normalised 0-100 attention metric, NOT a message count
        out["message_volume_score"] = _score(vol_item.get("valueNormalized"))
    return out


def fetch_stocktwits_approved(symbol: str, username: str | None, password: str | None,
                              window: str = "24h", timeout: float = 8.0,
                              opener=None) -> dict:
    """Synchronous approved-API client (HTTP Basic, documented endpoint). Caller is
    responsible for caching, request budgeting and the circuit breaker.
    No credentials -> no request. Credentials must be APPROVED API credentials, not
    a consumer login. Never logs headers, credentials or response bodies.
    `opener` exists for tests (defaults to urllib.request.urlopen)."""
    symbol = (symbol or "").strip().upper()
    empty = normalize_stocktwits({}, symbol, window)
    if not username or not password:
        return dict(empty, status="not_configured")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,19}", symbol):
        return dict(empty, status="invalid_symbol")
    url = STOCKTWITS_ENDPOINT.format(symbol=urllib.parse.quote(symbol, safe=""))
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    req = urllib.request.Request(url, headers={
        "Authorization": "Basic " + token, "Accept": "application/json",
        "User-Agent": "AlphaQuantResearch/1.0",
    })
    opener = opener or urllib.request.urlopen
    try:
        with opener(req, timeout=timeout) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            return dict(empty, status="invalid_response")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return dict(empty, status="invalid_response")
        return normalize_stocktwits(payload, symbol, window)
    except urllib.error.HTTPError as exc:
        status = {401: "unauthorized", 403: "forbidden", 429: "rate_limited"}.get(
            exc.code, "http_error")
        return dict(empty, status=status)
    except (urllib.error.URLError, TimeoutError, OSError):
        return dict(empty, status="network_error")
    except (ValueError, UnicodeError):
        return dict(empty, status="invalid_response")
