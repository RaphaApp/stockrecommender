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
JP_PARSER_VERSION = "jp-board-v3-dom-scoped-shared-validity"
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


def _five_category_detail(scoped_text: str) -> tuple[dict | None, str]:
    """(result, reason) for LABEL-THEN-PERCENT text from ONE isolated component."""
    alternatives = "|".join(re.escape(x) for x in sorted(JP_LABELS, key=len, reverse=True))
    pattern = (rf"(?P<label>{alternatives})\s*[:\uff1a]?\s*"
               rf"(?P<value>\d+(?:\.\d+)?)\s*[%\uff05]")
    values: dict[str, float] = {}
    half: dict[str, float] = {}
    for match in re.finditer(pattern, scoped_text):
        label, raw = match["label"], match["value"]
        value = float(raw)
        if label in values:
            return None, "duplicate_label"
        if not 0 <= value <= 100:
            return None, "value_out_of_range"
        values[label], half[label] = value, _half_unit(raw)
    if set(values) != set(JP_LABELS):
        return None, "missing_category"
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
    return {"buy_pct": buy, "neutral_pct": neutral, "sell_pct": sell,
            "categories": values}, ""


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
    starts = sorted(m.start() for h in JP_POLL_HEADINGS for m in re.finditer(re.escape(h), raw))
    out = []
    for i, st in enumerate(starts):
        end = min(starts[i + 1] if i + 1 < len(starts) else len(raw), st + _RAW_SCOPE_CHARS)
        cut = _BOUNDARY.search(raw, st, end)
        if cut is not None:
            end = cut.start()
        out.append(html_to_text(raw[st:end])[:_SCOPE_CHARS])
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
    }
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
               reason=f"{len(oks)} agreeing component(s)")
    return out


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
