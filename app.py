from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import ssl
import threading
import time
import urllib.parse
import urllib.request
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    import yfinance as yf
except Exception:
    yf = None

# Optional: powers the six-factor radar in the Deep-Dive tab. Guarded like yfinance
# so the app runs without it (the radar section is simply skipped). Add `plotly`
# to requirements.txt to enable it.
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Optional companion module (sec_research.py). Powers the US-only Conviction tab from
# SEC EDGAR fundamentals. Guarded so the app still runs if it's absent.
try:
    import sec_research
except Exception:
    sec_research = None

logger = logging.getLogger("alpha_quant")

# ----------------------------------------------------------------------------
# Networking — corporate-friendly SSL handling for yfinance
# ----------------------------------------------------------------------------
def _build_yf_session():
    """Return a curl_cffi session for yfinance that copes with corporate SSL.

    Large corporate networks and many VPNs run TLS inspection: a proxy re-signs
    HTTPS traffic with an internal root CA. libcurl (which yfinance uses under the
    hood) doesn't trust that internal CA, so every fetch fails with
    'curl: (60) ... unable to get local issuer certificate'. Two opt-in escape
    hatches, both via environment variables so the secure default is unchanged:

        STOCKREC_CA_BUNDLE    = C:\\path\\to\\corp-ca.pem   (secure: trust your corp CA)
        STOCKREC_INSECURE_SSL = 1                           (quick: skip verification)

    On a private/home network you set neither, and behaviour is the normal secure
    default. The helper degrades gracefully if curl_cffi isn't importable.
    """
    if yf is None:
        return None
    try:
        from curl_cffi import requests as cffi_requests
    except Exception:
        return None

    kwargs = {"impersonate": "chrome"}
    ca_bundle = os.environ.get("STOCKREC_CA_BUNDLE")
    if ca_bundle:
        kwargs["verify"] = ca_bundle                     # trust a specific CA file
    elif os.environ.get("STOCKREC_INSECURE_SSL") == "1":
        kwargs["verify"] = False                         # skip verification entirely
        warnings.filterwarnings("ignore", message=".*[Vv]erif.*")
    try:
        return cffi_requests.Session(**kwargs)
    except Exception:
        return None

_YF_SESSION = _build_yf_session()

def _ticker(symbol: str):
    """Construct a yfinance Ticker, reusing the shared curl_cffi session if present.

    This is the single entry point every fetch (history, fundamentals, analyst and
    insider data) goes through. The session built in _build_yf_session impersonates
    Chrome, which both copes with corporate TLS inspection and makes Yahoo far less
    likely to rate-limit the IP. If no session was created (curl_cffi missing) or a
    given yfinance version doesn't accept the `session=` argument, fall back to a
    plain Ticker so the app still works.
    """
    if _YF_SESSION is not None:
        try:
            return yf.Ticker(symbol, session=_YF_SESSION)
        except TypeError:
            pass   # this yfinance build doesn't take session=; use the default
    return yf.Ticker(symbol)

def _reddit_ssl_context() -> ssl.SSLContext | None:
    """Match the rest of the app's SSL behaviour for the urllib Reddit call.

    urllib uses Python's own ssl module (not curl_cffi), so on a TLS-inspecting
    corporate network it would fail independently. Honour the same env vars used
    for yfinance so the Reddit feed behaves consistently. Returns None to let
    urllib use its default context on a normal network.
    """
    ca_bundle = os.environ.get("STOCKREC_CA_BUNDLE")
    if ca_bundle:
        try:
            return ssl.create_default_context(cafile=ca_bundle)
        except Exception:
            return None
    if os.environ.get("STOCKREC_INSECURE_SSL") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None

def _get_secret(name: str, default: str | None = None) -> str | None:
    """Read a credential from Streamlit secrets first (set in the Cloud dashboard),
    then fall back to environment variables (handy for local runs). Never raises if
    no secrets file is configured."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)

def _count_ticker_mentions(blob_upper: str, tickers: list) -> dict:
    """Count mentions of each ticker in an uppercased text blob.

    Sub-3-char symbols (e.g. 'V', 'KO') require an explicit '$' cashtag so ordinary
    words aren't mistaken for tickers; longer symbols use a plain word boundary,
    which still matches the $ form because '$' is itself a boundary-forming char.
    """
    counts = {}
    for t in tickers:
        pattern = (r"\$" + re.escape(t) + r"\b") if len(t) < 3 else (r"\b" + re.escape(t) + r"\b")
        counts[t] = len(re.findall(pattern, blob_upper, re.IGNORECASE))
    return counts

# Small finance lexicons for a *gentle* directional tilt on Reddit buzz. Mention count is
# volume; this nudges it by the bull/bear words appearing near each mention, so a heavily
# bearish-discussed name doesn't earn the same hype as a bullish one. The tilt is bounded
# and multiplicative — it can dampen the positive kick toward zero but never manufactures
# negative hype (the sell scanner owns bearishness). With sparse text it's ~0, i.e. a
# plain count. Empty either set to disable that side.
_BULL_WORDS = {"BUY", "BUYING", "BOUGHT", "LONG", "CALL", "CALLS", "MOON", "MOONING",
               "BULLISH", "BULL", "BREAKOUT", "RALLY", "SQUEEZE", "PUMP", "ROCKET",
               "UNDERVALUED", "BEAT", "BEATS", "UPGRADE", "UPGRADED", "GAINS", "GREEN",
               "HODL", "ACCUMULATE", "OVERSOLD", "RIP"}
_BEAR_WORDS = {"SELL", "SELLING", "SOLD", "SHORT", "SHORTING", "PUT", "PUTS", "CRASH",
               "CRASHING", "BEARISH", "BEAR", "DUMP", "DUMPING", "DROP", "DROPPING",
               "OVERVALUED", "MISS", "MISSES", "MISSED", "DOWNGRADE", "DOWNGRADED",
               "RED", "BAGHOLDER", "PUKE", "TANK", "TANKING", "DEAD"}
_TONE_WINDOW = 80     # chars of context examined each side of a mention
_TONE_GAIN = 0.15     # tilt per net (bull - bear) distinct word
_TONE_CAP = 0.5       # max |tilt|, so buzz is scaled within [0.5x, 1.5x]

def _tone_weighted_counts(blob_upper: str, tickers: list) -> dict:
    """Mention count per ticker, gently scaled by the bull/bear tone of the text around
    each mention: volume * (1 + tilt), tilt in [-_TONE_CAP, +_TONE_CAP], floored at 0.
    Distinct sentiment words per window (so spam can't dominate). Degrades to a plain
    count when no sentiment words are nearby."""
    out = {}
    for t in tickers:
        pattern = (r"\$" + re.escape(t) + r"\b") if len(t) < 3 else (r"\b" + re.escape(t) + r"\b")
        matches = list(re.finditer(pattern, blob_upper, re.IGNORECASE))
        n = len(matches)
        if n == 0:
            out[t] = 0.0
            continue
        net = 0
        for m in matches:
            ctx = blob_upper[max(0, m.start() - _TONE_WINDOW): m.end() + _TONE_WINDOW]
            words = set(re.findall(r"[A-Z]+", ctx))
            net += len(words & _BULL_WORDS) - len(words & _BEAR_WORDS)
        tilt = max(-_TONE_CAP, min(_TONE_CAP, _TONE_GAIN * net))
        out[t] = max(0.0, n * (1.0 + tilt))
    return out

def _reddit_get(url: str, headers: dict, data: bytes | None = None) -> str:
    """Single urllib request honouring the app's SSL context. Raises on failure."""
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=10, context=_reddit_ssl_context()) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def _reddit_oauth_token() -> str | None:
    """Application-only (userless) OAuth token via the client_credentials grant.

    Needs only a client id + secret from a Reddit 'script' app — no username or
    password. Returns None when credentials aren't configured, so the caller can
    fall back to the public RSS feed.
    """
    cid = _get_secret("REDDIT_CLIENT_ID")
    csec = _get_secret("REDDIT_CLIENT_SECRET")
    if not cid or not csec:
        return None
    ua = _get_secret("REDDIT_USER_AGENT", "stockrec-hype/1.0")
    auth = base64.b64encode(f"{cid}:{csec}".encode()).decode()
    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    raw = _reddit_get("https://www.reddit.com/api/v1/access_token", data=body,
                      headers={"Authorization": f"Basic {auth}", "User-Agent": ua})
    return json.loads(raw).get("access_token")

# Records which path last produced data, for an at-a-glance UI status indicator.
_LAST_REDDIT_SOURCE = "offline"

# Public finance subreddits polled for the hype signal. More subs = broader coverage
# but more requests and higher block risk on unauthenticated access, so this is a
# deliberate handful of high ticker-density communities. Each is polled ONCE per scan
# (titles + bodies counted across the whole blob), never per ticker — per-ticker search
# would be one request per ticker per sub and get throttled fast.
REDDIT_SUBS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]

def _reddit_collect(url_fn, headers_fn, parse_fn) -> str | None:
    """Fetch each REDDIT_SUBS listing, run parse_fn(raw) -> list[str], and join them.
    Per-sub failures are skipped; returns the combined UPPERCASED text, or None if
    every subreddit failed (so the caller can drop to the next tier)."""
    chunks, ok = [], False
    for sub in REDDIT_SUBS:
        try:
            chunks.extend(parse_fn(_reddit_get(url_fn(sub), headers_fn())))
            ok = True
        except Exception:
            continue   # this subreddit unavailable -> skip, keep polling the rest
        time.sleep(0.3)   # courtesy spacing between public requests
    return (" ".join(chunks).upper() if ok else None)

def _parse_reddit_json(raw: str) -> list:
    texts = []
    for child in json.loads(raw).get("data", {}).get("children", []):
        d = child.get("data", {})
        texts.append(str(d.get("title", "")))
        texts.append(str(d.get("selftext", "")))
    return texts

def _parse_reddit_rss(raw: str) -> list:
    blocks = re.findall(r"<title[^>]*>(.*?)</title>", raw, re.DOTALL | re.IGNORECASE)
    blocks += re.findall(r"<content[^>]*>(.*?)</content>", raw, re.DOTALL | re.IGNORECASE)
    return blocks if blocks else [raw]

def fetch_reddit_hype(tickers: list) -> dict:
    """Count mentions per ticker across several public finance subreddits, preferring
    authenticated access.

    Three-tier, fully fail-safe:
      1. Authenticated OAuth (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET) — most reliable
         from cloud IPs; up to 100 hot posts (title + body) per subreddit.
      2. Public RSS hot feed per subreddit — no account/registration; works locally,
         often blocked on cloud datacenter IPs.
      3. Zero mentions for every ticker — guarantees the scan never crashes.
    Subreddits are polled once each (not per ticker). Standard library only.
    """
    global _LAST_REDDIT_SOURCE

    # --- Tier 1: authenticated OAuth across all subs ---------------------------
    try:
        token = _reddit_oauth_token()
        if token:
            ua = _get_secret("REDDIT_USER_AGENT", "stockrec-hype/1.0")
            blob = _reddit_collect(
                lambda s: f"https://oauth.reddit.com/r/{s}/hot?limit=100",
                lambda: {"Authorization": f"Bearer {token}", "User-Agent": ua},
                _parse_reddit_json)
            if blob is not None:
                _LAST_REDDIT_SOURCE = "authenticated"
                return _tone_weighted_counts(blob, tickers)
    except Exception:
        pass   # fall through to the unauthenticated RSS path

    # --- Tier 2: unauthenticated public RSS across all subs --------------------
    blob = _reddit_collect(
        lambda s: f"https://www.reddit.com/r/{s}/hot.rss",
        lambda: {"User-Agent": "Mozilla/5.0"},
        _parse_reddit_rss)
    if blob is not None:
        _LAST_REDDIT_SOURCE = "rss"
        return _tone_weighted_counts(blob, tickers)

    # --- Tier 3: safe zero-mention fallback ------------------------------------
    _LAST_REDDIT_SOURCE = "offline"
    return {t: 0 for t in tickers}

# ----------------------------------------------------------------------------
# Pluggable, per-market sentiment sources
# ----------------------------------------------------------------------------
# Each source declares which MARKETS (TICKER_UNIVERSE region keys) it covers and a
# fetch fn (tickers, names) -> {ticker: mention_count}. The orchestrator routes each
# market to the sources that cover it and merges the counts. Reddit handles English
# names; GDELT's multilingual news index reaches Japanese & Chinese coverage Reddit
# never sees. The market routing lives here in code (not in the UI), so the blend
# stays private while the sidebar still lets a user toggle whole sources on/off.
_LAST_GDELT_STATUS = "offline"

def _sentiment_reddit(tickers: list, names: dict) -> dict:
    """Reddit mention counts across several finance subs (English/US retail buzz)."""
    return fetch_reddit_hype(tickers)

def _sentiment_gdelt(tickers: list, names: dict) -> dict:
    """News-volume buzz per company from GDELT's free DOC API (no API key needed).

    Batches the company names into short OR-queries (one request per ~12 names),
    aggregates the returned article titles, and counts each name — the same blob-
    and-count approach as Reddit, but over a 100+ language global news index, so it
    reaches the Japanese and Chinese coverage Reddit lacks. Fully fail-safe: any
    network/parse error returns zero mentions. Title matching is on the English name,
    so native-language-only articles are under-counted (a known, acceptable limit).
    """
    global _LAST_GDELT_STATUS
    counts = {t: 0 for t in tickers}
    items = [(t, names.get(t, t)) for t in tickers if names.get(t, t)]
    ua = _get_secret("REDDIT_USER_AGENT", "stockrec-hype/1.0")
    got_any = False
    try:
        for i in range(0, len(items), 12):
            chunk = items[i:i + 12]
            or_terms = " OR ".join(f'"{nm}"' for _, nm in chunk)
            url = ("https://api.gdeltproject.org/api/v2/doc/doc?query="
                   + urllib.parse.quote(f"({or_terms})")
                   + "&mode=artlist&format=json&timespan=3d&maxrecords=250&sort=hybridrel")
            raw = _reddit_get(url, {"User-Agent": ua})
            try:
                arts = json.loads(raw).get("articles", [])
            except Exception:
                arts = []
            if arts:
                got_any = True
            blob = " ".join(str(a.get("title", "")) for a in arts).upper()
            for tkr, nm in chunk:
                counts[tkr] += len(re.findall(r"\b" + re.escape(nm.upper()) + r"\b", blob))
            time.sleep(0.5)   # GDELT asks for gentle pacing between requests
        _LAST_GDELT_STATUS = "ok" if got_any else "offline"
        return counts
    except Exception:
        _LAST_GDELT_STATUS = "offline"
        return {t: 0 for t in tickers}

SENTIMENT_SOURCES = {
    "reddit": {"label_key": "src_reddit", "markets": {"USA", "Europe"},
               "fn": _sentiment_reddit},
    "gdelt":  {"label_key": "src_gdelt",  "markets": {"USA", "Japan", "Europe", "China"},
               "fn": _sentiment_gdelt},
}
DEFAULT_SOURCES = list(SENTIMENT_SOURCES.keys())

# Human-readable summary of what the last hype scan actually used, for the UI.
_LAST_HYPE_STATUS = ""
_LAST_HYPE_FETCHED_AT = 0.0   # epoch seconds of the last ACTUAL (cache-miss) hype fetch

def _fetch_hype_uncached(universe: dict, enabled: list) -> tuple:
    """Run every enabled source over the markets it covers; merge per-ticker counts.
    Returns (combined, sources_ran). Sets the per-source status globals as a side effect."""
    combined = {t: 0 for ticks in universe.values() for t in ticks}
    sources_ran = []
    for key in enabled:
        src = SENTIMENT_SOURCES.get(key)
        if not src:
            continue
        sub = [t for region, ticks in universe.items() if region in src["markets"] for t in ticks]
        if not sub:
            continue
        names = {t: COMPANY_NAMES.get(t, t) for t in sub}
        try:
            res = src["fn"](sub, names) or {}
        except Exception:
            res = {}
        for t, c in res.items():
            combined[t] = combined.get(t, 0) + float(c or 0)
        sources_ran.append(key)
    return combined, sources_ran

@st.cache_data(ttl=1800, show_spinner=False)
def _hype_payload_cached(universe_key: tuple, enabled_key: tuple) -> dict:
    """Cached real fetch (30-min TTL): repeat scans of the same universe + sources reuse
    this instead of re-hitting Reddit/GDELT. Captures the fetch time and the raw per-source
    status keys so the wrapper can rebuild a localized status string on cache hits too."""
    universe = {region: list(tks) for region, tks in universe_key}
    combined, sources_ran = _fetch_hype_uncached(universe, list(enabled_key))
    return {"counts": combined, "sources_ran": sources_ran,
            "reddit": _LAST_REDDIT_SOURCE, "gdelt": _LAST_GDELT_STATUS,
            "fetched_at": time.time()}

def fetch_hype_signals(universe: dict, enabled_sources: list | None = None) -> dict:
    """Merged per-ticker hype counts ({ticker: combined_count}); analyze_ticker turns that
    into the capped hype bonus. Results are cached for 30 min and stamped with a fetch time
    (shown as 'updated X ago'); each source is independently fail-safe."""
    global _LAST_HYPE_STATUS, _LAST_HYPE_FETCHED_AT, _LAST_REDDIT_SOURCE, _LAST_GDELT_STATUS
    enabled = enabled_sources if enabled_sources is not None else DEFAULT_SOURCES
    universe_key = tuple((region, tuple(ticks)) for region, ticks in sorted(universe.items()))
    payload = _hype_payload_cached(universe_key, tuple(enabled))
    _LAST_HYPE_FETCHED_AT = payload["fetched_at"]
    _LAST_REDDIT_SOURCE, _LAST_GDELT_STATUS = payload["reddit"], payload["gdelt"]
    status_parts = []
    for key in payload["sources_ran"]:
        if key == "reddit":
            status_parts.append(f"{tr('src_reddit')}: {tr('hype_status_' + payload['reddit'])}")
        elif key == "gdelt":
            status_parts.append(f"{tr('src_gdelt')}: {tr('hype_status_' + payload['gdelt'])}")
    _LAST_HYPE_STATUS = (tr("hype_sources_prefix") + " " + " · ".join(status_parts)) if status_parts else ""
    return payload["counts"]




# ----------------------------------------------------------------------------
# Configuration & Global Market Universes
# ----------------------------------------------------------------------------
DB_PATH = "stock_engine.db"
FACTORS = ["momentum", "value", "technical", "hype", "quality", "theme"]

DEFAULT_WEIGHTS = {
    "momentum": 0.20,
    "value": 0.18,
    "technical": 0.18,
    "hype": 0.16,
    "quality": 0.16,
    # Theme (industry-rotation) starts as the smallest factor: it's the newest and
    # the most correlated with momentum, so it must earn a bigger weight through the
    # walk-forward loop rather than being granted one. Old kpi_weights rows lack the
    # column; get_latest_weights substitutes this default and renormalises.
    "theme": 0.12,
}

LEARNING_RATE = 0.04
MIN_WEIGHT = 0.05
EVAL_HORIZON_DAYS = 14
BUY_THRESHOLD = 65.0
SELL_THRESHOLD = 45.0

from config import (
    TICKER_UNIVERSE, COMPANY_NAMES, THEMES, BENCHMARKS, TRANSLATIONS,
    DEEP_FINALISTS, DEEP_US_TICKERS, JP_DEEP_TICKERS, CN_DEEP_TICKERS, DEEP_UNIVERSES,
)
# Deep-scan promotion settings (new). getattr-style fallback so an older config.py
# without these blocks still runs — promotions just start empty with the 5/3/2 quota.
try:
    from config import PROMOTED_TICKERS, PROMOTION_QUOTA
except ImportError:
    PROMOTED_TICKERS = {"USA": [], "Japan": [], "China": []}
    PROMOTION_QUOTA = {"USA": 5, "Japan": 3, "China": 2}
# Live-universe + theme-bridge settings (new); fallbacks keep an older config.py running.
try:
    from config import DYNAMIC_UNIVERSE_SPEC, DYNAMIC_ALLOWED_SUFFIXES, THEME_INDUSTRY_KEYWORDS
except ImportError:
    DYNAMIC_UNIVERSE_SPEC, DYNAMIC_ALLOWED_SUFFIXES, THEME_INDUSTRY_KEYWORDS = {}, {}, []
from indicators import (
    compute_rsi, compute_macd, compute_bollinger, compute_hype, clamp, screen_metrics,
    forum_sentiment_score, forum_euphoria_sell_score, theme_strength_score,
)
ALL_TICKERS = [ticker for region in TICKER_UNIVERSE.values() for ticker in region]







DEFAULT_BENCHMARK = "SPY"

def benchmark_for(ticker: str) -> str:
    """Return the benchmark index symbol for a ticker, based on its exchange suffix."""
    suffix = ticker.rsplit(".", 1)[1].upper() if "." in ticker else ""
    return BENCHMARKS.get(suffix, DEFAULT_BENCHMARK)


# ----------------------------------------------------------------------------
# Internationalization (i18n) — English / Japanese
# ----------------------------------------------------------------------------
# The translation helper is named `tr` (not `t`) on purpose: the scan loop uses
# a local variable `t` for the ticker symbol, so a `t()` helper would be shadowed.
LANGUAGES = {"English": "en", "日本語": "ja"}

REGION_NAMES = {
    "en": {"USA": "USA", "Japan": "Japan", "Europe": "Europe", "China": "China"},
    "ja": {"USA": "アメリカ", "Japan": "日本", "Europe": "ヨーロッパ", "China": "中国"},
}


def get_lang() -> str:
    return st.session_state.get("lang", "en")

def tr(key: str, **kwargs) -> str:
    """Look up a UI string for the active language, falling back to English."""
    table = TRANSLATIONS.get(get_lang(), TRANSLATIONS["en"])
    text = table.get(key) or TRANSLATIONS["en"].get(key, key)
    return text.format(**kwargs) if kwargs else text

def region_name(key: str) -> str:
    """Translate an internal region key (e.g. 'Japan') for display."""
    return REGION_NAMES.get(get_lang(), REGION_NAMES["en"]).get(key, key)

st.set_page_config(
    page_title="Alpha Quant Engine",
    page_icon="📈",
    layout="wide",
    # "auto": expanded on desktop, collapsed on phones. The page NAVIGATION now
    # lives in the sidebar (st.navigation), so hiding it by default on desktop
    # would hide the app's entire structure; on mobile "auto" still collapses it
    # and the first-run hero CTA covers the cold start.
    initial_sidebar_state="auto",
)

# ----------------------------------------------------------------------------
# Styling & UI Engine
# ----------------------------------------------------------------------------
def inject_css(accent: str = "#3b82f6", card_bg: str = "rgba(255,255,255,0.03)") -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --accent: {accent};
            --card-bg: {card_bg};
            --pos: #16a34a;
            --neg: #dc2626;
            --muted: #94a3b8;
        }}
        /* Clear Streamlit's fixed top toolbar so the tab bar below the header
           doesn't slide underneath it. The previous 1.4rem was too tight. */
        .block-container,
        [data-testid="stMainBlockContainer"] {{ padding-top: 3.75rem; padding-bottom: 3rem; }}
        .qc-card {{
            background: var(--card-bg);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.10);
            transition: transform .12s ease, border-color .12s ease;
        }}
        .qc-card:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
        .qc-label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
        .qc-value {{ font-size: 1.55rem; font-weight: 700; line-height: 1.15; margin-top: 2px; }}
        .qc-delta {{ font-size: 0.9rem; font-weight: 600; margin-top: 2px; }}
        .qc-pos {{ color: var(--pos); }}
        .qc-neg {{ color: var(--neg); }}
        .qc-pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.8rem; }}
        .qc-buy  {{ background: rgba(22,163,74,0.15);  color: var(--pos); }}
        .qc-hold {{ background: rgba(148,163,184,0.18); color: var(--muted); }}
        .qc-sell {{ background: rgba(220,38,38,0.15);  color: var(--neg); }}
        .qc-ticker {{ font-size: 1.25rem; font-weight: 800; }}
        .qc-sub {{ font-size: 0.82rem; color: var(--muted); }}
        .qc-chip {{
            display: inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; color: var(--muted);
            background: var(--card-bg); border: 1px solid rgba(148,163,184,0.25);
            margin-bottom: 8px;
        }}
        @media (max-width: 640px) {{
            .qc-value {{ font-size: 1.3rem; }}
            .qc-ticker {{ font-size: 1.1rem; }}
            .qc-card {{ padding: 12px 14px; margin-bottom: 10px; }}
            .block-container {{ padding-left: .6rem; padding-right: .6rem; }}
            /* Streamlit does not stack st.columns on narrow screens by default,
               so the 3- and 4-up card rows get badly squished on a phone.
               Force every column in a row to take the full width and wrap. */
            div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: .4rem; }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def metric_card(label: str, value: str, delta: str | None = None, positive: bool | None = None) -> str:
    delta_html = ""
    if delta is not None:
        cls = "qc-pos" if positive else "qc-neg" if positive is not None else ""
        arrow = "▲ " if positive else "▼ " if positive is not None else ""
        delta_html = f'<div class="qc-delta {cls}">{arrow}{delta}</div>'
    return f'<div class="qc-card"><div class="qc-label">{label}</div><div class="qc-value">{value}</div>{delta_html}</div>'

def rec_pill(rec: str) -> str:
    cls = {"BUY": "qc-buy", "HOLD": "qc-hold", "SELL": "qc-sell"}.get(rec, "qc-hold")
    label = tr(f"rec_{rec}")
    return f'<span class="qc-pill {cls}">{label}</span>'

# --- None-proofing Parsing Helpers ---
def safe_float(value, default: float = float("nan")) -> float:
    try:
        if value is None: return default
        if isinstance(value, str):
            # Yahoo occasionally returns currency-formatted strings (e.g. "$1,250,000",
            # common in insider-transaction Value fields). Strip $ signs, commas, and
            # surrounding whitespace before casting so the number isn't lost.
            value = value.replace("$", "").replace(",", "").strip()
            if value == "": return default
        f = float(value)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError): return default

def fmt_money(v: float) -> str: return "—" if math.isnan(v) else f"${v:,.2f}"
def fmt_pct(v: float) -> str: return "—" if math.isnan(v) else f"{v:+.2f}%"
def fmt_num(v: float, nd: int = 1) -> str: return "—" if math.isnan(v) else f"{v:.{nd}f}"
def fmt_big(v: float) -> str:
    if math.isnan(v): return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(v) >= div: return f"${v / div:,.2f}{unit}"
    return f"${v:,.0f}"

# ----------------------------------------------------------------------------
# Database layer — dual backend.
#
# Default: local SQLite (zero config, exactly as before). If a DATABASE_URL
# secret/env var is set (e.g. a Supabase Postgres connection string) AND
# psycopg2 is installed, everything persists to Postgres instead — which is
# what survives Streamlit Cloud redeploys (walk-forward weights, mock
# portfolio, promotions, theme history). All call sites are unchanged: the
# adapter speaks the sqlite3 dialect the app already uses ('?' placeholders,
# conn.execute(), rows addressable by name AND position, `with` = commit).
# ----------------------------------------------------------------------------
try:
    import psycopg2                     # optional; only needed for Postgres mode
    import psycopg2.extras
    import psycopg2.pool
except Exception:
    psycopg2 = None


def _db_url() -> str | None:
    return _get_secret("DATABASE_URL")


def db_backend() -> str:
    return "postgres" if (psycopg2 is not None and _db_url()) else "sqlite"


def _pg_sql(sql: str) -> str:
    """Translate the app's sqlite dialect to Postgres: '?' params -> '%s'."""
    return sql.replace("?", "%s")


class _PgConn:
    """psycopg2 adapter with the sqlite3-ish surface the app uses.

    - conn.execute()/executemany() directly (no explicit cursor at call sites)
    - '?' placeholders
    - DictCursor rows: addressable by column name AND position, like sqlite3.Row
      (RealDictCursor would break the app's positional access — r[0] etc.)
    - context manager commits on success, rolls back on error, and RELEASES the
      connection back to the shared pool (or closes it, for a non-pooled fallback
      connection) — see _pg_checkout below.
    """
    def __init__(self, raw, pool=None):
        self._raw = raw
        self._pool = pool   # None -> direct connection; close() really closes

    def execute(self, sql: str, params=()):
        cur = self._raw.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(_pg_sql(sql), tuple(params))
        return cur

    def executemany(self, sql: str, seq):
        cur = self._raw.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.executemany(_pg_sql(sql), [tuple(p) for p in seq])
        return cur

    def cursor(self, *args, **kwargs):
        # pandas.read_sql_query drives the raw DBAPI cursor protocol directly.
        return self._raw.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def commit(self):
        self._raw.commit()

    def close(self):
        """Return the connection to the pool (pooled) or close it (fallback)."""
        if self._pool is not None:
            try:
                self._pool.putconn(self._raw)
            except Exception:
                try: self._raw.close()
                except Exception: pass
        else:
            self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._raw.commit()
            else:
                self._raw.rollback()
        finally:
            self.close()
        return False


# --- Postgres connection pool -----------------------------------------------
# Every get_conn() used to open a brand-new TCP+TLS connection to the hosted
# database (100-300ms each to e.g. Supabase) — multiplied by the scan loop's
# per-ticker writes and by concurrent Streamlit sessions. A module-level
# ThreadedConnectionPool amortises that to near-zero, and the SELECT 1 pre-ping
# on checkout transparently replaces connections the server dropped while idle
# (hosted Postgres commonly reaps idle sockets), so callers never see a stale
# one. Every call site is unchanged: _PgConn.close()/__exit__ now *release*
# instead of closing.
_PG_POOL = None
_PG_POOL_LOCK = threading.Lock()


def _pg_pool():
    global _PG_POOL
    if _PG_POOL is None:
        with _PG_POOL_LOCK:
            if _PG_POOL is None:
                _PG_POOL = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1, maxconn=6, dsn=_db_url())
    return _PG_POOL


def _pg_checkout() -> "_PgConn":
    """Checkout with pre-ping: a pooled connection that fails SELECT 1 (dropped by
    the server while idle) is discarded and replaced. If the pool itself is broken
    or exhausted, fall back to one direct connection so the request still succeeds —
    _PgConn knows to really close that one instead of returning it to the pool."""
    try:
        pool = _pg_pool()
    except Exception:
        return _PgConn(psycopg2.connect(_db_url()), pool=None)
    for _ in range(2):
        try:
            raw = pool.getconn()
        except Exception:
            break
        try:
            with raw.cursor() as cur:
                cur.execute("SELECT 1")
            raw.rollback()            # clear the ping's implicit transaction
            return _PgConn(raw, pool=pool)
        except Exception:
            try:
                pool.putconn(raw, close=True)   # stale/dead -> drop from the pool
            except Exception:
                pass
    return _PgConn(psycopg2.connect(_db_url()), pool=None)


def get_conn():
    if db_backend() == "postgres":
        return _pg_checkout()
    # SQLite path — unchanged semantics.
    # check_same_thread=False because Streamlit serves reruns on a pool of threads;
    # timeout gives writers up to 10s to wait out a lock instead of erroring out;
    # WAL journaling lets reads and a writer proceed concurrently, which together
    # prevent the 'database is locked' OperationalError under multi-session load.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except sqlite3.Error:
        pass   # some network filesystems reject WAL; fall back to default journaling
    return conn

def _ensure_column(conn, table: str, column: str,
                   decl: str = "REAL DEFAULT 0.0") -> None:
    """Add `column` to `table` if it isn't already present (safe schema migration)."""
    if db_backend() == "postgres":
        # Postgres types: REAL is only 4 bytes there; store doubles like sqlite does.
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} "
                     f"{decl.replace('REAL', 'DOUBLE PRECISION')}")
        return
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _ddl(sql: str) -> str:
    """Translate CREATE TABLE statements for the active backend. The app's DDL is
    written in sqlite dialect; for Postgres swap the auto-id and widen floats."""
    if db_backend() == "postgres":
        return (sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                   .replace(" REAL", " DOUBLE PRECISION"))
    return sql

def init_db() -> None:
    with get_conn() as conn:
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL, rec_date TEXT NOT NULL, recommendation TEXT NOT NULL,
                composite REAL NOT NULL, price_at_rec REAL, momentum REAL, value REAL, 
                technical REAL, hype REAL, quality REAL, price_after REAL, outcome INTEGER, eval_date TEXT
            )"""))
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS kpi_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT, update_date TEXT NOT NULL,
                momentum REAL, value REAL, technical REAL, hype REAL, quality REAL, note TEXT
            )"""))
        # Walk-forward optimisation tables (new). CREATE ... IF NOT EXISTS is
        # inherently non-destructive, so existing data is never touched.
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS mock_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, ticker TEXT NOT NULL,
                recommendation_price REAL, reason TEXT, kpi_snapshot TEXT,
                evaluated INTEGER DEFAULT 0
            )"""))
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS system_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, factor_name TEXT NOT NULL, current_weight REAL
            )"""))
        # Deep-scan winners promoted into the main scan universe. NOTE: on Streamlit
        # Cloud this file-backed table is wiped on restart/redeploy — the persistent
        # layer is PROMOTED_TICKERS in config.py (the app offers a paste-able snippet).
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS promoted_tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL, ticker TEXT NOT NULL, name TEXT,
                composite REAL, promoted_at TEXT NOT NULL
            )"""))
        # Live screener names merged into each market's deep-scan universe (rotates
        # on every "Refresh universe" click; curated lists remain the quality floor).
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS dynamic_universe (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL, ticker TEXT NOT NULL,
                rank INTEGER, fetched_at TEXT NOT NULL
            )"""))
        # Per-theme aggregates recorded once per FULL scan (quick scans skipped) so
        # the Themes tab can show momentum deltas vs the previous scan. Local-only
        # persistence: on Streamlit Cloud this resets at every redeploy.
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS theme_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot TEXT NOT NULL, theme TEXT NOT NULL,
                momentum REAL, hype REAL, composite REAL,
                n INTEGER, cover INTEGER, created_at TEXT NOT NULL
            )"""))
        # Full scan-result snapshots (JSON payload, history frames stripped) so the
        # last scan survives a page refresh / new session instead of living only in
        # st.session_state. Pruned to the newest few rows on every save.
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS scan_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL, is_quick INTEGER DEFAULT 1,
                n_results INTEGER, payload TEXT
            )"""))
        # --- Safe migration for pre-existing stock_engine.db files ---
        # Databases created before the "quality" factor existed are missing the
        # column. Add it (defaulting old rows to 0.0) only when absent, so this is
        # a no-op on fresh installs and never errors on an upgrade.
        _ensure_column(conn, "recommendations", "quality")
        _ensure_column(conn, "kpi_weights", "quality")
        # Theme (industry-rotation) factor — sixth KPI, added the same safe way.
        _ensure_column(conn, "recommendations", "theme")
        _ensure_column(conn, "kpi_weights", "theme")
        # Bookkeeping flag so each logged pick feeds the walk-forward loop once.
        _ensure_column(conn, "mock_portfolio", "evaluated", "INTEGER DEFAULT 0")
        # Company name stored alongside the ticker for a friendlier audit table.
        _ensure_column(conn, "mock_portfolio", "name", "TEXT")
        # Indexes for the hot queries: per-day replace in save_recommendation,
        # the pending-outcome sweep, and the walk-forward's evaluated=0 select.
        # CREATE INDEX IF NOT EXISTS is valid on both sqlite and Postgres.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reco_ticker_date ON recommendations(ticker, rec_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reco_date_outcome ON recommendations(rec_date, outcome)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mock_evaluated ON mock_portfolio(evaluated)")
        conn.commit()
    if get_weight_history().empty:
        save_weights(DEFAULT_WEIGHTS, note="initial defaults")

def get_latest_weights() -> dict[str, float]:
    with get_conn() as conn:
        row = conn.execute("SELECT momentum, value, technical, hype, quality, theme FROM kpi_weights ORDER BY id DESC LIMIT 1").fetchone()
    if row is None: return dict(DEFAULT_WEIGHTS)
    raw = {f: safe_float(row[f], DEFAULT_WEIGHTS[f]) for f in FACTORS}
    # A pre-migration weight row carries quality=0.0 (the ALTER TABLE default). The
    # auto-tuner floors every genuine weight at MIN_WEIGHT, so any value <= 0 is a
    # migration artifact: substitute the default and renormalise so the active
    # weights still sum to 1.0 (clean rows already do, making this a no-op for them).
    raw = {f: (w if w > 0 else DEFAULT_WEIGHTS[f]) for f, w in raw.items()}
    total = sum(raw.values()) or 1.0
    return {f: raw[f] / total for f in FACTORS}

def save_weights(weights: dict[str, float], note: str = "") -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO kpi_weights (update_date, momentum, value, technical, hype, quality, theme, note) VALUES (?,?,?,?,?,?,?,?)",
            (ts, weights["momentum"], weights["value"], weights["technical"], weights["hype"], weights["quality"], weights["theme"], note),
        )
        # Mirror each factor into the long-format system_weights ledger (one row per
        # factor per change) so the walk-forward audit can track them individually.
        conn.executemany(
            "INSERT INTO system_weights (timestamp, factor_name, current_weight) VALUES (?,?,?)",
            [(ts, f, float(weights[f])) for f in FACTORS],
        )
        conn.commit()

def get_weight_history() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT update_date, momentum, value, technical, hype, quality, theme, note FROM kpi_weights ORDER BY id ASC", conn)
    if not df.empty: df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")
    return df

def save_recommendation(rec: dict) -> None:
    """Persist one recommendation per ticker per day.

    FIX: the analysis dict produced by analyze_ticker stores the price under
    "price" (not "price_at_rec") and stores the full hype breakdown dict under
    "hype" (the numeric score is "hype_score"). Reading rec["price_at_rec"] or
    binding the hype *dict* into a REAL column raised on every ticker and sent
    them all to the "failed" list. We map both robustly here.
    """
    today = date.today().isoformat()

    # Accept either the analysis dict (key "price") or a pre-built record.
    price_val = safe_float(rec.get("price_at_rec", rec.get("price")))

    # Pull the numeric hype score, never the breakdown dict.
    hype_val = rec.get("hype_score", rec.get("hype", 0.0))
    if isinstance(hype_val, dict):
        hype_val = safe_float(hype_val.get("score", 0.0), 0.0)
    else:
        hype_val = safe_float(hype_val, 0.0)

    quality_val = safe_float(rec.get("quality"))

    with get_conn() as conn:
        conn.execute(
            "DELETE FROM recommendations WHERE ticker=? AND rec_date=? AND outcome IS NULL",
            (rec["ticker"], today),
        )
        conn.execute(
            "INSERT INTO recommendations (ticker, rec_date, recommendation, composite, price_at_rec, momentum, value, technical, hype, quality, theme) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rec["ticker"], today, rec["recommendation"], float(rec["composite"]),
             price_val, safe_float(rec["momentum"]), safe_float(rec["value"]),
             safe_float(rec["technical"]), hype_val, quality_val,
             safe_float(rec.get("theme"), 50.0)),
        )
        conn.commit()

def get_recommendations() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM recommendations ORDER BY rec_date DESC, id DESC", conn)
    if not df.empty: df["rec_date"] = pd.to_datetime(df["rec_date"], errors="coerce")
    return df

# ----------------------------------------------------------------------------
# Deep-scan promotions — a small rotating layer of deep-scan winners that rides
# along with the standard TICKER_UNIVERSE on every main scan (quota: US 5 / JP 3 /
# CN 2). Two layers, merged: the SQLite table (written by the Deep Scan tab's
# Promote button; ephemeral on Streamlit Cloud) takes priority, then the
# PROMOTED_TICKERS block in config.py (the persistent, hand-pasted layer). Names
# already anywhere in the base universe are dropped, and each region is capped to
# its quota — so the total scan is at most base + 10.
# ----------------------------------------------------------------------------
_BASE_UNIVERSE_SET = {t for ticks in TICKER_UNIVERSE.values() for t in ticks}

def get_promoted_universe() -> dict[str, list[str]]:
    db_layer: dict[str, list[str]] = {}
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT region, ticker FROM promoted_tickers ORDER BY id ASC").fetchall()
        for r in rows:
            db_layer.setdefault(r["region"], []).append(r["ticker"])
    except Exception:
        pass   # a missing/locked DB must never break universe assembly
    out: dict[str, list[str]] = {}
    for region, quota in PROMOTION_QUOTA.items():
        merged, seen = [], set()
        for t in db_layer.get(region, []) + list(PROMOTED_TICKERS.get(region, [])):
            if t and t not in seen and t not in _BASE_UNIVERSE_SET:
                merged.append(t); seen.add(t)
            if len(merged) >= quota:
                break
        if merged:
            out[region] = merged
    return out

def save_promotions(region: str, picks: list[tuple[str, str, float]]) -> None:
    """Replace `region`'s DB promotion layer with `picks` [(ticker, name, composite)]."""
    ts = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.execute("DELETE FROM promoted_tickers WHERE region=?", (region,))
        conn.executemany(
            "INSERT INTO promoted_tickers (region, ticker, name, composite, promoted_at) VALUES (?,?,?,?,?)",
            [(region, t, n, safe_float(c, 0.0), ts) for t, n, c in picks],
        )
        conn.commit()

def clear_promotions() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM promoted_tickers")
        conn.commit()

def effective_universe() -> dict[str, list[str]]:
    """Base TICKER_UNIVERSE plus the promoted deep-scan layer, per region, deduped."""
    promoted = get_promoted_universe()
    out = {}
    for region, ticks in TICKER_UNIVERSE.items():
        extra = [t for t in promoted.get(region, []) if t not in ticks]
        out[region] = list(ticks) + extra
    return out

def promotions_config_snippet() -> str:
    """Ready-to-paste config.py block reflecting the CURRENT merged promotions, so a
    Streamlit Cloud user can persist them locally before a redeploy wipes the DB."""
    promoted = get_promoted_universe()
    lines = ["PROMOTED_TICKERS = {"]
    for region in PROMOTION_QUOTA:
        ticks = ", ".join(f'"{t}"' for t in promoted.get(region, []))
        lines.append(f'    "{region}": [{ticks}],')
    lines.append("}")
    return "\n".join(lines)

# ----------------------------------------------------------------------------
# LIVE deep-universe layer — Yahoo screener (top market cap + biggest movers)
# merged with the hand-curated deep lists, so the funnel's INPUT rotates with the
# market instead of being frozen at the knowledge cutoff. Curated lists are the
# quality floor and always remain; the screener adds churn: new listings, names
# that grew into the cap floor, and the day's genuine movers. Fully fail-safe —
# the screener is an unofficial endpoint and is often blocked on cloud IPs, in
# which case the deep scan simply runs on the curated list as before.
# ----------------------------------------------------------------------------
def refresh_dynamic_universe(region: str) -> list[str]:
    """Query Yahoo's screener for `region`. Two pulls per Yahoo region code:
    top-N by market cap (stability, data quality) and top movers above the same
    cap floor (discovery). Returns deduped symbols filtered to the market's
    expected listing suffixes; [] on any failure. Never raises."""
    spec = DYNAMIC_UNIVERSE_SPEC.get(region)
    if not spec or yf is None or not hasattr(yf, "screen") or not hasattr(yf, "EquityQuery"):
        return []
    allowed = DYNAMIC_ALLOWED_SUFFIXES.get(region, {""})
    symbols, seen = [], set()

    def _take(resp) -> None:
        for q in (resp or {}).get("quotes", []):
            sym = str(q.get("symbol") or "").upper().strip()
            if not sym or sym in seen or "=" in sym or "^" in sym:
                continue
            if q.get("quoteType") not in (None, "EQUITY"):
                continue
            suffix = "." + sym.rsplit(".", 1)[1] if "." in sym else ""
            if suffix not in allowed:
                continue
            seen.add(sym)
            symbols.append(sym)

    for code in spec["regions"]:
        base = yf.EquityQuery("and", [
            yf.EquityQuery("eq", ["region", code]),
            yf.EquityQuery("gt", ["intradaymarketcap", spec["min_cap"]]),
        ])
        try:   # largest names first: keeps data quality high and the list current
            _take(yf.screen(base, sortField="intradaymarketcap", sortAsc=False,
                            size=spec["top_cap"]))
        except Exception as e:
            logger.warning("screener market-cap pull failed for %s/%s: %s", region, code, e)
        try:   # movers overlay: today's biggest gainers above the same cap floor
            _take(yf.screen(base, sortField="percentchange", sortAsc=False,
                            size=spec["movers"]))
        except Exception as e:
            logger.warning("screener movers pull failed for %s/%s: %s", region, code, e)
    return symbols

def save_dynamic_universe(region: str, tickers: list[str]) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.execute("DELETE FROM dynamic_universe WHERE region=?", (region,))
        conn.executemany(
            "INSERT INTO dynamic_universe (region, ticker, rank, fetched_at) VALUES (?,?,?,?)",
            [(region, t, i, ts) for i, t in enumerate(tickers)],
        )
        conn.commit()

def get_dynamic_universe(region: str) -> tuple[list[str], str | None]:
    """This market's stored screener names and when they were fetched (ISO), if any."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT ticker, fetched_at FROM dynamic_universe WHERE region=? ORDER BY rank ASC",
                (region,)).fetchall()
    except Exception:
        return [], None
    if not rows:
        return [], None
    return [r["ticker"] for r in rows], rows[0]["fetched_at"]

def deep_universe_for(region: str) -> tuple[list[str], int]:
    """Curated deep list + live screener names (deduped). Returns (tickers, n_live)."""
    curated = list(DEEP_UNIVERSES.get(region, []))
    dyn, _ = get_dynamic_universe(region)
    cur_set = set(curated)
    extra = [t for t in dyn if t not in cur_set]
    return curated + extra, len(extra)

def match_theme(region: str, sector: str | None, industry: str | None) -> str | None:
    """Classify a stock into one of the THEMES baskets from its Yahoo sector/industry
    (captured for free during stage-2 fundamentals). First keyword match wins,
    specific industries before catch-alls; China + internet routes to China Internet.
    Returns None when nothing fits — not every stock belongs to a theme."""
    blob = f"{sector or ''} {industry or ''}".lower().strip()
    if not blob:
        return None
    if region == "China" and any(k in blob for k in ("internet", "e-commerce", "online")):
        return "China Internet"
    for theme, kws in THEME_INDUSTRY_KEYWORDS:
        if any(k in blob for k in kws):
            return theme
    return None

# ----------------------------------------------------------------------------
# Quantitative Math & Indicators
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _stooq_symbol(ticker: str) -> str | None:
    """Map a US ticker to its Stooq symbol (AAPL -> aapl.us). None for suffixed/non-US
    tickers — Stooq is our US-only, keyless price fallback."""
    return None if "." in ticker else ticker.lower() + ".us"


def _period_to_days(period: str) -> int | None:
    m = re.match(r"(\d+)\s*(d|wk|mo|y)?", str(period).strip().lower())
    if not m:
        return None
    n, unit = int(m.group(1)), (m.group(2) or "d")
    return {"d": 1, "wk": 7, "mo": 30, "y": 365}.get(unit, 1) * n


def fetch_history_stooq(ticker: str, period: str = "8mo") -> "pd.DataFrame | None":
    """Free, keyless EOD price history from Stooq, used as a fallback when yfinance is
    throttled (US tickers only). Returns a yfinance-shaped frame (Date index, OHLCV with
    a 'Close' column) trimmed to `period`, or None. Never raises."""
    sym = _stooq_symbol(ticker)
    if sym is None:
        return None
    try:
        raw = _reddit_get(f"https://stooq.com/q/d/l/?s={sym}&i=d", {"User-Agent": "Mozilla/5.0"})
    except Exception:
        return None
    lines = raw.strip().splitlines()
    if not lines or not lines[0].lower().startswith("date"):
        return None   # '<no data>', error page, or a block — degrade quietly
    recs = []
    for line in lines[1:]:
        p = line.split(",")
        if len(p) < 5:
            continue
        try:
            d = pd.to_datetime(p[0])
            o, h, l, c = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        except Exception:
            continue
        v = float(p[5]) if len(p) > 5 and p[5] not in ("", "N/D") else float("nan")
        recs.append((d, o, h, l, c, v))
    if not recs:
        return None
    df = (pd.DataFrame(recs, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
          .set_index("Date").sort_index())
    days = _period_to_days(period)
    if days:
        df = df[df.index >= (df.index.max() - pd.Timedelta(days=days))]
    return df if not df.empty else None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker: str, period: str = "8mo") -> pd.DataFrame:
    """Fetch OHLCV with retry/backoff so a throttled burst doesn't wipe a scan.

    Cached 30 min per (ticker, period): repeat scans, the Deep-Dive lazy refetch,
    and fallback paths reuse the frame instead of re-hitting Yahoo. RAISES
    RuntimeError when nothing comes back after the retries. @st.cache_data
    never caches an exception, so a transient throttle is retried next call instead
    of being frozen as a permanent None for the whole TTL. Callers that loop over
    many tickers wrap this in try/except and skip the offending name.
    """
    if yf is None:
        raise RuntimeError("yfinance unavailable")
    last_err = None
    for attempt in range(3):
        try:
            df = _ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                return df.dropna(subset=["Close"])
        except Exception as e:
            last_err = e
        time.sleep(1.2 * (attempt + 1))   # back off, then retry
    # yfinance exhausted -> try the free Stooq fallback (US tickers) before giving up,
    # so a Yahoo throttle degrades to EOD data instead of dropping the name entirely.
    stq = fetch_history_stooq(ticker, period)
    if stq is not None and not stq.empty and "Close" in stq.columns:
        return stq.dropna(subset=["Close"])
    raise RuntimeError(f"no price history for {ticker} after retries") from last_err

@st.cache_data(ttl=1800, show_spinner=False)
def _fmp_fundamentals(ticker: str) -> dict:
    """Optional secondary fundamentals from Financial Modeling Prep, used only when
    yfinance fails or returns empty (its `.info` is the flakiest source in the app).
    Enabled by setting FMP_API_KEY (st.secrets or env); otherwise a no-op. Free-tier
    coverage is US-focused, so suffixed/non-US tickers are skipped. Fail-safe -> {}.
    Values are normalized to the app's conventions: roe as a fraction, div_yield as a
    percent. short_pct isn't available from these endpoints, so it stays NaN."""
    key = _get_secret("FMP_API_KEY")
    if not key or "." in ticker:
        return {}
    base = "https://financialmodelingprep.com/api/v3"
    hdr = {"User-Agent": "Mozilla/5.0"}
    out = {"pe": float("nan"), "div_yield": float("nan"), "market_cap": float("nan"),
           "roe": float("nan"), "short_pct": float("nan"), "name": ticker,
           "sector": "", "industry": ""}
    got = False
    try:
        prof = json.loads(_reddit_get(f"{base}/profile/{ticker}?apikey={key}", hdr))
        if isinstance(prof, list) and prof:
            p = prof[0]
            out["market_cap"] = safe_float(p.get("mktCap"))
            out["pe"] = safe_float(p.get("pe"))
            out["name"] = str(p.get("companyName") or ticker)
            out["sector"] = str(p.get("sector") or "")
            out["industry"] = str(p.get("industry") or "")
            got = True
    except Exception as e:
        logger.warning("FMP profile fallback failed for %s: %s", ticker, e)
    try:
        rat = json.loads(_reddit_get(f"{base}/ratios-ttm/{ticker}?apikey={key}", hdr))
        if isinstance(rat, list) and rat:
            r = rat[0]
            pe = r.get("peRatioTTM", r.get("priceEarningsRatioTTM"))
            if pe is not None and math.isnan(out["pe"]):
                out["pe"] = safe_float(pe)
            out["roe"] = safe_float(r.get("returnOnEquityTTM", r.get("roeTTM")))   # fraction
            dy = r.get("dividendYieldTTM")
            if dy is not None:
                out["div_yield"] = safe_float(dy) * 100.0   # FMP fraction -> app percent
            got = True
    except Exception as e:
        logger.warning("FMP ratios fallback failed for %s: %s", ticker, e)
    return out if got else {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    out = {"pe": float("nan"), "div_yield": float("nan"), "market_cap": float("nan"),
           "roe": float("nan"), "short_pct": float("nan"), "name": ticker,
           "sector": "", "industry": ""}
    if yf is None: return out
    # Distinguish a transient failure (rate-limit / network) from a reachable ticker
    # that simply lacks some fields. On the former we RAISE — @st.cache_data never
    # caches an exception, so the broken state isn't frozen for the whole TTL and the
    # next scan retries. Field-level gaps stay as NaN and cache normally.
    try:
        time.sleep(0.15)   # pace the un-bulkable .info endpoint; cache misses only
        info = _ticker(ticker).info
    except Exception as e:
        fb = _fmp_fundamentals(ticker)   # secondary source kicks in only on failure
        if fb:
            logger.info("fundamentals for %s served by FMP fallback (yfinance error)", ticker)
            return fb
        raise RuntimeError(f"fundamentals fetch failed for {ticker} (likely rate-limited)") from e
    if not info:
        fb = _fmp_fundamentals(ticker)
        if fb:
            logger.info("fundamentals for %s served by FMP fallback (empty yfinance)", ticker)
            return fb
        raise RuntimeError(f"empty fundamentals for {ticker} (likely rate-limited)")
    out["pe"] = safe_float(info.get("trailingPE"))
    out["market_cap"] = safe_float(info.get("marketCap"))
    out["name"] = str(info.get("shortName") or info.get("longName") or ticker)
    out["div_yield"] = resolve_div_yield(info)
    # New fundamentals for the quality factor and short-squeeze modifier. safe_float
    # keeps these as NaN when yfinance omits them, so downstream scoring never breaks.
    out["roe"] = safe_float(info.get("returnOnEquity"))
    out["short_pct"] = safe_float(info.get("shortPercentOfFloat"))
    # Sector/industry ride along for free from the same .info call — they feed the
    # Deep Scan <-> Themes bridge (match_theme) at zero extra API cost.
    out["sector"] = str(info.get("sector") or "")
    out["industry"] = str(info.get("industry") or "")
    return out

def resolve_div_yield(info: dict) -> float:
    """Return an annual dividend yield in PERCENT, or NaN.

    yfinance's `dividendYield` field is inconsistent across versions — sometimes a
    fraction (0.004), sometimes already a percent (0.4) — which can scale a real
    ~0.4% yield up to 15%+. The unambiguous route is dividend-rate / price, so we
    compute that first and only fall back to the raw field, then cap obviously
    broken values (equity yields above ~25% are virtually always a data error).
    """
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    rate = safe_float(info.get("dividendRate") or info.get("trailingAnnualDividendRate"))

    dy = float("nan")
    if not math.isnan(rate) and not math.isnan(price) and price > 0:
        dy = rate / price * 100.0
    else:
        raw = safe_float(info.get("dividendYield"))
        if not math.isnan(raw):
            dy = raw if raw > 1 else raw * 100.0   # normalize fraction vs percent

    if math.isnan(dy) or dy < 0 or dy > 25:
        return float("nan")
    return dy


def _parse_jp_forum(html: str) -> tuple[float, float] | None:
    """Extract (買いたい%, 売りたい%) from a Yahoo! Japan Finance forum page. The
    markup is unofficial and can change, so this is deliberately tolerant — but
    binding a percentage to a label is ambiguity-prone (in `買いたい：72% 売りたい：9%`
    the naive percent-before-label pattern lets 売りたい steal the 72). So: pass 1
    binds label→following-% (unambiguous), claiming those percents; pass 2 binds any
    still-unresolved label to the nearest UNCLAIMED preceding % (the screenshot's
    `64% 買いたい` layout). Returns None rather than guessing."""
    if not html:
        return None
    labels = ("買いたい", "売りたい")
    got: dict[str, float] = {}
    claimed: set[int] = set()
    # pass 1: label ... NN%   (claim the percent's position)
    for lab in labels:
        m = re.search(rf"{lab}[^0-9%]{{0,20}}(\d{{1,3}})\s*%", html)
        if m and 0.0 <= float(m.group(1)) <= 100.0:
            got[lab] = float(m.group(1))
            claimed.add(m.start(1))
    # pass 2: NN% ... label   (only percents nobody claimed, window free of labels)
    for lab in labels:
        if lab in got:
            continue
        for m in re.finditer(rf"(\d{{1,3}})\s*%([^0-9%]{{0,20}}){lab}", html):
            gap = m.group(2)
            if m.start(1) not in claimed and not any(l in gap for l in labels) \
                    and 0.0 <= float(m.group(1)) <= 100.0:
                got[lab] = float(m.group(1))
                claimed.add(m.start(1))
                break
    if len(got) < 2:
        mb = re.search(r'"(?:buy|bullish)(?:Ratio|Percent)"\s*:\s*(\d{1,3})', html)
        ms = re.search(r'"(?:sell|bearish)(?:Ratio|Percent)"\s*:\s*(\d{1,3})', html)
        if mb and ms:
            return (float(mb.group(1)), float(ms.group(1)))
        return None
    return (got["買いたい"], got["売りたい"])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_jp_forum_rating(ticker: str) -> tuple[float, float] | None:
    """みんなの評価 poll for one Japanese ticker (####.T) from the Yahoo! Japan
    Finance forum page. Unofficial HTML endpoint: cached 30 min, courtesy-throttled,
    fail-safe (None on any error/blocked/markup change — hype falls back to the
    volume signal alone). Personal-use scraping; Yahoo! JP could block or change it."""
    if not ticker.endswith(".T"):
        return None
    try:
        html = _reddit_get(f"https://finance.yahoo.co.jp/quote/{ticker}/forum",
                           {"User-Agent": "Mozilla/5.0", "Accept-Language": "ja"})
        time.sleep(0.25)   # only on cache miss; be polite to the page
        return _parse_jp_forum(html)
    except Exception as e:
        logger.warning("JP forum fetch failed for %s: %s", ticker, e)
        return None


def analyze_ticker(ticker: str, region: str, hype_mentions: float = 0.0,
                   jp_forum: tuple | None = None,
                   hist: pd.DataFrame | None = None) -> dict | None:
    # `hist` lets bulk callers (run_engine / run_deep_scan) hand in the frame from
    # ONE yf.download for the whole universe instead of paying a per-ticker request;
    # names the bulk call missed fall back to the per-ticker fetch (retry + Stooq).
    if hist is None:
        hist = fetch_history(ticker)
    if hist is None or len(hist) < 60: return None
    close, volume = hist["Close"], hist.get("Volume", pd.Series(dtype=float))
    funds = fetch_fundamentals(ticker)

    price = safe_float(close.iloc[-1])
    sma20 = safe_float(close.rolling(20).mean().iloc[-1])
    sma50 = safe_float(close.rolling(50).mean().iloc[-1])
    rsi = safe_float(compute_rsi(close).iloc[-1])
    macd_line, signal_line, hist_macd = compute_macd(close)
    _, bb_up, bb_lo, bb_pct = compute_bollinger(close)
    # 1-month return = 21 TRADING days back, i.e. iloc[-22] relative to the last
    # bar (iloc[-21] was only 20 days back — a subtle off-by-one).
    ret_1m = safe_float((price / close.iloc[-22] - 1) * 100) if len(close) >= 22 else float("nan")
    hype = compute_hype(volume)

    # Scoring Engine Logic
    mom = 50.0 + (clamp((price / sma50 - 1) * 200, -25, 25) if sma50 > 0 else 0)
    mom += 12 if safe_float(hist_macd.iloc[-1]) > 0 else -12
    momentum_score = clamp(mom + clamp(ret_1m if not math.isnan(ret_1m) else 0.0, -13, 13))

    pe = funds["pe"]
    value_score = 50.0 if math.isnan(pe) or pe <= 0 else clamp(100 - (pe - 10) * 2.0)
    if not math.isnan(funds["div_yield"]): value_score = clamp(value_score + min(funds["div_yield"] * 3, 10))

    rsi_safe = rsi if not math.isnan(rsi) else 50.0
    tech = 50.0 - abs(rsi_safe - 50) * 0.6 + (15 if 45 <= rsi_safe <= 65 else 0)
    pct_b = safe_float(bb_pct.iloc[-1])
    technical_score = clamp(tech + (15 if (not math.isnan(pct_b) and 0.2 <= pct_b <= 0.8) else -10))

    # Quality factor — rewards return-on-equity (profitability). A missing ROE is
    # treated as neutral (50) rather than penalised.
    roe = funds["roe"]
    quality_score = 50.0 if math.isnan(roe) else clamp(50 + (roe * 200))

    # Social & news buzz — a live retail/news-sentiment kicker combining every
    # enabled source for this market (Reddit, GDELT news, …). Each mention adds 10
    # points to the hype score (capped at +35), then the whole score is re-clamped
    # to 0-100. Zero mentions (or a failed fetch) leave hype untouched.
    if hype_mentions > 0:
        hype["score"] = clamp(hype["score"] + min(35, hype_mentions * 10))

    # Japan-native sentiment: Yahoo!掲示板 みんなの評価 poll, blended 50/50 into the
    # hype score when available. This gives Japanese names a real sentiment signal
    # (Reddit/GDELT are English-centric); the transform (net bullishness, damped)
    # is forum_sentiment_score in indicators.py. Fail-safe: no poll -> volume-only.
    if jp_forum is not None:
        fs = forum_sentiment_score(jp_forum[0], jp_forum[1])
        if fs is not None:
            hype["score"] = clamp(0.5 * hype["score"] + 0.5 * fs)

    # Short-squeeze modifier — a heavily-shorted name (>10% of float) that is also
    # printing sustained volume breakouts can squeeze violently, so we boost its
    # hype score by +30 (capped at 100). The boost flows into hype_score below.
    short_pct = funds["short_pct"]
    if not math.isnan(short_pct) and short_pct > 0.10 and hype["sustained"]:
        hype["score"] = clamp(hype["score"] + 30.0)

    return {
        "ticker": ticker, "region": region, "name": funds["name"], "price": price,
        "sma20": sma20, "sma50": sma50, "rsi": rsi, "macd_hist": safe_float(hist_macd.iloc[-1]),
        "bb_pct": pct_b, "ret_1m": ret_1m, "pe": pe, "div_yield": funds["div_yield"],
        "market_cap": funds["market_cap"], "roe": roe, "short_pct": short_pct,
        "sector": funds.get("sector", ""), "industry": funds.get("industry", ""),
        "hype": hype, "momentum": momentum_score, "value": value_score,
        "technical": technical_score, "quality": quality_score,
        # Theme (industry-rotation) KPI: neutral until the caller's cross-sectional
        # second pass fills it in (run_engine / run_deep_scan) — the score depends on
        # the OTHER scanned members of the basket, which a per-ticker function can't
        # see. Neutral 50 contributes weight*50 to the composite, i.e. no tilt.
        "theme": 50.0, "theme_match": None,
        "hype_score": hype["score"], "hype_mentions": hype_mentions,
        "jp_bull": (jp_forum[0] if jp_forum else float("nan")),
        # Keep ONLY the Close column of the history frame in the result dict: the
        # sole post-scan consumer is the Deep-Dive price chart (Close + SMAs derived
        # from Close). Storing the full 7-column OHLCV frame per ticker kept ~85%
        # dead weight alive in st.session_state for the whole session — multiplied
        # by the universe size and by every concurrent user, a real contributor to
        # Streamlit Cloud's memory limit (which kills the app until a reboot).
        "jp_bear": (jp_forum[1] if jp_forum else float("nan")), "history": hist[["Close"]]
    }

def score_with_weights(analysis: dict, weights: dict[str, float]) -> dict:
    composite = sum(weights[f] * analysis[f if f != "hype" else "hype_score"] for f in FACTORS)
    rec = "BUY" if composite >= BUY_THRESHOLD else "SELL" if composite < SELL_THRESHOLD else "HOLD"
    return {"composite": float(composite), "recommendation": rec}

# ----------------------------------------------------------------------------
# Sell-Signal Scanner
# ----------------------------------------------------------------------------
# A separate, ticker-driven view that surfaces *bearish* indicators. Each KPI
# returns a 0-100 sub-score where HIGHER = stronger sell pressure; the composite
# is a weighted average over whichever KPIs actually returned data (so a symbol
# with no analyst/insider coverage isn't unfairly scored on missing fields).
SELL_HIGH = 65.0          # composite >= this  -> "elevated" sell pressure
SELL_LOW = 45.0           # composite <  this  -> "low" sell pressure
SELL_LOOKBACK_DAYS = 180  # window for analyst rating changes & insider filings

SELL_KPI_WEIGHTS = {
    "analyst": 0.20,      # consensus rating (1 strong buy .. 5 strong sell)
    "downgrades": 0.18,   # recent analyst rating changes (hold->sell etc.)
    "target": 0.15,       # price vs mean analyst price target
    "insider": 0.17,      # net insider selling
    "technical": 0.15,    # trend breakdown (moving averages, MACD)
    "momentum": 0.10,     # recent price momentum
    "short": 0.05,        # short interest as bearish positioning
    "forum": 0.05,        # Yahoo!掛示板 euphoria (JP only; asymmetric, skipped elsewhere)
}
_SELL_GRADES = ("sell", "underperform", "reduce", "underweight", "negative", "strong sell")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_analyst_signals(ticker: str) -> dict:
    """Analyst consensus, mean price target, and recent rating changes."""
    out = {"rec_mean": float("nan"), "rec_key": "", "num_analysts": float("nan"),
           "target_mean": float("nan"), "currency": "", "changes": []}
    if yf is None:
        return out
    tk = _ticker(ticker)
    # RAISE on a failed/empty .info so a rate-limited response isn't cached for the
    # whole TTL (see fetch_fundamentals). A reachable name with no analyst coverage
    # returns a populated info dict but NaN fields — that's legitimate and caches.
    try:
        time.sleep(0.2)    # pace the per-ticker endpoint; runs on cache misses only
        info = tk.info
    except Exception as e:
        raise RuntimeError(f"analyst fetch failed for {ticker} (likely rate-limited)") from e
    if not info:
        raise RuntimeError(f"empty analyst info for {ticker} (likely rate-limited)")
    out["rec_mean"] = safe_float(info.get("recommendationMean"))
    out["rec_key"] = str(info.get("recommendationKey") or "")
    out["num_analysts"] = safe_float(info.get("numberOfAnalystOpinions"))
    out["target_mean"] = safe_float(info.get("targetMeanPrice"))
    out["currency"] = str(info.get("currency") or "")
    # Rating-change history is optional and frequently empty (esp. non-US); its
    # absence is not a fetch failure, so we swallow errors here rather than raise.
    try:
        ud = tk.upgrades_downgrades
        if ud is not None and not ud.empty:
            ud = ud.reset_index()
            date_col = ud.columns[0]
            parsed = pd.to_datetime(ud[date_col], errors="coerce", utc=True)
            ud[date_col] = parsed.dt.tz_convert(None)
            cutoff = pd.Timestamp(date.today() - timedelta(days=SELL_LOOKBACK_DAYS))
            recent = ud[ud[date_col] >= cutoff].sort_values(date_col, ascending=False)
            for _, row in recent.head(12).iterrows():
                d = row[date_col]
                out["changes"].append({
                    "date": d.date().isoformat() if pd.notna(d) else "",
                    "firm": str(row.get("Firm", "")),
                    "from": str(row.get("FromGrade", "")),
                    "to": str(row.get("ToGrade", "")),
                    "action": str(row.get("Action", "")),
                })
    except Exception:
        pass
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_insider_activity(ticker: str) -> dict:
    """Recent insider transactions, split into buy vs sell value/counts."""
    out = {"buy_value": 0.0, "sell_value": 0.0, "n_buys": 0, "n_sells": 0,
           "rows": [], "available": False}
    if yf is None:
        return out
    # RAISE when the underlying call throws (transient / rate-limit) so the failure
    # isn't cached. A successful call that returns an EMPTY frame is a legitimate
    # 'no insider transactions' result for this name and is cached as such.
    try:
        time.sleep(0.2)    # pace the per-ticker endpoint; runs on cache misses only
        it = _ticker(ticker).insider_transactions
    except Exception as e:
        raise RuntimeError(f"insider fetch failed for {ticker} (likely rate-limited)") from e
    if it is None or it.empty:
        return out

    cols = list(it.columns)
    date_col = next((c for c in ("Start Date", "Date", "startDate") if c in cols), None)

    def classify(row) -> str:
        txt = ""
        for c in ("Transaction", "Text"):
            if c in cols and pd.notna(row.get(c)):
                txt = str(row.get(c)).lower(); break
        if any(k in txt for k in ("sale", "sell", "sold", "dispos")):
            return "sell"
        if any(k in txt for k in ("buy", "purchase", "acqui")):
            return "buy"
        return "other"

    for _, row in it.iterrows():
        kind = classify(row)
        val = safe_float(row.get("Value"), 0.0)
        val = 0.0 if math.isnan(val) else abs(val)
        if kind == "sell":
            out["sell_value"] += val; out["n_sells"] += 1
        elif kind == "buy":
            out["buy_value"] += val; out["n_buys"] += 1
        if len(out["rows"]) < 12:
            out["rows"].append({
                "date": str(row.get(date_col)) if date_col else "",
                "insider": str(row.get("Insider", "")),
                "transaction": str(row.get("Transaction") if "Transaction" in cols else row.get("Text", "")),
                "shares": safe_float(row.get("Shares"), float("nan")),
                "value": val,
            })
    out["available"] = (out["n_buys"] + out["n_sells"]) > 0
    return out

def analyze_sell_signals(ticker: str, hist: pd.DataFrame | None = None) -> dict | None:
    """Compose every bearish KPI into a single sell-pressure profile for one ticker.

    `hist` lets a bulk caller pass in a pre-fetched 1y price history (from a single
    yf.download for the whole portfolio) so we don't make a per-ticker history
    request here. When it's None we fetch individually, as the single-ticker path does.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None
    if hist is None:
        hist = fetch_history(ticker, period="1y")   # 1y so SMA200 is available (cached)
    if hist is None or len(hist) < 60:
        return None

    # Per-ticker endpoints below are cached; each paces itself on cache misses only.
    funds = fetch_fundamentals(ticker)
    analyst = fetch_analyst_signals(ticker)
    insider = fetch_insider_activity(ticker)

    close = hist["Close"]
    price = safe_float(close.iloc[-1])
    sma50 = safe_float(close.rolling(50).mean().iloc[-1])
    sma200 = safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
    _, _, macd_hist = compute_macd(close)
    macd_h = safe_float(macd_hist.iloc[-1])
    ret_1m = safe_float((price / close.iloc[-22] - 1) * 100) if len(close) >= 22 else float("nan")
    ret_3m = safe_float((price / close.iloc[-64] - 1) * 100) if len(close) >= 64 else float("nan")

    signals: list[tuple[str, float | None, str]] = []

    # 1) Analyst consensus — 1 (strong buy) .. 5 (strong sell)
    rm = analyst["rec_mean"]
    if not math.isnan(rm) and 1.0 <= rm <= 5.0:
        sc = clamp((rm - 1.0) / 4.0 * 100.0)
        key = analyst["rec_key"].replace("_", " ") or "n/a"
        n = "" if math.isnan(analyst["num_analysts"]) else f", {int(analyst['num_analysts'])} analysts"
        signals.append(("analyst", sc, f"Consensus: {key} (mean {rm:.2f}{n})"))
    else:
        signals.append(("analyst", None, "No analyst consensus available"))

    # 2) Recent rating changes / downgrades (the 'hold -> sell' indicator)
    changes = analyst["changes"]
    if changes:
        downs = sum(1 for c in changes if c["action"].lower() == "down")
        ups = sum(1 for c in changes if c["action"].lower() == "up")
        to_sell = sum(1 for c in changes if any(g in c["to"].lower() for g in _SELL_GRADES))
        sc = clamp(45 + 18 * downs - 15 * ups + (22 if to_sell else 0))
        detail = f"{downs} downgrade(s), {ups} upgrade(s) in {SELL_LOOKBACK_DAYS // 30}mo"
        if to_sell:
            detail += f"; {to_sell} cut to a sell-type rating"
        signals.append(("downgrades", sc, detail))
    else:
        signals.append(("downgrades", None, "No recent rating changes on file"))

    # 3) Price vs mean analyst target (price above target => implied downside)
    tgt = analyst["target_mean"]
    if not math.isnan(tgt) and tgt > 0 and not math.isnan(price) and price > 0:
        upside = (tgt / price - 1.0) * 100.0
        sc = clamp(50.0 - upside * 1.5)
        signals.append(("target", sc, f"Mean target {tgt:,.2f} vs price {price:,.2f} ({upside:+.1f}% implied)"))
    else:
        signals.append(("target", None, "No price target available"))

    # 4) Insider selling
    if insider["available"]:
        sv, bv = insider["sell_value"], insider["buy_value"]
        if (sv + bv) > 0:
            ratio = sv / (sv + bv)
        else:
            ratio = insider["n_sells"] / max(1, insider["n_sells"] + insider["n_buys"])
        sc = clamp(ratio * 100.0)
        signals.append(("insider", sc, f"{insider['n_sells']} sells vs {insider['n_buys']} buys in recent filings"))
    else:
        signals.append(("insider", None, "No insider transactions reported"))

    # 5) Technical breakdown
    tech, notes = 50.0, []
    if not math.isnan(sma50) and not math.isnan(price):
        if price < sma50: tech += 15; notes.append("below SMA50")
        else: tech -= 8
    if not math.isnan(sma200) and not math.isnan(price):
        if price < sma200: tech += 15; notes.append("below SMA200")
        else: tech -= 8
    if not math.isnan(sma50) and not math.isnan(sma200) and sma50 < sma200:
        tech += 10; notes.append("death cross (SMA50<SMA200)")
    if not math.isnan(macd_h):
        if macd_h < 0: tech += 12; notes.append("bearish MACD")
        else: tech -= 6
    signals.append(("technical", clamp(tech), ", ".join(notes) if notes else "trend intact / no breakdown"))

    # 6) Momentum trend (negative returns raise sell pressure)
    mb = 50.0
    if not math.isnan(ret_1m): mb -= ret_1m * 1.3
    if not math.isnan(ret_3m): mb -= ret_3m * 0.6
    signals.append(("momentum", clamp(mb), f"1M {fmt_pct(ret_1m)}, 3M {fmt_pct(ret_3m)}"))

    # 7) Short interest — only BEARISH when momentum is also negative. A heavily
    # shorted name that is RISING may be squeezing (the exact condition the buy
    # engine in analyze_ticker treats as BULLISH), so to avoid scoring the same
    # fact as both bullish and bearish we ignore short interest here unless the
    # 1-month momentum is negative.
    sp = funds["short_pct"]
    if math.isnan(sp):
        signals.append(("short", None, "Short interest not reported"))
    elif not math.isnan(ret_1m) and ret_1m < 0:
        signals.append(("short", clamp(40 + sp * 200),
                        f"{sp * 100:.1f}% of float short, with negative 1M momentum ({fmt_pct(ret_1m)})"))
    else:
        signals.append(("short", None,
                        f"{sp * 100:.1f}% short ignored — 1M momentum not negative (possible squeeze)"))

    # 8) Yahoo!掛示板 euphoria (Japan only). ASYMMETRIC and contrarian-aware:
    # only lopsided retail *bullishness* on a .T name adds sell pressure (the retail-top
    # pattern); a bearish or balanced board is neutral, never a sell push. Skipped
    # (None) for non-Japanese names and whenever no usable poll is available, so it
    # never dilutes a US/EU/CN composite.
    if ticker.endswith(".T"):
        poll = fetch_jp_forum_rating(ticker)
        es = forum_euphoria_sell_score(poll[0], poll[1]) if poll else None
        if es is None:
            signals.append(("forum", None, "No Yahoo!掛示板 poll available"))
        else:
            signals.append(("forum", es,
                f"Yahoo!掛示板 買いたい {poll[0]:.0f}% / 売りたい {poll[1]:.0f}% "
                + ("(euphoric — possible retail top)" if es > 50 else "(not euphoric — neutral)")))
    else:
        signals.append(("forum", None, "Yahoo!掛示板 poll — Japan only"))

    # Composite over only the KPIs that returned a score
    num = sum(SELL_KPI_WEIGHTS[k] * s for k, s, _ in signals if s is not None)
    den = sum(SELL_KPI_WEIGHTS[k] for k, s, _ in signals if s is not None)
    composite = (num / den) if den > 0 else float("nan")
    if math.isnan(composite):
        verdict = "low"
    else:
        verdict = "high" if composite >= SELL_HIGH else "mixed" if composite >= SELL_LOW else "low"

    return {
        "ticker": ticker, "name": funds["name"], "price": price,
        "currency": analyst["currency"], "composite": composite, "verdict": verdict,
        "signals": signals, "changes": changes, "insider": insider,
    }

# ----------------------------------------------------------------------------
# Backtesting / Evaluation Systems
# ----------------------------------------------------------------------------
# NOTE: the OLD recommendations-based gradient (evaluate_pending +
# update_weights_from_outcomes) was removed deliberately. It wrote the same
# kpi_weights as walk_forward_update, so the two loops fought each other and muddied
# the optimisation. walk_forward_update — driven by the curated mock_portfolio KPI
# snapshots — is now the SINGLE learning loop. evaluate_outcomes_only() below still
# resolves recommendation win/lose for the audit accuracy panel, but it carries NO
# gradient and never touches kpi_weights, so it can't reignite that conflict.

def _close_on_or_after(h: pd.DataFrame | None, when) -> float:
    """First close on or after `when`, compared on pure DATES (both sides normalized)
    so we align to the right end-of-day bar and never grab an intraday/unclosed price
    through a timezone mismatch. Shared by the optimiser and the outcome evaluator."""
    if h is None or h.empty:
        return float("nan")
    idx = pd.to_datetime(h.index)
    try:
        idx = idx.tz_localize(None)       # strip tz on a tz-aware index
    except (TypeError, AttributeError):
        pass                              # already tz-naive
    dates = idx.normalize()               # 00:00 -> date-level granularity
    when_ts = pd.Timestamp(when).normalize()
    sub = h[dates >= when_ts]
    return safe_float(sub["Close"].iloc[0]) if not sub.empty else float("nan")

def evaluate_outcomes_only() -> int:
    """Resolve the forward OUTCOME of matured recommendations for the Systems Audit
    accuracy / win-rate panel — WITHOUT any weight update. This intentionally has no
    gradient: walk_forward_update() is the single learning loop; this only fills
    win/lose so the audit reflects live picks rather than only seeded history.

    Win logic mirrors the optimiser exactly: the pick's dividend-inclusive TOTAL
    return (both endpoints read from one fresh auto_adjusted series) must beat its
    HOME-market index over the same [rec_date, rec_date+horizon] window.
    """
    df = get_recommendations()
    if df.empty:
        return 0
    cutoff = date.today() - timedelta(days=EVAL_HORIZON_DAYS)
    pending = df[(df["outcome"].isna()) & (df["rec_date"].dt.date <= cutoff)]
    if pending.empty:
        return 0

    # Batch the price fetches, same pattern as walk_forward_update: one bulk
    # download for the pending tickers and their benchmarks, per-ticker fallback.
    pend_syms = list(dict.fromkeys(pending["ticker"].tolist()))
    bench_syms = sorted({benchmark_for(t) for t in pend_syms})
    bulk = get_histories(pend_syms + bench_syms, period="1y")

    bench_cache: dict[str, pd.DataFrame | None] = {s: bulk[s] for s in bench_syms if s in bulk}
    def bench_hist(sym: str) -> pd.DataFrame | None:
        if sym not in bench_cache:
            try:
                bench_cache[sym] = fetch_history(sym, period="1y")
            except Exception:
                bench_cache[sym] = None   # missing benchmark -> absolute-return fallback
        return bench_cache[sym]

    evaluated = 0
    with get_conn() as conn:
        for _, row in pending.iterrows():
            hist = bulk.get(row["ticker"])
            if hist is None:
                try:
                    hist = fetch_history(row["ticker"], period="1y")
                except Exception:
                    continue   # transient fetch failure -> skip this name, keep the batch alive
            rec_dt = row["rec_date"]
            target = rec_dt + timedelta(days=EVAL_HORIZON_DAYS)
            price_then = _close_on_or_after(hist, rec_dt)
            price_after = _close_on_or_after(hist, target)
            if math.isnan(price_after) or math.isnan(price_then) or price_then <= 0:
                continue
            stock_ret = price_after / price_then - 1.0

            bh = bench_hist(benchmark_for(row["ticker"]))
            b_then = _close_on_or_after(bh, rec_dt)
            b_after = _close_on_or_after(bh, target)
            if not math.isnan(b_then) and not math.isnan(b_after) and b_then > 0:
                win = 1 if stock_ret > (b_after / b_then - 1.0) else 0
            else:
                win = 1 if stock_ret > 0 else 0   # fallback when benchmark window missing

            conn.execute(
                "UPDATE recommendations SET price_after=?, outcome=?, eval_date=? WHERE id=?",
                (price_after, win, date.today().isoformat(), int(row["id"])),
            )
            evaluated += 1
        conn.commit()
    return evaluated

# ----------------------------------------------------------------------------
# Walk-Forward Optimisation: mock portfolio + factor-attribution feedback loop
# ----------------------------------------------------------------------------
def save_mock_portfolio(results: list[dict]) -> None:
    """Log today's Top-3 Growth and Top-3 Dividend picks with a full KPI snapshot.

    Growth = highest 1-month momentum; Dividend = highest composite among names
    yielding >= 1.5%. Idempotent per day: re-running a scan the same day refreshes
    the entry for a given ticker+reason rather than duplicating it.
    """
    if not results:
        return

    def _val(r, k, d=float("nan")):
        return safe_float(r.get(k), d)

    growth = sorted(
        [r for r in results if not math.isnan(_val(r, "ret_1m"))],
        key=lambda r: _val(r, "ret_1m"), reverse=True,
    )[:3]
    dividend = sorted(
        [r for r in results if not math.isnan(_val(r, "div_yield")) and r["div_yield"] >= 1.5],
        key=lambda r: _val(r, "composite", 0.0), reverse=True,
    )[:3]
    picks = [(r, "Top Growth") for r in growth] + [(r, "Top Dividend") for r in dividend]

    today = date.today().isoformat()
    ts = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        for r, reason in picks:
            snapshot = json.dumps({
                "momentum": round(_val(r, "momentum", 0.0), 2),
                "value": round(_val(r, "value", 0.0), 2),
                "technical": round(_val(r, "technical", 0.0), 2),
                "hype": round(_val(r, "hype_score", 0.0), 2),
                "quality": round(_val(r, "quality", 0.0), 2),
                "theme": round(_val(r, "theme", 50.0), 2),
                "composite": round(_val(r, "composite", 0.0), 2),
            })
            conn.execute(
                "DELETE FROM mock_portfolio WHERE date(timestamp)=? AND ticker=? AND reason=?",
                (today, r["ticker"], reason),
            )
            conn.execute(
                "INSERT INTO mock_portfolio (timestamp, ticker, recommendation_price, reason, kpi_snapshot, evaluated, name) VALUES (?,?,?,?,?,0,?)",
                (ts, r["ticker"], _val(r, "price"), reason, snapshot, str(r.get("name") or r["ticker"])),
            )
        conn.commit()

def get_mock_portfolio() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT timestamp, ticker, name, recommendation_price, reason FROM mock_portfolio ORDER BY id DESC", conn
        )

# ----------------------------------------------------------------------------
# Scan snapshots — persist the last scan so a refresh / new session restores it
# instead of greeting a returning user with the empty state. The price history
# frames are STRIPPED before serialising (they're the bulk of the memory and
# aren't JSON-serialisable); the Deep-Dive page lazily refetches the one frame
# it needs. Everything else in an analysis dict round-trips through json.
# ----------------------------------------------------------------------------
_SCAN_SNAPSHOT_KEEP = 5   # newest rows retained; older ones pruned on save


def _json_safe(o):
    """json.dumps default hook: numpy scalars sneak into analysis dicts (np.mean,
    Series.iloc) and json refuses them — coerce to plain Python."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def save_scan_snapshot(results: list[dict], is_quick: bool) -> None:
    """Persist one scan's results (sans history frames). Never raises — snapshot
    persistence must not be able to sink an otherwise-successful scan."""
    try:
        payload = json.dumps(
            [{k: v for k, v in r.items() if k != "history"} for r in results],
            default=_json_safe)
        ts = datetime.now().isoformat(timespec="seconds")
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO scan_snapshots (created_at, is_quick, n_results, payload) VALUES (?,?,?,?)",
                (ts, 1 if is_quick else 0, len(results), payload))
            # Prune: keep only the newest N snapshots (valid on sqlite AND Postgres).
            conn.execute(
                "DELETE FROM scan_snapshots WHERE id NOT IN "
                "(SELECT id FROM scan_snapshots ORDER BY id DESC LIMIT ?)",
                (_SCAN_SNAPSHOT_KEEP,))
            conn.commit()
    except Exception as e:
        logger.warning("scan snapshot save failed: %s", e)


def load_latest_scan_snapshot() -> tuple[list[dict], str, bool] | None:
    """(results, created_at_iso, is_quick) of the newest snapshot, or None.
    Restored dicts have no 'history' key — consumers must treat it as optional."""
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT created_at, is_quick, payload FROM scan_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        results = json.loads(row["payload"] or "[]")
        if not isinstance(results, list) or not results:
            return None
        return results, str(row["created_at"]), bool(row["is_quick"])
    except Exception as e:
        logger.warning("scan snapshot load failed: %s", e)
        return None


def _bulk_history(tickers: list, period: str = "1y") -> dict:
    """Fetch history for many tickers in ONE yf.download, returned as
    {ticker: DataFrame} so a bulk scan reuses it instead of one request per name.

    Tickers with no usable Close series (delisted, or fewer than 60 rows) are
    omitted, so the caller falls back to an individual fetch for those. Never raises.
    """
    out: dict[str, pd.DataFrame] = {}
    uniq = list(dict.fromkeys(t for t in tickers if t))
    if yf is None or not uniq:
        return out
    try:
        kwargs = dict(period=period, progress=False, threads=False,
                      group_by="ticker", auto_adjust=True)
        try:
            data = yf.download(uniq, session=_YF_SESSION, **kwargs)   # reuse SSL-aware session
        except TypeError:
            data = yf.download(uniq, **kwargs)                        # older yfinance: no session kwarg
    except Exception:
        return out
    if data is None or len(data) == 0:
        return out
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # group_by="ticker" -> column level 0 = ticker, level 1 = field.
            available = set(data.columns.get_level_values(0))
            for t in uniq:
                if t not in available:
                    continue
                frame = data[t]
                if "Close" in frame.columns:
                    sub = frame.dropna(subset=["Close"])
                    if len(sub) >= 60:
                        out[t] = sub
        elif "Close" in data.columns and len(uniq) == 1:
            sub = data.dropna(subset=["Close"])     # single ticker -> flat columns
            if len(sub) >= 60:
                out[uniq[0]] = sub
    except Exception:
        return out
    return out

_HIST_CACHE_TTL = 1800   # seconds a per-ticker frame stays fresh in the session cache

def get_histories(tickers: list, period: str = "1y") -> dict:
    """Session-level history cache in front of _bulk_history.

    Every bulk consumer (quick scan, deep scan, sell scanner, walk-forward and
    maturity updates) goes through here, so a ticker's frame is downloaded ONCE
    per session per TTL and shared across features — previously each feature paid
    its own full bulk download even for overlapping names (benchmarks, promoted
    tickers, portfolio names that were just scanned). Only the tickers that are
    missing or stale are fetched, in a single yf.download. Returns
    {ticker: DataFrame} like _bulk_history; missing names are simply absent so
    callers' per-ticker fallbacks keep working unchanged.
    """
    cache = st.session_state.setdefault("_hist_cache", {})   # {(t, period): (df, ts)}
    now = time.time()
    # Prune expired entries first: without this, frames for tickers that are never
    # requested again (rotated screener names, one-off portfolio symbols) would sit
    # in session memory for the lifetime of the session.
    for k in [k for k, (_, ts) in cache.items() if now - ts > _HIST_CACHE_TTL]:
        del cache[k]
    missing = [t for t in dict.fromkeys(t for t in tickers if t)
               if (t, period) not in cache or now - cache[(t, period)][1] > _HIST_CACHE_TTL]
    if missing:
        for t, df in _bulk_history(missing, period).items():
            cache[(t, period)] = (df, now)
    return {t: cache[(t, period)][0] for t in tickers if (t, period) in cache}

def clear_market_data_caches() -> None:
    """Escape hatch for the 30-min data TTLs: wipe the session history cache and
    the per-ticker market-data fetchers so the NEXT scan re-hits Yahoo and picks
    up sudden intraday price action instead of serving cached frames.

    Deliberately surgical — sentiment caches (Reddit/GDELT hype, JP forum) are
    left alone: they aren't price data, and re-hitting those courtesy-throttled
    endpoints early would be both slow and impolite."""
    st.session_state.pop("_hist_cache", None)          # shared bulk-history frames
    for fn in (fetch_history, fetch_fundamentals, fetch_analyst_signals,
               fetch_insider_activity, _bulk_latest_close):
        try:
            fn.clear()                                 # per-function st.cache_data wipe
        except Exception:
            pass   # decorator absent (e.g. hot-reload edge) -> nothing cached anyway

@st.cache_data(ttl=300, show_spinner=False)
def _bulk_latest_close(tickers: tuple) -> dict:
    """Most-recent close for many tickers in ONE bulk yf.download request.

    Cached 5 min per ticker tuple: on the audit tab any widget interaction reruns
    the whole script, and while "update prices" is ticked that used to refire the
    full bulk download every rerun. Callers pass a TUPLE (hashable cache key).

    Fetching prices one ticker at a time hammered Yahoo and tripped its IP rate
    limiter (crashing the app on Streamlit Cloud). A single `yf.download(...)` for
    the whole list fixes that. Returns {ticker: close} with NaN for anything that
    didn't come back, and never raises.

    Uses a 5-day window and takes each ticker's LAST NON-NaN close (not the single
    most-recent row) — a mixed US/European batch otherwise leaves the off-calendar
    market (e.g. MC.PA, ASML) NaN on the last row, which showed up as a blank
    "Current Price". Any symbol the bulk request drops entirely gets a bounded
    single-ticker fallback.
    """
    out = {t: float("nan") for t in tickers}
    uniq = list(dict.fromkeys(tickers))
    if yf is None or not uniq:
        return out
    kwargs = dict(period="5d", progress=False, threads=False, auto_adjust=True)
    data = None
    for attempt in range(2):                  # one retry: Yahoo throttles request bursts
        try:
            try:
                data = yf.download(uniq, session=_YF_SESSION, **kwargs)   # SSL-aware session
            except TypeError:
                data = yf.download(uniq, **kwargs)                        # older yfinance: no session kwarg
        except Exception:
            data = None
        if data is not None and len(data):
            break
        if attempt == 0:
            time.sleep(1.0)                   # brief pause, then one more attempt
    if data is not None and len(data):
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]                 # columns = tickers
                for t in uniq:
                    if t in close.columns:
                        col = close[t].dropna()       # last VALID close for THIS ticker,
                        if not col.empty:             # not the global last row
                            out[t] = safe_float(col.iloc[-1])
            else:
                series = data["Close"].dropna()       # single-ticker frame: flat columns
                if not series.empty and len(uniq) == 1:
                    out[uniq[0]] = safe_float(series.iloc[-1])
        except Exception:
            pass
    # Per-ticker fallback for anything still unresolved (yfinance can silently omit
    # symbols from a mixed-exchange batch). Bounded to the misses, so it can't
    # reintroduce a full per-ticker fetch storm.
    for t in [t for t in uniq if math.isnan(out[t])]:
        try:
            s = fetch_history(t, period="5d")["Close"].dropna()   # has retry/backoff
            if not s.empty:
                out[t] = safe_float(s.iloc[-1])
        except Exception:
            pass
    return out

def walk_forward_update() -> int:
    """Walk-forward optimiser driven by the mock_portfolio KPI snapshots.

    For each logged pick that has matured (>= EVAL_HORIZON_DAYS old) and hasn't yet
    been scored, compute its benchmark-relative forward return, then attribute that
    outcome to the KPI profile it was picked on: factors that scored highly on
    winners get nudged UP, factors that scored highly on losers get nudged DOWN
    (so e.g. Value rises if value-heavy picks are beating their market). The new
    weights are persisted via save_weights — meaning they drive scoring through
    score_with_weights AND land in the system_weights ledger. Each pick is flagged
    evaluated so it contributes exactly once. Returns the count newly evaluated.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, ticker, recommendation_price, kpi_snapshot FROM mock_portfolio WHERE evaluated=0"
        ).fetchall()
    if not rows:
        return 0

    cutoff = date.today() - timedelta(days=EVAL_HORIZON_DAYS)

    # Batch the price fetches: ONE bulk download covering every matured ticker plus
    # its benchmark, instead of a per-ticker fetch (with 3-retry backoff) inside the
    # loop. Bulk misses fall back to the per-ticker path below.
    matured_syms: list[str] = []
    for row in rows:
        _dt = pd.to_datetime(row["timestamp"], errors="coerce")
        if not pd.isna(_dt) and _dt.date() <= cutoff:
            matured_syms.append(row["ticker"])
    bench_syms = sorted({benchmark_for(t) for t in matured_syms})
    bulk = get_histories(list(dict.fromkeys(matured_syms)) + bench_syms, period="1y")

    bench_cache: dict[str, pd.DataFrame | None] = {s: bulk[s] for s in bench_syms if s in bulk}
    def bench_hist(sym: str) -> pd.DataFrame | None:
        if sym not in bench_cache:
            try:
                bench_cache[sym] = fetch_history(sym, period="1y")
            except Exception:
                bench_cache[sym] = None   # missing benchmark -> absolute-return fallback
        return bench_cache[sym]

    grad = {f: 0.0 for f in FACTORS}
    evaluated_ids, n = [], 0
    for row in rows:
        rec_dt = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(rec_dt) or rec_dt.date() > cutoff:
            continue   # not matured yet — leave unflagged so it's re-checked later
        try:
            snap = json.loads(row["kpi_snapshot"] or "{}")
        except Exception:
            snap = {}
        hist = bulk.get(row["ticker"])
        if hist is None:
            try:
                hist = fetch_history(row["ticker"], period="1y")
            except Exception:
                continue   # transient fetch failure -> skip this name, keep the batch alive
        target = rec_dt + timedelta(days=EVAL_HORIZON_DAYS)
        # Fix 5 — derive BOTH endpoints from the SAME freshly-fetched, dividend-
        # adjusted series (fetch_history uses auto_adjust=True, which folds dividends
        # into the close). That makes stock_ret a proper TOTAL return, so Value /
        # Dividend picks get credit for payouts. Re-deriving the entry price here
        # (rather than using the stored recommendation_price) also keeps both ends on
        # one adjustment baseline — adding a separate dividend-yield term on top would
        # double-count, since the series is already adjusted.
        price_then = _close_on_or_after(hist, rec_dt)
        price_after = _close_on_or_after(hist, target)
        if math.isnan(price_after) or math.isnan(price_then) or price_then <= 0:
            continue
        stock_ret = price_after / price_then - 1.0

        bh = bench_hist(benchmark_for(row["ticker"]))
        b_then = _close_on_or_after(bh, rec_dt)
        b_after = _close_on_or_after(bh, target)
        if not math.isnan(b_then) and not math.isnan(b_after) and b_then > 0:
            win = stock_ret > (b_after / b_then - 1.0)   # beat its home market
        else:
            win = stock_ret > 0                          # fallback: absolute return
        direction = 1.0 if win else -1.0
        for f in FACTORS:
            grad[f] += direction * (safe_float(snap.get(f), 50.0) / 100.0 - 0.5)
        evaluated_ids.append(int(row["id"]))
        n += 1

    if n == 0:
        return 0

    weights = get_latest_weights()
    new = {f: max(MIN_WEIGHT, weights[f] + LEARNING_RATE * grad[f] / n) for f in FACTORS}
    total = sum(new.values())
    new = {f: new[f] / total for f in FACTORS}
    save_weights(new, note=f"Walk-forward update on {n} matured mock-portfolio pick(s)")

    with get_conn() as conn:
        conn.executemany("UPDATE mock_portfolio SET evaluated=1 WHERE id=?", [(i,) for i in evaluated_ids])
        conn.commit()
    return n

def seed_demo_history() -> None:
    rng = np.random.default_rng(42)
    with get_conn() as conn:
        for i in range(40):
            d = (date.today() - timedelta(days=80 - i * 2)).isoformat()
            t = ALL_TICKERS[i % len(ALL_TICKERS)]
            scores = {f: float(rng.uniform(40, 85)) for f in FACTORS}
            comp = float(np.mean(list(scores.values())))
            win = int(rng.random() < 0.62)
            price = float(rng.uniform(100, 300))
            conn.execute(
                "INSERT INTO recommendations (ticker, rec_date, recommendation, composite, price_at_rec, momentum, value, technical, hype, quality, theme, price_after, outcome, eval_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t, d, "BUY" if comp > 60 else "HOLD", comp, price, scores["momentum"], scores["value"], scores["technical"], scores["hype"], scores["quality"], scores["theme"], price * (1.06 if win else 0.94), win, d)
            )
        conn.commit()
    w = dict(DEFAULT_WEIGHTS)
    for k in range(5):
        w["momentum"] = min(0.45, w["momentum"] + 0.02)
        w["value"] = max(MIN_WEIGHT, w["value"] - 0.01)
        tot = sum(w.values())
        save_weights({f: w[f] / tot for f in FACTORS}, note="Demo backfill loop simulation")

def run_engine(limit_per_region: int | None = None,
               regions: list | None = None,
               sources: list | None = None,
               use_jp_forum: bool = True) -> tuple[list[dict], list[str]]:
    weights = get_latest_weights()
    results, failed = [], []

    # Only scan the regions the user selected (defaults to all). On a quick scan,
    # also cap to the first N tickers per region — fewer requests means a faster
    # run and a much lower chance of Yahoo rate-limiting the server's IP.
    # Promoted deep-scan names (US 5 / JP 3 / CN 2 max) ride along AFTER the cap,
    # so even a quick scan always covers the current promotion layer.
    selected = regions if regions else list(TICKER_UNIVERSE.keys())
    promoted_map = get_promoted_universe()
    promoted_set = {t for ticks in promoted_map.values() for t in ticks}
    universe = {}
    for region, ticks in TICKER_UNIVERSE.items():
        if region not in selected:
            continue
        base = ticks[:limit_per_region] if limit_per_region else list(ticks)
        extra = [t for t in promoted_map.get(region, []) if t not in base]
        universe[region] = base + extra

    # Pull sentiment buzz ONCE for the whole scan, routing each market to the
    # enabled sources (Reddit for English names, GDELT news for JP/CN, etc.), then
    # feed each ticker its own combined count. Fully fail-safe: a blocked source
    # contributes zeros, so the scan proceeds with hype unaffected.
    hype_counts = fetch_hype_signals(universe, sources)

    # ONE bulk price download for the whole scan (the deep scan already worked this
    # way; the main scan previously paid a per-ticker fetch with retry/backoff plus
    # a fixed 0.4s sleep per name — most of a quick scan's wall clock). Anything the
    # bulk request misses (delisted, exchange quirk) transparently falls back to the
    # per-ticker path inside analyze_ticker. Fundamentals (.info) remain per-ticker —
    # Yahoo has no bulk endpoint for those — pacing lives inside the cached fetchers.
    # period="1y" (was 8mo) deliberately matches the deep scan / sell scanner /
    # walk-forward period, so get_histories serves ALL of them from one download;
    # analyze_ticker only reads tail slices, so the longer frame changes nothing.
    all_symbols = [t for ticks in universe.values() for t in ticks]
    bulk = get_histories(all_symbols, period="1y")

    progress = st.progress(0.0, text=tr("scanning"))

    total_symbols = sum(len(ticks) for ticks in universe.values())
    current_index = 0

    analyses: list[dict] = []
    for region, tickers in universe.items():
        for t in tickers:
            try:
                jp_rating = (fetch_jp_forum_rating(t)
                             if (use_jp_forum and region == "Japan") else None)
                analysis = analyze_ticker(t, region, hype_counts.get(t, 0),
                                          jp_forum=jp_rating, hist=bulk.get(t))
                if analysis is None:
                    failed.append(f"{t} (no data)")
                else:
                    analysis["promoted"] = t in promoted_set
                    if analysis["promoted"]:
                        # Star the display name so promoted ride-alongs are
                        # recognisable in every tab without new columns.
                        analysis["name"] = f"⭐ {analysis.get('name') or t}"
                    analyses.append(analysis)
            except Exception as e:
                # Surface the real reason instead of silently swallowing it.
                failed.append(f"{t} ({type(e).__name__})")
            current_index += 1
            # Pacing now lives INSIDE the cached fetchers (fetch_fundamentals /
            # fetch_history), so it only fires on actual network calls — a repeat
            # scan within the TTL runs at full speed instead of sleeping per name.
            progress.progress(current_index / total_symbols, text=tr("processing", ticker=t, region=region_name(region)))
    progress.empty()

    # ---- Theme (industry-rotation) KPI: cross-sectional second pass ----
    # A ticker's theme score is the damped average MOMENTUM of the *other* scanned
    # members of its basket (theme_strength_score in indicators.py), so it must be
    # computed after every name has been analysed. Uses the sector/industry strings
    # stage-2 fundamentals fetched anyway — zero extra API calls. Themeless names
    # keep the neutral 50 set by analyze_ticker.
    theme_members: dict[str, list[dict]] = {}
    for a in analyses:
        a["theme_match"] = match_theme(a["region"], a.get("sector"), a.get("industry"))
        if a["theme_match"]:
            theme_members.setdefault(a["theme_match"], []).append(a)
    for a in analyses:
        peers = [p["momentum"] for p in theme_members.get(a.get("theme_match"), []) if p is not a]
        a["theme"] = theme_strength_score(peers)
        a.update(score_with_weights(a, weights))
        save_recommendation(a)
        results.append(a)

    results.sort(key=lambda r: r["composite"], reverse=True)
    # Log today's curated Top-3 Growth / Top-3 Dividend picks for the walk-forward
    # loop, persist the scan snapshot (survives refresh/new session), and stamp the
    # scan time for the freshness chip. All wrapped/fail-safe so bookkeeping can
    # never sink an otherwise-successful scan.
    try:
        save_mock_portfolio(results)
    except Exception:
        pass
    save_scan_snapshot(results, is_quick=limit_per_region is not None)
    st.session_state["scan_ts"] = time.time()
    st.session_state["restored_scan"] = False
    return results, failed

# ----------------------------------------------------------------------------
# Layout Tab Renderers
# ----------------------------------------------------------------------------
def render_daily_top_3(results: list[dict]) -> None:
    st.subheader(tr("top3_header"))
    if not results:
        st.info(tr("need_scan_sidebar"))
        return
    top_3 = results[:3]
    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        if i < len(top_3):
            r = top_3[i]
            col.markdown(metric_card(f"{r['ticker']} · {region_name(r['region'])}", f"{r['composite']:.1f}/100 {tr('score_suffix')}", f"{tr('price_label')}: {fmt_money(r['price'])}", positive=r['composite'] >= BUY_THRESHOLD), unsafe_allow_html=True)
            div_txt = f"{r['div_yield']:.2f}%" if not math.isnan(r['div_yield']) else "—"
            col.markdown(
                f"**{tr('company_profile')}:** {r['name']} <br>"
                f"**{tr('trend_return_1m')}:** {fmt_pct(r['ret_1m'])} <br>"
                f"**{tr('pe_ratio')}:** {fmt_num(r['pe'], 1)} &nbsp;·&nbsp; **{tr('dividend_yield')}:** {div_txt}",
                unsafe_allow_html=True,
            )

# --- US Conviction tab (US-only; SEC EDGAR fundamentals) -------------------------
# Deliberately isolated: it blends each US name's existing scan composite with SEC
# growth/quality OUTSIDE the global factor model, so this US-only data never enters
# FACTORS, DEFAULT_WEIGHTS, or the walk-forward gradient.
US_CONVICTION_WEIGHTS = {"composite": 0.5, "growth": 0.3, "quality": 0.2}

@st.cache_data(ttl=86400, show_spinner=False)
def _sec_fundamentals_cached(ticker: str) -> dict:
    """SEC fundamentals for one US ticker, cached a day (filings are quarterly). Raises
    on a transient fetch failure so the broken state isn't frozen for the whole TTL."""
    ua = _get_secret("SEC_USER_AGENT", "stockrec-research/1.0 (set SEC_USER_AGENT to your email)")
    return sec_research.sec_fundamentals(ticker, ua=ua)

# 13F superinvestor conviction layer (Phase 2): each manager that NEWLY BOUGHT the name
# this quarter (added it quarter-over-quarter) adds a few
# points, capped — so this quarterly, ~45-day-lagged signal tilts the ranking rather than
# dominating the SEC-fundamentals core.
SUPERINVESTOR_PT = 4.0
SUPERINVESTOR_CAP = 3

@st.cache_data(ttl=86400, show_spinner=False)
def _superinvestor_counts_cached(tickers: tuple) -> dict:
    """How many tracked superinvestors hold each US ticker (latest 13F-HR), cached a day.
    Raises on a total outage so an all-zero result isn't frozen for the whole TTL."""
    ua = _get_secret("SEC_USER_AGENT", "stockrec-research/1.0 (set SEC_USER_AGENT to your email)")
    return sec_research.superinvestor_counts(list(tickers), ua=ua)

def _blend_conviction(comp: float, growth: float, quality: float) -> float:
    """Weighted, NaN-aware blend. If SEC scores are missing, it renormalises onto the
    scan composite alone, so a name still ranks (just without the SEC tilt)."""
    num = den = 0.0
    for key, val in (("composite", comp), ("growth", growth), ("quality", quality)):
        if val is not None and not math.isnan(val):
            num += US_CONVICTION_WEIGHTS[key] * val
            den += US_CONVICTION_WEIGHTS[key]
    return num / den if den > 0 else float("nan")

def render_us_conviction(results: list[dict]) -> None:
    st.subheader(tr("us_header"))
    if sec_research is None:
        st.warning(tr("us_module_missing"))
        return
    us = [r for r in results if r.get("region") == "USA"]
    if not us:
        st.info(tr("need_scan_sidebar"))
        return

    us_tickers = sorted({r["ticker"] for r in us})
    rows = []
    with st.spinner(tr("us_spinner")):
        try:
            sci = _superinvestor_counts_cached(tuple(us_tickers))
        except Exception:
            sci = {}   # 13F unavailable this run -> no conviction bonus, no crash
        for r in us:
            comp = safe_float(r.get("composite"), float("nan"))
            g = q = rev_cagr = rev_yoy = eps_yoy = float("nan")
            try:
                f = _sec_fundamentals_cached(r["ticker"])
                g, q = f["growth_score"], f["quality_score"]
                rev_cagr, rev_yoy, eps_yoy = f["rev_cagr_3y"], f["rev_yoy"], f["earnings_yoy"]
            except Exception:
                pass   # SEC unavailable for this name -> rank on the composite alone
            base = _blend_conviction(comp, g, q)
            n_si = int(sci.get(r["ticker"], 0) or 0)
            conv = base if math.isnan(base) else clamp(base + SUPERINVESTOR_PT * min(n_si, SUPERINVESTOR_CAP))
            rows.append({
                "ticker": r["ticker"], "name": r.get("name", r["ticker"]),
                "conviction": conv, "composite": comp, "n_si": n_si,
                "growth": g, "quality": q, "rev_cagr": rev_cagr, "eps_yoy": eps_yoy,
                "price": safe_float(r.get("price"), float("nan")),
            })
    rows.sort(key=lambda x: x["conviction"] if not math.isnan(x["conviction"]) else float("-inf"),
              reverse=True)

    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        if i < len(rows):
            x = rows[i]
            conv = "—" if math.isnan(x["conviction"]) else f"{x['conviction']:.1f}/100"
            cagr = "—" if math.isnan(x["rev_cagr"]) else fmt_pct(x["rev_cagr"] * 100.0)
            si = f"<br>⭐ {tr('us_held_by', n=x['n_si'])}" if x["n_si"] > 0 else ""
            col.markdown(metric_card(x["ticker"], f"{conv} {tr('us_conviction_suffix')}",
                                     f"{tr('price_label')}: {fmt_money(x['price'])}",
                                     positive=(not math.isnan(x["conviction"]) and x["conviction"] >= BUY_THRESHOLD)),
                         unsafe_allow_html=True)
            col.markdown(
                f"**{tr('company_profile')}:** {x['name']} <br>"
                f"**{tr('col_sec_growth')}:** {fmt_num(x['growth'], 0)} &nbsp;·&nbsp; "
                f"**{tr('col_sec_quality')}:** {fmt_num(x['quality'], 0)} <br>"
                f"**{tr('col_rev_cagr')}:** {cagr}{si}",
                unsafe_allow_html=True,
            )

    def _r(v, nd=1):
        return float("nan") if math.isnan(v) else round(v, nd)
    table = pd.DataFrame([{
        tr("col_ticker"): x["ticker"], tr("col_company"): x["name"],
        tr("col_us_conviction"): _r(x["conviction"]), tr("col_overall_score"): _r(x["composite"]),
        tr("col_sec_growth"): _r(x["growth"], 0), tr("col_sec_quality"): _r(x["quality"], 0),
        tr("col_rev_cagr"): _r(x["rev_cagr"] * 100.0), tr("col_eps_yoy"): _r(x["eps_yoy"] * 100.0),
        tr("col_superinvestors"): x["n_si"],
    } for x in rows])
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(tr("us_caption"))

def render_category_views(results: list[dict]) -> None:
    if not results:
        st.info(tr("need_run_engine"))
        return

    v1, v2 = st.tabs([tr("subtab_growth"), tr("subtab_dividend")])
    df = pd.DataFrame(results)

    with v1:
        st.markdown(f"### {tr('growth_header_text')}")
        growth_df = df.sort_values(by="ret_1m", ascending=False).head(5).copy()
        growth_df["region"] = growth_df["region"].map(region_name)
        st.dataframe(growth_df[["ticker", "name", "region", "price", "ret_1m", "composite"]].rename(columns={"ticker": tr("col_ticker"), "name": tr("col_company"), "region": tr("col_region"), "price": tr("col_price"), "ret_1m": tr("col_momentum_1m"), "composite": tr("col_overall_score")}), width="stretch", hide_index=True)

    with v2:
        st.markdown(f"### {tr('dividend_header_text')}")
        div_df = df[df["div_yield"] >= 1.5].sort_values(by="composite", ascending=False).head(5).copy()
        if not div_df.empty:
            div_df["region"] = div_df["region"].map(region_name)
            st.dataframe(div_df[["ticker", "name", "region", "price", "div_yield", "composite"]].rename(columns={"ticker": tr("col_ticker"), "name": tr("col_company"), "region": tr("col_region"), "price": tr("col_price"), "div_yield": tr("col_dividend_return"), "composite": tr("col_overall_score")}), width="stretch", hide_index=True)
        else:
            st.info(tr("no_dividend_match"))

def render_global_sectors(results: list[dict]) -> None:
    if not results:
        st.info(tr("need_activate"))
        return
    region_keys = list(TICKER_UNIVERSE.keys())
    target_region = st.selectbox(tr("select_region"), region_keys, format_func=region_name)
    regional_filtered = [r for r in results if r["region"] == target_region]

    # One sortable, selectable table instead of a card row per ticker (38 US names
    # made a very long scroll with no sorting). ProgressColumn gives an instant
    # visual ranking; clicking a row opens a detail card below AND preselects the
    # name in the Deep-Dive tab (row order == regional_filtered order, so the
    # selection index maps straight back to the analysis dict).
    df = pd.DataFrame([{
        "ticker": r["ticker"],
        "name": str(r.get("name", r["ticker"])),
        "call": tr(f"rec_{r['recommendation']}") if r.get("recommendation") in ("BUY", "HOLD", "SELL")
                else str(r.get("recommendation", "—")),
        "composite": safe_float(r.get("composite")),
        "price": safe_float(r.get("price")),
        "hype": safe_float(r.get("hype_score")),
        "ret1m": safe_float(r.get("ret_1m")),
    } for r in regional_filtered])
    event = st.dataframe(
        df,
        column_config={
            "ticker": st.column_config.TextColumn(tr("col_ticker"), width="small"),
            "name": st.column_config.TextColumn(tr("col_company")),
            "call": st.column_config.TextColumn(tr("why_call"), width="small"),
            "composite": st.column_config.ProgressColumn(
                tr("card_system_rating"), min_value=0, max_value=100, format="%.1f"),
            "price": st.column_config.NumberColumn(tr("card_trading_close"), format="%.2f"),
            "hype": st.column_config.ProgressColumn(
                tr("card_sustained_hype"), min_value=0, max_value=100, format="%.0f"),
            "ret1m": st.column_config.NumberColumn(tr("evi_ret_1m"), format="%+.1f%%"),
        },
        hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row",
        key=f"regional_table_{target_region}",
    )
    sel_rows = []
    try:
        sel_rows = list(event.selection.rows)   # DataframeState from on_select
    except Exception:
        sel_rows = []
    if not sel_rows:
        st.caption(tr("row_select_prompt"))
        return

    r = regional_filtered[sel_rows[0]]
    # Hand the pick to the Deep-Dive tab: its selectbox is keyed "deep_dive_pick",
    # and this tab renders BEFORE it in the same rerun, so the state lands in time.
    st.session_state["deep_dive_pick"] = r["ticker"]

    cols = st.columns([2, 1, 1, 1])
    cols[0].markdown(
        f'<div class="qc-card"><span class="qc-ticker">{r["ticker"]}</span> '
        f'{rec_pill(r["recommendation"])}<br><span class="qc-sub">{r["name"]}</span></div>',
        unsafe_allow_html=True)
    cols[1].markdown(metric_card(tr("card_system_rating"), f"{r['composite']:.1f}"), unsafe_allow_html=True)
    cols[2].markdown(metric_card(tr("card_trading_close"), fmt_money(r["price"])), unsafe_allow_html=True)
    cols[3].markdown(metric_card(tr("card_sustained_hype"), f"{r['hype_score']:.0f}",
                     tr("breakout") if r["hype"]["sustained"] else tr("flat"),
                     positive=r["hype"]["sustained"]), unsafe_allow_html=True)
    # Real navigation, not just preselection: the Deep-Dive page is registered in
    # _PAGES by _build_navigation (runs before any page renders). The pick was
    # already stored under the Deep-Dive selectbox's key above, so the landing
    # page opens straight on this name. Fallback caption if navigation isn't up
    # (e.g. bare-mode tests).
    if st.button(tr("open_deep_dive_btn", ticker=r["ticker"]), type="primary",
                 key=f"open_dd_{target_region}"):
        page = _PAGES.get("deep_dive")
        if page is not None:
            st.switch_page(page)
    st.caption(tr("row_selected_hint", ticker=r["ticker"]))

def _ago(epoch: float) -> str:
    """Compact 'time since' label, e.g. '<1m', '12m', '2h 5m', '3d 4h'. Unit suffixes
    only, so the surrounding 'updated … ago' phrasing handles localization."""
    mins = max(0, int(time.time() - epoch)) // 60
    if mins < 1:
        return "<1m"
    if mins < 60:
        return f"{mins}m"
    if mins < 60 * 48:
        return f"{mins // 60}h {mins % 60}m"
    return f"{mins // 1440}d {(mins % 1440) // 60}h"   # restored snapshots can be days old


def _pick_breakdown(r: dict, weights: dict) -> pd.DataFrame:
    """Per-factor score, weight and weighted contribution to the composite.
    contribution = weight * factor_score, and the contributions sum to the composite —
    so this is an exact decomposition of *why* the name scored what it did."""
    rows = []
    for f in FACTORS:
        score = safe_float(r.get("hype_score" if f == "hype" else f), float("nan"))
        w = float(weights.get(f, 0.0))
        rows.append({"factor": f, "score": score, "weight": w, "contribution": w * score})
    return pd.DataFrame(rows)


def _evidence_rows(r: dict) -> list:
    """The actual KPI numbers behind each factor, as (factor, metric, value) rows."""
    fm, fv, ft, fh, fq = (tr("factor_momentum"), tr("factor_value"), tr("factor_technical"),
                          tr("factor_hype"), tr("factor_quality"))
    fth = tr("factor_theme")
    def _f(x): return safe_float(x, float("nan"))
    price, sma50 = _f(r.get("price")), _f(r.get("sma50"))
    vs50 = "—" if (math.isnan(price) or math.isnan(sma50) or sma50 == 0) else f"{(price / sma50 - 1) * 100:+.1f}%"
    macd = _f(r.get("macd_hist"))
    macd_s = "—" if math.isnan(macd) else (tr("evi_bullish") if macd > 0 else tr("evi_bearish"))
    pe, div, rsi, bbp = _f(r.get("pe")), _f(r.get("div_yield")), _f(r.get("rsi")), _f(r.get("bb_pct"))
    roe, shortp = _f(r.get("roe")), _f(r.get("short_pct"))
    ret1m = _f(r.get("ret_1m"))
    jp_bull, jp_bear = _f(r.get("jp_bull")), _f(r.get("jp_bear"))
    rows = [
        (fm, tr("evi_ret_1m"),  "—" if math.isnan(ret1m) else f"{ret1m:+.1f}%"),
        (fm, tr("evi_vs_sma50"), vs50),
        (fm, tr("evi_macd"),     macd_s),
        (fv, tr("evi_pe"),       "—" if (math.isnan(pe) or pe <= 0) else f"{pe:,.1f}"),
        (fv, tr("evi_div"),      "—" if math.isnan(div) else f"{div:.2f}%"),       # div_yield already in %
        (ft, tr("evi_rsi"),      "—" if math.isnan(rsi) else f"{rsi:.0f}"),
        (ft, tr("evi_bbpct"),    "—" if math.isnan(bbp) else f"{bbp:.2f}"),
        (fq, tr("evi_roe"),      "—" if math.isnan(roe) else f"{roe * 100:.1f}%"),  # roe is a fraction
        (fh, tr("evi_mentions"), f"{safe_float(r.get('hype_mentions'), 0):g}"),
        (fh, tr("evi_short"),    "—" if math.isnan(shortp) else f"{shortp * 100:.1f}%"),  # fraction of float
        (fth, tr("evi_theme_basket"), str(r.get("theme_match") or "—")),
        (fth, tr("evi_theme_peers"),  fmt_num(safe_float(r.get("theme"), float("nan")), 0)),
    ]
    # Yahoo!掲示板 みんなの評価 — shown only for names where a poll was actually
    # fetched (Japanese tickers with the toggle on), so the row's presence is itself
    # the "is the forum signal live?" confirmation Gemini wanted for the live test.
    if not (math.isnan(jp_bull) or math.isnan(jp_bear)):
        rows.append((fh, tr("evi_jp_forum"), tr("evi_jp_forum_val").format(
            bull=f"{jp_bull:.0f}", bear=f"{jp_bear:.0f}")))
    return rows


def _screen_rank(histories: dict, n: int) -> list:
    """Stage-1 price screen: rank tickers by the multi-factor screen score from
    indicators.screen_metrics (momentum blend + volume surge + 52w-high proximity,
    with a falling-knife cap). Returns the top n as [(ticker, metrics_dict), ...].
    Pure function over the bulk histories — still zero extra API calls."""
    scored = []
    for t, h in histories.items():
        vol = h["Volume"] if "Volume" in h.columns else None
        m = screen_metrics(h["Close"], vol)
        if m is not None:
            scored.append((t, m))
    scored.sort(key=lambda x: x[1]["score"], reverse=True)
    return scored[:n]


def run_deep_scan(region: str) -> tuple[list, int]:
    """Deep scan funnel for one market. Stage 1: ONE bulk price download for the
    market's curated deep list MERGED with any live screener names (see
    deep_universe_for), ranked on the multi-factor screen score. Stage 2: full
    fundamental scoring on the top finalists only — Yahoo sees ~1 bulk request +
    DEEP_FINALISTS per-ticker calls. Finalists also get a theme classification
    from the sector/industry that stage 2 fetches anyway."""
    tickers, _n_live = deep_universe_for(region)
    # 1y (~252 trading days), not 8mo: screen_metrics' 52-week-high proximity reads
    # closes.iloc[-252:], so an 8-month frame silently turned it into an 8-month
    # high — flattering names that peaked 9-12 months ago and bled since, the exact
    # profile the falling-knife cap exists to exclude. One bulk download either way.
    histories = get_histories(tickers, period="1y")
    n_screened = len(histories)
    finalists = _screen_rank(histories, DEEP_FINALISTS)
    weights = get_latest_weights()
    scored: list[tuple[dict, dict]] = []
    prog = st.progress(0.0, text=tr("deep_scan_scoring"))
    for i, (t, m) in enumerate(finalists, start=1):
        try:
            # Reuse the stage-1 bulk frame — stage 2 previously re-fetched each
            # finalist's history per ticker even though it was already in memory.
            a = analyze_ticker(t, region, 0.0, hist=histories.get(t))
            if a:
                a["screen_score"] = m["score"]
                a["screen_blend"] = m["blend_pct"]
                a["screen_dhigh"] = m["dist_high_pct"]
                a["screen_vsurge"] = m["vol_surge"]
                a["theme_match"] = match_theme(region, a.get("sector"), a.get("industry"))
                scored.append((a, m))
        except Exception:
            pass   # a throttled finalist is dropped, not fatal
        prog.progress(i / max(1, len(finalists)), text=tr("deep_scan_scoring"))
    prog.empty()
    # Theme KPI across the finalist cohort (same cross-sectional logic as the main
    # scan): a finalist in a basket whose scanned peers are also strong gets a tilt.
    theme_members: dict[str, list[dict]] = {}
    for a, _m in scored:
        if a.get("theme_match"):
            theme_members.setdefault(a["theme_match"], []).append(a)
    results = []
    for a, _m in scored:
        peers = [p["momentum"] for p in theme_members.get(a.get("theme_match"), []) if p is not a]
        a["theme"] = theme_strength_score(peers)
        a.update(score_with_weights(a, weights))
        results.append(a)
    results.sort(key=lambda x: safe_float(x.get("composite"), float("-inf")), reverse=True)
    return results, n_screened


def render_deep_scan() -> None:
    st.caption(tr("deep_scan_intro"))
    # Current promotion layer — visible up front, with the persistence snippet and a
    # clear button, so the state of "what rides along with the main scan" is obvious.
    promoted_now = get_promoted_universe()
    if any(promoted_now.values()):
        items = " · ".join(
            f"{region_name(r)}: {', '.join(ts)}" for r, ts in promoted_now.items() if ts)
        st.caption(tr("promoted_current", items=items))
        with st.expander(tr("promote_snippet_hint")):
            st.code(promotions_config_snippet(), language="python")
        if st.button(tr("promote_clear"), key="promote_clear_btn"):
            clear_promotions()
            st.info(tr("promote_clear_note"))
            st.rerun()
    region_labels = {tr("deep_region_us"): "USA", tr("deep_region_jp"): "Japan",
                     tr("deep_region_cn"): "China"}
    choice = st.radio(tr("deep_region_label"), list(region_labels.keys()),
                      horizontal=True, key="deep_region")
    region = region_labels[choice]
    if region != "USA":
        st.caption(tr("deep_scan_jpcn_note"))
    # Universe status: curated floor + live screener overlay, with freshness.
    _dyn, _dyn_ts = get_dynamic_universe(region)
    _curated_n = len(DEEP_UNIVERSES.get(region, []))
    _live_n = deep_universe_for(region)[1]
    _ago_part = ""
    if _dyn_ts:
        try:
            _ago_part = f" · {_ago(pd.Timestamp(_dyn_ts).timestamp())}"
        except Exception:
            pass
    st.caption(tr("dyn_universe_status", curated=_curated_n, dynamic=_live_n, ago=_ago_part))
    col_run, col_refresh = st.columns(2)
    if col_refresh.button(tr("dyn_refresh_btn"), key=f"dyn_refresh_{region}"):
        with st.spinner(tr("deep_scan_running")):
            syms = refresh_dynamic_universe(region)
        if syms:
            save_dynamic_universe(region, syms)
            st.success(tr("dyn_refresh_done", n=len(syms)))
            st.rerun()
        else:
            st.warning(tr("dyn_refresh_fail"))
    if col_run.button(tr("deep_scan_btn"), key="deep_scan_run"):
        with st.spinner(tr("deep_scan_running")):
            res, n = run_deep_scan(region)
        st.session_state[f"deep_results_{region}"] = res
        st.session_state[f"deep_screened_{region}"] = n
    res = st.session_state.get(f"deep_results_{region}")
    if not res:
        st.info(tr("deep_scan_hint"))
        return
    st.caption(tr("deep_scan_done", screened=st.session_state.get(f"deep_screened_{region}", 0),
               total=len(deep_universe_for(region)[0]), finalists=len(res)))

    cols = st.columns(3)
    for i, item in enumerate(res[:3]):
        comp = safe_float(item.get("composite"), float("nan"))
        with cols[i]:
            st.markdown(metric_card(f"{item['ticker']} · {item.get('name', item['ticker'])}",
                        fmt_num(comp, 1), positive=(not math.isnan(comp) and comp >= BUY_THRESHOLD)),
                        unsafe_allow_html=True)
            st.markdown(rec_pill(str(item.get("recommendation", ""))), unsafe_allow_html=True)

    df = pd.DataFrame([{
        tr("col_ticker"): x["ticker"], tr("col_company"): x.get("name", x["ticker"]),
        tr("col_theme_strength"): safe_float(x.get("composite"), float("nan")),
        tr("why_call"): str(x.get("recommendation", "—")),
        tr("deep_col_score"): safe_float(x.get("screen_score"), float("nan")),
        tr("deep_col_dhigh"): safe_float(x.get("screen_dhigh"), float("nan")),
        tr("deep_col_vsurge"): safe_float(x.get("screen_vsurge"), float("nan")),
        tr("deep_col_theme"): x.get("theme_match") or "—",
        tr("pe_ratio"): safe_float(x.get("pe"), float("nan")),
    } for x in res])
    st.dataframe(df.style.format({tr("col_theme_strength"): "{:.0f}", tr("deep_col_score"): "{:.0f}",
                 tr("deep_col_dhigh"): "{:+.1f}%", tr("deep_col_vsurge"): "{:.1f}×",
                 tr("pe_ratio"): "{:,.1f}"}, na_rep="—"),
                 width="stretch", hide_index=True)
    st.caption(tr("deep_scan_note"))

    # Promote this market's best finalists (US 5 / JP 3 / CN 2) into the main scan.
    # Ranked by composite; names already in the base TICKER_UNIVERSE are skipped so
    # promotion never double-scans a name. Replaces this region's previous
    # promotions — re-running deep scans naturally rotates the layer.
    quota = PROMOTION_QUOTA.get(region, 0)
    if quota and st.button(tr("promote_btn", n=quota), key=f"promote_{region}"):
        picks, skipped = [], 0
        for x in res:
            if len(picks) >= quota:
                break
            t = x["ticker"]
            if t in _BASE_UNIVERSE_SET:
                skipped += 1
                continue
            picks.append((t, str(x.get("name") or t), safe_float(x.get("composite"), 0.0)))
        if picks:
            save_promotions(region, picks)
            st.success(tr("promote_done", added=len(picks), skipped=skipped))
            st.rerun()
        else:
            st.info(tr("promote_none"))


def record_theme_snapshot(rows: list[dict]) -> None:
    """Persist per-theme aggregates for one FULL scan. Quick scans are skipped by the
    caller (comparing quick vs full would be noise, not signal). A content signature
    dedupes Streamlit's reruns: the same scan rendering ten times inserts once."""
    sig = hashlib.md5(json.dumps(
        [(r["theme"], round(safe_float(r.get("momentum"), 0.0), 2), r["n"]) for r in rows],
        sort_keys=True).encode()).hexdigest()
    try:
        with get_conn() as conn:
            last = conn.execute(
                "SELECT snapshot FROM theme_history ORDER BY id DESC LIMIT 1").fetchone()
            if last and last[0] == sig:
                return
            now = datetime.now().isoformat(timespec="seconds")
            conn.executemany(
                "INSERT INTO theme_history (snapshot, theme, momentum, hype, composite, n, cover, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(sig, r["theme"], safe_float(r.get("momentum")), safe_float(r.get("hype")),
                  safe_float(r.get("composite")), int(r["n"]), int(r["cover"]), now)
                 for r in rows])
    except Exception as e:
        logger.warning("theme history write skipped: %s", e)


def theme_momentum_deltas() -> dict[str, float]:
    """Momentum change per theme vs the PREVIOUS recorded snapshot (latest minus
    prior). Empty dict when fewer than two snapshots exist — the Δ column shows '—'
    until a second full scan is recorded."""
    try:
        with get_conn() as conn:
            snaps = [r[0] for r in conn.execute(
                # GROUP BY + MAX(id) instead of DISTINCT + ORDER BY id: Postgres
                # rejects ordering a DISTINCT by a column outside the select list
                # (sqlite tolerates it); this form is valid and identical on both.
                "SELECT snapshot FROM theme_history GROUP BY snapshot "
                "ORDER BY MAX(id) DESC LIMIT 2")]
            if len(snaps) < 2:
                return {}
            cur = dict(conn.execute(
                "SELECT theme, momentum FROM theme_history WHERE snapshot = ?", (snaps[0],)))
            prev = dict(conn.execute(
                "SELECT theme, momentum FROM theme_history WHERE snapshot = ?", (snaps[1],)))
    except Exception as e:
        logger.warning("theme history read skipped: %s", e)
        return {}
    return {t: float(cur[t]) - float(prev[t])
            for t in cur if t in prev
            and cur[t] is not None and prev[t] is not None}


def render_themes(results: list[dict]) -> None:
    if not results:
        st.info(tr("need_run_engine"))
        return
    by_ticker = {r["ticker"]: r for r in results}

    def _avg(present, key):
        vals = [safe_float(x.get(key), float("nan")) for x in present]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    # Smart-money overlay: superinvestor 13F NEW buys, summed per theme. US-only and
    # ~45-day lagged by nature; cached a day. Independent of the yfinance scan, so it
    # covers every US member regardless of quick-scan. Fail-safe -> NaN if unavailable.
    sci = {}
    if sec_research is not None:
        all_us = sorted({t for ticks in THEMES.values() for t in ticks if "." not in t})
        try:
            sci = _superinvestor_counts_cached(tuple(all_us))
        except Exception:
            sci = {}
    smart_ok = bool(sci)

    rows = []
    for theme, ticks in THEMES.items():
        present = [by_ticker[t] for t in ticks if t in by_ticker]
        if not present:
            continue   # nothing from this theme was scanned -> omit (coverage note explains)
        smart = float(sum(int(sci.get(t, 0) or 0) for t in ticks)) if smart_ok else float("nan")
        # Breadth: does the whole basket participate, or is one megacap carrying the
        # average? (>SMA50 % and BUY %). This is the honest fix for megacap skew —
        # not cap-weighting, which would amplify it.
        above = sum(1 for x in present
                    if safe_float(x.get("price"), float("nan")) > safe_float(x.get("sma50"), float("inf")))
        buys = sum(1 for x in present if str(x.get("recommendation")) == "BUY")
        rows.append({"theme": theme,
                     "momentum": _avg(present, "momentum"),
                     "hype": _avg(present, "hype_score"),
                     "smart": smart,
                     "composite": _avg(present, "composite"),
                     "ret1m": _avg(present, "ret_1m"),
                     "breadth": above / len(present) * 100.0,
                     "buyspct": buys / len(present) * 100.0,
                     "n": len(present), "cover": len(ticks)})
    if not rows:
        st.warning(tr("themes_no_coverage"))
        return
    # Rotation memory: record full scans only, then diff vs the previous snapshot.
    if not st.session_state.get("scan_is_quick", True):
        record_theme_snapshot(rows)
    deltas = theme_momentum_deltas()
    df = pd.DataFrame(rows)
    df["delta"] = df["theme"].map(lambda t: deltas.get(t, float("nan")))

    st.caption(tr("themes_caption"))

    # Multi-lens bubble: momentum (x) vs attention (y), bubble size = smart-money buys
    # (falls back to coverage when 13F is unavailable). Lets you read quadrants at a
    # glance — strong-but-quiet (right/low) vs loud-but-weak (left/high).
    size_field, size_title = (("smart", tr("col_theme_smart")) if smart_ok
                              else ("n", tr("col_theme_coverage")))
    bubble = (alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X("momentum:Q", axis=alt.Axis(title=tr("themes_momentum_axis"))),
        y=alt.Y("hype:Q", axis=alt.Axis(title=tr("themes_hype_axis"))),
        size=alt.Size(f"{size_field}:Q", title=size_title, scale=alt.Scale(range=[80, 900])),
        color=alt.Color("theme:N", legend=None),
        tooltip=[alt.Tooltip("theme:N", title=tr("col_theme")),
                 alt.Tooltip("momentum:Q", title=tr("col_theme_momentum"), format=".0f"),
                 alt.Tooltip("hype:Q", title=tr("col_theme_hype"), format=".0f"),
                 alt.Tooltip("smart:Q", title=tr("col_theme_smart"), format=".0f"),
                 alt.Tooltip("composite:Q", title=tr("col_theme_strength"), format=".0f")])
        .properties(height=380, width="container"))
    st.altair_chart(bubble)
    st.caption(tr("themes_bubble_caption"))

    # Rank-by lens: momentum (strong now), attention (hype volume), or smart-money (13F
    # new buys). Triangulation, not prediction — see the caption caveats.
    lenses = {tr("themes_rank_momentum"): ("momentum", tr("themes_momentum_axis")),
              tr("themes_rank_hype"): ("hype", tr("themes_hype_axis")),
              tr("themes_rank_smart"): ("smart", tr("themes_smart_axis"))}
    choice = st.radio(tr("themes_rank_by"), list(lenses.keys()), horizontal=True)
    sort_key, axis_title = lenses[choice]
    df = df.sort_values(sort_key, ascending=False, na_position="last")

    chart = (alt.Chart(df).mark_bar()
             .encode(
                 x=alt.X(f"{sort_key}:Q", axis=alt.Axis(title=axis_title)),
                 y=alt.Y("theme:N", sort="-x", axis=alt.Axis(title=None)),
                 color=alt.Color("theme:N", legend=None),
                 tooltip=[alt.Tooltip("theme:N", title=tr("col_theme")),
                          alt.Tooltip("momentum:Q", title=tr("col_theme_momentum"), format=".0f"),
                          alt.Tooltip("hype:Q", title=tr("col_theme_hype"), format=".0f"),
                          alt.Tooltip("smart:Q", title=tr("col_theme_smart"), format=".0f"),
                          alt.Tooltip("composite:Q", title=tr("col_theme_strength"), format=".0f")])
             .properties(height=max(180, 28 * len(df)), width="container"))
    st.altair_chart(chart)

    table = df.assign(
        coverage=lambda d: d["n"].astype(str) + "/" + d["cover"].astype(str),
        delta_disp=lambda d: d["delta"].map(
            lambda v: "—" if math.isnan(v) else f"{'▲' if v >= 0 else '▼'} {v:+.1f}"))[
        ["theme", "momentum", "delta_disp", "breadth", "buyspct", "hype", "smart",
         "composite", "ret1m", "coverage"]].rename(columns={
        "theme": tr("col_theme"), "momentum": tr("col_theme_momentum"),
        "delta_disp": tr("col_theme_delta"),
        "breadth": tr("col_theme_breadth"), "buyspct": tr("col_theme_buyspct"),
        "hype": tr("col_theme_hype"), "smart": tr("col_theme_smart"),
        "composite": tr("col_theme_strength"), "ret1m": tr("col_theme_ret1m"),
        "coverage": tr("col_theme_coverage")})
    st.dataframe(table.style.format({tr("col_theme_momentum"): "{:.0f}", tr("col_theme_hype"): "{:.0f}",
                 tr("col_theme_breadth"): "{:.0f}%", tr("col_theme_buyspct"): "{:.0f}%",
                 tr("col_theme_smart"): "{:.0f}", tr("col_theme_strength"): "{:.0f}",
                 tr("col_theme_ret1m"): "{:+.1f}%"}, na_rep="—"),
                 width="stretch", hide_index=True)
    st.caption(tr("themes_coverage_note"))
    st.caption(tr("themes_delta_note"))

    if not smart_ok:
        st.caption(tr("themes_smart_unavailable"))

    # drill into one theme -> its scanned stocks, with per-name smart-money buys
    st.markdown(f"#### {tr('themes_top_header')}")
    pick = st.selectbox(tr("themes_select"), list(df["theme"]))
    members = sorted([by_ticker[t] for t in THEMES[pick] if t in by_ticker],
                     key=lambda x: safe_float(x.get("composite"), float("-inf")), reverse=True)
    mdf = pd.DataFrame([{
        tr("col_ticker"): x["ticker"], tr("col_company"): x.get("name", x["ticker"]),
        tr("col_theme_strength"): safe_float(x.get("composite"), float("nan")),
        tr("col_theme_momentum"): safe_float(x.get("momentum"), float("nan")),
        tr("col_theme_hype"): safe_float(x.get("hype_score"), float("nan")),
        tr("col_theme_smart"): float(int(sci.get(x["ticker"], 0) or 0)) if smart_ok else float("nan"),
        tr("col_theme_ret1m"): safe_float(x.get("ret_1m"), float("nan")),
        tr("why_call"): str(x.get("recommendation", "—")),
    } for x in members])
    st.dataframe(mdf.style.format({tr("col_theme_strength"): "{:.0f}", tr("col_theme_momentum"): "{:.0f}",
                 tr("col_theme_hype"): "{:.0f}", tr("col_theme_smart"): "{:.0f}",
                 tr("col_theme_ret1m"): "{:+.1f}%"}, na_rep="—"),
                 width="stretch", hide_index=True)

    # Deep-scan echo: cross-reference the latest deep-scan finalists (any market,
    # pulled from session state — no new fetching) against these theme baskets via
    # their Yahoo sector/industry. This is the Themes <-> Deep Scan bridge: theme
    # heat measured on the curated basket, confirmed or contradicted by the top
    # names from the much wider deep universe — including names the baskets don't
    # know about, which is exactly the discovery the curated lists can't provide.
    # NOTE: placed at the very end of the tab (nothing else follows), so an empty
    # state here can never skip the leaderboard or drill-down rendered above.
    deep_all = []
    for reg in DEEP_UNIVERSES:
        deep_all.extend(st.session_state.get(f"deep_results_{reg}") or [])
    st.markdown(f"##### {tr('themes_deep_echo')}")
    if not deep_all:
        st.caption(tr("themes_deep_echo_hint"))
    else:
        echo_rows = [{
            tr("col_theme"): x["theme_match"],
            tr("col_ticker"): x["ticker"],
            tr("col_company"): x.get("name", x["ticker"]),
            "_region": region_name(x.get("region", "")),
            tr("col_theme_strength"): safe_float(x.get("composite"), float("nan")),
            tr("deep_col_score"): safe_float(x.get("screen_score"), float("nan")),
        } for x in deep_all if x.get("theme_match")]
        if not echo_rows:
            st.caption(tr("themes_deep_echo_none"))
        else:
            edf = (pd.DataFrame(echo_rows)
                   .sort_values([tr("col_theme"), tr("col_theme_strength")], ascending=[True, False])
                   .groupby(tr("col_theme"), sort=False).head(3)
                   .rename(columns={"_region": tr("regions_to_scan")}))
            st.dataframe(edf.style.format({tr("col_theme_strength"): "{:.0f}",
                         tr("deep_col_score"): "{:.0f}"}, na_rep="—"),
                         width="stretch", hide_index=True)
            st.caption(tr("themes_deep_echo_note"))


def render_deep_dive(results: list[dict]) -> None:
    if not results:
        st.info(tr("need_run_history"))
        return
    rank_of = {x["ticker"]: i + 1 for i, x in enumerate(
        sorted(results, key=lambda x: safe_float(x.get("composite"), float("-inf")), reverse=True))}

    names = {r["ticker"]: r.get("name", r["ticker"]) for r in results}
    opts = [r["ticker"] for r in results]
    # "deep_dive_pick" may have been set by a row click in the Regional tab (which
    # renders earlier in the same rerun). Sanitize: a stale pick from a previous
    # scan that's no longer in the results must not crash the keyed selectbox.
    if st.session_state.get("deep_dive_pick") not in opts:
        st.session_state.pop("deep_dive_pick", None)
    pick = st.selectbox(tr("select_profile"), opts,
                        format_func=lambda t: f"{t} — {names.get(t, t)}",
                        key="deep_dive_pick")
    r = next(x for x in results if x["ticker"] == pick)
    weights = get_latest_weights()
    st.markdown(f"### {r['ticker']} — {r.get('name', r['ticker'])}")

    # --- headline: composite, call, rank ---
    comp = safe_float(r.get("composite"), float("nan"))
    rec = str(r.get("recommendation", "—"))
    hc = st.columns(3)
    hc[0].markdown(metric_card(tr("why_composite"), fmt_num(comp, 1),
                   positive=(not math.isnan(comp) and comp >= BUY_THRESHOLD)), unsafe_allow_html=True)
    hc[1].markdown(metric_card(tr("why_call"), rec, positive=(rec == "BUY")), unsafe_allow_html=True)
    hc[2].markdown(metric_card(tr("why_rank"), f"#{rank_of.get(pick, '—')} / {len(results)}"),
                   unsafe_allow_html=True)

    # --- factor profile: six-axis radar, this pick vs the scan average ---
    # The contribution bars below answer "why this composite"; the radar answers
    # "what KIND of stock is this" (momentum monster vs. value dog) at a glance.
    # plotly is optional (guarded import) — without it the section is skipped.
    if go is not None:
        def _fval(x: dict, f: str) -> float:
            return safe_float(x.get("hype_score" if f == "hype" else f), 50.0)
        cats = [tr(f"factor_{f}") for f in FACTORS]
        vals = [_fval(r, f) for f in FACTORS]
        avgs = [float(np.mean([_fval(x, f) for x in results])) for f in FACTORS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=avgs + avgs[:1], theta=cats + cats[:1], name=tr("radar_scan_avg"),
            line=dict(color="#94a3b8", dash="dot"), opacity=0.65))
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=cats + cats[:1], name=r["ticker"],
            fill="toself", line=dict(color="#3b82f6"),
            fillcolor="rgba(59,130,246,0.25)"))
        fig.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(range=[0, 100], showticklabels=True,
                                       tickfont=dict(size=10), gridcolor="rgba(148,163,184,0.25)"),
                       angularaxis=dict(gridcolor="rgba(148,163,184,0.25)")),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
            height=360, margin=dict(l=50, r=50, t=30, b=30))
        st.markdown(f"#### {tr('radar_header')}")
        st.plotly_chart(fig, width="stretch", key=f"radar_{pick}")
    else:
        st.caption(tr("radar_plotly_hint"))

    # --- why this pick: exact weighted-contribution decomposition ---
    st.markdown(f"#### {tr('why_header')}")
    name_map = {"momentum": tr("factor_momentum"), "value": tr("factor_value"),
                "technical": tr("factor_technical"), "hype": tr("factor_hype"),
                "quality": tr("factor_quality"), "theme": tr("factor_theme")}
    bd = _pick_breakdown(r, weights)
    bd["label"] = bd["factor"].map(name_map)
    total = float(bd["contribution"].sum())
    bd_sorted = bd.sort_values("contribution", ascending=False)

    chart = (alt.Chart(bd_sorted)
             .mark_bar()
             .encode(
                 x=alt.X("contribution:Q", axis=alt.Axis(title=tr("why_contrib_axis"))),
                 y=alt.Y("label:N", sort="-x", axis=alt.Axis(title=None)),
                 color=alt.Color("label:N", legend=None),
                 tooltip=[alt.Tooltip("label:N", title=tr("col_factor")),
                          alt.Tooltip("score:Q", title=tr("col_score"), format=".0f"),
                          alt.Tooltip("weight:Q", title=tr("col_weight"), format=".0%"),
                          alt.Tooltip("contribution:Q", title=tr("col_contribution"), format=".1f")])
             .properties(height=210, width="container"))
    st.altair_chart(chart)
    st.caption(tr("why_contrib_caption"))

    show = (bd_sorted.assign(share=lambda d: (d["contribution"] / total * 100.0) if total else float("nan"))
            [["label", "score", "weight", "contribution", "share"]]
            .rename(columns={"label": tr("col_factor"), "score": tr("col_score"),
                             "weight": tr("col_weight"), "contribution": tr("col_contribution"),
                             "share": tr("col_share")}))
    st.dataframe(show.style.format({tr("col_score"): "{:.0f}", tr("col_weight"): "{:.0%}",
                 tr("col_contribution"): "{:.1f}", tr("col_share"): "{:.0f}%"}, na_rep="—"),
                 width="stretch", hide_index=True)

    # plain-language rationale: top two contributors as drivers, lowest score as the drag
    top, second = bd_sorted.iloc[0], bd_sorted.iloc[1]
    weak = bd.sort_values("score").iloc[0]
    share2 = ((top["contribution"] + second["contribution"]) / total * 100.0) if total else float("nan")
    st.info(tr("why_rationale", d1=top["label"], s1=f"{top['score']:.0f}",
               d2=second["label"], s2=f"{second['score']:.0f}", share=f"{share2:.0f}",
               w=weak["label"], sw=f"{weak['score']:.0f}"))

    # --- evidence: the real numbers behind each factor ---
    st.markdown(f"#### {tr('why_evidence')}")
    ev = pd.DataFrame(_evidence_rows(r), columns=[tr("col_factor"), tr("col_metric"), tr("col_value")])
    st.dataframe(ev, width="stretch", hide_index=True)

    # --- price context ---
    # Restored snapshot results carry no history frame (stripped before
    # serialising) — lazily refetch just the picked ticker's series, cache it
    # back onto the dict, and degrade to a caption if the fetch fails.
    hist = r.get("history")
    if hist is None:
        try:
            hist = fetch_history(r["ticker"])
            r["history"] = hist
        except Exception:
            hist = None
    if hist is not None and "Close" in getattr(hist, "columns", []):
        st.markdown(f"#### {tr('price_trend')}")
        chart_df = pd.DataFrame({"Close": hist["Close"],
                                 "SMA20": hist["Close"].rolling(20).mean(),
                                 "SMA50": hist["Close"].rolling(50).mean()})
        st.line_chart(chart_df.dropna())
    else:
        st.caption(tr("price_chart_unavailable"))

    mentions = safe_float(r.get("hype_mentions", 0), 0.0)
    st.markdown(metric_card(tr("hype_buzz_label"), f"{mentions:g}", positive=mentions > 0),
                unsafe_allow_html=True)
    st.caption(tr("hype_buzz_caption"))
    if _LAST_HYPE_STATUS:
        st.caption(_LAST_HYPE_STATUS)
    if _LAST_HYPE_FETCHED_AT:
        st.caption(tr("hype_updated", ago=_ago(_LAST_HYPE_FETCHED_AT)))

def render_engine_audit(update_prices: bool = True) -> None:
    st.markdown(f"### {tr('audit_header_text')}")
    st.caption(tr("persistence_note"))
    st.caption(tr("benchmark_note"))
    df = get_recommendations()
    done = df[df["outcome"].notna()] if not df.empty else pd.DataFrame()

    if df.empty:
        st.info(tr("no_tracks"))
        return

    overall_win = 100 * done["outcome"].mean() if not done.empty else float("nan")
    c = st.columns(3)
    c[0].markdown(metric_card(tr("card_archived_calls"), str(len(df))), unsafe_allow_html=True)
    c[1].markdown(metric_card(tr("card_perf_verified"), str(len(done))), unsafe_allow_html=True)
    c[2].markdown(metric_card(tr("card_accuracy"), f"{overall_win:.0f}%" if not math.isnan(overall_win) else "—", positive=(not math.isnan(overall_win) and overall_win >= 50)), unsafe_allow_html=True)

    if not done.empty:
        st.markdown(f"#### {tr('winrate_over_time')}")
        wr = (done.assign(week=done["rec_date"].dt.to_period("W").dt.start_time)
                  .groupby("week")["outcome"].mean().mul(100).round(1))
        st.bar_chart(wr.rename(tr("winrate_col")))

    wh = get_weight_history()
    if len(wh) >= 2:
        st.markdown(f"#### {tr('kpi_weight_evolution')}")
        plot = wh.copy()
        plot["update_date"] = pd.to_datetime(plot["update_date"], errors="coerce")
        long = (plot.melt(id_vars="update_date", value_vars=FACTORS,
                          var_name="factor", value_name="weight")
                    .dropna(subset=["update_date"]))
        # Altair lets us force the x-axis to show the DATE only (not 6AM/6PM ticks).
        # width="container" keeps it responsive without the deprecated container flag.
        chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("update_date:T", axis=alt.Axis(format="%Y-%m-%d", title=None)),
                y=alt.Y("weight:Q", axis=alt.Axis(title=None)),
                color=alt.Color("factor:N", legend=alt.Legend(title=None)),
            )
            .properties(height=320, width="container")
        )
        st.altair_chart(chart)

    # --- Historical Performance: live return of every logged mock-portfolio pick ---
    st.markdown(f"#### {tr('historical_performance')}")
    mp = get_mock_portfolio()
    if mp.empty:
        st.caption(tr("no_portfolio_picks"))
        return
    mp["date"] = pd.to_datetime(mp["timestamp"], errors="coerce")
    one_per = st.checkbox(tr("audit_one_per_stock"), value=True)

    if one_per:
        # one row per stock: anchor on its FIRST recommendation (so return reflects the
        # full development since the engine first flagged it) and count how often picked
        view = mp.sort_values("date").drop_duplicates("ticker", keep="first").copy()
        view["times"] = view["ticker"].map(mp.groupby("ticker").size()).astype(int)
    else:
        view = mp.head(40).copy()

    uniq = list(view["ticker"].unique())
    if update_prices:
        # SINGLE bulk request for every unique ticker — replaces the per-ticker fetch
        # storm that was tripping Yahoo's rate limiter and crashing the app. Tuple:
        # the function is cached (5 min), so the argument must be hashable.
        prices = _bulk_latest_close(tuple(uniq))
    else:
        prices = {t: float("nan") for t in uniq}   # checkbox off -> leave prices blank
    view["current"] = view["ticker"].map(prices)
    view["return_pct"] = (view["current"] / view["recommendation_price"] - 1.0) * 100.0
    view["date_str"] = view["date"].dt.date.astype(str)
    view["name"] = view["name"].fillna("")   # older rows logged before names were stored
    reason_map = {"Top Growth": tr("reason_growth"), "Top Dividend": tr("reason_dividend")}
    view["reason"] = view["reason"].map(lambda x: reason_map.get(x, x))

    ret_col = tr("col_return_pct")
    rec_col, cur_col = tr("col_rec_price"), tr("col_current_price")
    if one_per:
        view = view.sort_values("return_pct", ascending=False, na_position="last")
        disp = view[["date_str", "ticker", "name", "reason", "times",
                     "recommendation_price", "current", "return_pct"]].rename(columns={
            "date_str": tr("col_first_rec"), "ticker": tr("col_ticker"), "name": tr("col_company"),
            "reason": tr("col_reason"), "times": tr("col_times"),
            "recommendation_price": rec_col, "current": cur_col, "return_pct": ret_col})
    else:
        disp = view[["date_str", "ticker", "name", "reason",
                     "recommendation_price", "current", "return_pct"]].rename(columns={
            "date_str": tr("col_date"), "ticker": tr("col_ticker"), "name": tr("col_company"),
            "reason": tr("col_reason"),
            "recommendation_price": rec_col, "current": cur_col, "return_pct": ret_col})

    def _style_returns(series):
        styles = []
        for v in series:
            if pd.isna(v): styles.append("")
            elif v >= 0: styles.append("color: #16a34a; font-weight: 600;")  # green winner
            else: styles.append("color: #dc2626; font-weight: 600;")          # red loser
        return styles

    styled = (disp.style
              .apply(_style_returns, subset=[ret_col])
              .format({rec_col: "{:,.2f}", cur_col: "{:,.2f}", ret_col: "{:+.2f}%"}, na_rep="—"))
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption(tr("audit_one_per_note") if one_per else tr("historical_perf_note"))
    if update_prices:
        unpriced = sorted({t for t in uniq if pd.isna(prices.get(t, float("nan")))})
        if unpriced:
            st.caption(tr("prices_unavailable", tickers=", ".join(unpriced)))
    else:
        st.caption(tr("prices_skipped"))

def _sell_pill(verdict: str) -> str:
    cls = {"high": "qc-sell", "mixed": "qc-hold", "low": "qc-buy"}.get(verdict, "qc-hold")
    return f'<span class="qc-pill {cls}">{tr(f"sell_verdict_{verdict}")}</span>'

def _render_sell_detail(data: dict) -> None:
    """Full single-ticker sell breakdown (reused by the optional drill-down)."""
    cur = (data["currency"] + " ") if data["currency"] else ""
    comp_txt = "—" if math.isnan(data["composite"]) else f"{data['composite']:.0f}/100"
    price_txt = "—" if math.isnan(data["price"]) else f"{cur}{data['price']:,.2f}"
    c = st.columns([2, 1, 1])
    c[0].markdown(
        f'<div class="qc-card"><span class="qc-ticker">{data["ticker"]}</span> {_sell_pill(data["verdict"])}'
        f'<br><span class="qc-sub">{data["name"]}</span></div>',
        unsafe_allow_html=True,
    )
    c[1].markdown(metric_card(tr("sell_pressure"), comp_txt,
                              positive=(not math.isnan(data["composite"]) and data["composite"] < SELL_LOW)),
                  unsafe_allow_html=True)
    c[2].markdown(metric_card(tr("price_label"), price_txt), unsafe_allow_html=True)

    # --- why this is a sell: exact weighted-contribution decomposition ---
    st.markdown(f"#### {tr('sell_why_header')}")
    scored = [(k, float(s)) for k, s, _ in data["signals"] if s is not None]
    if not scored:
        st.caption(tr("sell_no_signal_data"))
    else:
        den = sum(SELL_KPI_WEIGHTS[k] for k, _ in scored) or 1.0
        sb = pd.DataFrame([{
            "label": tr(f"kpi_{k}"), "score": s, "weight": SELL_KPI_WEIGHTS[k],
            "contribution": SELL_KPI_WEIGHTS[k] * s / den,    # contributions sum to the composite
        } for k, s in scored])
        total = float(sb["contribution"].sum())
        sb_sorted = sb.sort_values("contribution", ascending=False)

        chart = (alt.Chart(sb_sorted).mark_bar()
                 .encode(
                     x=alt.X("contribution:Q", axis=alt.Axis(title=tr("sell_why_axis"))),
                     y=alt.Y("label:N", sort="-x", axis=alt.Axis(title=None)),
                     color=alt.Color("label:N", legend=None),
                     tooltip=[alt.Tooltip("label:N", title=tr("col_kpi")),
                              alt.Tooltip("score:Q", title=tr("col_score"), format=".0f"),
                              alt.Tooltip("weight:Q", title=tr("col_weight"), format=".0%"),
                              alt.Tooltip("contribution:Q", title=tr("col_contribution"), format=".1f")])
                 .properties(height=210, width="container"))
        st.altair_chart(chart)
        st.caption(tr("sell_why_caption"))

        show = (sb_sorted.assign(share=lambda d: (d["contribution"] / total * 100.0) if total else float("nan"))
                [["label", "score", "weight", "contribution", "share"]]
                .rename(columns={"label": tr("col_kpi"), "score": tr("col_score"),
                                 "weight": tr("col_weight"), "contribution": tr("col_contribution"),
                                 "share": tr("col_share")}))
        st.dataframe(show.style.format({tr("col_score"): "{:.0f}", tr("col_weight"): "{:.0%}",
                     tr("col_contribution"): "{:.1f}", tr("col_share"): "{:.0f}%"}, na_rep="—"),
                     width="stretch", hide_index=True)

        top = sb_sorted.iloc[0]
        if len(sb_sorted) >= 2:
            second = sb_sorted.iloc[1]
            share2 = ((top["contribution"] + second["contribution"]) / total * 100.0) if total else float("nan")
            st.info(tr("sell_why_rationale", d1=top["label"], s1=f"{top['score']:.0f}",
                       d2=second["label"], s2=f"{second['score']:.0f}", share=f"{share2:.0f}"))
        else:
            share1 = (top["contribution"] / total * 100.0) if total else float("nan")
            st.info(tr("sell_why_rationale_one", d1=top["label"], s1=f"{top['score']:.0f}",
                       share=f"{share1:.0f}"))

        skipped = [tr(f"kpi_{k}") for k, s, _ in data["signals"] if s is None]
        if skipped:
            st.caption(tr("sell_not_scored", items=", ".join(skipped)))

    st.markdown(f"#### {tr('sell_breakdown_title')}")

    readings = [{
        tr("col_kpi"): tr(f"kpi_{k}"),
        tr("col_signal"): "—" if s is None else f"{s:.0f}",
        tr("col_reading"): detail,
    } for k, s, detail in data["signals"]]
    st.dataframe(pd.DataFrame(readings), width="stretch", hide_index=True)

    st.markdown(f"#### {tr('sell_recent_downgrades')}")
    if data["changes"]:
        cdf = pd.DataFrame(data["changes"]).rename(columns={
            "date": tr("col_date"), "firm": tr("col_firm"),
            "from": tr("col_from"), "to": tr("col_to"), "action": tr("col_action"),
        })
        st.dataframe(cdf, width="stretch", hide_index=True)
    else:
        st.caption(tr("sell_no_analyst"))

    st.markdown(f"#### {tr('sell_insider_activity')}")
    ins = data["insider"]
    if ins["available"] and ins["rows"]:
        idf = pd.DataFrame(ins["rows"]).rename(columns={
            "date": tr("col_date"), "insider": tr("col_insider"),
            "transaction": tr("col_transaction"), "shares": tr("col_shares"), "value": tr("col_value"),
        })
        st.dataframe(idf, width="stretch", hide_index=True)
    else:
        st.caption(tr("sell_no_insider"))

def _parse_portfolio_input(pasted: str, uploaded) -> list:
    """Build a deduped ticker list from a comma/space/newline text area and/or a CSV
    whose 'Ticker' or 'Symbol' column (case-insensitive) holds the symbols."""
    tickers = []
    if pasted:
        tickers += [s.strip().upper() for s in re.split(r"[,\n;\s]+", pasted) if s.strip()]
    if uploaded is not None:
        try:
            cdf = pd.read_csv(uploaded)
            col = next((c for c in cdf.columns if str(c).strip().lower() in ("ticker", "symbol")), None)
            if col is None:
                st.warning(tr("sell_csv_nocol"))
            else:
                tickers += [str(x).strip().upper() for x in cdf[col].dropna().tolist() if str(x).strip()]
        except Exception:
            st.warning(tr("sell_csv_error"))
    return list(dict.fromkeys(t for t in tickers if t))   # dedupe, preserve order

def render_sell_signals() -> None:
    st.markdown(f"### {tr('sell_header')}")
    st.caption(tr("sell_disclaimer"))
    st.caption(tr("sell_privacy_note"))

    pasted = st.text_area(tr("sell_paste_label"), key="sell_paste", placeholder="AAPL, MSFT, NVDA")
    uploaded = st.file_uploader(tr("sell_upload_label"), type=["csv"], key="sell_csv")

    tickers = _parse_portfolio_input(pasted, uploaded)
    if tickers:
        # Stored in session_state only — never written to the SQLite DB — so each
        # user's portfolio stays private in a shared/multi-user deployment.
        st.session_state["portfolio_tickers"] = tickers

    portfolio = st.session_state.get("portfolio_tickers", [])
    if portfolio:
        preview = ", ".join(portfolio[:20]) + (" …" if len(portfolio) > 20 else "")
        st.caption(tr("sell_loaded", n=len(portfolio), tickers=preview))

    if st.button(tr("sell_scan_btn"), width="stretch", disabled=not portfolio):
        rows = []
        details = {}
        # OPTIMISATION: one bulk yf.download for every name's history up front, so
        # the per-ticker loop reuses it instead of making a history request each.
        with st.spinner(tr("sell_bulk_fetch", total=len(portfolio))):
            histories = get_histories(portfolio, period="1y")
        prog = st.progress(0.0, text=tr("sell_scan_progress", done=0, total=len(portfolio)))
        for i, tk in enumerate(portfolio, start=1):
            try:
                data = analyze_sell_signals(tk, hist=histories.get(tk))
            except Exception:
                data = None
            if data is not None:
                rows.append({"ticker": data["ticker"], "price": data["price"],
                             "composite": data["composite"],
                             "verdict": tr(f"sell_verdict_{data['verdict']}")})
                details[data["ticker"]] = data   # keep the full breakdown for the drill-down
            else:
                rows.append({"ticker": tk, "price": float("nan"),
                             "composite": float("nan"), "verdict": tr("sell_verdict_na")})
            prog.progress(i / len(portfolio), text=tr("sell_scan_progress", done=i, total=len(portfolio)))
        prog.empty()
        st.session_state["portfolio_results"] = rows
        st.session_state["portfolio_details"] = details

    results = st.session_state.get("portfolio_results", [])
    if not results:
        st.info(tr("sell_need_portfolio"))
        return

    rdf = pd.DataFrame(results).sort_values(by="composite", ascending=False, na_position="last")
    comp_col, price_col = tr("sell_pressure"), tr("col_current_price")
    verdict_col, tick_col = tr("col_verdict"), tr("col_ticker")
    disp = rdf.rename(columns={"ticker": tick_col, "price": price_col, "composite": comp_col, "verdict": verdict_col})

    def _style_pressure(series):
        out = []
        for v in series:
            if pd.isna(v): out.append("")
            elif v >= SELL_HIGH: out.append("color: #dc2626; font-weight: 600;")   # elevated -> red
            elif v < SELL_LOW: out.append("color: #16a34a; font-weight: 600;")      # low -> green
            else: out.append("color: #d97706; font-weight: 600;")                   # mixed -> amber
        return out

    styled = (disp.style.apply(_style_pressure, subset=[comp_col])
              .format({price_col: "{:,.2f}", comp_col: "{:.0f}"}, na_rep="—"))
    st.dataframe(styled, width="stretch", hide_index=True)

    # Drill-down: show the full breakdown computed during the scan — no re-fetch, so it
    # works for any number of stocks (re-fetching here got rate-limited after a big scan).
    with st.expander(tr("sell_detail_expander")):
        choices = [r["ticker"] for r in results]
        sel = st.selectbox(tr("sell_detail_select"), choices, key="sell_detail_pick")
        detail = st.session_state.get("portfolio_details", {}).get(sel)
        if detail is None and sel:        # fallback (e.g. results from an older session)
            with st.spinner(tr("sell_spinner", ticker=sel)):
                try:
                    detail = analyze_sell_signals(sel)
                except Exception:
                    detail = None   # a raised (rate-limited) sub-fetch -> show the notice
        if detail is None:
            st.warning(tr("sell_no_data", ticker=sel))
        else:
            _render_sell_detail(detail)

# ----------------------------------------------------------------------------
# Help / About tab — in-app documentation, bilingual (follows the language menu).
# Content lives here (not in TRANSLATIONS) because it's long-form prose per
# language, not templated UI strings.
# ----------------------------------------------------------------------------
_HELP_SECTIONS = {
    "en": [
        ("🚀 What is this app?",
         """The **Alpha Quant Engine** scans a global stock universe (US, Japan, Europe, China), scores
every name on **six KPI factors**, and turns the weighted composite into a BUY / HOLD / SELL call.
It is a *screening and research* tool: it surfaces candidates and shows its work — it is **not
financial advice**, and its signals are based on free, sometimes-delayed data sources."""),
        ("🧭 Quick start",
         """1. In the sidebar, pick the **regions** to scan and leave **Quick scan** on (first 10 names
per region — fast, and gentle on the data sources).
2. Press **Run scan**. Results populate the Top Selections, Regional, Category, Themes and Deep-Dive tabs.
3. Open **Deep-Dive Analysis** on any name to see *exactly* why it scored what it did — an exact
weighted-contribution breakdown, plus the raw evidence behind every factor.
4. Use the **Deep Scan** tab to hunt outside the curated universe, and **promote** winners so they
ride along with your daily scans.
5. The engine logs its picks and, two weeks later, grades itself and **re-tunes its own factor
weights** (see "How the engine learns" below)."""),
        ("📊 The six KPI factors",
         """Every stock gets a 0–100 score per factor; the composite is the weighted average.
- **Momentum** — price vs. its 50-day average, MACD direction, and the 1-month return.
- **Value** — trailing P/E mapped to a score (cheaper = higher), plus a dividend-yield bonus.
- **Technical** — RSI positioning and Bollinger %B; rewards healthy, mid-band setups over
overbought/oversold extremes.
- **Hype** — volume-breakout detection vs. a 30-day baseline, boosted by live retail/news buzz
(Reddit, GDELT), the Yahoo! Japan forum poll for Japanese names, and a short-squeeze modifier.
- **Quality** — return on equity (profitability). Missing data scores neutral, never penalised.
- **Theme** *(new)* — industry-rotation strength: the average momentum of the *other* scanned members
of the stock's theme basket (AI, Semis, Defense, Clean Energy…), damped so a hot industry helps a
name without drowning out its own signals. Stocks without a theme score a neutral 50.

**Calls:** composite ≥ 65 → BUY · < 45 → SELL · otherwise HOLD."""),
        ("🔁 How the engine learns (walk-forward loop)",
         """Each full scan logs the day's Top-3 Growth and Top-3 Dividend picks with a snapshot of their
factor scores. Once a pick is **14 days** old, the engine checks whether it beat its home-market
benchmark (S&P 500 for US names, Nikkei for Japan, etc.). Factors that scored highly on *winners*
get nudged **up**; factors that scored highly on *losers* get nudged **down** — with a floor so no
factor ever dies completely. The **Systems Audit** tab charts the weight evolution and win rate.
The new Theme factor starts small (12%) and must earn a bigger weight through this loop."""),
        ("🔬 Deep Scan & promotions",
         """The Deep Scan is a two-stage funnel per market. **Stage 1** downloads prices for a few hundred
names in one bulk request and ranks them on a pure price screen (momentum blend, volume surge,
52-week-high proximity, with a falling-knife cap). **Stage 2** runs full fundamental scoring on the
top finalists only. The best finalists can be **promoted** (US 5 / JP 3 / CN 2) into your main daily
scan; promotions rotate each time you re-run and promote. On cloud deployments the promotion table
resets on redeploy — paste the offered `PROMOTED_TICKERS` snippet into `config.py` to make it stick."""),
        ("📡 Data sources & limitations",
         """Prices and fundamentals come from **Yahoo Finance** (with a free Stooq fallback for US names and
an optional FMP key for fundamentals). Sentiment comes from **Reddit**, **GDELT news**, and the
**Yahoo! Japan forum poll** for Japanese tickers. All sources are free/unofficial: they can be
delayed, throttled, or blocked, and every fetch is wrapped so a dead source degrades a signal to
neutral instead of crashing a scan. If a whole scan fails, it is almost always a temporary Yahoo
rate limit — wait a few minutes and use Quick scan."""),
        ("⚠️ Disclaimer",
         """This app is an educational screening tool. Scores are mechanical transformations of public,
sometimes-delayed data; they can be wrong, stale, or based on incomplete inputs. Nothing here is a
recommendation to buy or sell any security. Always do your own research and consider consulting a
licensed financial advisor before trading."""),
    ],
    "ja": [
        ("🚀 このアプリについて",
         """**Alpha Quant Engine** は、グローバル株式ユニバース（米国・日本・欧州・中国）をスキャンし、
各銘柄を **6つのKPIファクター** で採点、加重平均のコンポジットスコアから BUY / HOLD / SELL を
判定します。これは*スクリーニング・リサーチ*ツールであり、**投資助言ではありません**。
シグナルは無料（遅延あり）のデータソースに基づきます。"""),
        ("🧭 クイックスタート",
         """1. サイドバーでスキャンする**地域**を選び、**クイックスキャン**はONのまま（各地域の先頭10銘柄のみ・高速）。
2. **スキャン実行**を押すと、トップ銘柄・地域別・カテゴリ・テーマ・詳細分析の各タブに結果が反映されます。
3. **詳細分析**タブで銘柄を選ぶと、スコアの根拠（ファクター別の正確な寄与分解と裏付けデータ）を確認できます。
4. **ディープスキャン**タブで通常ユニバース外を探索し、勝者を**プロモート**すると日次スキャンに同乗します。
5. エンジンは自らのピックを記録し、2週間後に成績を判定して**ファクターの重みを自動調整**します（下記参照）。"""),
        ("📊 6つのKPIファクター",
         """各銘柄はファクターごとに0–100点、コンポジットはその加重平均です。
- **モメンタム** — 50日移動平均との乖離、MACDの向き、1ヶ月リターン。
- **バリュー** — 実績PER（割安ほど高得点）＋配当利回りボーナス。
- **テクニカル** — RSIの位置とボリンジャー%B。過熱・売られすぎより健全な中間帯を評価。
- **ハイプ** — 30日ベースライン比の出来高ブレイクアウトに、Reddit・GDELTニュース・
Yahoo!掲示板「みんなの評価」（日本株）・踏み上げ（ショートスクイーズ）補正を加味。
- **クオリティ** — ROE（収益性）。データ欠損は中立扱いで減点しません。
- **テーマ**（新設）— 業種ローテーションの強さ。同じテーマバスケット（AI・半導体・防衛・
クリーンエネルギー等）の*他の*スキャン銘柄の平均モメンタムを減衰付きで反映します。
テーマなしの銘柄は中立の50点です。

**判定:** コンポジット 65以上 → BUY ・ 45未満 → SELL ・ それ以外 → HOLD"""),
        ("🔁 エンジンの学習（ウォークフォワード）",
         """フルスキャンごとに、その日のトップ3グロース／トップ3配当ピックをファクタースコア付きで記録します。
**14日**経過後、各ピックが自国市場ベンチマーク（米国株はS&P 500、日本株は日経平均など）に勝ったかを
判定し、勝者で高得点だったファクターの重みを**引き上げ**、敗者で高得点だったファクターを**引き下げ**
ます（下限付き）。**システム監査**タブで重みの推移と勝率を確認できます。新設のテーマファクターは
12%の小さな重みから始まり、このループで実力に応じて調整されます。"""),
        ("🔬 ディープスキャンとプロモート",
         """市場ごとの2段階ファネルです。**ステージ1**は数百銘柄の株価を一括取得し、価格スクリーン
（モメンタムブレンド・出来高サージ・52週高値近接度・急落キャップ）でランク付け。**ステージ2**は
上位ファイナリストのみファンダメンタルズを含むフル採点を行います。最上位は（米5・日3・中2の枠で）
メインスキャンに**プロモート**でき、再実行のたびに入れ替わります。クラウド環境では再デプロイで
プロモートが消えるため、表示される `PROMOTED_TICKERS` スニペットを `config.py` に貼り付けると永続化できます。"""),
        ("📡 データソースと制限",
         """株価・ファンダメンタルズは **Yahoo Finance**（米国株はStooqフォールバック、任意でFMP APIキー）、
センチメントは **Reddit**・**GDELTニュース**・日本株は **Yahoo!掲示板「みんなの評価」** を利用します。
いずれも無料・非公式のソースで、遅延・制限・ブロックの可能性があります。全取得処理はフェイルセーフで、
ソース停止時はシグナルが中立に劣化するだけでスキャンは継続します。スキャン全体が失敗する場合は
ほぼYahooの一時的なレート制限です。数分待ってクイックスキャンをお試しください。"""),
        ("⚠️ 免責事項",
         """本アプリは教育目的のスクリーニングツールです。スコアは公開データ（遅延あり）の機械的な変換であり、
誤り・陳腐化・入力不足の可能性があります。特定の証券の売買を推奨するものではありません。
投資判断はご自身の調査に基づき、必要に応じて有資格のファイナンシャルアドバイザーにご相談ください。"""),
    ],
}


def render_help() -> None:
    """In-app documentation. First section open by default; the rest collapsed."""
    sections = _HELP_SECTIONS.get(get_lang(), _HELP_SECTIONS["en"])
    for i, (title, body) in enumerate(sections):
        with st.expander(title, expanded=(i == 0)):
            st.markdown(body)


# ----------------------------------------------------------------------------
# Main Application Controller Setup — st.navigation page structure.
#
# The old single-script st.tabs(10) bar had two costs: every widget click
# re-rendered ALL ten tabs (Audit queries, Sell scanner, everything), and the
# bar itself scrolled horizontally, worse in Japanese. st.navigation renders
# only the selected page per rerun and groups the app into four sections in the
# sidebar. Page url_paths are language-independent, so switching 言語 keeps you
# on the same page; _PAGES lets other modules jump programmatically
# (st.switch_page) — e.g. the Regional table's "Open full analysis" button.
# Requires Streamlit >= 1.36 (the app already uses >= 1.35 APIs elsewhere).
# ----------------------------------------------------------------------------
_PAGES: dict[str, "st.Page"] = {}


def _session_results() -> list[dict]:
    return st.session_state.get("results", [])


# Page wrappers: st.Page callables take no arguments, so each page pulls the
# shared scan results (and the audit toggle) from session state itself.
def page_top() -> None: render_daily_top_3(_session_results())
def page_regional() -> None: render_global_sectors(_session_results())
def page_category() -> None: render_category_views(_session_results())
def page_themes() -> None: render_themes(_session_results())
def page_deep_dive() -> None: render_deep_dive(_session_results())
def page_deep_scan() -> None: render_deep_scan()
def page_us() -> None: render_us_conviction(_session_results())
def page_audit() -> None: render_engine_audit(st.session_state.get("_update_prices", True))
def page_sell() -> None: render_sell_signals()
def page_help() -> None: render_help()


def _build_navigation() -> "st.navigation":
    """Grouped page tree. Titles are translated per rerun (tr follows the language
    menu, which is read before this runs); url_paths stay fixed so bookmarks and
    the current selection survive a language switch."""
    global _PAGES
    _PAGES = {
        "top": st.Page(page_top, title=tr("tab_top"), url_path="top", default=True),
        "regional": st.Page(page_regional, title=tr("tab_regional"), url_path="regional"),
        "category": st.Page(page_category, title=tr("tab_category"), url_path="category"),
        "themes": st.Page(page_themes, title=tr("tab_themes"), url_path="themes"),
        "deep_dive": st.Page(page_deep_dive, title=tr("tab_deep"), url_path="deep-dive"),
        "deep_scan": st.Page(page_deep_scan, title=tr("tab_deep_scan"), url_path="deep-scan"),
        "us": st.Page(page_us, title=tr("tab_us"), url_path="us-conviction"),
        "audit": st.Page(page_audit, title=tr("tab_audit"), url_path="audit"),
        "sell": st.Page(page_sell, title=tr("tab_sell"), url_path="sell"),
        "help": st.Page(page_help, title=tr("tab_help"), url_path="help"),
    }
    return st.navigation({
        tr("nav_scan"): [_PAGES["top"], _PAGES["regional"], _PAGES["category"], _PAGES["themes"]],
        tr("nav_research"): [_PAGES["deep_dive"], _PAGES["deep_scan"], _PAGES["us"]],
        tr("nav_audit"): [_PAGES["audit"], _PAGES["sell"]],
        tr("nav_info"): [_PAGES["help"]],
    })


def main() -> None:
    init_db()
    inject_css()

    # Resolve matured recommendation outcomes for the audit accuracy panel (no weight
    # change), then run the single learning loop — the walk-forward optimiser — which
    # re-tunes kpi_weights from matured mock-portfolio picks.
    # Guarded to ONCE PER SESSION: Streamlit reruns main() on every widget
    # interaction, and both loops hit the network when matured picks exist — a
    # persistently-failing ticker (e.g. delisted) would otherwise re-block the UI
    # for several seconds on every single click.
    if not st.session_state.get("_maintenance_done"):
        try: evaluate_outcomes_only()
        except Exception: pass
        try: walk_forward_update()
        except Exception: pass
        st.session_state["_maintenance_done"] = True

    # Sidebar Interface Controller Layout
    # Language menu first: tr() must reflect the chosen language BEFORE the page
    # tree is built, so section headers and page titles localize on this rerun.
    # (st.navigation pins the nav to the top of the sidebar; these controls
    # render below it regardless of call order.)
    lang_choice = st.sidebar.selectbox("🌐 Language / 言語", list(LANGUAGES.keys()), key="lang_select")
    st.session_state["lang"] = LANGUAGES[lang_choice]

    pg = _build_navigation()

    st.sidebar.title(tr("console_title"))
    region_keys = list(TICKER_UNIVERSE.keys())
    selected_regions = st.sidebar.multiselect(tr("regions_to_scan"), region_keys,
                                              default=region_keys, format_func=region_name)
    _n_promoted = sum(len(v) for v in get_promoted_universe().values())
    if _n_promoted:
        st.sidebar.caption(tr("promoted_sidebar", n=_n_promoted))
    quick_scan = st.sidebar.toggle(tr("scan_quick_toggle"), value=True, help=tr("scan_quick_help"))
    jp_forum_on = st.sidebar.toggle(tr("jp_forum_toggle"), value=True, help=tr("jp_forum_help"))
    update_prices = st.sidebar.checkbox(tr("update_hist_prices"), value=True)
    # The Audit page reads this from session state (page callables take no args).
    st.session_state["_update_prices"] = bool(update_prices)
    source_keys = list(SENTIMENT_SOURCES.keys())
    selected_sources = st.sidebar.multiselect(
        tr("sentiment_sources"), source_keys, default=source_keys,
        format_func=lambda k: tr(SENTIMENT_SOURCES[k]["label_key"]),
    )
    if st.sidebar.button(tr("scan_btn"), width="stretch"):
        if not selected_regions:
            st.sidebar.warning(tr("no_region_selected"))
        else:
            with st.spinner(tr("scan_spinner")):
                limit = 10 if quick_scan else None
                res, fail = run_engine(limit_per_region=limit, regions=selected_regions,
                                       sources=selected_sources, use_jp_forum=jp_forum_on)
                st.session_state["results"], st.session_state["failed"] = res, fail
                st.session_state["scan_is_quick"] = bool(quick_scan)
            if res:
                st.success(tr("scan_success"))
            elif fail:
                # Nothing came back and tickers failed → almost always an IP-level
                # rate limit from Yahoo, which is common on Streamlit Cloud.
                st.error(tr("all_failed"))

    if st.sidebar.button(tr("seed_btn"), width="stretch"):
        seed_demo_history()
        st.sidebar.success(tr("seed_success"))
        st.rerun()

    # Escape hatch for the 30-min data caches: without this, a sudden intraday
    # move is invisible until the TTL lapses because every scan is served from
    # cached frames. Clears price/fundamentals caches only (sentiment keeps its
    # own TTL); the user then re-runs the scan to pull fresh data.
    if st.sidebar.button(tr("refresh_btn"), width="stretch", help=tr("refresh_help")):
        clear_market_data_caches()
        st.sidebar.success(tr("refresh_success"))

    # Read existing session metrics safely. A brand-new session (refresh, new
    # browser tab, cloud session timeout) restores the last persisted scan
    # snapshot so returning users land on their results, not the empty state.
    if "results" not in st.session_state:
        snap = load_latest_scan_snapshot()
        if snap:
            res, created_at, was_quick = snap
            st.session_state["results"] = res
            st.session_state["failed"] = []
            st.session_state["scan_is_quick"] = was_quick
            st.session_state["restored_scan"] = True
            try:
                st.session_state["scan_ts"] = pd.Timestamp(created_at).timestamp()
            except Exception:
                st.session_state["scan_ts"] = None
    results = st.session_state.get("results", [])
    failed = st.session_state.get("failed", [])

    if failed:
        st.sidebar.warning(tr("skipped", items=", ".join(failed)))

    # Freshness chip: when, how many, quick or full — visible on every page.
    if results:
        scan_ts = st.session_state.get("scan_ts")
        ago_txt = _ago(scan_ts) if scan_ts else "—"
        mode = tr("scan_mode_quick") if st.session_state.get("scan_is_quick") else tr("scan_mode_full")
        st.markdown(
            f'<span class="qc-chip">🕒 {tr("scan_chip", ago=ago_txt, n=len(results), mode=mode)}</span>',
            unsafe_allow_html=True)
        if st.session_state.get("restored_scan"):
            st.caption(tr("scan_restored_note"))

    # --- First-run empty state ---------------------------------------------
    # A new user — especially on mobile, where the sidebar starts collapsed —
    # would otherwise land on "please run a scan" pages with the actual Run
    # button hidden. Give them a hero card with a primary CTA right on the main
    # page; it disappears once results exist, and is skipped on the Help page
    # (reading the docs shouldn't compete with a call to action).
    if not results and pg.url_path != "help":
        st.markdown(
            f'<div class="qc-card" style="padding:26px 24px;">'
            f'<div class="qc-ticker">{tr("hero_title")}</div>'
            f'<div class="qc-sub" style="margin-top:6px; max-width:60ch;">{tr("hero_sub")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(tr("hero_btn"), type="primary", key="hero_scan_btn"):
            with st.spinner(tr("scan_spinner")):
                res, fail = run_engine(limit_per_region=10,
                                       regions=list(TICKER_UNIVERSE.keys()),
                                       sources=None, use_jp_forum=True)
                st.session_state["results"], st.session_state["failed"] = res, fail
                st.session_state["scan_is_quick"] = True
            if not res and fail:
                st.error(tr("all_failed"))
            else:
                st.rerun()
        st.caption(tr("hero_hint"))

    # Global sentiment telemetry: surface the buzz sources + freshness once a scan has
    # run, so it's visible app-wide rather than only inside the Deep-Dive page.
    if _LAST_HYPE_FETCHED_AT:
        _msg = tr("hype_updated", ago=_ago(_LAST_HYPE_FETCHED_AT))
        st.caption(f"{_LAST_HYPE_STATUS} · {_msg}" if _LAST_HYPE_STATUS else _msg)

    # Render ONLY the selected page (lazy — the whole point of st.navigation).
    pg.run()

if __name__ == "__main__":
    main()