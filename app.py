from __future__ import annotations

import base64
import json
import math
import os
import re
import sqlite3
import ssl
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

def fetch_reddit_hype(tickers: list) -> dict:
    """Count r/wallstreetbets mentions per ticker, preferring authenticated access.

    Three-tier, fully fail-safe:
      1. Authenticated OAuth (set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in
         Streamlit secrets or env vars) — by far the most reliable from cloud /
         datacenter IPs, and pulls up to 100 hot posts (title + selftext).
      2. Public RSS feed via urllib + browser UA — works locally, often blocked
         on cloud IPs.
      3. Zero mentions for every ticker — guarantees the scan never crashes.
    No extra pip dependencies: base64/json/urllib/ssl are all standard library.
    """
    global _LAST_REDDIT_SOURCE
    counts = {t: 0 for t in tickers}

    # --- Tier 1: authenticated OAuth -------------------------------------------
    try:
        token = _reddit_oauth_token()
        if token:
            ua = _get_secret("REDDIT_USER_AGENT", "stockrec-hype/1.0")
            raw = _reddit_get("https://oauth.reddit.com/r/wallstreetbets/hot?limit=100",
                              {"Authorization": f"Bearer {token}", "User-Agent": ua})
            payload = json.loads(raw)
            texts = []
            for child in payload.get("data", {}).get("children", []):
                d = child.get("data", {})
                texts.append(str(d.get("title", "")))
                texts.append(str(d.get("selftext", "")))
            _LAST_REDDIT_SOURCE = "authenticated"
            return _count_ticker_mentions(" ".join(texts).upper(), tickers)
    except Exception:
        pass   # fall through to the unauthenticated RSS path

    # --- Tier 2: unauthenticated public RSS ------------------------------------
    try:
        raw = _reddit_get("https://www.reddit.com/r/wallstreetbets/hot.rss",
                          {"User-Agent": "Mozilla/5.0"})
        blocks = re.findall(r"<title[^>]*>(.*?)</title>", raw, re.DOTALL | re.IGNORECASE)
        blocks += re.findall(r"<content[^>]*>(.*?)</content>", raw, re.DOTALL | re.IGNORECASE)
        blob = (" ".join(blocks) if blocks else raw).upper()
        _LAST_REDDIT_SOURCE = "rss"
        return _count_ticker_mentions(blob, tickers)
    except Exception:
        # --- Tier 3: safe zero-mention fallback --------------------------------
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
    """Reddit r/wallstreetbets mention counts (English/US retail buzz)."""
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

def fetch_hype_signals(universe: dict, enabled_sources: list | None = None) -> dict:
    """Run every enabled sentiment source over the markets it covers and merge the
    per-ticker mention counts. Returns {ticker: combined_count}; analyze_ticker turns
    that into the capped hype bonus. Each source is independently fail-safe."""
    global _LAST_HYPE_STATUS
    enabled = enabled_sources if enabled_sources is not None else DEFAULT_SOURCES
    combined = {t: 0 for ticks in universe.values() for t in ticks}
    status_parts = []
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
            combined[t] = combined.get(t, 0) + int(c or 0)
        if key == "reddit":
            status_parts.append(f"{tr('src_reddit')}: {tr('hype_status_' + _LAST_REDDIT_SOURCE)}")
        elif key == "gdelt":
            status_parts.append(f"{tr('src_gdelt')}: {tr('hype_status_' + _LAST_GDELT_STATUS)}")
    _LAST_HYPE_STATUS = (tr("hype_sources_prefix") + " " + " · ".join(status_parts)) if status_parts else ""
    return combined




# ----------------------------------------------------------------------------
# Configuration & Global Market Universes
# ----------------------------------------------------------------------------
DB_PATH = "stock_engine.db"
FACTORS = ["momentum", "value", "technical", "hype", "quality"]

DEFAULT_WEIGHTS = {
    "momentum": 0.20,
    "value": 0.20,
    "technical": 0.20,
    "hype": 0.20,
    "quality": 0.20,
}

LEARNING_RATE = 0.04
MIN_WEIGHT = 0.05
EVAL_HORIZON_DAYS = 14
BUY_THRESHOLD = 65.0
SELL_THRESHOLD = 45.0

# Expanded global universe. Trim any list to speed up scans (each symbol adds
# a price fetch + a fundamentals call, so ~60 tickers takes a couple of minutes).
TICKER_UNIVERSE = {
    "USA": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "META",
            "JPM", "V", "WMT", "JNJ", "PG", "XOM", "KO", "DIS",
            "NFLX", "AMD", "BAC", "PFE", "CSCO", "MRK", "HD"],
    "Japan": ["7203.T", "6758.T", "9984.T", "6501.T", "7751.T",
              "8306.T", "9432.T", "6902.T", "4063.T", "8035.T",
              "7267.T", "6098.T", "9433.T", "6954.T", "8058.T"],
    "Europe": ["ASML", "MC.PA", "VOW3.DE", "SAP", "OR.PA",
               "SIE.DE", "AIR.PA", "ALV.DE", "BMW.DE", "BAS.DE",
               "SU.PA", "DTE.DE", "AI.PA", "RMS.PA"],
    "China": ["0700.HK", "9988.HK", "BABA", "JD", "BIDU",
              "1810.HK", "3690.HK", "PDD", "NIO", "2318.HK",
              "0939.HK", "1299.HK"]
}
ALL_TICKERS = [ticker for region in TICKER_UNIVERSE.values() for ticker in region]

# Plain-English company names, used by news-based sentiment sources (a ticker like
# "7203.T" never appears in an article, but "Toyota" does). Edit here to tune what
# the news scan searches for. Falls back to the ticker itself if a name is missing.
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "TSLA": "Tesla", "META": "Meta Platforms", "JPM": "JPMorgan",
    "V": "Visa", "WMT": "Walmart", "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble",
    "XOM": "Exxon Mobil", "KO": "Coca-Cola", "DIS": "Disney", "NFLX": "Netflix",
    "AMD": "AMD", "BAC": "Bank of America", "PFE": "Pfizer", "CSCO": "Cisco",
    "MRK": "Merck", "HD": "Home Depot",
    "7203.T": "Toyota", "6758.T": "Sony", "9984.T": "SoftBank", "6501.T": "Hitachi",
    "7751.T": "Canon", "8306.T": "Mitsubishi UFJ", "9432.T": "NTT", "6902.T": "Denso",
    "4063.T": "Shin-Etsu Chemical", "8035.T": "Tokyo Electron", "7267.T": "Honda",
    "6098.T": "Recruit Holdings", "9433.T": "KDDI", "6954.T": "Fanuc", "8058.T": "Mitsubishi Corp",
    "ASML": "ASML", "MC.PA": "LVMH", "VOW3.DE": "Volkswagen", "SAP": "SAP",
    "OR.PA": "L'Oreal", "SIE.DE": "Siemens", "AIR.PA": "Airbus", "ALV.DE": "Allianz",
    "BMW.DE": "BMW", "BAS.DE": "BASF", "SU.PA": "Schneider Electric",
    "DTE.DE": "Deutsche Telekom", "AI.PA": "Air Liquide", "RMS.PA": "Hermes",
    "0700.HK": "Tencent", "9988.HK": "Alibaba", "BABA": "Alibaba", "JD": "JD.com",
    "BIDU": "Baidu", "1810.HK": "Xiaomi", "3690.HK": "Meituan", "PDD": "Pinduoduo",
    "NIO": "NIO", "2318.HK": "Ping An", "0939.HK": "China Construction Bank", "1299.HK": "AIA",
}

# Benchmark index per home market for the walk-forward evaluator. A pick is judged
# against the index of the exchange it actually trades on (keyed by ticker suffix),
# so a Tokyo-listed name is measured against the Nikkei rather than the S&P 500.
# Suffix-based mapping also handles dual listings correctly: a US-listed ADR with
# no suffix (e.g. BABA, ASML) is benchmarked against SPY, matching its currency and
# trading calendar. To change a mapping, edit this one dict. Anything not listed
# falls back to DEFAULT_BENCHMARK.
#
# KNOWN ASYMMETRY (accepted): stock returns are dividend-inclusive (auto_adjust=True),
# but most of these benchmarks are PRICE indices (^N225/^HSI/^FCHI/000300.SS/etc.),
# while SPY is a dividend-paying ETF (total-return when adjusted) and ^GDAXI is itself
# a total-return index. So a dividend-paying non-US/non-DE stock is judged slightly
# generously vs a price-only index. Over the 14-day EVAL_HORIZON_DAYS window the
# accrued dividend is ~0.1-0.2%, so the bias is marginal; the correct symmetric fix
# would need total-return indices in each LOCAL currency, which aren't freely
# available for JP/CN (USD ETFs like EWJ would trade this small bias for a worse
# currency mismatch), so we accept it rather than introduce that error.
BENCHMARKS = {
    "":   "SPY",        # US / NYSE / NASDAQ (no suffix)  -> S&P 500 ETF
    "T":  "^N225",      # Tokyo                           -> Nikkei 225
    "HK": "^HSI",       # Hong Kong                       -> Hang Seng
    "PA": "^FCHI",      # Paris                           -> CAC 40
    "DE": "^GDAXI",     # Frankfurt / Xetra               -> DAX
    "SS": "000300.SS",  # Shanghai (A-shares)             -> CSI 300
    "SZ": "000300.SS",  # Shenzhen (A-shares)             -> CSI 300
    "SI": "^STI",       # Singapore                       -> Straits Times
    "JK": "^JKSE",      # Jakarta                         -> IDX Composite
    "L":  "^FTSE",      # London                          -> FTSE 100
}
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

TRANSLATIONS = {
    "en": {
        # Sidebar
        "console_title": "🎛️ Engine Console",
        "scan_btn": "⚙️ Execute Market Analysis Scan",
        "scan_spinner": "Compiling cross-market indices...",
        "scan_success": "Global tracking array sync completed.",
        "seed_btn": "🧪 Seed Simulated Audit Dataset",
        "seed_success": "Synthetic data arrays appended to tracking engine database.",
        "skipped": "Skipped tickers: {items}",
        "scan_quick_toggle": "⚡ Quick scan (recommended on Cloud)",
        "scan_quick_help": "Scans only the first few tickers per region — faster and far less likely to be rate-limited by Yahoo Finance.",
        "all_failed": "Every fetch failed. Yahoo Finance is most likely rate-limiting this server's IP — this is common on Streamlit Cloud. Wait a few minutes and retry, or run the app locally.",
        "persistence_note": "ℹ️ On Streamlit Community Cloud, saved history is not guaranteed to persist — it resets whenever the app reboots or sleeps (after 12h idle). Use the seed button to repopulate the demo, or wire up an external database for permanent storage.",
        "benchmark_note": "📈 Accuracy is benchmark-relative: each pick counts as a win only if it beat its home market over the holding window (US→S&P 500, Japan→Nikkei 225, Hong Kong→Hang Seng, France→CAC 40, Germany→DAX).",
        # Tabs
        "tab_top": "🔥 Top Selections",
        "tab_regional": "📊 Regional Desks",
        "tab_category": "⚡ Category Screening",
        "tab_deep": "🔍 Chart Deep Dive",
        "tab_audit": "🧠 Systems Audit",
        # Engine progress
        "scanning": "Scanning Global Markets...",
        "processing": "Processing {ticker} ({region})...",
        # Top 3
        "top3_header": "🔥 Top 3 System Picks Across All Global Regions",
        "need_scan_sidebar": "Please process market metrics in the sidebar analyzer.",
        "score_suffix": "Score",
        "price_label": "Price",
        "company_profile": "Company Profile",
        "trend_return_1m": "Trend Return (1M)",
        "pe_ratio": "P/E Ratio",
        "dividend_yield": "Dividend Yield",
        # Category views
        "need_run_engine": "Run the calculations engine in the sidebar.",
        "subtab_growth": "🚀 Top Weekly Growth Selections",
        "subtab_dividend": "💰 High-Yield Dividend Picks",
        "growth_header_text": "Highest Weekly Capital Expansion Momentum",
        "dividend_header_text": "Balanced Top Recommendations with Dividends > 1.5%",
        "no_dividend_match": "No active candidates matched dividend targets in this cycle.",
        "col_ticker": "Ticker",
        "col_company": "Company",
        "col_region": "Region",
        "col_price": "Price",
        "col_momentum_1m": "1M Momentum %",
        "col_overall_score": "Overall Score",
        "col_dividend_return": "Dividend Return %",
        # Regional desks
        "need_activate": "Activate engine matrix to sort international regional distributions.",
        "select_region": "Select Geographical Workspace Focus:",
        "card_system_rating": "System Rating",
        "card_trading_close": "Trading Close",
        "card_sustained_hype": "Sustained Hype",
        "breakout": "Breakout",
        "flat": "Flat",
        # Deep dive
        "need_run_history": "Run the engine to inspect price history.",
        "select_profile": "Select Target Analytics Profile:",
        "history_workspace": "Historical Data Plot Workspace: {ticker}",
        "factor_breakdown": "Factor Breakdown",
        "factor_momentum": "Momentum",
        "factor_value": "Value",
        "factor_technical": "Technical",
        "factor_hype": "Hype",
        "factor_quality": "Quality",
        "wsb_mentions": "WallStreetBets Mentions",
        "wsb_caption": "Live mention count for this ticker in r/wallstreetbets hot posts during the last scan (feeds the Hype factor).",
        "wsb_source_authenticated": "🟢 Source: authenticated Reddit API (most reliable).",
        "wsb_source_rss": "🟡 Source: public RSS feed — may be blocked or throttled on cloud hosts.",
        "wsb_source_offline": "⚪ Source: offline — Reddit was unreachable, so mentions counted as 0.",
        "historical_performance": "Historical Performance (Mock Portfolio)",
        "historical_perf_note": "Return % is the live price vs. the price when the pick was logged. Green = ahead, red = behind. These curated picks also feed the walk-forward weight optimizer.",
        "no_portfolio_picks": "No saved picks yet — run a scan to start logging the mock portfolio.",
        "col_reason": "Reason",
        "col_rec_price": "Rec. Price",
        "col_current_price": "Current Price",
        "col_return_pct": "Return %",
        "reason_growth": "Top Growth",
        "reason_dividend": "Top Dividend",
        "regions_to_scan": "Regions to Scan",
        "sentiment_sources": "Sentiment Sources",
        "src_reddit": "Reddit (r/wallstreetbets)",
        "src_gdelt": "Global News (GDELT)",
        "hype_buzz_label": "Social & News Buzz (mentions)",
        "hype_buzz_caption": "Combined mentions of this stock across the enabled sentiment sources during the last scan (feeds the Hype factor).",
        "hype_sources_prefix": "Buzz sources —",
        "hype_status_authenticated": "Reddit API ✓",
        "hype_status_rss": "RSS feed",
        "hype_status_ok": "live ✓",
        "hype_status_offline": "offline (0)",
        "update_hist_prices": "Update Historical Portfolio Prices",
        "no_region_selected": "Select at least one region to scan.",
        "prices_skipped": "Live price update is off — current prices and returns are left blank. Tick 'Update Historical Portfolio Prices' to refresh.",
        "sell_privacy_note": "Your portfolio list stays in this session only — it is never saved to the database.",
        "sell_paste_label": "Paste tickers (comma, space, or newline separated)",
        "sell_upload_label": "…or upload a CSV with a 'Ticker' or 'Symbol' column",
        "sell_csv_nocol": "No 'Ticker' or 'Symbol' column found in that CSV.",
        "sell_csv_error": "Couldn't read that CSV file.",
        "sell_loaded": "{n} ticker(s) loaded: {tickers}",
        "sell_scan_btn": "Scan Portfolio",
        "sell_bulk_fetch": "Bulk-fetching price history for {total} tickers in one request…",
        "sell_scan_progress": "Scanning {done}/{total}…",
        "sell_need_portfolio": "Paste tickers or upload a CSV above, then press Scan Portfolio.",
        "sell_verdict_na": "NO DATA",
        "col_verdict": "Verdict",
        "sell_detail_expander": "🔎 View full breakdown for one ticker",
        "sell_detail_select": "Select a ticker from your scan",
        # Sell-Signal scanner
        "tab_sell": "🔻 Sell Signals",
        "sell_header": "🔻 Sell-Signal Scanner",
        "sell_disclaimer": "Informational signals only — not financial advice. Data via Yahoo Finance; analyst and insider coverage is richest for US-listed symbols.",
        "sell_input_label": "Enter a ticker symbol",
        "sell_btn": "Analyze",
        "sell_spinner": "Scanning sell signals for {ticker}...",
        "sell_need_input": "Type a ticker symbol above (e.g. AAPL, MSFT, ABBV) to scan for sell signals.",
        "sell_no_data": "Couldn't retrieve price data for '{ticker}'. Check the symbol — for the richest analyst and insider data, try the US listing (e.g. ABBV instead of 4AB, MSFT instead of MSF).",
        "sell_pressure": "Sell Pressure",
        "sell_verdict_high": "ELEVATED",
        "sell_verdict_mixed": "MIXED",
        "sell_verdict_low": "LOW",
        "sell_breakdown_title": "Signal Breakdown (higher = more bearish)",
        "sell_recent_downgrades": "Recent Analyst Rating Changes",
        "sell_insider_activity": "Recent Insider Transactions",
        "sell_no_analyst": "No recent analyst rating changes available for this symbol.",
        "sell_no_insider": "No insider transactions reported for this symbol.",
        "kpi_analyst": "Analyst Consensus",
        "kpi_downgrades": "Rating Changes",
        "kpi_target": "Price-Target Downside",
        "kpi_insider": "Insider Selling",
        "kpi_technical": "Technical Breakdown",
        "kpi_momentum": "Momentum Trend",
        "kpi_short": "Short Interest",
        "col_kpi": "Indicator",
        "col_signal": "Signal (0-100)",
        "col_reading": "Reading",
        "col_date": "Date",
        "col_firm": "Firm",
        "col_from": "From",
        "col_to": "To",
        "col_action": "Action",
        "col_insider": "Insider",
        "col_transaction": "Transaction",
        "col_shares": "Shares",
        "col_value": "Value",
        # Audit
        "audit_header_text": "🧠 Machine Weight Analytics & Historical Integrity",
        "no_tracks": "No calculation tracks logged yet.",
        "card_archived_calls": "Archived Calls",
        "card_perf_verified": "Performance Verified",
        "card_accuracy": "Historical Accuracy Score",
        "winrate_over_time": "Win rate over time",
        "winrate_col": "Win Rate %",
        "kpi_weight_evolution": "KPI weight evolution",
        # Recommendation pills
        "rec_BUY": "BUY",
        "rec_HOLD": "HOLD",
        "rec_SELL": "SELL",
    },
    "ja": {
        # サイドバー
        "console_title": "🎛️ エンジンコンソール",
        "scan_btn": "⚙️ 市場分析スキャンを実行",
        "scan_spinner": "市場横断インデックスを集計中...",
        "scan_success": "グローバル追跡データの同期が完了しました。",
        "seed_btn": "🧪 監査用サンプルデータを生成",
        "seed_success": "サンプルデータをトラッキングDBに追加しました。",
        "skipped": "スキップした銘柄: {items}",
        "scan_quick_toggle": "⚡ クイックスキャン（クラウド推奨）",
        "scan_quick_help": "各地域の先頭数銘柄のみをスキャンします。高速で、Yahoo Finance のレート制限にかかりにくくなります。",
        "all_failed": "すべての取得に失敗しました。Yahoo Finance がこのサーバーのIPをレート制限している可能性が高いです（Streamlit Cloud ではよくあります）。数分待って再試行するか、ローカルで実行してください。",
        "persistence_note": "ℹ️ Streamlit Community Cloud では、保存された履歴は永続化が保証されません。アプリの再起動やスリープ（12時間無操作）のたびにリセットされます。シードボタンでデモを再生成するか、外部データベースを接続して永続保存してください。",
        "benchmark_note": "📈 的中率はベンチマーク相対です。各推奨は保有期間中に自国市場を上回った場合のみ「勝ち」と判定されます（米国→S&P500、日本→日経225、香港→ハンセン、フランス→CAC40、ドイツ→DAX）。",
        # タブ
        "tab_top": "🔥 トップ銘柄",
        "tab_regional": "📊 地域別デスク",
        "tab_category": "⚡ カテゴリ別スクリーニング",
        "tab_deep": "🔍 チャート詳細分析",
        "tab_audit": "🧠 システム監査",
        # 進捗
        "scanning": "グローバル市場をスキャン中...",
        "processing": "{ticker}（{region}）を処理中...",
        # トップ3
        "top3_header": "🔥 全地域から選んだトップ3銘柄",
        "need_scan_sidebar": "サイドバーのアナライザーで市場データを処理してください。",
        "score_suffix": "スコア",
        "price_label": "価格",
        "company_profile": "企業プロフィール",
        "trend_return_1m": "トレンドリターン（1ヶ月）",
        "pe_ratio": "PER（株価収益率）",
        "dividend_yield": "配当利回り",
        # カテゴリ別
        "need_run_engine": "サイドバーで計算エンジンを実行してください。",
        "subtab_growth": "🚀 週間グロース上位銘柄",
        "subtab_dividend": "💰 高配当銘柄",
        "growth_header_text": "週間モメンタム上位銘柄",
        "dividend_header_text": "配当利回り1.5%超のおすすめ銘柄",
        "no_dividend_match": "今回のサイクルでは配当条件に合う銘柄がありませんでした。",
        "col_ticker": "銘柄",
        "col_company": "会社名",
        "col_region": "地域",
        "col_price": "価格",
        "col_momentum_1m": "1ヶ月モメンタム %",
        "col_overall_score": "総合スコア",
        "col_dividend_return": "配当利回り %",
        # 地域別デスク
        "need_activate": "エンジンを実行して地域別の分布を表示してください。",
        "select_region": "対象地域を選択:",
        "card_system_rating": "システム評価",
        "card_trading_close": "終値",
        "card_sustained_hype": "継続的な注目度",
        "breakout": "急騰",
        "flat": "横ばい",
        # 詳細分析
        "need_run_history": "エンジンを実行すると価格履歴を確認できます。",
        "select_profile": "分析する銘柄を選択:",
        "history_workspace": "価格履歴チャート: {ticker}",
        "factor_breakdown": "ファクター内訳",
        "factor_momentum": "モメンタム",
        "factor_value": "バリュー",
        "factor_technical": "テクニカル",
        "factor_hype": "ハイプ",
        "factor_quality": "クオリティ",
        "wsb_mentions": "WallStreetBets 言及数",
        "wsb_caption": "直近のスキャン時に r/wallstreetbets の人気投稿でこの銘柄が言及された回数（ハイプ・ファクターに反映されます）。",
        "wsb_source_authenticated": "🟢 ソース：認証済み Reddit API（最も安定）。",
        "wsb_source_rss": "🟡 ソース：公開RSSフィード（クラウド環境ではブロック・制限される場合があります）。",
        "wsb_source_offline": "⚪ ソース：オフライン（Redditに接続できず、言及数は0としてカウント）。",
        "historical_performance": "ヒストリカル・パフォーマンス（模擬ポートフォリオ）",
        "historical_perf_note": "リターン％は、記録時の株価に対する現在株価の変化です。緑＝プラス、赤＝マイナス。これらの厳選銘柄はウォークフォワード重み最適化にも反映されます。",
        "no_portfolio_picks": "保存された推奨はまだありません。スキャンを実行すると模擬ポートフォリオの記録が始まります。",
        "col_reason": "理由",
        "col_rec_price": "推奨時株価",
        "col_current_price": "現在株価",
        "col_return_pct": "リターン %",
        "reason_growth": "成長株トップ",
        "reason_dividend": "高配当トップ",
        "regions_to_scan": "スキャンする地域",
        "sentiment_sources": "センチメント・ソース",
        "src_reddit": "Reddit（r/wallstreetbets）",
        "src_gdelt": "グローバルニュース（GDELT）",
        "hype_buzz_label": "ソーシャル・ニュース言及数",
        "hype_buzz_caption": "直近のスキャン時に、有効化したセンチメント・ソース全体でこの銘柄が言及された合計回数（ハイプ・ファクターに反映されます）。",
        "hype_sources_prefix": "バズの取得元 —",
        "hype_status_authenticated": "Reddit API ✓",
        "hype_status_rss": "RSSフィード",
        "hype_status_ok": "ライブ ✓",
        "hype_status_offline": "オフライン (0)",
        "update_hist_prices": "履歴ポートフォリオの株価を更新",
        "no_region_selected": "スキャンする地域を1つ以上選択してください。",
        "prices_skipped": "ライブ株価の更新はオフです。現在株価とリターンは空欄です。「履歴ポートフォリオの株価を更新」をオンにすると更新されます。",
        "sell_privacy_note": "ポートフォリオのリストはこのセッション内にのみ保持され、データベースには保存されません。",
        "sell_paste_label": "ティッカーを貼り付け（カンマ・スペース・改行区切り）",
        "sell_upload_label": "…または「Ticker」「Symbol」列を含むCSVをアップロード",
        "sell_csv_nocol": "そのCSVに「Ticker」または「Symbol」列が見つかりませんでした。",
        "sell_csv_error": "そのCSVファイルを読み込めませんでした。",
        "sell_loaded": "{n} 銘柄を読み込みました：{tickers}",
        "sell_scan_btn": "ポートフォリオをスキャン",
        "sell_bulk_fetch": "{total} 銘柄の価格履歴を一括取得中…",
        "sell_scan_progress": "スキャン中 {done}/{total}…",
        "sell_need_portfolio": "上にティッカーを貼り付けるかCSVをアップロードし、「ポートフォリオをスキャン」を押してください。",
        "sell_verdict_na": "データなし",
        "col_verdict": "判定",
        "sell_detail_expander": "🔎 個別銘柄の詳細を表示",
        "sell_detail_select": "スキャンした銘柄から選択",
        # 売りシグナル・スキャナー
        "tab_sell": "🔻 売りシグナル",
        "sell_header": "🔻 売りシグナル・スキャナー",
        "sell_disclaimer": "本機能は情報提供のみを目的とし、投資助言ではありません。データはYahoo Finance提供。アナリスト・インサイダー情報は米国上場銘柄が最も充実しています。",
        "sell_input_label": "ティッカーシンボルを入力",
        "sell_btn": "分析",
        "sell_spinner": "{ticker} の売りシグナルを分析中...",
        "sell_need_input": "上の欄にティッカー（例：AAPL、MSFT、ABBV）を入力すると売りシグナルを分析します。",
        "sell_no_data": "'{ticker}' の価格データを取得できませんでした。シンボルをご確認ください。アナリスト・インサイダー情報を充実させるには米国上場銘柄（例：4AB→ABBV、MSF→MSFT）をお試しください。",
        "sell_pressure": "売り圧力",
        "sell_verdict_high": "強い",
        "sell_verdict_mixed": "中程度",
        "sell_verdict_low": "弱い",
        "sell_breakdown_title": "シグナル内訳（高いほど弱気）",
        "sell_recent_downgrades": "最近のアナリスト格付け変更",
        "sell_insider_activity": "最近のインサイダー取引",
        "sell_no_analyst": "この銘柄の最近のアナリスト格付け変更はありません。",
        "sell_no_insider": "この銘柄のインサイダー取引は報告されていません。",
        "kpi_analyst": "アナリスト・コンセンサス",
        "kpi_downgrades": "格付け変更",
        "kpi_target": "目標株価との乖離",
        "kpi_insider": "インサイダー売却",
        "kpi_technical": "テクニカルの崩れ",
        "kpi_momentum": "モメンタム動向",
        "kpi_short": "空売り比率",
        "col_kpi": "指標",
        "col_signal": "シグナル (0-100)",
        "col_reading": "内容",
        "col_date": "日付",
        "col_firm": "証券会社",
        "col_from": "変更前",
        "col_to": "変更後",
        "col_action": "アクション",
        "col_insider": "インサイダー",
        "col_transaction": "取引",
        "col_shares": "株数",
        "col_value": "金額",
        # 監査
        "audit_header_text": "🧠 重み学習の分析と過去実績",
        "no_tracks": "まだ記録された計算履歴はありません。",
        "card_archived_calls": "記録された判定",
        "card_perf_verified": "結果検証済み",
        "card_accuracy": "過去の的中率",
        "winrate_over_time": "勝率の推移",
        "winrate_col": "勝率 %",
        "kpi_weight_evolution": "重み（KPI）の推移",
        # 推奨ピル
        "rec_BUY": "買い",
        "rec_HOLD": "中立",
        "rec_SELL": "売り",
    },
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
    # Collapsed by default: on a phone an expanded sidebar covers the whole
    # screen. Users tap the ☰ menu to open the console.
    initial_sidebar_state="collapsed",
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
# Database layer
# ----------------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
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

def _ensure_column(conn: sqlite3.Connection, table: str, column: str,
                   decl: str = "REAL DEFAULT 0.0") -> None:
    """Add `column` to `table` if it isn't already present (safe schema migration)."""
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL, rec_date TEXT NOT NULL, recommendation TEXT NOT NULL,
                composite REAL NOT NULL, price_at_rec REAL, momentum REAL, value REAL, 
                technical REAL, hype REAL, quality REAL, price_after REAL, outcome INTEGER, eval_date TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kpi_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT, update_date TEXT NOT NULL,
                momentum REAL, value REAL, technical REAL, hype REAL, quality REAL, note TEXT
            )""")
        # Walk-forward optimisation tables (new). CREATE ... IF NOT EXISTS is
        # inherently non-destructive, so existing data is never touched.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mock_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, ticker TEXT NOT NULL,
                recommendation_price REAL, reason TEXT, kpi_snapshot TEXT,
                evaluated INTEGER DEFAULT 0
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, factor_name TEXT NOT NULL, current_weight REAL
            )""")
        # --- Safe migration for pre-existing stock_engine.db files ---
        # Databases created before the "quality" factor existed are missing the
        # column. Add it (defaulting old rows to 0.0) only when absent, so this is
        # a no-op on fresh installs and never errors on an upgrade.
        _ensure_column(conn, "recommendations", "quality")
        _ensure_column(conn, "kpi_weights", "quality")
        # Bookkeeping flag so each logged pick feeds the walk-forward loop once.
        _ensure_column(conn, "mock_portfolio", "evaluated", "INTEGER DEFAULT 0")
        # Company name stored alongside the ticker for a friendlier audit table.
        _ensure_column(conn, "mock_portfolio", "name", "TEXT")
        conn.commit()
    if get_weight_history().empty:
        save_weights(DEFAULT_WEIGHTS, note="initial defaults")

def get_latest_weights() -> dict[str, float]:
    with get_conn() as conn:
        row = conn.execute("SELECT momentum, value, technical, hype, quality FROM kpi_weights ORDER BY id DESC LIMIT 1").fetchone()
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
            "INSERT INTO kpi_weights (update_date, momentum, value, technical, hype, quality, note) VALUES (?,?,?,?,?,?,?)",
            (ts, weights["momentum"], weights["value"], weights["technical"], weights["hype"], weights["quality"], note),
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
        df = pd.read_sql_query("SELECT update_date, momentum, value, technical, hype, quality, note FROM kpi_weights ORDER BY id ASC", conn)
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
            "INSERT INTO recommendations (ticker, rec_date, recommendation, composite, price_at_rec, momentum, value, technical, hype, quality) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rec["ticker"], today, rec["recommendation"], float(rec["composite"]),
             price_val, safe_float(rec["momentum"]), safe_float(rec["value"]),
             safe_float(rec["technical"]), hype_val, quality_val),
        )
        conn.commit()

def get_recommendations() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM recommendations ORDER BY rec_date DESC, id DESC", conn)
    if not df.empty: df["rec_date"] = pd.to_datetime(df["rec_date"], errors="coerce")
    return df

# ----------------------------------------------------------------------------
# Quantitative Math & Indicators
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_history(ticker: str, period: str = "8mo") -> pd.DataFrame:
    """Fetch OHLCV with retry/backoff so a throttled burst doesn't wipe a scan.

    RAISES RuntimeError when nothing comes back after the retries. @st.cache_data
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
    raise RuntimeError(f"no price history for {ticker} after retries") from last_err

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    out = {"pe": float("nan"), "div_yield": float("nan"), "market_cap": float("nan"),
           "roe": float("nan"), "short_pct": float("nan"), "name": ticker}
    if yf is None: return out
    # Distinguish a transient failure (rate-limit / network) from a reachable ticker
    # that simply lacks some fields. On the former we RAISE — @st.cache_data never
    # caches an exception, so the broken state isn't frozen for the whole TTL and the
    # next scan retries. Field-level gaps stay as NaN and cache normally.
    try:
        info = _ticker(ticker).info
    except Exception as e:
        raise RuntimeError(f"fundamentals fetch failed for {ticker} (likely rate-limited)") from e
    if not info:
        raise RuntimeError(f"empty fundamentals for {ticker} (likely rate-limited)")
    out["pe"] = safe_float(info.get("trailingPE"))
    out["market_cap"] = safe_float(info.get("marketCap"))
    out["name"] = str(info.get("shortName") or info.get("longName") or ticker)
    out["div_yield"] = resolve_div_yield(info)
    # New fundamentals for the quality factor and short-squeeze modifier. safe_float
    # keeps these as NaN when yfinance omits them, so downstream scoring never breaks.
    out["roe"] = safe_float(info.get("returnOnEquity"))
    out["short_pct"] = safe_float(info.get("shortPercentOfFloat"))
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

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b

def compute_hype(volume: pd.Series) -> dict:
    result = {"score": 0.0, "breakout_days": 0, "avg_ratio": float("nan"), "sustained": False}
    vol = volume.dropna()
    if len(vol) < 33: return result
    baseline = float(vol.iloc[-33:-3].mean())
    if baseline <= 0: return result
    ratios = vol.iloc[-3:] / baseline
    breakout_days = int((ratios > 1.5).sum())
    avg_ratio = float(ratios.mean())
    raw = (breakout_days / 3) * 60 + min(max(avg_ratio - 1.0, 0.0), 2.0) / 2.0 * 40
    result.update(score=float(min(raw, 100.0)), breakout_days=breakout_days, avg_ratio=avg_ratio, sustained=breakout_days == 3)
    return result

def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float: return max(lo, min(hi, v))

def analyze_ticker(ticker: str, region: str, hype_mentions: int = 0) -> dict | None:
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
    ret_1m = safe_float((price / close.iloc[-21] - 1) * 100) if len(close) > 21 else float("nan")
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
        "hype": hype, "momentum": momentum_score, "value": value_score,
        "technical": technical_score, "quality": quality_score,
        "hype_score": hype["score"], "hype_mentions": hype_mentions, "history": hist
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
        hist = fetch_history(ticker, period="1y")   # 1y so SMA200 is available
        time.sleep(0.2)                              # pace only when we hit the network
    if hist is None or len(hist) < 60:
        return None

    # Pace the remaining (un-bulkable) per-ticker endpoints to avoid IP-level rate limits.
    funds = fetch_fundamentals(ticker)
    time.sleep(0.2)
    analyst = fetch_analyst_signals(ticker)
    time.sleep(0.2)
    insider = fetch_insider_activity(ticker)

    close = hist["Close"]
    price = safe_float(close.iloc[-1])
    sma50 = safe_float(close.rolling(50).mean().iloc[-1])
    sma200 = safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
    _, _, macd_hist = compute_macd(close)
    macd_h = safe_float(macd_hist.iloc[-1])
    ret_1m = safe_float((price / close.iloc[-21] - 1) * 100) if len(close) > 21 else float("nan")
    ret_3m = safe_float((price / close.iloc[-63] - 1) * 100) if len(close) > 63 else float("nan")

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

    bench_cache: dict[str, pd.DataFrame | None] = {}
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

def _bulk_latest_close(tickers: list) -> dict:
    """Most-recent close for many tickers in ONE bulk yf.download request.

    Fetching prices one ticker at a time hammered Yahoo and tripped its IP rate
    limiter (crashing the app on Streamlit Cloud). A single `yf.download(...)` for
    the whole list fixes that. Returns {ticker: close} with NaN for anything that
    didn't come back, and never raises.
    """
    out = {t: float("nan") for t in tickers}
    uniq = list(dict.fromkeys(tickers))
    if yf is None or not uniq:
        return out
    try:
        kwargs = dict(period="1d", progress=False, threads=False)
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
            # Multi-ticker frame: column level 0 = field, level 1 = ticker.
            close = data["Close"].dropna(how="all")
            if not close.empty:
                last = close.iloc[-1]
                for t in uniq:
                    if t in last.index:
                        out[t] = safe_float(last[t])
        else:
            # Single-ticker frame: flat columns, 'Close' is a Series.
            series = data["Close"].dropna()
            if not series.empty and len(uniq) == 1:
                out[uniq[0]] = safe_float(series.iloc[-1])
    except Exception:
        return out
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
    bench_cache: dict[str, pd.DataFrame | None] = {}
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
                "INSERT INTO recommendations (ticker, rec_date, recommendation, composite, price_at_rec, momentum, value, technical, hype, quality, price_after, outcome, eval_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t, d, "BUY" if comp > 60 else "HOLD", comp, price, scores["momentum"], scores["value"], scores["technical"], scores["hype"], scores["quality"], price * (1.06 if win else 0.94), win, d)
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
               sources: list | None = None) -> tuple[list[dict], list[str]]:
    weights = get_latest_weights()
    results, failed = [], []

    # Only scan the regions the user selected (defaults to all). On a quick scan,
    # also cap to the first N tickers per region — fewer requests means a faster
    # run and a much lower chance of Yahoo rate-limiting the server's IP.
    selected = regions if regions else list(TICKER_UNIVERSE.keys())
    universe = {
        region: (ticks[:limit_per_region] if limit_per_region else ticks)
        for region, ticks in TICKER_UNIVERSE.items()
        if region in selected
    }

    # Pull sentiment buzz ONCE for the whole scan, routing each market to the
    # enabled sources (Reddit for English names, GDELT news for JP/CN, etc.), then
    # feed each ticker its own combined count. Fully fail-safe: a blocked source
    # contributes zeros, so the scan proceeds with hype unaffected.
    hype_counts = fetch_hype_signals(universe, sources)

    progress = st.progress(0.0, text=tr("scanning"))

    total_symbols = sum(len(ticks) for ticks in universe.values())
    current_index = 0

    for region, tickers in universe.items():
        for t in tickers:
            try:
                analysis = analyze_ticker(t, region, hype_counts.get(t, 0))
                if analysis is None:
                    failed.append(f"{t} (no data)")
                else:
                    analysis.update(score_with_weights(analysis, weights))
                    save_recommendation(analysis)
                    results.append(analysis)
            except Exception as e:
                # Surface the real reason instead of silently swallowing it.
                failed.append(f"{t} ({type(e).__name__})")
            current_index += 1
            time.sleep(0.4)   # gentle pacing so Yahoo doesn't throttle the burst
            progress.progress(current_index / total_symbols, text=tr("processing", ticker=t, region=region_name(region)))
    progress.empty()
    results.sort(key=lambda r: r["composite"], reverse=True)
    # Log today's curated Top-3 Growth / Top-3 Dividend picks for the walk-forward
    # loop. Wrapped so a logging hiccup can never sink an otherwise-successful scan.
    try:
        save_mock_portfolio(results)
    except Exception:
        pass
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

    for r in regional_filtered:
        cols = st.columns([2, 1, 1, 1])
        cols[0].markdown(f'<div class="qc-card"><span class="qc-ticker">{r["ticker"]}</span> {rec_pill(r["recommendation"])}<br><span class="qc-sub">{r["name"]}</span></div>', unsafe_allow_html=True)
        cols[1].markdown(metric_card(tr("card_system_rating"), f"{r['composite']:.1f}"), unsafe_allow_html=True)
        cols[2].markdown(metric_card(tr("card_trading_close"), fmt_money(r["price"])), unsafe_allow_html=True)
        cols[3].markdown(metric_card(tr("card_sustained_hype"), f"{r['hype_score']:.0f}", tr("breakout") if r['hype']['sustained'] else tr("flat"), positive=r['hype']['sustained']), unsafe_allow_html=True)

def render_deep_dive(results: list[dict]) -> None:
    if not results:
        st.info(tr("need_run_history"))
        return
    pick = st.selectbox(tr("select_profile"), [r["ticker"] for r in results])
    r = next(x for x in results if x["ticker"] == pick)

    st.markdown(f"### {tr('history_workspace', ticker=r['ticker'])}")
    chart_df = pd.DataFrame({"Close": r["history"]["Close"], "SMA20": r["history"]["Close"].rolling(20).mean(), "SMA50": r["history"]["Close"].rolling(50).mean()})
    st.line_chart(chart_df.dropna())

    st.markdown(f"#### {tr('factor_breakdown')}")
    breakdown = pd.Series({
        tr("factor_momentum"): r["momentum"],
        tr("factor_value"): r["value"],
        tr("factor_technical"): r["technical"],
        tr("factor_hype"): r["hype_score"],
        tr("factor_quality"): r["quality"],
    })
    st.bar_chart(breakdown)

    mentions = int(r.get("hype_mentions", 0))
    st.markdown(metric_card(tr("hype_buzz_label"), f"{mentions}",
                            positive=mentions > 0), unsafe_allow_html=True)
    st.caption(tr("hype_buzz_caption"))
    if _LAST_HYPE_STATUS:
        st.caption(_LAST_HYPE_STATUS)

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
    mp = mp.head(40)
    uniq = list(mp["ticker"].unique())
    if update_prices:
        # SINGLE bulk request for every unique ticker — replaces the per-ticker
        # fetch storm that was tripping Yahoo's rate limiter and crashing the app.
        prices = _bulk_latest_close(uniq)
    else:
        prices = {t: float("nan") for t in uniq}   # checkbox off -> leave prices blank
    mp["current"] = mp["ticker"].map(prices)
    mp["return_pct"] = (mp["current"] / mp["recommendation_price"] - 1.0) * 100.0
    mp["date"] = pd.to_datetime(mp["timestamp"], errors="coerce").dt.date.astype(str)
    mp["name"] = mp["name"].fillna("")   # older rows logged before names were stored
    reason_map = {"Top Growth": tr("reason_growth"), "Top Dividend": tr("reason_dividend")}
    mp["reason"] = mp["reason"].map(lambda x: reason_map.get(x, x))

    ret_col = tr("col_return_pct")
    rec_col, cur_col = tr("col_rec_price"), tr("col_current_price")
    disp = mp[["date", "ticker", "name", "reason", "recommendation_price", "current", "return_pct"]].rename(columns={
        "date": tr("col_date"), "ticker": tr("col_ticker"), "name": tr("col_company"), "reason": tr("col_reason"),
        "recommendation_price": rec_col, "current": cur_col, "return_pct": ret_col,
    })

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
    st.caption(tr("historical_perf_note"))
    if not update_prices:
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

    st.markdown(f"#### {tr('sell_breakdown_title')}")
    scored = {tr(f"kpi_{k}"): s for k, s, _ in data["signals"] if s is not None}
    if scored:
        st.bar_chart(pd.Series(scored))

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
        # OPTIMISATION: one bulk yf.download for every name's history up front, so
        # the per-ticker loop reuses it instead of making a history request each.
        with st.spinner(tr("sell_bulk_fetch", total=len(portfolio))):
            histories = _bulk_history(portfolio, period="1y")
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
            else:
                rows.append({"ticker": tk, "price": float("nan"),
                             "composite": float("nan"), "verdict": tr("sell_verdict_na")})
            prog.progress(i / len(portfolio), text=tr("sell_scan_progress", done=i, total=len(portfolio)))
        prog.empty()
        st.session_state["portfolio_results"] = rows

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

    # Optional drill-down: full breakdown for any one scanned ticker.
    with st.expander(tr("sell_detail_expander")):
        choices = [r["ticker"] for r in results]
        sel = st.selectbox(tr("sell_detail_select"), choices, key="sell_detail_pick")
        if sel:
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
# Main Application Controller Setup
# ----------------------------------------------------------------------------
def main() -> None:
    init_db()
    inject_css()

    # Resolve matured recommendation outcomes for the audit accuracy panel (no weight
    # change), then run the single learning loop — the walk-forward optimiser — which
    # re-tunes kpi_weights from matured mock-portfolio picks.
    try: evaluate_outcomes_only()
    except Exception: pass
    try: walk_forward_update()
    except Exception: pass

    # Sidebar Interface Controller Layout 
    # Language menu first, so changing it re-renders the whole UI in the chosen language.
    lang_choice = st.sidebar.selectbox("🌐 Language / 言語", list(LANGUAGES.keys()), key="lang_select")
    st.session_state["lang"] = LANGUAGES[lang_choice]

    st.sidebar.title(tr("console_title"))
    region_keys = list(TICKER_UNIVERSE.keys())
    selected_regions = st.sidebar.multiselect(tr("regions_to_scan"), region_keys,
                                              default=region_keys, format_func=region_name)
    quick_scan = st.sidebar.toggle(tr("scan_quick_toggle"), value=True, help=tr("scan_quick_help"))
    update_prices = st.sidebar.checkbox(tr("update_hist_prices"), value=True)
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
                limit = 6 if quick_scan else None
                res, fail = run_engine(limit_per_region=limit, regions=selected_regions,
                                       sources=selected_sources)
                st.session_state["results"], st.session_state["failed"] = res, fail
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

    # Read existing session metrics safely
    results = st.session_state.get("results", [])
    failed = st.session_state.get("failed", [])

    if failed:
        st.sidebar.warning(tr("skipped", items=", ".join(failed)))

    # Main Segment View tabs routing setup
    t1, t2, t3, t4, t5, t6 = st.tabs([tr("tab_top"), tr("tab_regional"), tr("tab_category"), tr("tab_deep"), tr("tab_audit"), tr("tab_sell")])

    with t1: render_daily_top_3(results)
    with t2: render_global_sectors(results)
    with t3: render_category_views(results)
    with t4: render_deep_dive(results)
    with t5: render_engine_audit(update_prices)
    with t6: render_sell_signals()

if __name__ == "__main__":
    main()