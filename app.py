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

# Benchmark index per home market for the walk-forward evaluator. A pick is judged
# against the index of the exchange it actually trades on (keyed by ticker suffix),
# so a Tokyo-listed name is measured against the Nikkei rather than the S&P 500.
# Suffix-based mapping also handles dual listings correctly: a US-listed ADR with
# no suffix (e.g. BABA, ASML) is benchmarked against SPY, matching its currency and
# trading calendar. To change a mapping, edit this one dict. Anything not listed
# falls back to DEFAULT_BENCHMARK.
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
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
        # --- Safe migration for pre-existing stock_engine.db files ---
        # Databases created before the "quality" factor existed are missing the
        # column. Add it (defaulting old rows to 0.0) only when absent, so this is
        # a no-op on fresh installs and never errors on an upgrade.
        _ensure_column(conn, "recommendations", "quality")
        _ensure_column(conn, "kpi_weights", "quality")
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
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO kpi_weights (update_date, momentum, value, technical, hype, quality, note) VALUES (?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), weights["momentum"], weights["value"], weights["technical"], weights["hype"], weights["quality"], note),
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
def fetch_history(ticker: str, period: str = "8mo") -> pd.DataFrame | None:
    """Fetch OHLCV with retry/backoff so a throttled burst doesn't wipe a scan."""
    if yf is None:
        return None
    for attempt in range(3):
        try:
            df = _ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                return df.dropna(subset=["Close"])
        except Exception:
            pass
        time.sleep(1.2 * (attempt + 1))   # back off, then retry
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    out = {"pe": float("nan"), "div_yield": float("nan"), "market_cap": float("nan"),
           "roe": float("nan"), "short_pct": float("nan"), "name": ticker}
    if yf is None: return out
    try: info = _ticker(ticker).info or {}
    except Exception: return out
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

def analyze_ticker(ticker: str, region: str, reddit_mentions: int = 0) -> dict | None:
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

    # WallStreetBets buzz — a live retail-sentiment kicker. Each mention adds 10
    # points to the hype score (capped at +35), then the whole score is re-clamped
    # to 0-100. Zero mentions (or a failed Reddit fetch) leave hype untouched.
    if reddit_mentions > 0:
        hype["score"] = clamp(hype["score"] + min(35, reddit_mentions * 10))

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
        "hype_score": hype["score"], "reddit_mentions": reddit_mentions, "history": hist
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
    try:
        tk = _ticker(ticker)
    except Exception:
        return out
    try:
        info = tk.info or {}
        out["rec_mean"] = safe_float(info.get("recommendationMean"))
        out["rec_key"] = str(info.get("recommendationKey") or "")
        out["num_analysts"] = safe_float(info.get("numberOfAnalystOpinions"))
        out["target_mean"] = safe_float(info.get("targetMeanPrice"))
        out["currency"] = str(info.get("currency") or "")
    except Exception:
        pass
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
    try:
        it = _ticker(ticker).insider_transactions
    except Exception:
        return out
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

def analyze_sell_signals(ticker: str) -> dict | None:
    """Compose every bearish KPI into a single sell-pressure profile for one ticker."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None
    hist = fetch_history(ticker, period="1y")   # 1y so SMA200 is available
    if hist is None or len(hist) < 60:
        return None

    # Pace the remaining requests slightly. Hitting four Yahoo endpoints back-to-back
    # for one ticker is a common trigger for IP-level rate limiting, so we space them.
    time.sleep(0.2)
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
def evaluate_pending() -> int:
    df = get_recommendations()
    if df.empty: return 0
    cutoff = date.today() - timedelta(days=EVAL_HORIZON_DAYS)
    pending = df[(df["outcome"].isna()) & (df["rec_date"].dt.date <= cutoff)]
    if pending.empty: return 0

    # Per-market benchmarking: each pick is judged against its HOME index over the
    # SAME holding window, so a "Win" means it beat its local market — not merely
    # the S&P 500. Benchmark histories are fetched once per distinct index and
    # reused across tickers (fetch_history is also memoised by Streamlit's cache).
    bench_cache: dict[str, pd.DataFrame | None] = {}
    def bench_hist(symbol: str) -> pd.DataFrame | None:
        if symbol not in bench_cache:
            bench_cache[symbol] = fetch_history(symbol, period="1y")
        return bench_cache[symbol]

    def _close_on_or_after(h: pd.DataFrame | None, when) -> float:
        if h is None or h.empty: return float("nan")
        sub = h[h.index.tz_localize(None) >= pd.Timestamp(when)]
        return safe_float(sub["Close"].iloc[0]) if not sub.empty else float("nan")

    evaluated = 0
    with get_conn() as conn:
        for _, row in pending.iterrows():
            hist = fetch_history(row["ticker"], period="1y")
            if hist is None: continue
            target = row["rec_date"] + timedelta(days=EVAL_HORIZON_DAYS)
            future = hist[hist.index.tz_localize(None) >= pd.Timestamp(target)]
            if future.empty: continue
            price_after = safe_float(future["Close"].iloc[0])
            price_then = safe_float(row["price_at_rec"])
            if math.isnan(price_after) or math.isnan(price_then) or price_then <= 0: continue
            stock_ret = price_after / price_then - 1.0

            # Home-market benchmark return over the identical [rec_date, target] window.
            bh = bench_hist(benchmark_for(row["ticker"]))
            b_then = _close_on_or_after(bh, row["rec_date"])
            b_after = _close_on_or_after(bh, target)
            if not math.isnan(b_then) and not math.isnan(b_after) and b_then > 0:
                bench_ret = b_after / b_then - 1.0
                win = 1 if stock_ret > bench_ret else 0
            else:
                # Benchmark data missing for this window (e.g. very old rec) — fall
                # back to the original absolute-return test so the record resolves.
                win = 1 if stock_ret > 0 else 0

            conn.execute("UPDATE recommendations SET price_after=?, outcome=?, eval_date=? WHERE id=?", (price_after, win, date.today().isoformat(), int(row["id"])))
            evaluated += 1
        conn.commit()
    if evaluated: update_weights_from_outcomes()
    return evaluated

def update_weights_from_outcomes() -> None:
    df = get_recommendations()
    done = df[df["outcome"].notna()]
    if len(done) < 4: return
    weights = get_latest_weights()
    grad = {f: 0.0 for f in FACTORS}
    for _, r in done.iterrows():
        direction = 1.0 if int(r["outcome"]) == 1 else -1.0
        for f in FACTORS:
            grad[f] += direction * (safe_float(r[f], 50.0) / 100.0 - 0.5)
    n = len(done)
    new = {f: max(MIN_WEIGHT, weights[f] + LEARNING_RATE * grad[f] / n) for f in FACTORS}
    total = sum(new.values())
    new = {f: new[f] / total for f in FACTORS}
    save_weights(new, note=f"Auto-tuned on {n} metrics (Win rate: {100 * done['outcome'].mean():.0f}%)")

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

def run_engine(limit_per_region: int | None = None) -> tuple[list[dict], list[str]]:
    weights = get_latest_weights()
    results, failed = [], []

    # On a quick scan, only take the first N tickers per region. Fewer requests
    # means a faster run, lighter memory use, and a much lower chance of Yahoo
    # rate-limiting the server's IP (a frequent issue on Streamlit Cloud).
    universe = {
        region: (ticks[:limit_per_region] if limit_per_region else ticks)
        for region, ticks in TICKER_UNIVERSE.items()
    }

    # Pull WallStreetBets mention counts ONCE for the whole scan (a single RSS
    # request), then feed each ticker its own count. Fail-safe: a blocked/failed
    # request returns all-zeros, so the scan proceeds with hype unaffected.
    scan_tickers = [t for ticks in universe.values() for t in ticks]
    reddit_counts = fetch_reddit_hype(scan_tickers)

    progress = st.progress(0.0, text=tr("scanning"))

    total_symbols = sum(len(ticks) for ticks in universe.values())
    current_index = 0

    for region, tickers in universe.items():
        for t in tickers:
            try:
                analysis = analyze_ticker(t, region, reddit_counts.get(t, 0))
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
        st.dataframe(growth_df[["ticker", "name", "region", "price", "ret_1m", "composite"]].rename(columns={"ticker": tr("col_ticker"), "name": tr("col_company"), "region": tr("col_region"), "price": tr("col_price"), "ret_1m": tr("col_momentum_1m"), "composite": tr("col_overall_score")}), use_container_width=True, hide_index=True)

    with v2:
        st.markdown(f"### {tr('dividend_header_text')}")
        div_df = df[df["div_yield"] >= 1.5].sort_values(by="composite", ascending=False).head(5).copy()
        if not div_df.empty:
            div_df["region"] = div_df["region"].map(region_name)
            st.dataframe(div_df[["ticker", "name", "region", "price", "div_yield", "composite"]].rename(columns={"ticker": tr("col_ticker"), "name": tr("col_company"), "region": tr("col_region"), "price": tr("col_price"), "div_yield": tr("col_dividend_return"), "composite": tr("col_overall_score")}), use_container_width=True, hide_index=True)
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

    mentions = int(r.get("reddit_mentions", 0))
    st.markdown(metric_card(tr("wsb_mentions"), f"{mentions}",
                            positive=mentions > 0), unsafe_allow_html=True)
    st.caption(tr("wsb_caption"))
    st.caption(tr(f"wsb_source_{_LAST_REDDIT_SOURCE}"))

def render_engine_audit() -> None:
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
        plot = wh.set_index("update_date")[FACTORS]
        st.line_chart(plot)

def _sell_pill(verdict: str) -> str:
    cls = {"high": "qc-sell", "mixed": "qc-hold", "low": "qc-buy"}.get(verdict, "qc-hold")
    return f'<span class="qc-pill {cls}">{tr(f"sell_verdict_{verdict}")}</span>'

def render_sell_signals() -> None:
    st.markdown(f"### {tr('sell_header')}")
    st.caption(tr("sell_disclaimer"))

    col_in, col_btn = st.columns([3, 1])
    typed = col_in.text_input(tr("sell_input_label"), key="sell_ticker_input", placeholder="AAPL")
    col_btn.button(tr("sell_btn"), use_container_width=True)   # affordance; Enter also submits

    target = (typed or "").strip().upper()
    if not target:
        st.info(tr("sell_need_input"))
        return

    with st.spinner(tr("sell_spinner", ticker=target)):
        data = analyze_sell_signals(target)

    if data is None:
        st.warning(tr("sell_no_data", ticker=target))
        return

    cur = (data["currency"] + " ") if data["currency"] else ""
    comp_txt = "—" if math.isnan(data["composite"]) else f"{data['composite']:.0f}/100"
    price_txt = "—" if math.isnan(data["price"]) else f"{cur}{data['price']:,.2f}"
    c = st.columns([2, 1, 1])
    c[0].markdown(
        f'<div class="qc-card"><span class="qc-ticker">{data["ticker"]}</span> {_sell_pill(data["verdict"])}'
        f'<br><span class="qc-sub">{data["name"]}</span></div>',
        unsafe_allow_html=True,
    )
    # For sell pressure, LOW is the "good/green" reading, so positive=True when low.
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
    st.dataframe(pd.DataFrame(readings), use_container_width=True, hide_index=True)

    st.markdown(f"#### {tr('sell_recent_downgrades')}")
    if data["changes"]:
        cdf = pd.DataFrame(data["changes"]).rename(columns={
            "date": tr("col_date"), "firm": tr("col_firm"),
            "from": tr("col_from"), "to": tr("col_to"), "action": tr("col_action"),
        })
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    else:
        st.caption(tr("sell_no_analyst"))

    st.markdown(f"#### {tr('sell_insider_activity')}")
    ins = data["insider"]
    if ins["available"] and ins["rows"]:
        idf = pd.DataFrame(ins["rows"]).rename(columns={
            "date": tr("col_date"), "insider": tr("col_insider"),
            "transaction": tr("col_transaction"), "shares": tr("col_shares"), "value": tr("col_value"),
        })
        st.dataframe(idf, use_container_width=True, hide_index=True)
    else:
        st.caption(tr("sell_no_insider"))

# ----------------------------------------------------------------------------
# Main Application Controller Setup
# ----------------------------------------------------------------------------
def main() -> None:
    init_db()
    inject_css()

    # Auto Execution of background performance auditor calculations
    try: evaluate_pending()
    except Exception: pass

    # Sidebar Interface Controller Layout 
    # Language menu first, so changing it re-renders the whole UI in the chosen language.
    lang_choice = st.sidebar.selectbox("🌐 Language / 言語", list(LANGUAGES.keys()), key="lang_select")
    st.session_state["lang"] = LANGUAGES[lang_choice]

    st.sidebar.title(tr("console_title"))
    quick_scan = st.sidebar.toggle(tr("scan_quick_toggle"), value=True, help=tr("scan_quick_help"))
    if st.sidebar.button(tr("scan_btn"), use_container_width=True):
        with st.spinner(tr("scan_spinner")):
            limit = 6 if quick_scan else None
            res, fail = run_engine(limit_per_region=limit)
            st.session_state["results"], st.session_state["failed"] = res, fail
        if res:
            st.success(tr("scan_success"))
        elif fail:
            # Nothing came back and tickers failed → almost always an IP-level
            # rate limit from Yahoo, which is common on Streamlit Cloud.
            st.error(tr("all_failed"))

    if st.sidebar.button(tr("seed_btn"), use_container_width=True):
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
    with t5: render_engine_audit()
    with t6: render_sell_signals()

if __name__ == "__main__":
    main()