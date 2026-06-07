from __future__ import annotations

import math
import sqlite3
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# ----------------------------------------------------------------------------
# Configuration & Global Market Universes
# ----------------------------------------------------------------------------
DB_PATH = "stock_engine.db"
FACTORS = ["momentum", "value", "technical", "hype"]

DEFAULT_WEIGHTS = {
    "momentum": 0.30,
    "value": 0.25,
    "technical": 0.25,
    "hype": 0.20,
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

def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL, rec_date TEXT NOT NULL, recommendation TEXT NOT NULL,
                composite REAL NOT NULL, price_at_rec REAL, momentum REAL, value REAL, 
                technical REAL, hype REAL, price_after REAL, outcome INTEGER, eval_date TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kpi_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT, update_date TEXT NOT NULL,
                momentum REAL, value REAL, technical REAL, hype REAL, note TEXT
            )""")
        conn.commit()
    if get_weight_history().empty:
        save_weights(DEFAULT_WEIGHTS, note="initial defaults")

def get_latest_weights() -> dict[str, float]:
    with get_conn() as conn:
        row = conn.execute("SELECT momentum, value, technical, hype FROM kpi_weights ORDER BY id DESC LIMIT 1").fetchone()
    if row is None: return dict(DEFAULT_WEIGHTS)
    return {f: safe_float(row[f], DEFAULT_WEIGHTS[f]) for f in FACTORS}

def save_weights(weights: dict[str, float], note: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO kpi_weights (update_date, momentum, value, technical, hype, note) VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), weights["momentum"], weights["value"], weights["technical"], weights["hype"], note),
        )
        conn.commit()

def get_weight_history() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT update_date, momentum, value, technical, hype, note FROM kpi_weights ORDER BY id ASC", conn)
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

    with get_conn() as conn:
        conn.execute(
            "DELETE FROM recommendations WHERE ticker=? AND rec_date=? AND outcome IS NULL",
            (rec["ticker"], today),
        )
        conn.execute(
            "INSERT INTO recommendations (ticker, rec_date, recommendation, composite, price_at_rec, momentum, value, technical, hype) VALUES (?,?,?,?,?,?,?,?,?)",
            (rec["ticker"], today, rec["recommendation"], float(rec["composite"]),
             price_val, safe_float(rec["momentum"]), safe_float(rec["value"]),
             safe_float(rec["technical"]), hype_val),
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
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                return df.dropna(subset=["Close"])
        except Exception:
            pass
        time.sleep(1.2 * (attempt + 1))   # back off, then retry
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    out = {"pe": float("nan"), "div_yield": float("nan"), "market_cap": float("nan"), "name": ticker}
    if yf is None: return out
    try: info = yf.Ticker(ticker).info or {}
    except Exception: return out
    out["pe"] = safe_float(info.get("trailingPE"))
    out["market_cap"] = safe_float(info.get("marketCap"))
    out["name"] = str(info.get("shortName") or info.get("longName") or ticker)
    out["div_yield"] = resolve_div_yield(info)
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

def analyze_ticker(ticker: str, region: str) -> dict | None:
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

    return {
        "ticker": ticker, "region": region, "name": funds["name"], "price": price,
        "sma20": sma20, "sma50": sma50, "rsi": rsi, "macd_hist": safe_float(hist_macd.iloc[-1]),
        "bb_pct": pct_b, "ret_1m": ret_1m, "pe": pe, "div_yield": funds["div_yield"],
        "market_cap": funds["market_cap"], "hype": hype, "momentum": momentum_score,
        "value": value_score, "technical": technical_score, "hype_score": hype["score"], "history": hist
    }

def score_with_weights(analysis: dict, weights: dict[str, float]) -> dict:
    composite = sum(weights[f] * analysis[f if f != "hype" else "hype_score"] for f in FACTORS)
    rec = "BUY" if composite >= BUY_THRESHOLD else "SELL" if composite < SELL_THRESHOLD else "HOLD"
    return {"composite": float(composite), "recommendation": rec}

# ----------------------------------------------------------------------------
# Backtesting / Evaluation Systems
# ----------------------------------------------------------------------------
def evaluate_pending() -> int:
    df = get_recommendations()
    if df.empty: return 0
    cutoff = date.today() - timedelta(days=EVAL_HORIZON_DAYS)
    pending = df[(df["outcome"].isna()) & (df["rec_date"].dt.date <= cutoff)]
    if pending.empty: return 0

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
            win = 1 if price_after > price_then else 0
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
                "INSERT INTO recommendations (ticker, rec_date, recommendation, composite, price_at_rec, momentum, value, technical, hype, price_after, outcome, eval_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (t, d, "BUY" if comp > 60 else "HOLD", comp, price, scores["momentum"], scores["value"], scores["technical"], scores["hype"], price * (1.06 if win else 0.94), win, d)
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

    progress = st.progress(0.0, text=tr("scanning"))

    total_symbols = sum(len(ticks) for ticks in universe.values())
    current_index = 0

    for region, tickers in universe.items():
        for t in tickers:
            try:
                analysis = analyze_ticker(t, region)
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

def render_engine_audit() -> None:
    st.markdown(f"### {tr('audit_header_text')}")
    st.caption(tr("persistence_note"))
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
    t1, t2, t3, t4, t5 = st.tabs([tr("tab_top"), tr("tab_regional"), tr("tab_category"), tr("tab_deep"), tr("tab_audit")])

    with t1: render_daily_top_3(results)
    with t2: render_global_sectors(results)
    with t3: render_category_views(results)
    with t4: render_deep_dive(results)
    with t5: render_engine_audit()

if __name__ == "__main__":
    main()
