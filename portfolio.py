"""Rakuten Securities trade-history parsing for the Alpha Quant Engine.

Pure pandas — no Streamlit, no network, no app state — so it is unit-testable in
isolation (see test_portfolio.py). The Streamlit page in app.py only handles widgets,
session caching and display; every transformation lives here.

Rakuten exports two shapes, both Shift-JIS (cp932) with the header on line 1:

  US file : 約定日, 受渡日, ティッカー,   銘柄名, ..., 約定代金［USドル］, 為替レート, 受渡金額［円］
  JP file : 約定日, 受渡日, 銘柄コード, 銘柄名, ..., 単価［円］, 受渡金額［円］

Both are normalised to one canonical frame so the UI never branches on file type.
"""
from __future__ import annotations

import pandas as pd

# Column names used for file-type detection (see detect_kind).
US_KEY_COL = "ティッカー"      # US export: free-text ticker (FB, NVDA)
JP_KEY_COL = "銘柄コード"      # JP export: 4-digit TSE code (3382)

SIDE_BUY = "買付"
SIDE_SELL = "売付"

# Share-count direction per 売買区分. Transfers matter: 入庫 (transferred IN) really
# does increase the holding and 出庫 (transferred OUT) decreases it — treating anything
# that isn't a 買付 as a sale silently turns a transfer-in into a short position.
# Cost basis for transfers is unknown, which affects P&L only, never the share count.
def side_sign(side) -> float:
    s = str(side or "").strip()
    if s == SIDE_BUY or s.startswith("入"):     # 買付 / 入庫 / 入庫（分割）
        return 1.0
    if s == SIDE_SELL or s.startswith("出"):    # 売付 / 出庫（買収）
        return -1.0
    return 0.0

# Canonical output schema — every parsed file produces exactly these columns.
CANON_COLS = ["date", "market", "symbol", "code", "name", "side", "qty",
              "price_local", "value_usd", "value_jpy", "jpy_estimated", "account"]


def detect_kind(df: pd.DataFrame) -> str | None:
    """Classify a raw Rakuten export by its headers: 'US', 'JP', or None if neither
    key column is present (e.g. a dividend statement or an unrelated CSV)."""
    cols = set(df.columns)
    if US_KEY_COL in cols:
        return "US"
    if JP_KEY_COL in cols:
        return "JP"
    return None


def to_num(series: pd.Series) -> pd.Series:
    """Rakuten writes numbers as quoted strings with thousands separators, and uses
    '-' as the placeholder for 'not applicable' (e.g. 受渡金額［円］ on a USD-settled
    trade). Both become NaN rather than raising or silently becoming 0."""
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.strip()
              .replace({"-": None, "": None, "nan": None, "None": None}),
        errors="coerce")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column if present, otherwise an all-NaN column of the right length — keeps the
    canonical schema stable across Rakuten's differing export layouts."""
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index)


def jp_symbol(code) -> str:
    """TSE code -> Yahoo symbol: 3382 (int or str) -> '3382.T'. Already-suffixed
    values pass through untouched."""
    s = str(code).strip()
    if s.endswith(".T") or not s or s.lower() == "nan":
        return s
    return f"{s.split('.')[0]}.T"


def normalize_trades(df: pd.DataFrame, kind: str | None = None) -> pd.DataFrame:
    """Raw Rakuten export -> canonical trade frame (CANON_COLS).

    Currency handling is the subtle part, and it is deliberately explicit:

    * value_usd — US rows use 約定代金［USドル］ (the trade value, always populated).
      JP rows have NO usd figure and the JP export carries no FX rate, so they stay
      NaN. We never invent a conversion rate.
    * value_jpy — 受渡金額［円］ when Rakuten reports it. On USD-settled US trades that
      field is '-', so we fall back to 約定代金［USドル］ x 為替レート and set
      jpy_estimated=True for those rows, so the UI can label them rather than pass
      off a derived number as a reported one.
    """
    kind = kind or detect_kind(df)
    if kind is None or df is None or df.empty:
        return pd.DataFrame(columns=CANON_COLS)

    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(_col(df, "約定日"), errors="coerce")
    out["market"] = kind
    out["name"] = _col(df, "銘柄名").astype(str).str.strip()
    out["side"] = _col(df, "売買区分").astype(str).str.strip()
    out["qty"] = to_num(_col(df, "数量［株］"))
    out["account"] = _col(df, "口座" if kind == "US" else "口座区分").astype(str).str.strip()

    if kind == "US":
        code = _col(df, US_KEY_COL).astype(str).str.strip().str.upper()
        out["code"] = code
        out["symbol"] = code                       # US tickers are Yahoo symbols as-is
        out["price_local"] = to_num(_col(df, "単価［USドル］"))
        out["value_usd"] = to_num(_col(df, "約定代金［USドル］"))
        reported_jpy = to_num(_col(df, "受渡金額［円］"))
        fx = to_num(_col(df, "為替レート"))
        derived_jpy = out["value_usd"] * fx
        out["value_jpy"] = reported_jpy.fillna(derived_jpy)
        out["jpy_estimated"] = reported_jpy.isna() & derived_jpy.notna()
    else:
        code = _col(df, JP_KEY_COL).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        out["code"] = code
        out["symbol"] = code.map(jp_symbol)         # 3382 -> 3382.T
        out["price_local"] = to_num(_col(df, "単価［円］"))
        out["value_usd"] = float("nan")             # no FX rate in the JP export
        out["value_jpy"] = to_num(_col(df, "受渡金額［円］"))
        out["jpy_estimated"] = False

    return out[CANON_COLS].sort_values("date", ascending=False).reset_index(drop=True)


def combine_trades(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge several normalised frames (e.g. the JP and US exports) into one, newest
    first. Exact duplicate rows are dropped so re-uploading a file is harmless."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame(columns=CANON_COLS)
    df = pd.concat(usable, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "symbol", "side", "qty", "price_local"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def recent_sells(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """The most recent 売付 (sell) rows — the Needs-Review queue. Returns an empty
    frame (never raises) when the upload contains no sells at all, which is a normal
    state: a buy-only export legitimately has nothing to review."""
    if df is None or df.empty or "side" not in df.columns:
        return pd.DataFrame(columns=CANON_COLS)
    sells = df[df["side"] == SIDE_SELL]
    return sells.head(limit).reset_index(drop=True)


def net_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol net share count and gross buy/sell cash flows.

    NOT profit and loss: matching sells to specific purchase lots (FIFO/average cost)
    isn't attempted, and transfers-in (入庫) carry no cost basis at all. These are
    factual cash-flow totals only, which is why the columns are named 'bought'/'sold'
    rather than anything implying return.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "name", "market", "net_qty",
                                     "bought_jpy", "sold_jpy", "bought_usd", "sold_usd"])
    d = df.copy()
    sign = d["side"].map(side_sign).fillna(0.0)
    d["signed_qty"] = d["qty"].fillna(0.0) * sign
    is_buy, is_sell = d["side"] == SIDE_BUY, d["side"] == SIDE_SELL
    rows = []
    for sym, g in d.groupby("symbol", sort=False):
        rows.append({
            "symbol": sym,
            "name": g["name"].iloc[0],
            "market": g["market"].iloc[0],
            "net_qty": float(g["signed_qty"].sum()),
            "bought_jpy": float(g.loc[is_buy.loc[g.index], "value_jpy"].sum(min_count=1) or 0.0),
            "sold_jpy": float(g.loc[is_sell.loc[g.index], "value_jpy"].sum(min_count=1) or 0.0),
            "bought_usd": float(g.loc[is_buy.loc[g.index], "value_usd"].sum(min_count=1) or 0.0),
            "sold_usd": float(g.loc[is_sell.loc[g.index], "value_usd"].sum(min_count=1) or 0.0),
        })
    return (pd.DataFrame(rows)
            .sort_values(["net_qty", "symbol"], ascending=[False, True])
            .reset_index(drop=True))


def search_options(df: pd.DataFrame) -> list[str]:
    """Autocomplete labels: 'FB - FACEBOOK INC.' / '3382 - セブン＆アイ・ＨＬＤＧＳ'.
    One entry per traded instrument, code-sorted. Streamlit's selectbox filters as
    you type, so these double as the search index."""
    if df is None or df.empty:
        return []
    seen, opts = set(), []
    for code, name in zip(df["code"], df["name"]):
        label = f"{code} - {name}"
        if label not in seen:
            seen.add(label)
            opts.append(label)
    return sorted(opts)


def option_to_symbol(label: str, df: pd.DataFrame) -> str | None:
    """Reverse an autocomplete label back to its Yahoo symbol ('3382 - ...' -> 3382.T)."""
    if not label or df is None or df.empty:
        return None
    code = label.split(" - ", 1)[0].strip()
    hit = df.loc[df["code"].astype(str) == code, "symbol"]
    return None if hit.empty else str(hit.iloc[0])


def trades_for(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Every row for one symbol, newest first."""
    if df is None or df.empty or not symbol:
        return pd.DataFrame(columns=CANON_COLS)
    return df[df["symbol"] == symbol].reset_index(drop=True)


def net_qty(df: pd.DataFrame, symbol: str) -> float:
    """Net shares held for one symbol: buys and transfers-in minus sells and
    transfers-out. Single source of truth for the share count — the UI must not
    re-derive this (an inline 'everything that isn't a buy is a sell' shortcut turns
    a 入庫 transfer into a negative position)."""
    rows = trades_for(df, symbol)
    if rows.empty:
        return 0.0
    return float((rows["qty"].fillna(0.0) * rows["side"].map(side_sign).fillna(0.0)).sum())


def summary(df: pd.DataFrame, currency: str = "JPY") -> dict:
    """Headline totals in the chosen currency ('JPY' or 'USD').

    `missing` counts rows that have no value in that currency — JP trades have no USD
    figure (no FX rate in the export), so a USD view legitimately can't include them.
    Surfacing the count keeps the totals honest instead of silently under-reporting.
    """
    col = "value_usd" if currency.upper() == "USD" else "value_jpy"
    if df is None or df.empty or col not in df.columns:
        return {"trades": 0, "buy_value": 0.0, "sell_value": 0.0,
                "symbols": 0, "missing": 0, "currency": currency.upper()}
    vals = df[col]
    return {
        "trades": int(len(df)),
        "buy_value": float(vals[df["side"] == SIDE_BUY].sum(min_count=1) or 0.0),
        "sell_value": float(vals[df["side"] == SIDE_SELL].sum(min_count=1) or 0.0),
        "symbols": int(df["symbol"].nunique()),
        "missing": int(vals.isna().sum()),
        "currency": currency.upper(),
    }


def holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Positions still OPEN (net shares > 0) — the set a sell scanner is actually for.

    avg_buy_price is the qty-weighted average of 買付 rows in the local currency. It is
    NOT a lot-matched cost basis: transfers-in (入庫) have no purchase price and, after
    a partial sale, which lots remain is unknown. Treat it as "what I paid on average",
    not as a tax-grade basis.
    """
    pos = net_positions(df)
    if pos.empty:
        return pos
    held = pos[pos["net_qty"] > 0].copy()
    if held.empty:
        return held
    buys = df[df["side"] == SIDE_BUY]
    avg, last = {}, {}
    for sym, g in buys.groupby("symbol", sort=False):
        q = g["qty"].fillna(0.0)
        px = g["price_local"]
        w = (px * q).sum(min_count=1)
        avg[sym] = float(w / q.sum()) if q.sum() > 0 and pd.notna(w) else float("nan")
    for sym, g in df.groupby("symbol", sort=False):
        last[sym] = g["date"].max()
    held["avg_buy_price"] = held["symbol"].map(avg)
    held["last_trade"] = held["symbol"].map(last)
    # invested = gross bought minus gross sold, in yen (available for every row because
    # US yen values are derived from the FX rate when Rakuten omits them)
    held["invested_jpy"] = held["bought_jpy"].fillna(0.0) - held["sold_jpy"].fillna(0.0)
    held["invested_usd"] = held["bought_usd"].fillna(0.0) - held["sold_usd"].fillna(0.0)
    return held.sort_values("invested_jpy", ascending=False).reset_index(drop=True)


def closed_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Fully exited positions (net == 0 and at least one sell) — a re-entry watchlist,
    not a sell-review queue: there is nothing left to sell."""
    pos = net_positions(df)
    if pos.empty:
        return pos
    sold = set(df.loc[df["side"] == SIDE_SELL, "symbol"])
    out = pos[(pos["net_qty"] == 0) & (pos["symbol"].isin(sold))].copy()
    if out.empty:
        return out
    last_sell = (df[df["side"] == SIDE_SELL].groupby("symbol")["date"].max())
    out["exited"] = out["symbol"].map(last_sell)
    return out.sort_values("exited", ascending=False).reset_index(drop=True)


def incomplete_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Symbols whose net share count is NEGATIVE. This is not a short position — it
    means the opening purchase happened before the uploaded export's date range, so
    the history is partial. Surfaced rather than silently treated as a holding."""
    pos = net_positions(df)
    if pos.empty:
        return pos
    return pos[pos["net_qty"] < 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Instrument index — searchable ticker / English name / Japanese name lookup
# ---------------------------------------------------------------------------
def instrument_index(names_en: dict | None = None, names_ja: dict | None = None,
                     trades: pd.DataFrame | None = None,
                     results: list | None = None) -> dict:
    """Merge every name source into {ticker: {"en": str, "ja": str}}.

    Priority is static-first, then live sources fill gaps rather than overwrite:
      1. config.COMPANY_NAMES / config.INSTRUMENT_JA — the curated base.
      2. An uploaded Rakuten CSV — 銘柄名 is Japanese for TSE rows and English for US
         rows, so each row fills whichever side is missing.
      3. Scan results — yfinance long names, English.

    That means a Rakuten upload automatically extends the picker to every stock you
    have ever traded, including names the app's universe has never heard of.
    """
    idx: dict[str, dict] = {}

    def _put(ticker, en=None, ja=None, overwrite=False):
        t = str(ticker or "").strip()
        if not t or t.lower() == "nan":
            return
        cur = idx.setdefault(t, {"en": "", "ja": ""})
        for k, v in (("en", en), ("ja", ja)):
            v = str(v or "").strip()
            if v and (overwrite or not cur[k]):
                cur[k] = v

    for t, n in (names_en or {}).items():
        _put(t, en=n)
    for t, n in (names_ja or {}).items():
        _put(t, ja=n)

    if trades is not None and not trades.empty:
        for sym, nm, mkt in zip(trades["symbol"], trades["name"], trades["market"]):
            # JP export names are Japanese; US export names are English.
            _put(sym, ja=nm) if mkt == "JP" else _put(sym, en=nm)

    for r in (results or []):
        _put(r.get("ticker"), en=r.get("name"))

    return idx


def instrument_labels(index: dict) -> list[str]:
    """Picker labels: 'AAPL — Apple Inc. / アップル'. Both names sit in the one string,
    so Streamlit's built-in selectbox/multiselect filtering matches a ticker, an
    English fragment or a Japanese fragment with no custom search code."""
    labels = []
    for t, n in (index or {}).items():
        en, ja = n.get("en", ""), n.get("ja", "")
        if en and ja and en != ja:
            labels.append(f"{t} — {en} / {ja}")
        elif en or ja:
            labels.append(f"{t} — {en or ja}")
        else:
            labels.append(str(t))
    return sorted(labels)


def label_to_ticker(label: str) -> str | None:
    """'AAPL — Apple Inc. / アップル' -> 'AAPL'."""
    if not label:
        return None
    return str(label).split(" — ", 1)[0].strip() or None
