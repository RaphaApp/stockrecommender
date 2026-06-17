"""
sec_research.py
===============
US-only fundamental research from the SEC's official, free EDGAR APIs — built to feed
a "US Conviction" view without touching the main app's global factor model.

What it does
------------
* ``cik_for(ticker)``        — map a US ticker to its 10-digit SEC CIK (company_tickers.json).
* ``company_facts(cik)``     — pull the XBRL companyfacts bundle (audited filings).
* ``parse_facts(facts)``     — PURE function: turn that bundle into multi-year revenue /
                               net-income / operating-cash-flow / R&D series and derived
                               metrics, then a 0-100 Growth score and a 0-100 Quality score.
* ``sec_fundamentals(ticker)`` — network wrapper: cik_for -> company_facts -> parse_facts.

Compliance / etiquette
----------------------
data.sec.gov is public and free (no key), but SEC's fair-access policy requires a
descriptive ``User-Agent`` with contact info and ~10 req/s max. Pass your own via the
``ua`` argument (the app reads it from the SEC_USER_AGENT secret/env). Honors the same
corporate-SSL escape hatches as the main app: STOCKREC_CA_BUNDLE / STOCKREC_INSECURE_SSL.

Scope
-----
EDGAR is US filers only, by design — this module is meant to power a US-only tab, so the
US-only data never leaks into the app's cross-region ranking or its walk-forward weights.

Phase 2 hook
------------
``SUPERINVESTOR_CIKS`` + ``superinvestor_counts()`` are scaffolded (the latter currently
returns zeros). The 13F-HR parser that fills it in is the next phase; the scoring already
leaves a slot for it so wiring it in is a drop-in.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET

NEUTRAL = float("nan")

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}"

# Candidate XBRL (us-gaap) concept names, tried in order until one yields a usable series.
_REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
_NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
_OCF_CONCEPTS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
_RND_CONCEPTS = ("ResearchAndDevelopmentExpense",)

# Superinvestor managers -> SEC CIK (for the Phase-2 13F overlay).
SUPERINVESTOR_CIKS = {
    "Berkshire Hathaway (Buffett)": "0001067983",
    "Duquesne (Druckenmiller)": "0001536411",
    "Scion (Burry)": "0001649339",
    "Pershing Square (Ackman)": "0001336528",
    "Tiger Global (Coleman)": "0001167483",
    "Coatue (Laffont)": "0001135730",
    "Greenlight (Einhorn)": "0001079114",
}

_CIK_MAP: dict[str, str] | None = None
_TITLE_MAP: dict[str, str] | None = None   # ticker -> normalized company title (13F name matching)


# ---------------------------------------------------------------------------
# networking
# ---------------------------------------------------------------------------
def _ssl_context() -> ssl.SSLContext | None:
    """Mirror the main app's corporate-network escape hatches."""
    if os.environ.get("STOCKREC_INSECURE_SSL") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    bundle = os.environ.get("STOCKREC_CA_BUNDLE")
    if bundle and os.path.exists(bundle):
        return ssl.create_default_context(cafile=bundle)
    return None


def _get_raw(url: str, ua: str, timeout: float = 20.0) -> str:
    """GET a URL with the required User-Agent, paced for SEC fair-access. Returns decoded
    text. RAISES on any failure (so callers' caches never store a transient error)."""
    time.sleep(0.15)   # SEC fair-access: stay under ~10 req/s regardless of how fast
                       # EDGAR replies. Sequential calls are spaced >=0.15s (~6.7/s).
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="replace")


def _get_json(url: str, ua: str, timeout: float = 20.0):
    return json.loads(_get_raw(url, ua, timeout))


# Tokens stripped when normalising a company name for fuzzy 13F issuer<->ticker matching.
_NAME_DROP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD",
              "LIMITED", "PLC", "LLC", "LP", "HOLDINGS", "HOLDING", "GROUP", "THE",
              "COM", "CL", "CLASS", "A", "B", "C", "SA", "NV", "AG", "TRUST", "FUND",
              "ADR", "NEW", "SHS", "SHARES", "ORD"}


def _norm_name(name: str) -> str:
    """Normalise a company name so a 13F 'nameOfIssuer' lines up with an SEC title:
    uppercase, strip punctuation, and drop generic corporate-suffix tokens."""
    s = re.sub(r"[^A-Za-z0-9 ]", " ", str(name or "").upper())
    return " ".join(t for t in s.split() if t and t not in _NAME_DROP)


def _load_ticker_map(ua: str) -> None:
    """One-time fetch of SEC's company_tickers.json -> ticker->CIK and ticker->normalised
    title. RAISES on a failed fetch (so a transient error isn't cached upstream)."""
    global _CIK_MAP, _TITLE_MAP
    if _CIK_MAP is not None:
        return
    data = _get_json(_TICKERS_URL, ua)
    cmap, tmap = {}, {}
    for row in data.values():
        tk = str(row["ticker"]).upper()
        cmap[tk] = str(row["cik_str"]).zfill(10)
        tmap[tk] = _norm_name(row.get("title", ""))
    _CIK_MAP, _TITLE_MAP = cmap, tmap


def cik_for(ticker: str, ua: str) -> str | None:
    """US ticker -> zero-padded 10-digit CIK, or None if not an SEC filer."""
    if "." in ticker:                       # foreign-suffixed tickers aren't US filers
        return None
    _load_ticker_map(ua)
    return (_CIK_MAP or {}).get(ticker.upper())


def _title_norm_for(ticker: str, ua: str) -> str | None:
    """US ticker -> normalised company title (for 13F name matching), or None."""
    if "." in ticker:
        return None
    _load_ticker_map(ua)
    return (_TITLE_MAP or {}).get(ticker.upper())


def company_facts(cik: str, ua: str) -> dict:
    return _get_json(_FACTS_URL.format(cik=cik), ua)


# ---------------------------------------------------------------------------
# parsing + scoring  (PURE — no network, fully unit-testable)
# ---------------------------------------------------------------------------
def _annual_series(facts: dict, concepts) -> list[tuple[int, float]]:
    """Pull annual (FY, 10-K) values for the first concept that yields >=2 years.
    Dedupes restatements by keeping, per fiscal year, the most recently filed value.
    Returns [(fiscal_year, value), ...] sorted ascending by year."""
    gaap = (facts.get("facts", {}) or {}).get("us-gaap", {}) or {}
    for concept in concepts:
        node = gaap.get(concept)
        if not node:
            continue
        units = node.get("units", {}) or {}
        rows = units.get("USD") or next((v for v in units.values()), [])
        by_year: dict[int, tuple[str, float]] = {}
        for e in rows:
            if e.get("fp") != "FY" or not str(e.get("form", "")).startswith("10-K"):
                continue
            fy = e.get("fy")
            val = e.get("val")
            if fy is None or val is None:
                continue
            stamp = str(e.get("filed") or e.get("end") or "")
            prev = by_year.get(int(fy))
            if prev is None or stamp >= prev[0]:
                by_year[int(fy)] = (stamp, float(val))
        series = sorted((fy, v) for fy, (_, v) in by_year.items())
        if len(series) >= 2:
            return series
    return []


def _yoy(series: list[tuple[int, float]]) -> float:
    if len(series) < 2:
        return NEUTRAL
    prev, cur = series[-2][1], series[-1][1]
    return (cur / prev - 1.0) if prev > 0 else NEUTRAL


def _cagr(series: list[tuple[int, float]]) -> float:
    if len(series) < 2:
        return NEUTRAL
    span = series[-4:] if len(series) >= 4 else series
    first, last, n = span[0][1], span[-1][1], len(span) - 1
    return ((last / first) ** (1.0 / n) - 1.0) if first > 0 and last > 0 and n > 0 else NEUTRAL


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _nanmean(xs) -> float:
    vals = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else NEUTRAL


def parse_facts(facts: dict) -> dict:
    """Turn a companyfacts bundle into derived metrics + 0-100 Growth/Quality scores.
    PURE and fail-soft: missing concepts become NaN, never an exception."""
    rev = _annual_series(facts, _REVENUE_CONCEPTS)
    ni = _annual_series(facts, _NET_INCOME_CONCEPTS)
    ocf = _annual_series(facts, _OCF_CONCEPTS)
    rnd = _annual_series(facts, _RND_CONCEPTS)

    rev_yoy, rev_cagr = _yoy(rev), _cagr(rev)
    eps_yoy, ocf_yoy = _yoy(ni), _yoy(ocf)
    ocf_consistency = (sum(1 for _, v in ocf if v > 0) / len(ocf)) if ocf else NEUTRAL
    ni_consistency = (sum(1 for _, v in ni if v > 0) / len(ni)) if ni else NEUTRAL
    rnd_intensity = (rnd[-1][1] / rev[-1][1]) if rnd and rev and rev[-1][1] > 0 else NEUTRAL

    # Growth 0-100: 3y revenue CAGR is the spine (+50pts per +25%), nudged by latest
    # revenue and earnings YoY. Mirrors the app's "50 + metric*k" factor convention.
    growth_score = _nanmean([
        _clamp(50.0 + rev_cagr * 200.0) if not math.isnan(rev_cagr) else NEUTRAL,
        _clamp(50.0 + rev_yoy * 150.0) if not math.isnan(rev_yoy) else NEUTRAL,
        _clamp(50.0 + eps_yoy * 100.0) if not math.isnan(eps_yoy) else NEUTRAL,
    ])
    # Quality 0-100: cash-flow consistency + earnings consistency + OCF momentum.
    quality_score = _nanmean([
        ocf_consistency * 100.0 if not math.isnan(ocf_consistency) else NEUTRAL,
        ni_consistency * 100.0 if not math.isnan(ni_consistency) else NEUTRAL,
        _clamp(50.0 + ocf_yoy * 100.0) if not math.isnan(ocf_yoy) else NEUTRAL,
    ])

    return {
        "entity": str(facts.get("entityName") or ""),
        "years": [fy for fy, _ in rev],
        "rev_yoy": rev_yoy, "rev_cagr_3y": rev_cagr,
        "earnings_yoy": eps_yoy, "ocf_yoy": ocf_yoy,
        "ocf_consistency": ocf_consistency, "ni_consistency": ni_consistency,
        "rnd_intensity": rnd_intensity,
        "growth_score": growth_score, "quality_score": quality_score,
        "available": bool(rev or ni or ocf),
    }


# ---------------------------------------------------------------------------
# network wrapper
# ---------------------------------------------------------------------------
def _empty() -> dict:
    return {"entity": "", "years": [], "rev_yoy": NEUTRAL, "rev_cagr_3y": NEUTRAL,
            "earnings_yoy": NEUTRAL, "ocf_yoy": NEUTRAL, "ocf_consistency": NEUTRAL,
            "ni_consistency": NEUTRAL, "rnd_intensity": NEUTRAL,
            "growth_score": NEUTRAL, "quality_score": NEUTRAL, "available": False}


def sec_fundamentals(ticker: str, ua: str = "stockrec-research/1.0 (set SEC_USER_AGENT)") -> dict:
    """Full pull for one US ticker. Returns the parsed-metrics dict. Non-US tickers (or
    names SEC doesn't list) return an empty/unavailable dict WITHOUT raising. A genuine
    network/HTTP failure RAISES so the app's cache won't freeze a transient error."""
    cik = cik_for(ticker, ua)        # may raise on a failed tickers fetch (not cached)
    if not cik:
        return _empty()              # not an SEC filer -> legitimately unavailable
    facts = company_facts(cik, ua)   # may raise on a failed facts fetch (not cached)
    return parse_facts(facts)


# ---------------------------------------------------------------------------
# 13F superinvestor overlap (Phase 2)
# ---------------------------------------------------------------------------
def _latest_13f_infotable_url(cik: str, ua: str) -> str | None:
    """Find a manager's most recent 13F-HR filing and return the URL of its holdings
    (information-table) XML, or None if not found."""
    sub = _get_json(_SUBMISSIONS_URL.format(cik=cik), ua)
    recent = (sub.get("filings", {}) or {}).get("recent", {}) or {}
    forms = recent.get("form", []) or []
    accns = recent.get("accessionNumber", []) or []
    for i, form in enumerate(forms):              # arrays are most-recent-first
        if form != "13F-HR":                      # skip 13F-NT notices, amendments, etc.
            continue
        accn = str(accns[i]).replace("-", "")
        base = _ARCHIVES.format(cik=int(cik), accn=accn)
        idx = _get_json(f"{base}/index.json", ua)
        names = [str(it.get("name", "")) for it in
                 (idx.get("directory", {}) or {}).get("item", []) or []]
        xmls = [n for n in names if n.lower().endswith(".xml")]
        # prefer an obviously-named info table; otherwise any XML that isn't the cover page
        for n in xmls:
            low = n.lower()
            if "primary_doc" not in low and any(k in low for k in ("infotable", "form13f", "table")):
                return f"{base}/{n}"
        for n in xmls:
            if "primary_doc" not in n.lower():
                return f"{base}/{n}"
        return None
    return None


def _parse_13f_issuers(xml_text: str) -> set:
    """Return the set of normalised issuer names held in a 13F information table.
    Namespace-tolerant; never raises."""
    issuers: set = set()
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return issuers
    for el in root.iter():
        if el.tag.split("}")[-1] == "nameOfIssuer" and el.text:
            n = _norm_name(el.text)
            if n:
                issuers.add(n)
    return issuers


def superinvestor_counts(tickers: list[str], ua: str = "") -> dict:
    """How many of SUPERINVESTOR_CIKS hold each ticker in their latest 13F-HR.

    13F discloses CUSIPs + issuer NAMES, not tickers, and there's no free official
    CUSIP->ticker map, so this matches on normalised company name — reliable for the
    large, well-known holdings these managers report (and that populate a US universe),
    best-effort at the edges. Quarterly data with a ~45-day filing lag; US tickers only.

    Fail-safe: a manager whose filing can't be fetched/parsed is skipped. If EVERY
    manager fails (a total outage) it RAISES, so the caller's cache won't freeze an
    all-zero result; a partial success returns normally.
    """
    counts = {t: 0 for t in tickers}
    if not ua:
        return counts
    name_to_ticker: dict[str, str] = {}
    for t in tickers:
        nm = _title_norm_for(t, ua)     # may raise on a failed tickers-map fetch (not cached)
        if nm:
            name_to_ticker[nm] = t
    if not name_to_ticker:
        return counts

    successes = 0
    for cik in SUPERINVESTOR_CIKS.values():
        try:
            url = _latest_13f_infotable_url(cik, ua)
            if not url:
                continue
            held = _parse_13f_issuers(_get_raw(url, ua))
            successes += 1
            for nm, tkr in name_to_ticker.items():
                if nm in held:
                    counts[tkr] += 1
        except Exception:
            continue                    # this manager unavailable -> skip, keep going
    if successes == 0:
        raise RuntimeError("no 13F filings could be retrieved")
    return counts


if __name__ == "__main__":
    for tk in ("AAPL", "NVDA", "MSFT"):
        try:
            print(tk, sec_fundamentals(tk, ua="demo demo@example.com"))
        except Exception as e:
            print(tk, "fetch failed:", type(e).__name__, e)