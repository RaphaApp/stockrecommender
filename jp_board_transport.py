"""Secure, bounded Yahoo board transport. No login, private endpoints or bypass."""
from __future__ import annotations
import os
import re
import ssl
import urllib.request
import urllib.error
from datetime import datetime, timezone

MAX_PAGE_BYTES = 5_000_000

def fetch_board_page(ticker: str, timeout: float = 15.0, opener=None) -> tuple[str, dict]:
    ticker = str(ticker or "").strip().upper()
    if not re.fullmatch(r"[0-9A-Z]{4}\.T", ticker):
        raise ValueError("Expected a four-character Tokyo ticker ending .T")
    url = f"https://finance.yahoo.co.jp/quote/{ticker}/forum"
    bundle = os.environ.get("STOCKREC_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "ja", "Accept": "text/html"})
    with (opener or urllib.request.urlopen)(req, timeout=timeout, context=ctx) as resp:
        final = resp.geturl()
        if final != url:
            raise ValueError("Unexpected redirect; possible login or consent page")
        raw = resp.read(MAX_PAGE_BYTES + 1)
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError("Page exceeds safe size limit")
        encoding = resp.headers.get_content_charset() or "utf-8"
        text = raw.decode(encoding, errors="replace")
        return text, {"source_url": url, "http_status": getattr(resp, "status", 200),
                      "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "transport": "urllib_verified_tls", "page_bytes": len(raw)}
