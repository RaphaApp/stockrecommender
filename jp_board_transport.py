"""Secure, bounded Yahoo board transport. No login, private endpoints or bypass.

Properties (all kept from the supplied patch): verified TLS (an optional trusted CA bundle
via STOCKREC_CA_BUNDLE; STOCKREC_INSECURE_SSL is deliberately NOT honoured here), a hard
cap on the bytes read, strict Tokyo-ticker validation, a refusal to follow a redirect to a
different URL (a login/consent page), and no cookies, credentials or access-control bypass.

Added in the review: every failure that is not an HTTP status is raised as a
BoardTransportError carrying a short machine-readable `reason`, so the app can show WHY
(`tls_verification_failed`, `unexpected_redirect`, `page_too_large`, `timeout`, ...) instead
of a bare exception class name. BoardTransportError is a ValueError, so code written for the
original contract keeps working. HTTP errors (403, 429, ...) still propagate as
urllib.error.HTTPError so the caller can see the status code.
"""
from __future__ import annotations
import os
import re
import socket
import ssl
import urllib.request
import urllib.error
from datetime import datetime, timezone

MAX_PAGE_BYTES = 5_000_000


class BoardTransportError(ValueError):
    """A transport failure with a stable, machine-readable reason code."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason


def _is_tls_failure(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, "reason", None), ssl.SSLError)


def fetch_board_page(ticker: str, timeout: float = 15.0, opener=None) -> tuple[str, dict]:
    ticker = str(ticker or "").strip().upper()
    if not re.fullmatch(r"[0-9A-Z]{4}\.T", ticker):
        raise BoardTransportError("invalid_ticker", "Expected a four-character Tokyo ticker ending .T")
    url = f"https://finance.yahoo.co.jp/quote/{ticker}/forum"
    bundle = os.environ.get("STOCKREC_CA_BUNDLE")
    try:
        ctx = ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()
    except (OSError, ssl.SSLError) as exc:
        raise BoardTransportError("ca_bundle_unreadable", "STOCKREC_CA_BUNDLE could not be loaded") from exc
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "ja", "Accept": "text/html"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout, context=ctx) as resp:
            final = resp.geturl()
            if final != url:
                raise BoardTransportError("unexpected_redirect", "Unexpected redirect; possible login or consent page")
            raw = resp.read(MAX_PAGE_BYTES + 1)
            if len(raw) > MAX_PAGE_BYTES:
                raise BoardTransportError("page_too_large", "Page exceeds safe size limit")
            encoding = resp.headers.get_content_charset() or "utf-8"
            try:
                text = raw.decode(encoding, errors="replace")
            except LookupError:                       # a charset name Python does not know
                text = raw.decode("utf-8", errors="replace")
            return text, {"source_url": url, "http_status": getattr(resp, "status", 200),
                          "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "transport": "urllib_verified_tls", "page_bytes": len(raw)}
    except urllib.error.HTTPError:
        raise                                          # keep the status code for the caller
    except BoardTransportError:
        raise
    except (socket.timeout, TimeoutError) as exc:
        raise BoardTransportError("timeout", "The request timed out") from exc
    except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
        if isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
            raise BoardTransportError("timeout", "The request timed out") from exc
        if _is_tls_failure(exc):
            raise BoardTransportError(
                "tls_verification_failed",
                "TLS certificate verification failed; set STOCKREC_CA_BUNDLE to your trusted CA file") from exc
        raise BoardTransportError("network_error", type(exc).__name__) from exc
