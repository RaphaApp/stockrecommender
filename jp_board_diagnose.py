"""Diagnose why the Yahoo! Japan board poll (みんなの評価) is or is not being read.

Run from the app folder (it needs sentiment_sources.py, indicators.py and
jp_board_transport.py beside it):

    python jp_board_diagnose.py 7203.T                       # fetch the page and explain it
    python jp_board_diagnose.py 7203.T --save page.html      # also keep the raw HTML
    python jp_board_diagnose.py 7203.T --record run.json     # keep a small run record
    python jp_board_diagnose.py 7203.T --poll-html poll.html # keep ONLY the poll block
    python jp_board_diagnose.py --file page.html             # analyse a page you saved

It tells you, in plain words, whether the parser can read the poll, and if not, which of
the usual causes applies: the heading is missing (layout change, login/consent page,
blocking), the poll numbers sit only inside script data (JSON) that the parser does not
read, the markup is too malformed to parse safely, or the poll is present but malformed.
Nothing here changes the app or its data.

--record preserves a WORKING run: ticker, retrieval time (UTC), HTTP status, size, parser
version, status, the percentages, and which source produced them. --poll-html writes just
the poll block's markup (not the whole page) and checks that parsing it gives the same
result, so a working layout can be kept as a small fixture without the rest of the page.
Look at any saved file before you share it; the checks below only catch obvious secrets.

Personal diagnostic use only. The app reads this unofficial page at personal-use level and
stops on blocking; Yahoo!'s terms apply to you. No credentials are involved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from sentiment_sources import describe_poll_page, poll_scope_html

URL = "https://finance.yahoo.co.jp/quote/{ticker}/forum"
_SECRET_HINTS = re.compile(r"(?i)(password|passwd|bearer\s+[a-z0-9]|csrf|session(id)?=|api[_-]?key|"
                           r"access[_-]?token|set-cookie|[\w.+-]+@[\w-]+\.[\w.]+)")


def fetch_page(ticker: str, timeout: float = 15.0) -> tuple[str, dict]:
    from jp_board_transport import fetch_board_page
    return fetch_board_page(ticker, timeout=timeout)


def fetch(ticker: str, timeout: float = 15.0) -> str:
    return fetch_page(ticker, timeout=timeout)[0]


def verdict(d: dict) -> str:
    r = d["result"]
    if r["status"] == "ok":
        extra = ""
        if r.get("summary_buy_pct") is not None or r.get("summary_sell_pct") is not None:
            extra = (f" A rounded headline summary (buy {r.get('summary_buy_pct')}%, "
                     f"sell {r.get('summary_sell_pct')}%) was found and agrees with the categories.")
        src = {"visible_text": "the visible page text",
               "accessible_label": "an ACCESSIBLE LABEL (aria-label); no visible copy of the breakdown parsed",
               "visible_and_accessible": "both the visible text and an accessible label, which agree"
               }.get(d.get("poll_source"), "the page")
        warn = ""
        if d.get("source_disagreement"):
            warn = ("\n  WARNING: the visible text and the accessible label parse to DIFFERENT numbers. "
                    "The score uses the combined reading shown above; check the page by eye.")
        return (f"READABLE. Buy {r['buy_pct']:.2f}% / watch {r['neutral_pct']}% / "
                f"sell {r['sell_pct']:.2f}% (layout: {r['layout']}). Read from {src}.{extra}{warn}")
    if d.get("malformed_html"):
        return ("NOT READABLE: the page markup is too malformed to parse safely "
                f"({d['malformed_html']}), for example a truncated download or thousands of unclosed "
                "tags. Nothing was guessed. Try again, or save the page again from your browser.")
    in_script = sum(d["label_counts_script_data"].values()) + d["heading_in_script_data"]
    in_text = sum(d["label_counts_visible"].values()) + d["heading_in_visible_text"]
    if d["heading_in_visible_text"] == 0 and in_script > 0 and in_text == 0:
        return ("NOT READABLE: the poll wording appears ONLY inside script data (embedded JSON), "
                "not in the visible page. The parser deliberately does not read script data, so "
                "it would need a different, verified adapter. Save the page (--save) and share it.")
    if d["heading_in_visible_text"] == 0:
        return ("NOT READABLE: the 「みんなの評価」 heading is not in the visible page text. The page "
                "may be a login/consent/blocking page, or Yahoo! changed its layout. "
                f"Parser reason: {r['reason']}.")
    return (f"HEADING FOUND but the poll did not validate: status={r['status']}, "
            f"reason={r['reason']}. See the scope text below; this is the case to share.")


def run_record(d: dict, ticker: str | None, meta: dict | None) -> dict:
    r = d["result"]
    return {"ticker": ticker, "fetched_at": (meta or {}).get("fetched_at"),
            "http_status": (meta or {}).get("http_status"), "page_bytes": (meta or {}).get("page_bytes"),
            "transport": (meta or {}).get("transport"), "parser_version": r.get("parser_version"),
            "status": r["status"], "reason": r["reason"], "buy_pct": r.get("buy_pct"),
            "neutral_pct": r.get("neutral_pct"), "sell_pct": r.get("sell_pct"),
            "summary_buy_pct": r.get("summary_buy_pct"), "summary_sell_pct": r.get("summary_sell_pct"),
            "poll_source": d.get("poll_source"), "source_disagreement": d.get("source_disagreement")}


def write_poll_fixture(html: str, path: str, original: dict) -> list[str]:
    """Write only the poll block(s) as a tiny page and report whether it reproduces the
    original parse. Returns human-readable lines."""
    slices = poll_scope_html(html)
    if not slices:
        return ["POLL FIXTURE: not written — no 「みんなの評価」 block was found on the page."]
    fixture = "<html><body>" + "".join(f"<section>{s}</section>" for s in slices) + "</body></html>"
    with open(path, "w", encoding="utf-8") as f:
        f.write(fixture)
    keys = ("status", "reason", "buy_pct", "neutral_pct", "sell_pct", "summary_buy_pct", "summary_sell_pct")
    again = describe_poll_page(fixture)["result"]
    same = all(again.get(k) == original["result"].get(k) for k in keys)
    out = [f"POLL FIXTURE: wrote {path} ({len(fixture):,} characters, {len(slices)} block(s)).",
           "  round-trip check: " + ("the fixture reproduces the original result."
                                     if same else "DIFFERS from the original result — do not rely on this fixture.")]
    if _SECRET_HINTS.search(fixture):
        out.append("  WARNING: the fixture contains text that looks like a secret or an e-mail address. "
                   "Open it and remove it before sharing.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ticker", nargs="?", help="a Tokyo ticker such as 7203.T")
    ap.add_argument("--file", help="analyse a saved HTML file instead of fetching")
    ap.add_argument("--save", help="write the fetched raw HTML to this file")
    ap.add_argument("--record", help="write a small JSON run record to this file")
    ap.add_argument("--poll-html", help="write only the poll block as a small HTML fixture")
    args = ap.parse_args(argv)
    if bool(args.file) == bool(args.ticker):
        ap.error("give either a ticker or --file")
    meta = None
    if args.file:
        html = open(args.file, encoding="utf-8", errors="ignore").read()
    else:
        if not args.ticker.upper().endswith(".T"):
            ap.error("the Yahoo! Japan board only covers Tokyo listings (ticker ending .T)")
        try:
            html, meta = fetch_page(args.ticker.upper())
        except Exception as e:                       # network, TLS, HTTP: say so, never guess
            reason = getattr(e, "reason", None)
            code = getattr(e, "code", None)
            print(f"FETCH FAILED ({type(e).__name__}"
                  f"{': ' + str(code) if code else ''}{', ' + str(reason) if isinstance(reason, str) else ''}): {e}")
            print("For TLS errors, set STOCKREC_CA_BUNDLE to your trusted CA file. Verification stays enabled.")
            print("HTTP 403/429: stop; no bypass is attempted. A browser-saved public HTML file can be analysed with --file.")
            return 2
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"saved raw page to {args.save} ({len(html):,} characters)")
    d = describe_poll_page(html)
    print("\n" + verdict(d) + "\n")
    if args.record:
        with open(args.record, "w", encoding="utf-8") as f:
            json.dump(run_record(d, args.ticker.upper() if args.ticker else None, meta), f,
                      ensure_ascii=False, indent=2)
        print(f"run record written to {args.record}")
    if args.poll_html:
        for line in write_poll_fixture(html, args.poll_html, d):
            print(line)
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
