"""Diagnose why the Yahoo! Japan board poll (みんなの評価) is or is not being read.

Run from the app folder (it needs sentiment_sources.py and indicators.py beside it):

    python jp_board_diagnose.py 7203.T                 # fetch the page and explain it
    python jp_board_diagnose.py 7203.T --save page.html   # also keep the raw HTML
    python jp_board_diagnose.py --file page.html       # analyse a page you already saved

It tells you, in plain words, whether the parser can read the poll, and if not, which of
the usual causes applies: the heading is missing (layout change, login/consent page,
blocking), the poll numbers sit only inside script data (JSON) that the parser does not
read, or the poll is present but malformed. Nothing here changes the app or its data.

Personal diagnostic use only. The app reads this unofficial page at personal-use level and
stops on blocking; Yahoo!'s terms apply to you. A saved page can contain data you may not
want to share, so look at it before sending it to anyone. No credentials are involved.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.request

from sentiment_sources import describe_poll_page

URL = "https://finance.yahoo.co.jp/quote/{ticker}/forum"


def fetch(ticker: str, timeout: float = 15.0) -> str:
    from jp_board_transport import fetch_board_page
    return fetch_board_page(ticker, timeout=timeout)[0]

def verdict(d: dict) -> str:
    r = d["result"]
    if r["status"] == "ok":
        extra = ""
        if r.get("summary_buy_pct") is not None or r.get("summary_sell_pct") is not None:
            extra = (f" A rounded headline summary (buy {r.get('summary_buy_pct')}%, "
                     f"sell {r.get('summary_sell_pct')}%) was found and agrees with the categories.")
        return (f"READABLE. Buy {r['buy_pct']:.2f}% / watch {r['neutral_pct']}% / "
                f"sell {r['sell_pct']:.2f}% (layout: {r['layout']}).{extra}")
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ticker", nargs="?", help="a Tokyo ticker such as 7203.T")
    ap.add_argument("--file", help="analyse a saved HTML file instead of fetching")
    ap.add_argument("--save", help="write the fetched raw HTML to this file")
    args = ap.parse_args(argv)
    if bool(args.file) == bool(args.ticker):
        ap.error("give either a ticker or --file")
    if args.file:
        html = open(args.file, encoding="utf-8", errors="ignore").read()
    else:
        if not args.ticker.upper().endswith(".T"):
            ap.error("the Yahoo! Japan board only covers Tokyo listings (ticker ending .T)")
        try:
            html = fetch(args.ticker.upper())
        except Exception as e:                       # network, TLS, HTTP: say so, never guess
            print(f"FETCH FAILED ({type(e).__name__}): {e}")
            print("For TLS errors, set STOCKREC_CA_BUNDLE to your trusted CA file. Verification stays enabled.")
            print("HTTP 403/429: stop; no bypass is attempted. A browser-saved public HTML file can be analysed with --file.")
            return 2
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"saved raw page to {args.save} ({len(html):,} characters)")
    d = describe_poll_page(html)
    print("\n" + verdict(d) + "\n")
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
