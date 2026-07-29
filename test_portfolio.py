"""Unit tests for portfolio.py — Rakuten trade-history parsing.

Synthetic cp932 CSVs modelled on the real exports: no network, no Streamlit.
Run: pytest -q
"""
import io
import math

import pandas as pd
import pytest

from portfolio import (detect_kind, to_num, jp_symbol, normalize_trades, combine_trades,
                       recent_sells, net_positions, search_options, option_to_symbol,
                       trades_for, summary, SIDE_BUY, SIDE_SELL, CANON_COLS)

US_CSV = (
    "約定日,受渡日,ティッカー,銘柄名,口座,取引区分,売買区分,信用区分,弁済期限,決済通貨,"
    "数量［株］,単価［USドル］,約定代金［USドル］,為替レート,手数料［USドル］,税金［USドル］,"
    "受渡金額［USドル］,受渡金額［円］\n"
    '"2026/6/18","2026/6/22","BAC","BANK OF AMERICA","特定","現物","売付","-","-","米ドル",'
    '"15","57.2000","858.00","160.470","0.00","0.00","853.74","-"\n'
    '"2026/7/9","2026/7/11","JPM","JP MORGAN CHASE","特定","現物","買付","-","-","円",'
    '"2","334.1000","668.20","163.540","0.00","0.00","-","109,278.00"\n'
)

JP_CSV = (
    "約定日,受渡日,銘柄コード,銘柄名,市場名称,口座区分,取引区分,売買区分,信用区分,弁済期限,"
    "数量［株］,単価［円］,手数料［円］,税金等［円］,諸費用［円］,税区分,受渡金額［円］\n"
    '"2023/1/4","2023/1/6","3382","セブン＆アイ・ＨＬＤＧＳ","東証","旧NISA","現物","買付",'
    '"-","-","100","5,610.0","0","0","0","-","561,000"\n'
    '"2024/2/27","2024/3/1","3382","セブン＆アイ・ＨＬＤＧＳ","東証","旧NISA","","入庫",'
    '"-","-","200","1,870.0","0","0","0","-","-"\n'
)


def _read(text):
    return pd.read_csv(io.BytesIO(text.encode("cp932")), encoding="cp932")


# ------------------------------------------------------------------ detection
def test_detect_us_and_jp():
    assert detect_kind(_read(US_CSV)) == "US"
    assert detect_kind(_read(JP_CSV)) == "JP"


def test_detect_unknown_layout():
    other = pd.DataFrame({"日付": ["2026/1/1"], "配当金": [100]})
    assert detect_kind(other) is None
    # and an unknown frame normalises to an empty canonical frame, not an exception
    out = normalize_trades(other)
    assert out.empty and list(out.columns) == CANON_COLS


# -------------------------------------------------------------- number parsing
def test_to_num_handles_commas_and_dashes():
    s = pd.Series(["1,010.85", "-", "561,000", "", None])
    out = to_num(s)
    assert out.iloc[0] == pytest.approx(1010.85)
    assert out.iloc[1] != out.iloc[1]          # '-' -> NaN, not 0
    assert out.iloc[2] == pytest.approx(561000)
    assert out.isna().sum() == 3


def test_jp_symbol_mapping():
    assert jp_symbol(3382) == "3382.T"
    assert jp_symbol("3382") == "3382.T"
    assert jp_symbol("3382.T") == "3382.T"     # idempotent


# ------------------------------------------------------------------ US parsing
def test_us_normalise_and_jpy_derivation():
    df = normalize_trades(_read(US_CSV))
    assert list(df.columns) == CANON_COLS
    sell = df[df["side"] == SIDE_SELL].iloc[0]
    buy = df[df["side"] == SIDE_BUY].iloc[0]
    # USD value always comes from 約定代金［USドル］
    assert sell["value_usd"] == pytest.approx(858.00)
    # 受渡金額［円］ is '-' on the USD-settled sell -> derived from USD x FX, flagged
    assert sell["value_jpy"] == pytest.approx(858.00 * 160.470)
    assert bool(sell["jpy_estimated"]) is True
    # the JPY-settled buy uses the reported figure and is NOT flagged
    assert buy["value_jpy"] == pytest.approx(109278.0)
    assert bool(buy["jpy_estimated"]) is False
    assert sell["symbol"] == "BAC" and sell["market"] == "US"


# ------------------------------------------------------------------ JP parsing
def test_jp_normalise_symbol_and_no_usd():
    df = normalize_trades(_read(JP_CSV))
    assert set(df["symbol"]) == {"3382.T"}
    assert df["value_usd"].isna().all()        # no FX in the JP export -> never invented
    assert not df["jpy_estimated"].any()
    # the 入庫 (transfer-in) row has 受渡金額 '-' -> NaN, not zero
    transfer = df[df["side"] == "入庫"].iloc[0]
    assert math.isnan(transfer["value_jpy"])


# ------------------------------------------------------------------- combining
def test_combine_dedupes_and_sorts_newest_first():
    us, jp = normalize_trades(_read(US_CSV)), normalize_trades(_read(JP_CSV))
    combined = combine_trades([us, jp, us])    # duplicate upload of the same file
    assert len(combined) == len(us) + len(jp)  # dupes dropped
    assert combined["date"].is_monotonic_decreasing


def test_combine_empty_is_safe():
    out = combine_trades([])
    assert out.empty and list(out.columns) == CANON_COLS


# ---------------------------------------------------------------- review queue
def test_recent_sells_filters_sells_only():
    combined = combine_trades([normalize_trades(_read(US_CSV)),
                               normalize_trades(_read(JP_CSV))])
    sells = recent_sells(combined)
    assert len(sells) == 1 and sells.iloc[0]["symbol"] == "BAC"


def test_recent_sells_empty_when_buy_only():
    # the real JP export in this account has zero sells — must be a clean empty frame
    assert recent_sells(normalize_trades(_read(JP_CSV))).empty
    assert recent_sells(pd.DataFrame()).empty


# --------------------------------------------------------------------- search
def test_search_options_and_reverse_lookup():
    combined = combine_trades([normalize_trades(_read(US_CSV)),
                               normalize_trades(_read(JP_CSV))])
    opts = search_options(combined)
    assert "BAC - BANK OF AMERICA" in opts
    assert any(o.startswith("3382 - ") for o in opts)
    assert len(opts) == len(set(opts))                     # deduped
    assert option_to_symbol("BAC - BANK OF AMERICA", combined) == "BAC"
    assert option_to_symbol([o for o in opts if o.startswith("3382")][0],
                            combined) == "3382.T"
    assert option_to_symbol("", combined) is None


def test_trades_for_symbol():
    combined = combine_trades([normalize_trades(_read(US_CSV)),
                               normalize_trades(_read(JP_CSV))])
    assert len(trades_for(combined, "3382.T")) == 2
    assert trades_for(combined, "NOPE").empty


# -------------------------------------------------------------------- summary
def test_summary_currency_switch_and_missing_count():
    combined = combine_trades([normalize_trades(_read(US_CSV)),
                               normalize_trades(_read(JP_CSV))])
    jpy = summary(combined, "JPY")
    usd = summary(combined, "USD")
    assert jpy["trades"] == usd["trades"] == 4
    # JP rows carry no USD figure -> counted as missing, not silently zeroed
    assert usd["missing"] >= 2
    assert usd["sell_value"] == pytest.approx(858.00)
    assert jpy["buy_value"] == pytest.approx(109278.0 + 561000.0)


def test_net_positions_signs_buys_and_sells():
    combined = combine_trades([normalize_trades(_read(US_CSV)),
                               normalize_trades(_read(JP_CSV))])
    pos = net_positions(combined).set_index("symbol")
    assert pos.loc["BAC", "net_qty"] == -15.0     # sold, no offsetting buy in this file
    assert pos.loc["JPM", "net_qty"] == 2.0
    # 買付 100 + 入庫 200: transferred-in shares are held, so the net is 300 —
    # they are neither ignored nor (worse) counted as a sale.
    assert pos.loc["3382.T", "net_qty"] == 300.0


# ------------------------------------------------------------- share-count signs
def test_side_sign_treats_transfers_correctly():
    from portfolio import side_sign
    assert side_sign("買付") == 1.0
    assert side_sign("売付") == -1.0
    assert side_sign("入庫") == 1.0          # transferred IN increases the holding…
    assert side_sign("入庫（分割）") == 1.0
    assert side_sign("出庫（買収）") == -1.0  # …and OUT decreases it
    assert side_sign("") == 0.0 and side_sign(None) == 0.0


def test_net_qty_transfer_in_is_not_a_sale():
    # regression: 買付 100 + 入庫 200 must be +300, never -100
    from portfolio import net_qty
    df = combine_trades([normalize_trades(_read(JP_CSV))])
    assert net_qty(df, "3382.T") == pytest.approx(300.0)
    assert net_qty(df, "NOPE") == 0.0


# ------------------------------------------------------- holdings vs closed vs gaps
ROUNDTRIP_CSV = (
    "約定日,受渡日,ティッカー,銘柄名,口座,取引区分,売買区分,信用区分,弁済期限,決済通貨,"
    "数量［株］,単価［USドル］,約定代金［USドル］,為替レート,手数料［USドル］,税金［USドル］,"
    "受渡金額［USドル］,受渡金額［円］\n"
    '"2025/1/10","2025/1/14","XYZ","XYZ CORP","特定","現物","買付","-","-","米ドル",'
    '"10","100.0000","1,000.00","150.000","0.00","0.00","1,000.00","-"\n'
    '"2025/6/10","2025/6/12","XYZ","XYZ CORP","特定","現物","売付","-","-","米ドル",'
    '"10","120.0000","1,200.00","155.000","0.00","0.00","1,200.00","-"\n'
    '"2025/2/10","2025/2/12","HOLD","HOLD CORP","特定","現物","買付","-","-","米ドル",'
    '"4","50.0000","200.00","150.000","0.00","0.00","200.00","-"\n'
    '"2025/3/10","2025/3/12","HOLD","HOLD CORP","特定","現物","買付","-","-","米ドル",'
    '"6","100.0000","600.00","150.000","0.00","0.00","600.00","-"\n'
)


def _combined():
    from portfolio import normalize_trades
    return combine_trades([normalize_trades(_read(US_CSV)),
                           normalize_trades(_read(JP_CSV)),
                           normalize_trades(_read(ROUNDTRIP_CSV))])


def test_holdings_only_open_positions():
    from portfolio import holdings
    h = holdings(_combined()).set_index("symbol")
    # open: JPM (bought 2), 3382.T (100 bought + 200 transferred in), HOLD (4+6)
    assert set(h.index) == {"JPM", "3382.T", "HOLD"}
    assert h.loc["3382.T", "net_qty"] == 300.0
    # XYZ was bought AND fully sold -> not a holding
    assert "XYZ" not in h.index
    # BAC has a sell with no matching buy in the file -> negative, not a holding
    assert "BAC" not in h.index


def test_holdings_avg_buy_is_qty_weighted():
    from portfolio import holdings
    h = holdings(_combined()).set_index("symbol")
    # HOLD: 4 @ $50 + 6 @ $100 -> (200+600)/10 = $80, not the $75 simple mean
    assert h.loc["HOLD", "avg_buy_price"] == pytest.approx(80.0)


def test_closed_positions_are_the_reentry_list():
    from portfolio import closed_positions
    c = closed_positions(_combined())
    assert list(c["symbol"]) == ["XYZ"]        # net 0 AND had a sell
    assert pd.notna(c.iloc[0]["exited"])


def test_incomplete_positions_flags_negative_net():
    from portfolio import incomplete_positions
    inc = incomplete_positions(_combined())
    assert list(inc["symbol"]) == ["BAC"] and inc.iloc[0]["net_qty"] == -15.0


def test_holdings_empty_frame_is_safe():
    from portfolio import holdings, closed_positions, incomplete_positions
    empty = pd.DataFrame()
    assert holdings(empty).empty
    assert closed_positions(empty).empty
    assert incomplete_positions(empty).empty


# ------------------------------------------------------- instrument index / search
def test_instrument_index_merges_sources_without_overwriting_base():
    from portfolio import instrument_index, normalize_trades
    trades = normalize_trades(_read(JP_CSV))          # JP names are Japanese
    idx = instrument_index({"AAPL": "Apple"}, {"AAPL": "アップル"},
                           trades=trades, results=[{"ticker": "MSFT", "name": "Microsoft"}])
    assert idx["AAPL"] == {"en": "Apple", "ja": "アップル"}   # curated base intact
    assert idx["3382.T"]["ja"].startswith("セブン")            # from the JP CSV
    assert idx["MSFT"]["en"] == "Microsoft"                    # from scan results
    # a US CSV name fills the ENGLISH side, not the Japanese one
    idx2 = instrument_index({}, {}, trades=normalize_trades(_read(US_CSV)))
    assert idx2["BAC"]["en"] == "BANK OF AMERICA" and idx2["BAC"]["ja"] == ""


def test_instrument_labels_and_reverse():
    from portfolio import instrument_labels, label_to_ticker
    idx = {"AAPL": {"en": "Apple", "ja": "アップル"},
           "GILD": {"en": "Gilead", "ja": ""},
           "9999.T": {"en": "", "ja": "テスト社"}}
    labels = instrument_labels(idx)
    assert "AAPL — Apple / アップル" in labels
    assert "GILD — Gilead" in labels                  # no JA -> English only
    assert "9999.T — テスト社" in labels               # no EN -> Japanese only
    for l in labels:                                   # round-trips for every label
        assert label_to_ticker(l) in idx
    assert label_to_ticker("") is None


def test_instrument_labels_searchable_by_all_three_fields():
    # this is what makes Streamlit's built-in filtering act as tri-lingual autocomplete
    from portfolio import instrument_labels
    labels = instrument_labels({"NVDA": {"en": "Nvidia", "ja": "エヌビディア"}})
    for query in ("NVDA", "nvidia", "エヌビ"):
        assert any(query.lower() in l.lower() for l in labels), query
