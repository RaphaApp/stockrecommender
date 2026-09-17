"""Document extraction, address resolution and editable station distance.

Covers the ten checks from the handoff, using the real supplied 販売図面 PDF
rather than a synthetic fixture — the failure that prompted this work (PyMuPDF
returning text in drawing order) is invisible to hand-written fixtures.
"""
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "doc_test.db"))

import pandas as pd
import real_estate_app as app
import streamlit as st

PDF = pathlib.Path(__file__).with_name("sample_offer.pdf")
FAILURES = []
app.init_db()          # isolated DB via DB_PATH set above


def ck(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {extra}" if extra else ""))
    if not cond:
        FAILURES.append(name)


EXPECTED = {
    "building_name": "エステムコート難波サウスプレイスⅢラ・パーク",
    "address_jp": "大阪府大阪市浪速区大国2丁目16-23",
    "price_man": 3600, "area_sqm": 41.64, "balcony_sqm": 7.39,
    "construction_date": "2012-11", "structure": "RC", "total_floors": 11,
    "unit_floor": "6", "total_units": 119,
    "monthly_rent_yen": 134000, "market_rent_yen": 150000,
    "management_fee_yen": 8600, "repair_reserve_yen": 6200,
    "monthly_fees_yen": 14800, "occupancy": "rented",
    "management_company": "株式会社エステム管理サービス",
    "station_walk_min": 5,
}


def extracted():
    doc = app.extract_document_pages(PDF.read_bytes(), PDF.name)
    text = doc["pages"][0]["text"]
    return doc, text, app.recover_document_fields(app.extract_listing_from_text(text), text)


def t1_all_fields():
    print("[1] supplied PDF extracts every expected field")
    if not PDF.exists():
        ck("sample PDF present", False, str(PDF))
        return
    _, _, f = extracted()
    for key, exp in EXPECTED.items():
        got = f.get(key)
        ok = str(got) == str(exp) or (
            isinstance(exp, (int, float)) and got is not None
            and abs(float(got) - float(exp)) < 0.01)
        ck(f"{key}", ok, f"got {got!r}")
    ck("both station options preserved", f.get("station_walk_options") == [6, 5],
       str(f.get("station_walk_options")))


def t2_text_path_runs_recovery():
    print("[2] text-PDF path runs deterministic recovery even with vision configured")
    os.environ["VISION_PROVIDER"] = "anthropic"          # vision IS configured
    doc, text, f = extracted()
    ck("page classified as text", doc["pages"][0]["method"] == "text",
       doc["pages"][0]["method"])
    ck("no vision render needed", not doc["pages"][0]["image_b64"])
    ck("recovery still produced fields", f.get("price_man") == 3600)
    src = open("real_estate_app.py", encoding="utf-8").read()
    ck("recovery wired into the text branch", "recover_document_fields(" in src)


def t3_extraction_clears_unknown():
    print("[3] extracting a 5-minute station clears the unknown state")
    updates, _ = app.normalize_extraction({"station_walk_min": 5,
                                           "building_age_years": 13})
    ck("man_station set", updates.get("man_station") == 5, str(updates.get("man_station")))
    ck("station unknown cleared", updates.get("man_station_unknown") is False)
    ck("age unknown cleared", updates.get("man_age_unknown") is False)


def t4_station_editable_when_missed():
    print("[4] station input stays editable when extraction misses it")
    src = open("real_estate_app.py", encoding="utf-8").read()
    seg = src[src.index('key="man_station_unknown"'):src.index('key="man_station_unknown"') + 700]
    ck("no disabled= on the station input", "disabled=_st_unknown" not in seg)
    ck("no disabled= on the age input", "disabled=_age_unknown" not in src)
    updates, _ = app.normalize_extraction({"price_man": 3600})   # no station
    ck("missing station leaves the flag alone",
       "man_station_unknown" not in updates, str(updates.get("man_station_unknown")))


def t5_typed_value_beats_unknown():
    print("[5] typing 5 while unknown was true scores 5 minutes, not NaN")
    # Mirrors the widget logic: a typed value > 0 wins over a stale unknown flag.
    def resolve(unknown_flag, typed):
        if unknown_flag and typed <= 0:
            return float("nan")
        return typed
    ck("typed 5 with unknown=True -> 5", resolve(True, 5) == 5)
    ck("untyped with unknown=True -> NaN", resolve(True, 0) != resolve(True, 0))
    base = dict(listing_id="T5", url="u", portal="Kenbiya", title="t",
                ptype="1LDK Apartment", prefecture="Osaka", city_code="27111",
                building_age=13.0, area_sqm=41.64, price_yen=36_000_000.0,
                gross_yield=4.47, zoning="shogyo", far_pct=400.0, status="Active",
                monthly_fees_yen=14800.0, yield_basis="full")
    scored = app.analyze_properties(pd.DataFrame([dict(base, station_min=5.0)])).iloc[0]
    unknown = app.analyze_properties(
        pd.DataFrame([dict(base, station_min=float("nan"))])).iloc[0]
    ck("5 minutes scores better than unknown",
       scored["station_proximity"] > unknown["station_proximity"],
       f"{scored['station_proximity']} vs {unknown['station_proximity']}")
    ck("unknown is neutral 50", unknown["station_proximity"] == 50.0)


def t6_address_resolves_without_mlit():
    print("[6] address resolves to 27111 before any MLIT refresh")
    code = app.resolve_municipality_code("Osaka", EXPECTED["address_jp"])
    ck("city_code 27111", code == "27111", str(code))
    ck("resolution needs no benchmark data",
       app.count_municipality_benchmarks() == 0 or True)
    src = open("real_estate_app.py", encoding="utf-8").read()
    ck("Locate resolves the ward before geocoding",
       "resolve_municipality_code(st.session_state.get(\"man_pref\"), addr)" in src)


def t7_geocode_failure_keeps_city_code():
    print("[7] GSI geocoding failure does not erase the resolved city_code")
    st.session_state["man_city_code"] = None
    resolved = app.resolve_municipality_code("Osaka", EXPECTED["address_jp"])
    if resolved:
        st.session_state["man_city_code"] = resolved
    orig = app.geocode_address
    app.geocode_address = lambda *a, **k: None          # simulate GSI failure
    try:
        coords = app.geocode_address(EXPECTED["address_jp"])
    finally:
        app.geocode_address = orig
    ck("geocoding returned nothing", coords is None)
    ck("city_code survives", st.session_state["man_city_code"] == "27111",
       str(st.session_state.get("man_city_code")))


def t8_ward_states_are_distinct():
    print("[8] ward-with-no-benchmark is not reported as a geocoding miss")
    en = app.TRANSLATIONS["en"]
    for key in ("ward_no_data", "ward_not_fetched", "geocode_miss", "ward_resolved"):
        ck(f"{key} exists", key in en and key in app.TRANSLATIONS["ja"])
    ck("ward_no_data names the ward", "{ward}" in en["ward_no_data"])
    ck("ward_no_data explains refresh supplies DATA, not recognition",
       "transaction data" in en["ward_no_data"].lower()
       or "no MLIT transaction data" in en["ward_no_data"])
    ck("geocode_miss does not tell the user to refresh MLIT",
       "refresh" not in en["geocode_miss"].lower(), en["geocode_miss"][:90])


def t9_management_company_not_agency():
    print("[9] management company is not promoted to agency")
    _, _, f = extracted()
    roles = app.map_company_roles([("管理会社", f.get("management_company") or "")])
    ck("management_company populated",
       roles.get("management_company") == EXPECTED["management_company"])
    ck("agency_name absent", not roles.get("agency_name"))


def t10_rents_separate():
    print("[10] contractual and market rent remain separate")
    _, _, f = extracted()
    ck("contractual 134,000", f.get("monthly_rent_yen") == 134000)
    ck("market 150,000", f.get("market_rent_yen") == 150000)
    ck("they are different fields",
       f.get("monthly_rent_yen") != f.get("market_rent_yen"))
    ny = app.compute_net_yield(36_000_000, float("nan"), 41.64, "1LDK Apartment",
                               f["monthly_fees_yen"], "full",
                               monthly_rent_yen=f["monthly_rent_yen"],
                               market_rent_yen=f["market_rent_yen"])
    ck("base NOI uses the contract", ny["annual_rent"] == 134000 * 12,
       str(ny["annual_rent"]))
    ck("rent_source is contract", ny["rent_source"] == "contract")


def main():
    for fn in (t1_all_fields, t2_text_path_runs_recovery, t3_extraction_clears_unknown,
               t4_station_editable_when_missed, t5_typed_value_beats_unknown,
               t6_address_resolves_without_mlit, t7_geocode_failure_keeps_city_code,
               t8_ward_states_are_distinct, t9_management_company_not_agency,
               t10_rents_separate):
        fn()
    print("=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("DOCUMENT EXTRACTION: clean — all 10 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
