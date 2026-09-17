"""Select All / Copy ingestion tests (Update A)."""
import os, sys, tempfile
os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "paste.db"))
import real_estate_app as app
app.init_db()

FAILURES = []
def ck(n, c, e=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f" — {e}" if e else ""))
    if not c: FAILURES.append(n)

FIXTURE = """区分マンション
新丸子駅徒歩2分　オーナーチェンジ物件　融資相談可能！
販売価格 1160万円
表面利回り 5.58%
想定年間収入 648,000円 (54,000円/月)
所在地 神奈川県川崎市中原区新丸子町
交通 東急東横線 新丸子駅 徒歩2分
建物構造 SRC造
築年月 1990年07月（築37年）
土地権利 所有権
専有面積 16.76㎡
バルコニー面積 2.84㎡
間取り 1R
方角 東向き
階数 3階／8階建て
総戸数 80戸
管理費（月額） 9,000円
修繕積立金（月額） 1,000円
管理形態 自主管理 (管理員なし)
用途地域 商業地域
取引態様 仲介
現況 賃貸中
引渡可能年月 相談
管理番号 RR-01377KM
次回更新予定日 2026/10/10
更新日 2026/09/10
情報登録日 2026/09/10"""

EXPECTED = {
    "portal_guess": "Rakumachi", "ingestion_channel": "pasted_text",
    "listing_title": "新丸子駅徒歩2分　オーナーチェンジ物件　融資相談可能！",
    "building_name": None, "address_jp": "神奈川県川崎市中原区新丸子町",
    "prefecture_jp": "神奈川県", "city_jp": "川崎市中原区", "station_walk_min": 2,
    "price_man": 1160, "gross_yield_pct": 5.58,
    "expected_annual_income_yen": 648000, "market_rent_yen": 54000,
    "monthly_rent_yen": None, "structure": "SRC", "construction_date": "1990-07",
    "land_right": "ownership", "area_sqm": 16.76, "balcony_sqm": 2.84,
    "layout": "1R", "orientation": "east", "unit_floor": 3, "total_floors": 8,
    "total_units": 80, "management_fee_yen": 9000, "repair_reserve_yen": 1000,
    "monthly_fees_yen": 10000, "management_type": "self_managed",
    "agency_role": "intermediary", "agency_name": None, "occupancy": "rented",
    "handover": "consultation", "management_number": "RR-01377KM",
    "registered_at": "2026-09-10", "updated_at": "2026-09-10",
    "next_update_at": "2026-10-10",
}


def main():
    f = app.parse_pasted_listing(FIXTURE)

    print("[1] Rakumachi fixture extracts all expected values")
    for k, e in EXPECTED.items():
        g = f.get(k)
        ok = (g == e) or (isinstance(e, (int, float)) and g is not None
                          and abs(float(g) - float(e)) < 0.01)
        ck(k, ok, f"got {g!r}")

    print("[2] marketing headline does not become building name")
    ck("building_name is None", f.get("building_name") is None)
    ck("headline stored as listing_title",
       f.get("listing_title") == EXPECTED["listing_title"])

    print("[3] 想定年間収入 does not become contractual rent")
    ck("monthly_rent_yen stays None", f.get("monthly_rent_yen") is None)
    ck("assumption goes to market_rent_yen", f.get("market_rent_yen") == 54000)
    warns = {w["key"] for w in app.pasted_listing_warnings(f)}
    ck("assumed-rent warning raised", "paste_warn_assumed_rent" in warns, str(warns))
    ny = app.compute_net_yield(11_600_000, 5.58, 16.76, "1R Mansion", 10000, "full",
                               monthly_rent_yen=None, market_rent_yen=54000)
    ck("base NOI never uses the assumption", ny["rent_source"] == "gross_yield")

    print("[4] annual/monthly income consistency")
    ck("648,000 == 54,000 x 12", 648000 == 54000 * 12)
    ck("no mismatch warning for the fixture",
       "paste_warn_income_mismatch" not in warns)
    bad = app.parse_pasted_listing("想定年間収入 700,000円 (54,000円/月)\n販売価格 1160万円\n専有面積 16.76㎡")
    ck("inconsistent pair warns",
       "paste_warn_income_mismatch" in {w["key"] for w in app.pasted_listing_warnings(bad)})

    print("[5] station 2 minutes clears the unknown state")
    updates, _ = app.normalize_extraction(f)
    ck("man_station = 2", updates.get("man_station") == 2, str(updates.get("man_station")))
    ck("unknown cleared", updates.get("man_station_unknown") is False)

    print("[6] manual station stays editable after an extraction miss")
    src = open("real_estate_app.py", encoding="utf-8").read()
    ck("no disabled= on station", "disabled=_st_unknown" not in src)
    ck("no disabled= on age", "disabled=_age_unknown" not in src)
    miss, _ = app.normalize_extraction({"price_man": 1160})
    ck("missing station leaves the flag alone", "man_station_unknown" not in miss)

    print("[7] municipality resolves to 14133 without MLIT data")
    ck("city_code 14133",
       app.resolve_municipality_code("Kanagawa", f["address_jp"]) == "14133")
    ck("no benchmarks needed", app.count_municipality_benchmarks() == 0)

    print("[8] extraction prefills and does not score automatically")
    seg = src[src.index('tr("paste_extract_btn")'):src.index('tr("paste_extract_btn")') + 1400]
    ck("no analyze_properties in the paste branch", "analyze_properties(" not in seg)
    ck("no save_manual_pick in the paste branch", "save_manual_pick(" not in seg)
    ck("writes only to session_state", "st.session_state[k] = v" in seg)

    print("[9] user corrections create manual_override provenance")
    prov = {k: {"extracted_value": v, "extraction_method": "pasted_text"}
            for k, v in f.items() if not isinstance(v, (list, dict))}
    ov = app.apply_manual_overrides(prov, {"price_man": 1200})
    ck("edited field stamped", ov["price_man"]["extraction_method"] == "manual_override")
    ck("original retained", ov["price_man"]["extracted_value"] == 1160)
    ck("untouched field keeps pasted_text",
       ov["area_sqm"]["extraction_method"] == "pasted_text")

    print("[10] pasted text is not persisted by default")
    ck("no column stores the page", "pasted_text_blob" not in src)
    seg2 = src[src.index('tr("paste_extract_btn")'):src.index('tr("paste_extract_btn")') + 1600]
    ck("provenance stores values, not the page", '"extracted_value": v' in seg2)
    ck("UI states it is not stored", "paste_not_stored" in src)

    print("=" * 58)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for x in FAILURES: print("  -", x)
        return 1
    print("PASTED LISTING: clean — all 10 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
