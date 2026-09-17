"""Full validation matrix required by CLAUDE_INSTRUCTIONS.md.

Run: python test_full_validation.py      (exit 0 = clean)
"""
import ast
import io
import json
import math
import os
import re
import sqlite3
import string
import sys
import types

import pandas as pd

# --- Streamlit/altair stubs so the module imports headlessly ---------------
_st = types.ModuleType("streamlit")
_st.session_state = {"lang": "en"}
_st.set_page_config = lambda *a, **k: None
_st.markdown = lambda *a, **k: None


class _Secrets(dict):
    pass


_st.secrets = _Secrets()
sys.modules.setdefault("streamlit", _st)
_alt = types.ModuleType("altair")


class _Any:
    def __getattr__(self, n):
        return self

    def __call__(self, *a, **k):
        return self


for _n in ("Chart", "X", "Y", "Color", "Row", "Tooltip", "Header", "Scale",
           "Axis", "condition", "datum", "value"):
    setattr(_alt, _n, _Any())
sys.modules.setdefault("altair", _alt)

import real_estate_app as app  # noqa: E402

FAILURES = []


def ck(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
        print(f"  FAIL  {name}" + (f" — {extra}" if extra else ""))
    return cond


BASE = dict(listing_id="V", url="https://www.kenbiya.com/pp1/x/re_1/", portal="Kenbiya",
            title="t", ptype="1LDK Apartment", prefecture="Osaka", city_code="27111",
            station_min=5.0, building_age=13.0, area_sqm=41.64, price_yen=36_000_000.0,
            gross_yield=4.47, zoning="shogyo", far_pct=400.0, status="Active",
            monthly_fees_yen=14_800.0, yield_basis="full")


def fresh_db(path):
    app.DB_PATH = path
    if os.path.exists(path):
        os.remove(path)
    _st.session_state.pop("_db_ready", None)
    app._memo_invalidate()
    app.init_db()


def t_clean_schema():
    print("[1] clean SQLite schema creation")
    fresh_db("/tmp/v_clean.db")
    c = sqlite3.connect(app.DB_PATH)
    cols = {r[1] for r in c.execute("PRAGMA table_info(historical_picks)")}
    ck("v16 size/rent columns present",
       {"size_model_mode", "size_adjustment", "size_evidence", "size_breakdown",
        "rent_source", "contractual_gross_pct", "gross_gap_pp"} <= cols)
    ck("schema version matches the app",
       c.execute("SELECT value FROM app_meta WHERE key='schema_version'").fetchone()[0]
       == app.SCHEMA_VERSION, app.SCHEMA_VERSION)
    ck("v17 rental_observations table",
       "rental_observations" in {r[0] for r in c.execute(
           "SELECT name FROM sqlite_master WHERE type='table'")})
    ck("v17 source_url column", "source_url" in cols)


def t_migration_v14():
    print("[2] migration from a v14-era database")
    path = "/tmp/v_old.db"
    if os.path.exists(path):
        os.remove(path)
    oc = sqlite3.connect(path)
    oc.execute("""CREATE TABLE historical_picks (id INTEGER PRIMARY KEY AUTOINCREMENT,
        listing_id TEXT, url TEXT, portal TEXT, title TEXT, ptype TEXT, prefecture TEXT,
        station_min REAL, building_age REAL, area_sqm REAL, price_yen REAL,
        gross_yield REAL, score REAL, factor_snapshot TEXT, recommendation_date TEXT,
        current_status TEXT, current_price REAL, days_listed INTEGER, outcome INTEGER,
        evaluated INTEGER, manual INTEGER)""")
    oc.execute("INSERT INTO historical_picks (listing_id,url,portal,title,ptype,"
               "prefecture,station_min,building_age,area_sqm,price_yen,gross_yield,"
               "score,factor_snapshot,recommendation_date,current_status) VALUES "
               "('OLD','u','Kenbiya','t','1K Apartment','Tokyo',5,10,25,25000000,"
               "5.0,60,'{}','2026-01-01','pending')")
    oc.commit()
    oc.close()
    app.DB_PATH = path
    _st.session_state.pop("_db_ready", None)
    app._memo_invalidate()
    app.init_db()
    back = app.get_historical_picks()
    ck("legacy row survives", len(back) == 1 and back.iloc[0]["listing_id"] == "OLD")
    row = back.iloc[0]
    ck("legacy row: all new fields null",
       all(pd.isna(row[f]) or row[f] is None
           for f in ("size_model_mode", "size_breakdown", "rent_source",
                     "building_name", "agency_name")))
    ck("legacy row still labels", app.listing_label(row) not in ("", None))
    ck("legacy row source falls back", app.source_display(row) == "Kenbiya",
       app.source_display(row))


def t_schema_parity():
    print("[3] Supabase/Postgres schema parity")
    fresh_db("/tmp/v_par.db")
    c = sqlite3.connect(app.DB_PATH)
    sql = open("supabase_schema.sql", encoding="utf-8").read()
    ck("schema file matches app version", f"SCHEMA_VERSION {app.SCHEMA_VERSION}" in sql)
    bad = []
    for t in [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]:
        live = {r[1] for r in c.execute(f"PRAGMA table_info({t})")}
        blk = re.search(rf"CREATE TABLE IF NOT EXISTS {t} \((.*?)\n\);", sql, re.S)
        decl = ({ln.strip().split()[0] for ln in blk.group(1).splitlines()
                 if ln.strip() and not ln.strip().startswith("PRIMARY KEY")}
                if blk else set())
        if live - decl:
            bad.append(f"{t}:{sorted(live - decl)}")
    ck("every table+column declared", not bad, str(bad)[:160])


def t_size_modes():
    print("[4] all three size modes on the same property")
    fresh_db("/tmp/v_modes.db")
    got = {}
    for mode in app.SIZE_MODEL_MODES:
        _st.session_state["size_model_mode"] = mode
        sc = app.analyze_properties(pd.DataFrame([dict(BASE, area_sqm=15.0)]))
        got[mode] = (sc.iloc[0]["score"], sc.iloc[0]["size_adjustment"])
    ck("legacy applies -8", got["legacy_penalty"][1] == -8.0, str(got))
    ck("descriptive zero effect", got["descriptive_only"][1] == 0.0)
    ck("evidence != legacy", got["evidence_weighted"][1] != -8.0)
    ck("three modes differ",
       len({round(v[0], 2) for v in got.values()}) >= 2, str(got))
    _st.session_state["size_model_mode"] = "evidence_weighted"
    adjs = [app.analyze_properties(pd.DataFrame([dict(BASE, area_sqm=a)]))
            .iloc[0]["size_adjustment"] for a in (10, 20, 30, 45, 70)]
    ck("bounded to ±6", all(abs(a) <= app.SIZE_ADJ_CAP + 1e-9 for a in adjs), str(adjs))


def t_continuity():
    print("[5] size-prior continuity at the anchors")
    worst = 0.0
    for a in (15, 18, 22, 25, 30, 35, 40, 50):
        vals = [app.size_prior(a - 0.1), app.size_prior(a), app.size_prior(a + 0.1)]
        worst = max(worst, max(vals) - min(vals))
    ck("no discontinuity > 1.5", worst < 1.5, f"max spread {worst:.3f}")


def t_no_legacy_leak():
    print("[6] demographics cannot reapply the legacy deduction")
    src = open("real_estate_app.py", encoding="utf-8").read()
    ck("resale_delta gated to legacy mode",
       'if _mode == "legacy_penalty" else 0.0' in src)
    for band in ("severe", "mild", "flat", "growth", None):
        adj = app.resale_gate_adjustment({"band": band} if band else {})
        ck(f"resale gate bounded ({band})", abs(adj) <= app.RESALE_GATE_CAP + 1e-9)
    ck("unknown outlook = exactly 0", app.resale_gate_adjustment({}) == 0.0)
    ck("low_base outlook = exactly 0",
       app.resale_gate_adjustment({"band": "low_base"}) == 0.0)


def t_rent_basis():
    print("[7] contractual rent drives NOI")
    a = app.compute_net_yield(36e6, 4.47, 41.64, "1LDK Apartment", 14800, "full",
                              monthly_rent_yen=134000)
    b = app.compute_net_yield(36e6, 4.47, 41.64, "1LDK Apartment", 14800, "full")
    c = app.compute_net_yield(36e6, float("nan"), 41.64, "1LDK Apartment", 14800,
                              "full", monthly_rent_yen=134000, market_rent_yen=999999)
    d = app.compute_net_yield(36e6, 6.50, 41.64, "1LDK Apartment", 14800, "full",
                              monthly_rent_yen=134000)
    ck("contract wins", a["rent_source"] == "contract")
    ck("gross-yield fallback", b["rent_source"] == "gross_yield")
    ck("market rent excluded from base NOI", abs(c["annual_rent"] - 134000 * 12) < 1)
    ck("annual rent = 1,608,000", a["annual_rent"] == 1_608_000)
    ck("conflict warns", d["rent_conflict"])
    ck("no false conflict", not a["rent_conflict"])


def t_rent_evidence_honesty():
    print("[8] rental evidence is not overclaimed")
    ev = app._rent_evidence_from_row({"monthly_rent_yen": 134000, "area_sqm": 41.64})
    ck("subject lease is NOT achieved_local", ev["source"] != "achieved_local", str(ev))
    ck("subject lease is contract_verified", ev["contract_verified"] is True)
    res = app.size_resilience(41.64, "1LDK Apartment", city_code="27111",
                              building_age=13, rent_evidence=ev)
    ck("flagged locally unbenchmarked", res.get("rent_locally_benchmarked") is False)
    ck("weight limited without local data", res.get("rent_weight", 1.0) <= 0.5,
       str(res.get("rent_weight")))
    ck("no flat 80", res["rental"] != 80.0, str(res["rental"]))


def t_financing_isolation():
    print("[9] financing cannot alter asset score / fair value")
    fresh_db("/tmp/v_fin.db")
    _st.session_state["size_model_mode"] = "evidence_weighted"
    sc = app.analyze_properties(pd.DataFrame([BASE]))
    score = float(sc.iloc[0]["score"])
    noi = app.compute_net_yield(36e6, 4.47, 41.64, "1LDK Apartment", 14800, "full",
                                monthly_rent_yen=134000)["noi"]
    variants = [app.financing_analysis(36e6, noi, down=d, rate=r, years=y, target_coc=t)
                for d, r, y, t in ((0, 0.5, 35, 3), (20, 2.0, 25, 5), (100, 9.9, 5, 20))]
    ck("score unchanged by financing",
       float(app.analyze_properties(pd.DataFrame([BASE])).iloc[0]["score"]) == score)
    ck("financing produces distinct statuses",
       len({v["status"] for v in variants}) >= 1)
    src = open("real_estate_app.py", encoding="utf-8").read()
    ana = src[src.index("def render_manual_analyzer"):]
    ck("exactly one financing panel", ana.count("financing_analysis(") == 1)


def t_company_roles():
    print("[10] 管理会社 never becomes agency_name")
    r = app.map_company_roles([("管理会社", "株式会社エステム管理サービス")])
    ck("management_company set", r.get("management_company") == "株式会社エステム管理サービス")
    ck("agency_name absent", not r.get("agency_name"))
    r2 = app.map_company_roles([("", "株式会社ナゾ")])
    ck("unlabelled -> extracted_entities", r2["extracted_entities"][0]["role"] == "unknown")
    ck("unlabelled never becomes agency", not r2.get("agency_name"))


def t_documents():
    print("[11] document ingestion matrix")
    ok, err = app.pdf_available()
    if not ck("PyMuPDF available", ok, err[:60]):
        return
    import pymupdf
    d = pymupdf.open()
    pg = d.new_page()
    for i, ln in enumerate(["物件名 テストマンション", "販売価格 3,600万円",
                            "専有面積 41.64m2", "管理会社 株式会社エステム管理サービス",
                            "賃料 134,000円", "想定賃料 150,000円", "現況 賃貸中"]):
        pg.insert_text((50, 72 + i * 18), ln, fontname="china-s", fontsize=10)
    text_pdf = d.tobytes()
    d.close()
    res = app.extract_document_pages(text_pdf, "offer.pdf")
    ck("text PDF uses embedded text", res["method"] == "text")
    fields = app.extract_listing_from_text(res["pages"][0]["text"])
    ck("text PDF yields price", fields.get("price_man") == 3600)
    ck("contractual vs market rent separated",
       fields.get("monthly_rent_yen") == 134000 and fields.get("market_rent_yen") == 150000)

    d2 = pymupdf.open()
    d2.new_page().insert_text((72, 72), "Logo")
    scan = app.extract_document_pages(d2.tobytes(), "cover.pdf")
    d2.close()
    ck("sparse page rendered for vision", bool(scan["pages"][0]["image_b64"]))

    big = pymupdf.open()
    for _ in range(12):
        big.new_page().insert_text((72, 72), "p")
    capped = app.extract_document_pages(big.tobytes(), "many.pdf")
    big.close()
    ck("page cap enforced", len(capped["pages"]) == app.MAX_PDF_PAGES and capped["warnings"])

    for data, name, label in ((b"%PDF-1.4 broken", "bad.pdf", "corrupt"),
                              (b"x" * (21 * 1024 * 1024), "big.pdf", "oversize"),
                              (b"", "empty.pdf", "empty")):
        try:
            app.extract_document_pages(data, name)
            ck(f"{label} raises DocumentError", False)
        except app.DocumentError:
            ck(f"{label} raises DocumentError", True)

    enc = pymupdf.open()
    enc.new_page().insert_text((72, 72), "secret")
    enc_bytes = enc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                            owner_pw="o", user_pw="u")
    enc.close()
    try:
        app.extract_document_pages(enc_bytes, "enc.pdf")
        ck("encrypted PDF raises DocumentError", False)
    except app.DocumentError:
        ck("encrypted PDF raises DocumentError", True)

    one = app.extract_document_pages(b"\x89PNG\r\n\x1a\nfake", "a.png")
    ck("single image one page", one["method"] == "image" and len(one["pages"]) == 1)
    merged, conflicts, _prov = app.merge_extractions([
        {"page": 1, "fields": {"building_name": "A", "price_man": 3600}},
        {"page": 2, "fields": {"agency_name": "株式会社アルファ不動産"}},
        {"page": 3, "fields": {"price_man": 3900}}])
    ck("multi-page merge", merged.get("agency_name") == "株式会社アルファ不動産"
       and merged.get("building_name") == "A")
    ck("conflict surfaced", len(conflicts) == 1 and conflicts[0]["field"] == "price_man")


def t_persistence():
    print("[12] field-by-field persistence")
    fresh_db("/tmp/v_persist.db")
    _st.session_state["size_model_mode"] = "evidence_weighted"
    sc = app.analyze_properties(pd.DataFrame([BASE]))
    r = sc.iloc[0]
    payload = {**{k: r[k] for k in r.index}, "lat": 34.6455, "lon": 135.4960,
               "hazard_flags": "flood:rank2", "dev_flags": "kodo",
               "nickname": "Namba", "building_name": "エステムコート難波サウスプレイスⅢラ・パーク",
               "unit_number": None, "unit_floor": "6",
               "source_document_name": "offer.jpeg", "source_document_date": "2026-08-01",
               "ingestion_channel": "agent_image", "source_type": "real_estate_agency",
               "agency_name": None, "agent_name": "山田", "agency_phone": "06-1234",
               "agency_address": "大阪市中央区1-1-1", "agency_role": "unknown",
               "seller_name": "売主", "management_company": "株式会社エステム管理サービス",
               "rent_guarantee_company": "保証", "monthly_rent_yen": 134000.0,
               "market_rent_yen": 150000.0, "occupancy": "rented", "structure": "RC",
               "rent_source": "contract", "contractual_gross_pct": 4.47,
               "gross_gap_pp": 0.0,
               "extracted_entities": [{"name": "ナゾ", "label": "", "role": "unknown"}],
               "provenance": {"monthly_rent_yen": {"page_number": 1,
                                                   "extraction_method": "vision"}},
               **{f: float(r[f]) for f in app.FACTORS}, "score": float(r["score"])}
    app.save_manual_pick(payload)
    b = app.get_historical_picks().iloc[0]
    for f in ("building_name", "unit_floor", "source_document_name",
              "source_document_date", "ingestion_channel", "source_type",
              "agent_name", "agency_phone", "agency_address", "seller_name",
              "management_company", "rent_guarantee_company", "occupancy",
              "structure", "nickname", "hazard_flags", "dev_flags", "rent_source",
              "size_model_mode"):
        ck(f"persist {f}", str(b[f]) == str(payload.get(f, b[f])), f"got {b[f]!r}")
    for f in ("monthly_rent_yen", "market_rent_yen", "contractual_gross_pct"):
        ck(f"persist {f}", abs(float(b[f]) - float(payload[f])) < 1e-6)
    ck("agency_name stays null", b["agency_name"] is None or pd.isna(b["agency_name"]))
    ck("unit_number stays null", b["unit_number"] is None or pd.isna(b["unit_number"]))
    ck("size_breakdown JSON round-trips",
       bool(b["size_breakdown"]) and "prior" in json.loads(b["size_breakdown"]))
    ck("extracted_entities JSON", json.loads(b["extracted_entities"])[0]["role"] == "unknown")
    ck("provenance JSON",
       json.loads(b["provenance"])["monthly_rent_yen"]["extraction_method"] == "vision")
    ck("rents distinct after reload", b["monthly_rent_yen"] != b["market_rent_yen"])


def t_size_consistency():
    print("[13] panel, persisted value and score share one computation")
    fresh_db("/tmp/v_cons.db")
    _st.session_state["size_model_mode"] = "evidence_weighted"
    sc = app.analyze_properties(pd.DataFrame([BASE]))
    row = sc.iloc[0]
    bd = json.loads(row["size_breakdown"])
    ck("breakdown carried on the row", bd.get("applicable") is True)
    ck("persisted final == breakdown final",
       abs(float(row["size_resilience"]) - float(bd["final"])) < 1e-9)
    # The row stores the rounded adjustment and the score consumes that same
    # rounded value, so recomputing from the breakdown must match to 2dp exactly.
    ck("row adjustment == adjustment from that breakdown",
       abs(float(row["size_adjustment"]) - round(app.size_score_adjustment(bd), 2)) < 1e-9,
       f"{row['size_adjustment']} vs {app.size_score_adjustment(bd)}")
    ck("score reproducible from stored adjustment",
       abs(round(float(row["size_adjustment"]), 2) - float(row["size_adjustment"])) < 1e-9)
    ck("evidence carried", abs(float(row["size_evidence"]) - float(bd["evidence"])) < 1e-9)


def t_regression_fixture():
    print("[14] Osaka regression fixture")
    fresh_db("/tmp/v_fix.db")
    ny = app.compute_net_yield(36_000_000, float("nan"), 41.64, "1LDK Apartment",
                               8_600 + 6_200, "full", monthly_rent_yen=134_000,
                               market_rent_yen=150_000)
    ck("annual rent 1,608,000", ny["annual_rent"] == 1_608_000)
    ck("fees 14,800/month", abs(ny["annual_fees"] - 14_800 * 12) < 1)
    ck("contract drives NOI", ny["rent_source"] == "contract")
    ck("no small-unit penalty at 41.64m²",
       app.area_liquidity(41.64, "1LDK Apartment")["penalty"] == 0.0)
    ck("prior ≈ 85", abs(app.size_prior(41.64) - 85.0) < 0.5, str(app.size_prior(41.64)))
    res = app.size_resilience(41.64, "1LDK Apartment", city_code="27111",
                              building_age=13, structure="RC",
                              rent_evidence=app._rent_evidence_from_row(
                                  {"monthly_rent_yen": 134000, "area_sqm": 41.64}))
    ck("confidence limited without local evidence", res["evidence"] < 0.5,
       str(res["evidence"]))


def t_smoke():
    print("[15] static analyser wiring (source-level)")
    src = open("real_estate_app.py", encoding="utf-8").read()
    ana = src[src.index("def render_manual_analyzer"):]
    for fn in ("extract_document_pages(", "merge_extractions(", "map_company_roles(",
               "financing_analysis("):
        ck(f"analyser calls {fn[:-1]}", fn in ana)
    ck("size read from row not recomputed", "size_breakdown" in ana)
    ck("main() present", hasattr(app, "main"))


def t_i18n():
    print("[16] i18n parity")
    en, ja = set(app.TRANSLATIONS["en"]), set(app.TRANSLATIONS["ja"])
    ck("key parity", en == ja, str(en ^ ja)[:120])
    bad = [k for k in en
           if {f for _, f, _, _ in string.Formatter().parse(app.TRANSLATIONS["en"][k]) if f}
           != {f for _, f, _, _ in string.Formatter().parse(app.TRANSLATIONS["ja"][k]) if f}]
    ck("placeholder parity", not bad, str(bad))
    src = open("real_estate_app.py", encoding="utf-8").read()
    refs = set(re.findall(r'tr\(\s*"([a-z0-9_]+)"\s*[,)]', src))
    ck("all tr() refs resolve", not (refs - en), str(sorted(refs - en))[:120])
    for k in ("col_source", "col_building", "col_unit_floor", "rent_unbenchmarked",
              "form_yield_optional"):
        ck(f"dedicated key {k}", k in en and k in ja)




def t_yield_display():
    print("[17] gross-yield display label")
    # contract present, advertised absent -> implied shown, not 0.00%
    a = app.compute_net_yield(36e6, 0.0, 41.64, "1LDK Apartment", 14800, "full",
                              monthly_rent_yen=134_000)
    ck("implied computed when advertised absent",
       abs(a["contractual_gross_pct"] - 4.47) < 0.01, str(a["contractual_gross_pct"]))
    ck("base source is the contract", a["rent_source"] == "contract")
    ck("no false conflict when advertised is blank", not a["rent_conflict"])
    # both present and consistent
    b = app.compute_net_yield(36e6, 4.47, 41.64, "1LDK Apartment", 14800, "full",
                              monthly_rent_yen=134_000)
    ck("consistent pair does not warn", not b["rent_conflict"])
    # both present and conflicting
    c = app.compute_net_yield(36e6, 6.50, 41.64, "1LDK Apartment", 14800, "full",
                              monthly_rent_yen=134_000)
    ck("conflicting pair warns", c["rent_conflict"])
    ck("gap reported", abs(c["gross_gap_pp"] + 2.03) < 0.05, str(c["gross_gap_pp"]))
    src = open("real_estate_app.py", encoding="utf-8").read()
    ck("dynamic label keys used", "yield_label_implied" in src and "yield_label_advertised" in src)
    for k in ("yield_label_advertised", "yield_label_implied"):
        ck(f"{k} bilingual", k in app.TRANSLATIONS["en"] and k in app.TRANSLATIONS["ja"])


def t_comparable_categories():
    print("[18] comparable category separation")
    fresh_db("/tmp/v_cat.db")
    rows = [{"comp_id": f"c{i}", "city_code": "27111", "year": 2025, "quarter": 1,
             "property_kind": "中古マンション等", "floor_plan": "1K",
             "area_sqm": 40 + i * 0.4, "building_year": 2012, "structure": "RC",
             "price_yen": 36_000_000 + i * 400_000,
             "price_per_sqm": (36_000_000 + i * 400_000) / (40 + i * 0.4)}
            for i in range(10)]
    app.save_transaction_comparables(rows)
    for pt in ("Office Unit", "Corner Retail Unit", "Detached House",
               "Whole Building Apartment"):
        v = app.comparable_valuation("27111", 41.64, 13.0, ptype=pt)
        ck(f"{pt} gets NO residential comparables", not v["available"] and v["n"] == 0,
           f"n={v['n']}")
    for pt in ("Office Unit", "Corner Retail Unit"):
        v = app.comparable_valuation("27111", 41.64, 13.0, ptype=pt)
        ck(f"{pt} reports commercial unavailability",
           v.get("unavailable_reason") == "no_commercial_comparables")
    for pt in ("1K Apartment", "2LDK Mansion"):
        ck(f"{pt} still valued", app.comparable_valuation("27111", 41.64, 13.0,
                                                          ptype=pt)["available"])


def t_provenance():
    print("[19] field-level provenance")
    pages = [{"page": 1, "document": "offer.pdf", "method": "text", "confidence": 0.97,
              "fields": {"building_name": "エステムコート", "price_man": 3600}},
             {"page": 2, "document": "offer.pdf", "method": "vision", "confidence": 0.82,
              "fields": {"agency_name": "株式会社アルファ不動産"}},
             {"page": 3, "document": "offer.pdf", "method": "vision",
              "fields": {"price_man": 3900}}]
    merged, conflicts, prov = app.merge_extractions(pages)
    ck("provenance returned, not discarded", bool(prov))
    e = prov["building_name"]
    ck("retains extracted value", e["extracted_value"] == "エステムコート")
    ck("retains source document", e["source_document"] == "offer.pdf")
    ck("retains page number", e["page_number"] == 1)
    ck("retains extraction method", e["extraction_method"] == "text")
    ck("retains confidence", e["confidence"] == 0.97)
    ck("multi-page: later page recorded", prov["agency_name"]["page_number"] == 2)
    ck("conflicts recorded in provenance",
       prov["price_man"]["conflicts"][0]["value"] == 3900)
    ov = app.apply_manual_overrides(prov, {"building_name": "エステムコート",
                                           "price_man": 3700})
    ck("unchanged field keeps its method", ov["building_name"]["extraction_method"] == "text")
    ck("edited field stamped manual_override",
       ov["price_man"]["extraction_method"] == "manual_override")
    ck("edited field keeps the original value", ov["price_man"]["extracted_value"] == 3600)
    ck("edited field records the reviewed value", ov["price_man"]["final_value"] == 3700)
    ck("edited field keeps the original method", ov["price_man"]["original_method"] == "text")
    # persistence + reload
    fresh_db("/tmp/v_prov.db")
    sc = app.analyze_properties(pd.DataFrame([BASE]))
    r = sc.iloc[0]
    app.save_manual_pick({**{k: r[k] for k in r.index}, "lat": 34.6, "lon": 135.5,
                          "hazard_flags": None, "dev_flags": None,
                          "provenance": ov,          # dict in, JSON out
                          **{f: float(r[f]) for f in app.FACTORS},
                          "score": float(r["score"])})
    back = json.loads(app.get_historical_picks().iloc[0]["provenance"])
    ck("provenance JSON persists and reloads",
       back["price_man"]["extraction_method"] == "manual_override")


def t_rent_label_precedence():
    print("[20] text-PDF rent label precedence")
    cases = [("estimate before contract", "想定賃料 150,000円 現行賃料 134,000円", 134000, 150000),
             ("contract before estimate", "現行賃料 134,000円 想定賃料 150,000円", 134000, 150000),
             ("only 想定賃料", "想定賃料 150,000円", None, 150000),
             ("only 参考賃料", "参考賃料 150,000円", None, 150000),
             ("only 市場賃料", "市場賃料 150,000円", None, 150000),
             ("only contractual", "現行賃料 134,000円", 134000, None),
             ("契約賃料", "契約賃料 134,000円", 134000, None),
             ("現況賃料", "現況賃料 134,000円", 134000, None)]
    for label, text, exp_c, exp_m in cases:
        f = app.extract_listing_from_text(text)
        ck(f"{label}: contractual", f.get("monthly_rent_yen") == exp_c,
           f"got {f.get('monthly_rent_yen')}")
        ck(f"{label}: market", f.get("market_rent_yen") == exp_m,
           f"got {f.get('market_rent_yen')}")


def t_docs_clean():
    print("[21] documentation has no stale simulation claims")
    src = open("real_estate_app.py", encoding="utf-8").read()
    stale = [ln.strip()[:90] for ln in src.splitlines()
             if ("シミュレート" in ln or "シミュレーション版" in ln
                 or ("simulat" in ln.lower() and "earlier" not in ln.lower()
                     and "previous" not in ln.lower()
                     and "Nothing here is simulated" not in ln))]
    ck("no stale simulation claims", not stale, str(stale)[:200])
    ck("delisting policy documented",
       "confirmed delisting is treated as a sale" in src
       or "delisting is treated as a sale" in src)


def main():
    for fn in (t_clean_schema, t_migration_v14, t_schema_parity, t_size_modes,
               t_continuity, t_no_legacy_leak, t_rent_basis, t_rent_evidence_honesty,
               t_financing_isolation, t_company_roles, t_documents, t_persistence,
               t_size_consistency, t_regression_fixture, t_smoke, t_i18n, t_yield_display,
               t_comparable_categories, t_provenance, t_rent_label_precedence,
               t_docs_clean):
        fn()
    print("=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("FULL VALIDATION: clean — all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
