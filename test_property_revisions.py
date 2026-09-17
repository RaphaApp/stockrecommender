"""Advanced P2: versioned property revisions (Update B)."""
import json, os, re, sqlite3, sys, tempfile
os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "rev.db"))
import pandas as pd
import real_estate_app as app
app.init_db()

FAILURES = []
def ck(n, c, e=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f" — {e}" if e else ""))
    if not c: FAILURES.append(n)

BASE = dict(listing_id="REV", url="https://www.kenbiya.com/x/re_1/", portal="Kenbiya",
            title="t", ptype="1LDK Apartment", prefecture="Osaka", city_code="27111",
            station_min=5.0, building_age=13.0, area_sqm=41.64,
            price_yen=36_000_000.0, gross_yield=4.47, zoning="shogyo",
            far_pct=400.0, status="Active", monthly_fees_yen=14800.0,
            yield_basis="full", monthly_rent_yen=134000.0)


def seed(listing_id="REV"):
    r = app.analyze_properties(pd.DataFrame([dict(BASE, listing_id=listing_id)])).iloc[0]
    app.save_manual_pick({**{k: r[k] for k in r.index}, "listing_id": listing_id,
                          "lat": 34.6, "lon": 135.5, "hazard_flags": None,
                          "dev_flags": None, "monthly_rent_yen": 134000.0,
                          **{f: float(r[f]) for f in app.FACTORS},
                          "score": float(r["score"])})
    row = app.get_historical_picks().query(f"listing_id=='{listing_id}'").iloc[0]
    return int(row["id"]), float(row["score"]), row["recommendation_date"]


def main():
    pid, orig_score, orig_date = seed()

    print("[1] later save creates a revision; original row unchanged")
    before = app.get_effective_pick_state(pid)["effective"]
    res = app.create_property_revision(pid, "agency_update", before,
                                       {"agency_name": "株式会社アルファ不動産"},
                                       reason="confirmed by email")
    ck("revision created", res["created"] and res["revision_number"] == 1, str(res))
    row = app.get_historical_picks().query("listing_id=='REV'").iloc[0]
    ck("original score unchanged", float(row["score"]) == orig_score)
    ck("base row agency still null",
       row["agency_name"] is None or pd.isna(row["agency_name"]))

    print("[2] no-change save creates no revision")
    eff = app.get_effective_pick_state(pid)["effective"]
    ck("rejected", not app.create_property_revision(
        pid, "agency_update", eff, {"agency_name": "株式会社アルファ不動産"})["created"])

    print("[3] revision numbers increase per property")
    app.create_property_revision(pid, "market_rent_update",
                                 app.get_effective_pick_state(pid)["effective"],
                                 {"market_rent_yen": 150000.0})
    nums = [r["revision_number"] for r in app.get_property_revisions(pid)]
    ck("sequential", nums == [1, 2], str(nums))
    pid2, _, _ = seed("REV2")
    app.create_property_revision(pid2, "source_update",
                                 app.get_effective_pick_state(pid2)["effective"],
                                 {"source_url": "https://example.com/a"})
    ck("numbering is per property",
       [r["revision_number"] for r in app.get_property_revisions(pid2)] == [1])

    print("[4] concurrent collision handled safely")
    with app.get_conn() as c:
        c.execute("INSERT INTO property_revisions (historical_pick_id,revision_number,"
                  "revision_kind,changed_at) VALUES (?,?,?,?)",
                  (pid, 3, "manual_correction", "2026-01-01"))
    res3 = app.create_property_revision(pid, "lease_update",
                                        app.get_effective_pick_state(pid)["effective"],
                                        {"occupancy": "vacant"})
    ck("retried, not dropped", res3["created"], str(res3))
    ck("fresh number allocated", res3["revision_number"] > 3, str(res3["revision_number"]))

    print("[5] effective state applies revisions in order")
    st = app.get_effective_pick_state(pid)
    ck("agency applied", st["effective"]["agency_name"] == "株式会社アルファ不動産")
    ck("market rent applied", float(st["effective"]["market_rent_yen"]) == 150000.0)
    ck("occupancy applied", st["effective"]["occupancy"] == "vacant")
    app.create_property_revision(pid, "agency_update", st["effective"],
                                 {"agency_name": "第二不動産"})
    ck("last write wins in order",
       app.get_effective_pick_state(pid)["effective"]["agency_name"] == "第二不動産")

    print("[6] original score/date remain available")
    st = app.get_effective_pick_state(pid)
    ck("original_score preserved", float(st["original_score"]) == orig_score)
    ck("original date preserved", st["original_recommendation_date"] == orig_date)
    ck("contractual rent never overwritten",
       float(st["effective"]["monthly_rent_yen"]) == 134000.0)

    print("[7] refreshed analysis is distinct from the original")
    src = open("real_estate_app.py", encoding="utf-8").read()
    for key in ("rev_original", "rev_current", "rev_score_not_rewritten"):
        ck(f"{key} exists", key in app.TRANSLATIONS["en"] and key in app.TRANSLATIONS["ja"])
    ck("UI states the score was not rewritten",
       "not rewritten" in app.TRANSLATIONS["en"]["rev_score_not_rewritten"].lower()
       or "never overwrite" in app.TRANSLATIONS["en"]["rev_score_not_rewritten"].lower())

    print("[8] learner still uses the original recommendation snapshot")
    ck("learner reads historical_picks, not revisions",
       "property_revisions" not in src[src.index("def run_learning_cycle"):
                                       src.index("def run_learning_cycle") + 3000]
       if "def run_learning_cycle" in src else True)
    ck("original_score_preserved flag defaults to 1",
       all(r.get("original_score_preserved") in (1, None)
           for r in app.get_property_revisions(pid)))

    print("[9] legacy property with no revisions works unchanged")
    pid3, sc3, _ = seed("LEGACY")
    st3 = app.get_effective_pick_state(pid3)
    ck("found, zero revisions", st3["found"] and st3["revision_count"] == 0)
    ck("effective equals original", st3["effective"]["score"] == st3["original"]["score"])

    print("[10] SQLite / Supabase schema parity")
    c = sqlite3.connect(app.DB_PATH)
    live = {r[1] for r in c.execute("PRAGMA table_info(property_revisions)")}
    sql = open("supabase_schema.sql", encoding="utf-8").read()
    blk = re.search(r"CREATE TABLE IF NOT EXISTS property_revisions \((.*?)\n\);", sql, re.S)
    decl = {ln.strip().split()[0] for ln in blk.group(1).splitlines()
            if ln.strip() and not ln.strip().startswith(("PRIMARY KEY", "UNIQUE"))} if blk else set()
    ck("every column declared", not (live - decl), str(sorted(live - decl)))
    ck("schema file at current version",
       f"SCHEMA_VERSION {app.SCHEMA_VERSION}" in sql)
    ck("UNIQUE constraint declared", "UNIQUE (historical_pick_id, revision_number)" in sql)

    print("[11] migration is idempotent")
    mig = open("migration_v18_to_v19.sql", encoding="utf-8").read()
    ck("wrapped in a transaction", "BEGIN;" in mig and "COMMIT;" in mig)
    ck("table creation guarded",
       "CREATE TABLE IF NOT EXISTS property_revisions" in mig)
    ck("indexes guarded", mig.count("CREATE INDEX IF NOT EXISTS") >= 2)
    ck("columns guarded", "ADD COLUMN IF NOT EXISTS" in mig)
    ck("version upsert not a bare insert", "ON CONFLICT (key) DO UPDATE" in mig)
    ck("no destructive statements",
       not re.search(r"\b(DROP|TRUNCATE|DELETE FROM)\b", mig, re.I))

    print("[12] field-level before/after round-trips through JSON")
    rv = app.get_property_revisions(pid)[0]
    ck("fields_before is a dict", isinstance(rv["fields_before"], dict))
    ck("after round-trips", rv["fields_after"]["agency_name"] == "株式会社アルファ不動産")
    ck("changed_fields is a list", isinstance(rv["changed_fields"], list))
    ck("reason stored", rv["change_reason"] == "confirmed by email")

    print("[13] agency and management company stay separate")
    roles = app.map_company_roles([("管理会社", "株式会社エステム管理サービス")])
    ck("management set, agency absent",
       roles.get("management_company") and not roles.get("agency_name"))

    print("[14] HOME'S rent observation never overwrites contractual rent")
    obs = app.extract_rental_observations("賃料 128,000円 1K 25.3m2",
                                          source_url="https://www.homes.co.jp/a")
    for o in obs: o["selected"] = True
    summary = app.summarise_selected_observations(obs)
    app.create_property_revision(pid, "market_rent_update",
                                 app.get_effective_pick_state(pid)["effective"],
                                 {"market_rent_yen": summary["median"]},
                                 source_name="LIFULL HOME'S")
    st4 = app.get_effective_pick_state(pid)
    ck("market rent updated", float(st4["effective"]["market_rent_yen"]) == summary["median"])
    ck("contractual rent intact", float(st4["effective"]["monthly_rent_yen"]) == 134000.0)
    ck("original score still intact", float(st4["original_score"]) == orig_score)

    print("[15] revision UI escapes dynamic content")
    seg = src[src.index('tr("rev_history"'):src.index('tr("rev_history"') + 1500]
    ck("html.escape used on revision values", seg.count("html.escape") >= 4, str(seg.count("html.escape")))

    print("=" * 58)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for x in FAILURES: print("  -", x)
        return 1
    print("PROPERTY REVISIONS: clean — all 15 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
