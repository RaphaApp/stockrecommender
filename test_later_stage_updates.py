"""Later-stage update tests.

The rent scanner now returns STRUCTURED observations rather than a flat list of
numbers, because a bare median cannot express that its inputs came from
different areas, layouts, floors or dates. These assertions keep the original
intent of the earlier version — an AI-predicted sale price must never be read as
rent — and extend it to the new contract.
"""
import real_estate_app as a

# --- original intent: 3,027万円 must not become 027万円 = 270,000 ------------
obs = a.extract_rental_observations(
    "賃料 12.5万円 月額 128,000円 AI予測価格 3,027万円", subject_area=56.6)
rents = sorted(o["monthly_rent_yen"] for o in obs)
assert rents == [125000.0, 128000.0], rents
assert 270000.0 not in rents, "AI sale price leaked into rent"

# A page with only an AI valuation yields no rent observation at all.
assert a.extract_rental_observations("AI予測価格 3,027万円〜3,862万円") == []

# --- structured contract ----------------------------------------------------
for o in obs:
    assert o["evidence_kind"] in ("actual", "estimate")
    assert o["raw_context"], "every observation must carry its source text"
    assert o["selected"] is False, "nothing is pre-selected"

# --- no median without explicit selection -----------------------------------
assert a.summarise_selected_observations(obs)["median"] is None
for o in obs:
    o["selected"] = True
summary = a.summarise_selected_observations(obs)
assert summary["count"] == 2 and summary["median"] == 126500.0, summary

# --- estimates are labelled, never merged into actuals ----------------------
est = a.extract_rental_observations("想定賃料 150,000円")
assert len(est) == 1 and est[0]["evidence_kind"] == "estimate", est

# --- excluded quantities never become rent ----------------------------------
for text in ("月額返済 125,000円", "敷金 250,000円", "礼金 125,000円",
             "販売価格 9,800万円", "AI査定価格 3,027万円"):
    assert a.extract_rental_observations(text) == [], text

# --- management fee recorded beside rent, never added to it -----------------
fee_case = a.extract_rental_observations("賃料 125,000円 管理費 8,000円")[0]
assert fee_case["monthly_rent_yen"] == 125000.0
assert fee_case["management_fee_yen"] == 8000.0
assert fee_case["total_monthly_cost_yen"] is None, "no invented total"

print("LATER-STAGE UPDATE TESTS: clean")
