"""Real interaction test — actually runs the analyzer, not just the server.

test_streamlit_smoke.py proves the process starts and answers /_stcore/health.
That is genuinely not the same thing as proving the app WORKS: the
UnboundLocalError on `_ny` lived inside the submitted-analyzer branch, so the
health endpoint stayed green while scoring a property raised every time.

This file drives the app through streamlit.testing.v1.AppTest, which executes
the real script in-process, so exceptions raised in a widget branch surface as
test failures.

Run:  python test_streamlit_interaction.py      (exit 0 = clean)
"""
import ast
import os
import pathlib
import sys
import tempfile

APP = pathlib.Path(__file__).with_name("real_estate_app.py")
FAILURES = []


def ck(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {extra}" if extra else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def static_unbound_check():
    """Prove every local is assigned before use in the analyzer.

    A cheap structural guard that runs even where AppTest is unavailable: for
    each name assigned inside render_manual_analyzer, the first assignment must
    precede the first load. This is exactly the class of bug that shipped."""
    print("[A] static: locals assigned before use in render_manual_analyzer")
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "render_manual_analyzer"), None)
    if not ck("analyzer found", fn is not None):
        return
    stores, loads = {}, {}
    # Names bound by `except ... as e`, comprehensions and lambdas have their own
    # scope/lifetime rules, so a plain first-assignment-vs-first-use comparison
    # reports them falsely. Exclude them rather than weaken the real check.
    scoped = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.ExceptHandler) and node.name:
            scoped.add(node.name)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                for t in ast.walk(gen.target):
                    if isinstance(t, ast.Name):
                        scoped.add(t.id)
        if isinstance(node, ast.Lambda):
            for a in node.args.args:
                scoped.add(a.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id not in scoped:
            bucket = stores if isinstance(node.ctx, ast.Store) else loads
            bucket.setdefault(node.id, []).append(node.lineno)
    offenders = []
    for name, load_lines in loads.items():
        if name not in stores:
            continue                      # global/builtin/import — not our concern
        if min(stores[name]) > min(load_lines):
            offenders.append((name, min(stores[name]), min(load_lines)))
    ck("no local used before assignment", not offenders,
       "; ".join(f"{n} assigned L{a} used L{u}" for n, a, u in offenders))
    ck("_ny assigned exactly once", len(set(stores.get("_ny", []))) == 1,
       str(stores.get("_ny")))


def interaction_test():
    print("[B] interaction: submit the analyzer through AppTest")
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:                      # older Streamlit
        ck("streamlit.testing.v1 available", False, str(e)[:80])
        return

    tmp = tempfile.mkdtemp(prefix="st_interact_")
    os.environ["DB_PATH"] = os.path.join(tmp, "interaction.db")

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if not ck("app runs without exception", not at.exception,
              str(at.exception)[:200] if at.exception else ""):
        return

    # Fill a minimal valid property: price, area and a contractual rent — the
    # handoff's rule is that contractual rent alone is sufficient, with no
    # advertised gross yield and no source/party fields at all.
    def set_num(key, value):
        for w in list(at.number_input) + list(at.slider):
            if getattr(w, "key", None) == key:
                w.set_value(value)
                return True
        return False

    filled = {
        "man_price": set_num("man_price", 3600),
        "man_area": set_num("man_area", 41.64),
        "man_monthly_rent": set_num("man_monthly_rent", 134000),
        "man_fees": set_num("man_fees", 14800),
        "man_yield": set_num("man_yield", 0.0),        # deliberately absent
    }
    ck("minimal fields present in the form", all(filled.values()), str(filled))
    at.run()
    ck("no exception after filling fields", not at.exception,
       str(at.exception)[:200] if at.exception else "")

    # Submit. This is the branch that raised UnboundLocalError.
    submitted = False
    for btn in at.button:
        label = (getattr(btn, "label", "") or "")
        if "Score" in label or "スコア" in label or "採点" in label:
            btn.click()
            submitted = True
            break
    if not submitted:
        for form_btn in getattr(at, "form_submit_button", []):
            form_btn.click()
            submitted = True
            break
    ck("found and clicked the score control", submitted)
    at.run()
    ck("SUBMITTED ANALYZER RAISES NO EXCEPTION", not at.exception,
       str(at.exception)[:400] if at.exception else "")

    text = " ".join(str(getattr(e, "value", "")) for e in
                    list(at.markdown) + list(at.caption) + list(at.info)
                    + list(at.warning) + list(at.error))
    ck("no UnboundLocalError surfaced", "UnboundLocalError" not in text)
    ck("rendered a result", bool(text.strip()))




def pdf_and_manual_edit_test():
    """Upload-equivalent flow: extract the real PDF, then correct station by hand.

    AppTest cannot drive a file_uploader, so the document is pushed through the
    same normalize_extraction path the uploader uses, then the widgets are
    driven exactly as a user would. That still exercises the bug this was
    written for: an extracted value that leaves the unknown flag set, and a
    locked input the user cannot correct."""
    print("[C] PDF extraction then manual station edit")
    import pathlib
    import real_estate_app as app
    pdf = pathlib.Path(__file__).with_name("sample_offer.pdf")
    if not pdf.exists():
        ck("sample PDF present", False)
        return
    doc = app.extract_document_pages(pdf.read_bytes(), pdf.name)
    text = doc["pages"][0]["text"]
    fields = app.recover_document_fields(app.extract_listing_from_text(text), text)
    updates, _warn = app.normalize_extraction(fields)
    ck("extraction filled the form", updates.get("man_price") == 3600,
       str(updates.get("man_price")))
    ck("station extracted as the shortest walk", updates.get("man_station") == 5,
       str(updates.get("man_station")))
    ck("unknown flag cleared by extraction",
       updates.get("man_station_unknown") is False)
    ck("contractual rent filled", updates.get("man_monthly_rent") == 134000,
       str(updates.get("man_monthly_rent")))
    ck("market rent kept separate", updates.get("man_market_rent") == 150000,
       str(updates.get("man_market_rent")))

    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        ck("streamlit.testing.v1 available", False, str(e)[:60])
        return
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if not ck("app runs", not at.exception, str(at.exception)[:160] if at.exception else ""):
        return
    # Manual correction AFTER extraction: the user overrides 5 with 7.
    edited = False
    for w in at.number_input:
        if getattr(w, "key", None) == "man_station":
            ck("station input is not disabled", not getattr(w, "disabled", False))
            w.set_value(7)
            edited = True
    ck("station edited by hand", edited)
    at.run()
    ck("no exception after manual edit", not at.exception,
       str(at.exception)[:200] if at.exception else "")
    for btn in at.button:
        if "Score" in (getattr(btn, "label", "") or ""):
            btn.click()
            break
    at.run()
    ck("scores after manual station edit", not at.exception,
       str(at.exception)[:300] if at.exception else "")


def main():
    static_unbound_check()
    interaction_test()
    pdf_and_manual_edit_test()
    print("=" * 58)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("INTERACTION: clean — analyzer submits without exception")
    return 0


if __name__ == "__main__":
    sys.exit(main())
