"""Tests for the feedback layer: recommendation outcomes and the Model Lab ledger.

app.py imports streamlit, so these stub it (plus yfinance/altair) before import and
point DB_PATH at a temp file. Nothing here touches Yahoo, Reddit, GDELT, SEC or
Stooq — histories are synthetic DataFrames and every fetch entry point is monkey-
patched. Run: pytest -q test_feedback.py
"""
import os
import sys
import types
from datetime import date, timedelta

import pandas as pd
import pytest


# --------------------------------------------------------------- streamlit stub
class _Any:
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return _Any()


class _St(types.ModuleType):
    session_state: dict = {}
    def __getattr__(self, n): return _Any()
    def cache_data(self, *a, **k): return (lambda f: f)
    def cache_resource(self, *a, **k): return (lambda f: f)


@pytest.fixture(scope="module")
def app():
    sys.modules["streamlit"] = _St("streamlit")
    sys.modules["yfinance"] = types.ModuleType("yfinance")
    sys.modules["altair"] = _St("altair")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    spec = importlib.util.spec_from_file_location("appmod", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def db(app, tmp_path):
    app.DB_PATH = str(tmp_path / "t.db")
    app.init_db()
    return app


def _series(start, n, step):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Close": [100.0 * (step ** i) for i in range(n)]}, index=idx)


# ------------------------------------------------- directional outcome (pure)
def test_buy_wins_on_positive_excess(app):
    assert app.directional_outcome("BUY", 0.05) == app.OUTCOME_WIN
    assert app.directional_outcome("BUY", -0.05) == app.OUTCOME_LOSS


def test_sell_wins_on_negative_excess(app):
    # the inverted-SELL bug: a SELL is RIGHT when the name underperforms
    assert app.directional_outcome("SELL", -0.05) == app.OUTCOME_WIN
    assert app.directional_outcome("SELL", 0.05) == app.OUTCOME_LOSS


def test_hold_is_neutral(app):
    assert app.directional_outcome("HOLD", 0.05) == app.OUTCOME_NEUTRAL
    assert app.directional_outcome("", 0.05) == app.OUTCOME_NEUTRAL


# ------------------------------------------------- outcome evaluation (db)
def _seed_recs(app, rows, rec_date):
    with app.get_conn() as c:
        for ticker, rec in rows:
            c.execute(
                "INSERT INTO recommendations (ticker, rec_date, recommendation, composite,"
                " price_at_rec, momentum, value, technical, hype, quality)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ticker, rec_date, rec, 70.0, 100.0, 70, 50, 60, 40, 65))


def test_evaluate_directions_and_hold_exclusion(db, monkeypatch):
    app = db
    rec_date = (date.today() - timedelta(days=40)).isoformat()
    start = pd.Timestamp(rec_date) - pd.Timedelta(days=5)
    up, down, flat = _series(start, 60, 1.01), _series(start, 60, 0.99), _series(start, 60, 1.0)
    _seed_recs(app, [("BW", "BUY"), ("BL", "BUY"), ("SW", "SELL"),
                     ("SL", "SELL"), ("H", "HOLD")], rec_date)
    hist = {"BW": up, "BL": down, "SW": down, "SL": up, "H": up, "^BM": flat}
    monkeypatch.setattr(app, "get_histories", lambda t, period="1y": {k: v for k, v in hist.items() if k in t})
    monkeypatch.setattr(app, "fetch_history", lambda t, period="1y": hist.get(t))
    monkeypatch.setattr(app, "benchmark_for", lambda t: "^BM")
    app.evaluate_outcomes_only()
    with app.get_conn() as c:
        got = {r[0]: r[1] for r in c.execute("SELECT ticker, outcome FROM recommendations")}
    assert got["BW"] == app.OUTCOME_WIN and got["BL"] == app.OUTCOME_LOSS
    assert got["SW"] == app.OUTCOME_WIN, "SELL on a falling name is a correct call"
    assert got["SL"] == app.OUTCOME_LOSS
    assert got["H"] == app.OUTCOME_NEUTRAL


def test_seeded_hold_rows_are_neutral(db, monkeypatch):
    app = db
    monkeypatch.setattr(app, "save_weights", lambda *a, **k: None)
    app.seed_demo_history()
    df = app.get_recommendations()
    holds = df[df["recommendation"] == "HOLD"]
    assert not holds.empty, "the demo seed should produce some HOLD rows"
    assert (holds["outcome"] == app.OUTCOME_NEUTRAL).all(), \
        "seeded HOLDs must not carry a directional win/loss"
    directional = df[df["recommendation"].isin(app.DIRECTIONAL_CALLS)]
    assert directional["outcome"].isin([app.OUTCOME_LOSS, app.OUTCOME_WIN]).all()


# ------------------------------------------------- observation ledger (db)
def _obs(app, ticker, region="USA", composite=80.0):
    return {"ticker": ticker, "region": region, "price": 100.0, "composite": composite,
            "recommendation": "BUY", "momentum": 70, "value": 50, "technical": 60,
            "hype_score": 40, "quality": 65, "theme": 55}


def test_same_day_regional_scans_coexist(db):
    app = db
    app.record_observations([_obs(app, f"US{i}") for i in range(3)])
    app.record_observations([_obs(app, f"JP{i}.T", region="Japan") for i in range(2)])
    with app.get_conn() as c:
        syms = {r[0] for r in c.execute("SELECT ticker FROM observations")}
    assert len([s for s in syms if s.startswith("US")]) == 3, \
        "a Japan scan must not erase the same day's US observations"
    assert len(syms) == 5


def test_same_day_rescan_replaces_only_that_ticker(db):
    app = db
    app.record_observations([_obs(app, "AAA"), _obs(app, "BBB")])
    app.record_observations([_obs(app, "AAA", composite=91.0)])
    with app.get_conn() as c:
        rows = dict(c.execute("SELECT ticker, composite FROM observations").fetchall())
    assert len(rows) == 2 and rows["AAA"] == 91.0 and rows["BBB"] == 80.0


def test_missing_benchmark_leaves_excess_null(db, monkeypatch):
    app = db
    # Must sit INSIDE the stale window, or the row is legitimately not selected and
    # the assertion would pass without exercising anything.
    obs_day = (date.today() - timedelta(days=100)).isoformat()
    with app.get_conn() as c:
        c.execute("INSERT INTO observations (scan_id, obs_date, ticker, benchmark, price,"
                  " composite, recommendation, rank_pct, factors) VALUES (?,?,?,?,?,?,?,?,?)",
                  ("s", obs_day, "X", "NOPE", 100.0, 88.0, "BUY", 100.0, "{}"))
    stock = _series(obs_day, 90, 1.01)
    monkeypatch.setattr(app, "get_histories", lambda t, period="1y": {"X": stock})
    app.evaluate_observations()
    with app.get_conn() as c:
        x5, x20 = c.execute("SELECT x5, x20 FROM observations").fetchone()
    assert x5 is None and x20 is None, "no benchmark must mean NULL, not a 0% substitute"


def test_partial_horizons_are_filled_and_row_stays_pending(db, monkeypatch):
    app = db
    obs_day = (date.today() - timedelta(days=100)).isoformat()
    with app.get_conn() as c:
        c.execute("INSERT INTO observations (scan_id, obs_date, ticker, benchmark, price,"
                  " composite, recommendation, rank_pct, factors) VALUES (?,?,?,?,?,?,?,?,?)",
                  ("s", obs_day, "X", "BM", 100.0, 88.0, "BUY", 100.0, "{}"))
    # only 10 sessions available -> 5d fills, 20d and 60d cannot
    stock, bench = _series(obs_day, 10, 1.01), _series(obs_day, 10, 1.0)
    monkeypatch.setattr(app, "get_histories", lambda t, period="1y": {"X": stock, "BM": bench})
    app.evaluate_observations()
    with app.get_conn() as c:
        x5, x20, x60 = c.execute("SELECT x5, x20, x60 FROM observations").fetchone()
    assert x5 is not None and x20 is None and x60 is None
    with app.get_conn() as c:
        still = c.execute("SELECT COUNT(*) FROM observations WHERE x5 IS NULL"
                          " OR x20 IS NULL OR x60 IS NULL").fetchone()[0]
    assert still == 1, "a row with x5 filled but x20/x60 missing must stay selectable"


def test_stale_rows_do_not_block_newer_ones(db):
    app = db
    old = (date.today() - timedelta(days=400)).isoformat()
    with app.get_conn() as c:
        c.execute("INSERT INTO observations (scan_id, obs_date, ticker, benchmark,"
                  " composite, rank_pct, factors) VALUES (?,?,?,?,?,?,?)",
                  ("s", old, "ANCIENT", "BM", 70.0, 50.0, "{}"))
        cutoff = (date.today() - timedelta(days=200)).isoformat()
        pend = {r[0] for r in c.execute(
            "SELECT ticker FROM observations WHERE (x5 IS NULL OR x20 IS NULL"
            " OR x60 IS NULL) AND obs_date >= ?", (cutoff,))}
    assert "ANCIENT" not in pend


# ------------------------------------------------- session-accurate horizons
def test_sessions_not_calendar_days(app):
    stock = _series("2026-01-05", 90, 1.01)      # business days only
    p0, p1 = app._close_after_sessions(stock, "2026-01-05", 20)
    assert p1 / p0 == pytest.approx(1.01 ** 20), "20 must mean 20 trading sessions"


def test_weekend_start_resolves_to_next_session(app):
    stock = _series("2026-01-05", 90, 1.01)      # Mon 5 Jan; 3/4 Jan is a weekend
    p0, _ = app._close_after_sessions(stock, "2026-01-03", 5)
    assert p0 == pytest.approx(100.0), "a weekend date resolves to the first real session"


def test_unmatured_horizon_returns_nan(app):
    stock = _series("2026-01-05", 30, 1.01)
    _, p1 = app._close_after_sessions(stock, "2026-01-05", 60)
    assert pd.isna(p1), "an unmatured horizon must be NaN, never a fabricated price"


# ------------------------------------------------- separation of the two loops
def test_model_lab_never_writes_weights(db, monkeypatch):
    app = db
    before = app.get_latest_weights()
    calls = []
    monkeypatch.setattr(app, "save_weights", lambda *a, **k: calls.append(a))
    app.record_observations([_obs(app, "AAA")])
    app.evaluate_observations()
    app.observation_buckets(20)
    assert calls == [], "the observation ledger must not touch kpi_weights"
    assert app.get_latest_weights() == before


# ------------------------------------------- Early Setup app wiring
def test_early_setup_receives_volume_from_the_scan(db, monkeypatch):
    """Regression: results carry Close ONLY (a deliberate memory fix), so volume has
    to arrive via the caller's full OHLCV frames. When it didn't, the accumulation
    component sat at neutral 50 forever and CONFIRMED was unreachable on live scans.
    The pure-function tests could not see this — they pass volume directly."""
    app = db
    idx = pd.bdate_range("2026-01-05", periods=200)
    close = pd.Series([100.0 * (1.002 ** i) for i in range(200)], index=idx)
    # volume arrives overwhelmingly on up days -> accumulation must be well above 50
    chg = close.pct_change().fillna(0.0)
    vol = pd.Series([3.0e6 if ch > 0 else 3.0e5 for ch in chg], index=idx)
    full = pd.DataFrame({"Close": close, "Volume": vol})
    bench = pd.DataFrame({"Close": pd.Series([100.0] * 200, index=idx)})
    monkeypatch.setattr(app, "get_histories", lambda t, period="1y": {b: bench for b in t})

    def _result():
        return [{"ticker": "AAA", "region": "USA", "price": 150.0, "composite": 70.0,
                 "recommendation": "BUY", "ret_1m": 1.0,
                 "history": full[["Close"]],          # exactly what analyze_ticker stores
                 "momentum": 70, "value": 50, "technical": 60,
                 "hype_score": 40, "quality": 65, "theme": 55}]

    without = _result()
    app.attach_early_setups(without)                       # no frames -> Close only
    with_vol = _result()
    app.attach_early_setups(with_vol, histories={"AAA": full})

    assert without[0]["setup_components"]["accumulation"] == 50.0, \
        "Close-only input must leave accumulation neutral"
    assert with_vol[0]["setup_components"]["accumulation"] > 60.0, \
        "the scan's OHLCV frame must reach the detector"
    assert with_vol[0]["setup_score"] != without[0]["setup_score"]


def test_scan_call_site_hands_over_full_frames():
    """Guard the WIRING, not just the function.

    The test above passes `histories` explicitly, so it still passes even if the scan
    forgets to — which is exactly how the original bug survived: the pure detector and
    its unit tests were correct while the live path silently ran without volume.
    A source-level assertion is crude, but it fails for the one mistake that actually
    happened, which a behavioural test at this boundary cannot reach.
    """
    import os
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")).read()
    assert "attach_early_setups(results, histories=bulk)" in src, (
        "run_engine must hand its OHLCV frames to attach_early_setups; without them "
        "accumulation is stuck at neutral and CONFIRMED is unreachable")
