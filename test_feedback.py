"""Tests for the feedback layer: recommendation outcomes and the Model Lab ledger.

app.py imports streamlit, so these stub it (plus yfinance/altair) before import and
point DB_PATH at a temp file. Nothing here touches Yahoo, Reddit, GDELT, SEC or
Stooq — histories are synthetic DataFrames and every fetch entry point is monkey-
patched. Run: pytest -q test_feedback.py
"""
import json
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


def st_stub_session_keys(app_mod) -> set:
    """Keys the app put in session state — used to prove scan-local data stays local."""
    import streamlit as _st
    return set(getattr(_st, "session_state", {}) or {})


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


def test_run_engine_hands_full_frames_including_bulk_misses(db, monkeypatch):
    """Behavioural wiring test: run the real run_engine and capture what it actually
    hands to attach_early_setups.

    Covers the two paths that matter:
      BULKED  — present in the initial bulk download;
      MISSED  — absent from bulk and recovered by fetch_history() inside the scan.

    The recovered frame used to vanish: analyze_ticker fetched it internally and the
    result kept Close only, so Early Setup saw no volume for that name. This replaces
    an earlier source-string assertion, which guarded the call site but proved nothing
    about behaviour.
    """
    app = db
    idx = pd.bdate_range("2026-01-05", periods=200)

    def _frame(mult):
        close = pd.Series([100.0 * (mult ** i) for i in range(200)], index=idx)
        vol = pd.Series([1e6 + i for i in range(200)], index=idx)
        return pd.DataFrame({"Close": close, "Volume": vol})

    bulked, missed, bench = _frame(1.002), _frame(1.003), _frame(1.0)

    monkeypatch.setattr(app, "TICKER_UNIVERSE", {"USA": ["BULKED", "MISSED"]})
    monkeypatch.setattr(app, "effective_universe", lambda: {"USA": ["BULKED", "MISSED"]})
    # bulk returns ONLY the first ticker -> the second must be recovered
    monkeypatch.setattr(app, "get_histories",
                        lambda syms, period="1y": {s: (bulked if s == "BULKED" else bench)
                                                   for s in syms if s != "MISSED"})
    fetched = []
    def _fetch(t, period="1y"):
        fetched.append(t)
        assert t == "MISSED", "a ticker already in the bulk result must not be refetched"
        return missed
    monkeypatch.setattr(app, "fetch_history", _fetch)
    monkeypatch.setattr(app, "fetch_hype_signals", lambda *a, **k: {})

    def _fake_analyze(ticker, region, hype=0, jp_forum=None, hist=None):
        assert hist is not None, f"{ticker} should receive a frame"
        return {"ticker": ticker, "region": region, "name": ticker, "price": 150.0,
                "momentum": 60.0, "value": 50.0, "technical": 55.0, "hype_score": 40.0,
                "quality": 60.0, "theme": 50.0, "theme_match": None, "ret_1m": 1.0,
                "history": hist[["Close"]]}          # exactly what the real one stores
    monkeypatch.setattr(app, "analyze_ticker", _fake_analyze)

    captured = {}
    real_attach = app.attach_early_setups
    def _spy(results, histories=None):
        captured["histories"] = histories or {}
        return real_attach(results, histories=histories)
    monkeypatch.setattr(app, "attach_early_setups", _spy)

    captured["ret"] = app.run_engine(regions=["USA"], sources=[])

    hists = captured.get("histories", {})
    assert "BULKED" in hists, "the bulk-downloaded frame must be handed over"
    assert "MISSED" in hists, "a frame recovered by fetch_history must be handed over too"
    assert fetched == ["MISSED"], "only the bulk miss may be fetched individually"
    for name, frame in hists.items():
        assert "Volume" in frame.columns, f"{name} frame reached Early Setup without Volume"
    # The memory optimisation must survive: results keep Close only, and the
    # scan-local map must not leak out of run_engine or into session state.
    res, _failed = captured.get("ret", (None, None))
    if res:
        for r in res:
            assert list(r["history"].columns) == ["Close"], \
                "result['history'] must stay Close-only (session-state memory fix)"
    assert "scan_histories" not in st_stub_session_keys(app), \
        "scan_histories must be scan-local, never persisted in session state"


def test_post_scan_tasks_are_isolated(app, monkeypatch):
    """Unit-test the helper directly: each job guarded, one failure never skips
    the rest, and Early Setup must run BEFORE the observation write (the ledger
    persists setup_score/state/components)."""
    order = []
    monkeypatch.setattr(app, "save_mock_portfolio",
                        lambda r: (_ for _ in ()).throw(RuntimeError("mock boom")))
    monkeypatch.setattr(app, "attach_early_setups",
                        lambda r, histories=None: order.append("setup"))
    monkeypatch.setattr(app, "record_observations", lambda r: order.append("obs"))
    app._run_post_scan_tasks([{"ticker": "AAA"}], {})
    assert order == ["setup", "obs"], "a mock-portfolio failure must not skip the others"

    order.clear()
    monkeypatch.setattr(app, "save_mock_portfolio", lambda r: order.append("mock"))
    monkeypatch.setattr(app, "attach_early_setups",
                        lambda r, histories=None: (_ for _ in ()).throw(RuntimeError("setup boom")))
    app._run_post_scan_tasks([{"ticker": "AAA"}], {})
    assert order == ["mock", "obs"], "an Early Setup failure must not skip observations"

    order.clear()
    monkeypatch.setattr(app, "attach_early_setups",
                        lambda r, histories=None: order.append("setup"))
    monkeypatch.setattr(app, "record_observations",
                        lambda r: (_ for _ in ()).throw(RuntimeError("obs boom")))
    app._run_post_scan_tasks([{"ticker": "AAA"}], {})
    assert order == ["mock", "setup"], "an observation failure must not hide the earlier work"


def test_setup_metadata_persists(db):
    """The new quality/metric columns must survive a write-read round trip."""
    app = db
    row = _obs(app, "META")
    row.update({
        "setup_score": 63.0, "setup_state": "SETUP STRENGTHENING",
        "setup_components": {"macd": 70.0},
        "setup_data_quality": {"volume_available": True, "coverage_pct": 83.3},
        "setup_metrics": {"distance_to_trigger_pct": 1.2, "risk_range_pct": 8.5},
        "setup_model_version": "early-setup-v3-quality-metrics",
        "setup_is_new": True, "setup_state_changed": False,
        "setup_previous_state": "NO SETUP",
    })
    assert app.record_observations([row]) == 1
    with app.get_conn() as conn:
        got = conn.execute(
            "SELECT setup_quality, setup_metrics, setup_model_version, setup_is_new, "
            "setup_state_changed, setup_previous_state FROM observations WHERE ticker=?",
            ("META",)).fetchone()
    assert json.loads(got[0])["volume_available"] is True
    assert json.loads(got[1])["distance_to_trigger_pct"] == pytest.approx(1.2)
    assert got[2] == "early-setup-v3-quality-metrics"
    assert got[3] == 1 and got[4] == 0 and got[5] == "NO SETUP"


def test_same_day_rescan_does_not_clear_the_new_flag(db, monkeypatch):
    """The prior-state lookup must ignore TODAY's rows: otherwise a second scan on
    the same day compares a name against its own earlier run and a genuinely NEW
    setup stops being reported as new."""
    app = db
    idx = pd.bdate_range("2026-01-05", periods=220)
    close = pd.Series([100.0 * (1.002 ** i) for i in range(220)], index=idx)
    vol = pd.Series([1e6] * 220, index=idx)
    full = pd.DataFrame({"Close": close, "Volume": vol})
    monkeypatch.setattr(app, "get_histories", lambda t, period="1y": {})

    def _mk():
        return [{"ticker": "NEWBIE", "region": "USA", "price": 150.0, "composite": 70.0,
                 "recommendation": "HOLD", "ret_1m": 1.0, "history": full[["Close"]],
                 "momentum": 70, "value": 50, "technical": 60, "hype_score": 40,
                 "quality": 65, "theme": 55}]

    first = _mk()
    app.attach_early_setups(first, histories={"NEWBIE": full})
    app.record_observations(first)
    second = _mk()
    app.attach_early_setups(second, histories={"NEWBIE": full})
    assert second[0]["setup_previous_state"] is None, \
        "today's own row must not count as the previous state"
