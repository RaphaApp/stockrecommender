"""Tests for the measurement-only forecasting diagnostics.

candidate_signals and cross_sectional_ic are pure; the app-level tests use a temp DB
and synthetic data. No network. Run: pytest -q test_diagnostics.py
"""
import json
import os
import sys
import types
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from indicators import candidate_signals, cross_sectional_ic


def _close(vals, start="2026-01-05"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype="float64")


# ---------------------------------------------------------------- candidates
def test_long_momentum_skips_the_last_month():
    """The whole point of this candidate: a last-month spike must NOT move it."""
    base = [100.0 * (1.001 ** i) for i in range(230)]
    spiked = base[:-21] + [base[-22] * 1.5] * 21            # huge final-month jump
    calm = _close(base)
    jump = _close(spiked)
    a = candidate_signals(calm)["mom_long_skip1m"]
    b = candidate_signals(jump)["mom_long_skip1m"]
    assert a == pytest.approx(b), "the skipped month must not influence the signal"


def test_long_momentum_needs_enough_history():
    assert np.isnan(candidate_signals(_close([100.0] * 150))["mom_long_skip1m"])


def test_relative_return_isolates_outperformance():
    c = _close([100.0 * (1.004 ** i) for i in range(60)])
    flat = _close([100.0] * 60)
    same = _close([100.0 * (1.004 ** i) for i in range(60)])
    assert candidate_signals(c, flat)["rel_ret_1m"] > 5.0
    assert candidate_signals(c, same)["rel_ret_1m"] == pytest.approx(0.0, abs=1e-9)


def test_relative_return_refuses_a_stale_benchmark():
    c = _close([100.0 * (1.004 ** i) for i in range(60)])
    stale = c.iloc[:-5] * 0 + 100.0                        # benchmark stops 5 bars early
    assert np.isnan(candidate_signals(c, stale)["rel_ret_1m"])


def test_candidates_have_no_lookahead():
    rng = np.random.default_rng(4)
    vals = list(100 + np.cumsum(rng.normal(0.05, 1.0, 260)))
    c = _close(vals)
    cut = 230
    before = candidate_signals(c.iloc[:cut])
    poisoned = c.copy()
    poisoned.iloc[cut:] *= 30.0
    after = candidate_signals(poisoned.iloc[:cut])
    for k in before:
        assert (np.isnan(before[k]) and np.isnan(after[k])) or \
            after[k] == pytest.approx(before[k]), f"{k} leaked future data"


# ---------------------------------------------------------------- IC
def _panel(n_dates=12, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        x = rng.normal(size=n_names)
        day = (date(2026, 1, 5) + timedelta(days=7 * d)).isoformat()
        for i in range(n_names):
            rows.append({"obs_date": day, "good": x[i], "bad": -x[i],
                         "noise": rng.normal(), "fwd": x[i] + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_ic_separates_signal_from_noise_and_direction():
    ic = cross_sectional_ic(_panel(), ["good", "bad", "noise"], "fwd").set_index("signal")
    assert ic.loc["good", "mean_ic"] > 0.5 and ic.loc["good", "pct_positive"] == 100.0
    assert ic.loc["bad", "mean_ic"] < -0.5, "a backwards signal must show NEGATIVE IC"
    assert abs(ic.loc["noise", "mean_ic"]) < 0.15


def test_ic_is_computed_within_dates_not_pooled():
    """A market-wide move shifts every name on a date equally. Within-date ranking must
    ignore it; pooling across dates would make a useless signal look predictive."""
    rng = np.random.default_rng(9)
    rows = []
    for d in range(12):
        drift = d * 5.0                                     # every name rises with the date
        day = (date(2026, 1, 5) + timedelta(days=7 * d)).isoformat()
        for i in range(25):
            rows.append({"obs_date": day,
                         "calendar": drift,                 # tracks the regime, not the name
                         "fwd": drift + rng.normal()})
    df = pd.DataFrame(rows)
    ic = cross_sectional_ic(df, ["calendar"], "fwd")
    # constant within each date -> no within-date ranking possible -> no IC
    assert ic.loc[0, "n_dates"] == 0 or np.isnan(ic.loc[0, "mean_ic"])


def test_ic_skips_thin_dates():
    df = _panel(n_dates=3, n_names=5)
    ic = cross_sectional_ic(df, ["good"], "fwd", min_names=8)
    assert ic.loc[0, "n_dates"] == 0, "dates with too few names must not produce an IC"


def test_ic_handles_missing_signal_columns():
    ic = cross_sectional_ic(_panel(), ["good", "does_not_exist"], "fwd")
    assert list(ic["signal"]) == ["good"]


# ---------------------------------------------------------------- app wiring
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
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location("appdiag", os.path.join(here, "app.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def db(app, tmp_path):
    app.DB_PATH = str(tmp_path / "d.db")
    app.init_db()
    return app


def _row(ticker, comp, cands=None):
    return {"ticker": ticker, "region": "USA", "price": 100.0, "composite": comp,
            "recommendation": "HOLD", "momentum": 60, "value": 50, "technical": 55,
            "hype_score": 40, "quality": 60, "theme": 50,
            "candidates": cands or {"mom_long_skip1m": 12.0, "rel_ret_1m": 1.5}}


def test_candidates_persist_and_feed_the_ic_report(db):
    app = db
    rng = np.random.default_rng(1)
    with app.get_conn() as c:
        for d in range(12):
            day = (date.today() - timedelta(days=200 - 7 * d)).isoformat()
            x = rng.normal(size=20)
            for i in range(20):
                c.execute(
                    "INSERT INTO observations (scan_id, obs_date, ticker, benchmark, composite,"
                    " rank_pct, factors, candidates, x20) VALUES (?,?,?,?,?,?,?,?,?)",
                    ("s", day, f"T{i}", "BM", 60.0, 50.0,
                     json.dumps({"momentum": float(x[i]), "value": float(rng.normal())}),
                     json.dumps({"mom_long_skip1m": float(-x[i]), "rel_ret_1m": float(rng.normal())}),
                     float(x[i] + rng.normal(0, 0.3))))
    table, n_dates = app.factor_ic_report(20)
    t = table.set_index("signal")
    assert n_dates == 12
    assert t.loc["momentum", "mean_ic"] > 0.5            # built to be predictive
    assert t.loc["mom_long_skip1m", "mean_ic"] < -0.5     # built to point the wrong way


def test_record_observations_stores_candidates(db):
    app = db
    app.record_observations([_row("AAA", 70.0, {"mom_long_skip1m": 9.5, "rel_ret_1m": -2.0})])
    with app.get_conn() as c:
        got = json.loads(c.execute("SELECT candidates FROM observations").fetchone()[0])
    assert got["mom_long_skip1m"] == pytest.approx(9.5)


def test_previous_composites_ignore_today(db):
    """A same-day rescan must compare against the PREVIOUS session, not itself."""
    app = db
    with app.get_conn() as c:
        c.execute("INSERT INTO observations (scan_id, obs_date, ticker, composite, rank_pct,"
                  " factors) VALUES (?,?,?,?,?,?)",
                  ("old", (date.today() - timedelta(days=3)).isoformat(), "AAA", 61.0, 50.0, "{}"))
    app.record_observations([_row("AAA", 72.0)])            # today's row
    prev = app.previous_composites(["AAA", "NEVER"])
    assert prev["AAA"][0] == pytest.approx(61.0), "today's own row must not be 'previous'"
    assert "NEVER" not in prev


def test_candidate_attachment_is_measurement_only(db, monkeypatch):
    """Candidate signals must never touch weights, scores or the recommendation."""
    app = db
    calls = []
    monkeypatch.setattr(app, "save_weights", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(app, "get_histories", lambda t, period="1y": {})
    idx = pd.bdate_range("2026-01-05", periods=230)
    frame = pd.DataFrame({"Close": [100.0 * (1.001 ** i) for i in range(230)]}, index=idx)
    r = _row("AAA", 70.0); r.pop("candidates")
    before = {k: r[k] for k in ("composite", "recommendation", "momentum")}
    app.attach_candidate_signals([r], histories={"AAA": frame})
    assert "candidates" in r
    assert {k: r[k] for k in before} == before
    assert calls == []


def test_candidate_task_runs_before_observation_write(app, monkeypatch):
    order = []
    monkeypatch.setattr(app, "save_mock_portfolio", lambda r: order.append("mock"))
    monkeypatch.setattr(app, "attach_early_setups", lambda r, histories=None: order.append("setup"))
    monkeypatch.setattr(app, "attach_candidate_signals", lambda r, histories=None: order.append("cand"))
    monkeypatch.setattr(app, "record_observations", lambda r: order.append("obs"))
    app._run_post_scan_tasks([{"ticker": "AAA"}], {})
    assert order == ["mock", "setup", "cand", "obs"]
