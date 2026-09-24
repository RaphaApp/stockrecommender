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
                    " rank_pct, factors, candidates, x20, model_version)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    ("s", day, f"T{i}", "BM", 60.0, 50.0,
                     json.dumps({"momentum": float(x[i]), "value": float(rng.normal())}),
                     json.dumps({"mom_long_skip1m": float(-x[i]), "rel_ret_1m": float(rng.normal())}),
                     float(x[i] + rng.normal(0, 0.3)), app.PRODUCTION_MODEL_VERSION))
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


# ================================================== relative momentum (v2 model)
from indicators import momentum_score, relative_return, MOMENTUM_REL_SESSIONS


def _trend(step, n=120):
    return _close([100.0 * (step ** i) for i in range(n)])


def test_relative_return_is_stock_minus_benchmark():
    s, b = _trend(1.002), _trend(1.001)
    expected = ((1.002 ** MOMENTUM_REL_SESSIONS) - (1.001 ** MOMENTUM_REL_SESSIONS)) * 100.0
    assert relative_return(s, b) == pytest.approx(expected)


def test_relative_return_rejects_stale_or_missing_benchmark():
    s = _trend(1.002)
    assert np.isnan(relative_return(s, None))
    assert np.isnan(relative_return(s, s.iloc[:-3])), "benchmark ending early is unusable"


def test_same_stock_scores_higher_when_it_beats_its_market():
    """The point of the change: identical price action, different market context."""
    s = _trend(1.002)
    p, sma = float(s.iloc[-1]), float(s.rolling(50).mean().iloc[-1])
    beating = momentum_score(p, sma, 0.5, relative_return(s, _trend(0.999)))
    lagging = momentum_score(p, sma, 0.5, relative_return(s, _trend(1.004)))
    assert beating > lagging + 15


def test_broad_rally_no_longer_lifts_every_name():
    """Two names rising exactly with the market must score neutral on the relative part
    — under the old absolute formula both got the full return bonus."""
    mkt = _trend(1.003)
    stock = _trend(1.003)
    assert relative_return(stock, mkt) == pytest.approx(0.0, abs=1e-9)


def test_missing_benchmark_is_neutral_not_invented():
    s = _trend(1.002)
    p, sma = float(s.iloc[-1]), float(s.rolling(50).mean().iloc[-1])
    with_zero = momentum_score(p, sma, 0.5, 0.0)
    without = momentum_score(p, sma, 0.5, float("nan"))
    assert without == pytest.approx(with_zero), "no benchmark must equal a neutral 0"


def test_momentum_scale_is_preserved():
    """Range stays 0..100 around 50 with a total swing of +/-50, so the factor's scale,
    the learned weights and the BUY/SELL thresholds keep their meaning."""
    assert momentum_score(120.0, 100.0, 1.0, 50.0) == pytest.approx(100.0)   # all max
    assert momentum_score(80.0, 100.0, -1.0, -50.0) == pytest.approx(0.0)    # all min
    assert momentum_score(100.0, 100.0, 1.0, 0.0) == pytest.approx(60.0)     # neutral + MACD


def test_trend_part_stays_absolute():
    """Pure relative momentum would reward a stock falling slower than a crash. A name
    below its own trend with a negative MACD must stay low even if it outperforms."""
    falling_slowly_in_a_crash = momentum_score(85.0, 100.0, -1.0, +20.0)
    assert falling_slowly_in_a_crash <= 50.0


def test_analyze_ticker_uses_the_benchmark(app, monkeypatch):
    """Integration: same stock history, strong vs weak benchmark -> different momentum;
    the rest of the composite's inputs are unaffected."""
    idx = pd.bdate_range("2026-01-05", periods=200)
    hist = pd.DataFrame({"Close": [100.0 * (1.002 ** i) for i in range(200)],
                         "Volume": [1e6] * 200}, index=idx)
    monkeypatch.setattr(app, "fetch_fundamentals", lambda t: {
        "pe": 15.0, "div_yield": 1.0, "market_cap": 1e10, "roe": 0.15, "short_pct": 0.02,
        "payout": 0.3, "name": t, "sector": "", "industry": ""})
    weak = pd.Series([100.0 * (0.999 ** i) for i in range(200)], index=idx)
    strong = pd.Series([100.0 * (1.005 ** i) for i in range(200)], index=idx)
    a = app.analyze_ticker("AAA", "USA", 0.0, hist=hist, bench_close=weak)
    b = app.analyze_ticker("AAA", "USA", 0.0, hist=hist, bench_close=strong)
    assert a["momentum"] > b["momentum"]
    assert a["momentum_basis"] == "relative" and a["rel_ret_3m"] > 0 > b["rel_ret_3m"]
    for k in ("value", "quality", "technical"):
        assert a[k] == pytest.approx(b[k]), f"{k} must not depend on the benchmark"


def test_analyze_ticker_flags_a_missing_benchmark(app, monkeypatch):
    idx = pd.bdate_range("2026-01-05", periods=200)
    hist = pd.DataFrame({"Close": [100.0 * (1.002 ** i) for i in range(200)]}, index=idx)
    monkeypatch.setattr(app, "fetch_fundamentals", lambda t: {
        "pe": 15.0, "div_yield": 1.0, "market_cap": 1e10, "roe": 0.15, "short_pct": 0.02,
        "payout": 0.3, "name": t, "sector": "", "industry": ""})
    monkeypatch.setattr(app, "_benchmark_close", lambda t: None)
    out = app.analyze_ticker("AAA", "USA", 0.0, hist=hist)
    assert out["momentum_basis"] == "no_benchmark" and np.isnan(out["rel_ret_3m"])


def test_observations_are_stamped_and_lab_excludes_old_model(db):
    app = db
    app.record_observations([_row("NEW", 70.0)])
    with app.get_conn() as c:
        assert c.execute("SELECT model_version FROM observations").fetchone()[0] == \
            app.PRODUCTION_MODEL_VERSION
        # an old-model row with a matured outcome must not enter the buckets
        c.execute("INSERT INTO observations (scan_id, obs_date, ticker, composite, rank_pct,"
                  " factors, x20, model_version) VALUES (?,?,?,?,?,?,?,?)",
                  ("o", "2026-01-05", "OLD", 95.0, 50.0, "{}", 9.9, None))
    assert app.observation_buckets(20).empty, "v1 rows must not be pooled with v2"


def test_candidate_keeps_the_old_basis_measurable():
    c = _close([100.0 * (1.003 ** i) for i in range(230)])
    out = candidate_signals(c)
    assert out["abs_ret_1m"] == pytest.approx((1.003 ** 21 - 1.0) * 100.0)
