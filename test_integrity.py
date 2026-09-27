"""Data-integrity tests for the measurement layer: as-of dates, common-session
forward returns, the uncontaminated volume baseline, the expectations layer, the
frozen legacy learner and per-observation context. Offline; no network.

Run: pytest -q test_integrity.py
"""
import json
import os
import sys
import types
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from indicators import (bar_date, forward_excess, expectations_from_frames,
                        early_setup)


def _s(vals, start="2026-01-05"):            # 5 Jan 2026 is a Monday
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype="float64")


# ------------------------------------------------------------ as-of dates
def test_bar_date_is_the_last_price_bar():
    assert bar_date(pd.bdate_range("2026-01-05", periods=5)) == "2026-01-09"   # Friday


def test_bar_date_handles_timezone_aware_index():
    idx = pd.bdate_range("2026-01-05", periods=3, tz="America/New_York")
    assert bar_date(idx) == "2026-01-07"


def test_weekend_observation_enters_at_friday_not_monday():
    """A Sunday scan scored Friday's close. Its forward return must start there;
    the old first-bar-on-or-after rule started from Monday, a price never seen."""
    stock = _s([100.0 + i for i in range(40)])
    bench = _s([100.0] * 40)
    friday = forward_excess(stock, bench, "2026-01-09", 5)
    sunday = forward_excess(stock, bench, "2026-01-11", 5)
    assert sunday == pytest.approx(friday)
    # Friday 9 Jan is bar 4 (price 104); 5 sessions later is bar 9 (price 109)
    assert friday[0] == pytest.approx((109.0 / 104.0 - 1) * 100)


def test_entry_is_never_moved_forward():
    stock = _s([100.0 + i for i in range(40)])
    bench = _s([100.0] * 40)
    sr, _, _ = forward_excess(stock, bench, "2026-01-11", 1)   # Sunday
    assert sr == pytest.approx((105.0 / 104.0 - 1) * 100), "entry must be Friday's bar"


def test_common_sessions_when_calendars_differ():
    """If the benchmark has a holiday the stock traded, both legs must still use the
    same dates — the stock's extra session is dropped, not silently counted."""
    idx = pd.bdate_range("2026-01-05", periods=30)
    stock = pd.Series(np.linspace(100, 130, 30), index=idx)
    bench = pd.Series(np.linspace(100, 110, 30), index=idx).drop(idx[3])   # holiday
    sr, br, xr = forward_excess(stock, bench, idx[0].date().isoformat(), 5)
    common = idx.drop(idx[3])
    exp_s = (stock[common[5]] / stock[common[0]] - 1) * 100
    exp_b = (bench[common[5]] / bench[common[0]] - 1) * 100
    assert sr == pytest.approx(exp_s) and br == pytest.approx(exp_b)
    assert xr == pytest.approx(exp_s - exp_b)


def test_missing_or_stale_benchmark_is_null():
    stock = _s([100.0 + i for i in range(40)])
    assert all(np.isnan(forward_excess(stock, None, "2026-01-09", 5)))
    stale = _s([100.0] * 3)                          # stops on 7 Jan
    assert all(np.isnan(forward_excess(stock, stale, "2026-02-20", 5)))


@pytest.mark.parametrize("horizon", [5, 20, 60])
def test_immature_horizons_stay_null(horizon):
    stock, bench = _s([100.0 + i for i in range(30)]), _s([100.0] * 30)
    out = forward_excess(stock, bench, "2026-01-05", horizon)
    assert all(np.isnan(out)) == (horizon >= 30)


# ------------------------------------------------------------ volume baseline
def test_volume_baseline_excludes_the_recent_window():
    """A 13% volume rise is 'expanding' against the PRECEDING 20 sessions. The old
    baseline contained those same 5 sessions, diluting it to 9.4% — missed."""
    closes = [100.0 * (1.001 ** i) for i in range(200)]
    vols = [1e6] * 195 + [1.13e6] * 5
    c, v = _s(closes), _s(vols)
    out = early_setup(c, v)
    assert out["volume_confirmed"] is True
    # and the contaminated arithmetic really would have missed it:
    contaminated = 1.13e6 / (sum([1e6] * 15 + [1.13e6] * 5) / 20)
    assert contaminated < 1.10


# ------------------------------------------------------------ expectations
def _df(**cols):
    return pd.DataFrame({k: [v] for k, v in cols.items()}, index=["0y"])


def test_revisions_are_normalised_by_how_many_analysts_revised():
    small = expectations_from_frames(eps_revisions=_df(upLast30days=3, downLast30days=0))
    large = expectations_from_frames(eps_revisions=_df(upLast30days=30, downLast30days=0))
    assert small["eps_revision_score"] == pytest.approx(large["eps_revision_score"])
    assert small["eps_revision_score"] == pytest.approx(90.0)


def test_estimate_trend_skips_sign_flips():
    out = expectations_from_frames(eps_trend=_df(current=0.5, **{"30daysAgo": -0.2}))
    assert np.isnan(out["eps_trend_score"]), "a % change across a sign flip is meaningless"


def test_yahoo_key_casing_is_tolerated():
    out = expectations_from_frames(eps_revisions=_df(UPLAST30DAYS=4, downlast30days=1))
    assert out["eps_revision_score"] == pytest.approx(50 + 3 / 5 * 40)


def test_no_coverage_is_nan_not_neutral():
    out = expectations_from_frames()
    assert out["expectations_parts"] == 0 and np.isnan(out["expectations_score"])


def test_revisions_reach_early_setup():
    """The Early Setup 'revisions' component was a dead input fixed at 50."""
    c, v = _s([100.0 * (1.002 ** i) for i in range(200)]), _s([1e6] * 200)
    assert early_setup(c, v, revision_score=82.0)["components"]["revisions"] == 82.0


# ------------------------------------------------------------ app wiring
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
    spec = importlib.util.spec_from_file_location("appint", os.path.join(here, "app.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def db(app, tmp_path):
    app.DB_PATH = str(tmp_path / "i.db")
    app.init_db()
    return app


FUNDS = {"pe": 15.0, "div_yield": 1.0, "market_cap": 5e10, "roe": 0.15,
         "short_pct": 0.02, "payout": 0.3, "name": "X", "sector": "Tech",
         "industry": "Software", "revenue_growth": float("nan")}


def test_analyze_ticker_stamps_the_price_date(app, monkeypatch):
    monkeypatch.setattr(app, "fetch_fundamentals", lambda t: dict(FUNDS))
    idx = pd.bdate_range("2026-01-05", periods=120)
    hist = pd.DataFrame({"Close": np.linspace(100, 120, 120)}, index=idx)
    out = app.analyze_ticker("AAA", "USA", 0.0, hist=hist,
                             bench_close=pd.Series(100.0, index=idx))
    assert out["price_date"] == idx[-1].date().isoformat()


def _res(ticker, price_date, comp=70.0):
    return {"ticker": ticker, "region": "USA", "price": 100.0, "composite": comp,
            "recommendation": "HOLD", "momentum": 60, "value": 50, "technical": 55,
            "hype_score": 40, "quality": 60, "theme": 50, "price_date": price_date,
            "sector": "Tech", "industry": "Software", "market_cap": 5e10}


def test_observation_date_is_the_price_date_and_weekend_rescans_dedupe(db):
    """Saturday and Sunday scans both score Friday: one row, dated Friday."""
    app = db
    app.record_observations([_res("AAA", "2026-01-09")], scan_type="full")
    app.record_observations([_res("AAA", "2026-01-09", comp=72.0)], scan_type="quick")
    with app.get_conn() as c:
        rows = c.execute("SELECT obs_date, composite, scan_type FROM observations").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "2026-01-09" and rows[0][1] == 72.0 and rows[0][2] == "quick"


def test_observation_stores_its_context(db):
    app = db
    app.record_observations([_res("AAA", "2026-01-09")])
    with app.get_conn() as c:
        w, bt, sec, cap = c.execute("SELECT weights_snapshot, benchmark_type, sector, "
                                    "market_cap FROM observations").fetchone()
    weights = json.loads(w)
    assert set(weights) == set(app.FACTORS) and abs(sum(weights.values()) - 1) < 1e-6
    assert bt == "etf_total_return" and sec == "Tech" and cap == pytest.approx(5e10)


def test_recommendation_date_is_the_price_date(db):
    app = db
    app.save_recommendation(dict(_res("AAA", "2026-01-09"), recommendation="BUY"))
    df = app.get_recommendations()
    assert str(df["rec_date"].iloc[0])[:10] == "2026-01-09"


def test_evaluator_uses_common_sessions(db, monkeypatch):
    """End-to-end through evaluate_observations, on the two cases where the old
    evaluator went wrong: a WEEKEND observation date (old rule entered on Monday)
    and a benchmark HOLIDAY inside the window (old rule counted each leg's sessions
    on its own calendar). A weekday date on matching calendars would pass under
    both implementations and prove nothing."""
    app = db
    # a Friday ~150 days back, and the Sunday after it
    fri = date.today() - timedelta(days=150)
    fri -= timedelta(days=(fri.weekday() - 4) % 7)
    sun = fri + timedelta(days=2)
    with app.get_conn() as c:
        c.execute("INSERT INTO observations (scan_id, obs_date, ticker, benchmark, "
                  "composite, rank_pct, factors, model_version) VALUES (?,?,?,?,?,?,?,?)",
                  ("s", sun.isoformat(), "AAA", "SPY", 70.0, 50.0, "{}",
                   app.PRODUCTION_MODEL_VERSION))
    idx = pd.bdate_range(fri, periods=90)
    stock = pd.DataFrame({"Close": [100.0 * 1.01 ** i for i in range(90)]}, index=idx)
    bench = pd.DataFrame({"Close": [100.0 * 1.002 ** i for i in range(90)]},
                         index=idx).drop(idx[2])            # benchmark holiday
    monkeypatch.setattr(app, "get_histories",
                        lambda t, period="1y": {"AAA": stock, "SPY": bench})
    app.evaluate_observations()
    with app.get_conn() as c:
        x5 = c.execute("SELECT x5 FROM observations").fetchone()[0]
    common = idx.drop(idx[2])
    # entry = Friday (common[0]); 5 COMMON sessions later = common[5]
    exp = ((stock.loc[common[5], "Close"] / stock.loc[common[0], "Close"])
           - (bench.loc[common[5], "Close"] / bench.loc[common[0], "Close"])) * 100
    assert x5 == pytest.approx(exp)


def test_attach_expectations_is_us_only_and_fail_safe(app, monkeypatch):
    calls = []
    def fake(t):
        calls.append(t)
        if t == "BAD":
            raise RuntimeError("throttled")
        return {"expectations_score": 66.0}
    monkeypatch.setattr(app, "fetch_expectations", fake)
    rs = [{"ticker": "AAA"}, {"ticker": "7203.T"}, {"ticker": "BAD"}]
    assert app.attach_expectations(rs) == 1
    assert calls == ["AAA", "BAD"], "non-US tickers must not be fetched"
    assert rs[0]["expectations"]["expectations_score"] == 66.0
    assert "expectations" not in rs[2], "a failed fetch leaves the result untouched"


def test_expectations_never_change_scores(app, monkeypatch):
    monkeypatch.setattr(app, "fetch_expectations", lambda t: {"expectations_score": 99.0})
    r = {"ticker": "AAA", "composite": 61.0, "recommendation": "HOLD", "momentum": 55.0}
    before = dict(r)
    app.attach_expectations([r])
    assert {k: r[k] for k in before} == before


def test_legacy_learner_is_frozen_but_still_logs(db, monkeypatch):
    """Shadow mode: the old loop evaluates and records its proposal, but production
    weights do not move."""
    app = db
    assert app.ENABLE_LEGACY_WEIGHT_LEARNING is False
    ts = (date.today() - timedelta(days=40)).isoformat() + "T10:00:00"
    snap = json.dumps({f: 70.0 for f in app.FACTORS} | {"composite": 70.0})
    with app.get_conn() as c:
        c.execute("INSERT INTO mock_portfolio (timestamp, ticker, recommendation_price, "
                  "reason, kpi_snapshot, evaluated, name, price_date) VALUES (?,?,?,?,?,0,?,?)",
                  (ts, "AAA", 100.0, "Top Growth", snap, "AAA", ts[:10]))
    idx = pd.bdate_range(ts[:10], periods=40)
    frame = pd.DataFrame({"Close": [100.0 * 1.01 ** i for i in range(40)]}, index=idx)
    flat = pd.DataFrame({"Close": [100.0] * 40}, index=idx)
    monkeypatch.setattr(app, "get_histories",
                        lambda t, period="1y": {s: (frame if s == "AAA" else flat) for s in t})
    monkeypatch.setattr(app, "fetch_history", lambda t, period="1y": frame if t == "AAA" else flat)
    before = app.get_latest_weights()
    app.walk_forward_update()
    assert app.get_latest_weights() == pytest.approx(before), "production weights must not move"
    with app.get_conn() as c:
        n = c.execute("SELECT COUNT(*) FROM shadow_weights").fetchone()[0]
    assert n == 1, "the frozen learner must still record what it would have done"


def test_composite_summary_rewards_a_predictive_composite(db):
    app = db
    rng = np.random.default_rng(3)
    with app.get_conn() as c:
        for d in range(6):
            day = (date.today() - timedelta(days=150 - 7 * d)).isoformat()
            comp = rng.uniform(30, 90, 20)
            for i in range(20):
                c.execute("INSERT INTO observations (scan_id, obs_date, ticker, composite, "
                          "rank_pct, factors, x20, model_version) VALUES (?,?,?,?,?,?,?,?)",
                          ("s", day, f"T{i}", float(comp[i]), 50.0, "{}",
                           float((comp[i] - 60) / 10 + rng.normal(0, 0.5)),
                           app.PRODUCTION_MODEL_VERSION))
    sm = app.composite_summary(20)
    assert sm["n"] == 120 and sm["spread"] > 3.0 and sm["ic"] > 0.5
    assert sm["top_q"] > sm["bottom_q"]


def test_composite_summary_withholds_on_thin_data(db):
    assert "spread" not in db.composite_summary(20)


# ------------------------------------------------ same-bar comparisons (regression)
def test_rescan_of_the_same_bar_keeps_the_new_flag(db, monkeypatch):
    """Observations are dated by price bar. When the bar predates today (weekends;
    US names scanned before the US open), a second scan must not treat the first
    scan's row — the SAME bar — as 'previous' and clear the NEW flag."""
    app = db
    idx = pd.bdate_range(end=date.today() - timedelta(days=1), periods=220)
    hist = pd.DataFrame({"Close": [100 * 1.002 ** i for i in range(len(idx))],
                         "Volume": [1e6] * len(idx)}, index=idx)
    monkeypatch.setattr(app, "get_histories", lambda *a, **k: {})
    bar = idx[-1].date().isoformat()

    def mk():
        return [dict(_res("AAA", bar), history=hist[["Close"]])]
    first = mk()
    app.attach_early_setups(first, histories={"AAA": hist})
    app.record_observations(first)
    second = mk()
    app.attach_early_setups(second, histories={"AAA": hist})
    assert second[0]["setup_previous_state"] is None
    assert second[0]["setup_is_new"] == first[0]["setup_is_new"]


def test_movers_compare_against_the_earlier_session(db):
    app = db
    y = (date.today() - timedelta(days=1)).isoformat()
    d3 = (date.today() - timedelta(days=3)).isoformat()
    app.record_observations([_res("AAA", d3, comp=61.0)])
    app.record_observations([_res("AAA", y, comp=72.0)])      # this scan, bar < today
    prev = app.previous_composites(["AAA"], asof={"AAA": y})
    assert prev["AAA"][0] == pytest.approx(61.0), "must not compare the scan with itself"


def test_existing_scan_dated_rows_still_evaluate(db, monkeypatch):
    """Rows written before this build are dated by SCAN date (possibly a Sunday).
    Requiring an exact calendar match would silently strand every one of them;
    the as-of rule resolves them to the Friday bar they actually scored."""
    app = db
    fri = date.today() - timedelta(days=150)
    fri -= timedelta(days=(fri.weekday() - 4) % 7)
    with app.get_conn() as c:
        c.execute("INSERT INTO observations (scan_id, obs_date, ticker, benchmark, composite,"
                  " rank_pct, factors, model_version) VALUES (?,?,?,?,?,?,?,?)",
                  ("old", (fri + timedelta(days=2)).isoformat(), "AAA", "SPY", 70.0, 50.0,
                   "{}", app.PRODUCTION_MODEL_VERSION))
    idx = pd.bdate_range(fri, periods=90)
    monkeypatch.setattr(app, "get_histories", lambda t, period="1y": {
        "AAA": pd.DataFrame({"Close": [100 * 1.01 ** i for i in range(90)]}, index=idx),
        "SPY": pd.DataFrame({"Close": [100.0] * 90}, index=idx)})
    app.evaluate_observations()
    with app.get_conn() as c:
        assert c.execute("SELECT x5 FROM observations").fetchone()[0] is not None


# ------------------------------------------------ expectations: wiring, not just the function
def test_earnings_surprise_uses_actual_vs_estimate():
    eh = pd.DataFrame({"epsActual": [1.00, 1.10], "epsEstimate": [1.00, 1.00]},
                      index=pd.to_datetime(["2026-03-31", "2026-06-30"]))
    out = expectations_from_frames(earnings_history=eh)
    assert out["earnings_surprise_score"] == pytest.approx(50 + 0.10 * 200)   # latest: +10%


def test_expectations_reach_the_ledger_and_early_setup_through_a_real_scan(db, monkeypatch):
    """Behavioural wiring test. An expectations layer fetched somewhere the ledger never
    sees would pass any test that injects the score by hand — so run the real
    run_engine and inspect what it wrote."""
    app = db
    idx = pd.bdate_range(end=date.today() - timedelta(days=1), periods=220)
    frame = pd.DataFrame({"Close": [100 * 1.002 ** i for i in range(len(idx))],
                          "Volume": [1e6] * len(idx)}, index=idx)
    monkeypatch.setattr(app, "effective_universe", lambda: {"USA": ["AAA"]})
    monkeypatch.setattr(app, "get_histories", lambda syms, period="1y": {s: frame for s in syms})
    monkeypatch.setattr(app, "fetch_hype_signals", lambda *a, **k: {})
    monkeypatch.setattr(app, "fetch_expectations", lambda t: {
        "expectations_score": 77.0, "eps_revision_score": 80.0,
        "earnings_surprise_score": 65.0, "expectations_parts": 3})
    monkeypatch.setattr(app, "analyze_ticker", lambda t, r, h=0, jp_forum=None, hist=None,
                        bench_close=None: {
        "ticker": t, "region": r, "name": t, "price": 1.0, "momentum": 50.0, "value": 50.0,
        "technical": 50.0, "hype_score": 50.0, "quality": 50.0, "theme": 50.0,
        "theme_match": None, "ret_1m": 0.0, "history": hist[["Close"]],
        "price_date": idx[-1].date().isoformat()})
    results, _ = app.run_engine(regions=["USA"], sources=[])
    assert results[0]["setup_components"]["revisions"] == pytest.approx(77.0), \
        "Early Setup's revisions input must receive the live expectations score"
    with app.get_conn() as c:
        cands = json.loads(c.execute("SELECT candidates FROM observations").fetchone()[0])
        exp = json.loads(c.execute("SELECT expectations FROM observations").fetchone()[0])
    assert cands["expectations_score"] == pytest.approx(77.0), "ledger must be able to test it"
    assert cands["earnings_surprise_score"] == pytest.approx(65.0)
    assert exp["eps_revision_score"] == pytest.approx(80.0)


def test_dax_is_classified_as_a_performance_index():
    import config
    assert config.BENCHMARK_TYPES["^GDAXI"] == "performance_index"
    assert config.BENCHMARK_TYPES["^N225"] == "price_index"
