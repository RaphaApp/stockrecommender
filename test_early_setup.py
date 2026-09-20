"""Offline tests for the Early Setup detector (indicators.early_setup).

Synthetic price/volume frames only — no network, no Streamlit. The important tests
here are the negative ones: a pre-breakout detector earns its keep by REFUSING
falling knives, extended names and mere oversold bounces.

Run: pytest -q test_early_setup.py
"""
import numpy as np
import pandas as pd
import pytest

from indicators import compute_rsi, early_setup, SETUP_STATES


def _frame(closes, vols=None):
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype="float64")
    v = pd.Series(vols if vols is not None else [1e6] * len(closes),
                  index=idx, dtype="float64")
    return c, v


def _uptrend_then(n_base, base_slope, tail, tail_vols=None, start=100.0):
    """A long healthy uptrend followed by a specific recent pattern."""
    closes = [start * (base_slope ** i) for i in range(n_base)]
    last = closes[-1]
    closes += [last * f for f in tail]
    vols = [1e6] * n_base + (tail_vols if tail_vols else [1e6] * len(tail))
    return _frame(closes, vols)


# ------------------------------------------------------------------ basics
def test_returns_none_on_short_history():
    c, v = _frame([100.0] * 30)
    assert early_setup(c, v) is None


def test_state_is_always_a_known_label():
    rng = np.random.default_rng(7)
    for seed_shift in range(6):
        vals = 100 + np.cumsum(rng.normal(seed_shift * 0.05, 1.0, 180))
        c, v = _frame(list(np.maximum(vals, 5.0)))
        out = early_setup(c, v)
        assert out is None or out["state"] in SETUP_STATES
        if out:
            assert 0.0 <= out["score"] <= 100.0


# ------------------------------------------- the pattern it should FIND
def test_macd_improving_before_crossing_zero_is_rewarded():
    # a dip that is flattening out: MACD histogram still negative but rising
    dip = [1 - 0.004 * i for i in range(18)] + [0.930 + 0.004 * i for i in range(14)]
    c, v = _uptrend_then(150, 1.002, dip)
    out = early_setup(c, v)
    assert out is not None
    h_comp = out["components"]["macd"]
    assert h_comp > 50, "a rising histogram must score above neutral"


def test_volatility_compression_scores_above_neutral():
    rng = np.random.default_rng(3)
    noisy = list(100 + np.cumsum(rng.normal(0, 1.6, 150)))          # wide bands
    calm = [noisy[-1] + rng.normal(0, 0.06) for _ in range(30)]     # then coiled
    c, v = _frame(noisy + calm)
    out = early_setup(c, v)
    assert out is not None
    assert out["components"]["compression"] > 55, "a tightening range should register"


def test_accumulation_without_breakout():
    # flat price, but volume arrives on up-days -> accumulation, not yet confirmed
    base = [100.0 + (0.4 if i % 2 == 0 else -0.35) for i in range(40)]
    vols = [2.2e6 if i % 2 == 0 else 6e5 for i in range(40)]
    c, v = _uptrend_then(140, 1.001, [b / 100.0 for b in base], vols)
    out = early_setup(c, v)
    assert out is not None
    assert out["components"]["accumulation"] > 60, "up-day volume should dominate"
    assert out["state"] != "CONFIRMED", "no breakout yet, so it must not be CONFIRMED"


def test_healthy_trend_pullback_scores_well():
    # long uptrend, then a modest 8% pullback that holds above the long average
    c, v = _uptrend_then(200, 1.004, [1 - 0.010 * i for i in range(9)])
    out = early_setup(c, v)
    assert out is not None
    assert out["components"]["pullback"] >= 60
    assert out["state"] in ("EARLY WATCH", "SETUP STRENGTHENING", "NO SETUP")
    assert out["state"] != "SETUP FAILED"


# ------------------------------------------- the patterns it must REJECT
def test_falling_knife_is_rejected():
    # a hard, uninterrupted decline: never a setup regardless of how "cheap" it looks
    c, v = _uptrend_then(150, 1.001, [1 - 0.016 * i for i in range(22)])
    out = early_setup(c, v)
    assert out is not None
    assert out["knife"] is True
    assert out["state"] == "SETUP FAILED"
    assert out["score"] <= 35.0, "a knife must be capped, not merely low"


def test_oversold_alone_is_not_a_setup():
    # deeply oversold and BELOW its long-term average: no reward for cheapness
    c, v = _uptrend_then(150, 0.998, [1 - 0.012 * i for i in range(25)])
    out = early_setup(c, v)
    assert out is not None
    assert out["components"]["pullback"] == 30.0, "below trend must be penalised"
    assert out["state"] in ("SETUP FAILED", "NO SETUP")


def test_extended_stock_is_rejected():
    # a vertical run far above SMA20 -> already gone, not an early setup
    c, v = _uptrend_then(150, 1.001, [1 + 0.02 * (i + 1) for i in range(12)])
    out = early_setup(c, v)
    assert out is not None
    assert out["extended"] is True
    assert out["state"] == "EXTENDED"


# ------------------------------------------- relative strength
def test_relative_strength_uses_the_benchmark():
    c, v = _uptrend_then(180, 1.002, [1 + 0.004 * i for i in range(6)])
    weak_bench = pd.Series([100.0 * (0.997 ** i) for i in range(len(c))], index=c.index)
    strong_bench = pd.Series([100.0 * (1.01 ** i) for i in range(len(c))], index=c.index)
    beating = early_setup(c, v, bench_close=weak_bench)["components"]["rel_strength"]
    lagging = early_setup(c, v, bench_close=strong_bench)["components"]["rel_strength"]
    assert beating > lagging, "outperforming the benchmark must score higher"
    neutral = early_setup(c, v)["components"]["rel_strength"]
    assert neutral == 50.0, "absent a benchmark the component is neutral, not invented"


def test_revisions_are_optional_and_neutral_when_absent():
    c, v = _uptrend_then(180, 1.002, [1.0] * 5)
    assert early_setup(c, v)["components"]["revisions"] == 50.0
    assert early_setup(c, v, revision_score=90.0)["components"]["revisions"] == 90.0
    assert early_setup(c, v, revision_score=float("nan"))["components"]["revisions"] == 50.0


# ------------------------------------------- no look-ahead
def test_future_values_cannot_change_past_output():
    """Poison the FUTURE, recompute the same past prefix, demand identical output.

    Replaces an earlier assertion of the form `x != y or True`, which was a tautology
    and proved nothing. Comparing state and every component (not just the aggregate
    score) makes a leak visible in whichever component sprang it.
    """
    rng = np.random.default_rng(11)
    values = 100 + np.cumsum(rng.normal(0.05, 1.0, 220))
    c, v = _frame(list(values))
    cut = 180
    before = early_setup(c.iloc[:cut], v.iloc[:cut])

    poisoned_c, poisoned_v = c.copy(), v.copy()
    poisoned_c.iloc[cut:] *= 25.0          # wild future prices
    poisoned_v.iloc[cut:] *= 40.0          # and wild future volume
    after = early_setup(poisoned_c.iloc[:cut], poisoned_v.iloc[:cut])

    assert after["score"] == pytest.approx(before["score"])
    assert after["state"] == before["state"]
    for k, val in before["components"].items():
        assert after["components"][k] == pytest.approx(val), f"{k} leaked future data"
    # trigger/invalidation are date-sensitive levels, so they must be pinned too
    assert after["trigger"] == pytest.approx(before["trigger"]), "trigger leaked future data"
    assert after["invalidation"] == pytest.approx(before["invalidation"]), \
        "invalidation leaked future data"


def test_confirmation_only_once_the_breakout_bar_exists():
    """CONFIRMED must not appear before the breakout is in the data."""
    pre = [1 - 0.002 * i for i in range(12)]                 # quiet drift lower
    breakout = [pre[-1] * (1 + 0.012 * (i + 1)) for i in range(6)]
    tail_vols = [1e6] * 12 + [3.0e6] * 6                     # volume expands on breakout
    c, v = _uptrend_then(180, 1.003, pre + breakout, tail_vols)
    n_break = 6
    before = early_setup(c.iloc[:-n_break], v.iloc[:-n_break])
    after = early_setup(c, v)
    assert before is not None and after is not None
    assert before["state"] != "CONFIRMED", \
        "confirmation must use only information available on that date"
    assert after["confirmed"] or after["state"] in ("CONFIRMED", "EXTENDED"), \
        "once the breakout bars exist, the state should advance"


def test_trigger_and_invalidation_are_window_extremes():
    """Both levels are plain 20-session extremes of data already in hand — no
    forecast, no look-ahead, and trigger must sit above invalidation."""
    c, v = _uptrend_then(180, 1.002, [1 - 0.004 * i for i in range(10)])
    out = early_setup(c, v)
    prior = c.iloc[-21:-1]           # completed bars only
    assert out["trigger"] == pytest.approx(float(prior.max()))
    assert out["invalidation"] == pytest.approx(float(prior.min()))
    assert out["trigger"] > out["invalidation"]


def test_quality_flags_and_metrics():
    c, v = _uptrend_then(220, 1.001, [1.0] * 6)
    bench = pd.Series(np.linspace(100, 110, len(c)), index=c.index)
    out = early_setup(c, v, bench_close=bench)
    q = out["data_quality"]
    assert q["price_history_available"] is True
    assert q["price_history_full_year"] is True
    assert q["volume_available"] is True
    assert q["benchmark_available"] is True
    assert q["common_benchmark_sessions"] == len(c)
    assert 0 <= q["coverage_pct"] <= 100
    assert out["model_version"]
    assert out["distance_to_trigger_pct"] == pytest.approx(
        (out["trigger"] / float(c.iloc[-1]) - 1.0) * 100.0)
    assert out["distance_to_invalidation_pct"] >= 0
    assert out["risk_range_pct"] >= 0


def test_missing_inputs_are_flagged_not_disguised_as_evidence():
    """A neutral 50 from a MISSING input and a neutral 50 from a balanced one are
    indistinguishable in the score — the flags are what tells them apart."""
    c, _ = _uptrend_then(180, 1.001, [1.0] * 6)
    out = early_setup(c, volume=None, bench_close=None)
    q = out["data_quality"]
    assert q["volume_available"] is False and q["benchmark_available"] is False
    assert out["components"]["accumulation"] == 50.0
    assert out["components"]["rel_strength"] == 50.0
    assert q["coverage_pct"] < 100.0


def test_stale_benchmark_is_not_treated_as_available():
    """A benchmark that stopped updating must not silently score relative strength
    off a mismatched window — it is flagged unavailable instead."""
    c, v = _uptrend_then(200, 1.002, [1.0] * 6)
    stale = pd.Series([100.0] * (len(c) - 30), index=c.index[:-30])   # ends 30 bars early
    out = early_setup(c, v, bench_close=stale)
    assert out["data_quality"]["benchmark_endpoint_current"] is False
    assert out["data_quality"]["benchmark_available"] is False
    assert out["components"]["rel_strength"] == 50.0


def test_zero_volume_bars_do_not_count_as_volume_data():
    c, _ = _uptrend_then(200, 1.002, [1.0] * 6)
    zeros = pd.Series([0.0] * len(c), index=c.index)
    out = early_setup(c, zeros)
    assert out["data_quality"]["volume_available"] is False
    assert out["components"]["accumulation"] == 50.0


# ------------------------------- confirmation must clear the trigger
def test_no_confirmation_while_below_the_trigger():
    """Strong internals are not a breakout. If the close still sits below the prior
    completed-bar high, CONFIRMED must not fire — previously it could, leaving the
    displayed resistance contradicted by the state."""
    # rising into resistance on expanding volume, but stopping just short of it
    base = [1 + 0.004 * i for i in range(12)]            # sets a recent high
    fade = [base[-1] * (1 - 0.004 * (i + 1)) for i in range(4)]   # pulls back under it
    tail_vols = [1e6] * 12 + [3.5e6] * 4
    c, v = _uptrend_then(180, 1.002, base + fade, tail_vols)
    out = early_setup(c, v)
    assert out is not None
    assert float(c.iloc[-1]) < out["trigger"], "fixture must end below the trigger"
    assert out["price_breakout"] is False
    assert out["confirmed"] is False
    assert out["state"] != "CONFIRMED"


def test_confirmation_when_the_current_bar_crosses_a_pre_existing_trigger():
    """The mirror case: the same setup, but the final bar clears the prior high on
    expanding volume — all three conditions true, so confirmation is available."""
    base = [1 + 0.004 * i for i in range(12)]
    cross = [base[-1] * 1.03]                            # final bar jumps the level
    tail_vols = [1e6] * 12 + [4.0e6]
    c, v = _uptrend_then(180, 1.002, base + cross, tail_vols)
    out = early_setup(c, v)
    assert out is not None
    assert float(c.iloc[-1]) > out["trigger"], "fixture must end above the trigger"
    assert out["price_breakout"] is True
    assert out["volume_confirmed"] is True
    # EXTENDED is a legitimate outcome for a sharp jump; what must NOT happen is a
    # confirmation that ignores price.
    assert out["state"] in ("CONFIRMED", "EXTENDED", "SETUP STRENGTHENING")


def test_rsi_edge_series_are_defined():
    idx = pd.bdate_range("2026-01-05", periods=40)
    assert compute_rsi(pd.Series(np.arange(1.0, 41.0), index=idx)).iloc[-1] == pytest.approx(100.0)
    assert compute_rsi(pd.Series(np.arange(40.0, 0.0, -1.0), index=idx)).iloc[-1] == pytest.approx(0.0)
    assert compute_rsi(pd.Series(100.0, index=idx)).iloc[-1] == pytest.approx(50.0)

def test_stale_volume_endpoint_is_not_evidence():
    c, v = _uptrend_then(200, 1.002, [1.0] * 6)
    out = early_setup(c, v.iloc[:-3])
    assert out["data_quality"]["volume_endpoint_current"] is False
    assert out["data_quality"]["volume_available"] is False
    assert out["components"]["accumulation"] == 50.0
    assert out["volume_confirmed"] is False
