"""Unit tests for the pure scoring core in indicators.py.

These run with no network and no Streamlit — just synthetic price/volume series with
known properties. Run: pytest -q
"""
import math

import numpy as np
import pandas as pd
import pytest

from indicators import compute_rsi, compute_macd, compute_bollinger, compute_hype, clamp


# --------------------------------------------------------------------------- clamp
def test_clamp_default_bounds():
    assert clamp(-5) == 0.0          # below floor
    assert clamp(150) == 100.0       # above ceiling
    assert clamp(42.5) == 42.5       # passes through


def test_clamp_custom_bounds():
    assert clamp(20, 0, 10) == 10
    assert clamp(-3, 0, 10) == 0
    assert clamp(5, 0, 10) == 5


# ----------------------------------------------------------------------------- RSI
def _series(vals):
    return pd.Series(vals, dtype="float64")


def test_rsi_bounded_0_100():
    rng = np.random.default_rng(0)
    close = _series(100 + np.cumsum(rng.normal(0, 1, 300)))
    rsi = compute_rsi(close).dropna()
    assert len(rsi) > 0
    assert rsi.between(0, 100).all()


def test_rsi_uptrend_higher_than_downtrend():
    # genuine up/down moves (otherwise avg_loss or avg_gain is 0 -> NaN by design).
    # Mostly +1 with occasional -1 trends up; the reverse trends down.
    up = _series(100 + np.cumsum([(-1.0 if i % 6 == 0 else 1.0) for i in range(60)]))
    down = _series(200 + np.cumsum([(1.0 if i % 6 == 0 else -1.0) for i in range(60)]))
    rsi_up = compute_rsi(up).iloc[-1]
    rsi_down = compute_rsi(down).iloc[-1]
    assert rsi_up > 65, rsi_up
    assert rsi_down < 35, rsi_down
    assert rsi_up > rsi_down


def test_rsi_constant_series_is_nan():
    # no gains and no losses -> division by zero -> NaN (documented behaviour)
    rsi = compute_rsi(_series([50.0] * 40))
    assert math.isnan(rsi.iloc[-1])


# ---------------------------------------------------------------------------- MACD
def test_macd_constant_series_is_zero():
    macd_line, signal_line, hist = compute_macd(_series([10.0] * 100))
    assert abs(macd_line.iloc[-1]) < 1e-9
    assert abs(hist.iloc[-1]) < 1e-9


def test_macd_uptrend_positive_and_histogram_identity():
    close = _series(np.linspace(100, 200, 200))
    macd_line, signal_line, hist = compute_macd(close)
    assert macd_line.iloc[-1] > 0          # fast EMA above slow EMA in an uptrend
    # histogram is exactly macd - signal across the whole series
    assert np.allclose((macd_line - signal_line).values, hist.values, equal_nan=True)


# ----------------------------------------------------------------------- Bollinger
def test_bollinger_band_ordering():
    rng = np.random.default_rng(1)
    close = _series(100 + np.cumsum(rng.normal(0, 1, 100)))
    mid, upper, lower, pct_b = compute_bollinger(close)
    tail = slice(-50, None)
    assert (upper[tail] >= mid[tail]).all()
    assert (mid[tail] >= lower[tail]).all()


def test_bollinger_mid_is_rolling_mean():
    close = _series(np.arange(1, 61, dtype="float64"))
    mid, *_ = compute_bollinger(close, window=20)
    assert np.allclose(mid.dropna().values,
                       close.rolling(20).mean().dropna().values)


def test_bollinger_constant_series_pctb_nan():
    # zero width band -> pct_b undefined (NaN), not an exception
    _, _, _, pct_b = compute_bollinger(_series([5.0] * 40), window=20)
    assert math.isnan(pct_b.iloc[-1])


# --------------------------------------------------------------------------- hype
def test_hype_too_short_is_zero():
    out = compute_hype(_series([1000.0] * 20))   # < 33 rows
    assert out["score"] == 0.0
    assert out["breakout_days"] == 0


def test_hype_flat_volume_no_breakout():
    out = compute_hype(_series([1000.0] * 40))
    assert out["breakout_days"] == 0
    assert out["score"] == 0.0
    assert out["sustained"] is False


def test_hype_sustained_spike_maxes_out():
    vol = [1000.0] * 37 + [3000.0, 3000.0, 3000.0]   # last 3 days 3x baseline
    out = compute_hype(_series(vol))
    assert out["breakout_days"] == 3
    assert out["sustained"] is True
    assert out["score"] == pytest.approx(100.0)


def test_hype_single_spike_partial():
    vol = [1000.0] * 37 + [1000.0, 1000.0, 3000.0]   # only the last day spikes
    out = compute_hype(_series(vol))
    assert out["breakout_days"] == 1
    assert out["sustained"] is False
    assert 0.0 < out["score"] < 60.0
