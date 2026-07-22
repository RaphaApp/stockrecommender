"""Unit tests for the pure scoring core in indicators.py.

These run with no network and no Streamlit — just synthetic price/volume series with
known properties. Run: pytest -q
"""
import math

import numpy as np
import pandas as pd
import pytest

from indicators import (compute_rsi, compute_macd, compute_bollinger, compute_hype,
                        clamp, screen_metrics, forum_sentiment_score,
                        forum_euphoria_sell_score, theme_strength_score)


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


# ------------------------------------------------------------------- screen_metrics
def test_screen_too_short_is_none():
    assert screen_metrics(_series([100.0] * 20)) is None


def test_screen_uptrend_near_high_beats_downtrend():
    up = _series(np.linspace(100, 160, 200))          # steady climb, sits at its high
    down = _series(np.linspace(160, 100, 200))        # steady bleed, far from high
    s_up = screen_metrics(up)
    s_down = screen_metrics(down)
    assert s_up["score"] > s_down["score"]
    assert s_up["dist_high_pct"] == pytest.approx(0.0, abs=1e-9)   # at the 52w high
    assert s_down["dist_high_pct"] < -30


def test_screen_falling_knife_capped():
    # big 6-month run-up (4x), then a ~20% collapse in the final month: the old
    # momentum-blend loved these; the new screen caps them at 40.
    vals = list(np.linspace(100, 400, 178)) + list(np.linspace(400, 320, 22))
    s = screen_metrics(_series(vals))
    assert s["ret_1m_pct"] < -15
    assert s["score"] <= 40.0
    assert s["blend_pct"] > 0                          # tail momentum is still positive…
    # …which is exactly why the cap (not the blend) has to do the work here.


def test_screen_volume_surge_raises_score():
    close = _series(np.linspace(100, 130, 200))
    flat_vol = _series([1e6] * 200)
    surge_vol = _series([1e6] * 195 + [3e6] * 5)       # 3x the trailing average
    s_flat = screen_metrics(close, flat_vol)
    s_surge = screen_metrics(close, surge_vol)
    assert s_surge["vol_surge"] > 2.5
    assert s_flat["vol_surge"] == pytest.approx(1.0, rel=0.05)
    assert s_surge["score"] > s_flat["score"]


def test_screen_no_volume_is_neutral_not_crash():
    close = _series(np.linspace(100, 130, 200))
    s_none = screen_metrics(close, None)
    s_flat = screen_metrics(close, _series([1e6] * 200))
    # missing volume (e.g. a fallback data source) scores like flat volume, ±rounding
    assert abs(s_none["score"] - s_flat["score"]) < 2.0
    assert math.isnan(s_none["vol_surge"])


# ------------------------------------------------------------ forum sentiment (JP)
def test_forum_bullish_poll():
    # the Rakuten screenshot case: 64% buy / 14% sell -> net +50 -> 85
    assert forum_sentiment_score(64, 14) == pytest.approx(85.0)


def test_forum_bearish_poll():
    assert forum_sentiment_score(20, 60) == pytest.approx(50 + (20 - 60) * 0.7)  # 22


def test_forum_balanced_poll_is_neutral():
    assert forum_sentiment_score(40, 40) == pytest.approx(50.0)


def test_forum_extremes_clamp():
    assert forum_sentiment_score(100, 0) == 100.0
    assert forum_sentiment_score(0, 100) == 0.0


def test_forum_invalid_inputs_none():
    assert forum_sentiment_score(float("nan"), 10) is None
    assert forum_sentiment_score(-5, 10) is None
    assert forum_sentiment_score(70, 40) is None     # sums past 100 -> malformed parse
    assert forum_sentiment_score(0, 0) is None       # no votes -> no signal


# --------------------------------------------------- forum euphoria (sell side)
def test_euphoria_bearish_board_is_neutral():
    # crowd bearishness must NOT push toward selling (contrarian) -> neutral 50
    assert forum_euphoria_sell_score(20, 60) == 50.0
    assert forum_euphoria_sell_score(40, 40) == 50.0


def test_euphoria_normal_optimism_is_neutral():
    # ordinary retail bullishness (net <= 35) is background noise, not a sell signal
    assert forum_euphoria_sell_score(60, 30) == 50.0   # net +30
    assert forum_euphoria_sell_score(64, 30) == 50.0   # net +34, just under threshold


def test_euphoria_extreme_raises_sell_pressure():
    # lopsided euphoria beyond +35 net starts adding sell pressure
    assert forum_euphoria_sell_score(64, 14) == pytest.approx(65.0)   # net +50 -> 50+15
    assert forum_euphoria_sell_score(90, 5) == pytest.approx(100.0)   # net +85 -> capped
    assert forum_euphoria_sell_score(80, 10) > forum_euphoria_sell_score(64, 14)


def test_euphoria_never_below_neutral():
    for b, s in [(0, 100), (10, 80), (50, 50), (55, 40)]:
        assert forum_euphoria_sell_score(b, s) >= 50.0


def test_euphoria_invalid_none():
    assert forum_euphoria_sell_score(float("nan"), 10) is None
    assert forum_euphoria_sell_score(70, 40) is None   # sums > 100
    assert forum_euphoria_sell_score(0, 0) is None


# ------------------------------------------------------- theme (industry) KPI
def test_theme_no_peers_is_neutral():
    # themeless names and single-member baskets carry no rotation signal
    assert theme_strength_score(None) == 50.0
    assert theme_strength_score([]) == 50.0
    assert theme_strength_score([float("nan")]) == 50.0


def test_theme_hot_basket_beats_cold():
    hot = theme_strength_score([80.0, 90.0, 85.0])
    cold = theme_strength_score([20.0, 30.0, 25.0])
    assert hot > 50.0 > cold
    assert hot > cold


def test_theme_damping_factor():
    # peers averaging 90 -> 50 + 40*0.8 = 82, NOT 90: industry beta is damped
    assert theme_strength_score([90.0, 90.0]) == pytest.approx(82.0)
    assert theme_strength_score([10.0, 10.0]) == pytest.approx(18.0)


def test_theme_nan_peers_dropped():
    assert theme_strength_score([80.0, float("nan"), 80.0]) == \
        pytest.approx(theme_strength_score([80.0, 80.0]))


def test_theme_bounded_0_100():
    for peers in ([100.0] * 5, [0.0] * 5, [50.0], [0.0, 100.0]):
        assert 0.0 <= theme_strength_score(peers) <= 100.0


def test_theme_none_peers_tolerated():
    # None must behave exactly like NaN (dropped, no TypeError) — a defensive
    # caller may pass None for a missing peer score.
    assert theme_strength_score([None]) == 50.0
    assert theme_strength_score([80.0, None, 80.0]) == \
        pytest.approx(theme_strength_score([80.0, 80.0]))
