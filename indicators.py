"""Pure technical-indicator and scoring math for the Alpha Quant Engine.

These take price/volume series (or plain numbers) and return numbers — no Streamlit,
no network, no app state — so they are unit-testable in isolation (see
test_indicators.py). Extracted from app.py to keep the scoring core verifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b

def compute_hype(volume: pd.Series) -> dict:
    result = {"score": 0.0, "breakout_days": 0, "avg_ratio": float("nan"), "sustained": False}
    vol = volume.dropna()
    if len(vol) < 33: return result
    baseline = float(vol.iloc[-33:-3].mean())
    if baseline <= 0: return result
    ratios = vol.iloc[-3:] / baseline
    breakout_days = int((ratios > 1.5).sum())
    avg_ratio = float(ratios.mean())
    raw = (breakout_days / 3) * 60 + min(max(avg_ratio - 1.0, 0.0), 2.0) / 2.0 * 40
    result.update(score=float(min(raw, 100.0)), breakout_days=breakout_days, avg_ratio=avg_ratio, sustained=breakout_days == 3)
    return result

def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float: return max(lo, min(hi, v))


def screen_metrics(close: pd.Series, volume: pd.Series | None = None) -> dict | None:
    """Stage-1 deep-scan screen from the bulk price/volume history alone (no extra
    API calls — that's what keeps stage 1 cheap). Replaces the momentum-only ranking
    with a blended 0-100 screen score:

      * momentum (50%)   — mean of the available 1M/3M/6M returns (as before)
      * volume surge (20%) — last 5 trading days vs the trailing 60-day average;
                              neutral (50) when no usable volume (e.g. some fallbacks)
      * 52w-high proximity (30%) — distance below the trailing 252-day high; names
                              basing near highs outrank equal-momentum names that are
                              still 40% underwater
      * falling-knife cap — a name that dropped >15% in the last month is capped at
                              40 no matter how strong its 6-month tail looks, so a
                              crash's *survivor bias* can't ride a stale rally in.

    Returns None when there are fewer than 30 closes; otherwise a dict with the raw
    components (blend_pct, ret_1m_pct, vol_surge, dist_high_pct) and the composite
    `score`. Pure function: pandas in, plain floats out.
    """
    closes = close.dropna()
    if len(closes) < 30:
        return None
    last = float(closes.iloc[-1])

    def _ret(k: int) -> float:
        return (last / float(closes.iloc[-k - 1]) - 1.0) if len(closes) > k else float("nan")

    rets = [_ret(21), _ret(63), _ret(126)]
    vals = [r for r in rets if not np.isnan(r)]
    if not vals or last <= 0:
        return None
    blend_pct = sum(vals) / len(vals) * 100.0
    ret_1m_pct = rets[0] * 100.0 if not np.isnan(rets[0]) else float("nan")

    # 52-week-high proximity: dist_high_pct <= 0, 0 == at the high.
    high_52w = float(closes.iloc[-252:].max())
    dist_high_pct = (last / high_52w - 1.0) * 100.0 if high_52w > 0 else float("nan")

    # Volume surge: 5d vs 60d average. NaN/absent/zero-baseline volume -> NaN (neutral).
    vol_surge = float("nan")
    if volume is not None:
        vol = volume.dropna()
        if len(vol) >= 65:
            base = float(vol.iloc[-65:-5].mean())
            if base > 0:
                vol_surge = float(vol.iloc[-5:].mean()) / base

    m_score = clamp(50.0 + blend_pct * 1.5)
    v_score = 50.0 if np.isnan(vol_surge) else clamp(50.0 + (vol_surge - 1.0) * 50.0)
    h_score = 50.0 if np.isnan(dist_high_pct) else clamp(100.0 + dist_high_pct * 2.5)
    score = 0.5 * m_score + 0.2 * v_score + 0.3 * h_score
    if not np.isnan(ret_1m_pct) and ret_1m_pct < -15.0:
        score = min(score, 40.0)   # falling knife

    return {"score": float(score), "blend_pct": float(blend_pct),
            "ret_1m_pct": float(ret_1m_pct), "vol_surge": float(vol_surge),
            "dist_high_pct": float(dist_high_pct)}
