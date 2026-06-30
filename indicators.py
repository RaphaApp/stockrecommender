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
