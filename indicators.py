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


def theme_strength_score(peer_momenta) -> float:
    """Cross-sectional THEME (industry-rotation) KPI: how strong is the rest of the
    ticker's theme basket in the current scan?

    Input: the momentum scores (0-100) of the OTHER scanned members of the ticker's
    theme — the ticker itself is excluded by the caller, so this measures *peer*
    strength and doesn't double-count the name's own momentum (which already has its
    own factor). The mapping is a damped average deviation from neutral:

        score = clamp(50 + (mean(peers) - 50) * 0.8)

    0.8 damping keeps industry beta from dominating stock-level signals: a red-hot
    basket (peers averaging 90) scores 82, not 90. Neutral 50 when the name has no
    theme, no scanned peers, or only NaN peers — a themeless stock is neither
    rewarded nor punished. Pure function: list in, float out; the theme membership
    lookup lives in the app layer (match_theme)."""
    if not peer_momenta:
        return 50.0
    # pd.notna, not np.isnan: np.isnan(None) raises TypeError, and a defensive
    # caller may hand us None for a missing peer. pd.notna rejects None and NaN
    # alike without raising.
    vals = [float(v) for v in peer_momenta if pd.notna(v)]
    if not vals:
        return 50.0
    return clamp(50.0 + (sum(vals) / len(vals) - 50.0) * 0.8)


def forum_sentiment_score(bull_pct: float, bear_pct: float) -> float | None:
    """Map a Yahoo! Japan 掲示板 みんなの評価 poll (買いたい% / 売りたい%) onto the
    app's 0-100 hype scale via NET bullishness, not raw buy%:

        score = clamp(50 + (bull - bear) * 0.7)

    Rationale: retail boards skew structurally bullish (buy% of 55-65 is normal, not
    a signal), so raw buy% would overstate everything; the buy-minus-sell spread is
    the informative part. 0.7 damping means a very strong poll (e.g. 64/14 -> +50 net)
    lands at 85, and only an extreme ~+71 net saturates at 100. Returns None when the
    inputs aren't a usable poll (NaN, out of [0,100], or 0/0 = no votes recorded).
    Pure function; the page fetching/parsing lives in the app layer.
    """
    b, s = float(bull_pct), float(bear_pct)
    if np.isnan(b) or np.isnan(s):
        return None
    if not (0.0 <= b <= 100.0 and 0.0 <= s <= 100.0) or b + s > 100.0:
        return None
    if b == 0.0 and s == 0.0:
        return None   # no votes -> no signal (distinct from a genuinely neutral poll)
    return clamp(50.0 + (b - s) * 0.7)


def forum_euphoria_sell_score(bull_pct: float, bear_pct: float) -> float | None:
    """ASYMMETRIC sell-side reading of the Yahoo!掲示板 poll — deliberately NOT the
    mirror of forum_sentiment_score.

    Rationale: crowd *bearishness* is a weak/contrarian sell signal (retail capitulation
    often marks bottoms), so it should NOT push toward selling. What genuinely precedes
    drops is retail *euphoria* — an unusually lopsided bullish board on a name the crowd
    is already crowded into. So only net bullishness ABOVE a normal-optimism threshold
    contributes, scaled into sell pressure:

        net = bull - bear
        net <= 35  -> 50   (neutral: normal retail optimism is not a sell signal)
        net  = 70  -> ~85  (extreme euphoria: elevated sell pressure)
        capped at 100

    A balanced or bearish board returns exactly 50 (neutral), never < 50 — bearishness
    is never scored as a reason to sell. Returns None on unusable input (same validity
    rules as forum_sentiment_score) so the caller can SKIP the KPI, not zero it.
    """
    b, s = float(bull_pct), float(bear_pct)
    if np.isnan(b) or np.isnan(s):
        return None
    if not (0.0 <= b <= 100.0 and 0.0 <= s <= 100.0) or b + s > 100.0:
        return None
    if b == 0.0 and s == 0.0:
        return None
    net = b - s
    if net <= 35.0:
        return 50.0                       # normal or bearish -> neutral, never a sell push
    return clamp(50.0 + (net - 35.0) * 1.0)   # only euphoria beyond +35 adds sell pressure


def payout_penalty(payout_ratio: float) -> float:
    """Dividend-coverage penalty (points to subtract from the VALUE factor score).

    `payout_ratio` is dividends / earnings as a FRACTION (yfinance `payoutRatio`).
    A dividend comfortably covered by earnings is fine; one consuming most or more
    than all of earnings is the classic yield-trap profile (a cut waiting to happen),
    which the raw value score otherwise *rewards* — a falling price inflates yield
    and deflates P/E simultaneously.

        payout <= 0.8        -> 0            (healthy coverage: no penalty)
        0.8 < payout <= 1.0  -> 0..8 ramp    (thin coverage)
        payout > 1.0         -> 8 + 20*(p-1) (paying out more than it earns)
        capped at 20 points total

    NaN / missing / non-positive payout -> 0.0: skip-don't-punish, consistent with
    every other optional fundamental (yfinance's payoutRatio is a flaky field, and
    REITs / irregular Japanese payout conventions can distort it — a missing value
    must never penalise). Pure function: float in, float out.
    """
    p = float(payout_ratio)
    if np.isnan(p) or p <= 0.0:
        return 0.0
    if p <= 0.8:
        return 0.0
    if p <= 1.0:
        return (p - 0.8) / 0.2 * 8.0
    return min(8.0 + (p - 1.0) * 20.0, 20.0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average True Range (Wilder): the stock's typical daily movement, used to scale
    trade levels to each name's own volatility. Returns NaN when the OHLC inputs are
    unusable (e.g. a fallback data source without High/Low)."""
    try:
        h, l, c = high.astype(float), low.astype(float), close.astype(float)
    except Exception:
        return float("nan")
    if len(c) < period + 1:
        return float("nan")
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) and atr > 0 else float("nan")


def trade_levels(price: float, sma20: float, sma50: float, atr: float,
                 high_52w: float) -> dict | None:
    """Entry / target / stop REFERENCE LEVELS (not forecasts) derived purely from
    structure and volatility:

      * entry zone  — the SMA20..SMA50 band (a pullback buy in an uptrend); when
                      price already trades below that band, the zone falls back to
                      [price - ATR, price] (you don't 'pull back' up to a level).
      * target      — the 52-week high when price is basing >2% below it (structure
                      to reclaim); otherwise price + 2 ATR (volatility-scaled
                      extension for a name already at highs).
      * stop        — thesis-invalidation: half an ATR below the lower of SMA50 and
                      price - 2 ATR, guaranteeing it sits below the entry zone.

    Volatile names automatically get wider zones/stops than stable ones via ATR.
    Returns None when price or ATR is unusable — a panel with made-up levels is
    worse than no panel. Pure function; honest 'levels, not predictions' framing
    is the caller's job in the UI."""
    p, a = float(price), float(atr)
    if not (p > 0) or not (a > 0) or np.isnan(p) or np.isnan(a):
        return None
    band = [v for v in (float(sma20), float(sma50)) if not np.isnan(v) and v > 0]
    if band:
        entry_lo, entry_hi = min(band), max(band)
        if p < entry_lo:                      # downtrend: MAs overhead, band is meaningless
            entry_lo, entry_hi = p - a, p
    else:
        entry_lo, entry_hi = p - a, p
    hi52 = float(high_52w)
    target = hi52 if (not np.isnan(hi52) and hi52 > p * 1.02) else p + 2.0 * a
    stop_cands = [v for v in (float(sma50), p - 2.0 * a) if not np.isnan(v)]
    stop = min(stop_cands) - 0.5 * a
    stop = min(stop, entry_lo - 0.25 * a)     # always strictly below the entry zone
    return {"entry_lo": float(entry_lo), "entry_hi": float(entry_hi),
            "target": float(target), "stop": float(stop)}
