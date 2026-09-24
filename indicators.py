"""Pure technical-indicator and scoring math for the Alpha Quant Engine.

These take price/volume series (or plain numbers) and return numbers — no Streamlit,
no network, no app state — so they are unit-testable in isolation (see
test_indicators.py). Extracted from app.py to keep the scoring core verifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with explicit up-only, down-only and flat-series handling."""
    delta = close.astype(float).diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    ready = avg_gain.notna() & avg_loss.notna()
    rsi = rsi.mask(ready & (avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask(ready & (avg_gain == 0) & (avg_loss > 0), 0.0)
    return rsi.mask(ready & (avg_gain == 0) & (avg_loss == 0), 50.0)

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

      * entry zone  — a pullback buy: the SMA20..SMA50 band, but FLOORED at
                      price - 1 ATR so a fast/parabolic mover (whose SMA50 lags far
                      below) doesn't produce an entry 15-20% under price that may
                      never fill. When price already trades below the band (downtrend,
                      MAs overhead) the zone is [price - ATR, price] instead.
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
            # Cap how deep the pullback can be: a healthy uptrend rarely retraces to a
            # far-lagging SMA50, so floor the zone at price - 1 ATR (a realistic dip)
            # and cap the top at price (never suggest buying above current price).
            floor = p - a
            entry_lo = max(entry_lo, floor)
            entry_hi = min(max(entry_hi, entry_lo), p)
            if entry_hi - entry_lo < 0.5 * a:     # keep a usable band (>= half an ATR)
                entry_hi = min(entry_lo + 0.5 * a, p)   # widen upward toward price first
                if entry_hi - entry_lo < 0.5 * a:       # only if capped at price, drop lo
                    entry_lo = entry_hi - 0.5 * a
    else:
        entry_lo, entry_hi = p - a, p
    hi52 = float(high_52w)
    target = hi52 if (not np.isnan(hi52) and hi52 > p * 1.02) else p + 2.0 * a
    stop_cands = [v for v in (float(sma50), p - 2.0 * a) if not np.isnan(v)]
    stop = min(stop_cands) - 0.5 * a
    stop = min(stop, entry_lo - 0.25 * a)     # always strictly below the entry zone
    return {"entry_lo": float(entry_lo), "entry_hi": float(entry_hi),
            "target": float(target), "stop": float(stop)}


# ---------------------------------------------------------------------------
# Early Setup — measurement-only pre-breakout detector
# ---------------------------------------------------------------------------
SETUP_STATES = ("SETUP FAILED", "NO SETUP", "EARLY WATCH", "SETUP STRENGTHENING",
                "CONFIRMED", "EXTENDED")
# Stamped onto every stored score so the Model Lab can tell which model version
# produced a row — without it, changing the detector silently mixes incomparable
# observations into the same buckets.
EARLY_SETUP_MODEL_VERSION = "early-setup-v4-trigger-gated-confirmation"

# Component weights. Deliberately spread: no single component can carry a setup,
# which is what stops a merely-oversold name from scoring well on one axis.
_SETUP_WEIGHTS = {"macd": 0.20, "slope": 0.18, "rel_strength": 0.15,
                  "compression": 0.15, "accumulation": 0.14, "pullback": 0.13,
                  "revisions": 0.05}


def setup_confirmation(technical_breakout: bool, price_breakout: bool,
                       volume_confirmed: bool) -> bool:
    """True only when technical, price and volume confirmation all agree.

    Extracted so the rule is testable as a truth table rather than only through a
    synthetic price series, and so the reported booleans and the reported
    `confirmed` flag cannot drift apart.
    """
    return bool(technical_breakout and price_breakout and volume_confirmed)


def _slope_pct(s: pd.Series, k: int = 5) -> float:
    s = s.dropna()
    if len(s) < k + 1:
        return 0.0
    base = float(s.iloc[-k - 1])
    return 0.0 if base == 0 else float(s.iloc[-1]) / base - 1.0


def early_setup(close: pd.Series, volume: pd.Series | None = None,
                bench_close: pd.Series | None = None,
                revision_score: float | None = None) -> dict | None:
    """Pre-breakout "conditions improving" score (0-100) + a state label.

    This is NOT a buy signal and never becomes one: it is deliberately measured
    against forward returns in the Model Lab before anyone trusts it.

    NO LOOK-AHEAD BY CONSTRUCTION: every component reads `close`/`volume` up to and
    including the LAST bar only, never an index beyond it. Truncating the series to
    an earlier date therefore reproduces exactly what would have been computed on
    that date — which is what the confirmation test pins down.

    Components (each 0-100, 50 = neutral):
      macd         — histogram acceleration, with a bonus while still BELOW zero,
                     since that is the improving-but-unconfirmed window this is for
      slope        — SMA20/SMA50 turning up
      rel_strength — 5-session return vs the home benchmark (needs bench_close)
      compression  — Bollinger bandwidth vs its own 60-day median (tight = coiled)
      accumulation — up-day volume vs down-day volume over 20 sessions
      pullback     — off the 52w high but still above the long-term average
      revisions    — optional external fundamental/analyst input; neutral if absent

    Guards, which matter more than the score:
      * falling knife  (1M return < -15%, or price far below the long average)
        -> SETUP FAILED, score capped. Stops the detector catching downtrends.
      * extended       (stretched above SMA20 relative to its own volatility)
        -> EXTENDED. Stops it calling a name that has already run.
      * being oversold is NOT rewarded anywhere: RSI is not an input, and the
        pullback component requires price ABOVE the long-term average.

    Returns None when there is too little history (<60 bars) to judge.
    """
    c = close.dropna()
    if len(c) < 60:
        return None
    last = float(c.iloc[-1])
    if last <= 0:
        return None

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    long_win = min(200, max(60, len(c) // 2))
    sma_long = c.rolling(long_win).mean()
    sl = float(sma_long.iloc[-1]) if pd.notna(sma_long.iloc[-1]) else last
    ret_1m = (last / float(c.iloc[-22]) - 1.0) if len(c) > 22 else float("nan")

    # ---- guards ----------------------------------------------------------
    knife = (not np.isnan(ret_1m) and ret_1m < -0.15) or (sl > 0 and last < sl * 0.90)
    daily_vol = float(c.pct_change().tail(20).std() or 0.0)
    s20v = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else last
    stretch = (last / s20v - 1.0) if s20v > 0 else 0.0
    extended = stretch > max(0.08, 3.0 * daily_vol)

    # ---- components ------------------------------------------------------
    comp: dict[str, float] = {}

    _, _, hist = compute_macd(c)
    h = hist.dropna()
    if len(h) >= 6:
        scale = float(h.tail(60).abs().mean()) or 1.0
        delta = float(h.iloc[-1]) - float(h.iloc[-4])
        base = clamp(50.0 + (delta / scale) * 25.0)
        # improving while still negative is the early window this detector targets
        comp["macd"] = clamp(base + (10.0 if (float(h.iloc[-1]) < 0 and delta > 0) else 0.0))
    else:
        comp["macd"] = 50.0

    comp["slope"] = clamp(50.0 + _slope_pct(sma20) * 1500.0 + _slope_pct(sma50) * 1000.0)

    # Align stock and benchmark on their SHARED sessions. Comparing iloc[-6] to
    # iloc[-6] positionally is wrong whenever the two calendars differ — a Japanese
    # holiday, a half day, or a benchmark that simply stopped updating silently
    # shifted the 5-session window against itself.
    aligned = pd.DataFrame()
    benchmark_endpoint_current = False
    if bench_close is not None:
        aligned = pd.concat([c.rename("stock"), bench_close.dropna().rename("bench")],
                            axis=1, join="inner").dropna()
        benchmark_endpoint_current = bool(not aligned.empty
                                          and aligned.index[-1] == c.index[-1])
    benchmark_available = len(aligned) > 6 and benchmark_endpoint_current
    if benchmark_available:
        r5 = float(aligned["stock"].iloc[-1]) / float(aligned["stock"].iloc[-6]) - 1.0
        br5 = float(aligned["bench"].iloc[-1]) / float(aligned["bench"].iloc[-6]) - 1.0
        comp["rel_strength"] = clamp(50.0 + (r5 - br5) * 1000.0)
    else:
        comp["rel_strength"] = 50.0

    mid, up, lo, _ = compute_bollinger(c)
    width = ((up - lo) / mid.replace(0, np.nan)).dropna()
    if len(width) >= 40:
        med = float(width.tail(60).median())
        comp["compression"] = clamp(50.0 + (1.0 - float(width.iloc[-1]) / med) * 100.0) \
            if med > 0 else 50.0
    else:
        comp["compression"] = 50.0

    v = volume.dropna() if volume is not None else pd.Series(dtype=float)
    pv = (pd.concat([c.rename("close"), v.rename("volume")], axis=1,
                    join="inner").dropna() if not v.empty else pd.DataFrame())
    volume_endpoint_current = bool(not pv.empty and pv.index[-1] == c.index[-1])
    volume_available = (len(pv) >= 21 and volume_endpoint_current
                        and bool((pv["volume"].tail(20) > 0).all()))
    if volume_available:
        ch = pv["close"].pct_change().tail(20)
        vv = pv["volume"].reindex(ch.index)
        up_v, dn_v = float(vv[ch > 0].sum()), float(vv[ch < 0].sum())
        ratio = (up_v / dn_v) if dn_v > 0 else (2.0 if up_v > 0 else 1.0)
        comp["accumulation"] = clamp(50.0 + (ratio - 1.0) * 50.0)
    else:
        comp["accumulation"] = 50.0

    hi = float(c.tail(252).max())
    dist_hi = (last / hi - 1.0) if hi > 0 else 0.0
    above_long = (last / sl - 1.0) if sl > 0 else 0.0
    if above_long > 0 and -0.18 <= dist_hi <= -0.02:
        comp["pullback"] = clamp(60.0 + above_long * 200.0)   # the textbook setup
    elif above_long > 0:
        comp["pullback"] = 55.0
    else:
        comp["pullback"] = 30.0                               # below trend: not a setup

    comp["revisions"] = 50.0 if (revision_score is None or np.isnan(float(revision_score))) \
        else clamp(float(revision_score))

    score = sum(_SETUP_WEIGHTS[k] * comp[k] for k in _SETUP_WEIGHTS)

    # Completed bars only (exclude the current one). With the current bar included,
    # trigger >= last by construction, so "distance to trigger" could never be
    # negative and the level could never actually be cleared — the current close
    # must be able to cross its trigger, not redefine it.
    window = c.iloc[-21:-1]
    trigger = float(window.max())
    invalidation = float(window.min())

    # ---- confirmation & state -------------------------------------------
    vol_expanding = False
    if volume is not None:
        v = volume.dropna()
        if len(v) >= 20:
            recent, basev = float(v.tail(5).mean()), float(v.tail(20).mean())
            vol_expanding = basev > 0 and recent > basev * 1.10
    # Three independent conditions, reported separately so a near-miss is legible:
    #   technical_breakout — MACD positive, price above a rising SMA20
    #   price_breakout     — the close actually clears the prior completed-bar high
    #   volume_confirmed   — participation is expanding
    # The price test is what was missing: without it CONFIRMED could fire while the
    # name still sat BELOW its own displayed resistance, which is not a breakout.
    technical_breakout = (len(h) > 0 and float(h.iloc[-1]) > 0 and last > s20v
                          and _slope_pct(sma20) > 0)
    price_breakout = last > trigger
    volume_confirmed = bool(vol_expanding)
    confirmed = setup_confirmation(technical_breakout, price_breakout, volume_confirmed)

    if knife:
        state, score = "SETUP FAILED", min(score, 35.0)
    elif extended:
        state = "EXTENDED"
    elif score >= 60.0 and confirmed:
        state = "CONFIRMED"
    elif score >= 60.0:
        state = "SETUP STRENGTHENING"
    elif score >= 52.0:
        state = "EARLY WATCH"
    else:
        state = "NO SETUP"

    # Actionable levels, from the same 20-session window as the rest of the read:
    #   trigger      — the recent high; clearing it is what confirms the coil
    #   invalidation — the recent low; losing it says the setup is void
    # Both are plain historical extremes of data already in hand: no look-ahead,
    # no forecast, and they move with the window like every other component.
    distance_to_trigger_pct = (trigger / last - 1.0) * 100.0
    distance_to_invalidation_pct = ((last / invalidation - 1.0) * 100.0
                                    if invalidation > 0 else float("nan"))
    risk_range_pct = ((trigger / invalidation - 1.0) * 100.0
                      if invalidation > 0 else float("nan"))
    # Which inputs were genuinely present. A neutral 50 from a MISSING input and a
    # neutral 50 from a balanced one are indistinguishable in the score, so record
    # the difference rather than letting absent data masquerade as evidence.
    data_quality = {
        "price_history_available": True,
        "price_history_full_year": len(c) >= 200,
        "price_observations": int(len(c)),
        "volume_available": bool(volume_available),
        "volume_endpoint_current": bool(volume_endpoint_current),
        "common_price_volume_sessions": int(len(pv)),
        "benchmark_available": bool(benchmark_available),
        "benchmark_endpoint_current": bool(benchmark_endpoint_current),
        "common_benchmark_sessions": int(len(aligned)),
        "revision_available": bool(revision_score is not None
                                   and not np.isnan(float(revision_score))),
    }
    available_core = 4 + int(volume_available) + int(benchmark_available)
    data_quality["core_inputs_available"] = available_core
    data_quality["core_inputs_total"] = 6
    data_quality["coverage_pct"] = available_core / 6.0 * 100.0

    return {"score": float(score), "state": state, "components": comp,
            "trigger": trigger, "invalidation": invalidation,
            "distance_to_trigger_pct": float(distance_to_trigger_pct),
            "distance_to_invalidation_pct": float(distance_to_invalidation_pct),
            "risk_range_pct": float(risk_range_pct),
            "data_quality": data_quality,
            "model_version": EARLY_SETUP_MODEL_VERSION,
            "ret_1m_pct": float(ret_1m * 100.0) if not np.isnan(ret_1m) else float("nan"),
            "confirmed": bool(confirmed), "extended": bool(extended), "knife": bool(knife),
            "technical_breakout": bool(technical_breakout),
            "price_breakout": bool(price_breakout),
            "volume_confirmed": bool(volume_confirmed)}


# ---------------------------------------------------------------------------
# Factor diagnostics — measurement only, never feeds scoring
# ---------------------------------------------------------------------------
def candidate_signals(close: pd.Series, bench_close: pd.Series | None = None) -> dict:
    """Alternative signals recorded ALONGSIDE the production factors, so the Model Lab
    can test whether they would forecast better before anyone changes the model.

      mom_long_skip1m — return from ~a year ago up to ONE MONTH ago. The last month is
                        deliberately skipped: 1-month returns tend to reverse, while the
                        momentum that persists is the longer-horizon kind. Production
                        momentum currently leans on the 1-month return; this is the
                        textbook alternative.
      rel_ret_1m      — 1-month return MINUS the home benchmark's, on shared sessions
                        with a current endpoint. Production momentum is absolute, so in a
                        broad rally everything scores well; this isolates outperformance.

    NaN when inputs don't support a value (never a guess). Uses only data up to the last
    bar, so there is no look-ahead. Pure function.
    """
    out = {"mom_long_skip1m": float("nan"), "rel_ret_1m": float("nan")}
    c = close.dropna()
    if len(c) >= 200:
        start, skip = float(c.iloc[0]), float(c.iloc[-22])
        if start > 0:
            out["mom_long_skip1m"] = (skip / start - 1.0) * 100.0
    if bench_close is not None and len(c) >= 22:
        al = pd.concat([c.rename("s"), bench_close.dropna().rename("b")],
                       axis=1, join="inner").dropna()
        if len(al) >= 22 and al.index[-1] == c.index[-1]:
            s0, s1 = float(al["s"].iloc[-22]), float(al["s"].iloc[-1])
            b0, b1 = float(al["b"].iloc[-22]), float(al["b"].iloc[-1])
            if s0 > 0 and b0 > 0:
                out["rel_ret_1m"] = ((s1 / s0) - (b1 / b0)) * 100.0
    return out


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation computed as Pearson on ranks (no scipy dependency)."""
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 3:
        return float("nan")
    rx, ry = m.iloc[:, 0].rank(), m.iloc[:, 1].rank()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cross_sectional_ic(df: pd.DataFrame, signals: list, target: str,
                       date_col: str = "obs_date", min_names: int = 8) -> pd.DataFrame:
    """Information Coefficient per signal: the rank correlation between a signal and the
    forward excess return, computed WITHIN each scan date and then averaged.

    Why within-date: pooling all dates mixes market regimes — in a month when everything
    rallied, every signal looks predictive. Ranking names against their same-day peers
    asks the right question: on a given day, did the higher-scored names outperform the
    lower-scored ones?

    Returns one row per signal: mean_ic, n_dates, pct_positive (share of dates with
    IC > 0 — stability), n_obs. Dates with fewer than `min_names` usable names are
    skipped rather than allowed to produce noisy extreme correlations.
    As a rough guide, a mean IC of 0.03–0.05 that is positive on most dates is a useful
    signal in practice; anything that flips sign date to date is noise.
    """
    rows = []
    for sig in signals:
        if sig not in df.columns:
            continue
        ics, n_obs = [], 0
        for _, g in df.groupby(date_col):
            g2 = g[[sig, target]].dropna()
            if len(g2) < min_names:
                continue
            ic = _rank_corr(g2[sig], g2[target])
            if not np.isnan(ic):
                ics.append(ic)
                n_obs += len(g2)
        rows.append({"signal": sig,
                     "mean_ic": float(np.mean(ics)) if ics else float("nan"),
                     "n_dates": len(ics),
                     "pct_positive": (100.0 * sum(i > 0 for i in ics) / len(ics)) if ics else float("nan"),
                     "n_obs": n_obs})
    return pd.DataFrame(rows, columns=["signal", "mean_ic", "n_dates", "pct_positive", "n_obs"])
