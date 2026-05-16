"""
qf_shared.py — Shared helpers for QUANTFLOW scanners
=====================================================
Extracted from app.py during Session 01 refactor (2026-05-11).
Used by:
  - Existing momentum scanner in app.py
  - NEW SMC Long scanner (qf_smc/ package — future sessions)

Includes:
  - Data fetching (Binance main + mirror, Gate.io fallback)
  - Indicators (EMA, ADX)
  - External market data (Fear & Greed, BTC dominance, funding rate, OI)
  - Data cleaning utilities (_clean_df, get_session, trim_by_days)
  - Walk-forward / purged CV (PurgedTimeSeriesSplit, _purge_is_oos)
  - Backtest outcome classification (_classify_outcome)
  - Time-decay bucketing (_compute_decay_buckets, _bucket_stats_for_trades)
  - Regime similarity weighting (_regime_similarity_weight)
  - Scanner universe + candle helpers
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import urllib3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple


# ============================================================================
# CONSTANTS
# ============================================================================

# Neutral-R threshold: ±0.30R band → NEUTRAL (excluded from ML training).
# See _classify_outcome docstring for full motivation.
NEUTRAL_R_THRESHOLD = 0.30

# Bar-limit caps for deep klines fetches (Binance /api/v3/klines max = 1000).
# Used by _deep_limit_for, _scanner_quick_backtest, _scanner_mini_wfo,
# and _scanner_train_ml so they all pull the same historical depth.
_DEEP_FETCH_LIMITS = {
    "1h": 1000,   # ~41 days
    "2h": 1000,   # ~83 days
    "4h": 1000,   # ~166 days
    "6h": 1000,   # ~250 days
    "12h": 1000,  # ~500 days
    "1d": 1000,   # ~2.7 years
}

# Binance REST interval strings mapped from app-level timeframe labels.
_BINANCE_INTERVAL = {"1D": "1d", "4H": "4h", "1H": "1h", "1W": "1w"}

# Klines endpoint list: main then ISP-bypass mirror (verify=False for mirror).
_BINANCE_KLINES_URLS = [
    ("https://api.binance.com/api/v3/klines",         True),   # main — verify SSL
    ("https://data-api.binance.vision/api/v3/klines",  False),  # mirror — ISP-bypass
]

# Stablecoins and wrapped tokens to exclude from altcoin universe scans.
_SCANNER_EXCLUDE = {
    "USDT", "BUSD", "USDC", "TUSD", "DAI", "FDUSD", "USDP", "USDD",
    "PYUSD", "AEUR", "EURI",
    "WBTC", "WETH", "WBETH",
}


# ============================================================================
# BACKTESTING UTILITIES
# ============================================================================

class PurgedTimeSeriesSplit:
    """
    Walk-forward CV with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of contiguous test folds (≥ 2).
    entry_bars : array-like of int
        Bar index where each sample's signal fires. Must be
        non-decreasing (samples ordered chronologically) — caller's
        responsibility.
    label_end_bars : array-like of int
        Bar index where each sample's label is determined (the bar
        at which WIN/LOSS resolves). Must satisfy label_end_bars[i]
        >= entry_bars[i].
    embargo_pct : float, default 0.01
        Fraction of total_bars used as post-test embargo width.
    total_bars : int, optional
        Total number of bars in the underlying time series. If None,
        defaults to max(label_end_bars)+1.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray
        Sample-index arrays. `train_idx` has been purged (no label
        overlap with any test sample) and embargoed (no immediate-
        post-test entries within E bars).
    """

    def __init__(self, n_splits=5, *, entry_bars, label_end_bars,
                 embargo_pct=0.01, total_bars=None):
        self.n_splits = max(2, int(n_splits))
        self.entry_bars     = np.asarray(entry_bars,     dtype=np.int64)
        self.label_end_bars = np.asarray(label_end_bars, dtype=np.int64)
        if self.entry_bars.shape != self.label_end_bars.shape:
            raise ValueError("entry_bars and label_end_bars must have equal length.")
        # Degenerate guard: ensure every label_end >= entry
        if len(self.entry_bars) and np.any(self.label_end_bars < self.entry_bars):
            # Auto-correct rather than throw — safer for production streaming UI
            self.label_end_bars = np.maximum(self.label_end_bars, self.entry_bars)
        if total_bars is None:
            total_bars = (int(self.label_end_bars.max()) + 1
                          if len(self.label_end_bars) else 1)
        self.total_bars   = max(1, int(total_bars))
        self.embargo_pct  = max(0.0, min(0.5, float(embargo_pct)))
        # At least 1 bar of embargo when embargo_pct > 0
        self.embargo_bars = (max(1, int(np.ceil(self.total_bars * self.embargo_pct)))
                             if self.embargo_pct > 0 else 0)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(self.entry_bars)
        # Not enough samples to form meaningful folds: yield nothing, caller
        # will see empty cv_scores and report cv_acc=None.
        if n < self.n_splits * 2:
            return

        # Partition sample-indices into n_splits contiguous groups (by array
        # order, which must already be time-ordered on entry_bar). Each group
        # in turn becomes the test fold.
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        all_idx = np.arange(n)
        cursor = 0
        for sz in fold_sizes:
            start, end = cursor, cursor + sz
            cursor = end
            test_idx = all_idx[start:end]
            if len(test_idx) == 0:
                continue

            test_entry_min = int(self.entry_bars[test_idx].min())
            test_label_max = int(self.label_end_bars[test_idx].max())

            # All non-test candidates
            candidate = np.concatenate([all_idx[:start], all_idx[end:]])
            if len(candidate) == 0:
                continue

            # Purge + embargo in a single vectorized mask:
            #   keep if (label_end < test_entry_min)  OR
            #          (entry_bar > test_label_max + embargo)
            keep = (
                (self.label_end_bars[candidate] < test_entry_min)
                | (self.entry_bars[candidate]    > test_label_max + self.embargo_bars)
            )
            train_idx = candidate[keep]
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# Purged IS/OOS partition — used by _scanner_mini_wfo
# ─────────────────────────────────────────────────────────────────────────────
def _purge_is_oos(trades: List[dict], is_end_bar: int, total_bars: int,
                  embargo_pct: float = 0.01) -> dict:
    """
    Split a list of trade dicts (each with `bar_index` and `label_end_bar`)
    into purged IS and embargoed OOS subsets at the cut point `is_end_bar`.

    Purge: drop IS trades whose label resolution crosses the cut.
    Embargo: drop OOS trades whose entry falls within `E` bars of the cut.

    Returns dict:
      {
        "is_trades":       [...purged IS trades...],
        "oos_trades":      [...embargoed OOS trades...],
        "n_is_raw":        int,   # IS candidates before purge
        "n_oos_raw":       int,   # OOS candidates before embargo
        "n_purged":        int,   # IS trades dropped for overlap
        "n_embargoed":     int,   # OOS trades dropped for embargo
        "embargo_bars":    int,   # actual embargo width used
      }
    """
    embargo_bars = (max(1, int(np.ceil(max(1, total_bars) * max(0.0, embargo_pct))))
                    if embargo_pct > 0 else 0)
    is_raw, oos_raw = [], []
    for t in trades:
        if int(t.get("bar_index", -1)) < is_end_bar:
            is_raw.append(t)
        else:
            oos_raw.append(t)
    # Purge IS: label must end BEFORE is_end_bar
    is_clean  = [t for t in is_raw
                 if int(t.get("label_end_bar",
                              t.get("bar_index", 0) + 20)) < is_end_bar]
    # Embargo OOS: entry must be AT OR AFTER is_end_bar + embargo_bars
    oos_clean = [t for t in oos_raw
                 if int(t.get("bar_index", 0)) >= is_end_bar + embargo_bars]
    return {
        "is_trades":    is_clean,
        "oos_trades":   oos_clean,
        "n_is_raw":     len(is_raw),
        "n_oos_raw":    len(oos_raw),
        "n_purged":     len(is_raw)  - len(is_clean),
        "n_embargoed":  len(oos_raw) - len(oos_clean),
        "embargo_bars": embargo_bars,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Outcome classification — separates PnL accounting from ML labels
# ─────────────────────────────────────────────────────────────────────────────
# Problem this solves: with Partial-mgmt (50% off at TP1 + move SL to BE),
# trades that hit TP1 then reverse to BE produce r_mult ≈ +0.498R. They are
# correctly counted as positive PnL (PF accounting), BUT labeling them as
# "WIN" for ML training is misleading — they are actually break-even outcomes
# of a strategy that almost can't lose once TP1 hits. Result: ML sees 100%
# wins on trending coins like REZ, can't train (single class), backtest looks
# invincible.
#
# Fix: classify outcomes into three buckets for ML purposes:
#   WIN     → clean profitable trade (r_mult > +threshold)
#   LOSS    → real loss (r_mult < -threshold)
#   NEUTRAL → essentially break-even (|r_mult| <= threshold) — excluded from ML
#
# The PF / WR computation continues to use r_mult directly, so reported
# backtest PnL doesn't change. Only the ML training set is filtered.
#
# Default threshold: 0.30R. Why? After Partial+BE, a "no-real-direction"
# outcome lands at +0.498R. Threshold 0.30R catches that as NEUTRAL while
# preserving genuine wins (TP2 hit → +1.498R) and genuine losses (Simple/SL
# direct hit → ≤ -0.998R) as WIN/LOSS.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_outcome(r_mult: float) -> str:
    """Return 'WIN' / 'LOSS' / 'NEUTRAL' based on r_mult and the ±threshold band.
    NEUTRAL trades are excluded from ML training but still contribute to PF."""
    if r_mult > NEUTRAL_R_THRESHOLD:
        return "WIN"
    if r_mult < -NEUTRAL_R_THRESHOLD:
        return "LOSS"
    return "NEUTRAL"


def _deep_limit_for(timeframe: str) -> int:
    """Return the deep-fetch bar limit for a timeframe."""
    interval = _BINANCE_INTERVAL.get(timeframe, "1d") if "_BINANCE_INTERVAL" in globals() else timeframe
    return _DEEP_FETCH_LIMITS.get(interval, 1000)


def _compute_decay_buckets(n_df: int) -> dict:
    """
    Adaptive time-decay bucket scheme based on total bars available.

    Returns dict:
      {
        "count":     int  (1..4),
        "weights":   list (oldest → newest, length == count),
        "edges":     list of (age_start, age_end)  where age is normalized
                     bar_index from newest (0.0) to oldest (1.0),
        "labels":    list of human-readable labels aligned with weights,
      }

    - n_df >= 400 : 4 buckets  [0.40, 0.60, 0.80, 1.00]
    - n_df >= 200 : 3 buckets  [0.50, 0.75, 1.00]
    - n_df >=  80 : 2 buckets  [0.60, 1.00]
    - n_df <   80 : 1 bucket   [1.00]
    """
    if n_df >= 400:
        return {
            "count":   4,
            "weights": [0.40, 0.60, 0.80, 1.00],   # oldest → newest
            "edges":   [(0.75, 1.00), (0.50, 0.75), (0.25, 0.50), (0.00, 0.25)],
            "labels":  ["Oldest 25%", "Older 25%", "Recent 25%", "Newest 25%"],
        }
    if n_df >= 200:
        return {
            "count":   3,
            "weights": [0.50, 0.75, 1.00],
            "edges":   [(0.667, 1.000), (0.333, 0.667), (0.000, 0.333)],
            "labels":  ["Oldest 33%", "Middle 33%", "Newest 33%"],
        }
    if n_df >= 80:
        return {
            "count":   2,
            "weights": [0.60, 1.00],
            "edges":   [(0.5, 1.0), (0.0, 0.5)],
            "labels":  ["Older 50%", "Newer 50%"],
        }
    return {
        "count":   1,
        "weights": [1.00],
        "edges":   [(0.0, 1.0)],
        "labels":  ["All bars"],
    }


def _bucket_stats_for_trades(trades_raw: list, n_df: int, buckets: dict,
                              current_regime_score: float = None) -> tuple:
    """
    Split trades across time buckets and compute weighted + per-bucket stats.

    Each trade must have 'bar_index' (entry bar) and 'r_mult'.
    age = (n_df - 1 - bar_index) / (n_df - 1)   # 0.0 newest, 1.0 oldest

    If current_regime_score is provided AND each trade has a 'regime_score'
    field, a regime-similarity weight (0.15 to 1.0) is multiplied into the
    time-decay weight when computing weighted EV/WR. Non-current-regime
    trades still contribute but with diminished influence — a "soft filter"
    that avoids the sample-size cliff of hard filtering. The per-bucket
    rows (unweighted WR/EV for each bucket) are UNAFFECTED so the user
    can still see the raw performance distribution.

    Returns (bucket_rows, weighted_ev, weighted_wr)
      bucket_rows: list of dicts with keys
        label, weight, n, wr, ev
    """
    if n_df <= 1 or not trades_raw:
        return ([], 0.0, 0.0)

    denom = float(n_df - 1)
    rows  = []
    # edges are listed oldest → newest in the buckets dict — keep that order
    for idx, (edge, w, lbl) in enumerate(zip(buckets["edges"], buckets["weights"], buckets["labels"])):
        lo, hi = edge
        sub = []
        for t in trades_raw:
            bi = t.get("bar_index")
            if bi is None:
                continue
            age = (n_df - 1 - bi) / denom
            # Include lower bound; include upper bound only for the oldest-most bucket
            in_range = (lo <= age < hi) or (idx == 0 and age == hi)
            if in_range:
                sub.append(t)
        if sub:
            rs  = [t["r_mult"] for t in sub]
            wr  = round(sum(1 for r in rs if r > 0) / len(rs) * 100, 1)
            ev  = round(float(np.mean(rs)), 3)
        else:
            wr, ev = 0.0, 0.0
        rows.append({
            "label":  lbl,
            "weight": w,
            "n":      len(sub),
            "wr":     wr,
            "ev":     ev,
        })

    # Weighted headline stats (sum of r_mult * weight / sum of weights used)
    # When current_regime_score is provided, multiply in regime similarity weight.
    _use_regime = current_regime_score is not None
    total_w, total_rw = 0.0, 0.0
    total_w_wins, total_w_all = 0.0, 0.0
    for t in trades_raw:
        bi = t.get("bar_index")
        if bi is None:
            continue
        age = (n_df - 1 - bi) / denom
        # Find matching bucket weight
        w = 1.0
        for idx, (edge, bw) in enumerate(zip(buckets["edges"], buckets["weights"])):
            lo, hi = edge
            if (lo <= age < hi) or (idx == 0 and age == hi):
                w = bw
                break
        # Multiply in regime similarity weight if available
        if _use_regime:
            rscore_hist = t.get("regime_score")
            if rscore_hist is not None:
                w *= _regime_similarity_weight(current_regime_score, rscore_hist)
        total_w   += w
        total_rw  += t["r_mult"] * w
        total_w_all  += w
        if t["r_mult"] > 0:
            total_w_wins += w

    weighted_ev = round(total_rw / total_w, 3)      if total_w    > 0 else 0.0
    weighted_wr = round(total_w_wins / total_w_all * 100, 1) if total_w_all > 0 else 0.0
    return (rows, weighted_ev, weighted_wr)


def _regime_similarity_weight(current_score: float, historical_score: float) -> float:
    """
    Smooth continuous similarity weight between current and historical regime
    scores (both on a 0-100 scale).

    - Exact match (diff=0):     weight = 1.00
    - Small diff (10 points):   weight = 0.90
    - Medium diff (30 points):  weight = 0.70
    - Large diff (50 points):   weight = 0.50
    - Max diff (100 points):    weight = 0.15 (floor)

    The 0.15 floor ensures opposite-regime trades still contribute some
    information rather than being zeroed out — we want graceful soft
    filtering, not hard filtering. This avoids the sample-size cliff on
    illiquid coins where hard regime filtering would leave 0 samples.

    Formula: max(0.15, 1 - abs(diff) / 100)
    """
    try:
        diff = abs(float(current_score) - float(historical_score))
    except (TypeError, ValueError):
        return 1.0   # missing data — don't penalize
    return max(0.15, 1.0 - diff / 100.0)


# ============================================================================
# SESSION DETECTION
# ============================================================================

def get_session(hour_wib: int) -> str:
    """Return trading session name for a given WIB hour (0-23)."""
    if hour_wib >= 20:
        return "NY+London"
    elif 15 <= hour_wib < 20:
        return "London"
    elif 7 <= hour_wib < 15:
        return "Asian"
    else:  # 0-6
        return "Dead Zone"


# ============================================================================
# DATA CLEANING UTILITIES
# ============================================================================

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex, lowercase columns, keep OHLCV, compute all derived cols."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in ["open","high","low","close","volume"] if c not in df.columns]
    if missing:
        print(f"[fetch] Missing columns: {missing}")
        return pd.DataFrame()
    df = df[["open","high","low","close","volume"]].copy()
    df.dropna(inplace=True)
    df["body"]         = df["close"] - df["open"]
    df["candle_range"] = df["high"]  - df["low"]
    # Avoid division by zero without a full replace pass
    cr = df["candle_range"].copy()
    cr[cr == 0] = float("nan")
    df["body_pct"]  = df["body"] / cr
    df["vol_avg_7"] = df["volume"].shift(1).rolling(7).mean()
    df["vol_mult"]  = df["volume"] / df["vol_avg_7"]
    # ATR(14) for trailing stop
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    # ── New computed fields ──────────────────────────────────────────────────
    # 1. ATR ratio: current ATR vs its 20-bar rolling average (volatility expansion)
    df["atr_ratio"] = df["atr14"] / df["atr14"].rolling(20).mean()
    # 2. Volume delta proxy: approximates buying vs selling pressure, 5-bar rolling sum
    close_pos = (df["close"] - df["low"]) / cr   # cr already has 0→NaN from above
    vol_delta = df["volume"] * (2 * close_pos - 1)
    df["vol_delta_5"] = vol_delta.rolling(5).sum()
    # 3. EMA stack with shift(1) to avoid lookahead bias
    df["ema5"]  = df["close"].shift(1).ewm(span=5,  adjust=False).mean()
    df["ema15"] = df["close"].shift(1).ewm(span=15, adjust=False).mean()
    df["ema21"] = df["close"].shift(1).ewm(span=21, adjust=False).mean()
    # 4. Candle rank: percentile rank of |body_pct| over 20 bars
    df["candle_rank_20"] = df["body_pct"].abs().rolling(20).rank(pct=True)
    # 5. Volume rank: percentile rank of volume over 20 bars
    df["vol_rank_20"] = df["volume"].rolling(20).rank(pct=True)
    # 6. vol_delta_20: 20-bar flow proxy (up-candles vs down-candles)
    df["vol_delta_20"] = vol_delta.rolling(20).sum()
    # 7. vol_delta_regime: vol_delta_5 relative to 20-bar mean (normalised flow)
    _vd5_mean = df["vol_delta_5"].rolling(20).mean()
    _vd5_std  = df["vol_delta_5"].rolling(20).std().replace(0, float("nan"))
    df["vol_delta_regime"] = (df["vol_delta_5"] - _vd5_mean) / _vd5_std
    # 8. body_vs_atr: absolute body size relative to ATR(14)
    #    Captures "explosiveness" — a 2% body in a 0.5% ATR regime is
    #    FAR more meaningful than a 2% body in a 3% ATR regime. body_pct
    #    alone (body/range) can't see this since it's scale-invariant.
    #    Typical values: 0.5 = normal candle, 1.5+ = large, 3.0+ = extreme.
    df["body_vs_atr"] = df["body"].abs() / df["atr14"].replace(0, float("nan"))
    # 9. dist_from_ema21_pct: signed % distance of close from EMA21.
    #    Positive = above mean, negative = below. Extreme stretch means
    #    mean-reversion risk — a long signal 8% above EMA21 is buying
    #    a blow-off top. Model can learn "go long, but not when stretched".
    df["dist_from_ema21_pct"] = ((df["close"] - df["ema21"]) / df["ema21"]) * 100
    return df


def trim_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = df.index[-1] - timedelta(days=days)
    return df[df.index >= cutoff].copy()


# ============================================================================
# INDICATORS
# ============================================================================

@st.cache_data(show_spinner=False)
def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ADX, DI+, DI- from OHLCV DataFrame.
    Returns a DataFrame with columns: adx, di_plus, di_minus — aligned to df.index.
    ADX = trend strength (direction-neutral, 0–100).
    DI+ > DI- = bullish trend. DI- > DI+ = bearish trend."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    up   = high - high.shift(1)
    down = low.shift(1) - low

    dm_plus  = pd.Series(np.where((up > down) & (up > 0),  up,   0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr_w    = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm( alpha=1/period, adjust=False).mean() / atr_w
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_w

    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, float("nan"))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return pd.DataFrame({"adx": adx, "di_plus": di_plus, "di_minus": di_minus},
                        index=df.index)


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Compute EMA(period) on close prices.
    Uses shift(1) so the EMA at bar N is computed from bars 0..N-1 only.
    This avoids lookahead bias — the current bar's close is NOT included
    in its own EMA calculation.
    Returns a Series aligned to df.index.
    """
    return df["close"].shift(1).ewm(span=period, adjust=False).mean()


# ============================================================================
# EXTERNAL MARKET DATA FETCHERS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed() -> dict:
    """
    Fetch current Fear & Greed index from Alternative.me (free API).
    Returns dict with 'value' (0-100) and 'classification' str.
    Falls back to neutral (50) on any error.
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=5
        )
        data = resp.json()
        entry = data["data"][0]
        return {
            "value":          int(entry["value"]),
            "classification": entry["value_classification"],
            "ok":             True,
        }
    except Exception:
        return {"value": 50, "classification": "Neutral", "ok": False}


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_historical_fng(n_days: int = 1200) -> dict:
    """
    Fetch N days of historical Fear & Greed values from alternative.me.
    Returns a dict: {"YYYY-MM-DD": int_value, ...} plus "ok" flag.

    Used by _scanner_train_ml to attach the F&G reading AT THE DATE of each
    historical training bar, turning "market-context regime" into an explicit
    ML feature. The current bar uses the live fetch_fear_greed() value.

    Why cached 6h: F&G updates once per day. A 6h cache is plenty fresh.
    Why 1200 days: covers all timeframes we fetch (1H/4H/1D max 1000 bars
    ≈ 1000 hours to 1000 days — the daily case is the longest span).
    """
    try:
        resp = requests.get(
            f"https://api.alternative.me/fng/?limit={int(n_days)}&format=json",
            timeout=15,
        )
        data = resp.json()
        out = {}
        for entry in data.get("data", []):
            # alternative.me returns unix timestamp as a STRING
            try:
                _ts = int(entry.get("timestamp", 0))
                _val = int(entry.get("value", 50))
                _date_key = pd.Timestamp(_ts, unit="s").strftime("%Y-%m-%d")
                out[_date_key] = _val
            except Exception:
                continue
        return {"map": out, "n": len(out), "ok": bool(out)}
    except Exception:
        return {"map": {}, "n": 0, "ok": False}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_btc_dominance() -> dict:
    """
    Fetch BTC dominance from CoinGecko global endpoint (free, no key).
    Returns dict with 'btc_d' (0-100 float) and 'ok' bool.
    """
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=8,
            headers={"Accept": "application/json"},
        )
        mkt = resp.json()["data"]["market_cap_percentage"]
        btc_d = float(mkt.get("btc", 50.0))
        return {"btc_d": btc_d, "ok": True}
    except Exception:
        return {"btc_d": 50.0, "ok": False}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_rate(symbol: str) -> dict:
    """
    Fetch latest perpetual funding rate.
    Tries Binance Futures → Bybit → OKX in order.
    Returns dict with 'rate' (float, e.g. 0.0001), 'ok' bool, 'source' str.
    """
    sym = symbol.upper()

    # ── 1. Binance Futures ────────────────────────────────────────────────────
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": sym, "limit": 1},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and "fundingRate" in data[0]:
                return {"rate": float(data[0]["fundingRate"]), "ok": True, "source": "binance"}
    except Exception:
        pass

    # ── 2. Bybit (linear perpetuals) ─────────────────────────────────────────
    try:
        resp = requests.get(
            "https://api.bybit.com/v5/market/funding/history",
            params={"category": "linear", "symbol": sym, "limit": 1},
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json()
            entries = data.get("result", {}).get("list", [])
            if entries:
                return {"rate": float(entries[0]["fundingRate"]), "ok": True, "source": "bybit"}
    except Exception:
        pass

    # ── 3. OKX (swap) ────────────────────────────────────────────────────────
    try:
        base = sym.replace("USDT", "")
        okx_inst = f"{base}-USDT-SWAP"
        resp = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": okx_inst},
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json()
            entries = data.get("data", [])
            if entries:
                return {"rate": float(entries[0]["fundingRate"]), "ok": True, "source": "okx"}
    except Exception:
        pass

    return {"rate": 0.0, "ok": False, "source": "none"}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_interest(symbol: str) -> dict:
    """
    Fetch open interest (current + 24h-ago delta).
    Tries Binance Futures → Bybit → OKX in order.
    Returns dict with 'oi_now', 'oi_24h_ago', 'oi_change_pct', 'ok', 'source'.
    """
    sym = symbol.upper()

    # ── 1. Binance Futures ────────────────────────────────────────────────────
    try:
        r_now = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": sym},
            timeout=5,
        )
        if r_now.status_code == 200:
            oi_now = float(r_now.json()["openInterest"])
            r_hist = requests.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": sym, "period": "1h", "limit": 25},
                timeout=5,
            )
            hist = r_hist.json() if r_hist.status_code == 200 else []
            if hist and isinstance(hist, list):
                oi_24h_ago    = float(hist[0]["sumOpenInterest"])
                oi_change_pct = (oi_now - oi_24h_ago) / max(oi_24h_ago, 1e-9) * 100
            else:
                oi_24h_ago, oi_change_pct = oi_now, 0.0
            return {"oi_now": oi_now, "oi_24h_ago": oi_24h_ago,
                    "oi_change_pct": oi_change_pct, "ok": True, "source": "binance"}
    except Exception:
        pass

    # ── 2. Bybit (linear perpetuals) ─────────────────────────────────────────
    try:
        # Bybit open-interest history: intervalTime=1h, limit=25 → 24h span
        resp = requests.get(
            "https://api.bybit.com/v5/market/open-interest",
            params={"category": "linear", "symbol": sym, "intervalTime": "1h", "limit": 25},
            timeout=7,
        )
        if resp.status_code == 200:
            result = resp.json().get("result", {}).get("list", [])
            if len(result) >= 2:
                oi_now      = float(result[0]["openInterest"])
                oi_24h_ago  = float(result[-1]["openInterest"])
                oi_change_pct = (oi_now - oi_24h_ago) / max(oi_24h_ago, 1e-9) * 100
                return {"oi_now": oi_now, "oi_24h_ago": oi_24h_ago,
                        "oi_change_pct": oi_change_pct, "ok": True, "source": "bybit"}
            elif len(result) == 1:
                oi_now = float(result[0]["openInterest"])
                return {"oi_now": oi_now, "oi_24h_ago": oi_now,
                        "oi_change_pct": 0.0, "ok": True, "source": "bybit"}
    except Exception:
        pass

    # ── 3. OKX (swap) ────────────────────────────────────────────────────────
    try:
        base    = sym.replace("USDT", "")
        okx_inst = f"{base}-USDT-SWAP"
        # Current OI
        r1 = requests.get(
            "https://www.okx.com/api/v5/public/open-interest",
            params={"instType": "SWAP", "instId": okx_inst},
            timeout=7,
        )
        if r1.status_code == 200:
            oi_data = r1.json().get("data", [])
            if oi_data:
                oi_now = float(oi_data[0]["oi"])
                return {"oi_now": oi_now, "oi_24h_ago": oi_now,
                        "oi_change_pct": 0.0, "ok": True, "source": "okx"}
    except Exception:
        pass

    return {"oi_now": 0.0, "oi_24h_ago": 0.0, "oi_change_pct": 0.0, "ok": False, "source": "none"}


# ============================================================================
# KLINES FETCHERS (Binance + Gate.io)
# ============================================================================

def _binance_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    Raw (uncached) Binance klines with backward pagination.
    Tries api.binance.com first; falls back to data-api.binance.vision.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    end_ms   = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    for url, verify in _BINANCE_KLINES_URLS:
        all_klines: list = []
        batch_end = end_ms
        success   = True

        while True:
            try:
                resp = requests.get(url, params={
                    "symbol": symbol, "interval": interval,
                    "endTime": batch_end, "limit": 1000,
                }, timeout=15, verify=verify)
                if resp.status_code != 200:
                    print(f"[Binance] {url} HTTP {resp.status_code} — trying next URL")
                    success = False
                    break
                klines = resp.json()
                if not klines:
                    break
                all_klines = klines + all_klines
                earliest_ts = klines[0][0]
                if earliest_ts <= start_ms or len(klines) < 1000:
                    break
                batch_end = earliest_ts - 1
            except Exception as e:
                print(f"[Binance] {url} error: {e} — trying next URL")
                success = False
                break

        if success and all_klines:
            print(f"[Binance] fetched {len(all_klines)} candles via {url}")
            df = pd.DataFrame(all_klines, columns=[
                "ts", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "n_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
            return _clean_df(df[df.index >= cutoff])

    return pd.DataFrame()


def _gateio_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    Gate.io klines fetch with backward pagination.
    symbol must be in Binance format (e.g. BTCUSDT) — converted internally to BTC_USDT.
    Gate.io format: [ts_s, quote_vol, close, high, low, open, base_vol, closed]
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Convert BTCUSDT → BTC_USDT
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    pair = f"{base}_USDT"

    url       = "https://api.gateio.ws/api/v4/spot/candlesticks"
    end_s     = int(datetime.utcnow().timestamp())
    start_s   = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    all_rows: list = []
    batch_end = end_s

    while True:
        try:
            resp = requests.get(url, params={
                "currency_pair": pair, "interval": interval,
                "to": batch_end, "limit": 1000,
            }, timeout=15, verify=False)
            if resp.status_code != 200:
                print(f"[Gate.io] HTTP {resp.status_code} for {pair}: {resp.text[:100]}")
                break
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            all_rows = batch + all_rows
            earliest_s = int(batch[0][0])
            if earliest_s <= start_s or len(batch) < 1000:
                break
            batch_end = earliest_s - 1
        except Exception as e:
            print(f"[Gate.io] Error: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    # Gate.io columns: [ts, quote_vol, close, high, low, open, base_vol, closed]
    df = pd.DataFrame(all_rows, columns=[
        "ts", "quote_vol", "close", "high", "low", "open", "volume", "closed"
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="s")
    df.set_index("ts", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    df = df[df.index >= cutoff]
    return _clean_df(df)


@st.cache_data(ttl=1800, show_spinner=False)
def _binance_fetch(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Cached fetch: Binance (main→mirror), Gate.io fallback for unlisted symbols."""
    interval = _BINANCE_INTERVAL.get(timeframe, "1d")
    df = _binance_klines(symbol, interval, days)
    if not df.empty:
        return df
    print(f"[Gate.io] fallback for {symbol} @ {interval} ({days}d)")
    return _gateio_klines(symbol, interval, days)


def fetch_live(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch fresh (uncached) recent candles for the live scanner.
    Tries Binance first; falls back to Gate.io for altcoins not on Binance."""
    live_days = {"1D": 30, "4H": 14, "1H": 5}
    days      = live_days.get(timeframe, 30)
    interval  = _BINANCE_INTERVAL.get(timeframe, "1d")
    df = _binance_klines(symbol, interval, days)
    if not df.empty:
        return df
    print(f"[Gate.io] live fallback for {symbol} @ {interval} ({days}d)")
    return _gateio_klines(symbol, interval, days)


# ============================================================================
# SCANNER HELPERS
# ============================================================================

@st.cache_data(show_spinner=False, ttl=600)
def _scanner_btc_regime_for_combos() -> str:
    """
    Compute the current BTC regime tag (BULL / BEAR / CHOP) for QuantFlow combo
    classification. MUST match exactly the regime definition used by the audit
    (oos_audit_v3a/v3c/v3d) so combo *-A alignment classification stays
    consistent between backtest and live scanner.

    Audit definition:
      Compute EMA50 of BTC daily close.
      BULL  if BTC close > EMA50 × 1.02
      BEAR  if BTC close < EMA50 × 0.98
      CHOP  otherwise (within ±2% band)

    Cached 10 minutes — daily-close based so doesn't need to be fresh per scan.
    Returns "UNKNOWN" if BTC fetch fails (combo *-A variants will then return
    no matches; user can switch to *-N variants which don't depend on regime).
    """
    try:
        df = _binance_klines("BTCUSDT", "1d", days=200)
        if df is None or df.empty or len(df) < 60:
            return "UNKNOWN"
        close = df["close"]
        # ewm with .shift(1) to keep parity with audit features (lookahead-safe).
        # Note: the audit's compute_btc_regime_series did NOT shift here because
        # it tagged trades by their signal_dt which is at-or-before close. For
        # the LIVE classifier we want "what's the current regime as of the
        # latest CLOSED candle" so we use the most recent close vs ema50 of
        # closes up to and including that bar. No shift needed in this path.
        ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        c, e = float(close.iloc[-1]), float(ema50.iloc[-1])
        if not (c > 0 and e > 0): return "UNKNOWN"
        if c > e * 1.02: return "BULL"
        if c < e * 0.98: return "BEAR"
        return "CHOP"
    except Exception:
        return "UNKNOWN"


@st.cache_data(show_spinner=False, ttl=300)
def _scanner_get_universe(min_volume_usdt: float) -> list:
    """
    Fetch all Binance USDT spot pairs with 24h quoteVolume >= min_volume_usdt.
    Returns list of dicts sorted by volume desc: {symbol, volume_24h, price}.
    Result cached 5 minutes so repeated scans don't re-fetch.
    """
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            timeout=15,
        )
        resp.raise_for_status()
        tickers = resp.json()
    except Exception:
        # Mirror fallback
        try:
            resp = requests.get(
                "https://data-api.binance.vision/api/v3/ticker/24hr",
                timeout=15,
                verify=False,
            )
            tickers = resp.json()
        except Exception:
            return []

    universe = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if base in _SCANNER_EXCLUDE:
            continue
        try:
            vol = float(t.get("quoteVolume", 0))
        except Exception:
            continue
        if vol < min_volume_usdt:
            continue
        universe.append({
            "symbol":     sym,
            "volume_24h": vol,
            "price":      float(t.get("lastPrice", 0)),
        })

    universe.sort(key=lambda x: x["volume_24h"], reverse=True)
    return universe


def _scanner_get_universe_all() -> list:
    """
    Fetch ALL Binance USDT-margined perpetuals (no volume gate, no top-N cap).
    Returns list of dicts sorted by volume desc: {symbol, volume_24h, price}.
    Covers ~340 symbols as of Phase 4 (May 2026).
    """
    return _scanner_get_universe(min_volume_usdt=0.0)


def _scanner_fetch_candles(
    symbol: str,
    interval: str,
    limit: int = 200,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """
    Fetch last `limit` klines for symbol/interval from Binance.
    Returns cleaned DataFrame or empty DataFrame on failure.
    No caching — called inside thread workers.

    Args:
        timeout: per-request HTTP timeout in seconds. The SMC scanner passes
                 a short value (e.g. 5s) so a slow/hung coin fails fast and
                 gets skipped instead of stalling the whole scan.
    """
    urls = [
        ("https://api.binance.com/api/v3/klines",         True),
        ("https://data-api.binance.vision/api/v3/klines",  False),
    ]
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    for url, verify in urls:
        try:
            resp = requests.get(
                url,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=timeout,
                verify=verify,
            )
            if resp.status_code != 200:
                continue
            klines = resp.json()
            if len(klines) < 20:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "taker_buy_base", "tbqav", "ignore",
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Compute taker buy ratio (handle division by zero → 0.5)
            df["taker_buy_ratio"] = df.apply(
                lambda r: r["taker_buy_base"] / r["volume"] if r["volume"] > 0 else 0.5, axis=1
            )

            df = _clean_df(df)
            return df if not df.empty else pd.DataFrame()

        except Exception:
            continue

    return pd.DataFrame()
