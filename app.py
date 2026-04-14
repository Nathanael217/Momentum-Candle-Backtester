"""
Market Scanner — AutoFinder
Scans all liquid Binance altcoins for live momentum signals.
Provides Backtest, WFO Mini-Validation, ML Probability, and AI Final Verdict.

Run standalone:  streamlit run app_autofinder.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ─── sklearn (optional — falls back to heuristic if missing) ──────────────────
try:
    from sklearn.linear_model    import LogisticRegression
    from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing   import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline        import Pipeline
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

# ─── Deep fetch limits for backtest / WFO / ML training ──────────────────────
# Binance /api/v3/klines caps at 1000 bars per call. These values are used by
# _scanner_quick_backtest, _scanner_mini_wfo, and _scanner_train_ml so they
# all pull the same historical depth.
_DEEP_FETCH_LIMITS = {
    "1h": 1000,   # ~41 days
    "2h": 1000,   # ~83 days
    "4h": 1000,   # ~166 days
    "6h": 1000,   # ~250 days
    "12h": 1000,  # ~500 days
    "1d": 1000,   # ~2.7 years
}

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


def _bucket_stats_for_trades(trades_raw: list, n_df: int, buckets: dict) -> tuple:
    """
    Split trades across time buckets and compute weighted + per-bucket stats.

    Each trade must have 'bar_index' (entry bar) and 'r_mult'.
    age = (n_df - 1 - bar_index) / (n_df - 1)   # 0.0 newest, 1.0 oldest

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
        total_w   += w
        total_rw  += t["r_mult"] * w
        total_w_all  += w
        if t["r_mult"] > 0:
            total_w_wins += w

    weighted_ev = round(total_rw / total_w, 3)      if total_w    > 0 else 0.0
    weighted_wr = round(total_w_wins / total_w_all * 100, 1) if total_w_all > 0 else 0.0
    return (rows, weighted_ev, weighted_wr)

st.set_page_config(
    page_title="Market Scanner — AutoFinder",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130; border: 1px solid #2d3250;
        border-radius: 8px; padding: 16px 20px; margin: 4px 0;
    }
    .metric-label { color: #8892b0; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ccd6f6; font-size: 24px; font-weight: 700; margin-top: 4px; }
    .metric-value.green { color: #64ffda; }
    .metric-value.red   { color: #ff6b6b; }
    .signal-card {
        background: #0d1f0d; border: 1px solid #238636;
        border-radius: 8px; padding: 16px 20px; margin: 12px 0;
        font-family: monospace;
    }
    .signal-card h4 { color: #3fb950; margin: 0 0 10px 0; }
    .signal-line { color: #ccd6f6; padding: 2px 0; font-size: 13px; }
    .signal-line span { color: #64ffda; font-weight: 600; }
    div[data-testid="stTabs"] button { font-size: 14px; font-weight: 600; }
    .main .block-container,
    section[data-testid="stSidebar"] { transition: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session Detection ────────────────────────────────────────────────────────

# ─── Session Detection ────────────────────────────────────────────────────────

WIB_OFFSET = timedelta(hours=7)

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


# ─── Data Cleaning & Indicators ──────────────────────────────────────────────

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
    return df


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




# ─── Market Context API Helpers ───────────────────────────────────────────────

# ─── Market Context API Helpers ───────────────────────────────────────────────

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



# ─── Regime Scoring ───────────────────────────────────────────────────────────

def calculate_regime_score(df, bar_index, direction, adx_df,
                           htf_ema_series=None, timeframe="1D", ticker="",
                           fear_greed_data=None, btc_dom_data=None):
    """
    Compute a 0-100 regime score from 7 components:
    - ADX(14): 30 points max
    - ATR Ratio: 25 points max
    - EMA/HTF alignment: 25 points max
    - Session: 15 points max (intraday only, redistributed for daily)
    - DI Gap: 5 points max
    - Volume Delta modifier: ±3
    - Fear & Greed modifier: ±10 (NEW)
    - BTC Dominance filter for altcoins (NEW)

    Returns dict with: score (0-100), verdict (GREEN/YELLOW/RED),
    breakdown_line (string), flip_condition (string), hard_overrides (list)
    """
    import datetime as _dt

    is_daily  = timeframe in ("1D", "1W")
    is_crypto = str(ticker).upper().endswith("USDT")

    # ── Resolve bar ────────────────────────────────────────────────────────────
    try:
        bar = df.iloc[bar_index]
    except (IndexError, TypeError):
        bar = df.iloc[-1]

    close      = float(bar.get("close",      0))
    atr        = float(bar.get("atr14",      0) or 0)
    atr_ratio  = float(bar.get("atr_ratio",  1) or 1)
    ema5       = float(bar.get("ema5",       close) or close)
    ema15      = float(bar.get("ema15",      close) or close)
    ema21      = float(bar.get("ema21",      close) or close)
    vol_delta5 = float(bar.get("vol_delta_5", 0) or 0)
    bar_ts     = df.index[bar_index] if bar_index < len(df) else df.index[-1]

    adx_val    = float(adx_df["adx"].iloc[bar_index])      if adx_df is not None and "adx"      in adx_df.columns else 0
    di_plus    = float(adx_df["di_plus"].iloc[bar_index])  if adx_df is not None and "di_plus"  in adx_df.columns else 0
    di_minus   = float(adx_df["di_minus"].iloc[bar_index]) if adx_df is not None and "di_minus" in adx_df.columns else 0

    # ADX 3-bars-ago for declining check
    adx_3ago   = 0.0
    if adx_df is not None and "adx" in adx_df.columns and bar_index >= 3:
        adx_3ago = float(adx_df["adx"].iloc[bar_index - 3])

    # ATR ratio 10-bars-ago for compression-to-expansion bonus
    atr_ratio_10ago = 1.0
    if bar_index >= 10 and "atr_ratio" in df.columns:
        atr_ratio_10ago = float(df["atr_ratio"].iloc[bar_index - 10] or 1)

    # ATR ratio streak > 1.5 check (last 10 bars)
    atr_high_streak = 0
    if "atr_ratio" in df.columns and bar_index >= 10:
        atr_high_streak = int(
            (df["atr_ratio"].iloc[max(0, bar_index - 10):bar_index + 1] > 1.5).sum()
        )

    # ── 1. ADX score (0-30) ────────────────────────────────────────────────────
    if adx_val < 15:
        adx_pts = 0
    elif adx_val < 20:
        adx_pts = 8
    elif adx_val < 25:
        adx_pts = 18
    elif adx_val < 30:
        adx_pts = 28
    elif adx_val <= 40:
        adx_pts = 30
    else:
        adx_pts = 25   # overheated penalty

    adx_declining = adx_val > 25 and adx_3ago > 0 and adx_val < adx_3ago
    if adx_declining:
        adx_pts -= 5

    adx_max = 30

    # ── 2. ATR Ratio score (0-25) ──────────────────────────────────────────────
    if atr_ratio < 0.6:
        atr_pts = 5
    elif atr_ratio < 0.8:
        atr_pts = 12
    elif atr_ratio < 1.0:
        atr_pts = 18
    elif atr_ratio < 1.5:
        atr_pts = 25
    elif atr_ratio < 2.0:
        atr_pts = 20
    else:
        atr_pts = 10

    # Compression→expansion bonus
    if atr_ratio > 1.0 and atr_ratio_10ago < 0.8:
        atr_pts = min(25, atr_pts + 5)
    # Prolonged overheated penalty
    if atr_high_streak >= 10:
        atr_pts = max(0, atr_pts - 5)

    atr_max = 25

    # ── 3. EMA / HTF alignment score (0-25) ───────────────────────────────────
    # EMA stack: ema5 > ema15 > ema21 for long; reverse for short
    if direction == "long":
        stack_full    = ema5 > ema15 and ema15 > ema21
        stack_partial = (ema5 > ema15) or (ema15 > ema21)
    else:
        stack_full    = ema5 < ema15 and ema15 < ema21
        stack_partial = (ema5 < ema15) or (ema15 < ema21)

    stack_pts = 10 if stack_full else (5 if stack_partial else 0)

    # HTF EMA
    htf_pts   = 0
    htf_score = 0
    if htf_ema_series is not None:
        try:
            htf_ema_val = float(htf_ema_series.reindex([bar_ts], method="ffill").iloc[0])
        except Exception:
            htf_ema_val = None

        if htf_ema_val is not None and htf_ema_val > 0 and atr > 0:
            dist = close - htf_ema_val
            if direction == "long":
                on_correct_side = dist > 0
                within_1atr     = abs(dist) <= atr
            else:
                on_correct_side = dist < 0
                within_1atr     = abs(dist) <= atr

            if on_correct_side:
                htf_pts   = 10
                htf_score = 10
            elif within_1atr:
                htf_pts   = 5
                htf_score = 5
            # else 0
    else:
        htf_score = 5   # neutral when no HTF data

    # Cross-TF agreement bonus
    cross_tf_pts = 5 if (stack_pts >= 5 and htf_score >= 5) else 0

    ema_pts = min(25, stack_pts + htf_pts + cross_tf_pts)
    ema_max = 25

    # ── 4. Session score (0-15) ────────────────────────────────────────────────
    sess_pts = 0
    sess_max = 15
    if is_daily:
        sess_pts = 0
        # Redistribute 15 pts: +5 to each of ADX, ATR, EMA caps
        adx_max  = 35
        atr_max  = 30
        ema_max  = 30
        adx_pts  = min(adx_max, adx_pts)
        atr_pts  = min(atr_max, atr_pts)
        ema_pts  = min(ema_max, ema_pts)
    else:
        # Determine WIB hour from bar timestamp
        try:
            if hasattr(bar_ts, "to_pydatetime"):
                _naive = bar_ts.to_pydatetime()
            else:
                _naive = bar_ts
            # Binance timestamps are UTC; WIB = UTC+7
            wib_hour = (_naive.hour + 7) % 24
        except Exception:
            wib_hour = 12

        sess_name = get_session(wib_hour)

        if is_crypto:
            sess_pts = 7 if sess_name == "Dead Zone" else 10
        else:
            _sess_map = {"NY+London": 15, "London": 13, "Asian": 4, "Dead Zone": 2}
            sess_pts  = _sess_map.get(sess_name, 4)

    # ── 5. DI Gap score (0-5) ─────────────────────────────────────────────────
    di_gap = di_plus - di_minus
    if direction == "long":
        di_aligned = di_plus > di_minus
        gap_abs    = di_gap
    else:
        di_aligned = di_minus > di_plus
        gap_abs    = -di_gap

    if di_aligned and gap_abs >= 15:
        di_pts = 5
    elif di_aligned and gap_abs >= 5:
        di_pts = 3
    elif abs(di_gap) < 5:
        di_pts = 1
    else:
        di_pts = 0   # opposed

    # ── 6. Volume delta modifier (±3) ─────────────────────────────────────────
    if direction == "long":
        vol_mod = 3 if vol_delta5 > 0 else (-3 if vol_delta5 < 0 else 0)
    else:
        vol_mod = 3 if vol_delta5 < 0 else (-3 if vol_delta5 > 0 else 0)

    # ── 7. Fear & Greed modifier (±10) ────────────────────────────────────────
    fg_val = 50
    fg_label = "Neutral"
    fg_mod = 0
    if fear_greed_data and fear_greed_data.get("ok"):
        fg_val   = int(fear_greed_data.get("value", 50))
        fg_label = fear_greed_data.get("classification", "Neutral")
        if fg_val < 20:
            fg_mod = -10   # Extreme Fear → wider stop-hunts, kills momentum
        elif fg_val > 75:
            fg_mod = 8 if direction == "long" else -8   # Greed → favour longs
        else:
            fg_mod = 0

    # ── 8. BTC Dominance altcoin penalty ──────────────────────────────────────
    btc_dom_penalty = 0
    btc_d_val = 50.0
    btc_dom_rising = False
    _is_btc = str(ticker).upper() in ("BTCUSDT", "BTC")
    if not _is_btc and btc_dom_data and btc_dom_data.get("ok"):
        btc_d_val = float(btc_dom_data.get("btc_d", 50.0))
        btc_dom_rising = bool(btc_dom_data.get("rising", False))
        if btc_d_val > 56 and btc_dom_rising and direction == "long":
            btc_dom_penalty = -8   # Capital rotating into BTC → altcoin longs weaker

    # ── Total ──────────────────────────────────────────────────────────────────
    raw_score = (adx_pts + atr_pts + ema_pts + sess_pts + di_pts
                 + vol_mod + fg_mod + btc_dom_penalty)
    score     = max(0, min(100, raw_score))

    # ── Hard overrides ─────────────────────────────────────────────────────────
    hard_overrides = []

    if atr_ratio > 3.0:
        hard_overrides.append(f"ATR Ratio {atr_ratio:.1f} > 3.0 — extreme volatility")

    if not is_crypto and not is_daily:
        try:
            if hasattr(bar_ts, "to_pydatetime"):
                _bdt = bar_ts.to_pydatetime()
            else:
                _bdt = bar_ts
            _wib_hour = (_bdt.hour + 7) % 24
            if _bdt.weekday() == 4 and _wib_hour >= 16:   # Friday WIB ≥ 16:00
                hard_overrides.append("Friday 16:00+ WIB — liquidity drying up")
        except Exception:
            pass

    if htf_ema_series is not None and htf_score == 0 and atr > 0:
        try:
            htf_ema_val2 = float(htf_ema_series.reindex([bar_ts], method="ffill").iloc[0])
            if abs(close - htf_ema_val2) > 2 * atr:
                hard_overrides.append("Counter-HTF extreme: price > 2×ATR from HTF EMA")
        except Exception:
            pass

    verdict = "RED"
    if not hard_overrides:
        if score >= 70:
            verdict = "GREEN"
        elif score >= 45:
            verdict = "YELLOW"

    # ── Breakdown line ─────────────────────────────────────────────────────────
    def _icon(pts, max_pts):
        ratio = pts / max_pts if max_pts > 0 else 0
        return "✅" if ratio >= 0.7 else ("⚠️" if ratio >= 0.35 else "❌")

    adx_icon  = _icon(adx_pts,  adx_max)
    atr_icon  = _icon(atr_pts,  atr_max)
    ema_icon  = _icon(ema_pts,  ema_max)
    sess_icon = _icon(sess_pts, sess_max) if not is_daily else "—"
    di_icon   = _icon(di_pts,   5)

    fg_mod_str   = f"{'+' if fg_mod >= 0 else ''}{fg_mod}"
    btc_pen_str  = f"{'+' if btc_dom_penalty >= 0 else ''}{btc_dom_penalty}" if btc_dom_penalty != 0 else "—"

    breakdown_line = (
        f"ADX: {adx_val:.1f} {adx_icon} ({adx_pts}/{adx_max}) | "
        f"ATR×: {atr_ratio:.2f} {atr_icon} ({atr_pts}/{atr_max}) | "
        f"EMA: {ema_icon} ({ema_pts}/{ema_max}) | "
        f"Session: {sess_icon} ({sess_pts}/{sess_max}) | "
        f"DI: {di_icon} ({di_pts}/5) | "
        f"VolΔ: {'+' if vol_mod >= 0 else ''}{vol_mod} | "
        f"F&G: {fg_val} ({fg_label}) {fg_mod_str} | "
        f"BTC.D: {btc_d_val:.1f}% {btc_pen_str}"
    )

    # ── Flip condition ─────────────────────────────────────────────────────────
    flip_condition = ""
    if verdict == "RED" and adx_val < 20:
        flip_condition = f"ADX crosses 20 (currently {adx_val:.1f})"
    elif verdict == "YELLOW" and adx_val < 25:
        needed = 25 - adx_val
        flip_condition = f"ADX crosses 25 (currently {adx_val:.1f}, needs +{needed:.1f})"
    elif verdict == "GREEN" and adx_declining:
        flip_condition = f"Watch: ADX declining. Below 25 → YELLOW."

    return {
        "score":            score,
        "verdict":          verdict,
        "breakdown_line":   breakdown_line,
        "flip_condition":   flip_condition,
        "hard_overrides":   hard_overrides,
        # component breakdown for callers that want raw values
        "adx_pts":          adx_pts,
        "atr_pts":          atr_pts,
        "ema_pts":          ema_pts,
        "sess_pts":         sess_pts,
        "di_pts":           di_pts,
        "vol_mod":          vol_mod,
        # new market-context fields
        "fg_val":           fg_val,
        "fg_label":         fg_label,
        "fg_mod":           fg_mod,
        "btc_d_val":        btc_d_val,
        "btc_dom_rising":   btc_dom_rising,
        "btc_dom_penalty":  btc_dom_penalty,
    }



# ─── Binance / Gate.io Data Fetch ─────────────────────────────────────────────

_BINANCE_INTERVAL = {"1D": "1d", "4H": "4h", "1H": "1h", "1W": "1w"}

# Higher timeframe mapping for ADX context
_HTF_MAP = {"1H": "4H", "4H": "1D", "1D": "1W"}
_HTF_LABEL = {"1H": "4H", "4H": "Daily", "1D": "Weekly"}

_BINANCE_KLINES_URLS = [
    ("https://api.binance.com/api/v3/klines",          True),   # main — verify SSL
    ("https://data-api.binance.vision/api/v3/klines",  False),  # mirror — ISP-bypass
]


def _binance_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    Raw (uncached) Binance klines with backward pagination.
    Tries api.binance.com first; falls back to data-api.binance.vision.
    """
    import urllib3
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
    import urllib3
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


def trim_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = df.index[-1] - timedelta(days=days)
    return df[df.index >= cutoff].copy()

# ─── Candle Detection ──────────────────────────────────────────────────────────


# ─── Market Scanner (AutoFinder) ──────────────────────────────────────────────

# ─── Auto Analyzer ────────────────────────────────────────────────────────────


# ─── Market Scanner (replaces Auto Finder) ────────────────────────────────────

# Stablecoins and wrapped tokens to exclude from altcoin scan
_SCANNER_EXCLUDE = {
    "USDT", "BUSD", "USDC", "TUSD", "DAI", "FDUSD", "USDP", "USDD",
    "PYUSD", "AEUR", "EURI",
    "WBTC", "WETH", "WBETH",
}

# Scoring weights — must sum to 100
_SCORE_WEIGHTS = {
    "body":    25,   # candle conviction
    "volume":  20,   # institutional participation
    "adx":     20,   # trend strength
    "regime":  25,   # market environment
    "recency": 10,   # how fresh the signal is (candle 0 = most recent closed)
}


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


def _scanner_fetch_candles(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetch last `limit` klines for symbol/interval from Binance.
    Returns cleaned DataFrame or empty DataFrame on failure.
    No caching — called inside thread workers.
    """
    urls = [
        ("https://api.binance.com/api/v3/klines",         True),
        ("https://data-api.binance.vision/api/v3/klines",  False),
    ]
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    for url, verify in urls:
        try:
            resp = requests.get(
                url,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10,
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


def _compute_enhanced_trade_plan(
    direction: str,
    close_px: float,
    open_px: float,
    high_px: float,
    low_px: float,
    atr14: float,
    body_pct: float,
) -> dict:
    """
    Compute a multi-zone trade plan that is:
    - ATR-adaptive (SL scales with coin volatility, not fixed %)
    - Structure-anchored (SL placed outside candle high/low, not flat %)
    - Entry-tiered (3 zones: aggressive at close, standard on retrace, sniper at 61.8% fib)
    - Multi-TP with partial-exit management guidance

    Returns a dict with entry zones, SL, TP1/TP2/TP3, R:R per zone, and
    management instructions.
    """
    if close_px <= 0:
        return {}

    body_size  = abs(close_px - open_px)
    candle_rng = high_px - low_px if high_px > low_px else close_px * 0.01

    # ── ATR-based stop distance ───────────────────────────────────────────────
    # Use 1.0× ATR14 as the base volatility buffer behind the candle structure.
    # For very low-ATR coins clamp to 0.8% minimum; for very high-ATR coins
    # clamp to 6% maximum so we don't get absurd stops.
    atr_buffer  = atr14 if atr14 > 0 else close_px * 0.02
    atr_pct     = atr_buffer / close_px

    if direction == "long":
        # Structural anchor = candle low; add 0.5× ATR buffer below it
        struct_sl = low_px  - atr_buffer * 0.5
        # Clamp: SL must be positive and within 0.8%–6% of close
        struct_sl = max(struct_sl, close_px * 0.94)   # never more than 6% away
        struct_sl = min(struct_sl, close_px * 0.992)  # never tighter than 0.8%
        sl_dist   = max(0.008, min(0.06, (close_px - struct_sl) / close_px))
    else:
        struct_sl = high_px + atr_buffer * 0.5
        struct_sl = min(struct_sl, close_px * 1.06)
        struct_sl = max(struct_sl, close_px * 1.008)
        sl_dist   = max(0.008, min(0.06, (struct_sl - close_px) / close_px))

    # ── Entry zones ───────────────────────────────────────────────────────────
    # Aggressive  = enter right at candle close (fills immediately, worst R:R)
    # Standard    = wait for 38.2% retrace into the candle body
    # Sniper      = wait for 61.8% Fib retrace (best R:R, lower fill probability)
    fib_382 = body_size * 0.382
    fib_618 = body_size * 0.618

    if direction == "long":
        agg_entry      = round(close_px, 8)
        standard_entry = round(close_px - fib_382, 8)
        sniper_entry   = round(close_px - fib_618, 8)
        # Clamp sniper entry so it never goes below candle open (that's a full reversal)
        sniper_entry   = max(sniper_entry, round(open_px * 1.002, 8))
    else:
        agg_entry      = round(close_px, 8)
        standard_entry = round(close_px + fib_382, 8)
        sniper_entry   = round(close_px + fib_618, 8)
        sniper_entry   = min(sniper_entry, round(open_px * 0.998, 8))

    # ── Zone validity check ───────────────────────────────────────────────────
    # For SHORT: entry must be BELOW struct_sl  (SL is above entry; short logic).
    # For LONG:  entry must be ABOVE struct_sl  (SL is below entry; long logic).
    #
    # When a large-body candle's Fibonacci retrace zone overshoots the structural
    # SL level, the resulting trade plan is physically impossible: the entry fill
    # would be past your own invalidation level, making the risk calculation and
    # every TP derived from it nonsensical (TP1 literally equals the SL price).
    #
    # Detection:
    #   SHORT std invalid  → standard_entry  >= struct_sl
    #   SHORT sniper invalid → sniper_entry  >= struct_sl
    #   LONG  std invalid  → standard_entry  <= struct_sl
    #   LONG  sniper invalid → sniper_entry  <= struct_sl
    #
    # Resolution: mark zone invalid and clamp entry to just inside the SL
    # (0.05% buffer) so _tps() produces a tiny-but-finite R rather than TP=SL.
    # The validity flags are returned so the display can warn the user.
    _eps = struct_sl * 0.0005   # 0.05% inside SL

    if direction == "short":
        std_valid    = standard_entry < struct_sl
        sniper_valid = sniper_entry   < struct_sl
        if not std_valid:
            standard_entry = round(struct_sl - _eps, 8)
        if not sniper_valid:
            sniper_entry   = round(struct_sl - _eps, 8)
    else:  # long
        std_valid    = standard_entry > struct_sl
        sniper_valid = sniper_entry   > struct_sl
        if not std_valid:
            standard_entry = round(struct_sl + _eps, 8)
        if not sniper_valid:
            sniper_entry   = round(struct_sl + _eps, 8)

    # ── SL per entry zone ─────────────────────────────────────────────────────
    # All zones share the same structural SL (candle low/high ± 0.5×ATR).
    # Only the entry price varies — Standard/Sniper entries are closer to the
    # structural SL, so their dollar risk is smaller → genuinely better R:R.
    # (Previously sl_dist% was re-applied per entry, pushing the Standard/Sniper
    # SL *below* the structural anchor and making risk inconsistent.)
    sl_agg      = round(struct_sl, 8)
    sl_standard = round(struct_sl, 8)
    sl_sniper   = round(struct_sl, 8)

    # ── Take-profit levels (1R / 2R / 3R) per entry ──────────────────────────
    def _tps(entry, sl):
        risk = abs(entry - sl)
        if direction == "long":
            return (
                round(entry + 1.0 * risk, 8),
                round(entry + 2.0 * risk, 8),
                round(entry + 3.0 * risk, 8),
            )
        else:
            return (
                round(entry - 1.0 * risk, 8),
                round(entry - 2.0 * risk, 8),
                round(entry - 3.0 * risk, 8),
            )

    tp1_agg, tp2_agg, tp3_agg           = _tps(agg_entry,      sl_agg)
    tp1_std, tp2_std, tp3_std           = _tps(standard_entry, sl_standard)
    tp1_sniper, tp2_sniper, tp3_sniper  = _tps(sniper_entry,   sl_sniper)

    # ── R:R to TP2 (headline metric) ─────────────────────────────────────────
    def _rr2(entry, sl):
        risk = abs(entry - sl)
        return 2.0  # always 2R by definition

    # ── Summary label for SL method ──────────────────────────────────────────
    sl_method = f"ATR-adaptive ({sl_dist*100:.1f}% — 1×ATR below/above candle structure)"

    return {
        # Aggressive zone (enter at close — legacy behaviour)
        "agg_entry":   agg_entry,
        "agg_sl":      sl_agg,
        "agg_tp1":     tp1_agg,
        "agg_tp2":     tp2_agg,
        "agg_tp3":     tp3_agg,
        # Standard zone (38.2% retrace)
        "std_entry":   standard_entry,
        "std_sl":      sl_standard,
        "std_tp1":     tp1_std,
        "std_tp2":     tp2_std,
        "std_tp3":     tp3_std,
        # Sniper zone (61.8% retrace)
        "sniper_entry": sniper_entry,
        "sniper_sl":    sl_sniper,
        "sniper_tp1":   tp1_sniper,
        "sniper_tp2":   tp2_sniper,
        "sniper_tp3":   tp3_sniper,
        # Meta
        "sl_dist_pct":  round(sl_dist * 100, 2),
        "atr_pct":      round(atr_pct * 100,  2),
        "sl_method":    sl_method,
        "struct_sl":    round(struct_sl, 8),
        "std_valid":    std_valid,
        "sniper_valid": sniper_valid,
    }


def _scanner_score_signal(
    df: pd.DataFrame,
    adx_df: pd.DataFrame,
    bar_idx: int,
    direction: str,
    timeframe: str,
    symbol: str,
    min_body_pct: float,
    min_vol_mult: float,
) -> dict | None:
    """
    Score a single bar as a momentum signal. Returns None if bar doesn't qualify.
    Score is 0–100 based on _SCORE_WEIGHTS.
    """
    try:
        bar = df.iloc[bar_idx]
    except IndexError:
        return None

    body_pct = float(bar.get("body_pct", 0) or 0)
    vol_mult  = float(bar.get("vol_mult",  0) or 0)
    atr_ratio = float(bar.get("atr_ratio", 1) or 1)
    ema5      = float(bar.get("ema5",  0) or 0)
    ema15     = float(bar.get("ema15", 0) or 0)
    ema21     = float(bar.get("ema21", 0) or 0)
    c_rank    = float(bar.get("candle_rank_20", 0.5) or 0.5)
    v_rank    = float(bar.get("vol_rank_20",    0.5) or 0.5)
    taker_buy_ratio = float(bar.get("taker_buy_ratio", 0.5) or 0.5)
    close_px  = float(bar.get("close", 0) or 0)
    body_abs  = float(bar.get("body",  0) or 0)
    high_px   = float(bar.get("high",  close_px) or close_px)
    low_px    = float(bar.get("low",   close_px) or close_px)
    open_px   = float(bar.get("open",  close_px) or close_px)
    atr14_val = float(bar.get("atr14", close_px * 0.02) or close_px * 0.02)

    # ── Direction check ────────────────────────────────────────────────────────
    is_bullish = body_pct > 0
    if direction == "long"  and not is_bullish:
        return None
    if direction == "short" and is_bullish:
        return None

    # ── Filter thresholds ──────────────────────────────────────────────────────
    if abs(body_pct) < min_body_pct:
        return None
    if vol_mult < min_vol_mult or pd.isna(vol_mult):
        return None

    # ── ADX values ────────────────────────────────────────────────────────────
    adx_val  = 0.0
    di_plus  = 0.0
    di_minus = 0.0
    if adx_df is not None and not adx_df.empty and bar_idx < len(adx_df):
        try:
            _adx = float(adx_df["adx"].iloc[bar_idx])
            _dip = float(adx_df["di_plus"].iloc[bar_idx])
            _dim = float(adx_df["di_minus"].iloc[bar_idx])
            # Guard against NaN — float(NaN) succeeds but poisons arithmetic
            adx_val  = _adx  if _adx  == _adx  else 0.0
            di_plus  = _dip  if _dip  == _dip  else 0.0
            di_minus = _dim  if _dim  == _dim  else 0.0
        except Exception:
            pass

    # ── Regime score ──────────────────────────────────────────────────────────
    try:
        regime = calculate_regime_score(
            df, bar_idx, direction, adx_df,
            timeframe=timeframe, ticker=symbol,
        )
        regime_score_val = regime.get("score",   0)
        regime_verdict   = regime.get("verdict", "RED")
    except Exception:
        regime_score_val = 0
        regime_verdict   = "RED"

    # Skip RED regime entirely — not worth showing
    if regime_verdict == "RED":
        return None

    # ── EMA stack alignment ───────────────────────────────────────────────────
    if direction == "long":
        ema_full    = (ema5 > ema15) and (ema15 > ema21)
        ema_partial = (ema5 > ema15) or  (ema15 > ema21)
    else:
        ema_full    = (ema5 < ema15) and (ema15 < ema21)
        ema_partial = (ema5 < ema15) or  (ema15 < ema21)

    # ── Composite score (0–100) ───────────────────────────────────────────────
    # Body component (0–25)
    body_pts  = min(abs(body_pct) / 0.95, 1.0) * _SCORE_WEIGHTS["body"]

    # Volume component (0–20): vol_mult 1.5→ ~0 pts, 5.0+ → 20 pts
    vol_norm  = max(0, (vol_mult - min_vol_mult) / max(1, 5.0 - min_vol_mult))
    vol_pts   = min(vol_norm, 1.0) * _SCORE_WEIGHTS["volume"]

    # ADX component (0–20)
    adx_norm  = min(adx_val / 40.0, 1.0)
    adx_pts   = adx_norm * _SCORE_WEIGHTS["adx"]

    # Regime component (0–25)
    regime_pts = (regime_score_val / 100.0) * _SCORE_WEIGHTS["regime"]

    # Recency: set by caller based on bar_offset (0=most recent closed candle)
    # We use 10/6/3 for bar_offset 1/2/3 — set later in caller
    recency_pts = 0  # placeholder, set by caller

    total_score = body_pts + vol_pts + adx_pts + regime_pts
    # Defend against any NaN that slipped through a component (x != x ↔ isnan)
    if total_score != total_score:
        total_score = 0.0
    # Note: recency added by caller

    # ── Entry levels — enhanced multi-zone trade plan ─────────────────────────
    _etp = _compute_enhanced_trade_plan(
        direction=direction,
        close_px=close_px,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        atr14=atr14_val,
        body_pct=body_pct,
    )
    # Legacy fields (aggressive entry = enter at close) kept for backward compat
    entry = _etp.get("agg_entry",  close_px)
    sl    = _etp.get("agg_sl",     close_px * (0.985 if direction == "long" else 1.015))
    tp2r  = _etp.get("agg_tp2",    close_px)
    tp3r  = _etp.get("agg_tp3",    close_px)

    # ── Build reasons list ────────────────────────────────────────────────────
    reasons = []

    # Candle body
    bp_pct = abs(body_pct) * 100
    if bp_pct >= 85:
        body_lbl = "exceptional conviction"
    elif bp_pct >= 75:
        body_lbl = "strong conviction"
    else:
        body_lbl = "clear momentum"
    reasons.append(f"Candle body {bp_pct:.1f}% of range — {body_lbl} (threshold: {min_body_pct*100:.0f}%)")

    # Volume
    if vol_mult >= 4:
        vol_lbl = "extreme institutional activity"
    elif vol_mult >= 2.5:
        vol_lbl = "strong volume surge"
    elif vol_mult >= 1.8:
        vol_lbl = "elevated participation"
    else:
        vol_lbl = "above-average volume"
    reasons.append(f"Volume {vol_mult:.1f}× the 7-bar average — {vol_lbl}")

    # ADX / trend
    if adx_val >= 35:
        reasons.append(f"ADX {adx_val:.0f} — strongly trending market (momentum likely to continue)")
    elif adx_val >= 25:
        reasons.append(f"ADX {adx_val:.0f} — trending market (signals work best here)")
    elif adx_val >= 18:
        reasons.append(f"ADX {adx_val:.0f} — moderate trend developing")
    else:
        reasons.append(f"ADX {adx_val:.0f} — weak trend (signal still qualifies but use caution)")

    # DI alignment
    di_gap = abs(di_plus - di_minus)
    if direction == "long" and di_plus > di_minus and di_gap >= 10:
        reasons.append(f"DI+ {di_plus:.0f} vs DI− {di_minus:.0f} (gap {di_gap:.0f}) — bulls clearly dominating")
    elif direction == "short" and di_minus > di_plus and di_gap >= 10:
        reasons.append(f"DI− {di_minus:.0f} vs DI+ {di_plus:.0f} (gap {di_gap:.0f}) — bears clearly dominating")

    # EMA stack
    if ema_full:
        reasons.append(f"EMA stack fully {'bullish (5>15>21)' if direction=='long' else 'bearish (5<15<21)'} — trend filter aligned")
    elif ema_partial:
        reasons.append(f"EMA partially aligned — trend direction consistent but not perfect")

    # ATR ratio — volatility context
    if atr_ratio > 1.2:
        reasons.append(f"ATR ratio {atr_ratio:.2f}× — volatility expanding, momentum candle has more room to run")
    elif atr_ratio < 0.8:
        reasons.append(f"ATR ratio {atr_ratio:.2f}× — low volatility context, compression before potential breakout")

    # Candle rank
    if c_rank >= 0.85:
        reasons.append(f"Candle rank top {(1-c_rank)*100:.0f}% — one of the strongest candles in the last 20 bars")
    elif c_rank >= 0.70:
        reasons.append(f"Candle rank top {(1-c_rank)*100:.0f}% — above-average candle size for this coin")

    # Volume rank
    if v_rank >= 0.85:
        reasons.append(f"Volume rank top {(1-v_rank)*100:.0f}% — exceptionally high volume for this coin recently")
    elif v_rank >= 0.70:
        reasons.append(f"Volume rank top {(1-v_rank)*100:.0f}% — above-average trading activity")

    # Regime
    regime_color_label = {"GREEN": "✅ GREEN", "YELLOW": "⚠️ YELLOW"}.get(regime_verdict, regime_verdict)
    reasons.append(f"Market regime {regime_color_label} ({regime_score_val}/100) — favorable conditions for momentum trades")

    return {
        "symbol":        symbol,
        "timeframe":     timeframe,
        "direction":     direction,
        "base_score":    round(total_score, 2),   # recency added later
        "regime":        regime_verdict,
        "regime_score":  regime_score_val,
        "body_pct":      round(abs(body_pct) * 100, 1),
        "vol_mult":      round(vol_mult, 2),
        "adx":           round(adx_val,  1),
        "di_plus":       round(di_plus,  1),
        "di_minus":      round(di_minus, 1),
        "atr_ratio":     round(atr_ratio, 2),
        "ema_full":      ema_full,
        "ema_partial":   ema_partial,
        "candle_rank":   round(c_rank,   2),
        "vol_rank":      round(v_rank,   2),
        "close":         close_px,
        "entry":         entry,
        "sl":            sl,
        "tp2r":          tp2r,
        "tp3r":          tp3r,
        "bar_offset":    None,   # filled by caller
        "reasons":       reasons,
        "_trade_plan":   _etp,
        "taker_buy_ratio": round(taker_buy_ratio, 4),
    }


def _scan_one_symbol(args: tuple) -> list:
    """
    Worker function for ThreadPoolExecutor.
    args = (symbol, timeframes_list, min_body_pct, min_vol_mult, directions)
    Returns list of scored signal dicts (may be empty).
    """
    symbol, timeframes, min_body_pct, min_vol_mult, directions = args
    results = []
    _RECENCY_PTS = {1: 10, 2: 6, 3: 3}   # bar_offset → recency score

    for tf in timeframes:
        interval = _BINANCE_INTERVAL.get(tf, "1d")
        # Bumped 120 → 200 bars: gives regime/ADX rolling windows more warmup
        # for more accurate scan-time ranking. Still only last 3 closed candles
        # are checked for signals — this is warmup data only.
        df = _scanner_fetch_candles(symbol, interval, limit=200)
        if df.empty or len(df) < 22:
            continue

        try:
            adx_df = calculate_adx(df)
        except Exception:
            adx_df = pd.DataFrame()

        # Check last 3 CLOSED candles (skip index -1 = current open candle)
        _now_utc = pd.Timestamp.utcnow().tz_localize(None)
        for bar_offset in [1, 2, 3]:
            bar_idx = len(df) - bar_offset - 1   # -1 skips the live candle
            if bar_idx < 14:   # need enough bars for indicators to warm up
                continue

            # ── Staleness guard: skip candles older than 5 days ──────────────
            try:
                _bar_ts = pd.Timestamp(df.index[bar_idx]).tz_localize(None)
                if (_now_utc - _bar_ts).total_seconds() > 5 * 86400:
                    continue   # inactive / delisted coin — skip entirely
            except Exception:
                pass

            for direction in directions:
                sig = _scanner_score_signal(
                    df, adx_df, bar_idx, direction,
                    tf, symbol, min_body_pct, min_vol_mult,
                )
                if sig is None:
                    continue

                recency_pts        = _RECENCY_PTS.get(bar_offset, 0)
                sig["bar_offset"]  = bar_offset
                _raw_score = sig["base_score"] + recency_pts
                # Guard against NaN (NaN != NaN) and clamp to valid range
                sig["score"] = round(_raw_score if _raw_score == _raw_score else 0.0, 2)
                # Skip signals with invalid entry price (bad data / stablecoin)
                if not sig.get("entry") or sig["entry"] != sig["entry"]:
                    continue
                # Convert UTC → WIB (UTC+7) for display
                _ts_utc = pd.Timestamp(df.index[bar_idx])
                _ts_wib = _ts_utc + pd.Timedelta(hours=7)
                sig["candle_date"] = _ts_wib.strftime("%Y-%m-%d %H:%M WIB")
                results.append(sig)

    return results


def _scanner_ai_verdict(sig: dict, ml_a: dict = None, ml_b: dict = None,
                         bt: dict = None, wfo: dict = None,
                         cand_a: dict = None, cand_b: dict = None) -> dict:
    """
    Dual-candidate AI verdict.

    Analyzes TWO candidate trading methods for the same signal:
      - Candidate A = best method in the NEWEST time-decay bucket
      - Candidate B = best method by WEIGHTED all-time EV

    When A == B (same method_cfg), runs a single analysis and mirrors the
    result to both sides. Otherwise the LLM is asked to evaluate each
    candidate independently and pick the winner if both are TRADE.

    Returns a dict with:
      {
        "dual": True,
        "candidate_a": {verdict, confidence, rationale, execution, risk, conflicts},
        "candidate_b": {verdict, confidence, rationale, execution, risk, conflicts},
        "winner": "A" | "B" | "NONE",
        "winner_rationale": "...",
        "unanimous": bool,
        "source": "groq/<model>",
      }
    """
    api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        _empty = {
            "verdict": "NO KEY", "confidence": "",
            "rationale": "Add a free Groq API key in the sidebar to enable AI analysis.",
            "execution": "", "risk": "", "conflicts": "",
        }
        return {
            "dual": True,
            "candidate_a": _empty, "candidate_b": _empty,
            "winner": "NONE", "winner_rationale": "",
            "unanimous": False, "source": "",
        }

    # ── Detect if A and B are the same method ────────────────────────────────
    def _cfg_of(c):
        if not c:
            return None
        mc = c.get("method_cfg") or {}
        return (mc.get("zone"), mc.get("sl_label"), mc.get("mgmt"),
                round(float(mc.get("tp_mult", 2.0)), 2))

    _cfg_a = _cfg_of(cand_a)
    _cfg_b = _cfg_of(cand_b)
    _unanimous = (_cfg_a is not None and _cfg_a == _cfg_b)

    ema_status = (
        "fully aligned"   if sig.get("ema_full")    else
        "partially aligned" if sig.get("ema_partial") else
        "not aligned"
    )
    # ── Build ML section helper for a single candidate ──────────────────────
    def _build_ml_section(ml, tag):
        if not ml:
            return f"{tag}: ML not trained"
        _ml_trained_p = ml.get("trained", False)
        _ml_mname_p   = ml.get("method_name", "Heuristic")
        _ml_ns_p      = ml.get("n_samples", 0)
        _ml_cv_p      = ml.get("cv_accuracy")
        _ml_cfg_p     = ml.get("method_cfg") or {}
        _ml_fi_p      = ml.get("feature_importance", [])
        if _ml_trained_p:
            _cv_str_p = f"CV: {_ml_cv_p*100:.1f}%" if _ml_cv_p is not None else "CV: n/a"
            _cfg_str_p = (f"{_ml_cfg_p.get('zone','?')}/{_ml_cfg_p.get('sl_label','?')}/"
                          f"{_ml_cfg_p.get('mgmt','?')}/TP{_ml_cfg_p.get('tp_mult',2.0):.1f}R")
            _top3 = ", ".join(f"{f['feature']}={f['importance']:.2f}" for f in _ml_fi_p[:3])
            return (
                f"{tag}: {ml['pct']:.1f}% ({ml['label']}) | {_ml_mname_p} "
                f"n={_ml_ns_p} ({ml.get('n_wins',0)}W/{ml.get('n_losses',0)}L) | "
                f"{_cv_str_p} | method={_cfg_str_p}"
                + (f" | top={_top3}" if _top3 else "")
            )
        return f"{tag}: {ml['pct']:.1f}% ({ml['label']}) — HEURISTIC not trained ({_ml_mname_p})"

    ml_section_a = _build_ml_section(ml_a, "ML-A")
    ml_section_b = _build_ml_section(ml_b, "ML-B") if not _unanimous else "ML-B: (same as A — unanimous)"

    # ── Build candidate detail helper ────────────────────────────────────────
    def _build_cand_detail(cand, tag):
        if not cand:
            return f"{tag}: not available"
        mc   = cand.get("method_cfg") or {}
        nb   = cand.get("newest_bucket") or {}
        _pf  = cand.get("pf", 0)
        _pfs = "∞" if _pf >= 9.9 else f"{_pf:.2f}"
        _lines = [
            f"{tag}: {mc.get('zone','?')}/{mc.get('sl_label','?')}/{mc.get('mgmt','?')}/TP{mc.get('tp_mult',2.0):.1f}R",
            f"  All-time: WR={cand.get('win_rate',0):.1f}% EV={cand.get('ev',0):+.2f}R "
            f"EVw={cand.get('ev_weighted',0):+.2f}R PF={_pfs} n={cand.get('n',0)}",
            f"  Newest bucket: WR={nb.get('wr',0):.1f}% EV={nb.get('ev',0):+.2f}R n={nb.get('n',0)}",
        ]
        # Time-decay trajectory
        buckets = cand.get("buckets", []) or []
        if buckets:
            _traj = " → ".join(
                f"{b.get('label','?').split()[0]}:{b.get('wr',0):.0f}%/{b.get('ev',0):+.1f}R(n{b.get('n',0)})"
                for b in buckets
            )
            _lines.append(f"  Decay trajectory (old→new): {_traj}")
        return "\n".join(_lines)

    cand_a_section = _build_cand_detail(cand_a, "CANDIDATE A (best newest-bucket)")
    cand_b_section = (_build_cand_detail(cand_b, "CANDIDATE B (best weighted all-time)")
                      if not _unanimous
                      else "CANDIDATE B: identical to Candidate A — single analysis")

    direction = sig["direction"].upper()
    reasons_text  = "\n".join(f"- {r}" for r in sig.get("reasons", []))
    _etp          = sig.get("_trade_plan", {})

    # ── Backtest: per-zone best ─────────────────────────────────────────────
    zone_best   = bt.get("zone_best", {}) if bt else {}
    best_key    = bt.get("best_key", "")  if bt else ""
    best        = bt.get("best", {})      if bt else {}
    per_method  = bt.get("per_method", {}) if bt else {}

    def _fmt(v): return f"{v:.6g}" if v else "N/A"

    if bt and bt.get("error") is None:
        zone_lines = []
        for zn in ("Aggressive", "Standard", "Sniper"):
            zd = zone_best.get(zn, {})
            if zd and not zd.get("insufficient") and zd.get("n", 0) >= 4:
                zone_lines.append(
                    f"  {zn} ({zd.get('sl_label','?')} / {zd.get('mgmt','?')}): "
                    f"WR={zd.get('win_rate',0):.1f}% EV={zd.get('ev',0):+.2f}R "
                    f"n={zd.get('n',0)} avg_hold={zd.get('avg_bars',0):.1f}bars"
                )
            else:
                zone_lines.append(f"  {zn}: insufficient data (<4 setups)")

        best_line = (
            f"OVERALL BEST: {best_key} "
            f"(WR={best.get('win_rate',0):.1f}% EV={best.get('ev',0):+.2f}R n={best.get('n',0)})"
            if best_key else "OVERALL BEST: undetermined"
        )

        # Price levels for best zone — route by SL method (Fixed vs ATR)
        _zone_etp_keys = {
            "Aggressive": ("agg_entry", "agg_sl", "agg_tp1", "agg_tp2", "agg_tp3"),
            "Standard":   ("std_entry", "std_sl", "std_tp1", "std_tp2", "std_tp3"),
            "Sniper":     ("sniper_entry","sniper_sl","sniper_tp1","sniper_tp2","sniper_tp3"),
        }
        _bzone         = best.get("zone", "Aggressive")
        _bsl_label_p   = best.get("sl_label", "Fixed SL")
        _b_use_atr_p   = "ATR" in _bsl_label_p
        _bkeys         = _zone_etp_keys.get(_bzone, ())
        _b_entry       = _etp.get(_bkeys[0], 0) if _bkeys else 0
        _b_atr_sl_p    = _etp.get(_bkeys[1], 0) if _bkeys else 0
        _b_tp1_atr_p   = _etp.get(_bkeys[2], 0) if _bkeys else 0
        _b_tp2_atr_p   = _etp.get(_bkeys[3], 0) if _bkeys else 0
        _b_tp3_atr_p   = _etp.get(_bkeys[4], 0) if _bkeys else 0
        # Compute Fixed SL prices so the AI gets the right levels when Fixed SL is chosen
        _FIXED_SL_PROMPT = 0.015
        if _b_entry:
            _b_fix_sl_p = round(_b_entry * ((1 - _FIXED_SL_PROMPT) if direction == "long"
                                            else (1 + _FIXED_SL_PROMPT)), 8)
            _b_risk_fix = abs(_b_entry - _b_fix_sl_p)
            _sign = 1 if direction == "long" else -1
            _b_fix_tp1 = round(_b_entry + _sign * 1 * _b_risk_fix, 8)
            _b_fix_tp2 = round(_b_entry + _sign * 2 * _b_risk_fix, 8)
            _b_fix_tp3 = round(_b_entry + _sign * 3 * _b_risk_fix, 8)
        else:
            _b_fix_sl_p = _b_fix_tp1 = _b_fix_tp2 = _b_fix_tp3 = 0
        _b_sl  = _b_atr_sl_p  if _b_use_atr_p else _b_fix_sl_p
        _b_tp1 = _b_tp1_atr_p if _b_use_atr_p else _b_fix_tp1
        _b_tp2 = _b_tp2_atr_p if _b_use_atr_p else _b_fix_tp2
        _b_tp3 = _b_tp3_atr_p if _b_use_atr_p else _b_fix_tp3

        bt_section = (
            f"Zone comparison (best config per zone):\n"
            + "\n".join(zone_lines)
            + f"\n{best_line}\n"
            f"Best execution prices — Entry: {_fmt(_b_entry)} | SL: {_fmt(_b_sl)} "
            f"| TP1: {_fmt(_b_tp1)} | TP2: {_fmt(_b_tp2)} | TP3: {_fmt(_b_tp3)}\n"
            f"Management for best: {best.get('mgmt','Simple')} with {best.get('sl_label','Fixed SL')} targeting {best.get('tp_mult',2.0):.1f}R"
        )
    elif bt and bt.get("error"):
        bt_section = f"Backtest: {bt['error']}"
    else:
        bt_section = "Backtest: not computed"

    # ── Also provide all 3 zone entry prices for reference ─────────────────
    price_ref = (
        f"Signal candle close: {_fmt(sig.get('close', 0))}\n"
        f"Aggressive entry: {_fmt(_etp.get('agg_entry',0))} | SL: {_fmt(_etp.get('agg_sl',0))} | TP2: {_fmt(_etp.get('agg_tp2',0))}\n"
        f"Standard entry:   {_fmt(_etp.get('std_entry',0))} | SL: {_fmt(_etp.get('std_sl',0))} | TP2: {_fmt(_etp.get('std_tp2',0))}\n"
        f"Sniper entry:     {_fmt(_etp.get('sniper_entry',0))} | SL: {_fmt(_etp.get('sniper_sl',0))} | TP2: {_fmt(_etp.get('sniper_tp2',0))}\n"
        f"ATR SL distance: {_etp.get('sl_dist_pct',1.5):.1f}% (ATR={_etp.get('atr_pct',0):.1f}%)"
    )

    # ── New context variables for enhanced prompt ─────────────────────────
    _btcd = sig.get("btc_dominance", None)
    _fng  = sig.get("fng_value", None)
    _session = sig.get("session", "Unknown")
    _candle_rank_pct = round((1 - sig.get("candle_rank", 0.5)) * 100, 0)
    _oi_chg   = sig.get("oi_change_pct", None)
    _fr_rate  = sig.get("funding_rate", None)
    _taker    = sig.get("taker_buy_ratio", None)

    # Build derivatives section string
    if _oi_chg is not None:
        _deriv_section = (
            f"OI 24h Change: {_oi_chg:+.1f}%\n"
            f"Funding Rate: {_fr_rate*100:.4f}% per 8h\n"
            f"Taker Buy Ratio (signal candle): {_taker*100:.1f}%"
        ) if _fr_rate is not None and _taker is not None else (
            f"OI 24h Change: {_oi_chg:+.1f}%\n"
            f"Funding Rate: N/A\n"
            f"Taker Buy Ratio: N/A"
        )
    else:
        _deriv_section = "Derivatives data: not available (spot or fetch failed)"

    # Build macro section string
    _macro_parts = []
    if _btcd is not None:
        _macro_parts.append(f"BTC Dominance: {_btcd:.1f}%")
    if _fng is not None:
        _fng_label = "Extreme Fear" if _fng < 20 else "Fear" if _fng < 40 else "Neutral" if _fng < 60 else "Greed" if _fng < 80 else "Extreme Greed"
        _macro_parts.append(f"Fear & Greed: {_fng} ({_fng_label})")
    _macro_section = "\n".join(_macro_parts) if _macro_parts else "Macro context: not available"

    # ── WFO section ──────────────────────────────────────────────────────────
    if wfo and wfo.get("ok"):
        _wfo_verdict = wfo.get("verdict", "INSUFFICIENT")
        _wfo_is_pf   = wfo.get("is_pf",  0)
        _wfo_oos_pf  = wfo.get("oos_pf", 0)
        _wfo_oos_wr  = wfo.get("oos_wr", 0)
        _wfo_n_is    = wfo.get("is_n",   0)
        _wfo_n_oos   = wfo.get("oos_n",  0)
        _wfo_ratio   = wfo.get("oos_is_ratio", 0)
        _wfo_note    = wfo.get("note",    "")
        _wfo_method  = wfo.get("method_used", "")
        wfo_section  = (
            f"WFO Verdict: {_wfo_verdict}\\n"
            f"IS: PF={'∞' if _wfo_is_pf>=9.9 else f'{_wfo_is_pf:.2f}'} n={_wfo_n_is} | OOS: PF={'∞' if _wfo_oos_pf>=9.9 else f'{_wfo_oos_pf:.2f}'} WR={_wfo_oos_wr:.1f}% n={_wfo_n_oos}\\n"
            f"OOS/IS Ratio: {_wfo_ratio:.2f} (>0.60 = good) | Method: {_wfo_method}\\n"
            f"Note: {_wfo_note}"
        )
    else:
        wfo_section = "WFO: not run yet (Step 1 required)"

    prompt = f"""You are a SKEPTICAL trading analyst. Evaluate TWO candidate trading methods for the same signal. For EACH candidate output TRADE / WAIT / NO TRADE independently. Then pick the WINNER if both are TRADE. Cite specific numbers. No markdown.

=== SIGNAL (shared between both candidates) ===
Symbol: {sig['symbol']} | Timeframe: {sig['timeframe']} | Direction: {direction}
Composite Score: {sig['score']:.1f}/100 | Signal age: {sig.get('bar_offset',1)} candle(s) old
Body: {sig['body_pct']:.1f}% of range | Candle rank: top {_candle_rank_pct:.0f}% of last 20 bars
Volume: {sig['vol_mult']:.2f}x average | ADX: {sig['adx']:.1f} | DI+: {sig['di_plus']:.1f} vs DI-: {sig['di_minus']:.1f}
ATR Ratio: {sig['atr_ratio']:.2f} | EMA Stack (5/15/21): {ema_status}
Market Regime: {sig['regime']} ({sig['regime_score']}/100) | Session: {_session}

=== MACRO CONTEXT ===
{_macro_section}

=== DERIVATIVES SENTIMENT ===
{_deriv_section}

=== CANDIDATE A — BEST NEWEST-BUCKET METHOD ===
{cand_a_section}
{ml_section_a}

=== CANDIDATE B — BEST WEIGHTED ALL-TIME METHOD ===
{cand_b_section}
{ml_section_b}

=== FULL BACKTEST CONTEXT ===
{bt_section}

=== WFO VALIDATION (parameter robustness — applies to whichever method WFO ran on) ===
{wfo_section}

=== ALL ENTRY PRICE LEVELS (for reference) ===
{price_ref}

=== SELECTION CRITERIA (signal reasons) ===
{reasons_text}

=== DECISION RULES (apply strictly to EACH candidate) ===
- If that candidate's all-time EV < 0 with n >= 8: NO TRADE
- If that candidate's all-time WR < 40% with n >= 8: NO TRADE
- If WFO verdict is FAIL: WAIT minimum, note overfit risk
- If that candidate's ML < 45%: lean WAIT
- If signal age > 2 candles and candidate zone requires retrace (Standard/Sniper): WAIT
- Newest-bucket stats dominate when they contradict all-time stats — markets drift
- Low sample (n<5) in newest bucket: treat newest-bucket stats as directional only

=== CONFLICTS TO CHECK (for each candidate) ===
1. All-time vs newest bucket: is edge strengthening or decaying?
2. ML probability vs backtest EV sign: agreement check
3. ML CV accuracy: <55% means ML is barely predictive
4. Funding/OI vs direction: crowded positioning?
5. Signal age vs entry zone retrace requirement

=== WINNER SELECTION (only if BOTH A and B are TRADE) ===
Pick the candidate with the strongest combination of:
  - recent edge (newest bucket WR/EV)
  - ML probability × CV accuracy
  - consistency (EVw vs all-time EV stability)
  - tighter SL / better R:R if tie

If A == B (unanimous): output same verdict for both and WINNER=A.

Respond in EXACTLY this format, no extra text, no markdown, no preamble:

=== CANDIDATE A ===
VERDICT: [TRADE / WAIT / NO TRADE]
CONFIDENCE: [HIGH / MEDIUM / LOW]
CONFLICTS: [List with numbers, or "None detected"]
RATIONALE: [3 sentences max. Lead with strongest factor. Cite WR, EV, EVw, ML%, CV.]
EXECUTION: [If TRADE: exact zone, entry, SL, TP1, TP2 prices, mgmt. If WAIT: what must change. If NO TRADE: what disqualifies.]
RISK: [1 sentence — specific failure mode.]

=== CANDIDATE B ===
VERDICT: [TRADE / WAIT / NO TRADE]
CONFIDENCE: [HIGH / MEDIUM / LOW]
CONFLICTS: [List with numbers, or "None detected"]
RATIONALE: [3 sentences max. Cite specific numbers.]
EXECUTION: [If TRADE: exact zone, entry, SL, TP1, TP2 prices, mgmt. Otherwise what must change or disqualifies.]
RISK: [1 sentence — specific failure mode.]

=== WINNER ===
PICK: [A / B / NONE]
WHY: [1-2 sentences explaining which is stronger and why. If NONE, explain why neither is tradeable.]"""

    try:
        _selected_model = st.session_state.get("groq_model", "openai/gpt-oss-120b")
        _is_reasoning   = ("gpt-oss" in _selected_model or "qwen" in _selected_model.lower())
        _body = {
            "model":       _selected_model,
            "max_tokens":  2500,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a SKEPTICAL systematic momentum trading analyst reviewing TWO "
                        "candidate trading methods for the same signal. Your job is to:\n"
                        "  1) analyze each candidate independently and mark it TRADE / WAIT / NO TRADE,\n"
                        "  2) if both are TRADE, explicitly pick the winner and justify why,\n"
                        "  3) find reasons NOT to take each trade.\n"
                        "Be decisive and concise. Follow the output format EXACTLY. "
                        "Always cite specific numbers (WR, EV, PF, ML%, CV, sample size, bucket stats). "
                        "Never add extra commentary or markdown. No preamble."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        if _is_reasoning:
            _body["reasoning_effort"] = "medium"

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=_body,
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

        # ── Parse dual-candidate response ──────────────────────────────────
        def _empty_section():
            return {
                "verdict": "WAIT", "confidence": "MEDIUM",
                "rationale": "", "execution": "", "risk": "", "conflicts": "",
            }

        def _parse_section(text_block):
            sec = _empty_section()
            for line in text_block.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.upper().startswith("VERDICT:"):
                    v = line.split(":", 1)[1].strip().upper()
                    sec["verdict"] = ("NO TRADE" if "NO TRADE" in v else
                                      "TRADE"    if "TRADE"    in v else "WAIT")
                elif line.upper().startswith("CONFIDENCE:"):
                    sec["confidence"] = line.split(":", 1)[1].strip().upper()
                elif line.upper().startswith("CONFLICTS:"):
                    sec["conflicts"] = line.split(":", 1)[1].strip()
                elif line.upper().startswith("RATIONALE:"):
                    sec["rationale"] = line.split(":", 1)[1].strip()
                elif line.upper().startswith("EXECUTION:"):
                    sec["execution"] = line.split(":", 1)[1].strip()
                elif line.upper().startswith("RISK:"):
                    sec["risk"] = line.split(":", 1)[1].strip()
            return sec

        # Split the raw text by section headers
        _upper_raw = raw.upper()
        _idx_a = _upper_raw.find("=== CANDIDATE A")
        _idx_b = _upper_raw.find("=== CANDIDATE B")
        _idx_w = _upper_raw.find("=== WINNER")

        if _idx_a == -1:
            _idx_a = 0
        _block_a = raw[_idx_a:_idx_b] if _idx_b != -1 else raw[_idx_a:]
        _block_b = raw[_idx_b:_idx_w] if (_idx_b != -1 and _idx_w != -1) else (
                    raw[_idx_b:] if _idx_b != -1 else "")
        _block_w = raw[_idx_w:] if _idx_w != -1 else ""

        cand_a_out = _parse_section(_block_a)
        cand_b_out = _parse_section(_block_b) if _block_b else _empty_section()

        # Parse winner block
        winner = "NONE"
        winner_rationale = ""
        for line in _block_w.split("\n"):
            line = line.strip()
            if line.upper().startswith("PICK:"):
                p = line.split(":", 1)[1].strip().upper()
                winner = "A" if p.startswith("A") else ("B" if p.startswith("B") else "NONE")
            elif line.upper().startswith("WHY:"):
                winner_rationale = line.split(":", 1)[1].strip()

        # If unanimous, mirror A to B
        if _unanimous:
            cand_b_out = dict(cand_a_out)
            winner = "A"
            if not winner_rationale:
                winner_rationale = "Candidate A and B resolved to the same method — analyzed once."

        # Auto-pick winner if LLM failed to and both are TRADE
        if winner == "NONE":
            _a_trade = cand_a_out["verdict"] == "TRADE"
            _b_trade = cand_b_out["verdict"] == "TRADE"
            if _a_trade and not _b_trade:
                winner = "A"
            elif _b_trade and not _a_trade:
                winner = "B"

        # Fallback: if rationales are empty because parse failed, dump raw into A
        if not cand_a_out["rationale"] and not cand_b_out["rationale"]:
            cand_a_out["rationale"] = raw[:400]

        _model_used = st.session_state.get("groq_model", "openai/gpt-oss-120b")
        return {
            "dual":             True,
            "candidate_a":      cand_a_out,
            "candidate_b":      cand_b_out,
            "winner":           winner,
            "winner_rationale": winner_rationale,
            "unanimous":        _unanimous,
            "source":           f"groq/{_model_used.split('/')[-1]}",
            "raw":              raw[:2000],  # kept for debug / display fallback
        }

    except Exception as exc:
        _err = {
            "verdict":   "ERROR", "confidence": "",
            "rationale": f"API error: {str(exc)[:120]}",
            "execution": "", "risk": "", "conflicts": "",
        }
        return {
            "dual":             True,
            "candidate_a":      _err, "candidate_b": _err,
            "winner":           "NONE", "winner_rationale": "",
            "unanimous":        _unanimous,
            "source":           "error",
        }


def _scanner_quick_backtest(sig: dict) -> dict:
    """
    Enhanced multi-method backtest that tests and compares:
      Entry Zones  : Aggressive (0%), Standard (38.2%), Sniper (61.8%)
      SL Methods   : Fixed 1.5% vs ATR-adaptive (from signal _trade_plan)
      Management   : Simple (hold to 2R), Partial (50%@1R->BE->2R), Trailing
      Expiry Logic : Retrace entries expire if not filled within 3 bars

    Returns per_method stats, zone_best, best overall method.
    """
    symbol    = sig["symbol"]
    timeframe = sig["timeframe"]
    direction = sig["direction"]
    _etp      = sig.get("_trade_plan", {})

    min_body = abs(sig["body_pct"]) * 0.70 / 100
    min_vol  = max(1.0, sig["vol_mult"] * 0.70)

    interval = _BINANCE_INTERVAL.get(timeframe, "1d")
    # Deep historical fetch — timeframe-aware. Uses Binance max of 1000 bars.
    # For new coins with less history, caller uses whatever is returned.
    deep_limit = _deep_limit_for(timeframe)
    df = _scanner_fetch_candles(symbol, interval, limit=deep_limit)

    if df.empty or len(df) < 30:
        return {"error": "Not enough data", "n": 0,
                "meta": {"bars_requested": deep_limit, "bars_used": len(df) if not df.empty else 0}}

    # SL distances
    atr_sl_pct = (_etp.get("sl_dist_pct", 2.0) or 2.0) / 100.0
    atr_sl_pct = max(0.008, min(0.06, atr_sl_pct))
    FIXED_SL   = 0.015

    ENTRY_ZONES = {
        "Aggressive": {"retrace": 0.000, "expiry_bars": 0},
        "Standard":   {"retrace": 0.382, "expiry_bars": 3},
        "Sniper":     {"retrace": 0.618, "expiry_bars": 3},
    }
    MGMT_MODES = ["Simple", "Partial", "Trailing"]
    TP_MULTS   = [2.0, 2.5, 3.0]   # test 2R / 2.5R / 3R targets per combo
    MAX_HOLD   = 20
    n_df       = len(df)
    method_results = {}

    # ── Time-decay bucket scheme (adaptive to n_df) ──────────────────────────
    # 4 buckets if n_df >= 400, 3 if >=200, 2 if >=80, else 1 bucket
    _decay_buckets = _compute_decay_buckets(n_df)

    for zone_name, zone_cfg in ENTRY_ZONES.items():
        ret_frac = zone_cfg["retrace"]
        expiry   = zone_cfg["expiry_bars"]

        for sl_label, sl_pct_val in [("Fixed SL", FIXED_SL), ("ATR SL", atr_sl_pct)]:
            for mgmt in MGMT_MODES:
              for tp_mult in TP_MULTS:
                key        = f"{zone_name} / {sl_label} / {mgmt} / TP{tp_mult:.1f}R"
                trades_raw = []

                for i in range(14, n_df - 2):
                    bar      = df.iloc[i]
                    body_pct = float(bar.get("body_pct", 0) or 0)
                    vol_mult = float(bar.get("vol_mult",  0) or 0)
                    is_bull  = body_pct > 0
                    if direction == "long"  and not is_bull: continue
                    if direction == "short" and is_bull:     continue
                    if abs(body_pct) < min_body: continue
                    if vol_mult < min_vol:       continue

                    close_v  = float(bar["close"])
                    open_v   = float(bar.get("open",  close_v))
                    body_abs = abs(close_v - open_v)
                    atr14    = float(bar.get("atr14", close_v * 0.02) or close_v * 0.02)
                    if close_v <= 0:
                        continue

                    if direction == "long":
                        entry_target = max(round(close_v - body_abs * ret_frac, 8), open_v * 1.001)
                        if sl_label == "ATR SL":
                            # Structural anchor: candle low minus 0.5×ATR14 (matches display logic)
                            bar_low      = float(bar.get("low", close_v))
                            _struct_sl   = bar_low - atr14 * 0.5
                            # Clamp SL to 0.8%–6% band (same as _compute_enhanced_trade_plan)
                            _struct_sl   = max(_struct_sl, close_v * 0.94)
                            _struct_sl   = min(_struct_sl, close_v * 0.992)
                            # ── Guard: skip zone if entry is at or below the structural SL ──
                            # For LONG, entry must be ABOVE sl; if a large-body candle's
                            # retrace target undershoots the SL level the zone is invalid.
                            if entry_target <= _struct_sl:
                                continue
                            _sl_pct      = max(0.008, min(0.06, (entry_target - _struct_sl) / entry_target))
                            sl_px        = round(entry_target - entry_target * _sl_pct, 8)
                        else:
                            sl_px        = round(entry_target * (1 - sl_pct_val), 8)
                    else:
                        entry_target = min(round(close_v + body_abs * ret_frac, 8), open_v * 0.999)
                        if sl_label == "ATR SL":
                            bar_high     = float(bar.get("high", close_v))
                            _struct_sl   = bar_high + atr14 * 0.5
                            # Clamp SL to 0.8%–6% band (same as _compute_enhanced_trade_plan)
                            _struct_sl   = min(_struct_sl, close_v * 1.06)
                            _struct_sl   = max(_struct_sl, close_v * 1.008)
                            # ── Guard: skip zone if entry is at or above the structural SL ──
                            # For SHORT, entry must be BELOW sl; if a large-body candle's
                            # retrace target overshoots the SL level the zone is invalid.
                            if entry_target >= _struct_sl:
                                continue
                            _sl_pct      = max(0.008, min(0.06, (_struct_sl - entry_target) / entry_target))
                            sl_px        = round(entry_target + entry_target * _sl_pct, 8)
                        else:
                            sl_px        = round(entry_target * (1 + sl_pct_val), 8)

                    risk_amt = abs(entry_target - sl_px)
                    if risk_amt <= 0:
                        continue

                    if direction == "long":
                        tp1_px = entry_target + 1.0    * risk_amt
                        tp2_px = entry_target + tp_mult * risk_amt
                    else:
                        tp1_px = entry_target - 1.0    * risk_amt
                        tp2_px = entry_target - tp_mult * risk_amt

                    entry_filled   = (ret_frac == 0.0)
                    entry_fill_bar = i if entry_filled else None
                    current_sl     = sl_px
                    be_moved       = False
                    partial_done   = False
                    result         = "OPEN"
                    bars_held      = 0
                    r_mult         = 0.0
                    scan_range_end = min(i + 1 + MAX_HOLD, n_df)

                    for j in range(i + 1, min(i + 1 + MAX_HOLD + max(expiry, 0) + 1, n_df)):
                        fb    = df.iloc[j]
                        hi    = float(fb["high"])
                        lo    = float(fb["low"])
                        atr_j = float(fb.get("atr14", atr14) or atr14)

                        if not entry_filled:
                            fill_cond = (lo <= entry_target if direction == "long"
                                         else hi >= entry_target)
                            if fill_cond:
                                entry_filled   = True
                                entry_fill_bar = j
                                scan_range_end = min(j + 1 + MAX_HOLD, n_df)
                            else:
                                if expiry > 0 and (j - i) >= expiry:
                                    result = "EXPIRED"; break
                                if direction == "long":
                                    if lo > entry_target + 2 * risk_amt:
                                        result = "EXPIRED"; break
                                else:
                                    if hi < entry_target - 2 * risk_amt:
                                        result = "EXPIRED"; break
                                continue

                        bars_held = j - entry_fill_bar

                        if j >= scan_range_end:
                            ep = float(fb.get("close", entry_target))
                            r_mult = (((ep - entry_target) / risk_amt) if direction == "long"
                                      else ((entry_target - ep) / risk_amt)) - 0.002
                            if partial_done:
                                r_mult = (1.0 * 0.5 + r_mult * 0.5) - 0.002
                            result = "WIN" if r_mult > 0 else "LOSS"; break

                        # Trailing SL update
                        if mgmt == "Trailing" and be_moved and atr_j > 0:
                            if direction == "long":
                                current_sl = max(current_sl, float(fb["close"]) - 0.5 * atr_j)
                            else:
                                current_sl = min(current_sl, float(fb["close"]) + 0.5 * atr_j)

                        # Breakeven at 1R
                        if mgmt in ("Partial", "Trailing") and not be_moved:
                            trigger_1r = (hi >= tp1_px if direction == "long" else lo <= tp1_px)
                            if trigger_1r:
                                be_moved   = True
                                current_sl = entry_target

                        # Partial exit at 1R
                        if mgmt == "Partial" and not partial_done:
                            if direction == "long":
                                if hi >= tp1_px:
                                    partial_done = True
                            else:
                                if lo <= tp1_px:
                                    partial_done = True

                        # SL hit
                        sl_hit = (lo <= current_sl if direction == "long" else hi >= current_sl)
                        if sl_hit:
                            sl_r   = ((current_sl - entry_target) / risk_amt if direction == "long"
                                      else (entry_target - current_sl) / risk_amt)
                            r_mult = ((1.0 * 0.5 + sl_r * 0.5) if partial_done else sl_r) - 0.002
                            result = "WIN" if r_mult > 0 else "LOSS"; break

                        # TP full exit (at tp_mult R)
                        tp2_hit = (hi >= tp2_px if direction == "long" else lo <= tp2_px)
                        if tp2_hit:
                            r_mult = ((1.0 * 0.5 + tp_mult * 0.5) if partial_done else tp_mult) - 0.002
                            result = "WIN"; break

                    if result in ("WIN", "LOSS"):
                        trades_raw.append({
                            "result":    result,
                            "r_mult":    r_mult,
                            "bars_held": bars_held,
                            "bar_index": i,   # entry signal bar — used for time-decay buckets
                        })

                if len(trades_raw) < 3:
                    method_results[key] = {
                        "zone": zone_name, "sl_label": sl_label, "mgmt": mgmt, "tp_mult": tp_mult,
                        "n": len(trades_raw), "win_rate": 0, "ev": 0, "pf": 0,
                        "ev_weighted": 0, "wr_weighted": 0,
                        "avg_r": 0, "avg_bars": 0, "insufficient": True,
                        "buckets": [],
                    }
                    continue

                rs    = [t["r_mult"] for t in trades_raw]
                wins  = [r for r in rs if r > 0]
                losses= [r for r in rs if r <= 0]
                wr    = len(wins) / len(rs)
                avg_r = float(np.mean(rs))
                avg_b = float(np.mean([t["bars_held"] for t in trades_raw]))

                # Profit factor = gross profit / gross loss
                gp = sum(wins)
                gl = abs(sum(losses))
                if gl > 0:
                    pf_val = round(gp / gl, 3)
                elif gp > 0:
                    pf_val = 9.99    # sentinel: all wins, no losses
                else:
                    pf_val = 0.0

                # Time-decay bucket stats for this method
                bucket_rows, ev_weighted, wr_weighted = _bucket_stats_for_trades(
                    trades_raw, n_df, _decay_buckets
                )
                # PF for the newest bucket specifically (for "best of last bucket" picker)
                _newest = bucket_rows[-1] if bucket_rows else {"n": 0, "wr": 0, "ev": 0}

                method_results[key] = {
                    "zone": zone_name, "sl_label": sl_label, "mgmt": mgmt, "tp_mult": tp_mult,
                    "n": len(trades_raw), "win_rate": round(wr * 100, 1),
                    "ev": round(avg_r, 3),
                    "pf": pf_val,
                    "ev_weighted": ev_weighted,
                    "wr_weighted": wr_weighted,
                    "avg_r": round(avg_r, 3),
                    "avg_bars": round(avg_b, 1),
                    "insufficient": False,
                    "buckets": bucket_rows,
                    "newest_bucket": {
                        "n":  _newest.get("n",  0),
                        "wr": _newest.get("wr", 0),
                        "ev": _newest.get("ev", 0),
                    },
                }

    # Determine structurally invalid zones from the signal's trade plan.
    # These zones must never be recommended even if historical trades were found
    # (the backtest ran with a clamped SL workaround — the display correctly rejects them).
    _etp_for_filter = sig.get("_trade_plan", {})
    _invalid_zones  = set()
    if not _etp_for_filter.get("std_valid",    True):
        _invalid_zones.add("Standard")
    if not _etp_for_filter.get("sniper_valid", True):
        _invalid_zones.add("Sniper")

    # Best overall method — exclude structurally invalid zones
    valid    = {k: v for k, v in method_results.items()
                if not v.get("insufficient")
                and v["n"] >= 4
                and v["win_rate"] >= 35
                and v.get("zone", "Aggressive") not in _invalid_zones}
    best_key = max(valid, key=lambda k: valid[k]["ev"]) if valid else None
    best     = method_results.get(best_key, {}) if best_key else {}

    # Best per zone — apply same 35% WR floor as the overall valid filter.
    # This ensures the card stats and EXECUTE THIS always describe comparable configs.
    # If nothing passes the floor, fall back to best available and flag it.
    zone_best = {}
    for zn in ("Aggressive", "Standard", "Sniper"):
        if zn in _invalid_zones:
            zone_best[zn] = {"structurally_invalid": True, "zone": zn}
            continue
        zm_all   = {k: v for k, v in method_results.items()
                    if v.get("zone") == zn and not v.get("insufficient") and v["n"] >= 4}
        zm_valid = {k: v for k, v in zm_all.items() if v.get("win_rate", 0) >= 35}
        if zm_valid:
            bk = max(zm_valid, key=lambda k: zm_valid[k]["ev"])
            zone_best[zn] = {**zm_valid[bk], "key": bk, "below_wr_floor": False}
        elif zm_all:
            # Nothing passes 35% floor — show best available but flag it
            bk = max(zm_all, key=lambda k: zm_all[k]["ev"])
            zone_best[zn] = {**zm_all[bk], "key": bk, "below_wr_floor": True}

    # ── Candidate A: best method in the NEWEST time bucket ──────────────────
    # (what's working right now, regardless of ancient history)
    _cand_newest_key = None
    _cand_newest     = None
    _newest_pool = {
        k: v for k, v in method_results.items()
        if not v.get("insufficient")
        and v.get("newest_bucket", {}).get("n", 0) >= 3
        and v.get("newest_bucket", {}).get("wr", 0) >= 35
        and v.get("zone", "Aggressive") not in _invalid_zones
    }
    if _newest_pool:
        _cand_newest_key = max(_newest_pool,
            key=lambda k: _newest_pool[k]["newest_bucket"]["ev"])
        _cand_newest = {
            **method_results[_cand_newest_key],
            "key": _cand_newest_key,
            "method_cfg": {
                "zone":     method_results[_cand_newest_key]["zone"],
                "sl_label": method_results[_cand_newest_key]["sl_label"],
                "mgmt":     method_results[_cand_newest_key]["mgmt"],
                "tp_mult":  method_results[_cand_newest_key]["tp_mult"],
            },
        }

    # ── Candidate B: best method by time-decay WEIGHTED EV (all-time) ────────
    # Accounts for all history but newer trades count more (via bucket weights)
    _cand_weighted_key = None
    _cand_weighted     = None
    _weighted_pool = {
        k: v for k, v in method_results.items()
        if not v.get("insufficient")
        and v["n"] >= 4
        and v["win_rate"] >= 35
        and v.get("zone", "Aggressive") not in _invalid_zones
    }
    if _weighted_pool:
        _cand_weighted_key = max(_weighted_pool,
            key=lambda k: _weighted_pool[k].get("ev_weighted", -99))
        _cand_weighted = {
            **method_results[_cand_weighted_key],
            "key": _cand_weighted_key,
            "method_cfg": {
                "zone":     method_results[_cand_weighted_key]["zone"],
                "sl_label": method_results[_cand_weighted_key]["sl_label"],
                "mgmt":     method_results[_cand_weighted_key]["mgmt"],
                "tp_mult":  method_results[_cand_weighted_key]["tp_mult"],
            },
        }

    # Legacy compat fields
    leg = method_results.get("Aggressive / Fixed SL / Simple / TP2.0R", {})

    # Data provenance metadata — surfaced in UI so user knows what data was used
    _meta = {
        "bars_requested": deep_limit,
        "bars_used":      n_df,
        "bars_coverage":  f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}",
        "bucket_count":   _decay_buckets["count"],
        "bucket_weights": _decay_buckets["weights"],
        "bucket_labels":  _decay_buckets["labels"],
    }

    return {
        "n":          leg.get("n", 0),
        "win_2r":     leg.get("win_rate", 0),
        "win_3r":     leg.get("win_rate", 0),
        "ev_2r":      leg.get("ev", 0),
        "ev_3r":      leg.get("ev", 0),
        "avg_bars":   leg.get("avg_bars", 0),
        "error":      None if method_results else "No matching historical setups found",
        "per_method": method_results,
        "zone_best":  zone_best,
        "best_key":   best_key,
        "best":       best,
        "meta":       _meta,
        "candidate_newest":   _cand_newest,
        "candidate_weighted": _cand_weighted,
    }


def _scanner_mini_wfo(sig: dict, bt_results: dict) -> dict:
    """
    Mini Walk-Forward Validation for the scanner.
    Uses the BEST method from _scanner_quick_backtest on the IS (first 70%)
    window, then re-runs it on the OOS (last 30%) window.

    ok=True  whenever WFO actually ran (even INSUFFICIENT) — UI always shows result.
    ok=False only when we cannot start at all (no data, no valid method to test).
    verdict: PASS / BORDERLINE / FAIL / INSUFFICIENT
    """
    import math

    symbol    = sig["symbol"]
    timeframe = sig["timeframe"]
    direction = sig["direction"]

    # ── Resolve best method FIRST so every return path can report it ──────────
    best     = bt_results.get("best", {})
    best_key = bt_results.get("best_key", "") or ""
    if not best_key or best.get("insufficient"):
        return {
            "ok":          False,
            "verdict":     "INSUFFICIENT",
            "method_used": best_key or "—",
            "note":        "Backtest found no valid method (need ≥ 4 trades). WFO cannot run.",
        }

    zone_name   = best.get("zone",     "Aggressive")
    sl_label    = best.get("sl_label", "Fixed SL")
    mgmt        = best.get("mgmt",     "Simple")
    _etp        = sig.get("_trade_plan", {})
    atr_sl_pct  = (_etp.get("sl_dist_pct", 2.0) or 2.0) / 100.0
    atr_sl_pct  = max(0.008, min(0.06, atr_sl_pct))
    FIXED_SL    = 0.015
    MAX_HOLD    = 20

    _ZONE_CFG = {
        "Aggressive": {"retrace": 0.000, "expiry_bars": 0},
        "Standard":   {"retrace": 0.382, "expiry_bars": 3},
        "Sniper":     {"retrace": 0.618, "expiry_bars": 3},
    }
    ret_frac = _ZONE_CFG.get(zone_name, {}).get("retrace", 0.0)
    expiry   = _ZONE_CFG.get(zone_name, {}).get("expiry_bars", 0)

    # Use same body/vol thresholds as _scanner_quick_backtest (70% of signal values)
    min_body = abs(sig["body_pct"]) * 0.70 / 100
    min_vol  = max(1.0, sig["vol_mult"] * 0.70)

    # ── Fetch data ────────────────────────────────────────────────────────────
    interval = _BINANCE_INTERVAL.get(timeframe, "1d")
    # Same deep fetch as _scanner_quick_backtest (up to 1000 bars)
    df = _scanner_fetch_candles(symbol, interval, limit=_deep_limit_for(timeframe))
    if df.empty or len(df) < 60:
        return {
            "ok":          False,
            "verdict":     "INSUFFICIENT",
            "method_used": best_key,
            "note":        "< 60 bars available — insufficient historical data. WFO skipped.",
        }

    # ── Split IS (70%) / OOS (30%) ────────────────────────────────────────────
    n_total = len(df)
    is_end  = int(n_total * 0.70)
    df_is   = df.iloc[:is_end]
    df_oos  = df.iloc[is_end:]

    if len(df_is) < 30 or len(df_oos) < 15:
        return {
            "ok":          False,
            "verdict":     "INSUFFICIENT",
            "method_used": best_key,
            "note":        "Not enough bars for IS/OOS split. WFO skipped.",
        }

    # ── Inner simulate function ───────────────────────────────────────────────
    def _run_split(df_split):
        trades = []
        n = len(df_split)
        for i in range(14, n - 2):
            bar      = df_split.iloc[i]
            body_pct = float(bar.get("body_pct", 0) or 0)
            vol_mult = float(bar.get("vol_mult",  0) or 0)
            is_bull  = body_pct > 0
            if direction == "long"  and not is_bull: continue
            if direction == "short" and is_bull:     continue
            if abs(body_pct) < min_body: continue
            if vol_mult < min_vol:       continue

            close_v  = float(bar["close"])
            open_v   = float(bar.get("open", close_v))
            body_abs = abs(close_v - open_v)
            atr14    = float(bar.get("atr14", close_v * 0.02) or close_v * 0.02)
            if close_v <= 0:
                continue

            if direction == "long":
                entry_target = max(round(close_v - body_abs * ret_frac, 8), open_v * 1.001)
                if sl_label == "ATR SL":
                    bar_low    = float(bar.get("low", close_v))
                    _struct_sl = bar_low - atr14 * 0.5
                    if entry_target <= _struct_sl:
                        continue
                    _sp = max(0.008, min(0.06, (entry_target - _struct_sl) / entry_target))
                    sl_px = round(entry_target - entry_target * _sp, 8)
                else:
                    sl_px = round(entry_target * (1 - FIXED_SL), 8)
            else:
                entry_target = min(round(close_v + body_abs * ret_frac, 8), open_v * 0.999)
                if sl_label == "ATR SL":
                    bar_high   = float(bar.get("high", close_v))
                    _struct_sl = bar_high + atr14 * 0.5
                    if entry_target >= _struct_sl:
                        continue
                    _sp = max(0.008, min(0.06, (_struct_sl - entry_target) / entry_target))
                    sl_px = round(entry_target + entry_target * _sp, 8)
                else:
                    sl_px = round(entry_target * (1 + FIXED_SL), 8)

            risk_amt = abs(entry_target - sl_px)
            if risk_amt <= 0:
                continue

            if direction == "long":
                tp1_px = entry_target + 1.0 * risk_amt
                tp2_px = entry_target + 2.0 * risk_amt
            else:
                tp1_px = entry_target - 1.0 * risk_amt
                tp2_px = entry_target - 2.0 * risk_amt

            entry_filled   = (ret_frac == 0.0)
            entry_fill_bar = i if entry_filled else None
            current_sl     = sl_px
            be_moved       = False
            partial_done   = False
            result         = "OPEN"
            r_mult         = 0.0
            scan_end       = min(i + 1 + MAX_HOLD, n)

            for j in range(i + 1, min(i + 1 + MAX_HOLD + max(expiry, 0) + 1, n)):
                fb = df_split.iloc[j]
                hi = float(fb["high"])
                lo = float(fb["low"])
                atr_j = float(fb.get("atr14", atr14) or atr14)

                if not entry_filled:
                    fill = (lo <= entry_target if direction == "long" else hi >= entry_target)
                    if fill:
                        entry_filled   = True
                        entry_fill_bar = j
                        scan_end       = min(j + 1 + MAX_HOLD, n)
                    else:
                        if expiry > 0 and (j - i) >= expiry:
                            break
                        if direction == "long" and lo > entry_target + 2 * risk_amt:
                            break
                        if direction == "short" and hi < entry_target - 2 * risk_amt:
                            break
                        continue

                if j >= scan_end:
                    ep     = float(fb.get("close", entry_target))
                    r_mult = (((ep - entry_target) / risk_amt) if direction == "long"
                               else ((entry_target - ep) / risk_amt)) - 0.002
                    if partial_done:
                        r_mult = (1.0 * 0.5 + r_mult * 0.5) - 0.002
                    result = "WIN" if r_mult > 0 else "LOSS"
                    break

                # Management
                if mgmt == "Trailing" and be_moved and atr_j > 0:
                    if direction == "long":
                        current_sl = max(current_sl, float(fb["close"]) - 0.5 * atr_j)
                    else:
                        current_sl = min(current_sl, float(fb["close"]) + 0.5 * atr_j)
                if mgmt in ("Partial", "Trailing") and not be_moved:
                    t1h = (hi >= tp1_px if direction == "long" else lo <= tp1_px)
                    if t1h:
                        be_moved   = True
                        current_sl = entry_target
                if mgmt == "Partial" and not partial_done:
                    t1h = (hi >= tp1_px if direction == "long" else lo <= tp1_px)
                    if t1h:
                        partial_done = True

                sl_hit = (lo <= current_sl if direction == "long" else hi >= current_sl)
                if sl_hit:
                    sl_r   = ((current_sl - entry_target) / risk_amt if direction == "long"
                              else (entry_target - current_sl) / risk_amt)
                    r_mult = ((1.0 * 0.5 + sl_r * 0.5) if partial_done else sl_r) - 0.002
                    result = "WIN" if r_mult > 0 else "LOSS"
                    break
                tp2h = (hi >= tp2_px if direction == "long" else lo <= tp2_px)
                if tp2h:
                    r_mult = ((1.0 * 0.5 + 2.0 * 0.5) if partial_done else 2.0) - 0.002
                    result = "WIN"
                    break

            if result in ("WIN", "LOSS"):
                trades.append(r_mult)

        return trades

    is_trades  = _run_split(df_is)
    oos_trades = _run_split(df_oos)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _pf(ts):
        wins   = [r for r in ts if r > 0]
        losses = [r for r in ts if r <= 0]
        gp = sum(wins)
        gl = abs(sum(losses))
        return round(gp / gl, 3) if gl > 0 else (9.99 if gp > 0 else 0.0)  # 9.99 = no losses yet

    def _wr(ts):
        return round(sum(1 for r in ts if r > 0) / len(ts) * 100, 1) if ts else 0.0

    is_n   = len(is_trades)
    oos_n  = len(oos_trades)
    is_pf  = _pf(is_trades)
    oos_pf = _pf(oos_trades)
    oos_wr = _wr(oos_trades)

    # Low IS sample — still return ok=True with all fields so UI can display
    # the situation clearly rather than hiding it. 3+ IS trades is still
    # informative even if not statistically robust.
    if is_n < 3:
        return {
            "ok":           True,
            "verdict":      "INSUFFICIENT",
            "is_pf":        is_pf,
            "is_n":         is_n,
            "oos_pf":       oos_pf,
            "oos_wr":       oos_wr,
            "oos_n":        oos_n,
            "oos_is_ratio": 0.0,
            "method_used":  best_key,
            "tier_label":   "Single IS/OOS split (70% IS / 30% OOS)",
            "note":         f"Only {is_n} IS trades — insufficient for statistical validation. WFO skipped; interpret backtest with caution.",
        }

    oos_is_ratio = round(min(oos_pf / is_pf, 2.0), 3) if is_pf > 0 else 0.0

    # ── Verdict ───────────────────────────────────────────────────────────────
    if oos_n < 3:
        verdict = "INSUFFICIENT"
        note    = f"Only {oos_n} OOS trades (need ≥ 3 to judge)"
    elif oos_pf >= 1.3 and oos_is_ratio >= 0.60:
        verdict = "PASS"
        note    = "OOS edge confirmed — params generalize"
    elif oos_pf >= 1.0 and oos_is_ratio >= 0.40:
        verdict = "BORDERLINE"
        note    = "Marginal OOS — edge may not fully generalize"
    else:
        verdict = "FAIL"
        note    = "OOS underperforms IS significantly — possible overfitting"

    return {
        "ok":           True,
        "verdict":      verdict,
        "is_pf":        is_pf,
        "is_n":         is_n,
        "oos_pf":       oos_pf,
        "oos_wr":       oos_wr,
        "oos_n":        oos_n,
        "oos_is_ratio": oos_is_ratio,
        "method_used":  best_key,
        "tier_label":   "Single IS/OOS split (70% IS / 30% OOS)",
        "note":         note,
    }


def _scanner_heuristic_ml(sig: dict) -> dict:
    """
    Compute a weighted heuristic ML probability from signal features.
    Acts as an ML confirmation without needing a pre-trained model.
    Returns probability (0-1), percentage, and HIGH/MEDIUM/LOW label.
    """
    score = 0.0
    total = 0.0

    # Body conviction — weight 2.0
    body_score = min(sig["body_pct"] / 90.0, 1.0)
    score += body_score * 2.0;  total += 2.0

    # Volume surge — weight 1.5
    vol_score = min(max(sig["vol_mult"] - 1.0, 0) / 4.0, 1.0)
    score += vol_score * 1.5;   total += 1.5

    # ADX trend strength — weight 1.5
    adx_score = min(sig["adx"] / 40.0, 1.0)
    score += adx_score * 1.5;   total += 1.5

    # DI directional alignment — weight 1.0
    if sig["direction"] == "long":
        di_gap = max(sig["di_plus"] - sig["di_minus"], 0)
    else:
        di_gap = max(sig["di_minus"] - sig["di_plus"], 0)
    di_score = min(di_gap / 30.0, 1.0)
    score += di_score * 1.0;    total += 1.0

    # EMA stack — weight 1.0
    ema_score = 1.0 if sig.get("ema_full") else (0.5 if sig.get("ema_partial") else 0.0)
    score += ema_score * 1.0;   total += 1.0

    # Market regime — weight 2.0
    regime_score = sig.get("regime_score", 0) / 100.0
    score += regime_score * 2.0; total += 2.0

    # Candle rank (top N% of last 20 bars) — weight 0.5
    score += sig.get("candle_rank", 0.5) * 0.5; total += 0.5

    # Volume rank — weight 0.5
    score += sig.get("vol_rank", 0.5) * 0.5; total += 0.5

    # ATR ratio (volatility expansion is bullish for momentum) — weight 0.5
    atr_score = min(max(sig.get("atr_ratio", 1.0) - 0.7, 0) / 1.3, 1.0)
    score += atr_score * 0.5;   total += 0.5

    prob = score / total if total > 0 else 0.5
    prob = max(0.30, min(0.95, prob))   # clamp to realistic range

    return {
        "probability": round(prob, 3),
        "pct":         round(prob * 100, 1),
        "label":       "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.55 else "LOW"),
        # Compat fields so display code can handle heuristic and trained ML uniformly
        "method_name":        "Heuristic (hand-weighted)",
        "method_reason":      "No backtest method chosen — showing weighted formula fallback",
        "n_samples":          0,
        "n_wins":             0,
        "n_losses":           0,
        "cv_accuracy":        None,
        "cv_std":             None,
        "feature_importance": [],
        "note":               "Not trained on historical outcomes — pick a method & click Train ML.",
        "method_cfg":         None,
        "ok":                 True,
        "trained":            False,
    }


def _scanner_train_ml(sig: dict, method_cfg: dict) -> dict:
    """
    Train an adaptive ML classifier on historical qualifying candles
    labeled by the outcome of a specific trade method (entry zone, SL, mgmt, TP).

    Auto-selects model based on training sample count:
      n <  50 : Logistic Regression   (StandardScaler pipeline)
      50-150  : Random Forest         (max_depth=5, min_leaf=5)
      n >=150 : Gradient Boosting     (max_depth=3, lr=0.05)

    Returns dict with:
      probability, pct, label, method_name, method_reason, n_samples,
      n_wins, n_losses, cv_accuracy, cv_std, feature_importance, note,
      method_cfg, ok, trained
    """
    # Fallback shell — we update & return it on any early-exit path
    def _heuristic_fallback(note: str, method_label: str):
        h = _scanner_heuristic_ml(sig)
        h.update({
            "method_name":  method_label,
            "note":         note,
            "method_cfg":   method_cfg,
            "ok":           False,
            "trained":      False,
        })
        return h

    if not _SKLEARN_OK:
        return _heuristic_fallback(
            "sklearn not installed — pip install scikit-learn to enable trained ML.",
            "Heuristic (sklearn missing)",
        )

    symbol    = sig["symbol"]
    timeframe = sig["timeframe"]
    direction = sig["direction"]
    interval  = _BINANCE_INTERVAL.get(timeframe, "1d")

    zone_name = method_cfg.get("zone",     "Aggressive")
    sl_label  = method_cfg.get("sl_label", "Fixed SL")
    mgmt      = method_cfg.get("mgmt",     "Simple")
    tp_mult   = float(method_cfg.get("tp_mult", 2.0))

    _etp       = sig.get("_trade_plan", {})
    atr_sl_pct = (_etp.get("sl_dist_pct", 2.0) or 2.0) / 100.0
    atr_sl_pct = max(0.008, min(0.06, atr_sl_pct))
    FIXED_SL   = 0.015
    MAX_HOLD   = 20

    _ZONE_CFG = {
        "Aggressive": {"retrace": 0.000, "expiry_bars": 0},
        "Standard":   {"retrace": 0.382, "expiry_bars": 3},
        "Sniper":     {"retrace": 0.618, "expiry_bars": 3},
    }
    ret_frac = _ZONE_CFG.get(zone_name, {}).get("retrace",     0.0)
    expiry   = _ZONE_CFG.get(zone_name, {}).get("expiry_bars", 0)

    min_body = abs(sig["body_pct"]) * 0.70 / 100
    min_vol  = max(1.0, sig["vol_mult"] * 0.70)

    # Deep fetch — same depth as backtest
    deep_limit = _deep_limit_for(timeframe)
    df = _scanner_fetch_candles(symbol, interval, limit=deep_limit)
    if df.empty or len(df) < 40:
        return _heuristic_fallback(
            f"Only {len(df) if not df.empty else 0} bars available — need ≥40 for ML training.",
            "Heuristic (insufficient data)",
        )

    # ADX frame for feature extraction at historical bars
    try:
        adx_df = calculate_adx(df)
    except Exception:
        adx_df = pd.DataFrame()

    n_df = len(df)
    features_list = []
    labels_list   = []

    for i in range(14, n_df - 2):
        bar      = df.iloc[i]
        body_pct = float(bar.get("body_pct", 0) or 0)
        vol_mult = float(bar.get("vol_mult",  0) or 0)
        is_bull  = body_pct > 0
        if direction == "long"  and not is_bull: continue
        if direction == "short" and is_bull:     continue
        if abs(body_pct) < min_body: continue
        if vol_mult < min_vol:       continue

        close_v  = float(bar["close"])
        open_v   = float(bar.get("open",  close_v))
        body_abs = abs(close_v - open_v)
        atr14    = float(bar.get("atr14", close_v * 0.02) or close_v * 0.02)
        if close_v <= 0:
            continue

        # Build entry/SL — mirrors _scanner_quick_backtest exactly
        if direction == "long":
            entry_target = max(round(close_v - body_abs * ret_frac, 8), open_v * 1.001)
            if sl_label == "ATR SL":
                bar_low    = float(bar.get("low", close_v))
                _struct_sl = bar_low - atr14 * 0.5
                _struct_sl = max(_struct_sl, close_v * 0.94)
                _struct_sl = min(_struct_sl, close_v * 0.992)
                if entry_target <= _struct_sl:
                    continue
                _sp   = max(0.008, min(0.06, (entry_target - _struct_sl) / entry_target))
                sl_px = round(entry_target - entry_target * _sp, 8)
            else:
                sl_px = round(entry_target * (1 - FIXED_SL), 8)
        else:
            entry_target = min(round(close_v + body_abs * ret_frac, 8), open_v * 0.999)
            if sl_label == "ATR SL":
                bar_high   = float(bar.get("high", close_v))
                _struct_sl = bar_high + atr14 * 0.5
                _struct_sl = min(_struct_sl, close_v * 1.06)
                _struct_sl = max(_struct_sl, close_v * 1.008)
                if entry_target >= _struct_sl:
                    continue
                _sp   = max(0.008, min(0.06, (_struct_sl - entry_target) / entry_target))
                sl_px = round(entry_target + entry_target * _sp, 8)
            else:
                sl_px = round(entry_target * (1 + FIXED_SL), 8)

        risk_amt = abs(entry_target - sl_px)
        if risk_amt <= 0:
            continue

        if direction == "long":
            tp1_px = entry_target + 1.0     * risk_amt
            tp2_px = entry_target + tp_mult * risk_amt
        else:
            tp1_px = entry_target - 1.0     * risk_amt
            tp2_px = entry_target - tp_mult * risk_amt

        # Simulate the trade with the specified management
        entry_filled   = (ret_frac == 0.0)
        entry_fill_bar = i if entry_filled else None
        current_sl     = sl_px
        be_moved       = False
        partial_done   = False
        result         = "OPEN"
        r_mult         = 0.0
        scan_end       = min(i + 1 + MAX_HOLD, n_df)

        for j in range(i + 1, min(i + 1 + MAX_HOLD + max(expiry, 0) + 1, n_df)):
            fb    = df.iloc[j]
            hi    = float(fb["high"])
            lo    = float(fb["low"])
            atr_j = float(fb.get("atr14", atr14) or atr14)

            if not entry_filled:
                fill = (lo <= entry_target if direction == "long" else hi >= entry_target)
                if fill:
                    entry_filled   = True
                    entry_fill_bar = j
                    scan_end       = min(j + 1 + MAX_HOLD, n_df)
                else:
                    if expiry > 0 and (j - i) >= expiry:
                        break
                    if direction == "long"  and lo > entry_target + 2 * risk_amt: break
                    if direction == "short" and hi < entry_target - 2 * risk_amt: break
                    continue

            if j >= scan_end:
                ep = float(fb.get("close", entry_target))
                r_mult = (((ep - entry_target) / risk_amt) if direction == "long"
                          else ((entry_target - ep) / risk_amt)) - 0.002
                if partial_done:
                    r_mult = (1.0 * 0.5 + r_mult * 0.5) - 0.002
                result = "WIN" if r_mult > 0 else "LOSS"
                break

            if mgmt == "Trailing" and be_moved and atr_j > 0:
                if direction == "long":
                    current_sl = max(current_sl, float(fb["close"]) - 0.5 * atr_j)
                else:
                    current_sl = min(current_sl, float(fb["close"]) + 0.5 * atr_j)
            if mgmt in ("Partial", "Trailing") and not be_moved:
                t1h = (hi >= tp1_px if direction == "long" else lo <= tp1_px)
                if t1h:
                    be_moved   = True
                    current_sl = entry_target
            if mgmt == "Partial" and not partial_done:
                t1h = (hi >= tp1_px if direction == "long" else lo <= tp1_px)
                if t1h:
                    partial_done = True

            sl_hit = (lo <= current_sl if direction == "long" else hi >= current_sl)
            if sl_hit:
                sl_r   = ((current_sl - entry_target) / risk_amt if direction == "long"
                          else (entry_target - current_sl) / risk_amt)
                r_mult = ((1.0 * 0.5 + sl_r * 0.5) if partial_done else sl_r) - 0.002
                result = "WIN" if r_mult > 0 else "LOSS"
                break
            tp2h = (hi >= tp2_px if direction == "long" else lo <= tp2_px)
            if tp2h:
                r_mult = ((1.0 * 0.5 + tp_mult * 0.5) if partial_done else tp_mult) - 0.002
                result = "WIN"
                break

        if result not in ("WIN", "LOSS"):
            continue

        # ── Extract features at bar i (what was KNOWN at signal time) ─────────
        adx_val, di_plus, di_minus = 0.0, 0.0, 0.0
        if adx_df is not None and not adx_df.empty and i < len(adx_df):
            try:
                _a = float(adx_df["adx"].iloc[i])
                _p = float(adx_df["di_plus"].iloc[i])
                _m = float(adx_df["di_minus"].iloc[i])
                adx_val  = _a if _a == _a else 0.0
                di_plus  = _p if _p == _p else 0.0
                di_minus = _m if _m == _m else 0.0
            except Exception:
                pass

        ema5_v  = float(bar.get("ema5",  0) or 0)
        ema15_v = float(bar.get("ema15", 0) or 0)
        ema21_v = float(bar.get("ema21", 0) or 0)
        if direction == "long":
            ema_full    = (ema5_v > ema15_v) and (ema15_v > ema21_v)
            ema_partial = (ema5_v > ema15_v) or  (ema15_v > ema21_v)
            di_gap      = max(di_plus - di_minus, 0)
        else:
            ema_full    = (ema5_v < ema15_v) and (ema15_v < ema21_v)
            ema_partial = (ema5_v < ema15_v) or  (ema15_v < ema21_v)
            di_gap      = max(di_minus - di_plus, 0)
        ema_score = 1.0 if ema_full else (0.5 if ema_partial else 0.0)

        # Regime score (best-effort — calculate_regime_score can be expensive
        # per-bar on long histories; a failure falls through to a neutral 50)
        try:
            _rgm = calculate_regime_score(df, i, direction, adx_df,
                                          timeframe=timeframe, ticker=symbol)
            regime_score = float(_rgm.get("score", 0) or 0)
        except Exception:
            regime_score = 50.0

        feat = [
            abs(body_pct),
            float(vol_mult),
            float(adx_val),
            float(di_gap),
            float(bar.get("atr_ratio", 1.0) or 1.0),
            float(ema_score),
            float(regime_score),
            float(bar.get("candle_rank_20", 0.5) or 0.5),
            float(bar.get("vol_rank_20",    0.5) or 0.5),
        ]
        if any(v != v for v in feat):   # NaN guard
            continue

        features_list.append(feat)
        labels_list.append(1 if result == "WIN" else 0)

    feature_names = ["body_pct", "vol_mult", "adx", "di_gap", "atr_ratio",
                     "ema_score", "regime_score", "candle_rank", "vol_rank"]
    n_samples = len(labels_list)

    if n_samples < 20:
        return _heuristic_fallback(
            f"Only {n_samples} training samples for this method (need ≥20).",
            f"Heuristic (only {n_samples} samples)",
        )

    n_pos = int(sum(labels_list))
    n_neg = n_samples - n_pos
    if n_pos == 0 or n_neg == 0:
        return _heuristic_fallback(
            f"All {n_samples} trades were {'wins' if n_pos else 'losses'} — can't train a classifier.",
            "Heuristic (single class)",
        )

    X = np.array(features_list, dtype=float)
    y = np.array(labels_list,   dtype=int)

    # ── Adaptive model selection based on sample count ───────────────────────
    if n_samples < 50:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
        ])
        method_name   = "Logistic Regression"
        method_reason = f"n={n_samples} < 50 — LR is safest on small samples"
    elif n_samples < 150:
        model = RandomForestClassifier(
            n_estimators=150, max_depth=5,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        method_name   = "Random Forest"
        method_reason = f"n={n_samples} ∈ [50,150) — RF captures non-linear patterns without overfit"
    else:
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=42,
        )
        method_name   = "Gradient Boosting"
        method_reason = f"n={n_samples} ≥ 150 — GB gives best generalization on larger datasets"

    # Time-series CV (walk-forward) — splits scale with sample count
    n_splits = min(5, max(2, n_samples // 15))
    cv_scores = []
    try:
        tss = TimeSeriesSplit(n_splits=n_splits)
        for tr_idx, te_idx in tss.split(X):
            if len(tr_idx) < 5 or len(te_idx) < 2:
                continue
            if len(set(y[tr_idx])) < 2:   # single-class fold — skip
                continue
            model.fit(X[tr_idx], y[tr_idx])
            cv_scores.append(model.score(X[te_idx], y[te_idx]))
    except Exception:
        cv_scores = []

    cv_acc = round(float(np.mean(cv_scores)), 3) if cv_scores else None
    cv_std = round(float(np.std(cv_scores)),  3) if cv_scores else None

    # Final fit on all data
    try:
        model.fit(X, y)
    except Exception as e:
        return _heuristic_fallback(
            f"Model training failed: {str(e)[:60]}",
            "Heuristic (training error)",
        )

    # Feature importance — RF/GB have feature_importances_; LR Pipeline has coef_
    feature_importance = []
    try:
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "coef_"):
            coefs = model.named_steps["clf"].coef_[0]
            _norm = np.abs(coefs).sum() + 1e-9
            imps  = np.abs(coefs) / _norm
        else:
            imps = []
        for name, imp in zip(feature_names, imps):
            feature_importance.append({"feature": name, "importance": round(float(imp), 3)})
        feature_importance.sort(key=lambda x: -x["importance"])
    except Exception:
        feature_importance = []

    # Build the CURRENT signal's feature vector and predict
    if sig["direction"] == "long":
        di_gap_cur = max(float(sig.get("di_plus", 0) or 0) - float(sig.get("di_minus", 0) or 0), 0)
    else:
        di_gap_cur = max(float(sig.get("di_minus", 0) or 0) - float(sig.get("di_plus", 0) or 0), 0)
    ema_score_cur = 1.0 if sig.get("ema_full") else (0.5 if sig.get("ema_partial") else 0.0)

    cur_feat = np.array([[
        abs(float(sig.get("body_pct",    0)   or 0)),
        float(sig.get("vol_mult",        0)   or 0),
        float(sig.get("adx",             0)   or 0),
        float(di_gap_cur),
        float(sig.get("atr_ratio",       1.0) or 1.0),
        float(ema_score_cur),
        float(sig.get("regime_score",    0)   or 0),
        float(sig.get("candle_rank",     0.5) or 0.5),
        float(sig.get("vol_rank",        0.5) or 0.5),
    ]], dtype=float)

    try:
        prob = float(model.predict_proba(cur_feat)[0, 1])
    except Exception:
        prob = 0.5
    prob = max(0.05, min(0.95, prob))

    label = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.50 else "LOW")

    return {
        "probability":        round(prob, 3),
        "pct":                round(prob * 100, 1),
        "label":              label,
        "method_name":        method_name,
        "method_reason":      method_reason,
        "n_samples":          n_samples,
        "n_wins":             n_pos,
        "n_losses":           n_neg,
        "cv_accuracy":        cv_acc,
        "cv_std":             cv_std,
        "feature_importance": feature_importance,
        "note":               "",
        "method_cfg":         method_cfg,
        "ok":                 True,
        "trained":            True,
    }


def _scanner_setup_grade(sig: dict, ml: dict, bt: dict) -> tuple:
    """
    Return (grade, color, description) based on all available evidence.
    Grades: A+ / A / B / C / D
    Backtest with n >= 10 is required to matter. Low-n or bad backtest downgrades.
    """
    score    = sig["score"]
    regime   = sig["regime"]
    ema_full = sig.get("ema_full", False)
    adx      = sig["adx"]
    ml_pct   = ml["pct"]

    bt_valid = bt.get("error") is None and bt.get("n", 0) >= 10
    bt_weak  = bt.get("error") is None and 3 <= bt.get("n", 0) < 10   # low sample
    win2     = bt.get("win_2r", 0)
    ev2      = bt.get("ev_2r",  0)

    # Hard downgrade: backtest has enough data and clearly fails
    bt_failed = bt_valid and (win2 < 40 or ev2 < -0.2)

    if bt_failed:
        # Even a high-score signal should not grade above B if backtest says it loses
        if ml_pct >= 65 and score >= 65 and regime == "GREEN":
            return "B",  "#e3b341", "Caution — historical backtest underperforms"
        return "C", "#f85149", "Backtest negative — historical edge not confirmed"

    # Low sample backtest — flag it but don't use it for grading
    bt_note = " (few historical setups)" if bt_weak else ""

    if (score >= 78 and regime == "GREEN" and ema_full
            and adx >= 28 and ml_pct >= 68
            and (not bt_valid or win2 >= 52)):
        return "A+", "#3fb950", f"Exceptional — all filters aligned{bt_note}"
    if (score >= 68 and regime == "GREEN" and ml_pct >= 60
            and (not bt_valid or win2 >= 48)):
        return "A",  "#64ffda", f"Strong — most filters confirmed{bt_note}"
    if (score >= 55 and regime in ("GREEN", "YELLOW") and ml_pct >= 50):
        return "B",  "#e3b341", f"Moderate — proceed with caution{bt_note}"
    return "C", "#f85149", "Weak — wait for better conditions"


def render_auto_analyzer(ticker: str, df_full_1d: pd.DataFrame, tc: float,
                          current_tf: str):
    """
    Market Scanner — scans ALL liquid Binance altcoins across 1H / 4H / Daily
    for live momentum signals. Ranks all qualifying signals by composite score with point-by-point reasons.
    (Replaces the old single-ticker parameter sweep Auto Finder.)
    """
    import concurrent.futures

    st.markdown("## 🔭 Market Scanner — Top Altcoin Opportunities Right Now")
    st.markdown(
        '<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;'
        'padding:12px 16px;margin-bottom:16px;font-size:13px;color:#ccd6f6;">'
        '<b style="color:#58a6ff;">How it works:</b> Fetches every liquid USDT altcoin on Binance, '
        'scans the last 3 closed candles on each timeframe you select, scores each signal '
        '0–100 using body strength, volume spike, ADX trend, and market regime — '
        'then shows you <b>all qualifying setups ranked by composite score</b> with '
        'point-by-point reasons for every pick. '
        '<b>Regime RED signals are automatically excluded.</b></div>',
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        _vol_options = [500_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000]
        _vol_labels  = ["$500K", "$1M", "$5M", "$10M", "$25M", "$50M"]
        _vol_idx     = st.select_slider(
            "Min 24h Volume",
            options=range(len(_vol_options)),
            value=2,
            format_func=lambda i: _vol_labels[i],
            key="mscanner_vol",
        )
        min_vol_usdt = _vol_options[_vol_idx]

    with rc2:
        max_coins = st.select_slider(
            "Coins to scan",
            options=[50, 100, 150, 200, 300],
            value=150,
            format_func=lambda x: f"Top {x} by volume",
            key="mscanner_coins",
        )

    with rc3:
        scan_tfs = st.multiselect(
            "Timeframes",
            ["1H", "4H", "1D"],
            default=["1H", "4H", "1D"],
            key="mscanner_tfs",
        )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        min_body_pct = st.slider(
            "Min body %", 50, 90, 65, 5, key="mscanner_body",
            help="Candle body as % of total range. 65% = solid momentum, 80% = very strong.",
        ) / 100
    with sc2:
        min_vol_mult = st.slider(
            "Min volume ×", 1.0, 5.0, 1.5, 0.5, key="mscanner_volmult",
            help="Volume multiplier vs 7-bar average. 1.5× = elevated, 3.0× = exceptional.",
        )
    with sc3:
        scan_dirs = st.multiselect(
            "Direction",
            ["long", "short"],
            default=["long"],
            key="mscanner_dir",
        )

    if not scan_tfs:
        st.warning("Select at least one timeframe.")
        return
    if not scan_dirs:
        st.warning("Select at least one direction (long/short).")
        return

    # ── Scan button ────────────────────────────────────────────────────────────
    scan_key = f"mscanner_{min_vol_usdt}_{max_coins}_{'_'.join(sorted(scan_tfs))}_{'_'.join(sorted(scan_dirs))}_{min_body_pct}_{min_vol_mult}"
    _prev_key     = st.session_state.get("mscanner_key", "")
    _has_results  = "mscanner_results" in st.session_state

    if _has_results and _prev_key != scan_key:
        st.sidebar.warning("⚠️ Scanner settings changed — click **Scan Now** to update.")

    scan_btn = st.button(
        "🔭 Scan Market Now",
        type="primary",
        use_container_width=True,
        key="mscanner_run",
    )

    if not scan_btn and not _has_results:
        st.info("Configure settings above then click **Scan Market Now**. "
                "A scan of 150 coins × 3 timeframes takes ~60–90 seconds.")
        return

    # ── Run scan ───────────────────────────────────────────────────────────────
    if scan_btn:
        # Step 1: Universe
        fetch_placeholder = st.empty()
        fetch_placeholder.info("📡 Fetching Binance universe…")
        universe = _scanner_get_universe(min_vol_usdt)

        if not universe:
            fetch_placeholder.error(
                "❌ Could not fetch Binance universe. Check internet connection.")
            return

        coins = [u["symbol"] for u in universe[:max_coins]]
        fetch_placeholder.success(
            f"✅ Universe: {len(coins)} coins with 24h volume ≥ {_vol_labels[_vol_idx]}")

        # Estimate
        total_tasks = len(coins)   # one task per symbol, all TFs inside
        st.caption(
            f"Scanning {len(coins)} coins × {len(scan_tfs)} timeframe(s) × {len(scan_dirs)} direction(s) "
            f"× 3 candles = up to {len(coins)*len(scan_tfs)*len(scan_dirs)*3:,} signal checks")

        # Step 2: Parallel scan
        progress_bar = st.progress(0.0)
        status_txt   = st.empty()
        all_signals: list = []
        done_count   = 0

        task_args = [
            (sym, scan_tfs, min_body_pct, min_vol_mult, scan_dirs)
            for sym in coins
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futs = {executor.submit(_scan_one_symbol, arg): arg[0] for arg in task_args}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    sigs = fut.result(timeout=15)
                    all_signals.extend(sigs)
                except Exception:
                    pass
                done_count += 1
                progress_bar.progress(done_count / total_tasks)
                if done_count % 10 == 0 or done_count == total_tasks:
                    status_txt.caption(
                        f"Scanned {done_count}/{total_tasks} coins — "
                        f"{len(all_signals)} signals found so far…")

        progress_bar.empty()
        status_txt.empty()

        # Step 3: Sort, deduplicate (keep best per symbol across TFs/dirs)
        # Drop any signal whose score is NaN or None before sorting
        all_signals = [s for s in all_signals
                       if s.get("score") is not None and s.get("score") == s.get("score")]
        all_signals.sort(key=lambda x: x["score"] if x.get("score") == x.get("score") else -1, reverse=True)

        # Deduplicate: keep highest-score signal per (symbol, direction) pair
        seen   = {}
        # Deduplicate: keep highest-score signal per (symbol, direction) — show ALL
        all_signals_deduped = []
        for s in all_signals:
            key = (s["symbol"], s["direction"])
            if key not in seen:
                seen[key] = True
                all_signals_deduped.append(s)

        st.session_state["mscanner_results"]    = all_signals_deduped
        st.session_state["mscanner_all"]        = all_signals[:100]
        st.session_state["mscanner_key"]        = scan_key
        st.session_state["mscanner_scanned_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        st.session_state["mscanner_total_found"] = len(all_signals)

    # ── Render results ─────────────────────────────────────────────────────────
    all_signals_deduped      = st.session_state.get("mscanner_results", [])
    scanned_at  = st.session_state.get("mscanner_scanned_at", "")
    total_found = st.session_state.get("mscanner_total_found", 0)

    if not all_signals_deduped:
        st.warning(
            "No qualifying signals found with current settings. "
            "Try lowering Min Body % or Min Volume ×, "
            "or expand the coin universe.")
        return

    # Summary banner
    regime_counts = {}
    for s in all_signals_deduped:
        regime_counts[s["regime"]] = regime_counts.get(s["regime"], 0) + 1

    _rc_g = regime_counts.get("GREEN",  0)
    _rc_y = regime_counts.get("YELLOW", 0)
    regime_summary = f"<span style='color:#3fb950;font-weight:700;'>{_rc_g} GREEN</span>"
    if _rc_y:
        regime_summary += f" &nbsp; <span style='color:#e3b341;font-weight:700;'>{_rc_y} YELLOW</span>"

    st.markdown(
        f'<div style="background:#0d2818;border:1px solid #238636;border-radius:8px;'
        f'padding:10px 16px;margin:8px 0;font-size:13px;">'
        f'✅ <b style="color:#3fb950;">Scan complete</b> — {scanned_at} &nbsp;|&nbsp; '
        f'{total_found} total signals found &nbsp;|&nbsp; '
        f'Showing {len(all_signals_deduped)} &nbsp;|&nbsp; {regime_summary}</div>',
        unsafe_allow_html=True,
    )

    # Quick summary table
    _dir_icon = {"long": "📈", "short": "📉"}
    _reg_color = {"GREEN": "#3fb950", "YELLOW": "#e3b341", "RED": "#f85149"}

    summary_rows = []
    for i, s in enumerate(all_signals_deduped):
        _etp_s = s.get("_trade_plan", {})
        _sc = s.get("score") or 0
        _sc = float(_sc) if _sc == _sc else 0.0   # NaN guard
        _entry = s.get("entry") or 0
        _entry = float(_entry) if _entry == _entry else 0.0
        summary_rows.append({
            "Rank":         f"#{i+1}",
            "Coin":         s["symbol"].replace("USDT", ""),
            "TF":           s["timeframe"],
            "Dir":          ("LONG" if s["direction"] == "long" else "SHORT"),
            "Score":        _sc,
            "Regime":       s["regime"],
            "Body%":        s["body_pct"],
            "Vol×":         s["vol_mult"],
            "ADX":          s["adx"],
            "Agg Entry":    _entry,
            "Std Entry":    _etp_s.get("std_entry", _entry),
            "Sniper Entry": _etp_s.get("sniper_entry", _entry),
            "SL%":          _etp_s.get("sl_dist_pct", 1.5),
            "TP2 (Std)":    _etp_s.get("std_tp2", s["tp2r"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        height=min(40 + len(all_signals_deduped) * 35, 750),
        column_config={
            "Score":        st.column_config.NumberColumn(width=60,  format="%.1f"),
            "Body%":        st.column_config.NumberColumn(width=65,  format="%.1f"),
            "Vol×":         st.column_config.NumberColumn(width=55,  format="%.2f"),
            "ADX":          st.column_config.NumberColumn(width=55,  format="%.1f"),
            "Agg Entry":    st.column_config.NumberColumn(width=95,  format="%.6g"),
            "Std Entry":    st.column_config.NumberColumn(width=95,  format="%.6g"),
            "Sniper Entry": st.column_config.NumberColumn(width=100, format="%.6g"),
            "SL%":          st.column_config.NumberColumn(width=60,  format="%.2f%%"),
            "TP2 (Std)":    st.column_config.NumberColumn(width=100, format="%.6g"),
        },
    )

    st.markdown("---")
    st.markdown("### 📋 Detailed Signal Cards — Point-by-Point Analysis")

    # Detailed cards
    for i, sig in enumerate(all_signals_deduped):
        dir_color   = "#64ffda" if sig["direction"] == "long"  else "#ff6b6b"
        dir_icon    = "📈"      if sig["direction"] == "long"  else "📉"
        reg_color   = _reg_color.get(sig["regime"], "#8b949e")
        ema_str     = "✅ Full" if sig["ema_full"] else ("⚠️ Partial" if sig["ema_partial"] else "❌ Not aligned")
        recency_map = {1: "🟢 Current candle (freshest)", 2: "🟡 1 candle ago", 3: "🟠 2 candles ago"}
        recency_str = recency_map.get(sig.get("bar_offset", 1), "")

        # Score bar (visual) — guard against None/NaN scores
        try:
            score_pct = min(int(sig.get("score") or 0), 100)
        except (TypeError, ValueError):
            score_pct = 0
        bar_filled = "█" * (score_pct // 5)
        bar_empty  = "░" * (20 - score_pct // 5)

        _score_display = score_pct  # already safe int from above
        header = (
            f"#{i+1} — {sig['symbol']} ({sig['timeframe']}) "
            f"| {dir_icon} {sig['direction'].upper()} "
            f"| Score {_score_display}/100 "
            f"| {sig['regime']}"
        )

        with st.expander(header, expanded=(i < 5)):
            col_l, col_r = st.columns([1.4, 1])

            with col_l:
                # Coin header
                coin_base = sig["symbol"].replace("USDT", "")
                st.markdown(
                    f'<div style="font-size:20px;font-weight:800;color:{dir_color};">'
                    f'{dir_icon} {coin_base}/USDT &nbsp;'
                    f'<span style="font-size:13px;color:#8892b0;font-weight:400;">'
                    f'{sig["timeframe"]} | Candle: {sig.get("candle_date","")}</span></div>',
                    unsafe_allow_html=True,
                )

                # ── Entry method explanation ──────────────────────────────────
                # The scanner uses 0% retracement (immediate entry at candle close) as
                # the aggressive baseline. The enhanced plan adds 2 better entry zones.
                _bar_off  = sig.get("bar_offset", 1)
                _etp      = sig.get("_trade_plan", {})
                _is_fresh = _bar_off == 1

                if _is_fresh:
                    _freshness_html = (
                        "<span style='color:#3fb950;font-weight:700;'>🟢 FRESH — candle just closed.</span> "
                        "All three entry zones are valid. Prefer Standard or Sniper for better R:R."
                    )
                else:
                    _freshness_html = (
                        f"<span style='color:#e3b341;font-weight:700;'>⚠️ Signal is {_bar_off-1} candle(s) old.</span> "
                        "Aggressive entry may already be missed. Use Standard or Sniper zone only, "
                        "or skip if price is >1R away."
                    )

                # ── Build the enhanced trade plan card ─────────────────────────
                if _etp:
                    _sl_pct   = _etp.get("sl_dist_pct", 1.5)
                    _atr_pct  = _etp.get("atr_pct", 0)
                    _dir      = sig["direction"]
                    _std_valid    = _etp.get("std_valid",    True)
                    _sniper_valid = _etp.get("sniper_valid", True)

                    def _fmt(v):
                        return f"{v:.6g}" if v else "—"

                    # Update freshness note if any zone is invalid
                    if not _std_valid or not _sniper_valid:
                        _invalid_names = []
                        if not _std_valid:    _invalid_names.append("Standard")
                        if not _sniper_valid: _invalid_names.append("Sniper")
                        _zone_warn = (
                            f" <span style='color:#ff6b6b;font-weight:700;'>⚠️ "
                            f"{' & '.join(_invalid_names)} zone(s) unavailable — "
                            f"candle body too large for SL distance.</span>"
                        )
                        _freshness_html += _zone_warn

                    # Aggressive zone (enter at close)
                    _agg_rr1  = abs(_etp['agg_tp1'] - _etp['agg_entry']) / max(abs(_etp['agg_entry'] - _etp['agg_sl']), 1e-10)
                    _std_rr2  = 2.0  # always 2R by construction
                    _snp_rr3  = 3.0

                    # ── Standard zone HTML ────────────────────────────────────
                    if _std_valid:
                        _std_zone_html = f"""
  <div style="background:#091a1a;border:1px solid #1a4a3a;border-radius:6px;padding:10px;">
    <div style="color:#3fb950;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
      ✅ Standard Entry (38.2%)</div>
    <div style="color:#aab;font-size:10px;margin-bottom:8px;">Wait for 38.2% retrace into candle body. Recommended default.</div>
    <div style="color:#8892b0;font-size:10px;">ENTRY</div>
    <div style="color:#ccd6f6;font-weight:700;font-size:13px;">{_fmt(_etp['std_entry'])}</div>
    <div style="color:#8892b0;font-size:10px;margin-top:5px;">STOP LOSS</div>
    <div style="color:#ff6b6b;font-weight:700;font-size:13px;">{_fmt(_etp['std_sl'])}</div>
    <div style="color:#8892b0;font-size:10px;margin-top:5px;">TP1 / TP2 / TP3</div>
    <div style="color:#64ffda;font-size:12px;">{_fmt(_etp['std_tp1'])} / {_fmt(_etp['std_tp2'])} / {_fmt(_etp['std_tp3'])}</div>
  </div>"""
                    else:
                        _sl_pct_used = _etp.get("sl_dist_pct", 0)
                        _std_zone_html = f"""
  <div style="background:#1a0a0a;border:2px solid #6b2222;border-radius:6px;padding:10px;opacity:0.75;">
    <div style="color:#ff6b6b;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
      ❌ Standard Entry — UNAVAILABLE</div>
    <div style="color:#cc8888;font-size:11px;line-height:1.4;">
      Candle body is too large relative to the structural SL distance
      ({_sl_pct_used:.1f}%). The 38.2% retrace zone falls at or beyond the
      stop-loss level — entering here would mean your SL is already hit.
      <br><br><strong style="color:#ffaa88;">Use Aggressive zone only.</strong>
    </div>
  </div>"""

                    # ── Sniper zone HTML ──────────────────────────────────────
                    if _sniper_valid:
                        _sniper_zone_html = f"""
  <div style="background:#14100a;border:1px solid #4a3a1a;border-radius:6px;padding:10px;">
    <div style="color:#e3b341;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
      🎯 Sniper Entry (61.8%)</div>
    <div style="color:#aab;font-size:10px;margin-bottom:8px;">Wait for 61.8% fib retrace. Best R:R, lower fill probability.</div>
    <div style="color:#8892b0;font-size:10px;">ENTRY</div>
    <div style="color:#ccd6f6;font-weight:700;font-size:13px;">{_fmt(_etp['sniper_entry'])}</div>
    <div style="color:#8892b0;font-size:10px;margin-top:5px;">STOP LOSS</div>
    <div style="color:#ff6b6b;font-weight:700;font-size:13px;">{_fmt(_etp['sniper_sl'])}</div>
    <div style="color:#8892b0;font-size:10px;margin-top:5px;">TP1 / TP2 / TP3</div>
    <div style="color:#64ffda;font-size:12px;">{_fmt(_etp['sniper_tp1'])} / {_fmt(_etp['sniper_tp2'])} / {_fmt(_etp['sniper_tp3'])}</div>
  </div>"""
                    else:
                        _sl_pct_used = _etp.get("sl_dist_pct", 0)
                        _sniper_zone_html = f"""
  <div style="background:#1a0a0a;border:2px solid #6b2222;border-radius:6px;padding:10px;opacity:0.75;">
    <div style="color:#ff6b6b;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
      ❌ Sniper Entry — UNAVAILABLE</div>
    <div style="color:#cc8888;font-size:11px;line-height:1.4;">
      Candle body is too large relative to the structural SL distance
      ({_sl_pct_used:.1f}%). The 61.8% retrace zone falls at or beyond the
      stop-loss level — entering here would mean your SL is already hit.
      <br><br><strong style="color:#ffaa88;">Use Aggressive zone only.</strong>
    </div>
  </div>"""

                    _zone_rows = f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:10px 0;">

  <div style="background:#0a1628;border:1px solid #1f3a5f;border-radius:6px;padding:10px;">
    <div style="color:#8892b0;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
      ⚡ Aggressive Entry</div>
    <div style="color:#aab;font-size:10px;margin-bottom:8px;">Enter at candle close. Highest fill chance, lowest R:R.</div>
    <div style="color:#8892b0;font-size:10px;">ENTRY</div>
    <div style="color:#ccd6f6;font-weight:700;font-size:13px;">{_fmt(_etp['agg_entry'])}</div>
    <div style="color:#8892b0;font-size:10px;margin-top:5px;">STOP LOSS</div>
    <div style="color:#ff6b6b;font-weight:700;font-size:13px;">{_fmt(_etp['agg_sl'])}</div>
    <div style="color:#8892b0;font-size:10px;margin-top:5px;">TP1 / TP2 / TP3</div>
    <div style="color:#64ffda;font-size:12px;">{_fmt(_etp['agg_tp1'])} / {_fmt(_etp['agg_tp2'])} / {_fmt(_etp['agg_tp3'])}</div>
  </div>

  {_std_zone_html}

  {_sniper_zone_html}

</div>"""

                    _mgmt_html = f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:10px 14px;margin-top:8px;">
  <div style="color:#58a6ff;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:8px;">
    📋 Trade Management Plan</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">
    <div>
      <div style="color:#8892b0;">SL Method</div>
      <div style="color:#ccd6f6;">ATR-adaptive — {_sl_pct:.1f}% (ATR = {_atr_pct:.1f}%)</div>
    </div>
    <div>
      <div style="color:#8892b0;">Invalidation Anchor</div>
      <div style="color:#ccd6f6;">{'Below candle low' if _dir=='long' else 'Above candle high'} + 0.5× ATR buffer</div>
    </div>
    <div style="margin-top:6px;">
      <div style="color:#8892b0;">At TP1</div>
      <div style="color:#ccd6f6;">Close 30–50% of position → move SL to breakeven</div>
    </div>
    <div style="margin-top:6px;">
      <div style="color:#8892b0;">At TP2</div>
      <div style="color:#ccd6f6;">Close another 30% → trail SL below last swing</div>
    </div>
    <div style="margin-top:6px;">
      <div style="color:#8892b0;">At TP3 / Let Run</div>
      <div style="color:#ccd6f6;">Hold remaining 20–40% with trailing SL for extended move</div>
    </div>
    <div style="margin-top:6px;">
      <div style="color:#8892b0;">Skip Signal If</div>
      <div style="color:#ccd6f6;">Price already >1R from aggressive entry without a retrace</div>
    </div>
  </div>
</div>"""

                    st.markdown(
                        f'<div style="background:#0d1f2d;border:1px solid #1f6feb;'
                        f'border-radius:8px;padding:12px 16px;margin:8px 0;font-size:13px;">'
                        f'<div style="color:#58a6ff;font-weight:700;font-size:14px;margin-bottom:6px;">🎯 Enhanced Trade Plan</div>'
                        f'<div style="font-size:12px;line-height:1.5;margin-bottom:4px;">{_freshness_html}</div>'
                        f'{_zone_rows}'
                        f'{_mgmt_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Fallback to old simple display if _trade_plan missing
                    st.markdown(
                        f'<div style="background:#0d1f2d;border:1px solid #1f6feb;'
                        f'border-radius:6px;padding:10px 14px;margin:8px 0;font-size:13px;">'
                        f'<div style="color:#58a6ff;font-weight:700;margin-bottom:6px;">🎯 Trade Setup</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">'
                        f'<div><div style="color:#8892b0;font-size:11px;">ENTRY</div>'
                        f'<div style="color:#ccd6f6;font-weight:700;">{sig["entry"]:.6g}</div></div>'
                        f'<div><div style="color:#8892b0;font-size:11px;">STOP LOSS</div>'
                        f'<div style="color:#ff6b6b;font-weight:700;">{sig["sl"]:.6g}</div></div>'
                        f'<div><div style="color:#8892b0;font-size:11px;">TAKE PROFIT (2R)</div>'
                        f'<div style="color:#64ffda;font-weight:700;">{sig["tp2r"]:.6g}</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                # Signal recency
                st.markdown(
                    f'<div style="color:#8892b0;font-size:12px;margin-bottom:8px;">'
                    f'{recency_str}</div>',
                    unsafe_allow_html=True,
                )

                # Reasons
                st.markdown(
                    '<div style="color:#58a6ff;font-size:13px;font-weight:700;'
                    'margin-bottom:6px;">Why this coin was selected:</div>',
                    unsafe_allow_html=True,
                )
                for reason in sig["reasons"]:
                    st.markdown(
                        f'<div style="color:#ccd6f6;font-size:13px;padding:3px 0;'
                        f'border-bottom:1px solid #21262d;">'
                        f'▸ {reason}</div>',
                        unsafe_allow_html=True,
                    )

            with col_r:
                # Score breakdown card — use safe score_pct already computed above
                score_color = (
                    "#3fb950" if score_pct >= 70 else
                    "#e3b341" if score_pct >= 50 else
                    "#f85149"
                )
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid {score_color};'
                    f'border-radius:8px;padding:14px 16px;">'

                    f'<div style="text-align:center;margin-bottom:12px;">'
                    f'<div style="color:#8892b0;font-size:11px;text-transform:uppercase;'
                    f'letter-spacing:1px;">Signal Score</div>'
                    f'<div style="color:{score_color};font-size:32px;font-weight:800;">'
                    f'{score_pct}<span style="font-size:16px;color:#8892b0;">/100</span></div>'
                    f'<div style="font-family:monospace;font-size:11px;color:{score_color};">'
                    f'{bar_filled}<span style="color:#3a3f4b;">{bar_empty}</span></div>'
                    f'</div>'

                    f'<div style="border-top:1px solid #21262d;padding-top:10px;">'

                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">Body %</span>'
                    f'<span style="color:#ccd6f6;font-size:12px;font-weight:600;">'
                    f'{sig["body_pct"]:.1f}%</span></div>'

                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">Volume ×</span>'
                    f'<span style="color:#ccd6f6;font-size:12px;font-weight:600;">'
                    f'{sig["vol_mult"]:.2f}×</span></div>'

                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">ADX</span>'
                    f'<span style="color:#ccd6f6;font-size:12px;font-weight:600;">'
                    f'{sig["adx"]:.0f}</span></div>'

                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">DI+ / DI−</span>'
                    f'<span style="color:#ccd6f6;font-size:12px;font-weight:600;">'
                    f'{sig["di_plus"]:.0f} / {sig["di_minus"]:.0f}</span></div>'

                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">ATR Ratio</span>'
                    f'<span style="color:#ccd6f6;font-size:12px;font-weight:600;">'
                    f'{sig["atr_ratio"]:.2f}×</span></div>'

                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">EMA Stack</span>'
                    f'<span style="color:#ccd6f6;font-size:12px;font-weight:600;">'
                    f'{ema_str}</span></div>'

                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">Candle Rank</span>'
                    f'<span style="color:#ccd6f6;font-size:12px;font-weight:600;">'
                    f'Top {(1-sig["candle_rank"])*100:.0f}%</span></div>'

                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 0 0 0;border-top:1px solid #21262d;margin-top:4px;">'
                    f'<span style="color:#8892b0;font-size:12px;">Regime</span>'
                    f'<span style="color:{reg_color};font-size:12px;font-weight:700;">'
                    f'{sig["regime"]} ({sig["regime_score"]}/100)</span></div>'

                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # ── OI + Funding Rate + Taker Buy block (fetched once per symbol, cached)
                _is_perp = sig["symbol"].upper().endswith("USDT")
                if _is_perp:
                    _deriv_cache_key = f"deriv_{sig['symbol']}"
                    if _deriv_cache_key not in st.session_state:
                        try:
                            _fr = fetch_funding_rate(sig["symbol"])
                            _oi = fetch_open_interest(sig["symbol"])
                        except Exception:
                            _fr = {"rate": 0.0, "ok": False, "source": "error"}
                            _oi = {"oi_change_pct": 0.0, "ok": False, "source": "error"}
                        st.session_state[_deriv_cache_key] = {"fr": _fr, "oi": _oi}
                    _deriv = st.session_state[_deriv_cache_key]
                    _af_fr  = _deriv["fr"]
                    _af_oi  = _deriv["oi"]
                    _data_source = _af_oi.get("source") or _af_fr.get("source") or "none"

                    _deriv_ok = _af_fr.get("ok") or _af_oi.get("ok")
                    if _deriv_ok:
                        _badge_html_parts = []

                        # ── OI 24h Change badge ──
                        _oi_chg_val = _af_oi.get("oi_change_pct", 0) if _af_oi.get("ok") else None
                        # Store for AI prompt
                        sig["oi_change_pct"] = _oi_chg_val
                        if _oi_chg_val is not None:
                            if _oi_chg_val >= 10:
                                _oi_badge_col, _oi_badge_lbl = "#3fb950", "Strong inflow — new positions opening"
                            elif _oi_chg_val >= 3:
                                _oi_badge_col, _oi_badge_lbl = "#7ee787", "Rising — new money entering"
                            elif _oi_chg_val >= -3:
                                _oi_badge_col, _oi_badge_lbl = "#8892b0", "Neutral — no clear positioning shift"
                            elif _oi_chg_val >= -10:
                                _oi_badge_col, _oi_badge_lbl = "#e3b341", "Falling — position unwinding"
                            else:
                                _oi_badge_col, _oi_badge_lbl = "#f85149", "Heavy unwind — possible squeeze or exit"
                            _oi_arrow = "▲" if _oi_chg_val >= 0 else "▼"
                            _badge_html_parts.append(
                                f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
                                f'<span style="color:#8892b0;font-size:12px;">OI 24h Δ</span>'
                                f'<span style="color:{_oi_badge_col};font-size:12px;font-weight:600;">'
                                f'{_oi_arrow} {abs(_oi_chg_val):.1f}% — {_oi_badge_lbl}</span></div>'
                            )

                        # ── Funding Rate badge ──
                        _fr_rate_val = _af_fr.get("rate", 0) if _af_fr.get("ok") else None
                        sig["funding_rate"] = _fr_rate_val
                        if _fr_rate_val is not None:
                            _fr_pct = _fr_rate_val * 100  # e.g. 0.0001 → 0.01%
                            if _fr_pct > 0.05:
                                _fr_badge_col, _fr_badge_lbl = "#f85149", "Crowded LONG — longs paying heavily, squeeze risk"
                            elif _fr_pct >= 0.01:
                                _fr_badge_col, _fr_badge_lbl = "#e3b341", "Longs paying shorts — mild crowding"
                            elif _fr_pct >= -0.01:
                                _fr_badge_col, _fr_badge_lbl = "#8892b0", "Neutral — balanced positioning"
                            elif _fr_pct >= -0.05:
                                _fr_badge_col, _fr_badge_lbl = "#7ee787", "Shorts paying longs — long tailwind"
                            else:
                                _fr_badge_col, _fr_badge_lbl = "#3fb950", "Heavily negative — strong long tailwind"
                            _badge_html_parts.append(
                                f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
                                f'<span style="color:#8892b0;font-size:12px;">Funding Rate</span>'
                                f'<span style="color:{_fr_badge_col};font-size:12px;font-weight:600;">'
                                f'{_fr_pct:.4f}% — {_fr_badge_lbl}</span></div>'
                            )

                        # ── Taker Buy Ratio badge ──
                        _tbr_val = sig.get("taker_buy_ratio", 0.5)
                        _tbr_real = _tbr_val != 0.5  # suppress display if default
                        if _tbr_real:
                            _tbr_pct = _tbr_val * 100
                            if _tbr_pct >= 65:
                                _tbr_badge_col, _tbr_badge_lbl = "#3fb950", "Buy-side dominant — strong aggressive buying"
                            elif _tbr_pct >= 55:
                                _tbr_badge_col, _tbr_badge_lbl = "#7ee787", "Buy-side lean — buyers in control"
                            elif _tbr_pct >= 45:
                                _tbr_badge_col, _tbr_badge_lbl = "#8892b0", "Balanced — no clear aggressor"
                            elif _tbr_pct >= 35:
                                _tbr_badge_col, _tbr_badge_lbl = "#e3b341", "Sell-side lean — sellers in control"
                            else:
                                _tbr_badge_col, _tbr_badge_lbl = "#f85149", "Sell-side dominant — aggressive selling"
                            _badge_html_parts.append(
                                f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
                                f'<span style="color:#8892b0;font-size:12px;">Taker Buy Ratio</span>'
                                f'<span style="color:{_tbr_badge_col};font-size:12px;font-weight:600;">'
                                f'{_tbr_pct:.1f}% — {_tbr_badge_lbl}</span></div>'
                            )

                        # ── Combination reading ──
                        _combo_html = ""
                        _oi_rising = _oi_chg_val is not None and _oi_chg_val >= 3
                        _oi_falling = _oi_chg_val is not None and _oi_chg_val < -3
                        _tbr_buy = _tbr_val >= 0.55
                        _tbr_sell = _tbr_val < 0.45
                        _fr_crowded = _fr_rate_val is not None and _fr_rate_val * 100 > 0.03
                        _fr_neutral_neg = _fr_rate_val is None or _fr_rate_val * 100 <= 0.03

                        if _oi_rising and _tbr_buy and _fr_neutral_neg and not _fr_crowded:
                            _combo_col, _combo_txt = "#3fb950", "✅ Organic momentum — new money + buyer aggression, not crowded"
                        elif _oi_rising and _tbr_buy and _fr_crowded:
                            _combo_col, _combo_txt = "#e3b341", "⚠️ Momentum but crowded — strong move, longs already heavy"
                        elif _oi_falling and _tbr_sell:
                            _combo_col, _combo_txt = "#f85149", "❌ Unwinding — positions closing, sellers aggressive"
                        elif _oi_rising and _tbr_sell:
                            _combo_col, _combo_txt = "#e3b341", "⚠️ OI rising but sellers dominant — possible short buildup"
                        elif _oi_falling and _tbr_buy:
                            _combo_col, _combo_txt = "#e3b341", "⚠️ Buyers aggressive but OI falling — short covering, not fresh longs"
                        else:
                            _combo_col, _combo_txt = "#8892b0", "➖ Mixed signals — use other confluence"

                        _combo_html = (
                            f'<div style="border-top:1px solid #21262d;margin-top:6px;padding-top:6px;">'
                            f'<span style="color:{_combo_col};font-size:11px;font-weight:600;">{_combo_txt}</span></div>'
                        )

                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid #2d3250;'
                            f'border-radius:8px;padding:12px 16px;margin-top:10px;">'
                            f'<div style="color:#8892b0;font-size:11px;text-transform:uppercase;'
                            f'letter-spacing:1px;margin-bottom:6px;">📊 Derivatives Sentiment</div>'
                            + "".join(_badge_html_parts)
                            + _combo_html
                            + f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="color:#3a3f4b;font-size:11px;padding:4px 0;">'
                            'Derivatives data unavailable</div>',
                            unsafe_allow_html=True,
                        )

            # ── Confluence Panel (full-width, below both columns) ────────────
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

            _sym_key       = f"{sig['symbol']}_{sig['timeframe']}_{sig['direction']}"
            _bt_cache_key  = f"bt_{_sym_key}"
            _ml_cache_key  = f"ml_{_sym_key}"            # legacy — primary/display ML
            _ml_a_key      = f"mlA_{_sym_key}"           # Candidate A (newest bucket)
            _ml_b_key      = f"mlB_{_sym_key}"           # Candidate B (weighted all-time)
            _ml_primary    = f"ml_primary_{_sym_key}"    # "A" or "B" — which ML the UI/AI uses
            _wfo_cache_key = f"wfo_{_sym_key}"
            _ai_key        = f"ai_result_{_sym_key}"
            _has_ai_key    = bool(st.session_state.get("groq_api_key", ""))

            # ── Step 1: Backtest + WFO ───────────────────────────────────────
            if st.button("📊 Step 1 — Backtest + WFO  (deep historical scan)",
                         key=f"step1_{_sym_key}_{i}",
                         use_container_width=True,
                         help=("Deep fetch (up to 1000 bars) + multi-method backtest "
                               "with time-decay buckets + WFO mini-validation.")):
                with st.spinner("Deep backtest + WFO…"):
                    _bt  = _scanner_quick_backtest(sig)
                    _wfo = _scanner_mini_wfo(sig, _bt)
                st.session_state[_bt_cache_key]  = _bt
                st.session_state[_wfo_cache_key] = _wfo
                # Clear any previously cached ML so user re-trains on fresh backtest
                for _k in (_ml_cache_key, _ml_a_key, _ml_b_key, _ml_primary, _ai_key):
                    st.session_state.pop(_k, None)

            _bt_ready = _bt_cache_key in st.session_state

            # ── Step 2: Train ML (single button for both candidates) ─────────
            if _bt_ready:
                _bt_for_pick  = st.session_state[_bt_cache_key]
                _cand_a_dict  = _bt_for_pick.get("candidate_newest")
                _cand_b_dict  = _bt_for_pick.get("candidate_weighted")

                def _cand_label(c):
                    if not c:
                        return "— n/a —"
                    return (f"{c.get('zone','?')} / {c.get('sl_label','?')} / "
                            f"{c.get('mgmt','?')} / TP{c.get('tp_mult',2.0):.1f}R")

                # Detect if A and B are the same method
                def _cfg_tuple(c):
                    if not c:
                        return None
                    mc = c.get("method_cfg") or {}
                    return (mc.get("zone"), mc.get("sl_label"), mc.get("mgmt"),
                            round(float(mc.get("tp_mult", 2.0)), 2))

                _a_cfg = _cfg_tuple(_cand_a_dict)
                _b_cfg = _cfg_tuple(_cand_b_dict)
                _ab_same = (_a_cfg is not None and _a_cfg == _b_cfg)

                # Intro panel
                _intro_note = (
                    "Candidate A &amp; B resolved to the <b>same method</b> — ML will be trained once."
                    if _ab_same else
                    "Train adaptive ML (LR/RF/GB auto-picked by sample size) on both candidates in one click. "
                    "Each candidate is labeled by its own method outcomes."
                )
                st.markdown(
                    f'<div style="margin-top:10px;padding:8px 12px;background:#0d1117;'
                    f'border:1px solid #30363d;border-radius:6px;">'
                    f'<div style="color:#58a6ff;font-size:11px;text-transform:uppercase;'
                    f'letter-spacing:1px;font-weight:700;margin-bottom:4px;">'
                    f'🧠 Step 2 — Train ML for Both Candidates</div>'
                    f'<div style="color:#8892b0;font-size:11px;">{_intro_note}</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">'
                    f'<div style="background:#0d1f0d;border:1px solid #238636;border-radius:4px;padding:6px 8px;">'
                    f'<div style="color:#3fb950;font-size:9px;font-weight:700;text-transform:uppercase;">'
                    f'🟢 Candidate A {"(= B)" if _ab_same else ""}</div>'
                    f'<div style="color:#ccd6f6;font-size:10px;font-family:monospace;margin-top:2px;">'
                    f'{_cand_label(_cand_a_dict)}</div></div>'
                    + (f'<div style="background:#0a1628;border:1px solid #1f6feb;border-radius:4px;padding:6px 8px;">'
                       f'<div style="color:#58a6ff;font-size:9px;font-weight:700;text-transform:uppercase;">🔵 Candidate B</div>'
                       f'<div style="color:#ccd6f6;font-size:10px;font-family:monospace;margin-top:2px;">'
                       f'{_cand_label(_cand_b_dict)}</div></div>'
                       if not _ab_same else
                       f'<div style="background:#1a1500;border:1px solid #e3b341;border-radius:4px;padding:6px 8px;opacity:0.7;">'
                       f'<div style="color:#e3b341;font-size:9px;font-weight:700;text-transform:uppercase;">'
                       f'🔵 Candidate B — Same as A</div>'
                       f'<div style="color:#8892b0;font-size:10px;margin-top:2px;">Unanimous — single training</div></div>')
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )

                _ml_btn_disabled = (_cand_a_dict is None and _cand_b_dict is None)
                _ml_btn_label = ("🧠 Step 2 — Train ML (Unanimous)"
                                 if _ab_same else
                                 "🧠 Step 2 — Train ML for Both Candidates")
                if st.button(_ml_btn_label,
                             key=f"ml_both_btn_{_sym_key}_{i}",
                             use_container_width=True,
                             disabled=_ml_btn_disabled):
                    if _ab_same and _cand_a_dict:
                        with st.spinner("Training ML (unanimous method)…"):
                            _ml_shared = _scanner_train_ml(sig, _cand_a_dict["method_cfg"])
                        st.session_state[_ml_a_key] = _ml_shared
                        st.session_state[_ml_b_key] = _ml_shared
                        st.session_state[_ml_cache_key] = _ml_shared
                    else:
                        with st.spinner("Training ML on Candidate A…"):
                            if _cand_a_dict:
                                _ml_a_new = _scanner_train_ml(sig, _cand_a_dict["method_cfg"])
                                st.session_state[_ml_a_key] = _ml_a_new
                        with st.spinner("Training ML on Candidate B…"):
                            if _cand_b_dict:
                                _ml_b_new = _scanner_train_ml(sig, _cand_b_dict["method_cfg"])
                                st.session_state[_ml_b_key] = _ml_b_new
                        # Primary display ML = A by default (can be changed)
                        st.session_state[_ml_cache_key] = st.session_state.get(
                            _ml_a_key, st.session_state.get(_ml_b_key)
                        )
                    st.session_state[_ml_primary] = "A"
                    st.session_state.pop(_ai_key, None)

            # ── Step 3: AI Final Verdict (dual-candidate analysis) ───────────
            _ml_ready = (_ml_a_key in st.session_state) or (_ml_b_key in st.session_state)
            _ai_disabled = not _has_ai_key or not (_bt_ready and _ml_ready)
            _ai_tip = (
                "Run Step 1 + Step 2 (train ML) first."
                if not (_bt_ready and _ml_ready) else
                "Ask Groq (gpt-oss-120b) to analyze both candidates and pick the winner."
                if _has_ai_key else
                "Add Groq API key in sidebar to enable."
            )
            if st.button("🤖 Step 3 — AI Dual-Candidate Analysis",
                         key=f"step3_{_sym_key}_{i}",
                         use_container_width=True,
                         type="primary",
                         disabled=_ai_disabled,
                         help=_ai_tip):
                with st.spinner("AI analyzing both candidates (may take 20-40s)…"):
                    _bt_for_ai = st.session_state.get(_bt_cache_key, {}) or {}
                    _ai_res = _scanner_ai_verdict(
                        sig,
                        ml_a   = st.session_state.get(_ml_a_key),
                        ml_b   = st.session_state.get(_ml_b_key),
                        bt     = _bt_for_ai,
                        wfo    = st.session_state.get(_wfo_cache_key),
                        cand_a = _bt_for_ai.get("candidate_newest"),
                        cand_b = _bt_for_ai.get("candidate_weighted"),
                    )
                st.session_state[_ai_key] = _ai_res

            _bt_res  = st.session_state.get(_bt_cache_key)
            _ml_res  = st.session_state.get(_ml_cache_key)
            _wfo_res = st.session_state.get(_wfo_cache_key)
            _ai_res  = st.session_state.get(_ai_key)

            if _bt_res or _ml_res:
                _ml_res = _ml_res or _scanner_heuristic_ml(sig)
                _bt_res = _bt_res or {}
                _grade, _grade_color, _grade_desc = _scanner_setup_grade(sig, _ml_res, _bt_res)

                # ── WFO Results Block ──────────────────────────────────────────
                _wfo_block_html = ""
                if _wfo_res:
                    _wv      = _wfo_res.get("verdict", "INSUFFICIENT")
                    _wv_col  = {"PASS": "#3fb950", "BORDERLINE": "#e3b341",
                                "FAIL": "#f85149", "INSUFFICIENT": "#8892b0"}.get(_wv, "#8892b0")
                    _wv_bg   = {"PASS": "#091a0d", "BORDERLINE": "#1a1500",
                                "FAIL": "#1a0505", "INSUFFICIENT": "#0d1117"}.get(_wv, "#0d1117")
                    _wv_icon = {"PASS": "✅", "BORDERLINE": "⚠️",
                                "FAIL": "❌", "INSUFFICIENT": "⚠️"}.get(_wv, "—")
                    _wfo_ran   = _wfo_res.get("ok", False)
                    _wfo_note  = _wfo_res.get("note", "")
                    _wfo_meth  = _wfo_res.get("method_used", "—") or "—"

                    if _wvo_ran := _wfo_ran and _wv != "INSUFFICIENT":
                        # Full result card with metric grid
                        _wfo_block_html = (
                            f'<div style="margin-top:10px;background:{_wv_bg};'
                            f'border:1px solid {_wv_col};border-radius:8px;padding:10px 14px;">'
                            f'<div style="color:{_wv_col};font-size:11px;text-transform:uppercase;'
                            f'letter-spacing:1px;font-weight:700;margin-bottom:6px;">'
                            f'🔬 WFO Mini-Validation — {_wv_icon} {_wv}</div>'
                            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;margin-bottom:6px;">'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">IS PF</div>'
                            f'<div style="color:#ccd6f6;font-size:14px;font-weight:800;">'+("∞" if _wfo_res.get("is_pf",0)>=9.9 else f"{_wfo_res.get('is_pf',0):.2f}")+'</div>'
                            f'<div style="color:#8892b0;font-size:9px;">n={_wfo_res.get("is_n",0)}</div></div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">OOS PF</div>'
                            f'<div style="color:{_wv_col};font-size:14px;font-weight:800;">'+("∞" if _wfo_res.get("oos_pf",0)>=9.9 else f"{_wfo_res.get('oos_pf',0):.2f}")+'</div>'
                            f'<div style="color:#8892b0;font-size:9px;">n={_wfo_res.get("oos_n",0)}</div></div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">OOS WR</div>'
                            f'<div style="color:#ccd6f6;font-size:14px;font-weight:800;">{_wfo_res.get("oos_wr",0):.1f}%</div></div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">OOS/IS Ratio</div>'
                            f'<div style="color:#ccd6f6;font-size:14px;font-weight:800;">{_wfo_res.get("oos_is_ratio",0):.2f}</div></div>'
                            f'</div>'
                            f'<div style="color:#8892b0;font-size:10px;">'
                            f'Method: {_wfo_meth} &nbsp;|&nbsp; {_wfo_res.get("tier_label","70% IS / 30% OOS")}</div>'
                            f'<div style="color:{_wv_col};font-size:11px;margin-top:4px;">{_wfo_note}</div>'
                            f'</div>'
                        )
                    else:
                        # INSUFFICIENT or failed-to-start — show simple explanatory card
                        _ins_is_n = _wfo_res.get("is_n", 0)
                        _ins_desc = (
                            f"IS: {_ins_is_n} trades, OOS: {_wfo_res.get('oos_n',0)} trades"
                            if _wfo_ran else ""
                        )
                        _wfo_block_html = (
                            f'<div style="margin-top:10px;background:#0d1117;'
                            f'border:1px solid #8892b0;border-radius:8px;padding:10px 14px;">'
                            f'<div style="color:#8892b0;font-size:11px;text-transform:uppercase;'
                            f'letter-spacing:1px;font-weight:700;margin-bottom:4px;">'
                            f'🔬 WFO Mini-Validation — ⚠️ INSUFFICIENT SAMPLE</div>'
                            f'<div style="color:#ccd6f6;font-size:12px;margin-bottom:4px;">'
                            f'Method tested: <b>{_wfo_meth}</b>'
                            + (f' &nbsp;|&nbsp; {_ins_desc}' if _ins_desc else '')
                            + f'</div>'
                            f'<div style="color:#e3b341;font-size:11px;">{_wfo_note}</div>'
                            f'<div style="color:#8892b0;font-size:10px;margin-top:4px;">'
                            f'WFO result ignored — signal may still be considered based on backtest and ML alone.</div>'
                            f'</div>'
                        )

                # ── 6 Intelligence Layers Expander ────────────────────────────
                # ── Layer 2: Macro Context ────────────────────────────────────
                # Read from session state first (already fetched by live scanner /
                # main analysis tab). Fall back to fresh cached fetch (alternative.me
                # for F&G, CoinGecko for BTC.D — both free, no API key needed).
                _l2_fg_data   = (st.session_state.get("live_fg_data")
                                 or st.session_state.get("_regime_fg_cache"))
                if not _l2_fg_data or not _l2_fg_data.get("ok"):
                    _l2_fg_data = fetch_fear_greed()
                    if _l2_fg_data.get("ok"):
                        st.session_state["_regime_fg_cache"] = _l2_fg_data

                _l2_btcd_data = st.session_state.get("_regime_btcd_cache")
                if not _l2_btcd_data or not _l2_btcd_data.get("ok"):
                    _l2_btcd_data = fetch_btc_dominance()
                    if _l2_btcd_data.get("ok"):
                        st.session_state["_regime_btcd_cache"] = _l2_btcd_data

                _l2_fng_val  = _l2_fg_data.get("value") if _l2_fg_data and _l2_fg_data.get("ok") else None
                _l2_fng_lbl  = _l2_fg_data.get("classification", "") if _l2_fg_data else ""
                _l2_btcd_val = _l2_btcd_data.get("btc_d") if _l2_btcd_data and _l2_btcd_data.get("ok") else None

                _layer2_btcd = (f"BTC.D: {_l2_btcd_val:.1f}%" if _l2_btcd_val is not None
                                else "BTC.D: N/A")
                _layer2_fng  = (f"F&G: {_l2_fng_val} ({_l2_fng_lbl})" if _l2_fng_val is not None
                                else "F&G: N/A")
                _layer2 = f"{_layer2_btcd} | {_layer2_fng}"

                # ── Layer 3: Derivatives Sentiment ────────────────────────────
                # OI / Funding are set on sig{} by the derivatives display block
                # that ran earlier in this same render cycle (above the columns).
                # Also check session-state cache as a fallback.
                _l3_cache_key = f"deriv_{sig['symbol']}"
                _l3_cached    = st.session_state.get(_l3_cache_key, {})
                _l3_oi_val    = (sig.get("oi_change_pct")
                                 if sig.get("oi_change_pct") is not None
                                 else (_l3_cached.get("oi", {}).get("oi_change_pct")
                                       if _l3_cached.get("oi", {}).get("ok") else None))
                _l3_fr_val    = (sig.get("funding_rate")
                                 if sig.get("funding_rate") is not None
                                 else (_l3_cached.get("fr", {}).get("rate")
                                       if _l3_cached.get("fr", {}).get("ok") else None))
                _l3_tbr_val   = sig.get("taker_buy_ratio", 0.5)
                _l3_tbr_real  = abs(_l3_tbr_val - 0.5) > 0.001   # False if still at default

                _layer3_oi  = (f"OI 24h: {_l3_oi_val:+.1f}%" if _l3_oi_val is not None
                               else "OI 24h: N/A (spot-only or derivatives API unavailable)")
                _layer3_fr  = (f"Funding: {_l3_fr_val*100:.4f}%" if _l3_fr_val is not None
                               else "Funding: N/A")
                _layer3_tbr = (f"Taker Buy: {_l3_tbr_val*100:.1f}%" if _l3_tbr_real
                               else "Taker Buy: N/A")
                _layer3 = f"{_layer3_oi} | {_layer3_fr} | {_layer3_tbr}"

                # ── Layers 1, 4, 5 ────────────────────────────────────────────
                _layer1 = (
                    f"Body {sig['body_pct']:.1f}% | Vol {sig['vol_mult']:.2f}× | "
                    f"ADX {sig['adx']:.1f} | DI+ {sig['di_plus']:.1f} vs DI− {sig['di_minus']:.1f} | "
                    f"ATR× {sig['atr_ratio']:.2f} | Candle Rank top {round((1-sig.get('candle_rank',0.5))*100):.0f}% | "
                    f"Regime {sig['regime']} ({sig['regime_score']}/100) | Age: {sig.get('bar_offset',1)} candle(s)"
                )
                _ml_res_disp = _ml_res or _scanner_heuristic_ml(sig)
                # Layer 4 — ML engine: show method name + CV accuracy + sample count
                _ml_pct_d   = _ml_res_disp.get('pct', 50)
                _ml_lbl_d   = _ml_res_disp.get('label', '—')
                _ml_mname_d = _ml_res_disp.get('method_name', 'Heuristic')
                _ml_ns_d    = _ml_res_disp.get('n_samples', 0)
                _ml_cv_d    = _ml_res_disp.get('cv_accuracy')
                _ml_trained = _ml_res_disp.get('trained', False)
                if _ml_trained:
                    _cv_str = f"CV={_ml_cv_d*100:.1f}%" if _ml_cv_d is not None else "CV=n/a"
                    _layer4 = (f"ML Probability: {_ml_pct_d:.1f}% ({_ml_lbl_d}) | "
                               f"Model: {_ml_mname_d} | n={_ml_ns_d} "
                               f"({_ml_res_disp.get('n_wins',0)}W/{_ml_res_disp.get('n_losses',0)}L) | "
                               f"{_cv_str}")
                else:
                    _layer4 = (f"ML Probability: {_ml_pct_d:.1f}% ({_ml_lbl_d}) | "
                               f"{_ml_mname_d} — train a model in Step 2 for real ML")

                # Layer 5 — Backtest: show best method + WR + EV + PF + bars used
                _best_disp = _bt_res.get("best", {})
                _bk_disp   = _bt_res.get("best_key", "—") or "—"
                _meta_d    = _bt_res.get("meta", {}) or {}
                _bars_used = _meta_d.get("bars_used", 0)
                _bkt_cnt   = _meta_d.get("bucket_count", 1)
                if _bk_disp != "—":
                    _pf_d = _best_disp.get('pf', 0)
                    _pf_str = "∞" if _pf_d >= 9.9 else f"{_pf_d:.2f}"
                    _layer5 = (
                        f"Best: {_bk_disp} | "
                        f"WR={_best_disp.get('win_rate',0):.1f}% | "
                        f"EV={_best_disp.get('ev',0):+.2f}R | "
                        f"EVw={_best_disp.get('ev_weighted',0):+.2f}R | "
                        f"PF={_pf_str} | n={_best_disp.get('n',0)} | "
                        f"Bars={_bars_used} ({_bkt_cnt} decay buckets)"
                    )
                else:
                    _layer5 = f"Backtest: no valid method found (bars={_bars_used})"

                # ── Rich ML card (detailed, shown inline in the main card) ───
                # Builds a block showing method name, sample size, CV, top features,
                # and — if Candidate A & B are BOTH trained — a comparison strip.
                _ml_a_show = st.session_state.get(_ml_a_key)
                _ml_b_show = st.session_state.get(_ml_b_key)
                _ml_primary_show = st.session_state.get(_ml_primary, "A")

                def _render_ml_block(ml_dict, title, accent_color, bg_color):
                    if not ml_dict:
                        return ""
                    _trained = ml_dict.get("trained", False)
                    _mname   = ml_dict.get("method_name", "Heuristic")
                    _mcfg    = ml_dict.get("method_cfg") or {}
                    _pct     = ml_dict.get("pct", 50)
                    _lbl     = ml_dict.get("label", "—")
                    _ns      = ml_dict.get("n_samples", 0)
                    _nw      = ml_dict.get("n_wins",  0)
                    _nl      = ml_dict.get("n_losses", 0)
                    _cv      = ml_dict.get("cv_accuracy")
                    _cv_std  = ml_dict.get("cv_std")
                    _note    = ml_dict.get("note", "")
                    _fi      = ml_dict.get("feature_importance", [])

                    _mcfg_str = (
                        f"{_mcfg.get('zone','?')} / {_mcfg.get('sl_label','?')} / "
                        f"{_mcfg.get('mgmt','?')} / TP{_mcfg.get('tp_mult',2.0):.1f}R"
                    ) if _mcfg else "n/a"

                    _prob_color = ("#3fb950" if _pct >= 65 else
                                   "#e3b341" if _pct >= 50 else "#f85149")
                    _cv_color   = ("#3fb950" if (_cv or 0) >= 0.65 else
                                   "#e3b341" if (_cv or 0) >= 0.55 else "#f85149")
                    _cv_str = (
                        f"{_cv*100:.1f}% ± {(_cv_std or 0)*100:.1f}%"
                        if _cv is not None else "n/a"
                    )

                    # Top-3 feature importance bars
                    _fi_html = ""
                    if _fi:
                        _top = _fi[:3]
                        _max_imp = max((f["importance"] for f in _fi), default=1.0) or 1.0
                        for _f in _top:
                            _pct_bar = int((_f["importance"] / _max_imp) * 100)
                            _fi_html += (
                                f'<div style="display:grid;grid-template-columns:90px 1fr 50px;'
                                f'gap:6px;align-items:center;padding:2px 0;">'
                                f'<div style="color:#ccd6f6;font-size:10px;font-family:monospace;">{_f["feature"]}</div>'
                                f'<div style="background:#21262d;border-radius:3px;height:8px;overflow:hidden;">'
                                f'<div style="background:{accent_color};width:{_pct_bar}%;height:100%;"></div></div>'
                                f'<div style="color:#8892b0;font-size:10px;text-align:right;">{_f["importance"]:.2f}</div>'
                                f'</div>'
                            )
                        _fi_html = (
                            f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #21262d;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;'
                            f'letter-spacing:1px;margin-bottom:3px;">Top Feature Importance</div>'
                            + _fi_html + '</div>'
                        )

                    _status_badge = (
                        f'<span style="background:#0d2818;color:#3fb950;font-size:9px;'
                        f'padding:2px 6px;border-radius:3px;margin-left:6px;">✓ TRAINED</span>'
                        if _trained else
                        f'<span style="background:#2d2200;color:#e3b341;font-size:9px;'
                        f'padding:2px 6px;border-radius:3px;margin-left:6px;">⚠ HEURISTIC</span>'
                    )

                    _note_html = (
                        f'<div style="color:#8892b0;font-size:10px;margin-top:4px;font-style:italic;">{_note}</div>'
                        if _note else ""
                    )

                    return (
                        f'<div style="background:{bg_color};border:1px solid {accent_color};'
                        f'border-radius:6px;padding:8px 10px;margin-top:6px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'margin-bottom:4px;">'
                        f'<div style="color:{accent_color};font-size:10px;font-weight:700;'
                        f'text-transform:uppercase;letter-spacing:1px;">{title}{_status_badge}</div>'
                        f'<div style="color:{_prob_color};font-size:16px;font-weight:800;">{_pct:.1f}%</div>'
                        f'</div>'
                        f'<div style="color:#ccd6f6;font-size:11px;font-family:monospace;">{_mname}</div>'
                        f'<div style="color:#8892b0;font-size:10px;margin-top:2px;">Labeled by: {_mcfg_str}</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:6px;'
                        f'padding-top:6px;border-top:1px solid #21262d;">'
                        f'<div><div style="color:#8892b0;font-size:9px;">Samples</div>'
                        f'<div style="color:#ccd6f6;font-size:12px;font-weight:700;">{_ns} ({_nw}W/{_nl}L)</div></div>'
                        f'<div><div style="color:#8892b0;font-size:9px;">CV Accuracy</div>'
                        f'<div style="color:{_cv_color};font-size:12px;font-weight:700;">{_cv_str}</div></div>'
                        f'<div><div style="color:#8892b0;font-size:9px;">Verdict</div>'
                        f'<div style="color:{_prob_color};font-size:12px;font-weight:700;">{_lbl}</div></div>'
                        f'</div>'
                        + _note_html
                        + _fi_html
                        + f'</div>'
                    )

                _ml_card_html = ""
                if _ml_a_show or _ml_b_show:
                    # Detect if A and B are the same object (unanimous case)
                    _ml_unanimous_disp = (
                        _ml_a_show is _ml_b_show and _ml_a_show is not None
                    ) or (
                        _ml_a_show and _ml_b_show
                        and (_ml_a_show.get("method_cfg") or {}) == (_ml_b_show.get("method_cfg") or {})
                    )

                    _header = (
                        f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #21262d;">'
                        f'<div style="color:#58a6ff;font-size:11px;text-transform:uppercase;'
                        f'letter-spacing:1px;font-weight:700;margin-bottom:4px;">'
                        f'🧠 Trained ML — Adaptive Model'
                        f'{" (Unanimous: A ≡ B)" if _ml_unanimous_disp else ""}</div>'
                        f'</div>'
                    )
                    if _ml_unanimous_disp:
                        _ml_card_html = _header + _render_ml_block(
                            _ml_a_show or _ml_b_show,
                            "🟢 A ≡ B — Unanimous Method",
                            "#3fb950", "#091a0d",
                        )
                    else:
                        _a_block = _render_ml_block(
                            _ml_a_show,
                            "🟢 Candidate A — Best Newest-Bucket Method",
                            "#3fb950", "#091a0d",
                        )
                        _b_block = _render_ml_block(
                            _ml_b_show,
                            "🔵 Candidate B — Weighted All-Time Best",
                            "#58a6ff", "#0a1628",
                        )
                        _ml_card_html = _header + _a_block + _b_block

                # ── Layer 6: WFO ───────────────────────────────────────────────
                # Show WFO result whenever it ran (ok=True = ran, even if INSUFFICIENT).
                # ok=False = could not start at all.
                _wfo_l = _wfo_res or {}
                if _wfo_l.get("ok"):
                    _wfo_v = _wfo_l.get("verdict", "—")
                    if _wfo_v == "INSUFFICIENT":
                        _layer6 = (
                            f"WFO ran: {_wfo_v} | IS={_wfo_l.get('is_n',0)} trades | "
                            f"OOS={_wfo_l.get('oos_n',0)} trades — "
                            f"insufficient sample — result ignored | Method: {_wfo_l.get('method_used','—')}"
                        )
                    else:
                        _layer6 = (
                            f"WFO: {_wfo_v} | "
                            f"IS PF={'∞' if _wfo_l.get('is_pf',0)>=9.9 else f"{_wfo_l.get('is_pf',0):.2f}"} (n={_wfo_l.get('is_n',0)}) | "
                            f"OOS PF={'∞' if _wfo_l.get('oos_pf',0)>=9.9 else f"{_wfo_l.get('oos_pf',0):.2f}"} WR={_wfo_l.get('oos_wr',0):.1f}% "
                            f"(n={_wfo_l.get('oos_n',0)}) | Ratio={_wfo_l.get('oos_is_ratio',0):.2f}"
                        )
                elif _wfo_l.get("verdict") == "INSUFFICIENT":
                    # Ran but failed before simulation (no data / no method)
                    _layer6 = f"WFO: could not run — {_wfo_l.get('note', 'insufficient data')}"
                else:
                    _layer6 = "WFO: not yet run (click Step 1 first)"

                _intelligence_rows = [
                    ("1. Signal Raw Data",       "#58a6ff", _layer1),
                    ("2. Macro Context",          "#7ee787", _layer2),
                    ("3. Derivatives Sentiment",  "#e3b341", _layer3),
                    ("4. ML Engine",              "#64ffda", _layer4),
                    ("5. Backtest",               "#ccd6f6", _layer5),
                    ("6. WFO Validation",         "#f0883e", _layer6),
                ]
                _intel_rows_html = "".join(
                    f'<div style="display:grid;grid-template-columns:160px 1fr;gap:8px;'
                    f'padding:5px 0;border-bottom:1px solid #21262d;">'
                    f'<div style="color:{c};font-size:11px;font-weight:700;">{lbl}</div>'
                    f'<div style="color:#ccd6f6;font-size:11px;font-family:monospace;">{val}</div></div>'
                    for lbl, c, val in _intelligence_rows
                )
                _intel_expander_html = (
                    f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                    f'padding:12px 14px;margin-top:8px;">'
                    f'<div style="color:#8892b0;font-size:11px;text-transform:uppercase;'
                    f'letter-spacing:1px;font-weight:700;margin-bottom:8px;">🔭 6 Intelligence Layers</div>'
                    + _intel_rows_html
                    + f'</div>'
                )


                # Build backtest rows — enhanced multi-method comparison
                _bt_valid    = _bt_res.get("error") is None and _bt_res.get("n", 0) >= 3
                _zone_best   = _bt_res.get("zone_best", {})
                _best        = _bt_res.get("best", {})
                _best_key    = _bt_res.get("best_key", "")
                _per_method  = _bt_res.get("per_method", {})

                def _ev_color(ev):
                    return "#3fb950" if ev > 0.3 else "#e3b341" if ev > 0 else "#f85149"
                def _wr_color(wr):
                    return "#3fb950" if wr >= 55 else "#e3b341" if wr >= 45 else "#f85149"


                # ── Zone comparison table with execution detail ───────────────
                _etp_card   = sig.get("_trade_plan", {})
                _direction  = sig["direction"]
                _close_ref  = sig.get("close", 0)

                # Zone → _etp field prefix mapping
                _zone_etp = {
                    "Aggressive": ("agg_entry",    "agg_sl",    "agg_tp1",    "agg_tp2",    "agg_tp3"),
                    "Standard":   ("std_entry",    "std_sl",    "std_tp1",    "std_tp2",    "std_tp3"),
                    "Sniper":     ("sniper_entry", "sniper_sl", "sniper_tp1", "sniper_tp2", "sniper_tp3"),
                }
                FIXED_SL_PCT = 0.015

                def _zone_fixed_sl(entry_px):
                    if _direction == "long":
                        return round(entry_px * (1 - FIXED_SL_PCT), 8)
                    else:
                        return round(entry_px * (1 + FIXED_SL_PCT), 8)

                def _zone_fixed_tps(entry_px, sl_px):
                    risk = abs(entry_px - sl_px)
                    if _direction == "long":
                        return (round(entry_px + risk, 8),
                                round(entry_px + 2 * risk, 8),
                                round(entry_px + 3 * risk, 8))
                    else:
                        return (round(entry_px - risk, 8),
                                round(entry_px - 2 * risk, 8),
                                round(entry_px - 3 * risk, 8))

                def _fmt_px(v):
                    return f"{v:.6g}" if v else "—"

                def _mgmt_detail_html(entry_px, sl_px, tp1_px, tp2_px, mgmt_mode, sl_label):
                    risk = abs(entry_px - sl_px)
                    be_px = entry_px
                    if mgmt_mode == "Simple":
                        return (
                            f'<div style="color:#8892b0;font-size:10px;margin-top:6px;">'
                            f'📋 <b style="color:#ccd6f6;">Simple:</b> '
                            f'Hold full position → TP at <b style="color:#64ffda;">{_fmt_px(tp2_px)}</b> (2R) '
                            f'or SL at <b style="color:#ff6b6b;">{_fmt_px(sl_px)}</b> ({sl_label})</div>'
                        )
                    elif mgmt_mode == "Partial":
                        return (
                            f'<div style="color:#8892b0;font-size:10px;margin-top:6px;">'
                            f'📋 <b style="color:#ccd6f6;">Partial:</b> '
                            f'At <b style="color:#64ffda;">{_fmt_px(tp1_px)}</b> (1R) → close 50% → '
                            f'move SL to BE <b style="color:#e3b341;">{_fmt_px(be_px)}</b> → '
                            f'hold rest to <b style="color:#64ffda;">{_fmt_px(tp2_px)}</b> (2R)</div>'
                        )
                    elif mgmt_mode == "Trailing":
                        return (
                            f'<div style="color:#8892b0;font-size:10px;margin-top:6px;">'
                            f'📋 <b style="color:#ccd6f6;">Trailing:</b> '
                            f'At <b style="color:#64ffda;">{_fmt_px(tp1_px)}</b> (1R) → move SL to BE '
                            f'<b style="color:#e3b341;">{_fmt_px(be_px)}</b> → '
                            f'trail SL by 0.5× ATR until TP or stopped out</div>'
                        )
                    return ""

                _zone_table_rows = ""
                _zone_icons = {"Aggressive": "⚡", "Standard": "✅", "Sniper": "🎯"}
                _zone_desc  = {
                    "Aggressive": "Enter at candle close — highest fill chance",
                    "Standard":   "Wait for 38.2% retrace into candle body",
                    "Sniper":     "Wait for 61.8% Fibonacci retrace — best R:R",
                }

                for _zn in ("Aggressive", "Standard", "Sniper"):
                    _zd       = _zone_best.get(_zn, {})
                    _is_best_zone = _best_key and _zd.get("key", "") == _best_key
                    _border   = "border:1px solid #3fb950;" if _is_best_zone else "border:1px solid #30363d;"
                    _crown    = " 👑 BEST" if _is_best_zone else ""
                    _bg       = "background:#091a0d;" if _is_best_zone else "background:#0d1117;"

                    # Pull prices from _etp
                    _ep_keys   = _zone_etp.get(_zn, ())
                    _ep        = _etp_card.get(_ep_keys[0], 0) if _ep_keys else 0
                    _atr_sl_p  = _etp_card.get(_ep_keys[1], 0) if _ep_keys else 0
                    _tp1_p     = _etp_card.get(_ep_keys[2], 0) if _ep_keys else 0
                    _tp2_p     = _etp_card.get(_ep_keys[3], 0) if _ep_keys else 0
                    _tp3_p     = _etp_card.get(_ep_keys[4], 0) if _ep_keys else 0
                    _fix_sl_p  = _zone_fixed_sl(_ep) if _ep else 0
                    _fix_tp1, _fix_tp2, _fix_tp3 = _zone_fixed_tps(_ep, _fix_sl_p) if _ep else (0, 0, 0)

                    # ── Structural validity check ─────────────────────────────
                    # If the Fibonacci retrace zone overshoots the structural SL,
                    # the zone is physically impossible — show a hard warning.
                    # We check BOTH the _etp_card validity flags AND the zone_best
                    # flag that _scanner_quick_backtest now sets for filtered zones.
                    _structurally_invalid = False
                    if _zd.get("structurally_invalid"):
                        _structurally_invalid = True
                    elif _zn == "Standard" and not _etp_card.get("std_valid", True):
                        _structurally_invalid = True
                    elif _zn == "Sniper" and not _etp_card.get("sniper_valid", True):
                        _structurally_invalid = True

                    if _structurally_invalid:
                        _sl_pct_conf = _etp_card.get("sl_dist_pct", 0)
                        _fib_label   = "38.2%" if _zn == "Standard" else "61.8%"
                        _zone_table_rows += (
                            f'<div style="background:#1a0a0a;border:2px solid #6b2222;border-radius:6px;'
                            f'padding:10px 12px;margin-bottom:6px;">'
                            f'<div style="color:#ff6b6b;font-size:12px;font-weight:700;margin-bottom:4px;">'
                            f'{_zone_icons.get(_zn,"•")} {_zn} — ❌ STRUCTURALLY INVALID</div>'
                            f'<div style="color:#cc8888;font-size:11px;line-height:1.4;">'
                            f'Candle body is too large for this SL distance ({_sl_pct_conf:.1f}%). '
                            f'The {_fib_label} retrace zone falls at or beyond the structural stop-loss level. '
                            f'Entering this zone would mean your SL is already triggered at fill. '
                            f'<b style="color:#ffaa88;">Use Aggressive zone only.</b></div>'
                            f'</div>'
                        )
                        continue

                    # Best config for this zone
                    _best_sl_label = _zd.get("sl_label", "Fixed SL") if _zd else "Fixed SL"
                    _best_mgmt     = _zd.get("mgmt", "Simple") if _zd else "Simple"
                    _best_tp_mult  = _zd.get("tp_mult", 2.0) if _zd else 2.0
                    _use_atr       = "ATR" in _best_sl_label

                    # ── Price alignment fix ───────────────────────────────────
                    # All prices (SL, TP1, TP2) must be derived from the SAME
                    # config that produced the EV/WR stats shown in the card.
                    # SL distance: ATR-based (from _etp_card) or Fixed 1.5%
                    # TP target:   entry ± tp_mult × risk (NOT always 2R)
                    if _use_atr and _atr_sl_p:
                        _sl_show     = _atr_sl_p
                        _sl_pct_show = _etp_card.get("sl_dist_pct", FIXED_SL_PCT * 100)
                    else:
                        _sl_show     = _fix_sl_p
                        _sl_pct_show = FIXED_SL_PCT * 100
                    # Recompute TP1 and TP2 from the actual risk distance of this config
                    _risk_show = abs(_ep - _sl_show) if _ep and _sl_show else 0
                    if _risk_show > 0 and _ep:
                        _sign      = 1 if _direction == "long" else -1
                        _tp1_show  = round(_ep + _sign * 1.0            * _risk_show, 8)
                        _tp2_show  = round(_ep + _sign * _best_tp_mult  * _risk_show, 8)
                        _tp3_show  = round(_ep + _sign * (_best_tp_mult + 1.0) * _risk_show, 8)
                    else:
                        # Fallback to _etp values if risk calc not possible
                        _tp1_show = _tp1_p if (_use_atr and _ep) else _fix_tp1
                        _tp2_show = _tp2_p if (_use_atr and _ep) else _fix_tp2
                        _tp3_show = _tp3_p if (_use_atr and _ep) else _fix_tp3

                    if _zd and not _zd.get("insufficient") and _zd.get("n", 0) >= 4:
                        _expiry_note = "" if _zn == "Aggressive" else (
                            f' <span style="color:#e3b341;font-size:10px;">· Expires in 3 bars if not filled</span>'
                        )
                        _below_wr_floor = _zd.get("below_wr_floor", False)
                        _wr_floor_badge = (
                            f' <span style="background:#2d1a00;color:#e3b341;font-size:9px;'
                            f'padding:1px 6px;border-radius:3px;margin-left:4px;">'
                            f'⚠️ WR {_zd.get("win_rate",0):.1f}% — below 35% floor (EV shown, not recommended)</span>'
                        ) if _below_wr_floor else ""
                        _mgmt_html = _mgmt_detail_html(_ep, _sl_show, _tp1_show, _tp2_show, _best_mgmt, _best_sl_label)

                        _zone_table_rows += (
                            f'<div style="{_bg}{_border}border-radius:8px;padding:12px 14px;margin-bottom:8px;">'

                            # Header row
                            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                            f'<div>'
                            f'<span style="color:#ccd6f6;font-size:13px;font-weight:700;">{_zone_icons.get(_zn,"•")} {_zn}'
                            f'<span style="color:#3fb950;font-size:12px;">{_crown}</span></span>'
                            f'{_wr_floor_badge}'
                            f'<div style="color:#8892b0;font-size:10px;margin-top:1px;">{_zone_desc.get(_zn,"")}{_expiry_note}</div>'
                            f'</div>'
                            f'<div style="text-align:right;">'
                            f'<span style="background:#1a2030;border-radius:4px;padding:2px 8px;font-size:10px;color:#58a6ff;">{_best_sl_label} · {_best_mgmt}</span>'
                            f'<div style="color:#8892b0;font-size:10px;margin-top:2px;">n={_zd.get("n",0)} historical setups</div>'
                            f'</div>'
                            f'</div>'

                            # Stats row
                            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px;">'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:10px;">Win Rate</div>'
                            f'<div style="color:{_wr_color(_zd.get("win_rate",0))};font-size:15px;font-weight:800;">{_zd.get("win_rate",0):.1f}%</div>'
                            f'</div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:10px;">Exp. Value</div>'
                            f'<div style="color:{_ev_color(_zd.get("ev",0))};font-size:15px;font-weight:800;">{_zd.get("ev",0):+.2f}R</div>'
                            f'</div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:10px;">Avg Hold</div>'
                            f'<div style="color:#ccd6f6;font-size:15px;font-weight:800;">{_zd.get("avg_bars",0):.1f} bars</div>'
                            f'</div>'
                            f'</div>'

                            # Price levels
                            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin-bottom:6px;">'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">Entry</div>'
                            f'<div style="color:#58a6ff;font-size:12px;font-weight:700;">{_fmt_px(_ep)}</div>'
                            f'</div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">SL ({_sl_pct_show:.1f}%)</div>'
                            f'<div style="color:#ff6b6b;font-size:12px;font-weight:700;">{_fmt_px(_sl_show)}</div>'
                            f'</div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP1 (1R)</div>'
                            f'<div style="color:#64ffda;font-size:12px;font-weight:700;">{_fmt_px(_tp1_show)}</div>'
                            f'</div>'
                            f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP2 ({_zd.get("tp_mult",2.0):.1f}R) / TP3</div>'
                            f'<div style="color:#64ffda;font-size:12px;font-weight:700;">{_fmt_px(_tp2_show)}</div>'
                            f'<div style="color:#3fb950;font-size:10px;">{_fmt_px(_tp3_show)}</div>'
                            f'</div>'
                            f'</div>'

                            # Management instructions
                            + _mgmt_html
                            + f'</div>'
                        )
                    else:
                        # Check if excluded due to low win rate (35% floor) vs truly insufficient data
                        _best_wr_for_zone = max(
                            (v.get("win_rate", 0) for v in _per_method.values()
                             if v.get("zone") == _zn and not v.get("insufficient") and v.get("n", 0) >= 4),
                            default=None
                        )
                        if _best_wr_for_zone is not None and _best_wr_for_zone < 35:
                            _zone_table_rows += (
                                f'<div style="background:#0d1117;border:1px solid #2d2200;border-radius:6px;'
                                f'padding:8px 10px;margin-bottom:6px;">' 
                                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                                f'<span style="color:#8892b0;font-size:12px;">{_zone_icons.get(_zn,"•")} {_zn}</span>'
                                f'<span style="background:#2d2200;color:#e3b341;font-size:10px;padding:2px 8px;border-radius:4px;">'
                                f'⚠️ Excluded — Win Rate {_best_wr_for_zone:.1f}% below 35% minimum</span></div>'
                                f'<div style="color:#8892b0;font-size:10px;margin-top:4px;">'
                                f'EV may be positive but strategy wins fewer than 1 in 3 trades — not recommended for live trading.</div>'
                                f'</div>'
                            )
                        else:
                            _zone_table_rows += (
                                f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;'
                                f'padding:8px 10px;margin-bottom:6px;opacity:0.5;">'
                                f'<span style="color:#8892b0;font-size:12px;">{_zone_icons.get(_zn,"•")} {_zn} — insufficient data (&lt;4 setups)</span>'
                                f'</div>'
                            )

                # ── Best method recommendation with full execution plan ────────
                # Extra safety: never recommend a structurally invalid zone even if
                # best_key somehow slipped through (e.g. cached from earlier run).
                _best_zone_name = _best.get("zone", "Aggressive") if _best else "Aggressive"
                _best_structurally_ok = True
                if _best_zone_name == "Standard" and not _etp_card.get("std_valid", True):
                    _best_structurally_ok = False
                elif _best_zone_name == "Sniper" and not _etp_card.get("sniper_valid", True):
                    _best_structurally_ok = False

                if _best and _best_key and not _best_structurally_ok:
                    # Demote to best VALID zone instead
                    _fallback_best_key = None
                    _fallback_best     = {}
                    for _fb_k, _fb_v in sorted(
                        _per_method.items(), key=lambda x: -x[1].get("ev", -99)
                    ):
                        if _fb_v.get("insufficient") or _fb_v.get("n", 0) < 4:
                            continue
                        _fb_zone = _fb_v.get("zone", "Aggressive")
                        if _fb_zone == "Standard" and not _etp_card.get("std_valid", True):
                            continue
                        if _fb_zone == "Sniper" and not _etp_card.get("sniper_valid", True):
                            continue
                        _fallback_best_key = _fb_k
                        _fallback_best     = _fb_v
                        break
                    _best     = _fallback_best
                    _best_key = _fallback_best_key

                if _best and _best_key:
                    _bev    = _best.get("ev", 0)
                    _bwr    = _best.get("win_rate", 0)
                    _bn     = _best.get("n", 0)
                    _bzone  = _best.get("zone", "Aggressive")
                    _bsl    = _best.get("sl_label", "Fixed SL")
                    _bmgmt  = _best.get("mgmt", "Simple")
                    _btp    = _best.get("tp_mult", 2.0)
                    _bbars  = _best.get("avg_bars", 0)

                    _bep_keys  = _zone_etp.get(_bzone, ())
                    _bep       = _etp_card.get(_bep_keys[0], 0) if _bep_keys else 0
                    _b_atr_sl  = _etp_card.get(_bep_keys[1], 0) if _bep_keys else 0
                    _b_tp1     = _etp_card.get(_bep_keys[2], 0) if _bep_keys else 0
                    _b_tp2     = _etp_card.get(_bep_keys[3], 0) if _bep_keys else 0
                    _b_fix_sl  = _zone_fixed_sl(_bep) if _bep else 0
                    _b_fix_tp1, _b_fix_tp2, _ = _zone_fixed_tps(_bep, _b_fix_sl) if _bep else (0, 0, 0)
                    _b_use_atr = "ATR" in _bsl
                    _b_sl_px   = _b_atr_sl if (_b_use_atr and _b_atr_sl) else _b_fix_sl
                    # Recompute TP prices from actual config SL distance and tp_mult
                    # so EXECUTE THIS prices align with the EV/WR stats shown
                    _b_risk    = abs(_bep - _b_sl_px) if _bep and _b_sl_px else 0
                    if _b_risk > 0 and _bep:
                        _b_sign    = 1 if _direction == "long" else -1
                        _b_tp1_px  = round(_bep + _b_sign * 1.0   * _b_risk, 8)
                        _b_tp2_px  = round(_bep + _b_sign * _btp  * _b_risk, 8)
                    else:
                        _b_tp1_px  = _b_tp1 if (_b_use_atr and _bep) else _b_fix_tp1
                        _b_tp2_px  = _b_tp2 if (_b_use_atr and _bep) else _b_fix_tp2

                    _exec_detail = _mgmt_detail_html(_bep, _b_sl_px, _b_tp1_px, _b_tp2_px, _bmgmt, _bsl)
                    _wait_note   = (
                        f'<div style="color:#e3b341;font-size:11px;margin-top:4px;">'
                        f'⏳ Wait for retrace to <b>{_fmt_px(_bep)}</b> — expires if not filled within 3 bars</div>'
                    ) if _bzone != "Aggressive" else ""

                    _recommendation_html = (
                        f'<div style="background:#091a0d;border:1px solid #3fb950;border-radius:8px;'
                        f'padding:12px 14px;margin-top:10px;">'
                        f'<div style="color:#3fb950;font-size:11px;text-transform:uppercase;'
                        f'letter-spacing:1px;font-weight:700;margin-bottom:8px;">🏆 EXECUTE THIS — Best Proven Method</div>'
                        f'<div style="color:#ccd6f6;font-size:13px;font-weight:700;margin-bottom:8px;">{_bzone} / {_bsl} / {_bmgmt} &nbsp;<span style="color:#e3b341;font-size:12px;">TP {_btp:.1f}R</span></div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin-bottom:8px;">'
                        f'<div style="background:#0a1a0a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">Entry</div>'
                        f'<div style="color:#58a6ff;font-size:13px;font-weight:800;">{_fmt_px(_bep)}</div></div>'
                        f'<div style="background:#0a1a0a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">Stop Loss</div>'
                        f'<div style="color:#ff6b6b;font-size:13px;font-weight:800;">{_fmt_px(_b_sl_px)}</div></div>'
                        f'<div style="background:#0a1a0a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP1 (1R)</div>'
                        f'<div style="color:#64ffda;font-size:13px;font-weight:800;">{_fmt_px(_b_tp1_px)}</div></div>'
                        f'<div style="background:#0a1a0a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP ({_btp:.1f}R)</div>'
                        f'<div style="color:#64ffda;font-size:13px;font-weight:800;">{_fmt_px(_b_tp2_px)}</div></div>'
                        f'</div>'
                        + _wait_note
                        + _exec_detail
                        + f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:10px;'
                        f'padding-top:8px;border-top:1px solid #1a3a1a;">'
                        f'<div><div style="color:#8892b0;font-size:10px;">Historical Win Rate</div>'
                        f'<div style="color:{_wr_color(_bwr)};font-size:16px;font-weight:800;">{_bwr:.1f}%</div></div>'
                        f'<div><div style="color:#8892b0;font-size:10px;">Expected Value</div>'
                        f'<div style="color:{_ev_color(_bev)};font-size:16px;font-weight:800;">{_bev:+.2f}R</div></div>'
                        f'<div><div style="color:#8892b0;font-size:10px;">Sample / Avg Hold</div>'
                        f'<div style="color:#ccd6f6;font-size:16px;font-weight:800;">{_bn}t / {_bbars:.0f}b</div></div>'
                        f'</div></div>'
                    )
                else:
                    _recommendation_html = (
                        f'<div style="color:#8892b0;font-size:12px;padding:8px 0;">'
                        f'Not enough data to determine best method (&lt;4 setups per zone).</div>'
                    )

                # ── NEW: 2 CANDIDATE EXECUTION CARDS (A = newest, B = weighted) ───
                # These replace the 3 zone cards at the top of the view. The 3 zone
                # cards are still available inside an expander for power users.
                _cand_a_card = _bt_res.get("candidate_newest")
                _cand_b_card = _bt_res.get("candidate_weighted")

                def _cfg_of_card(c):
                    if not c:
                        return None
                    mc = c.get("method_cfg") or {}
                    return (mc.get("zone"), mc.get("sl_label"), mc.get("mgmt"),
                            round(float(mc.get("tp_mult", 2.0)), 2))
                _a_cfg_disp = _cfg_of_card(_cand_a_card)
                _b_cfg_disp = _cfg_of_card(_cand_b_card)
                _ab_unanimous_disp = (_a_cfg_disp is not None and _a_cfg_disp == _b_cfg_disp)

                def _build_cand_exec_card(cand, letter, title, accent, bg, border):
                    """Render one candidate execution card with prices + decay buckets."""
                    if not cand:
                        return (
                            f'<div style="background:{bg};border:1px solid {border};'
                            f'border-radius:8px;padding:12px 14px;margin-top:10px;">'
                            f'<div style="color:{accent};font-size:11px;font-weight:700;'
                            f'text-transform:uppercase;letter-spacing:1px;">{letter} · {title}</div>'
                            f'<div style="color:#8892b0;font-size:12px;margin-top:8px;">'
                            f'No valid method found — not enough historical data or all filters fail.</div>'
                            f'</div>'
                        )

                    _mc   = cand.get("method_cfg") or {}
                    _czn  = _mc.get("zone", "Aggressive")
                    _csl  = _mc.get("sl_label", "Fixed SL")
                    _cmg  = _mc.get("mgmt", "Simple")
                    _ctp  = float(_mc.get("tp_mult", 2.0))
                    _cwr  = cand.get("win_rate", 0)
                    _cev  = cand.get("ev", 0)
                    _cevw = cand.get("ev_weighted", 0)
                    _cpf  = cand.get("pf", 0)
                    _cpfs = "∞" if _cpf >= 9.9 else f"{_cpf:.2f}"
                    _cpfc = ("#3fb950" if _cpf >= 1.5 else
                             "#e3b341" if _cpf >= 1.0 else "#f85149")
                    _cn   = cand.get("n", 0)
                    _cbars= cand.get("avg_bars", 0)
                    _cnb  = cand.get("newest_bucket", {}) or {}

                    # Resolve entry/SL/TP prices using same logic as the old zone cards
                    _ekeys = _zone_etp.get(_czn, ())
                    _c_ep  = _etp_card.get(_ekeys[0], 0) if _ekeys else 0
                    _c_atr_sl = _etp_card.get(_ekeys[1], 0) if _ekeys else 0
                    _c_use_atr = "ATR" in _csl
                    if _c_ep:
                        if _c_use_atr and _c_atr_sl:
                            _c_sl_px = _c_atr_sl
                        else:
                            _c_sl_px = _zone_fixed_sl(_c_ep)
                        _c_risk = abs(_c_ep - _c_sl_px)
                        _c_sign = 1 if _direction == "long" else -1
                        if _c_risk > 0:
                            _c_tp1_px = round(_c_ep + _c_sign * 1.0   * _c_risk, 8)
                            _c_tp2_px = round(_c_ep + _c_sign * _ctp  * _c_risk, 8)
                            _c_sl_pct = round((_c_risk / _c_ep) * 100, 2)
                        else:
                            _c_tp1_px = _c_tp2_px = 0
                            _c_sl_pct = 0
                    else:
                        _c_sl_px = _c_tp1_px = _c_tp2_px = 0
                        _c_sl_pct = 0

                    _exec_detail = _mgmt_detail_html(_c_ep, _c_sl_px, _c_tp1_px, _c_tp2_px, _cmg, _csl) if _c_ep else ""
                    _wait_note = (
                        f'<div style="color:#e3b341;font-size:11px;margin-top:4px;">'
                        f'⏳ Wait for retrace to <b>{_fmt_px(_c_ep)}</b> — expires if not filled within 3 bars</div>'
                    ) if _czn != "Aggressive" and _c_ep else ""

                    # Time-decay bucket strip for this candidate
                    _buckets = cand.get("buckets", []) or []
                    _bkt_cells = ""
                    if _buckets:
                        _n_bkt = len(_buckets)
                        for _bi, _br in enumerate(_buckets):
                            _bn_i   = _br.get("n", 0)
                            _bwr_i  = _br.get("wr", 0)
                            _bev_i  = _br.get("ev", 0)
                            _bw_i   = _br.get("weight", 1.0)
                            _blbl_i = _br.get("label", "—")
                            _is_newest = (_bi == _n_bkt - 1)
                            _cell_bg = "#091a0d" if _is_newest else "#0d1117"
                            _cell_border = accent if _is_newest else "#21262d"
                            _wr_col_c = _wr_color(_bwr_i) if _bn_i >= 2 else "#555"
                            _ev_col_c = _ev_color(_bev_i) if _bn_i >= 2 else "#555"
                            _bkt_cells += (
                                f'<div style="background:{_cell_bg};border:1px solid {_cell_border};'
                                f'border-radius:4px;padding:5px 6px;">'
                                f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">'
                                f'{_blbl_i} · w={_bw_i:.2f}</div>'
                                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:2px;">'
                                f'<span style="color:{_wr_col_c};font-size:11px;font-weight:700;">{_bwr_i:.0f}%</span>'
                                f'<span style="color:{_ev_col_c};font-size:10px;">{_bev_i:+.1f}R</span>'
                                f'<span style="color:#8892b0;font-size:9px;">n={_bn_i}</span>'
                                f'</div></div>'
                            )
                        _bkt_strip = (
                            f'<div style="margin-top:8px;">'
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;'
                            f'letter-spacing:1px;margin-bottom:4px;">⏱ Time-Decay Breakdown (oldest → newest)</div>'
                            f'<div style="display:grid;grid-template-columns:repeat({_n_bkt},1fr);gap:4px;">'
                            f'{_bkt_cells}</div></div>'
                        )
                    else:
                        _bkt_strip = ""

                    return (
                        f'<div style="background:{bg};border:1px solid {border};'
                        f'border-radius:8px;padding:12px 14px;margin-top:10px;">'
                        # Header
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'margin-bottom:6px;">'
                        f'<div style="color:{accent};font-size:11px;font-weight:700;'
                        f'text-transform:uppercase;letter-spacing:1px;">{letter} · {title}</div>'
                        f'<div style="color:#8892b0;font-size:10px;">'
                        f'{_czn} / {_csl} / {_cmg} · <span style="color:#e3b341;">TP{_ctp:.1f}R</span></div>'
                        f'</div>'
                        # Price grid
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin-bottom:6px;">'
                        f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">Entry</div>'
                        f'<div style="color:#58a6ff;font-size:13px;font-weight:800;">{_fmt_px(_c_ep)}</div></div>'
                        f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">SL ({_c_sl_pct:.1f}%)</div>'
                        f'<div style="color:#ff6b6b;font-size:13px;font-weight:800;">{_fmt_px(_c_sl_px)}</div></div>'
                        f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP1 (1R)</div>'
                        f'<div style="color:#64ffda;font-size:13px;font-weight:800;">{_fmt_px(_c_tp1_px)}</div></div>'
                        f'<div style="background:#0a0f1a;border-radius:4px;padding:5px 8px;">'
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP ({_ctp:.1f}R)</div>'
                        f'<div style="color:#64ffda;font-size:13px;font-weight:800;">{_fmt_px(_c_tp2_px)}</div></div>'
                        f'</div>'
                        + _wait_note
                        + _exec_detail
                        # Stats strip
                        + f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:6px;margin-top:8px;'
                        f'padding-top:8px;border-top:1px solid #21262d;">'
                        f'<div><div style="color:#8892b0;font-size:9px;">All-time WR</div>'
                        f'<div style="color:{_wr_color(_cwr)};font-size:14px;font-weight:800;">{_cwr:.1f}%</div></div>'
                        f'<div><div style="color:#8892b0;font-size:9px;">EV</div>'
                        f'<div style="color:{_ev_color(_cev)};font-size:14px;font-weight:800;">{_cev:+.2f}R</div></div>'
                        f'<div><div style="color:#8892b0;font-size:9px;">EVw</div>'
                        f'<div style="color:{_ev_color(_cevw)};font-size:14px;font-weight:800;">{_cevw:+.2f}R</div></div>'
                        f'<div><div style="color:#8892b0;font-size:9px;">PF</div>'
                        f'<div style="color:{_cpfc};font-size:14px;font-weight:800;">{_cpfs}</div></div>'
                        f'<div><div style="color:#8892b0;font-size:9px;">Samples</div>'
                        f'<div style="color:#ccd6f6;font-size:14px;font-weight:800;">{_cn}t/{_cbars:.0f}b</div></div>'
                        f'</div>'
                        + _bkt_strip
                        + f'</div>'
                    )

                if _ab_unanimous_disp:
                    _candidate_cards_html = _build_cand_exec_card(
                        _cand_a_card, "🟢 A ≡ B",
                        "UNANIMOUS — Best in Both Views",
                        "#3fb950", "#091a0d", "#3fb950",
                    )
                else:
                    _card_a_html = _build_cand_exec_card(
                        _cand_a_card, "🟢 A",
                        "Best in Newest Bucket",
                        "#3fb950", "#091a0d", "#238636",
                    ) if _cand_a_card else ""
                    _card_b_html = _build_cand_exec_card(
                        _cand_b_card, "🔵 B",
                        "Best Weighted All-Time",
                        "#58a6ff", "#0a1628", "#1f6feb",
                    ) if _cand_b_card else ""
                    _candidate_cards_html = _card_a_html + _card_b_html

                # ── Full management breakdown (expandable) ────────────────────
                _mgmt_table = ""
                if _per_method:
                    _mgmt_rows_html = ""
                    # Sort by weighted EV so time-decay ranking surfaces the best recent methods first
                    for _mk, _mv in sorted(_per_method.items(),
                                           key=lambda x: -x[1].get("ev_weighted", x[1].get("ev", -99))):
                        if _mv.get("insufficient") or _mv.get("n", 0) < 4:
                            continue
                        _is_best = (_mk == _best_key)
                        _row_bg  = "background:#091a0d;" if _is_best else ""
                        _crown2  = " 👑" if _is_best else ""
                        _tp_label = f"TP{_mv.get('tp_mult',2.0):.1f}R"
                        _pf_val  = _mv.get("pf", 0)
                        _pf_str  = "∞" if _pf_val >= 9.9 else f"{_pf_val:.2f}"
                        _pf_c    = ("#3fb950" if _pf_val >= 1.5 else
                                    "#e3b341" if _pf_val >= 1.0 else "#f85149")
                        _evw     = _mv.get("ev_weighted", _mv.get("ev", 0))
                        _nbkt    = _mv.get("newest_bucket", {}) or {}
                        _nbkt_wr = _nbkt.get("wr", 0)
                        _nbkt_n  = _nbkt.get("n",  0)
                        _nbkt_ev = _nbkt.get("ev", 0)
                        _nbkt_txt = f"{_nbkt_wr:.0f}%/{_nbkt_ev:+.1f}R (n{_nbkt_n})" if _nbkt_n > 0 else "—"
                        _nbkt_color = _wr_color(_nbkt_wr) if _nbkt_n >= 3 else "#8892b0"
                        _mgmt_rows_html += (
                            f'<div style="{_row_bg}display:grid;grid-template-columns:2.6fr 0.7fr 0.7fr 0.7fr 0.7fr 0.7fr 1.1fr 0.8fr;'
                            f'gap:4px;padding:5px 6px;border-bottom:1px solid #1a1f2e;font-size:11px;">'
                            f'<div style="color:#ccd6f6;">{_mk}{_crown2}</div>'
                            f'<div style="color:{_wr_color(_mv["win_rate"])};text-align:right;font-weight:700;">{_mv["win_rate"]:.0f}%</div>'
                            f'<div style="color:{_ev_color(_mv["ev"])};text-align:right;font-weight:700;">{_mv["ev"]:+.2f}R</div>'
                            f'<div style="color:{_ev_color(_evw)};text-align:right;font-weight:700;">{_evw:+.2f}R</div>'
                            f'<div style="color:{_pf_c};text-align:right;font-weight:700;">{_pf_str}</div>'
                            f'<div style="color:#e3b341;text-align:right;font-weight:600;">{_tp_label}</div>'
                            f'<div style="color:{_nbkt_color};text-align:right;font-size:10px;">{_nbkt_txt}</div>'
                            f'<div style="color:#8892b0;text-align:right;">{_mv["n"]}n/{_mv["avg_bars"]:.0f}b</div>'
                            f'</div>'
                        )
                    if _mgmt_rows_html:
                        _mgmt_table = (
                            f'<div style="margin-top:10px;border:1px solid #21262d;border-radius:6px;overflow:hidden;">'
                            f'<div style="background:#161b22;display:grid;grid-template-columns:2.6fr 0.7fr 0.7fr 0.7fr 0.7fr 0.7fr 1.1fr 0.8fr;'
                            f'gap:4px;padding:5px 6px;border-bottom:1px solid #30363d;">'
                            f'<div style="color:#8892b0;font-size:10px;text-transform:uppercase;">Method (sorted by EVw)</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">WR%</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">EV</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">EVw</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">PF</div>'
                            f'<div style="color:#e3b341;font-size:10px;text-align:right;">TP</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">Newest bkt</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">n/bars</div>'
                            f'</div>'
                            f'{_mgmt_rows_html}'
                            f'</div>'
                        )

                # ── Data provenance strip ──────────────────────────────────
                # Shows what historical data the backtest ran on so the user
                # knows whether the numbers are backed by enough history.
                _meta_bt       = _bt_res.get("meta", {}) or {}
                _bars_used_p   = _meta_bt.get("bars_used", 0)
                _bars_req_p    = _meta_bt.get("bars_requested", 0)
                _coverage_p    = _meta_bt.get("bars_coverage", "—")
                _bkt_cnt_p     = _meta_bt.get("bucket_count", 1)
                _bkt_weights_p = _meta_bt.get("bucket_weights", [1.0])
                _bkt_labels_p  = _meta_bt.get("bucket_labels", ["All bars"])

                _is_short_history = (_bars_req_p > 0 and _bars_used_p < _bars_req_p * 0.9)
                _weights_str = " → ".join(f"{int(w*100)}%" for w in _bkt_weights_p)
                _provenance_note = (
                    f"⚠️ Coin is new: only {_bars_used_p} bars available (requested {_bars_req_p})"
                    if _is_short_history else
                    f"📅 {_bars_used_p} bars used"
                )
                _provenance_html = (
                    f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;'
                    f'padding:8px 12px;margin-top:10px;font-family:monospace;">'
                    f'<div style="color:#58a6ff;font-size:10px;text-transform:uppercase;'
                    f'letter-spacing:1px;font-weight:700;margin-bottom:4px;">📊 Backtest Data &amp; Time-Decay Scheme</div>'
                    f'<div style="color:#ccd6f6;font-size:11px;">'
                    f'{_provenance_note} · Coverage: {_coverage_p}'
                    f'</div>'
                    f'<div style="color:#8892b0;font-size:10px;margin-top:3px;">'
                    f'Time-decay: {_bkt_cnt_p} buckets (oldest→newest) with weights [{_weights_str}] · '
                    f'Candidate A = best WR/EV in newest bucket · Candidate B = best by weighted-EV all-time'
                    f'</div>'
                    f'</div>'
                ) if _bt_valid else ""

                # ── Time-decay bucket breakdown for the BEST method ────────
                # Shows how the edge evolved over time for the winning method.
                _best_buckets_html = ""
                _best_for_buckets = _best if _best else {}
                _best_buckets     = _best_for_buckets.get("buckets", []) if _best_for_buckets else []
                if _best_buckets and _bt_valid:
                    _bkt_row_html = ""
                    for _br in _best_buckets:
                        _br_wr = _br.get("wr", 0)
                        _br_ev = _br.get("ev", 0)
                        _br_n  = _br.get("n",  0)
                        _br_w  = _br.get("weight", 1.0)
                        _br_lb = _br.get("label", "—")
                        _wr_c  = _wr_color(_br_wr) if _br_n > 0 else "#444"
                        _ev_c  = _ev_color(_br_ev) if _br_n > 0 else "#444"
                        _bkt_row_html += (
                            f'<div style="display:grid;grid-template-columns:1.6fr 0.6fr 1fr 1fr 1fr;'
                            f'gap:4px;padding:4px 6px;border-bottom:1px solid #1a1f2e;font-size:11px;">'
                            f'<div style="color:#ccd6f6;">{_br_lb}</div>'
                            f'<div style="color:#8892b0;text-align:right;">×{_br_w:.2f}</div>'
                            f'<div style="color:{_wr_c};text-align:right;font-weight:700;">{_br_wr:.1f}%</div>'
                            f'<div style="color:{_ev_c};text-align:right;font-weight:700;">{_br_ev:+.2f}R</div>'
                            f'<div style="color:#8892b0;text-align:right;">n={_br_n}</div>'
                            f'</div>'
                        )
                    _best_buckets_html = (
                        f'<div style="margin-top:8px;border:1px solid #21262d;border-radius:6px;overflow:hidden;">'
                        f'<div style="background:#161b22;padding:6px 8px;color:#58a6ff;font-size:10px;'
                        f'text-transform:uppercase;letter-spacing:1px;font-weight:700;border-bottom:1px solid #30363d;">'
                        f'⏱ Time-Decay Breakdown — Best Method ({_best_for_buckets.get("zone","?")} / '
                        f'{_best_for_buckets.get("sl_label","?")} / {_best_for_buckets.get("mgmt","?")} / '
                        f'TP{_best_for_buckets.get("tp_mult",2.0):.1f}R)'
                        f'</div>'
                        f'<div style="background:#161b22;display:grid;grid-template-columns:1.6fr 0.6fr 1fr 1fr 1fr;'
                        f'gap:4px;padding:4px 6px;border-bottom:1px solid #30363d;">'
                        f'<div style="color:#8892b0;font-size:10px;">Bucket</div>'
                        f'<div style="color:#8892b0;font-size:10px;text-align:right;">Weight</div>'
                        f'<div style="color:#8892b0;font-size:10px;text-align:right;">WR%</div>'
                        f'<div style="color:#8892b0;font-size:10px;text-align:right;">EV</div>'
                        f'<div style="color:#8892b0;font-size:10px;text-align:right;">Trades</div>'
                        f'</div>'
                        f'{_bkt_row_html}'
                        f'</div>'
                    )

                _bt_rows = (
                    (
                        f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #21262d;">'
                        f'<div style="color:#58a6ff;font-size:11px;text-transform:uppercase;'
                        f'letter-spacing:1px;font-weight:700;margin-bottom:8px;">'
                        f'🎯 Top Candidates — Chosen from Time-Decay Analysis</div>'
                        + _provenance_html
                        + _candidate_cards_html
                        + _best_buckets_html
                        + f'</div>'
                    ) if _bt_valid else
                    f'<div style="color:#8892b0;font-size:12px;padding:5px 0;">📊 Backtest: {_bt_res.get("error","No matching setups")}</div>'
                )

                # AI verdict block
                _ai_block = ""
                if _ai_res:
                    # Handle legacy single-verdict fallback (shouldn't happen but safe)
                    if not _ai_res.get("dual"):
                        # Legacy format wrapper
                        _ai_res = {
                            "dual": True,
                            "candidate_a": {
                                "verdict":   _ai_res.get("verdict", "WAIT"),
                                "confidence":_ai_res.get("confidence", "MEDIUM"),
                                "rationale": _ai_res.get("rationale", ""),
                                "execution": _ai_res.get("execution", ""),
                                "risk":      _ai_res.get("risk", ""),
                                "conflicts": _ai_res.get("conflicts", ""),
                            },
                            "candidate_b": {
                                "verdict": "—", "confidence": "",
                                "rationale": "", "execution": "", "risk": "", "conflicts": "",
                            },
                            "winner": "A", "winner_rationale": "",
                            "unanimous": True,
                            "source": _ai_res.get("source", ""),
                        }

                    _cA = _ai_res.get("candidate_a", {}) or {}
                    _cB = _ai_res.get("candidate_b", {}) or {}
                    _winner = _ai_res.get("winner", "NONE")
                    _winner_why = _ai_res.get("winner_rationale", "")
                    _unanimous_ai = _ai_res.get("unanimous", False)
                    _src = _ai_res.get("source", "")

                    def _render_cand_verdict(c, letter, accent, title, is_winner):
                        _v = c.get("verdict", "WAIT")
                        _cc= c.get("confidence", "")
                        _v_color = ("#3fb950" if _v == "TRADE"
                                    else "#e3b341" if _v == "WAIT"
                                    else "#f85149" if _v == "NO TRADE"
                                    else "#8892b0")
                        _v_bg    = ("#091a0d" if _v == "TRADE"
                                    else "#1a1500" if _v == "WAIT"
                                    else "#1a0505" if _v == "NO TRADE"
                                    else "#0d1117")
                        _c_badge = (f'<span style="background:#1f2b1f;color:#3fb950;font-size:9px;'
                                    f'border-radius:3px;padding:1px 5px;margin-left:5px;">{_cc}</span>'
                                    if _cc in ("HIGH", "MEDIUM", "LOW") else "")
                        _winner_badge = (
                            f'<span style="background:#2d2200;color:#ffd700;font-size:10px;'
                            f'border-radius:3px;padding:2px 6px;margin-left:6px;font-weight:800;">👑 WINNER</span>'
                            if is_winner else ""
                        )

                        _exec_str = c.get("execution", "")
                        _exec_row = (
                            f'<div style="background:#0a1628;border:1px solid #1f6feb;border-radius:4px;'
                            f'padding:6px 8px;margin-top:6px;">'
                            f'<div style="color:#58a6ff;font-size:9px;text-transform:uppercase;'
                            f'letter-spacing:1px;margin-bottom:2px;">📋 Execution</div>'
                            f'<div style="color:#ccd6f6;font-size:11px;line-height:1.5;">{_exec_str}</div>'
                            f'</div>'
                        ) if _exec_str else ""

                        _conflicts_str = c.get("conflicts", "")
                        _conflicts_is_clean = (not _conflicts_str
                                               or _conflicts_str.lower() == "none detected"
                                               or _conflicts_str.lower() == "none")
                        if _conflicts_is_clean:
                            _conflicts_row = (
                                f'<div style="background:#0a1a0a;border-radius:4px;padding:5px 8px;margin-top:4px;">'
                                f'<span style="color:#3fb950;font-size:9px;text-transform:uppercase;">✅ Conflicts:</span>'
                                f'<span style="color:#ccd6f6;font-size:10px;"> None detected</span></div>'
                            )
                        elif _conflicts_str:
                            _conflicts_row = (
                                f'<div style="background:#1a1500;border-radius:4px;padding:5px 8px;margin-top:4px;">'
                                f'<span style="color:#e3b341;font-size:9px;text-transform:uppercase;">⚠️ Conflicts:</span>'
                                f'<span style="color:#ccd6f6;font-size:10px;"> {_conflicts_str}</span></div>'
                            )
                        else:
                            _conflicts_row = ""

                        _risk_str = c.get("risk", "")
                        _risk_row = (
                            f'<div style="background:#1a0a0a;border-radius:4px;padding:5px 8px;margin-top:4px;">'
                            f'<span style="color:#e3b341;font-size:9px;text-transform:uppercase;">⚠️ Risk:</span>'
                            f'<span style="color:#ccd6f6;font-size:10px;"> {_risk_str}</span></div>'
                        ) if _risk_str else ""

                        return (
                            f'<div style="background:{_v_bg};border:1px solid {accent};'
                            f'border-radius:6px;padding:10px 12px;">'
                            f'<div style="color:{accent};font-size:10px;font-weight:700;'
                            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'
                            f'{letter} · {title}{_winner_badge}</div>'
                            f'<div style="color:{_v_color};font-size:20px;font-weight:900;margin-bottom:4px;">'
                            f'{_v}{_c_badge}</div>'
                            f'<div style="color:#ccd6f6;font-size:11px;line-height:1.5;">'
                            f'{c.get("rationale","")}</div>'
                            + _exec_row
                            + _conflicts_row
                            + _risk_row
                            + f'</div>'
                        )

                    if _unanimous_ai:
                        # Single card
                        _ai_cards_html = _render_cand_verdict(
                            _cA, "🟢 A ≡ B", "#3fb950",
                            "UNANIMOUS Analysis",
                            is_winner=True,
                        )
                    else:
                        _cardA = _render_cand_verdict(
                            _cA, "🟢 A", "#238636",
                            "Best Newest-Bucket",
                            is_winner=(_winner == "A"),
                        )
                        _cardB = _render_cand_verdict(
                            _cB, "🔵 B", "#1f6feb",
                            "Best Weighted All-Time",
                            is_winner=(_winner == "B"),
                        )
                        _ai_cards_html = (
                            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'
                            f'{_cardA}{_cardB}</div>'
                        )

                    # Winner banner (only when dual and one is picked)
                    _winner_banner = ""
                    if not _unanimous_ai and _winner in ("A", "B") and _winner_why:
                        _w_color = "#3fb950" if _winner == "A" else "#58a6ff"
                        _w_bg    = "#091a0d" if _winner == "A" else "#0a1628"
                        _ab_trade_a = _cA.get("verdict") == "TRADE"
                        _ab_trade_b = _cB.get("verdict") == "TRADE"
                        if _ab_trade_a and _ab_trade_b:
                            _banner_label = f"👑 AI Recommends Candidate {_winner}"
                        elif _ab_trade_a or _ab_trade_b:
                            _banner_label = f"👑 Only Candidate {_winner} is Tradeable"
                        else:
                            _banner_label = "⚠️ Neither Candidate is Tradeable"
                        _winner_banner = (
                            f'<div style="margin-top:10px;background:{_w_bg};'
                            f'border:2px solid {_w_color};border-radius:8px;padding:10px 14px;">'
                            f'<div style="color:{_w_color};font-size:12px;font-weight:800;'
                            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'
                            f'{_banner_label}</div>'
                            f'<div style="color:#ccd6f6;font-size:12px;line-height:1.5;">'
                            f'{_winner_why}</div></div>'
                        )
                    elif _winner == "NONE" and not _unanimous_ai:
                        # Both untradeable or parse error
                        _winner_banner = (
                            f'<div style="margin-top:10px;background:#1a0a0a;'
                            f'border:2px solid #6b2222;border-radius:8px;padding:10px 14px;">'
                            f'<div style="color:#ff6b6b;font-size:12px;font-weight:800;'
                            f'text-transform:uppercase;letter-spacing:1px;">'
                            f'⚠️ No Clear Winner</div>'
                            f'<div style="color:#ccd6f6;font-size:11px;margin-top:4px;">'
                            f'{_winner_why or "Neither candidate passed the decision rules — wait for better conditions."}</div></div>'
                        )

                    _ai_block = (
                        f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #21262d;">'
                        f'<div style="color:#8892b0;font-size:10px;text-transform:uppercase;'
                        f'letter-spacing:1px;margin-bottom:8px;">'
                        f'🤖 AI Dual-Candidate Analysis{" (Unanimous)" if _unanimous_ai else ""}</div>'
                        + _ai_cards_html
                        + _winner_banner
                        + (f'<div style="color:#3a3f4b;font-size:10px;margin-top:6px;">{_src}</div>'
                           if _src and _src != "error" else "")
                        + f'</div>'
                    )

                _ml_color = "#3fb950" if _ml_res["pct"] >= 70 else "#e3b341" if _ml_res["pct"] >= 55 else "#f85149"
                _edge_bt  = (
                    f' · Best: {_best_key} WR={_best.get("win_rate",0):.0f}% EV={_best.get("ev",0):+.2f}R'
                    if _bt_valid and _best_key else
                    f' · {_bt_res["win_2r"]:.0f}% hist win · EV {_bt_res["ev_2r"]:+.2f}R'
                    if _bt_valid else ""
                )
                _html = (
                    f'<div style="background:#0d1117;border:1px solid #2d3250;border-radius:10px;padding:16px 20px;margin-top:8px;">'
                    f'<div style="display:flex;align-items:center;gap:16px;padding-bottom:12px;border-bottom:1px solid #21262d;margin-bottom:12px;">'
                    f'<div style="text-align:center;"><div style="color:#8892b0;font-size:10px;text-transform:uppercase;letter-spacing:1px;">Grade</div>'
                    f'<div style="color:{_grade_color};font-size:40px;font-weight:900;line-height:1;">{_grade}</div></div>'
                    f'<div><div style="color:#58a6ff;font-size:13px;font-weight:700;">📋 CONFLUENCE ANALYSIS</div>'
                    f'<div style="color:#8892b0;font-size:12px;margin-top:2px;">{_grade_desc}</div></div></div>'
                    f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
                    f'<span style="color:#8892b0;font-size:12px;">🤖 ML Probability</span>'
                    f'<span style="color:{_ml_color};font-size:13px;font-weight:700;">'
                    f'{_ml_res["pct"]:.1f}% <span style="font-size:10px;color:#8892b0;">{_ml_res["label"]}</span></span></div>'
                    + _bt_rows
                    + _ml_card_html
                    + _wfo_block_html
                    + _intel_expander_html
                    + f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #21262d;">'
                    f'<div style="color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">Edge Summary</div>'
                    f'<div style="color:#ccd6f6;font-size:12px;">ML {_ml_res["pct"]:.0f}%'
                    + _edge_bt
                    + f' · Score {score_pct}/100 · {sig["regime"]} regime</div></div>'
                    + _ai_block
                    + f'</div>'
                )
                st.markdown(_html, unsafe_allow_html=True)

                # ── Expanders for advanced details (collapsed by default) ─────
                if _bt_valid:
                    # Expander 1: Full 3-zone comparison (Aggressive/Standard/Sniper)
                    with st.expander("▸ View Full 3-Zone Comparison  (Aggressive / Standard / Sniper)", expanded=False):
                        _zone_expander_html = (
                            f'<div style="padding:6px 0;">'
                            f'<div style="color:#8892b0;font-size:11px;margin-bottom:8px;">'
                            f'Best config found for each of the three entry zones, '
                            f'plus the legacy "EXECUTE THIS" recommendation.</div>'
                            + _zone_table_rows
                            + _recommendation_html
                            + f'</div>'
                        )
                        st.markdown(_zone_expander_html, unsafe_allow_html=True)

                    # Expander 2: Full method breakdown (all 54 combinations)
                    if _mgmt_table:
                        with st.expander("▸ Full Method Breakdown  (all 54 combinations sorted by EVw)", expanded=False):
                            st.markdown(
                                f'<div style="padding:6px 0;">'
                                f'<div style="color:#8892b0;font-size:11px;margin-bottom:8px;">'
                                f'All tested combinations of Entry Zone × SL Method × Management × TP multiplier. '
                                f'Rows are sorted by <b>EVw (time-decay weighted EV)</b> so recent performance '
                                f'surfaces first. The crown 👑 marks the overall best.</div>'
                                + _mgmt_table
                                + f'</div>',
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown(
                    '<div style="color:#8892b0;font-size:12px;padding:8px 0;">'
                    '▸ Click <b>Step 1</b> (Backtest + WFO) → <b>Step 2</b> (Train ML for Both Candidates) '
                    '→ <b>Step 3</b> (AI Dual-Candidate Analysis).</div>',
                    unsafe_allow_html=True,
                )

    # Download button
    st.markdown("---")
    _dl_rows = []
    for s in all_signals_deduped:
        _etp_dl = s.get("_trade_plan", {})
        _dl_rows.append({
            "Symbol":          s["symbol"],
            "Timeframe":       s["timeframe"],
            "Direction":       s["direction"].upper(),
            "Score":           s["score"],
            "Regime":          s["regime"],
            "Body%":           s["body_pct"],
            "VolMult":         s["vol_mult"],
            "ADX":             s["adx"],
            # Aggressive zone (enter at close)
            "Agg_Entry":       s["entry"],
            "Agg_SL":          s["sl"],
            "Agg_TP1":         _etp_dl.get("agg_tp1", ""),
            "Agg_TP2":         s["tp2r"],
            "Agg_TP3":         s["tp3r"],
            # Standard zone (38.2% retrace)
            "Std_Entry":       _etp_dl.get("std_entry", ""),
            "Std_SL":          _etp_dl.get("std_sl",   ""),
            "Std_TP1":         _etp_dl.get("std_tp1",  ""),
            "Std_TP2":         _etp_dl.get("std_tp2",  ""),
            "Std_TP3":         _etp_dl.get("std_tp3",  ""),
            # Sniper zone (61.8% retrace)
            "Sniper_Entry":    _etp_dl.get("sniper_entry", ""),
            "Sniper_SL":       _etp_dl.get("sniper_sl",   ""),
            "Sniper_TP1":      _etp_dl.get("sniper_tp1",  ""),
            "Sniper_TP2":      _etp_dl.get("sniper_tp2",  ""),
            "Sniper_TP3":      _etp_dl.get("sniper_tp3",  ""),
            # Meta
            "SL_Pct":          _etp_dl.get("sl_dist_pct", ""),
            "ATR_Pct":         _etp_dl.get("atr_pct",     ""),
            "CandleDate":      s.get("candle_date", ""),
            "Reasons":         " | ".join(s["reasons"]),
        })
    _dl_df = pd.DataFrame(_dl_rows)
    st.download_button(
        "⬇ Download All Results as CSV",
        _dl_df.to_csv(index=False).encode("utf-8"),
        f"market_scanner_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
    )



# ─── AutoFinder Entry Point ────────────────────────────────────────────────────

def main():
    """Standalone AutoFinder — Market Scanner only."""
    if "ai_provider" not in st.session_state:
        st.session_state["ai_provider"] = "Groq (Free)"
    if "groq_api_key" not in st.session_state:
        st.session_state["groq_api_key"] = ""

    with st.sidebar:
        st.markdown("## 🔭 AutoFinder")
        st.caption("Scans all liquid Binance altcoins for live momentum signals.")
        st.markdown("---")

        with st.expander("🤖 AI Analysis (optional)", expanded=False):
            st.markdown(
                '<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:6px;' +
                'padding:8px 10px;font-size:12px;color:#ccd6f6;margin-bottom:8px;">' +
                '<b style="color:#58a6ff;">Groq is FREE</b> — sign up at ' +
                '<b>console.groq.com</b>, no credit card needed.</div>',
                unsafe_allow_html=True,
            )
            _ai_provider = st.selectbox(
                "AI Provider",
                ["Groq (Free)", "Anthropic (Claude)"],
                key="ai_provider",
            )
            if "Groq" in _ai_provider:
                st.text_input(
                    "Groq API Key", type="password", key="groq_api_key",
                    placeholder="gsk_...",
                    help="Get free key at console.groq.com → API Keys",
                )
                if st.session_state.get("groq_api_key"):
                    st.caption("✅ Groq key set — AI analysis ready")
                    st.session_state["groq_model"] = st.selectbox(
                        "Groq Model",
                        [
                            "openai/gpt-oss-120b",     # flagship reasoning (default)
                            "openai/gpt-oss-20b",      # faster reasoning
                            "qwen/qwen3-32b",          # alt reasoning
                            "llama-3.3-70b-versatile", # non-reasoning fallback
                            "meta-llama/llama-4-scout-17b-16e-instruct",  # long context
                        ],
                        index=0,
                        key="groq_model_select",
                        help=("gpt-oss-120b is the strongest free reasoning model on Groq "
                              "and is recommended. Falls back to 70B versatile if rate-limited."),
                    )
            else:
                st.text_input(
                    "Anthropic API Key", type="password", key="anthropic_api_key",
                    placeholder="sk-ant-...",
                )
                if st.session_state.get("anthropic_api_key"):
                    st.caption("✅ Anthropic key set (Claude)")

        st.markdown("---")
        st.caption("Data: Binance · Bybit · OKX | All free APIs")

    render_auto_analyzer(
        ticker="",
        df_full_1d=pd.DataFrame(),
        tc=0.001,
        current_tf="1D",
    )


if __name__ == "__main__":
    main()
