"""
Momentum Candle Strategy Backtester
Daily candle pattern detection, backtesting, and optimization dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from itertools import product
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Momentum Candle Backtester",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},          # hide "Deploy" / report bug to reduce header overhead
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
    .best-box {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #238636; border-radius: 10px;
        padding: 20px 24px; margin: 12px 0;
    }
    .best-box h3 { color: #3fb950; margin: 0 0 12px 0; font-size: 16px; }
    .best-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }
    .best-key { color: #8b949e; font-size: 13px; }
    .best-val { color: #e6edf3; font-size: 13px; font-weight: 600; }
    .info-box {
        background: #0d1f2d; border: 1px solid #1f6feb;
        border-radius: 8px; padding: 16px 20px; margin: 12px 0;
    }
    .info-box h4 { color: #58a6ff; margin: 0 0 10px 0; }
    .signal-card {
        background: #0d1f0d; border: 1px solid #238636;
        border-radius: 8px; padding: 16px 20px; margin: 12px 0;
        font-family: monospace;
    }
    .signal-card h4 { color: #3fb950; margin: 0 0 10px 0; }
    .signal-line { color: #ccd6f6; padding: 2px 0; font-size: 13px; }
    .signal-line span { color: #64ffda; font-weight: 600; }
    div[data-testid="stTabs"] button { font-size: 14px; font-weight: 600; }
    /* No transitions — instant re-renders */
    .main .block-container,
    section[data-testid="stSidebar"] { transition: none !important; }
    div[data-testid="stTabContent"] { min-height: 200px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────

# Feature 2: full Fibonacci + round retracement grid
RETRACEMENTS = [0.0, 0.10, 0.20, 0.236, 0.30, 0.382, 0.40, 0.50, 0.618, 0.786]
RETRACE_LABELS = {
    0.0:   "Immediate (0%)",
    0.10:  "10% Retrace",
    0.20:  "20% Retrace",
    0.236: "Fib 23.6%",
    0.30:  "30% Retrace",
    0.382: "Fib 38.2%",
    0.40:  "40% Retrace",
    0.50:  "50% Retrace",
    0.618: "Fib 61.8%",
    0.786: "Fib 78.6%",
}

SL_DISTANCES = [0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040]
SL_LABELS    = {
    0.005: "0.5%", 0.008: "0.8%", 0.010: "1.0%", 0.015: "1.5%",
    0.020: "2.0%", 0.025: "2.5%", 0.030: "3.0%", 0.040: "4.0%",
}

TP_MODES = ["2R", "3R", "Partial"]

# Entry management constants used by the enhanced backtest and optimizer
ENTRY_EXPIRY_BARS = 3          # retrace entries expire if not filled within N bars
SL_MODES_ENHANCED = ["Fixed", "Breakeven", "Trailing ATR", "Breakeven+Trail"]

ANALYSIS_SYSTEM_PROMPT = """You are a systematic momentum trading analyst. You produce structured trade decisions — not commentary, not education, not opinions. Every response follows the exact output format below. No exceptions.

OUTPUT FORMAT (mandatory):

1. REGIME: [GREEN/YELLOW/RED] — Score [X]/100 — [one sentence]
2. CANDLE: [body%]% body | [vol]x vol | ADX [X] — [STRONG/MODERATE/WEAK]
3. EVIDENCE: [N] similar -> [X]% win rate [CI] | ML: [Y]% | Combined: [Z]%
4. FAILURE MODE: [How this setup typically loses — 1-2 sentences]
5. ENTRY (only if TRADE): use the backtested parameters from Section D exactly.
   Dir: [LONG/SHORT] | Entry: [price from D] | SL: [price from D] | TP: [price from D] | Hold: per TP mode | Size: [FULL/HALF]
6. VERDICT: **[TRADE / NO TRADE / WAIT]** — [one sentence reason]
7. FLIP: [what changes this verdict]

HARD OVERRIDES (always NO TRADE):
- Regime RED (< 45)
- ML < 40%
- ML-historical gap > 20pp with N >= 20
- ATR ratio > 3.0

SCORING:
- Combined >= 60% AND GREEN -> TRADE (full)
- Combined >= 65% AND YELLOW -> TRADE (half)
- Combined 50-59% AND GREEN -> WAIT
- Combined < 50% -> NO TRADE

RULES: Biased toward NO TRADE in ambiguity. Never exceed 250 words. Never say "it depends." Never give entry for NO TRADE/WAIT. Lead with weakest signal when candle is strong but regime is weak. Never construct narratives for why candles won — report statistics only."""

HISTORY_OPTIONS = {
    "3 months": 90, "6 months": 180, "1 year": 365,
    "2 years": 730, "3 years": 1095, "5 years": 1825,
}

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

# ─── Data Layer ────────────────────────────────────────────────────────────────

def normalise_ticker(raw: str) -> str:
    """Convert any ticker input to Binance symbol format (e.g. BTCUSDT)."""
    t = raw.upper().strip().replace("-", "").replace("/", "").replace("_", "")
    if t.endswith("USDT"):
        return t
    if t.endswith("USD"):
        return t[:-3] + "USDT"
    return t + "USDT"




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
    Fetch latest perpetual funding rate from Binance Futures API.
    Returns dict with 'rate' (float, e.g. 0.0001) and 'ok' bool.
    """
    try:
        sym = symbol.upper()
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": sym, "limit": 1},
            timeout=5,
        )
        data = resp.json()
        if data and isinstance(data, list):
            rate = float(data[0]["fundingRate"])
            return {"rate": rate, "ok": True}
        return {"rate": 0.0, "ok": False}
    except Exception:
        return {"rate": 0.0, "ok": False}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_interest(symbol: str) -> dict:
    """
    Fetch open interest (current + 24h-ago) from Binance Futures API.
    Returns dict with 'oi_now', 'oi_24h_ago', 'oi_change_pct', 'ok'.
    """
    try:
        sym = symbol.upper()
        # Current OI
        r_now = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": sym},
            timeout=5,
        )
        oi_now = float(r_now.json()["openInterest"])

        # Historical OI — last 25h of 1h candles gives us 24h delta
        r_hist = requests.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": sym, "period": "1h", "limit": 25},
            timeout=5,
        )
        hist = r_hist.json()
        if hist and isinstance(hist, list):
            oi_24h_ago = float(hist[0]["sumOpenInterest"])
            oi_change_pct = (oi_now - oi_24h_ago) / max(oi_24h_ago, 1e-9) * 100
        else:
            oi_24h_ago    = oi_now
            oi_change_pct = 0.0

        return {
            "oi_now":        oi_now,
            "oi_24h_ago":    oi_24h_ago,
            "oi_change_pct": oi_change_pct,
            "ok":            True,
        }
    except Exception:
        return {"oi_now": 0.0, "oi_24h_ago": 0.0, "oi_change_pct": 0.0, "ok": False}


def detect_wyckoff_phase(df: pd.DataFrame, bar_index: int, adx_df=None) -> dict:
    """
    Classify the Wyckoff market phase from ATR ratio, EMA alignment, vol_delta.
    Returns dict with 'phase' (str) and 'label' (display string).

    Phases:
      ACCUMULATION — Low vol, flat EMAs, rising vol_delta_5 (smart money absorbing)
      MARKUP       — Rising ATR, aligned EMAs, positive vol_delta (highest quality)
      DISTRIBUTION — High ATR, flattening EMAs, negative vol_delta (traps forming)
      MARKDOWN     — Declining EMAs, negative vol_delta, high ATR (skip longs)
      RANGING      — Low ATR, flat EMAs, mixed vol_delta (no edge)
    """
    try:
        bar = df.iloc[bar_index]
    except (IndexError, TypeError):
        bar = df.iloc[-1]

    atr_ratio  = float(bar.get("atr_ratio",   1.0) or 1.0)
    vol_delta5 = float(bar.get("vol_delta_5", 0.0) or 0.0)
    ema5       = float(bar.get("ema5",  bar.get("close", 0)) or bar.get("close", 0))
    ema15      = float(bar.get("ema15", bar.get("close", 0)) or bar.get("close", 0))
    ema21      = float(bar.get("ema21", bar.get("close", 0)) or bar.get("close", 0))

    # EMA slope direction — compare current vs 5 bars ago
    ema21_5ago = 0.0
    if bar_index >= 5 and "ema21" in df.columns:
        ema21_5ago = float(df["ema21"].iloc[bar_index - 5] or 0.0)
    ema_rising   = ema21 > ema21_5ago
    ema_aligned_bull = ema5 > ema15 > ema21
    ema_aligned_bear = ema5 < ema15 < ema21
    ema_flat     = not ema_aligned_bull and not ema_aligned_bear

    adx_val = 0.0
    if adx_df is not None and "adx" in adx_df.columns:
        try:
            adx_val = float(adx_df["adx"].iloc[bar_index])
        except Exception:
            pass

    # Classification logic
    if atr_ratio < 0.8 and ema_flat and vol_delta5 > 0:
        phase = "ACCUMULATION"
        color = "#58a6ff"
        note  = "Early — reduce size"
    elif atr_ratio >= 0.8 and ema_aligned_bull and vol_delta5 > 0:
        phase = "MARKUP"
        color = "#3fb950"
        note  = "Highest quality longs"
    elif atr_ratio > 1.5 and ema_flat and vol_delta5 < 0:
        phase = "DISTRIBUTION"
        color = "#e3b341"
        note  = "Caution — momentum traps"
    elif ema_aligned_bear and vol_delta5 < 0 and atr_ratio > 1.0:
        phase = "MARKDOWN"
        color = "#f85149"
        note  = "Skip longs"
    else:
        phase = "RANGING"
        color = "#8892b0"
        note  = "No directional edge"

    label = f"{phase} ({note})"
    return {"phase": phase, "color": color, "label": label, "note": note}



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


def render_regime_banner(regime_data):
    """
    Full-width colored banner at top of page.
    Green background for GREEN, yellow-ish for YELLOW, red-ish for RED.
    Shows: icon + verdict label + score + breakdown line + flip condition.
    """
    if regime_data is None:
        return

    verdict    = regime_data.get("verdict",        "RED")
    score      = regime_data.get("score",          0)
    breakdown  = regime_data.get("breakdown_line", "")
    flip_cond  = regime_data.get("flip_condition", "")
    overrides  = regime_data.get("hard_overrides", [])
    wyckoff    = regime_data.get("wyckoff",        {})
    fg_val     = regime_data.get("fg_val",         None)
    fg_label   = regime_data.get("fg_label",       "")
    btc_d_val  = regime_data.get("btc_d_val",      None)

    _styles = {
        "GREEN":  ("background:#0d2818;border:1px solid #238636;",
                   "#3fb950", "✅", "REGIME: GO"),
        "YELLOW": ("background:#2d2200;border:1px solid #d29922;",
                   "#e3b341", "⚠️", "REGIME: CAUTION"),
        "RED":    ("background:#2d0d0d;border:1px solid #da3633;",
                   "#f85149", "❌", "REGIME: WAIT"),
    }
    box_style, accent, icon, label = _styles.get(verdict, _styles["RED"])

    override_html = ""
    if overrides:
        items = "".join(
            f'<div style="color:#f85149;font-size:12px;margin-top:2px;">🚫 {o}</div>'
            for o in overrides
        )
        override_html = f'<div style="margin-top:6px;">{items}</div>'

    flip_html = ""
    if flip_cond:
        flip_html = (f'<div style="color:#8b949e;font-size:12px;margin-top:4px;">'
                     f'↳ {flip_cond}</div>')

    # Wyckoff phase pill
    wyckoff_html = ""
    if wyckoff:
        wc = wyckoff.get("color", "#8892b0")
        wl = wyckoff.get("label", "")
        wyckoff_html = (f'<span style="background:{wc}22;border:1px solid {wc};'
                        f'color:{wc};font-size:11px;border-radius:4px;padding:2px 7px;'
                        f'margin-left:8px;font-weight:600;">PHASE: {wl}</span>')

    # F&G pill
    fg_html = ""
    if fg_val is not None:
        fg_color = "#f85149" if fg_val < 25 else ("#e3b341" if fg_val < 45 else
                   "#3fb950" if fg_val > 75 else "#8892b0")
        fg_html = (f'<span style="background:{fg_color}22;border:1px solid {fg_color};'
                   f'color:{fg_color};font-size:11px;border-radius:4px;padding:2px 7px;'
                   f'margin-left:8px;font-weight:600;">F&G {fg_val} {fg_label}</span>')

    # BTC.D pill
    btcd_html = ""
    if btc_d_val is not None:
        btcd_color = "#f85149" if btc_d_val > 56 else "#3fb950"
        btcd_html = (f'<span style="background:{btcd_color}22;border:1px solid {btcd_color};'
                     f'color:{btcd_color};font-size:11px;border-radius:4px;padding:2px 7px;'
                     f'margin-left:8px;font-weight:600;">BTC.D {btc_d_val:.1f}%</span>')

    st.markdown(f"""
<div style="{box_style}border-radius:8px;padding:14px 18px;margin-bottom:12px;">
  <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">
    <span style="font-size:18px;">{icon}</span>
    <span style="color:{accent};font-weight:700;font-size:16px;">{label}</span>
    <span style="color:{accent};font-size:22px;font-weight:800;">{score}/100</span>
    {wyckoff_html}{fg_html}{btcd_html}
  </div>
  <div style="color:#8b949e;font-size:12px;margin-top:6px;font-family:monospace;">
    {breakdown}
  </div>
  {flip_html}{override_html}
</div>
""", unsafe_allow_html=True)


# ─── Similarity Engine ─────────────────────────────────────────────────────────

def compute_buckets(df, feature_cols):
    """
    For each feature in feature_cols, compute 33rd/66th percentile thresholds
    and classify each row into 'Low', 'Med', or 'High'.
    Adds {feature}_bucket columns. Returns modified DataFrame.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col + "_bucket"] = "Med"
            continue
        p33 = df[col].quantile(0.33)
        p66 = df[col].quantile(0.66)
        def _classify(v, lo=p33, hi=p66):
            if v <= lo:
                return "Low"
            elif v <= hi:
                return "Med"
            return "High"
        df[col + "_bucket"] = df[col].apply(_classify)
    return df


def find_similar_candles(target, historical_qualifying, direction,
                         regime_zone, session=None, min_matches=15,
                         ticker=""):
    """
    Match the target candle against historical qualifying candles by bucket
    proximity, then rank by weighted Euclidean distance.

    Returns (matched_df, metadata_dict).
    metadata keys: n, relaxed_dim, quality_pct, target_buckets
    """
    import math

    FEATURE_COLS  = ["body_pct_abs", "vol_mult", "atr_ratio", "adx_value"]
    WEIGHTS       = {"body_pct_abs": 0.35, "vol_mult": 0.30,
                     "atr_ratio": 0.20,    "adx_value": 0.15}
    RELAX_ORDER   = ["atr_ratio", "adx_value", "vol_mult", "body_pct_abs"]

    # ── Prepare historical df ─────────────────────────────────────────────────
    hq = historical_qualifying.copy()
    if hq.empty:
        return pd.DataFrame(), {"n": 0, "relaxed_dim": None,
                                 "quality_pct": 0, "target_buckets": {}}

    # body_pct_abs
    if "body_pct_abs" not in hq.columns:
        hq["body_pct_abs"] = hq["body_pct"].abs() if "body_pct" in hq.columns else 0.0

    # adx_value column (may be pre-joined or absent)
    if "adx_value" not in hq.columns:
        hq["adx_value"] = 25.0   # neutral fallback

    # Bucket thresholds from historical distribution
    thresholds = {}
    for col in FEATURE_COLS:
        if col in hq.columns:
            thresholds[col] = (hq[col].quantile(0.33), hq[col].quantile(0.66))
        else:
            thresholds[col] = (0.5, 1.0)

    hq = compute_buckets(hq, FEATURE_COLS)

    # ── Determine target buckets using same thresholds ────────────────────────
    target_buckets = {}
    for col in FEATURE_COLS:
        val = float(target.get(col, 0) if isinstance(target, dict)
                    else getattr(target, col, 0))
        lo, hi = thresholds[col]
        if val <= lo:
            target_buckets[col] = "Low"
        elif val <= hi:
            target_buckets[col] = "Med"
        else:
            target_buckets[col] = "High"

    # ── Direction filter ──────────────────────────────────────────────────────
    if "body" in hq.columns:
        if direction == "long":
            hq = hq[hq["body"] > 0]
        else:
            hq = hq[hq["body"] < 0]

    # ── Regime zone filter ────────────────────────────────────────────────────
    if regime_zone and "regime_zone" in hq.columns:
        hq = hq[hq["regime_zone"] == regime_zone]

    # ── Session filter ────────────────────────────────────────────────────────
    if session and "session" in hq.columns:
        hq = hq[hq["session"] == session]

    if hq.empty:
        return pd.DataFrame(), {"n": 0, "relaxed_dim": None,
                                 "quality_pct": 0, "target_buckets": target_buckets}

    # ── Adjacent bucket helper ────────────────────────────────────────────────
    _adjacent = {"Low": {"Low", "Med"}, "Med": {"Low", "Med", "High"},
                 "High": {"Med", "High"}}

    def _bucket_filter(frame, fixed_dims, relaxed_dim=None):
        mask = pd.Series(True, index=frame.index)
        for col in FEATURE_COLS:
            bcol = col + "_bucket"
            if bcol not in frame.columns:
                continue
            tgt = target_buckets[col]
            if col == relaxed_dim:
                allowed = _adjacent[tgt]
                mask &= frame[bcol].isin(allowed)
            else:
                mask &= frame[bcol] == tgt
        return frame[mask]

    # ── Primary match: all 4 dims exact ──────────────────────────────────────
    matched      = _bucket_filter(hq, FEATURE_COLS)
    relaxed_dim  = None

    # ── Relaxation passes ────────────────────────────────────────────────────
    if len(matched) < min_matches:
        for dim in RELAX_ORDER:
            candidate = _bucket_filter(hq, FEATURE_COLS, relaxed_dim=dim)
            if len(candidate) >= min_matches:
                matched     = candidate
                relaxed_dim = dim
                break
        else:
            # Use all relaxations if still insufficient
            matched = hq.copy()

    # ── Z-score normalise features for distance ───────────────────────────────
    avail_feats = [c for c in FEATURE_COLS if c in matched.columns]
    matched     = matched.copy()

    means = {c: hq[c].mean() for c in avail_feats if c in hq.columns}
    stds  = {c: max(hq[c].std(), 1e-9) for c in avail_feats if c in hq.columns}

    tgt_z = {}
    for c in avail_feats:
        val = float(target.get(c, 0) if isinstance(target, dict)
                    else getattr(target, c, 0))
        tgt_z[c] = (val - means[c]) / stds[c]

    def _dist(row):
        d = 0.0
        for c in avail_feats:
            w   = WEIGHTS.get(c, 0.25)
            z   = (row[c] - means[c]) / stds[c]
            d  += w * (z - tgt_z[c]) ** 2
        return math.sqrt(d)

    matched["_distance"] = matched.apply(_dist, axis=1)

    # ── Temporal weight: exp(-0.693 * days_ago / half_life) ──────────────────
    half_life = 365 if "USDT" in str(ticker).upper() else 730
    now_ts    = pd.Timestamp.now(tz=None).normalize()

    def _tw(idx_val):
        try:
            ts = pd.Timestamp(idx_val)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            days_ago = max(0, (now_ts - ts).days)
        except Exception:
            days_ago = 365
        return math.exp(-0.693 * days_ago / half_life)

    matched["_temporal_weight"] = [_tw(i) for i in matched.index]
    matched = matched.sort_values("_distance")

    quality_pct = min(100, round(len(matched) / max(1, len(hq)) * 100))

    meta = {
        "n":              len(matched),
        "relaxed_dim":    relaxed_dim,
        "quality_pct":    quality_pct,
        "target_buckets": target_buckets,
    }
    return matched, meta


def aggregate_outcomes(matched_candles, all_trades, validated_params=None):
    """
    Join matched candles to trade outcomes. Only TP and SL exits count.
    Apply temporal weighting. Return outcome dict.
    """
    if matched_candles.empty or all_trades.empty:
        return {"n": 0, "confidence": "INSUFFICIENT"}

    # Normalise trade index to date for joining
    trades = all_trades.copy()
    if hasattr(trades.index, "normalize"):
        trades.index = trades.index.normalize()

    # Keep only TP/SL exits
    if "exit_type" in trades.columns:
        trades = trades[trades["exit_type"].isin(["TP", "SL", "tp", "sl",
                                                   "Take Profit", "Stop Loss"])]

    if trades.empty:
        return {"n": 0, "confidence": "INSUFFICIENT"}

    # Join on date index
    mc_dates = matched_candles.index
    if hasattr(mc_dates, "normalize"):
        mc_dates = mc_dates.normalize()

    joined = trades[trades.index.isin(mc_dates)].copy()
    if joined.empty:
        return {"n": 0, "confidence": "INSUFFICIENT"}

    # Attach temporal weights
    tw_map = dict(zip(
        matched_candles.index.normalize()
        if hasattr(matched_candles.index, "normalize")
        else matched_candles.index,
        matched_candles["_temporal_weight"]
    ))
    joined["_tw"] = joined.index.map(lambda x: tw_map.get(x, 1.0))

    # r_multiple column (try common names)
    r_col = next((c for c in ["r_multiple", "r", "R", "r_mult"] if c in joined.columns), None)
    if r_col is None:
        return {"n": 0, "confidence": "INSUFFICIENT"}

    n = len(joined)

    # Wins: r > 0
    wins   = joined[joined[r_col] > 0]
    losses = joined[joined[r_col] <= 0]

    tw_total  = joined["_tw"].sum()
    tw_wins   = wins["_tw"].sum()

    win_rate_w   = tw_wins / tw_total if tw_total > 0 else 0.0
    win_rate_raw = len(wins) / n if n > 0 else 0.0

    # Wilson score interval (90% CI, z=1.645)
    z = 1.645
    p = win_rate_w
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)

    avg_win_r  = float(wins[r_col].mean())  if not wins.empty  else 0.0
    avg_loss_r = float(losses[r_col].mean()) if not losses.empty else 0.0

    gross_profit = wins[r_col].sum()
    gross_loss   = abs(losses[r_col].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    hold_col = next((c for c in ["hold_bars", "bars_held", "hold"] if c in joined.columns), None)
    avg_hold = float(joined[hold_col].mean()) if hold_col else 0.0

    # Confidence label
    if n >= 50:       confidence = "HIGH"
    elif n >= 30:     confidence = "MODERATE"
    elif n >= 20:     confidence = "LOW"
    elif n >= 10:     confidence = "VERY LOW"
    else:             confidence = "INSUFFICIENT"

    # Best params combo: (retracement, sl, tp) with highest PF, min 5 trades
    best_params = None
    param_cols  = [c for c in ["retracement", "sl", "tp"] if c in joined.columns]
    if param_cols:
        try:
            grp = joined.groupby(param_cols)
            candidates = []
            for key, g in grp:
                if len(g) < 5:
                    continue
                gw = g[g[r_col] > 0][r_col].sum()
                gl = abs(g[g[r_col] <= 0][r_col].sum())
                pf = gw / gl if gl > 0 else float("inf")
                candidates.append((pf, key))
            if candidates:
                best_params = max(candidates, key=lambda x: x[0])[1]
        except Exception:
            pass

    # Exit breakdown
    tp_count = len(wins)
    sl_count = len(losses)
    total    = tp_count + sl_count
    exit_breakdown = {
        "tp_pct": round(tp_count / total * 100, 1) if total > 0 else 0.0,
        "sl_pct": round(sl_count / total * 100, 1) if total > 0 else 0.0,
    }

    return {
        "n":              n,
        "win_rate":       win_rate_w,
        "win_rate_raw":   win_rate_raw,
        "ci_lower":       ci_lower,
        "ci_upper":       ci_upper,
        "avg_win_r":      avg_win_r,
        "avg_loss_r":     avg_loss_r,
        "profit_factor":  profit_factor,
        "avg_hold_bars":  avg_hold,
        "confidence":     confidence,
        "best_params":    best_params,
        "exit_breakdown": exit_breakdown,
    }


def generate_probability_anchor(outcomes, target_buckets, regime_zone,
                                 session, ml_score, ml_meta):
    """
    Build a multiline probability anchor string from similarity outcomes.
    Returns a plain string (no HTML) with newline separators.
    """
    import math

    n = outcomes.get("n", 0)

    if n < 10:
        return (
            "Insufficient historical data for probability estimate.\n"
            f"Found only {n} similar candles (need >= 10).\n"
            "Run more history or loosen filter settings."
        )

    confidence  = outcomes.get("confidence",    "INSUFFICIENT")
    win_rate    = outcomes.get("win_rate",       0.0)
    ci_lower    = outcomes.get("ci_lower",       0.0)
    ci_upper    = outcomes.get("ci_upper",       0.0)
    pf          = outcomes.get("profit_factor",  0.0)
    avg_loss_r  = outcomes.get("avg_loss_r",     0.0)
    avg_hold    = outcomes.get("avg_hold_bars",  0.0)
    best_params = outcomes.get("best_params",    None)

    # Bucket label mapping
    _blabel = {"Low": "moderate", "Med": "strong", "High": "exceptional"}
    body_lbl = _blabel.get(target_buckets.get("body_pct_abs", "Med"), "strong")
    vol_lbl  = _blabel.get(target_buckets.get("vol_mult",     "Med"), "strong")

    zone_str = regime_zone or "—"
    sess_str = session     or "—"

    # Line 4 — Win Rate
    wr_pct   = round(win_rate * 100, 1)
    ci_lo_p  = round(ci_lower * 100, 1)
    ci_hi_p  = round(ci_upper * 100, 1)
    pf_str   = f"{pf:.2f}" if pf < 90 else "inf"

    # Line 7 — context
    if best_params:
        if isinstance(best_params, (list, tuple)):
            ctx_line = f"Best entry params: {best_params}"
        else:
            ctx_line = f"Best entry params: {best_params}"
    else:
        ctx_line = f"Typical resolution: ~{avg_hold:.0f} bars"

    # Lines 9-10 — ML
    ml_pct      = round(float(ml_score or 0) * 100, 1)
    gap         = ml_pct - wr_pct
    if abs(gap) <= 10:
        agree = "ML confirms historical"
    elif gap > 20:
        agree = "ML and historical DISAGREE"
    elif gap > 10:
        agree = "ML more optimistic"
    else:
        agree = "ML more cautious"

    # Sigmoid-weighted combined estimate
    hist_weight = 1 / (1 + math.exp(-0.08 * (n - 30)))
    ml_weight   = 1 - hist_weight
    combined    = hist_weight * win_rate + ml_weight * float(ml_score or 0)
    combined_pct = round(combined * 100, 1)

    lines = [
        f"[{n}] similar candles found ({confidence})",
        f"Body: {body_lbl} | Volume: {vol_lbl} | Regime: {zone_str} | Session: {sess_str}",
        "",
        f"Win Rate: {wr_pct}% [90% CI: {ci_lo_p}%-{ci_hi_p}%]",
        f"Profit Factor: {pf_str}",
        f"Avg heat before resolution: {avg_loss_r:.2f}R",
        ctx_line,
        "",
        f"ML Model: {ml_pct}% probability  {agree}",
        f"Combined Estimate: {combined_pct}%",
    ]
    return "\n".join(lines)


# ─── End Similarity Engine ─────────────────────────────────────────────────────

# ─── ML Classifier ────────────────────────────────────────────────────────────

ML_FEATURES = [
    "body_pct_abs", "vol_mult", "atr_ratio",
    "adx_value", "ema_aligned", "di_gap_signed", "candle_rank_20",
    # Market-context features (Improvement 5)
    "fear_greed_normalized",   # F&G index / 100  (0–1)
    "funding_rate",            # current perp funding rate
    "oi_change_pct_24h",       # open interest 24h change %
    "vol_delta_regime",        # vol_delta_5 z-score vs 20-bar baseline
]


def build_ml_training_data(qualifying_df, all_trades, direction,
                            validated_params=None):
    """
    Build (X, y) from qualifying candles that produced a TP or SL trade.
    validated_params: optional tuple (retracement, sl_dist, tp_mode) to filter.
    Returns (X DataFrame, y numpy array).
    """
    rows   = []
    labels = []

    # Flatten all_trades to a date→trade lookup
    # all_trades may be a list of dicts, or a dict keyed by param tuple
    if isinstance(all_trades, dict):
        # keyed by (ret, sl, tp) — flatten with optional param filter
        trade_list = []
        for key, trades in all_trades.items():
            if validated_params is not None and key != validated_params:
                continue
            trade_list.extend(trades)
    else:
        trade_list = list(all_trades)

    # Keep only TP / SL exits
    _tp_labels = {"TP", "tp", "Take Profit"}
    _sl_labels = {"SL", "sl", "Stop Loss"}
    trade_list = [t for t in trade_list
                  if t.get("exit_type") in _tp_labels | _sl_labels]

    if not trade_list:
        return pd.DataFrame(), np.array([])

    # Build a lookup: entry_date → exit_type
    date_to_exit = {}
    for t in trade_list:
        ed = t.get("entry_date") or t.get("date")
        if ed is None:
            continue
        # Normalise to date
        try:
            ed = pd.Timestamp(ed).normalize()
        except Exception:
            continue
        # If multiple trades on same date, prefer TP
        if ed not in date_to_exit or t["exit_type"] in _tp_labels:
            date_to_exit[ed] = t["exit_type"]

    if not date_to_exit:
        return pd.DataFrame(), np.array([])

    # Match qualifying candles to trade exits
    qdf = qualifying_df.copy()
    if "body_pct_abs" not in qdf.columns:
        qdf["body_pct_abs"] = qdf["body_pct"].abs() if "body_pct" in qdf.columns else 0.0

    for ts in qdf.index:
        date_key = pd.Timestamp(ts).normalize()
        if date_key not in date_to_exit:
            continue

        exit_type = date_to_exit[date_key]
        label     = 1 if exit_type in _tp_labels else 0

        row_s = qdf.loc[ts]

        body_pct_abs  = float(row_s.get("body_pct_abs",  abs(float(row_s.get("body_pct", 0.7)))))
        vol_mult      = float(row_s.get("vol_mult",       1.5))
        atr_ratio     = float(row_s.get("atr_ratio",      1.0))
        adx_val       = float(row_s.get("adx_value",      20.0))
        if np.isnan(adx_val):
            adx_val = 20.0
        candle_rank   = float(row_s.get("candle_rank_20", 0.5))
        if np.isnan(candle_rank):
            candle_rank = 0.5

        ema5  = float(row_s.get("ema5",  float(row_s.get("close", 0))))
        ema15 = float(row_s.get("ema15", float(row_s.get("close", 0))))
        ema21 = float(row_s.get("ema21", float(row_s.get("close", 0))))
        if direction == "long":
            ema_aligned = 1 if (ema5 > ema15 and ema15 > ema21) else 0
        else:
            ema_aligned = 1 if (ema5 < ema15 and ema15 < ema21) else 0

        di_plus  = float(row_s.get("di_plus",  0.0))
        di_minus = float(row_s.get("di_minus", 0.0))
        if direction == "long":
            di_gap_signed = di_plus - di_minus
        else:
            di_gap_signed = di_minus - di_plus

        rows.append({
            "body_pct_abs":          body_pct_abs,
            "vol_mult":              vol_mult,
            "atr_ratio":             atr_ratio,
            "adx_value":             adx_val,
            "ema_aligned":           ema_aligned,
            "di_gap_signed":         di_gap_signed,
            "candle_rank_20":        candle_rank,
            # Market-context features — use bar-level values where stored,
            # else fall back to neutral so historical rows still contribute
            "fear_greed_normalized": float(row_s.get("fear_greed_normalized", 0.5)),
            "funding_rate":          float(row_s.get("funding_rate",          0.0)),
            "oi_change_pct_24h":     float(row_s.get("oi_change_pct_24h",     0.0)),
            "vol_delta_regime":      float(row_s.get("vol_delta_regime",       0.0)
                                          if not (isinstance(row_s.get("vol_delta_regime"), float)
                                                  and np.isnan(row_s.get("vol_delta_regime", 0.0)))
                                          else 0.0),
        })
        labels.append(label)

    if not rows:
        return pd.DataFrame(), np.array([])

    X = pd.DataFrame(rows, columns=ML_FEATURES)
    y = np.array(labels, dtype=int)
    return X, y


def train_signal_classifier(qualifying_df, all_trades, direction,
                              validated_params=None):
    """
    Train a logistic-regression signal classifier with time-series CV.
    Returns a model dict, or {"error": "..."} if insufficient data.
    """
    X, y = build_ml_training_data(qualifying_df, all_trades, direction,
                                   validated_params)

    if len(y) < 30:
        return {"error": f"Only {len(y)} samples. Need >= 30."}

    # Fill NaNs with column median before scaling (some features missing on early bars)
    X = X.fillna(X.median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv        = TimeSeriesSplit(n_splits=5)
    cv_scores   = []
    for train_idx, val_idx in tscv.split(X_scaled):
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(X_scaled[train_idx], y[train_idx])
        preds = clf.predict(X_scaled[val_idx])
        cv_scores.append(accuracy_score(y[val_idx], preds))

    cv_accuracy = float(np.mean(cv_scores))

    # Holdout: last 20%
    split_at        = int(len(X_scaled) * 0.80)
    X_ho, y_ho      = X_scaled[split_at:], y[split_at:]
    final_model     = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                          max_iter=1000)
    final_model.fit(X_scaled, y)

    holdout_accuracy = (float(accuracy_score(y_ho, final_model.predict(X_ho)))
                        if len(y_ho) > 0 else cv_accuracy)

    feature_importance = dict(zip(ML_FEATURES,
                                   final_model.coef_[0].tolist()))

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    class_balance = f"{n_pos} TP / {n_neg} SL ({n_pos/len(y)*100:.0f}% win)"

    return {
        "model":               final_model,
        "scaler":              scaler,
        "cv_accuracy":         cv_accuracy,
        "holdout_accuracy":    holdout_accuracy,
        "n_samples":           len(y),
        "class_balance":       class_balance,
        "feature_importance":  feature_importance,
        "trained_at":          datetime.now(),
    }


def score_candle_ml(candle_series, model_data, direction):
    """
    Score a single candle using the trained classifier.
    Returns dict: probability, confidence, available.
    """
    if not model_data or "model" not in model_data:
        return {"probability": 0.5, "confidence": "LOW", "available": False}

    try:
        row_s = candle_series

        body_pct_abs  = float(row_s.get("body_pct_abs",
                               abs(float(row_s.get("body_pct", 0.7)))))
        vol_mult      = float(row_s.get("vol_mult",      1.5))
        atr_ratio     = float(row_s.get("atr_ratio",     1.0))
        adx_val       = float(row_s.get("adx_value",     20.0))
        if np.isnan(adx_val):
            adx_val = 20.0
        candle_rank   = float(row_s.get("candle_rank_20", 0.5))
        if np.isnan(candle_rank):
            candle_rank = 0.5

        ema5  = float(row_s.get("ema5",  float(row_s.get("close", 0))))
        ema15 = float(row_s.get("ema15", float(row_s.get("close", 0))))
        ema21 = float(row_s.get("ema21", float(row_s.get("close", 0))))
        if direction == "long":
            ema_aligned = 1 if (ema5 > ema15 and ema15 > ema21) else 0
        else:
            ema_aligned = 1 if (ema5 < ema15 and ema15 < ema21) else 0

        di_plus  = float(row_s.get("di_plus",  0.0))
        di_minus = float(row_s.get("di_minus", 0.0))
        di_gap_signed = (di_plus - di_minus) if direction == "long" else (di_minus - di_plus)

        feat_vec = np.array([[body_pct_abs, vol_mult, atr_ratio, adx_val,
                               ema_aligned, di_gap_signed, candle_rank]])
        # Replace any NaN with 0 before scaling (safe neutral value after StandardScaler)
        feat_vec = np.where(np.isnan(feat_vec), 0.0, feat_vec)
        feat_scaled = model_data["scaler"].transform(feat_vec)
        prob        = float(model_data["model"].predict_proba(feat_scaled)[0][1])

        ho_acc = model_data.get("holdout_accuracy", 0.5)
        if ho_acc >= 0.65:
            conf = "HIGH"
        elif ho_acc >= 0.55:
            conf = "MODERATE"
        else:
            conf = "LOW"

        return {"probability": prob, "confidence": conf, "available": True}

    except Exception:
        return {"probability": 0.5, "confidence": "LOW", "available": False}


# ─── WFO Pipeline ─────────────────────────────────────────────────────────────

def run_wfo_pipeline(df, qualifying_func, adx_df, timeframe, direction,
                     transaction_cost=0.001):
    """
    Walk-forward optimisation.  For each rolling IS/OOS window:
      1. Find best params on IS (highest PF, >=5 trades).
      2. Apply best params to OOS.
    Returns dict with cycles, summary, best_params, validated_at, timeframe.
    """
    _WFO_WINDOWS = {
        "1D": {"is_bars": 252,  "oos_bars": 63},
        "4H": {"is_bars": 1512, "oos_bars": 378},
        "1H": {"is_bars": 5040, "oos_bars": 1260},
    }
    cfg      = _WFO_WINDOWS.get(timeframe, _WFO_WINDOWS["1D"])
    is_bars  = cfg["is_bars"]
    oos_bars = cfg["oos_bars"]
    step     = oos_bars   # non-overlapping

    # WFO-specific param grid (smaller than full optimiser)
    _BODY_PCTS   = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    _VOL_MULTS   = [1.5, 2.0, 2.5, 3.0]
    _RETS        = [0.0, 0.236, 0.382, 0.50, 0.618]
    _SL_DISTS    = [0.008, 0.010, 0.015, 0.020, 0.025, 0.030]
    _TP_MODES    = ["2R", "3R", "Partial"]

    n_total = len(df)
    starts  = list(range(0, n_total - is_bars - oos_bars + 1, step))

    if not starts:
        return {
            "cycles":       [],
            "summary":      {"verdict": "FAIL", "reason": "Insufficient data for WFO"},
            "best_params":  None,
            "validated_at": datetime.now(),
            "timeframe":    timeframe,
        }

    cycles      = []
    prog_bar    = st.progress(0, text="Running Walk-Forward Optimisation...")

    for cycle_i, start in enumerate(starts):
        is_end   = start + is_bars
        oos_end  = min(is_end + oos_bars, n_total)

        df_is    = df.iloc[start:is_end]
        df_oos   = df.iloc[is_end:oos_end]

        if df_is.empty or df_oos.empty:
            continue

        # ── IS optimisation ───────────────────────────────────────────────────
        best_is_pf     = -1.0
        best_is_params = None
        best_is_trades = 0

        for body_p, vol_m in product(_BODY_PCTS, _VOL_MULTS):
            qual_is = qualifying_func(
                df_is, min_body_pct=body_p, min_vol_mult=vol_m,
                adx_filter=False, direction=direction,
            )
            if qual_is.empty:
                continue
            for ret, sl, tp in product(_RETS, _SL_DISTS, _TP_MODES):
                trades_is = run_backtest(
                    df_is, qual_is, ret, sl, tp,
                    sl_mode="Fixed", trail_atr_mult=1.5,
                    transaction_cost=transaction_cost,
                    max_hold_bars=9999, use_time_exit=False,
                    direction=direction,
                )
                m = calc_metrics(trades_is)
                if m and m["total_trades"] >= 5 and m["profit_factor"] > best_is_pf:
                    best_is_pf     = m["profit_factor"]
                    best_is_params = {
                        "body_pct":    body_p,
                        "vol_mult":    vol_m,
                        "retracement": ret,
                        "sl_dist":     sl,
                        "tp_mode":     tp,
                    }
                    best_is_trades = m["total_trades"]

        if best_is_params is None:
            continue

        # ── OOS evaluation ────────────────────────────────────────────────────
        qual_oos = qualifying_func(
            df_oos,
            min_body_pct=best_is_params["body_pct"],
            min_vol_mult=best_is_params["vol_mult"],
            adx_filter=False,
            direction=direction,
        )
        trades_oos = run_backtest(
            df_oos, qual_oos,
            best_is_params["retracement"],
            best_is_params["sl_dist"],
            best_is_params["tp_mode"],
            sl_mode="Fixed", trail_atr_mult=1.5,
            transaction_cost=transaction_cost,
            max_hold_bars=9999, use_time_exit=False,
            direction=direction,
        ) if not qual_oos.empty else []

        m_oos      = calc_metrics(trades_oos) if trades_oos else {}
        oos_pf     = float(m_oos.get("profit_factor", 0.0)) if m_oos else 0.0
        oos_wr     = float(m_oos.get("win_rate",       0.0)) if m_oos else 0.0
        oos_trades = int(m_oos.get("total_trades",     0))   if m_oos else 0
        ratio      = oos_pf / best_is_pf if best_is_pf > 0 else 0.0

        cycles.append({
            "cycle":         cycle_i + 1,
            "is_start":      str(df_is.index[0])[:10],
            "is_end":        str(df_is.index[-1])[:10],
            "oos_start":     str(df_oos.index[0])[:10],
            "oos_end":       str(df_oos.index[-1])[:10],
            "best_params":   best_is_params,
            "is_pf":         round(best_is_pf, 3),
            "oos_pf":        round(oos_pf, 3),
            "oos_is_ratio":  round(ratio, 3),
            "oos_trades":    oos_trades,
            "oos_wr":        round(oos_wr * 100, 1),
        })

        prog_bar.progress(
            min(1.0, (cycle_i + 1) / max(1, len(starts))),
            text=f"WFO cycle {cycle_i + 1}/{len(starts)} — OOS PF {oos_pf:.2f}",
        )

    prog_bar.empty()

    if not cycles:
        return {
            "cycles":       [],
            "summary":      {"verdict": "FAIL", "reason": "No valid cycles produced"},
            "best_params":  None,
            "validated_at": datetime.now(),
            "timeframe":    timeframe,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    oos_pfs         = [c["oos_pf"]       for c in cycles]
    ratios          = [c["oos_is_ratio"] for c in cycles]
    avg_oos_pf      = float(np.mean(oos_pfs))
    pct_profitable  = sum(1 for p in oos_pfs if p > 1.0) / len(oos_pfs) * 100
    ratio_avg       = float(np.mean(ratios))

    verdict = ("PASS"
               if avg_oos_pf > 1.0 and ratio_avg > 0.65 and len(cycles) >= 6
               else "FAIL")

    # Most common best_params across cycles (by serialised key)
    from collections import Counter
    param_counts = Counter(
        str(sorted(c["best_params"].items())) for c in cycles
    )
    most_common_str = param_counts.most_common(1)[0][0]
    # Recover the actual dict from cycles
    best_params = next(
        c["best_params"] for c in cycles
        if str(sorted(c["best_params"].items())) == most_common_str
    )

    summary = {
        "verdict":              verdict,
        "n_cycles":             len(cycles),
        "avg_oos_pf":           round(avg_oos_pf, 3),
        "pct_profitable_cycles": round(pct_profitable, 1),
        "oos_is_ratio_avg":     round(ratio_avg, 3),
    }

    return {
        "cycles":       cycles,
        "summary":      summary,
        "best_params":  best_params,
        "validated_at": datetime.now(),
        "timeframe":    timeframe,
    }


# ─── End ML / WFO ─────────────────────────────────────────────────────────────

# ─── AI Analysis ──────────────────────────────────────────────────────────────

def build_analysis_prompt(candle, regime, anchor_text, ml_score,
                           strategy_params, wfo_results, direction,
                           ticker, timeframe, risk_context):
    """
    Build a structured user prompt for the AI analysis endpoint.
    Returns a plain string with 5 labelled sections.
    """
    # ── Section A — Regime Context ────────────────────────────────────────────
    verdict_str   = regime.get("verdict",        "UNKNOWN") if regime else "UNKNOWN"
    score_str     = str(regime.get("score",      0))        if regime else "0"
    breakdown_str = regime.get("breakdown_line", "—")       if regime else "—"
    overrides     = regime.get("hard_overrides", [])        if regime else []
    override_str  = "; ".join(overrides) if overrides else "None"

    section_a = (
        f"=== A. REGIME CONTEXT ===\n"
        f"Verdict: {verdict_str}  Score: {score_str}/100\n"
        f"Breakdown: {breakdown_str}\n"
        f"Hard overrides: {override_str}"
    )

    # ── Section B — Candle ────────────────────────────────────────────────────
    def _fget(key, default=0.0):
        try:
            v = candle.get(key, default) if isinstance(candle, dict) else getattr(candle, key, default)
            return float(v) if v is not None and str(v) != "nan" else default
        except Exception:
            return default

    body_pct    = _fget("body_pct",    0.0)
    vol_mult    = _fget("vol_mult",    1.0)
    atr_ratio   = _fget("atr_ratio",  1.0)
    adx_val     = _fget("adx_value",  0.0)
    rank_20     = _fget("candle_rank_20", 0.5)
    vol_rank    = _fget("vol_rank_20",    0.5)
    vol_delta5  = _fget("vol_delta_5",    0.0)
    ema5        = _fget("ema5",  0.0)
    ema15       = _fget("ema15", 0.0)
    ema21       = _fget("ema21", 0.0)
    di_plus     = _fget("di_plus",  0.0)
    di_minus    = _fget("di_minus", 0.0)

    # EMA stack alignment
    if ema5 > 0 and ema15 > 0 and ema21 > 0:
        if ema5 > ema15 > ema21:
            ema_stack = "BULLISH (EMA5>EMA15>EMA21)"
        elif ema5 < ema15 < ema21:
            ema_stack = "BEARISH (EMA5<EMA15<EMA21)"
        else:
            ema_stack = "MIXED (no clean stack)"
    else:
        ema_stack = "N/A"

    # Vol delta interpretation
    vol_delta_str = (f"+{vol_delta5:.0f} (net buying pressure)"
                     if vol_delta5 > 0 else
                     f"{vol_delta5:.0f} (net selling pressure)")

    # DI alignment
    di_gap = abs(di_plus - di_minus)
    if di_plus > di_minus:
        di_str = f"DI+ {di_plus:.1f} > DI- {di_minus:.1f} | gap {di_gap:.1f} (bullish)"
    elif di_minus > di_plus:
        di_str = f"DI- {di_minus:.1f} > DI+ {di_plus:.1f} | gap {di_gap:.1f} (bearish)"
    else:
        di_str = f"DI neutral (gap {di_gap:.1f})"

    try:
        if isinstance(candle, dict):
            bar_date = candle.get("_date_str") or "—"
        else:
            bar_date = str(candle.name)[:10] if hasattr(candle, "name") else "—"
    except Exception:
        bar_date = "—"

    section_b = (
        f"=== B. CANDLE ===\n"
        f"Ticker: {ticker}  TF: {timeframe}  Direction: {direction.upper()}  Date: {bar_date}\n"
        f"Body: {abs(body_pct)*100:.1f}%  Vol mult: {vol_mult:.2f}x  "
        f"ATR ratio: {atr_ratio:.2f}  ADX: {adx_val:.1f}  Candle rank(20): {rank_20:.2f}\n"
        f"EMA stack: {ema_stack}\n"
        f"Vol delta 5-bar: {vol_delta_str}  Vol rank(20): {vol_rank:.2f}\n"
        f"DI alignment: {di_str}"
    )

    # ── Section C — Evidence ──────────────────────────────────────────────────
    ml_pct_str = f"{float(ml_score or 0)*100:.1f}%" if ml_score is not None else "N/A"
    section_c = (
        f"=== C. EVIDENCE ===\n"
        f"ML probability: {ml_pct_str}\n"
        f"{anchor_text if anchor_text else 'No similarity data available.'}"
    )

    # ── Section D — Parameters ────────────────────────────────────────────────
    wfo_status = "UNVALIDATED"
    wfo_detail = ""
    if wfo_results and isinstance(wfo_results, dict):
        summary = wfo_results.get("summary", {})
        wfo_status = summary.get("verdict", "UNVALIDATED")
        bp         = wfo_results.get("best_params", {})
        if bp:
            wfo_detail = (
                f"  Best params: body>={bp.get('body_pct','?')} "
                f"vol>={bp.get('vol_mult','?')} "
                f"ret={bp.get('retracement','?')} "
                f"SL={bp.get('sl_dist','?')} "
                f"TP={bp.get('tp_mode','?')}"
            )
        avg_pf = summary.get("avg_oos_pf", 0)
        n_cycles = summary.get("n_cycles", 0)
        wfo_detail += f"\n  OOS avg PF: {avg_pf:.2f}  Cycles: {n_cycles}"

    sp = strategy_params or {}
    ret   = float(sp.get("retracement", 0.0) or 0.0)
    sl_d  = float(sp.get("sl_dist",     0.01) or 0.01)
    tp_m  = str(sp.get("tp_mode", "2R"))
    wr_sp = float(sp.get("win_rate", 0.0) or 0.0)
    pf_sp = float(sp.get("pf",      0.0) or 0.0)

    # Compute actual price levels from close price
    close_for_entry = _fget("close", 0.0)
    if close_for_entry > 0:
        body_val = _fget("body", 0.0)
        if direction == "long":
            entry_px  = round(close_for_entry - abs(body_val) * ret, 6)
            sl_px     = round(entry_px * (1 - sl_d), 6)
            risk_amt  = entry_px - sl_px
        else:
            entry_px  = round(close_for_entry + abs(body_val) * ret, 6)
            sl_px     = round(entry_px * (1 + sl_d), 6)
            risk_amt  = sl_px - entry_px
        tp1_px = round(entry_px + (1 * risk_amt if direction == "long" else -1 * risk_amt), 6)
        tp2_px = round(entry_px + (2 * risk_amt if direction == "long" else -2 * risk_amt), 6)
        tp3_px = round(entry_px + (3 * risk_amt if direction == "long" else -3 * risk_amt), 6)
        if tp_m == "1R":
            tp_str = f"TP={tp1_px} (1R)"
        elif tp_m == "3R":
            tp_str = f"TP={tp3_px} (3R)"
        else:
            tp_str = f"TP1={tp1_px} (1R partial), TP2={tp2_px} (2R full)"
        ret_label = "Immediate (0% retrace — enter at close)" if ret == 0.0 else f"{ret*100:.0f}% retrace from close body"
        entry_detail = (
            f"  Entry method: {ret_label}\n"
            f"  Entry price: {entry_px}  SL: {sl_px} ({sl_d*100:.1f}% distance)  {tp_str}\n"
            f"  TP mode: {tp_m}  Historical win rate: {wr_sp:.1f}%  PF: {pf_sp:.2f}"
        )
    else:
        ret_label = "Immediate (0%)" if ret == 0.0 else f"{ret*100:.0f}% retrace"
        entry_detail = (
            f"  Entry: {ret_label}  SL dist: {sl_d*100:.1f}%  TP mode: {tp_m}\n"
            f"  Historical win rate: {wr_sp:.1f}%  PF: {pf_sp:.2f}"
        )

    section_d = (
        f"=== D. PARAMETERS (from backtester — use these for entry) ===\n"
        f"WFO status: {wfo_status}{wfo_detail}\n"
        f"Backtested params:{entry_detail}"
    )

    # ── Section E — Risk ──────────────────────────────────────────────────────
    rc             = risk_context or {}
    consec_losses  = rc.get("consecutive_losses", 0)
    trades_today   = rc.get("trades_today",       0)

    section_e = (
        f"=== E. RISK ===\n"
        f"Consecutive losses: {consec_losses}  "
        f"Trades today: {trades_today}"
    )

    return "\n\n".join([section_a, section_b, section_c, section_d, section_e])


def rule_based_fallback(regime, outcomes, ml_score):
    """
    Simple rule-based verdict when the AI API is unavailable.
    Returns dict: verdict, full_text, source='fallback'.
    """
    import math

    verdict_label = regime.get("verdict", "RED") if regime else "RED"
    score         = regime.get("score",   0)     if regime else 0
    ml            = float(ml_score or 0)
    n             = outcomes.get("n", 0)          if outcomes else 0
    win_rate      = outcomes.get("win_rate", 0.0) if outcomes else 0.0

    # Sigmoid-weighted combined (same as generate_probability_anchor)
    hist_weight = 1 / (1 + math.exp(-0.08 * (n - 30)))
    ml_weight   = 1 - hist_weight
    combined    = hist_weight * win_rate + ml_weight * ml

    # Hard overrides
    if verdict_label == "RED" or score < 45:
        verdict = "NO TRADE"
        reason  = f"Regime {verdict_label} (score {score}/100) — waiting for better conditions."
    elif ml < 0.40:
        verdict = "NO TRADE"
        reason  = f"ML probability {ml*100:.0f}% is below 40% threshold."
    elif combined >= 0.60 and verdict_label == "GREEN":
        verdict = "TRADE"
        reason  = f"Combined estimate {combined*100:.0f}% with GREEN regime — proceed full size."
    elif combined >= 0.65 and verdict_label == "YELLOW":
        verdict = "TRADE"
        reason  = f"Combined estimate {combined*100:.0f}% with YELLOW regime — half size."
    elif combined >= 0.50 and verdict_label == "GREEN":
        verdict = "WAIT"
        reason  = f"Combined estimate {combined*100:.0f}% — borderline, wait for confirmation."
    elif combined < 0.50:
        verdict = "NO TRADE"
        reason  = f"Combined estimate {combined*100:.0f}% below 50% — edge insufficient."
    else:
        verdict = "NO TRADE"
        reason  = "Insufficient evidence to justify entry."

    full_text = (
        f"[Rule-based fallback — AI unavailable]\n\n"
        f"VERDICT: **{verdict}** — {reason}\n\n"
        f"Regime: {verdict_label} ({score}/100)  "
        f"ML: {ml*100:.0f}%  "
        f"Combined: {combined*100:.0f}%  "
        f"N: {n}"
    )

    return {"verdict": verdict, "full_text": full_text, "source": "fallback"}


def analyze_candle_ai(candle, regime, anchor_text, ml_score,
                      strategy_params, wfo_results, direction,
                      ticker, timeframe, risk_context):
    """
    Call AI API (Anthropic or Groq) for a structured trade decision.
    Provider is selected from session_state["ai_provider"].
    Falls back to rule_based_fallback on missing key or any API error.
    Returns dict: verdict, full_text, source, error (if failed).
    """
    import os

    outcomes = None  # used by fallback

    # ── Resolve provider and key from sidebar session_state ──────────────────
    provider  = st.session_state.get("ai_provider", "Groq (Free)")
    use_groq  = "Groq" in provider

    if use_groq:
        api_key = st.session_state.get("groq_api_key", "")
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY", "")
    else:
        api_key = st.session_state.get("anthropic_api_key", "")
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                try:
                    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                except Exception:
                    api_key = ""

    if not api_key:
        fb = rule_based_fallback(regime, outcomes, ml_score)
        fb["error"] = "No API key — set key in sidebar"
        return fb

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = build_analysis_prompt(
        candle, regime, anchor_text, ml_score,
        strategy_params, wfo_results, direction,
        ticker, timeframe, risk_context,
    )

    # ── Call API ──────────────────────────────────────────────────────────────
    error_detail = ""
    raw_text     = ""
    try:
        if use_groq:
            # Groq uses OpenAI-compatible /chat/completions format
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model":       "llama-3.3-70b-versatile",
                    "max_tokens":  900,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                },
                timeout=20,
            )
            resp.raise_for_status()
            data     = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            source   = "groq/llama-3.3-70b"
        else:
            # Anthropic Messages API
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model":       "claude-3-5-sonnet-20241022",
                    "max_tokens":  900,
                    "temperature": 0.1,
                    "system":      ANALYSIS_SYSTEM_PROMPT,
                    "messages":    [{"role": "user", "content": prompt}],
                },
                timeout=20,
            )
            resp.raise_for_status()
            data     = resp.json()
            raw_text = data["content"][0]["text"]
            source   = "anthropic/claude-3.5-sonnet"

    except Exception as exc:
        error_detail = str(exc)
        # Try to extract HTTP error body for better diagnostics
        try:
            error_detail = f"{exc} | {resp.text[:300]}"
        except Exception:
            pass
        fb = rule_based_fallback(regime, outcomes, ml_score)
        fb["error"] = error_detail
        return fb

    # ── Detect verdict ────────────────────────────────────────────────────────
    if "**NO TRADE**" in raw_text:
        verdict = "NO TRADE"
    elif "**WAIT**" in raw_text:
        verdict = "WAIT"
    elif "**TRADE**" in raw_text:
        verdict = "TRADE"
    else:
        verdict = "WAIT"   # safe default if format is unexpected

    return {"verdict": verdict, "full_text": raw_text, "source": source}


# ─── End AI Analysis ──────────────────────────────────────────────────────────

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

def detect_qualifying_candles(df, min_body_pct=0.70, min_vol_mult=1.5,
                               adx_filter=False, adx_threshold=25,
                               adx_series: pd.Series = None,
                               di_plus_series: pd.Series = None,
                               di_minus_series: pd.Series = None,
                               di_gap_min: float = 0.0,
                               direction: str = "long",
                               ema_filter: bool = False,
                               ema_series: pd.Series = None):
    """Return qualifying momentum candles.
    When adx_filter=True: requires ADX >= threshold AND DI gap >= di_gap_min
    AND DI aligned with direction (DI+ > DI- for long, DI- > DI+ for short).
    di_gap_min=0 disables the gap/alignment filter even when ADX is ON.
    """
    if direction == "short":
        mask = (
            (df["body"] < 0) &
            (df["body_pct"].abs() >= min_body_pct) &
            (df["vol_mult"] >= min_vol_mult)
        )
    else:
        mask = (
            (df["body"] > 0) &
            (df["body_pct"] >= min_body_pct) &
            (df["vol_mult"] >= min_vol_mult)
        )

    if adx_filter and adx_series is not None:
        adx_aligned = adx_series.reindex(df.index, method="ffill")
        mask = mask & (adx_aligned >= adx_threshold)

        # DI gap + alignment filter (only if gap threshold > 0 and series provided)
        if di_gap_min > 0 and di_plus_series is not None and di_minus_series is not None:
            dip = di_plus_series.reindex(df.index, method="ffill")
            dim = di_minus_series.reindex(df.index, method="ffill")
            gap = (dip - dim).abs()
            if direction == "long":
                # DI+ must lead DI- by at least di_gap_min
                mask = mask & (dip > dim) & (gap >= di_gap_min)
            else:
                # DI- must lead DI+ by at least di_gap_min
                mask = mask & (dim > dip) & (gap >= di_gap_min)

    # ── EMA Trend Filter
    # Only take LONG signals when close > EMA (price above trend)
    # Only take SHORT signals when close < EMA (price below trend)
    # ema_series is already shift(1) — no lookahead bias
    if ema_filter and ema_series is not None:
        ema_aligned = ema_series.reindex(df.index, method="ffill")
        if direction == "long":
            mask = mask & (df["close"] > ema_aligned)
        else:
            mask = mask & (df["close"] < ema_aligned)

    result = df.loc[mask].copy()
    if adx_filter and adx_series is not None:
        adx_aligned_full = adx_series.reindex(df.index, method="ffill")
        result["adx_value"] = adx_aligned_full.loc[mask].round(1)
    else:
        result["adx_value"] = float("nan")
    return result

# ─── Backtest Engine ───────────────────────────────────────────────────────────

def simulate_trade(
    trigger_idx: int,
    df: pd.DataFrame,
    retracement: float,
    sl_dist: float,
    tp_mode: str,
    sl_mode: str = "Fixed",
    trail_atr_mult: float = 1.5,
    transaction_cost: float = 0.001,
    max_hold_bars: int = 5,
    use_time_exit: bool = False,
    direction: str = "long",
) -> Optional[Dict]:
    trigger  = df.iloc[trigger_idx]
    body_abs = abs(float(trigger["body"]))
    # Session detection for advanced stats (WIB = UTC+7)
    _ts_wib  = df.index[trigger_idx] + WIB_OFFSET
    _session = get_session(_ts_wib.hour)

    # ── Entry / SL / TP prices ──────────────────────────────────────────────
    if direction == "short":
        # Entry: retrace UP from close of bearish candle
        entry_price = float(trigger["close"]) + body_abs * retracement
        sl_price    = entry_price * (1 + sl_dist)   # SL ABOVE entry
        risk        = sl_price - entry_price
        if risk <= 0:
            return None
        if tp_mode == "2R":
            tp1 = entry_price - 2 * risk; tp2 = None
        elif tp_mode == "3R":
            tp1 = entry_price - 3 * risk; tp2 = None
        else:
            tp1 = entry_price - 1 * risk
            tp2 = entry_price - 2 * risk
    else:  # long
        # Entry: retrace DOWN from close of bullish candle
        entry_price = float(trigger["close"]) - float(trigger["body"]) * retracement
        sl_price    = entry_price * (1 - sl_dist)   # SL BELOW entry
        risk        = entry_price - sl_price
        if risk <= 0:
            return None
        if tp_mode == "2R":
            tp1 = entry_price + 2 * risk; tp2 = None
        elif tp_mode == "3R":
            tp1 = entry_price + 3 * risk; tp2 = None
        else:
            tp1 = entry_price + 1 * risk
            tp2 = entry_price + 2 * risk

    future_start     = trigger_idx + 1
    ENTRY_SEARCH_BARS = 30
    future_end       = min(future_start + ENTRY_SEARCH_BARS, len(df))
    if future_start >= len(df):
        return None

    if retracement == 0.0:
        entry_filled = True
        entry_date   = df.index[trigger_idx]
        entry_bar_i  = trigger_idx
        future_end   = min(trigger_idx + 1 + max_hold_bars, len(df))
        current_sl   = sl_price
        be_moved     = False
        scan_start   = future_start
    else:
        entry_filled = False
        entry_date   = None
        entry_bar_i  = trigger_idx
        current_sl   = sl_price
        be_moved     = False
        scan_start   = future_start

    partial_exited     = False
    partial_exit_price = None

    for fwd_i in range(scan_start, future_end):
        bar      = df.iloc[fwd_i]
        bar_date = df.index[fwd_i]

        # ── Wait for entry fill ──────────────────────────────────────────────
        if not entry_filled:
            fill_cond = bar["high"] >= entry_price if direction == "short" else bar["low"] <= entry_price
            if fill_cond:
                entry_filled = True
                entry_date   = bar_date
                entry_bar_i  = fwd_i
                future_end   = min(entry_bar_i + 1 + max_hold_bars, len(df))
                # Check if SL also hit on the same fill bar
                sl_same_bar = bar["high"] >= current_sl if direction == "short" else bar["low"] <= current_sl
                if sl_same_bar:
                    if direction == "short":
                        r = (entry_price - current_sl) / risk - transaction_cost * 2
                    else:
                        r = (current_sl - entry_price) / risk - transaction_cost * 2
                    result_sb = "Win" if r > 0 else "Loss"
                    return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                       entry_date, sl_price, tp1, current_sl,
                                       bar_date, result_sb, r, retracement,
                                       sl_dist, tp_mode, sl_mode,
                                       exit_type="SL", candles_held=0,
                                       session=_session)
            continue

        # ── Time exit ────────────────────────────────────────────────────────
        if use_time_exit and fwd_i == future_end - 1:
            ep = float(bar["open"])
            # Gap check: if open already crossed SL or TP
            if direction == "short":
                sl_gapped = ep >= current_sl
                tp_gapped = ep <= tp1
            else:
                sl_gapped = ep <= current_sl
                tp_gapped = ep >= tp1

            if sl_gapped:
                if direction == "short":
                    r = (entry_price - current_sl) / risk - transaction_cost * 2
                else:
                    r = (current_sl - entry_price) / risk - transaction_cost * 2
                result_sg = "Win" if r > 0 else "Loss"
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp1, current_sl,
                                   bar_date, result_sg, r, retracement,
                                   sl_dist, tp_mode, sl_mode,
                                   exit_type="SL", candles_held=fwd_i - entry_bar_i,
                                       session=_session)
            if tp_gapped:
                r = (tp1 - entry_price) / risk * (-1 if direction == "short" else 1) - transaction_cost * 2
                r = abs(r)  # TP always positive
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp1, tp1,
                                   bar_date, "Win", r, retracement,
                                   sl_dist, tp_mode, sl_mode,
                                   exit_type="TP", candles_held=fwd_i - entry_bar_i,
                                       session=_session)

            if tp_mode == "Partial" and partial_exited:
                avg = (partial_exit_price + ep) / 2
                r   = (entry_price - avg) / risk - transaction_cost * 2 if direction == "short" \
                      else (avg - entry_price) / risk - transaction_cost * 2
                ep  = avg
                res = "Partial Win" if r > 0 else "Loss"
            else:
                r   = (entry_price - ep) / risk - transaction_cost * 2 if direction == "short" \
                      else (ep - entry_price) / risk - transaction_cost * 2
                res = "Win" if r > 0 else "Loss"
            return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                               entry_date, sl_price, tp1, ep,
                               bar_date, res, r, retracement,
                               sl_dist, tp_mode, sl_mode,
                               exit_type="Time Exit", candles_held=fwd_i - entry_bar_i,
                                       session=_session)

        # ── Trailing / breakeven SL updates ─────────────────────────────────
        if direction == "short":
            if sl_mode in ("Breakeven", "Breakeven+Trail"):
                if not be_moved and bar["low"] <= entry_price - risk:
                    current_sl = entry_price
                    be_moved   = True
            if sl_mode in ("Trailing ATR", "Breakeven+Trail"):
                atr = float(bar.get("atr14", 0) or 0)
                if atr > 0:
                    trail_level = bar["close"] + trail_atr_mult * atr
                    if sl_mode == "Trailing ATR":
                        current_sl = min(current_sl, trail_level)
                    elif be_moved:
                        current_sl = min(current_sl, trail_level)
        else:
            if sl_mode in ("Breakeven", "Breakeven+Trail"):
                if not be_moved and bar["high"] >= entry_price + risk:
                    current_sl = entry_price
                    be_moved   = True
            if sl_mode in ("Trailing ATR", "Breakeven+Trail"):
                atr = float(bar.get("atr14", 0) or 0)
                if atr > 0:
                    trail_level = bar["close"] - trail_atr_mult * atr
                    if sl_mode == "Trailing ATR":
                        current_sl = max(current_sl, trail_level)
                    elif be_moved:
                        current_sl = max(current_sl, trail_level)

        # ── SL check ─────────────────────────────────────────────────────────
        sl_hit = bar["high"] >= current_sl if direction == "short" else bar["low"] <= current_sl
        if sl_hit:
            # Use actual current_sl (may be breakeven or trailed) for accurate R calc
            if direction == "short":
                r = (entry_price - current_sl) / risk - transaction_cost * 2
            else:
                r = (current_sl - entry_price) / risk - transaction_cost * 2
            result = "Win" if r > 0 else "Loss"
            return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                               entry_date, sl_price, tp1, current_sl,
                               bar_date, result, r, retracement,
                               sl_dist, tp_mode, sl_mode,
                               exit_type="SL", candles_held=fwd_i - entry_bar_i,
                                       session=_session)

        # ── TP checks ────────────────────────────────────────────────────────
        if tp_mode == "Partial":
            tp1_hit = bar["low"] <= tp1 if direction == "short" else bar["high"] >= tp1
            if not partial_exited and tp1_hit:
                partial_exited     = True
                partial_exit_price = tp1
                if sl_mode in ("Breakeven", "Breakeven+Trail"):
                    current_sl = entry_price
            tp2_hit = bar["low"] <= tp2 if direction == "short" else bar["high"] >= tp2
            if partial_exited and tp2_hit:
                avg = (partial_exit_price + tp2) / 2
                r   = (entry_price - avg) / risk - transaction_cost * 2 if direction == "short" \
                      else (avg - entry_price) / risk - transaction_cost * 2
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp2, avg,
                                   bar_date, "Partial Win", r, retracement,
                                   sl_dist, tp_mode, sl_mode,
                                   exit_type="TP", candles_held=fwd_i - entry_bar_i,
                                       session=_session)
        else:
            tp1_hit = bar["low"] <= tp1 if direction == "short" else bar["high"] >= tp1
            if tp1_hit:
                r = abs((tp1 - entry_price) / risk) - transaction_cost * 2
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp1, tp1,
                                   bar_date, "Win", r, retracement,
                                   sl_dist, tp_mode, sl_mode,
                                   exit_type="TP", candles_held=fwd_i - entry_bar_i,
                                       session=_session)

    if not entry_filled:
        return None

    last_i    = min(future_end - 1, len(df) - 1)
    last_bar  = df.iloc[last_i]
    last_date = df.index[last_i]

    # Truly unlimited mode (sentinel ≥ 9999): only TP/SL exits count.
    # Do NOT produce a Max Hold exit — return None so this trade is excluded
    # from the optimization rather than booking an artificial close-price profit.
    if max_hold_bars >= 9999 and not use_time_exit:
        return None

    if use_time_exit:
        exit_price = float(last_bar["open"])
        exit_type  = "Time Exit"
    else:
        exit_price = float(last_bar["close"])
        exit_type  = "Max Hold"

    candles_held = last_i - entry_bar_i

    if tp_mode == "Partial" and partial_exited:
        avg        = (partial_exit_price + exit_price) / 2
        r          = (entry_price - avg) / risk - transaction_cost * 2 if direction == "short" \
                     else (avg - entry_price) / risk - transaction_cost * 2
        exit_price = avg
        result     = "Partial Win" if r > 0 else "Loss"
    else:
        r      = (entry_price - exit_price) / risk - transaction_cost * 2 if direction == "short" \
                 else (exit_price - entry_price) / risk - transaction_cost * 2
        result = "Win" if r > 0 else "Loss"

    return _trade_dict(trigger, df.index[trigger_idx], entry_price, entry_date,
                       sl_price, tp1, exit_price, last_date, result, r,
                       retracement, sl_dist, tp_mode, sl_mode,
                       exit_type=exit_type, candles_held=candles_held,
                       session=_session)


def _trade_dict(trigger, trigger_date, entry_price, entry_date,
                sl_price, tp_price, exit_price, exit_date,
                result, r_mult, retracement, sl_dist, tp_mode, sl_mode,
                exit_type: str = "Max Hold", candles_held: int = 0,
                session: str = "") -> Dict:
    hold = (exit_date - entry_date).days if entry_date else 0
    return {
        "trigger_date":     trigger_date,
        "trigger_close":    round(float(trigger["close"]),    6),
        "trigger_body_pct": round(float(trigger["body_pct"]) * 100, 1),
        "trigger_vol_mult": round(float(trigger["vol_mult"]), 2),
        "retracement":      retracement,
        "sl_dist":          sl_dist,
        "tp_mode":          tp_mode,
        "sl_mode":          sl_mode,
        "entry_price":      round(float(entry_price),  6),
        "entry_date":       entry_date,
        "sl_price":         round(float(sl_price),     6),
        "tp_price":         round(float(tp_price),     6),
        "exit_price":       round(float(exit_price),   6),
        "exit_date":        exit_date,
        "result":           result,
        "r_mult":           round(float(r_mult),       3),
        "hold_days":        hold,
        "candles_held":     candles_held,
        "exit_type":        exit_type,
        "session":          session,
        "adx_value":        round(float(trigger.get("adx_value", float("nan"))), 1)
                            if "adx_value" in trigger.index else float("nan"),
    }


def run_backtest(df, qualifying, retracement, sl_dist, tp_mode,
                 sl_mode="Fixed", trail_atr_mult=1.5,
                 transaction_cost=0.001,
                 max_hold_bars=5, use_time_exit=False,
                 direction="long") -> List[Dict]:
    trades      = []
    date_index  = pd.Series(range(len(df)), index=df.index)
    for date in qualifying.index:
        if date not in date_index.index:
            continue
        idx   = int(date_index[date])
        trade = simulate_trade(idx, df, retracement, sl_dist, tp_mode,
                               sl_mode, trail_atr_mult, transaction_cost,
                               max_hold_bars, use_time_exit, direction)
        if trade:
            trades.append(trade)
    trades.sort(key=lambda t: t["entry_date"] if t["entry_date"] else datetime.min)
    return trades


def calc_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {}
    r      = np.array([t["r_mult"] for t in trades])
    wins   = r[r > 0]
    losses = r[r <= 0]
    total  = len(trades)
    gp     = wins.sum()        if len(wins)   else 0.0
    gl     = abs(losses.sum()) if len(losses) else 0.0
    pf     = gp / gl if gl > 0 else np.inf
    equity = np.cumprod(1 + r * 0.02)
    rm     = np.maximum.accumulate(equity)
    dd     = (equity - rm) / rm
    sharpe = (r.mean() / r.std() * np.sqrt(total)) if r.std() > 0 else 0.0
    # Note: this is a t-statistic (edge significance), not annualised Sharpe.
    # Labelled "Edge Score" in the UI to avoid confusion.
    streak = max_s = 0
    for v in r:
        streak = streak + 1 if v <= 0 else 0
        max_s  = max(max_s, streak)
    holds      = [t["hold_days"] for t in trades if t["hold_days"] > 0]
    time_exits = sum(1 for t in trades if t.get("exit_type") == "Time Exit")
    avg_win  = float(wins.mean())  if len(wins)   else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "total_trades": total, "win_rate": len(wins) / total,
        "avg_r": r.mean(), "profit_factor": pf,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "max_drawdown": dd.min(), "sharpe": sharpe,
        "longest_losing_streak": max_s,
        "avg_hold_days": np.mean(holds) if holds else 0,
        "gross_profit": gp, "gross_loss": gl,
        "equity": equity, "r_mults": r,
        "time_exits": time_exits,
    }

# ─── Optimization ──────────────────────────────────────────────────────────────

def optimize(df, qualifying, sl_mode="Fixed", trail_atr_mult=1.5,
             transaction_cost=0.001, max_hold_bars=5, use_time_exit=False,
             direction="long"):
    """Returns (results_df, all_trades_dict) — runs each combo only once."""
    results    = []
    all_trades = {}
    combos     = list(product(RETRACEMENTS, SL_DISTANCES, TP_MODES))
    n          = len(combos)
    prog       = st.progress(0, text=f"Testing {n} parameter combinations...")
    for i, (ret, sl, tp) in enumerate(combos):
        trades = run_backtest(df, qualifying, ret, sl, tp,
                              sl_mode, trail_atr_mult, transaction_cost,
                              max_hold_bars=9999, use_time_exit=False,
                              direction=direction)
        all_trades[(ret, sl, tp)] = trades
        m = calc_metrics(trades)
        if m and m["total_trades"] >= 5:
            results.append({
                "retracement": ret, "sl_dist": sl, "tp_mode": tp,
                "total_trades": m["total_trades"],
                "win_rate":     round(m["win_rate"] * 100, 1),
                "avg_r":        round(m["avg_r"], 3),
                "profit_factor": round(m["profit_factor"], 2)
                    if not np.isinf(m["profit_factor"]) else 99.0,
                "max_drawdown": round(m["max_drawdown"] * 100, 1),
                "sharpe":       round(m["sharpe"], 2),
                "longest_losing_streak": m["longest_losing_streak"],
                "avg_hold_days": round(m["avg_hold_days"], 1),
            })
        prog.progress((i + 1) / n,
                      text=f"Testing {i+1}/{n} combinations...")
    prog.empty()
    if not results:
        return pd.DataFrame(), all_trades
    rdf = pd.DataFrame(results)
    rdf.sort_values(["profit_factor", "win_rate", "sharpe"],
                    ascending=False, inplace=True)
    rdf.reset_index(drop=True, inplace=True)
    return rdf, all_trades

# ─── Chart Helpers ─────────────────────────────────────────────────────────────

DARK = dict(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(color="#ccd6f6"), margin=dict(l=0, r=0, t=40, b=0))


def _dark(fig, title="", **kw):
    fig.update_layout(title=title, **DARK, **kw)
    return fig


def plot_equity_curve(trades, ticker):
    if not trades:
        return go.Figure()
    # Sort by exit_date for correct chronological equity curve
    sorted_t = sorted(trades, key=lambda t: t["exit_date"] if t["exit_date"] else datetime.min)
    r  = np.array([t["r_mult"] for t in sorted_t])
    eq = np.cumprod(1 + r * 0.02) * 10000
    dt = [t["exit_date"] for t in sorted_t]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dt, y=eq, mode="lines",
                             line=dict(color="#64ffda", width=2),
                             fill="tozeroy", fillcolor="rgba(100,255,218,0.05)",
                             name="Strategy"))
    fig.add_hline(y=10000, line_dash="dash", line_color="#8892b0", line_width=1)
    return _dark(fig, f"{ticker} -- Equity Curve (2% risk/trade, $10k start)",
                 xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                 hovermode="x unified")


def plot_monthly_heatmap(trades):
    if not trades:
        return go.Figure()
    rows = [{"year": t["exit_date"].year, "month": t["exit_date"].month,
             "r": t["r_mult"]} for t in trades if t["exit_date"]]
    if not rows:
        return go.Figure()
    pivot = pd.DataFrame(rows).groupby(["year","month"])["r"].sum().unstack(fill_value=0)
    mn    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [mn[c-1] for c in pivot.columns]
    fig = px.imshow(pivot,
                    color_continuous_scale=[[0,"#ff6b6b"],[0.5,"#1e2130"],[1,"#64ffda"]],
                    color_continuous_midpoint=0, text_auto=".2f", aspect="auto")
    return _dark(fig, "Monthly R-Multiple Returns Heatmap",
                 coloraxis_colorbar=dict(title="R Sum"))


def plot_retrace_reach(df, qualifying, direction="long"):
    """Bar chart: % of qualifying candles that price retraced to each level."""
    hits  = {r: 0 for r in RETRACEMENTS}
    total = 0
    date_pos = pd.Series(range(len(df)), index=df.index)
    for date in qualifying.index:
        if date not in date_pos.index:
            continue
        idx     = int(date_pos[date])
        trigger = df.iloc[idx]
        future  = df.iloc[idx + 1: min(idx + 6, len(df))]
        if future.empty:
            continue
        body_abs = abs(float(trigger["body"]))
        total += 1
        for r in RETRACEMENTS:
            if direction == "short":
                level = float(trigger["close"]) + body_abs * r
                if future["high"].max() >= level:
                    hits[r] += 1
            else:
                level = float(trigger["close"]) - float(trigger["body"]) * r
                if future["low"].min() <= level:
                    hits[r] += 1
    if not total:
        return go.Figure()
    pcts  = [hits[r] / total * 100 for r in RETRACEMENTS]
    lbls  = [RETRACE_LABELS[r] for r in RETRACEMENTS]
    colors = ["#64ffda" if p >= 50 else "#ffd700" if p >= 30 else "#ff6b6b"
              for p in pcts]
    direction_label = "Bounce Up" if direction == "short" else "Retrace Down"
    fig = go.Figure(go.Bar(x=lbls, y=pcts, marker_color=colors,
                           text=[f"{p:.1f}%" for p in pcts],
                           textposition="outside"))
    return _dark(fig,
                 f"How Far Did Price {direction_label} After Trigger Candle? (5-bar window)",
                 xaxis_title="Retracement Level", yaxis_title="% of Candles Reaching Level",
                 yaxis=dict(range=[0, 110]))


def plot_winrate_by_retrace(trades_all):
    data = {}
    for (ret, sl, tp), trades in trades_all.items():
        if not trades:
            continue
        wr = sum(1 for t in trades if t["r_mult"] > 0) / len(trades) * 100
        data.setdefault(ret, []).append(wr)
    if not data:
        return go.Figure()
    avgs = [np.mean(data[r]) for r in RETRACEMENTS if r in data]
    lbls = [RETRACE_LABELS[r] for r in RETRACEMENTS if r in data]
    fig  = go.Figure(go.Bar(
        x=lbls, y=avgs,
        marker_color=["#64ffda" if w >= 50 else "#ff6b6b" for w in avgs],
        text=[f"{w:.1f}%" for w in avgs], textposition="outside"))
    return _dark(fig, "Average Win Rate by Entry Retracement Level",
                 xaxis_title="Retracement Level", yaxis_title="Win Rate (%)",
                 yaxis=dict(range=[0, 110]))


def plot_seasonal(qualifying):
    mc  = pd.Series(qualifying.index.month).value_counts().sort_index()
    mn  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Bar(x=[mn[m-1] for m in mc.index], y=mc.values,
                           marker_color="#7c83fd", text=mc.values,
                           textposition="outside"))
    return _dark(fig, "Qualifying Candles by Month",
                 xaxis_title="Month", yaxis_title="Count")

# ─── UI Helpers ────────────────────────────────────────────────────────────────

def metric_card(label, value, color=""):
    cls = f"metric-value {color}" if color else "metric-value"
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="{cls}">{value}</div></div>')


def render_metrics_grid(m, cols=4):
    pf     = m["profit_factor"]
    pf_str = f"{pf:.2f}" if pf < 90 else "inf"
    cards  = [
        ("Total Trades",          str(m["total_trades"]),                                ""),
        ("Win Rate",              f"{m['win_rate']*100:.1f}%",                          "green" if m["win_rate"] >= 0.5 else "red"),
        ("Avg R Multiple",        f"{m['avg_r']:.3f}R",                                 "green" if m["avg_r"] > 0 else "red"),
        ("Profit Factor",         pf_str,                                                "green" if pf >= 1.5 else ("red" if pf < 1 else "")),
        ("Avg Win",               f"+{m.get('avg_win', 0):.3f}R",                      "green"),
        ("Avg Loss",              f"{m.get('avg_loss', 0):.3f}R",                       "red"),
        ("Max Drawdown",          f"{m['max_drawdown']*100:.1f}%",                      "red" if m["max_drawdown"] < -0.15 else ""),
        ("Edge Score ⓘ",          f"{m['sharpe']:.2f}",                                 "green" if m["sharpe"] > 1 else ""),
        ("Longest Losing Streak", str(m["longest_losing_streak"]),                       ""),
        ("Avg Hold Time",         f"{m['avg_hold_days']:.1f} days",                     ""),
    ]
    gc = st.columns(cols)
    for i, (lbl, val, col) in enumerate(cards):
        with gc[i % cols]:
            st.markdown(metric_card(lbl, val, col), unsafe_allow_html=True)
    st.caption("ⓘ Edge Score = mean÷std×√N (t-statistic). >1.0 indicates statistically meaningful edge. "
               "Not the same as annualised Sharpe Ratio.")

# ─── Portfolio Simulator ───────────────────────────────────────────────────────

def render_portfolio_simulator(trades, df, ticker):
    st.markdown("### Portfolio Growth Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        start_cap = st.number_input("Starting Capital ($)", value=1000,
                                    min_value=100, step=100, key="sim_cap")
    with c2:
        risk_pct = st.number_input("Risk per Trade (%)", value=2.0,
                                   min_value=0.1, max_value=20.0, step=0.1,
                                   key="sim_risk") / 100
    with c3:
        compound = st.toggle("Compounding", value=True, key="sim_compound")

    if not trades:
        st.info("No trades to simulate.")
        return

    portfolio    = start_cap
    equity_vals  = [start_cap]
    equity_dates = []
    rows         = []

    # Sort by exit_date for correct chronological equity curve
    sorted_trades = sorted(trades, key=lambda t: t["exit_date"] if t["exit_date"] else datetime.min)
    for t in sorted_trades:
        risk_amt  = portfolio * risk_pct if compound else start_cap * risk_pct
        pnl       = t["r_mult"] * risk_amt
        portfolio = portfolio + pnl
        equity_vals.append(portfolio)
        equity_dates.append(t["exit_date"])
        rows.append({"exit_date": t["exit_date"], "pnl": pnl,
                     "portfolio": portfolio, "win": t["r_mult"] > 0})

    final     = portfolio
    total_ret = (final - start_cap) / start_cap * 100

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Starting Capital", f"${start_cap:,.0f}")
    mc2.metric("Ending Capital",   f"${final:,.0f}", f"{total_ret:+.1f}%")
    mc3.metric("Net P&L",          f"${final - start_cap:+,.0f}")

    bah_start = bah_end = None
    if trades and not df.empty:
        try:
            s = df.index.get_indexer([trades[0]["entry_date"]], method="nearest")[0]
            e = df.index.get_indexer([trades[-1]["exit_date"]], method="nearest")[0]
            bah_start = float(df.iloc[s]["close"])
            bah_end   = float(df.iloc[e]["close"])
        except Exception:
            pass

    all_dates = [trades[0]["entry_date"]] + equity_dates
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_dates, y=equity_vals, mode="lines",
                             line=dict(color="#ff8c42", width=2), name="Strategy",
                             fill="tozeroy", fillcolor="rgba(255,140,66,0.06)"))
    if bah_start and bah_end:
        bah_vals = [start_cap, start_cap * (bah_end / bah_start)]
        fig.add_trace(go.Scatter(x=[all_dates[0], all_dates[-1]], y=bah_vals,
                                 mode="lines",
                                 line=dict(color="#6e7681", width=1.5, dash="dash"),
                                 name=f"Buy & Hold {ticker}"))
        bah_ret = (bah_vals[-1] - start_cap) / start_cap * 100
        beat    = total_ret > bah_ret
        label   = "Strategy beats buy & hold" if beat else "Strategy underperforms buy & hold"
        st.markdown(f"{'checkmark' if beat else 'warning'} **{label}** "
                    f"({total_ret:+.1f}% vs {bah_ret:+.1f}%)")

    fig.add_hline(y=start_cap, line_dash="dash", line_color="#8892b0", line_width=1)
    fig.update_layout(title="Portfolio Growth vs Buy & Hold",
                      xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                      hovermode="x unified", **DARK)
    st.plotly_chart(fig, use_container_width=True)

    if rows:
        df_r = pd.DataFrame(rows)
        df_r["year"]  = pd.to_datetime(df_r["exit_date"]).dt.year
        df_r["month"] = pd.to_datetime(df_r["exit_date"]).dt.month
        monthly = df_r.groupby(["year","month"]).agg(
            Trades=("pnl","count"), Wins=("win","sum"),
            Monthly_PnL=("pnl","sum")).reset_index()
        monthly["Losses"]           = monthly["Trades"] - monthly["Wins"]
        monthly["End Portfolio"]    = monthly["Monthly_PnL"].cumsum() + start_cap
        monthly["Month"]            = monthly.apply(
            lambda r: datetime(int(r["year"]), int(r["month"]), 1).strftime("%b %Y"), axis=1)
        monthly["Monthly P&L ($)"]  = monthly["Monthly_PnL"].round(2)
        monthly["End Portfolio ($)"] = monthly["End Portfolio"].round(2)
        st.markdown("#### Monthly Breakdown")
        st.dataframe(monthly[["Month","Trades","Wins","Losses",
                               "Monthly P&L ($)","End Portfolio ($)"]],
                     use_container_width=True)

        yearly = df_r.groupby("year").agg(
            Trades=("pnl","count"), Wins=("win","sum"),
            Annual_PnL=("pnl","sum")).reset_index()
        yearly["Win Rate %"]      = (yearly["Wins"] / yearly["Trades"] * 100).round(1)
        yearly["End Portfolio"]   = yearly["Annual_PnL"].cumsum() + start_cap
        prev                      = yearly["End Portfolio"] - yearly["Annual_PnL"]
        yearly["Annual Return %"] = (yearly["Annual_PnL"] / prev.replace(0, float("nan")) * 100).round(1)
        yearly["End Portfolio ($)"] = yearly["End Portfolio"].round(2)
        st.markdown("#### Yearly Summary")
        st.dataframe(yearly[["year","Trades","Win Rate %",
                              "Annual Return %","End Portfolio ($)"]].rename(
            columns={"year":"Year"}), use_container_width=True)

# ─── Partial Exit Explanation ──────────────────────────────────────────────────

def render_partial_explanation(all_trades):
    st.markdown("""<div class="info-box">
<h4>HOW PARTIAL EXIT WORKS</h4>
<p>
<b>When price reaches 1R (first target):</b><br>
&nbsp;&nbsp;Close 50% of your position - Move Stop Loss to breakeven (entry price)<br>
&nbsp;&nbsp;Your remaining 50% is now a risk-free trade<br><br>
<b>When price reaches 2R (second target):</b><br>
&nbsp;&nbsp;Close the remaining 50% of your position<br><br>
<b>Example with $1,000 position, SL = 2%:</b><br>
&nbsp;&nbsp;Entry: $100 | SL: $98 | TP1 (1R): $102 | TP2 (2R): $104<br>
&nbsp;&nbsp;At TP1: sell $500 worth, pocket +$10, move SL to $100<br>
&nbsp;&nbsp;At TP2: sell remaining $500, pocket +$20<br>
&nbsp;&nbsp;<b>Total profit: $30 on $1,000 trade</b>
</p>
</div>""", unsafe_allow_html=True)

    stats_wr = {tp: [] for tp in TP_MODES}
    stats_pf = {tp: [] for tp in TP_MODES}
    for (ret, sl, tp), trades in all_trades.items():
        if not trades or tp not in stats_wr:
            continue
        m = calc_metrics(trades)
        if not m:
            continue
        stats_wr[tp].append(m["win_rate"] * 100)
        pf = m["profit_factor"]
        if not np.isinf(pf):
            stats_pf[tp].append(pf)

    st.markdown("#### TP Mode Comparison (avg across all parameter combos)")
    labels = {"2R": "Full 2R", "3R": "Full 3R", "Partial": "Partial (1R+2R)"}
    cols   = st.columns(len(TP_MODES))
    for col, tp in zip(cols, TP_MODES):
        avg_wr = np.mean(stats_wr[tp]) if stats_wr[tp] else 0
        avg_pf = np.mean(stats_pf[tp]) if stats_pf[tp] else 0
        with col:
            st.markdown(metric_card(f"{labels[tp]} — Win Rate", f"{avg_wr:.1f}%"),
                        unsafe_allow_html=True)
            st.markdown(metric_card(f"{labels[tp]} — Profit Factor", f"{avg_pf:.2f}"),
                        unsafe_allow_html=True)

# ─── SL Comparison ────────────────────────────────────────────────────────────

def render_sl_comparison(df, qualifying, best_ret, best_sl_dist,
                          best_tp, trail_mult, tc, direction="long"):
    st.markdown("### Stop Loss Method Comparison")

    # Cache: only re-run when params change (not on every widget interaction)
    _sl_cache_key = f"slcomp_{best_ret}_{best_sl_dist}_{best_tp}_{trail_mult}_{tc}_{direction}_{id(qualifying)}"
    cached = st.session_state.get("_sl_comp_cache")
    if cached is None or cached.get("key") != _sl_cache_key:
        tests  = [
            ("Fixed",          best_tp),
            ("Breakeven",      best_tp),
            ("Trailing ATR",   best_tp),
            ("Breakeven+Trail",best_tp),
        ]
        labels = [
            f"Fixed SL ({best_tp})",
            f"Breakeven SL ({best_tp})",
            f"Trailing ATR ({best_tp})",
            f"Breakeven+Trail ({best_tp})",
        ]
        rows = []
        for (mode, tp), lbl in zip(tests, labels):
            trades = run_backtest(df, qualifying, best_ret, best_sl_dist,
                                  tp, mode, trail_mult, tc, direction=direction)
            m = calc_metrics(trades)
            if not m:
                continue
            pf = m["profit_factor"]
            rows.append({
                "SL Method":     lbl,
                "Win Rate":      f"{m['win_rate']*100:.1f}%",
                "Avg R":         f"{m['avg_r']:.3f}",
                "Profit Factor": f"{pf:.2f}" if not np.isinf(pf) else "inf",
                "Max DD":        f"{m['max_drawdown']*100:.1f}%",
                "_pf_raw":       pf if not np.isinf(pf) else 999,
            })
        st.session_state["_sl_comp_cache"] = {"key": _sl_cache_key, "rows": rows}
    else:
        rows = cached["rows"]

    if not rows:
        st.info("Not enough trades to compare SL methods.")
        return

    best_lbl = max(rows, key=lambda r: r["_pf_raw"])["SL Method"]
    display  = pd.DataFrame(rows).drop(columns=["_pf_raw"])

    def highlight_best(row):
        if row["SL Method"] == best_lbl:
            return ["background-color: rgba(63,185,80,0.15); color:#3fb950"] * len(row)
        return [""] * len(row)

    st.dataframe(display.style.apply(highlight_best, axis=1),
                 use_container_width=True)
    st.caption(f"Best performing method (by Profit Factor): **{best_lbl}**")

# ─── Hold Time Analysis ────────────────────────────────────────────────────────

def analyze_hold_times(df, qualifying, ret, sl_dist, tp_mode,
                       sl_mode, trail_mult, tc, direction="long"):
    """Test each hold time 1-20 (time exit) + no-limit. Returns comparison DataFrame."""
    rows = []
    for n in range(1, 21):
        trades = run_backtest(df, qualifying, ret, sl_dist, tp_mode,
                              sl_mode, trail_mult, tc,
                              max_hold_bars=n, use_time_exit=True, direction=direction)
        m     = calc_metrics(trades)
        label = f"{n} candle{'s' if n > 1 else ''} after entry"
        if m:
            pf = m["profit_factor"]
            rows.append({
                "hold_time":     n,
                "label":         label,
                "total_trades":  m["total_trades"],
                "win_rate":      round(m["win_rate"] * 100, 1),
                "profit_factor": round(pf, 2) if not np.isinf(pf) else 99.0,
                "avg_r":         round(m["avg_r"], 3),
                "time_exits":    m.get("time_exits", 0),
            })
        else:
            rows.append({"hold_time": n, "label": label,
                         "total_trades": 0, "win_rate": 0.0,
                         "profit_factor": 0.0, "avg_r": 0.0, "time_exits": 0})

    # No-limit row: large max_hold_bars, no time exit = original behaviour
    trades_nl = run_backtest(df, qualifying, ret, sl_dist, tp_mode,
                             sl_mode, trail_mult, tc,
                             max_hold_bars=9999, use_time_exit=False, direction=direction)
    m_nl = calc_metrics(trades_nl)
    if m_nl:
        pf_nl = m_nl["profit_factor"]
        rows.append({
            "hold_time":     999,
            "label":         "No limit",
            "total_trades":  m_nl["total_trades"],
            "win_rate":      round(m_nl["win_rate"] * 100, 1),
            "profit_factor": round(pf_nl, 2) if not np.isinf(pf_nl) else 99.0,
            "avg_r":         round(m_nl["avg_r"], 3),
            "time_exits":    0,
        })
    return pd.DataFrame(rows)


def plot_hold_time_chart(hold_df):
    """Line chart: profit factor vs hold time 1-20, with no-limit reference line."""
    if hold_df.empty:
        return go.Figure()
    chart_df = hold_df[hold_df["hold_time"] < 999].copy()
    if chart_df.empty:
        return go.Figure()
    best_pf    = chart_df["profit_factor"].max()
    colors     = ["#64ffda" if pf == best_pf else "#7c83fd"
                  for pf in chart_df["profit_factor"]]
    sizes      = [14 if pf == best_pf else 7 for pf in chart_df["profit_factor"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df["label"], y=chart_df["profit_factor"],
        mode="lines+markers",
        line=dict(color="#7c83fd", width=2),
        marker=dict(color=colors, size=sizes),
        name="Profit Factor",
        hovertemplate="%{x}: PF=%{y:.2f}<extra></extra>",
    ))
    nolimit_row = hold_df[hold_df["hold_time"] == 999]
    if not nolimit_row.empty:
        nl_pf = nolimit_row.iloc[0]["profit_factor"]
        fig.add_hline(
            y=nl_pf, line_dash="dash", line_color="#8892b0", line_width=1,
            annotation_text=f"No time limit — hold until TP/SL ({nl_pf:.2f})",
            annotation_position="bottom right",
            annotation_font_color="#8892b0",
        )
    return _dark(fig, "Profit Factor by Max Hold (candles counted from entry fill)",
                 xaxis_title="Max Hold Time (candle 1 = first candle after entry)",
                 yaxis_title="Profit Factor",
                 hovermode="x unified")


def plot_candle_resolution(df, qualifying, ret, sl_dist, tp_mode,
                           sl_mode, trail_mult, tc, direction="long"):
    """Stacked bar: cumulative % of trades that hit TP or SL by each candle.
    Uses max_hold_bars=9999 (no cap) so every trade eventually closes via TP or SL.
    'Still Open' at candle N means genuine TP/SL not yet hit — no artificial
    Max Hold exits polluting the counts."""
    trades = run_backtest(df, qualifying, ret, sl_dist, tp_mode,
                          sl_mode, trail_mult, tc,
                          max_hold_bars=9999, use_time_exit=False, direction=direction)
    if not trades:
        return go.Figure()
    n_trades   = len(trades)
    candles    = list(range(1, 21))
    tp_cum, sl_cum, still_open = [], [], []
    for c in candles:
        tp_n = sum(1 for t in trades
                   if t.get("exit_type") == "TP" and t["candles_held"] <= c)
        sl_n = sum(1 for t in trades
                   if t.get("exit_type") == "SL" and t["candles_held"] <= c)
        tp_cum.append(tp_n / n_trades * 100)
        sl_cum.append(sl_n / n_trades * 100)
        still_open.append(max(0.0, 100 - tp_cum[-1] - sl_cum[-1]))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=candles, y=tp_cum, name="TP Hit (cumul.)",
        marker_color="#64ffda",
        text=[f"{v:.0f}%" if v >= 8 else "" for v in tp_cum],
        textposition="inside",
        textfont=dict(size=11, color="#0d1117"),
    ))
    fig.add_trace(go.Bar(
        x=candles, y=sl_cum, name="SL Hit (cumul.)",
        marker_color="#ff6b6b",
        text=[f"{v:.0f}%" if v >= 8 else "" for v in sl_cum],
        textposition="inside",
        textfont=dict(size=11, color="#0d1117"),
    ))
    fig.add_trace(go.Bar(
        x=candles, y=still_open, name="Still Open",
        marker_color="#4a5568",
        text=[f"{v:.0f}%" if v >= 8 else "" for v in still_open],
        textposition="inside",
        textfont=dict(size=11, color="#ccd6f6"),
    ))
    fig.update_layout(barmode="stack")
    return _dark(fig, "Cumulative Trade Exits by Candle Number (20-bar window)",
                 xaxis_title="Candle Number from Entry",
                 yaxis_title="% of All Trades",
                 yaxis=dict(range=[0, 110]),
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))


def render_hold_time_analysis(df, qualifying, best_ret, best_sl_dist,
                               best_tp, sl_mode, trail_mult, tc, ticker,
                               default_hold_bars=5, direction="long"):
    """Optimal Hold Time section rendered inside Strategy Finder tab."""
    st.markdown("---")
    st.markdown("### Optimal Hold Time Analysis")

    # hold_key includes run_key so ANY new Run Analysis (different timeframe, ticker,
    # data range, direction, etc.) always invalidates the cached hold time results.
    _current_run_key = st.session_state.get("run_key", "")
    hold_key = (f"hold_{_current_run_key}_{best_ret}_{best_sl_dist}_{best_tp}_"
                f"{sl_mode}_{trail_mult}_{tc}_{direction}")
    _stored_hold_key = st.session_state.get("hold_key", "")
    if _stored_hold_key != hold_key or "hold_df" not in st.session_state:
        with st.spinner("Testing hold times 1–20 candles..."):
            hold_df = analyze_hold_times(df, qualifying, best_ret, best_sl_dist,
                                         best_tp, sl_mode, trail_mult, tc, direction=direction)
        st.session_state["hold_df"]  = hold_df
        st.session_state["hold_key"] = hold_key

    hold_df = st.session_state.get("hold_df", pd.DataFrame())
    if hold_df.empty:
        st.info("Not enough trades for hold time analysis.")
        return

    # ── Line chart (cached) ──────────────────────────────────────────────────
    _ht_fig_key = st.session_state.get("hold_key", "")
    if st.session_state.get("_ht_fig_key") != _ht_fig_key:
        st.session_state["_ht_fig"]     = plot_hold_time_chart(hold_df)
        st.session_state["_ht_fig_key"] = _ht_fig_key
    st.plotly_chart(st.session_state["_ht_fig"], use_container_width=True,
                    config={"displayModeBar": False})

    # ── Table with green/red row highlights ──────────────────────────────────
    timed_df = hold_df[hold_df["hold_time"] < 999]
    if timed_df.empty:
        return
    best_idx   = timed_df["profit_factor"].idxmax()
    worst_idx  = timed_df["profit_factor"].idxmin()
    best_ht    = int(timed_df.loc[best_idx,  "hold_time"])
    worst_ht   = int(timed_df.loc[worst_idx, "hold_time"])
    best_ht_pf = timed_df.loc[best_idx, "profit_factor"]

    def _hl(row):
        if row["hold_time"] == best_ht:
            return ["background-color:rgba(63,185,80,0.18);color:#3fb950"] * len(row)
        if row["hold_time"] == worst_ht:
            return ["background-color:rgba(255,107,107,0.15);color:#ff6b6b"] * len(row)
        return [""] * len(row)

    display = hold_df.rename(columns={
        "label": "Hold Time", "total_trades": "Trades",
        "win_rate": "Win%", "profit_factor": "Profit Factor",
        "avg_r": "Avg R", "time_exits": "Time Exits",
    })[["Hold Time", "Trades", "Win%", "Profit Factor",
        "Avg R", "Time Exits", "hold_time"]]
    st.dataframe(
        display.style.apply(_hl, axis=1).hide(axis="columns", subset=["hold_time"]),
        use_container_width=True,
    )

    # ── Recommendation box ───────────────────────────────────────────────────
    nolimit_row = hold_df[hold_df["hold_time"] == 999]
    nolimit_pf  = nolimit_row.iloc[0]["profit_factor"] if not nolimit_row.empty else 0.0

    default_row = timed_df[timed_df["hold_time"] == default_hold_bars]
    default_pf  = float(default_row.iloc[0]["profit_factor"]) if not default_row.empty else None
    default_label = f"{default_hold_bars} candle{'s' if default_hold_bars > 1 else ''}"

    if best_ht_pf > nolimit_pf + 0.05:
        verdict       = f"Time exit IMPROVES results — exit at candle {best_ht} open after entry"
        verdict_color = "#64ffda"
    else:
        verdict       = "Time exit HURTS results — hold until TP/SL hit"
        verdict_color = "#ff6b6b"

    default_pf_row = ""
    if default_pf is not None and default_hold_bars != best_ht:
        default_pf_row = f"""
<div class="best-row">
  <span class="best-key">PF at your current setting ({default_label} after entry)</span>
  <span class="best-val">{default_pf:.2f}</span>
</div>"""

    st.markdown(f"""<div class="best-box">
<h3>Optimal Hold Time for {ticker}</h3>
<div class="best-row">
  <span class="best-key">Best timed hold</span>
  <span class="best-val">{best_ht} candle{'s' if best_ht > 1 else ''} after entry &nbsp;<span style="color:#8892b0;font-size:12px;">(exit at candle {best_ht} open)</span></span>
</div>
<div class="best-row">
  <span class="best-key">Profit factor at optimal hold</span>
  <span class="best-val">{best_ht_pf:.2f}</span>
</div>{default_pf_row}
<div class="best-row">
  <span class="best-key">Profit factor — no time limit (hold until TP/SL)</span>
  <span class="best-val">{nolimit_pf:.2f}</span>
</div>
<div class="best-row">
  <span class="best-key">Verdict</span>
  <span class="best-val" style="color:{verdict_color};">{verdict}</span>
</div>
</div>""", unsafe_allow_html=True)

    # ── Resolution chart (cached) ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Where Do Trades Close?")
    st.markdown("""<div class="info-box">
<h4>📊 How to read this chart</h4>
<p>
<b>Candle 1 = the first full candle after your entry fills.</b>
Each bar shows, cumulatively, what percentage of all trades have closed by that candle via
<b>Take Profit</b> (teal) or <b>Stop Loss</b> (red).
<b>Still Open</b> (grey) = the trade has not yet hit TP or SL by that candle.
</p>
<p>
<b>Trade exit priority order:</b><br>
&nbsp;&nbsp;1. <b>Stop Loss hit</b> — price touches SL → closes immediately as a loss<br>
&nbsp;&nbsp;2. <b>Take Profit hit</b> — price touches TP → closes immediately as a win<br>
&nbsp;&nbsp;3. <b>Time exit</b> (if enabled) — still open at candle N? → exits at <b>candle N's open price</b><br>
&nbsp;&nbsp;4. <b>Still open</b> — trade is live beyond the 20-candle window shown here
</p>
<p style="color:#8892b0;font-size:12px;">
This chart runs with <em>no time cap</em> — every trade runs until it genuinely
hits TP or SL. "Still Open" at candle 20 means the trade truly hasn't resolved yet.
</p>
</div>""", unsafe_allow_html=True)
    _res_key = hold_key + "_res"
    if st.session_state.get("_res_fig_key") != _res_key:
        with st.spinner("Building resolution chart..."):
            _res_fig = plot_candle_resolution(df, qualifying, best_ret, best_sl_dist,
                                              best_tp, sl_mode, trail_mult, tc, direction=direction)
        st.session_state["_res_fig"]     = _res_fig
        st.session_state["_res_fig_key"] = _res_key
    st.plotly_chart(st.session_state["_res_fig"], use_container_width=True)


# ─── Feature 3: Live Scanner ───────────────────────────────────────────────────

def render_live_scanner(ticker, timeframe, min_body_pct, min_vol_mult,
                         best_ret, best_sl_dist, best_tp, best_wr,
                         direction="long", adx_filter_on=False, adx_di_gap_min=0.0,
                         adx_threshold=25,
                         ema_filter=False, ema_period=200,
                         ema_tf_mode="Same as signal timeframe", ema_htf_period=200):
    htf         = _HTF_MAP.get(timeframe, "1D")
    htf_label   = _HTF_LABEL.get(timeframe, "Daily")
    tf_label    = {"1D": "Daily", "4H": "4H", "1H": "1H"}[timeframe]
    dir_label   = "📉 SHORT — bearish candles" if direction == "short" else "📈 LONG — bullish candles"
    dir_color   = "#ff6b6b" if direction == "short" else "#64ffda"

    st.markdown(f"### Live Scanner — {tf_label} Candles")
    st.markdown(f"""<div class="info-box" style="padding:10px 16px;margin-bottom:10px;">
<span style="color:{dir_color};font-weight:600;">{dir_label}</span>
&nbsp;|&nbsp; <span style="color:#58a6ff;font-weight:600;">ADX columns are informational only</span>
— signal is triggered by <b>Body %</b> + <b>Volume</b> alone.<br>
<span style="color:#8892b0;font-size:12px;">
Signal TF: <b>{tf_label}</b> &nbsp;|&nbsp; Higher TF (HTF): <b>{htf_label}</b>
&nbsp;({timeframe} → {htf} for ADX context)
</span>
</div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_b:
        refresh = st.button("Refresh Scanner", key="scanner_refresh",
                            use_container_width=True)

    # Re-fetch on: explicit refresh, ticker/TF change, or TTL expired
    # TTL: 1H → 5 min, 4H → 15 min, 1D → 60 min
    # Short TTLs ensure new candles appear within minutes of close
    _LIVE_TTL = {"1H": 5, "4H": 15, "1D": 60}
    live_key  = f"{ticker}_{timeframe}"
    _fetched_at = st.session_state.get("live_fetched_at")
    _ttl_minutes = _LIVE_TTL.get(timeframe, 30)
    _ttl_expired = (
        _fetched_at is None or
        (datetime.now() - _fetched_at).total_seconds() > _ttl_minutes * 60
    )
    if refresh or "live_df" not in st.session_state or \
            st.session_state.get("live_key") != live_key or _ttl_expired:
        with st.spinner("Fetching latest candles..."):
            live_df     = fetch_live(ticker, timeframe)
            # Fetch HTF for ADX context — always, regardless of ADX filter toggle
            htf_days    = {"1H": 14, "4H": 60, "1D": 365}.get(timeframe, 60)
            live_df_htf = _binance_klines(ticker, _BINANCE_INTERVAL[htf], htf_days)
            if live_df_htf.empty:
                live_df_htf = _gateio_klines(ticker, _BINANCE_INTERVAL[htf], htf_days)
        # Fetch derivatives & sentiment data (short TTL, free APIs)
        _is_perp = str(ticker).upper().endswith("USDT")
        _fr_data  = fetch_funding_rate(ticker)  if _is_perp else {"rate": 0.0, "ok": False}
        _oi_data  = fetch_open_interest(ticker) if _is_perp else {"oi_change_pct": 0.0, "ok": False}
        _fg_data  = fetch_fear_greed()
        st.session_state["live_df"]        = live_df
        st.session_state["live_df_htf"]    = live_df_htf
        st.session_state["live_ts"]        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["live_key"]       = live_key
        st.session_state["live_fetched_at"] = datetime.now()
        st.session_state["live_fr_data"]   = _fr_data
        st.session_state["live_oi_data"]   = _oi_data
        st.session_state["live_fg_data"]   = _fg_data

    live_df     = st.session_state.get("live_df",     pd.DataFrame())
    live_df_htf = st.session_state.get("live_df_htf", pd.DataFrame())
    live_ts     = st.session_state.get("live_ts",     "unknown")
    _fr_data    = st.session_state.get("live_fr_data",  {"rate": 0.0, "ok": False})
    _oi_data    = st.session_state.get("live_oi_data",  {"oi_change_pct": 0.0, "ok": False})
    _fg_data    = st.session_state.get("live_fg_data",  {"value": 50, "classification": "Neutral", "ok": False})

    with col_a:
        _next_refresh_min = _ttl_minutes - max(0, int((datetime.now() - st.session_state.get("live_fetched_at", datetime.now())).total_seconds() / 60))
        st.caption(f"Last updated: {live_ts} (WIB)  |  Auto-refresh in ~{_next_refresh_min} min")

    # ── Market Context Pills ───────────────────────────────────────────────────
    _fg_val   = _fg_data.get("value", 50)
    _fg_clf   = _fg_data.get("classification", "")
    _fr_rate  = _fr_data.get("rate", 0.0)
    _oi_chg   = _oi_data.get("oi_change_pct", 0.0)
    _fg_col   = "#f85149" if _fg_val < 25 else "#e3b341" if _fg_val < 45 else "#3fb950" if _fg_val > 75 else "#8892b0"
    _fr_col   = "#64ffda" if _fr_rate < -0.01 else "#f85149" if _fr_rate > 0.03 else "#8892b0"
    _fr_note  = "shorts crowded 🔥" if _fr_rate < -0.01 else "longs crowded ⚠️" if _fr_rate > 0.03 else "neutral"
    _oi_col   = "#64ffda" if _oi_chg > 5 else "#f85149" if _oi_chg < -5 else "#8892b0"
    _oi_note  = f"{'▲' if _oi_chg >= 0 else '▼'} {abs(_oi_chg):.1f}%"
    _context_parts = [
        f'<span style="border:1px solid {_fg_col};color:{_fg_col};border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;">F&G {_fg_val} — {_fg_clf}</span>',
    ]
    if _fr_data.get("ok"):
        _context_parts.append(
            f'<span style="border:1px solid {_fr_col};color:{_fr_col};border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;margin-left:6px;">Funding {_fr_rate*100:.4f}% ({_fr_note})</span>'
        )
    if _oi_data.get("ok"):
        _context_parts.append(
            f'<span style="border:1px solid {_oi_col};color:{_oi_col};border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;margin-left:6px;">OI 24h {_oi_note}</span>'
        )
    st.markdown(
        '<div style="margin-bottom:10px;display:flex;flex-wrap:wrap;gap:4px;">'
        + "".join(_context_parts) + "</div>",
        unsafe_allow_html=True,
    )

    if live_df.empty:
        st.error("Could not fetch live data.")
        return

    # Compute ADX on signal TF and HTF
    adx_df_signal  = calculate_adx(live_df) if not live_df.empty else None
    adx_df_htf_raw = calculate_adx(live_df_htf) if not live_df_htf.empty else None
    # Extract series
    adx_signal   = adx_df_signal["adx"]      if adx_df_signal  is not None else pd.Series(dtype=float)
    dip_signal   = adx_df_signal["di_plus"]  if adx_df_signal  is not None else pd.Series(dtype=float)
    dim_signal   = adx_df_signal["di_minus"] if adx_df_signal  is not None else pd.Series(dtype=float)
    adx_htf_raw  = adx_df_htf_raw["adx"]     if adx_df_htf_raw is not None else pd.Series(dtype=float)
    # Forward-fill HTF ADX onto signal TF index
    adx_htf = adx_htf_raw.reindex(live_df.index, method="ffill") if not adx_htf_raw.empty else pd.Series(dtype=float)

    # ── EMA for Live Scanner ─────────────────────────────────────────────────
    # live_df only has ~30 days — not enough to initialize EMA 50/100/200.
    # Fetch cached longer history specifically for EMA warm-up.
    # EMA needs ~3× the period in bars to converge properly.
    _ema_tf_days = {"1D": max(ema_period * 3, 90), "4H": max(ema_period // 2, 60),
                    "1H": max(ema_period // 6, 30)}.get(timeframe, max(ema_period * 3, 90))
    _df_for_ema = _binance_fetch(ticker, timeframe, _ema_tf_days)
    _ema_base_df = _df_for_ema if not _df_for_ema.empty else live_df
    ema_signal_series = calculate_ema(_ema_base_df, ema_period)   # shift(1) inside
    ema_htf_series    = pd.Series(dtype=float)
    if ema_filter and ema_tf_mode == "Higher timeframe" and not live_df_htf.empty:
        _ema_htf_raw   = calculate_ema(live_df_htf, ema_htf_period)
        ema_htf_series = _ema_htf_raw.reindex(live_df.index, method="ffill")

    # Determine if the last fetched candle is still forming or already closed.
    # Each candle opens at its index timestamp; it closes exactly one period later.
    _tf_duration = {"1H": pd.Timedelta(hours=1), "4H": pd.Timedelta(hours=4), "1D": pd.Timedelta(days=1)}
    _candle_dur  = _tf_duration.get(timeframe, pd.Timedelta(hours=4))
    _last_open   = live_df.index[-1]
    _last_close  = _last_open + _candle_dur
    _now_utc     = pd.Timestamp.utcnow().tz_localize(None)
    _last_is_open = _last_close > _now_utc   # True = candle still forming

    if _last_is_open:
        # Exclude the still-forming candle — take the 20 before it
        scan = live_df.iloc[-21:-1].copy() if len(live_df) > 20 else live_df.iloc[:-1].copy()
    else:
        # Last candle already closed — include it in the last 20
        scan = live_df.iloc[-20:].copy() if len(live_df) >= 20 else live_df.copy()

    if scan.empty:
        st.warning("Not enough candle data for scanner.")
        return

    st.markdown(f"**Last 20 {tf_label} candles for {ticker}**")

    # Build scanner table
    table_rows = []
    for i, (idx, row) in enumerate(scan.iterrows()):
        bp_ok  = (
            (row["body"] < 0) and (abs(row["body_pct"]) >= min_body_pct)
            if direction == "short"
            else (row["body"] > 0) and (row["body_pct"] >= min_body_pct)
        )
        vol_ok = (row["vol_mult"] >= min_vol_mult)

        # ADX values at this candle — must be fetched BEFORE the adx_filter_on check
        adx_s_val = adx_signal.get(idx, float("nan"))
        adx_h_val = adx_htf.get(idx,    float("nan"))

        # EMA checks — same-TF and HTF
        ema_s_val = ema_signal_series.get(idx, float("nan"))
        ema_h_val = ema_htf_series.get(idx,   float("nan")) if not ema_htf_series.empty else float("nan")
        close_val = row["close"]
        if pd.notna(ema_s_val):
            ema_same_ok = (close_val > ema_s_val) if direction == "long" else (close_val < ema_s_val)
        else:
            ema_same_ok = False
        if pd.notna(ema_h_val):
            ema_htf_ok = (close_val > ema_h_val) if direction == "long" else (close_val < ema_h_val)
        else:
            ema_htf_ok = None  # HTF not available / not enabled

        # DI gap check — mandatory when ADX filter is ON, informational otherwise
        dip_val   = dip_signal.get(idx,  float("nan"))
        dim_val   = dim_signal.get(idx,  float("nan"))
        if pd.notna(dip_val) and pd.notna(dim_val):
            di_gap_val = abs(dip_val - dim_val)
            if direction == "long":
                di_aligned = dip_val > dim_val
            else:
                di_aligned = dim_val > dip_val
            di_gap_10_ok = di_aligned and di_gap_val >= 10
            di_gap_20_ok = di_aligned and di_gap_val >= 20
        else:
            di_gap_val   = float("nan")
            di_aligned   = False
            di_gap_10_ok = False
            di_gap_20_ok = False

        # When ADX filter is ON, signal requires body + volume + ADX >= threshold + DI gap
        # This exactly mirrors detect_qualifying_candles() logic in the backtest
        if adx_filter_on:
            adx_ok = pd.notna(adx_s_val) and adx_s_val >= adx_threshold
            if adx_di_gap_min > 0:
                req_di_gap = di_aligned and di_gap_val >= adx_di_gap_min
                is_signal  = bp_ok and vol_ok and adx_ok and req_di_gap
            else:
                is_signal  = bp_ok and vol_ok and adx_ok
        else:
            is_signal  = bp_ok and vol_ok

        sig = "SIGNAL" if is_signal else ("partial" if (bp_ok or vol_ok) else "")
        bp_val = abs(row["body_pct"]) * 100 if pd.notna(row["body_pct"]) else 0
        vm_val = row["vol_mult"]       if pd.notna(row["vol_mult"])  else 0
        adx_s_str = f"{adx_s_val:.1f}" if pd.notna(adx_s_val) else "—"
        adx_h_str = f"{adx_h_val:.1f}" if pd.notna(adx_h_val) else "—"

        # Trend direction from DI+/DI- with gap magnitude
        if pd.notna(dip_val) and pd.notna(dim_val):
            di_gap     = abs(dip_val - dim_val)
            strength   = "strong" if di_gap >= 20 else "moderate" if di_gap >= 10 else "weak" if di_gap >= 5 else "neutral"
            trend_dir  = (f"▲ Bullish {strength} (gap {di_gap:.0f})"
                          if dip_val > dim_val else
                          f"▼ Bearish {strength} (gap {di_gap:.0f})")
        else:
            di_gap    = float("nan")
            trend_dir = "—"

        # Alignment: ADX strength + direction alignment with trade mode
        if pd.notna(adx_s_val) and pd.notna(adx_h_val):
            trending = adx_s_val >= 25 and adx_h_val >= 25
            ranging  = adx_s_val < 25  and adx_h_val < 25
            if trending:
                adx_align = "✅ Both trending"
            elif ranging:
                adx_align = "⚪ Both ranging"
            else:
                adx_align = "⚠️ Mixed"
        else:
            adx_align = "—"

        table_rows.append({
            "#":               len(scan) - i,
            "Date":            (
                (idx + pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M")
                if timeframe in ("4H", "1H") else
                str(idx.date()) if hasattr(idx, "date") else str(idx)
            ),
            "Body %":          round(bp_val, 1),
            "Vol Mult":        round(vm_val, 2),
            "Body OK":         "YES" if bp_ok  else "no",
            "Vol OK":          "YES" if vol_ok else "no",
            "SIGNAL":          sig,
            f"ADX ({tf_label})":  adx_s_str,
            "Trend Dir":       trend_dir,
            f"ADX ({htf_label})": adx_h_str,
            "ADX Align":       adx_align,
            "DI≥10":           "✅" if di_gap_10_ok else "—",
            "DI≥20":           "✅" if di_gap_20_ok else "—",
            "_body_ok":        bp_ok,
            "_vol_ok":         vol_ok,
            "_signal":         is_signal,
            "_adx_s":          adx_s_val,
            "_adx_h":          adx_h_val,
            "_dip":            dip_val,
            "_dim":            dim_val,
            "_ema_same_ok":    ema_same_ok,
            "_ema_htf_ok":     ema_htf_ok,
            "_ema_s_val":      ema_s_val,
            "_ema_h_val":      ema_h_val,
        })

    scan_df   = pd.DataFrame(table_rows)
    adx_s_col  = f"ADX ({tf_label})"
    adx_h_col  = f"ADX ({htf_label})"
    ema_s_col  = f"EMA{ema_period} ({tf_label})"
    ema_h_col  = f"EMA{ema_htf_period} ({htf_label})" if ema_tf_mode == "Higher timeframe" else None
    # Populate EMA display columns from hidden private columns
    scan_df[ema_s_col] = scan_df["_ema_same_ok"].apply(
        lambda v: "✅" if v else ("❌" if v is False else "—"))
    if ema_h_col:
        scan_df[ema_h_col] = scan_df["_ema_htf_ok"].apply(
            lambda v: "✅" if v else ("❌" if v is False else "—"))

    # Show confluence requirement note
    if adx_filter_on and adx_di_gap_min > 0:
        st.markdown(
            f'<div style="background:#1a2d1a;border:1px solid #238636;border-radius:6px;'
            f'padding:8px 12px;margin-bottom:8px;font-size:12px;color:#3fb950;">'
            f'✅ <b>ADX filter ON</b> — SIGNAL requires: Body ✓ + Volume ✓ + '
            f'DI aligned + gap ≥ {adx_di_gap_min:.0f} ✓</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#1a1a2d;border:1px solid #1f6feb;border-radius:6px;'
            'padding:8px 12px;margin-bottom:8px;font-size:12px;color:#58a6ff;">'
            'ℹ️ <b>ADX filter OFF</b> — SIGNAL requires: Body ✓ + Volume ✓ only. '
            'DI≥10 / DI≥20 columns are extra confluence (informational).</div>',
            unsafe_allow_html=True)

    def style_row(row):
        if row["_signal"]:
            return ["background-color: rgba(63,185,80,0.18); color:#3fb950"] * len(row)
        if row["_body_ok"] or row["_vol_ok"]:
            return ["background-color: rgba(255,215,0,0.10); color:#ccd6f6"] * len(row)
        return ["color:#6e7681"] * len(row)

    _ema_display_cols = [ema_s_col] + ([ema_h_col] if ema_h_col else [])
    display_cols = ["#", "Date", "Body %", "Vol Mult", "Body OK", "Vol OK",
                    "SIGNAL"] + _ema_display_cols + [adx_s_col, "Trend Dir", "DI≥10", "DI≥20", adx_h_col, "ADX Align"]
    hidden_cols  = ["_body_ok", "_vol_ok", "_signal", "_adx_s", "_adx_h", "_dip", "_dim",
                    "_ema_same_ok", "_ema_htf_ok", "_ema_s_val", "_ema_h_val"]

    st.dataframe(
        scan_df[display_cols + hidden_cols].style.apply(
            style_row, axis=1).hide(axis="columns", subset=hidden_cols),
        use_container_width=True,
        height=460,
        column_config={
            "#":           st.column_config.NumberColumn("#",        width=40),
            "Date":        st.column_config.TextColumn("Date",       width=130),
            "Body %":      st.column_config.NumberColumn("Body %",   width=65,  format="%.1f"),
            "Vol Mult":    st.column_config.NumberColumn("Vol×",      width=58,  format="%.2f"),
            "Body OK":     st.column_config.TextColumn("Body",       width=48),
            "Vol OK":      st.column_config.TextColumn("Vol",        width=43),
            "SIGNAL":      st.column_config.TextColumn("Signal",     width=68),
            ema_s_col:     st.column_config.TextColumn(ema_s_col,    width=90,
                               help=f"✅ = close {'above' if direction == 'long' else 'below'} EMA{ema_period} on {tf_label}. EMA filter {'ON' if ema_filter else 'OFF (informational)'}"),
            **(  {ema_h_col: st.column_config.TextColumn(ema_h_col, width=90,
                               help=f"✅ = close {'above' if direction == 'long' else 'below'} EMA{ema_htf_period} on {htf_label} HTF. Multi-TF EMA confluence.")}
                 if ema_h_col else {}  ),
            adx_s_col:     st.column_config.TextColumn(adx_s_col,    width=72,
                               help=f"ADX(14) strength on {tf_label}. ≥25 = trending market"),
            "Trend Dir":   st.column_config.TextColumn("Trend Dir",  width=195,
                               help="DI+ > DI- = bullish. DI- > DI+ = bearish. Gap = strength."),
            "DI≥10":       st.column_config.TextColumn("DI≥10",      width=55,
                               help="✅ = DI aligned with direction AND gap ≥ 10 (moderate confluence)"),
            "DI≥20":       st.column_config.TextColumn("DI≥20",      width=55,
                               help="✅ = DI aligned with direction AND gap ≥ 20 (strong confluence)"),
            "Trend Dir":   st.column_config.TextColumn("Trend Dir",   width=200,
                               help="DI+ > DI- = bullish trend. DI- > DI+ = bearish trend. "
                                    "For LONG signals, look for ▲ Bullish. "
                                    "For SHORT signals, look for ▼ Bearish."),
            adx_h_col:     st.column_config.TextColumn(adx_h_col,     width=85,
                               help=f"ADX(14) strength on {htf_label} HTF. ≥25 = trending"),
            "ADX Align":   st.column_config.TextColumn("Strength",    width=130,
                               help="Are both TFs in a trending market?"),
        },
    )

    n_signals = scan_df["_signal"].sum()
    st.markdown(f"**Found {n_signals} qualifying candle{'s' if n_signals != 1 else ''} "
                f"in the last {len(scan)} {tf_label} candles**")

    # Signal cards for each qualifying candle
    signal_rows = scan_df[scan_df["_signal"]].copy()
    if signal_rows.empty:
        st.info("No qualifying candles in the most recent 20 bars.")
        return

    # Match back to live_df to get price data
    live_reset = live_df.reset_index()
    for _, sr in signal_rows.iterrows():
        date_str = sr["Date"]
        match = live_reset[live_reset.iloc[:, 0].astype(str).str.startswith(date_str)]
        if match.empty:
            continue
        row      = live_df.iloc[match.index[0]]
        close    = row["close"]
        body_abs = abs(float(row["body"]))
        sl_dist  = best_sl_dist

        if direction == "short":
            entry    = round(close + body_abs * best_ret, 6)
            sl       = round(entry * (1 + sl_dist), 6)
            risk_amt = sl - entry
            tp1r     = round(entry - 1 * risk_amt, 6)
            tp2r     = round(entry - 2 * risk_amt, 6)
            tp3r     = round(entry - 3 * risk_amt, 6)
            sl_desc  = f"{SL_LABELS[sl_dist]} above entry"
        else:
            entry    = round(close - float(row["body"]) * best_ret, 6)
            sl       = round(entry * (1 - sl_dist), 6)
            risk_amt = entry - sl
            tp1r     = round(entry + 1 * risk_amt, 6)
            tp2r     = round(entry + 2 * risk_amt, 6)
            tp3r     = round(entry + 3 * risk_amt, 6)
            sl_desc  = f"{SL_LABELS[sl_dist]} below entry"

        wr_str   = f"{best_wr:.1f}%" if best_wr else "N/A"

        if best_tp == "2R":
            tp_lines = f'<div class="signal-line">TP &nbsp;(2R): <span>${tp2r:,.6f}</span></div>'
        elif best_tp == "3R":
            tp_lines = f'<div class="signal-line">TP &nbsp;(3R): <span>${tp3r:,.6f}</span></div>'
        else:
            tp_lines = (
                f'<div class="signal-line">TP1 (1R — partial exit): <span>${tp1r:,.6f}</span></div>'
                f'<div class="signal-line">TP2 (2R — full exit):    <span>${tp2r:,.6f}</span></div>'
            )

        # ADX context for signal card
        adx_s_v  = sr["_adx_s"]
        adx_h_v  = sr["_adx_h"]
        dip_v    = sr["_dip"]
        dim_v    = sr["_dim"]
        adx_s_disp = f"{adx_s_v:.1f}" if pd.notna(adx_s_v) else "N/A"
        adx_h_disp = f"{adx_h_v:.1f}" if pd.notna(adx_h_v) else "N/A"
        adx_s_color = "#64ffda" if pd.notna(adx_s_v) and adx_s_v >= 25 else "#ff6b6b" if pd.notna(adx_s_v) else "#8892b0"
        adx_h_color = "#64ffda" if pd.notna(adx_h_v) and adx_h_v >= 25 else "#ff6b6b" if pd.notna(adx_h_v) else "#8892b0"

        # DI direction — gap magnitude matters, not just which is larger
        if pd.notna(dip_v) and pd.notna(dim_v):
            di_gap = abs(dip_v - dim_v)
            if di_gap < 5:
                strength_label = "neutral"
            elif di_gap < 10:
                strength_label = "weak"
            elif di_gap < 20:
                strength_label = "moderate"
            else:
                strength_label = "strong"

            if dip_v > dim_v:
                di_dir_str = f"▲ Bullish {strength_label} (DI+{dip_v:.0f} vs DI-{dim_v:.0f}, gap={di_gap:.0f})"
                # Color: green only if aligns with LONG + gap meaningful
                if direction == "long" and di_gap >= 10:
                    di_dir_color = "#64ffda"
                elif direction == "short" and di_gap >= 10:
                    di_dir_color = "#ff6b6b"  # counter-trend for short
                else:
                    di_dir_color = "#ffd700"  # weak/neutral = yellow
            else:
                di_dir_str = f"▼ Bearish {strength_label} (DI-{dim_v:.0f} vs DI+{dip_v:.0f}, gap={di_gap:.0f})"
                if direction == "short" and di_gap >= 10:
                    di_dir_color = "#64ffda"
                elif direction == "long" and di_gap >= 10:
                    di_dir_color = "#ff6b6b"  # counter-trend for long
                else:
                    di_dir_color = "#ffd700"
        else:
            di_dir_str   = "N/A"
            di_dir_color = "#8892b0"

        if pd.notna(adx_s_v) and pd.notna(adx_h_v):
            if adx_s_v >= 25 and adx_h_v >= 25:
                adx_verdict = "✅ Both TFs trending — stronger signal context"
                adx_vcolor  = "#64ffda"
            elif adx_s_v < 25 and adx_h_v < 25:
                adx_verdict = "⚪ Both TFs ranging — momentum candle in choppy market"
                adx_vcolor  = "#8892b0"
            else:
                adx_verdict = "⚠️ Mixed — one TF trending, one ranging"
                adx_vcolor  = "#ffd700"
        else:
            adx_verdict = "ADX data unavailable"
            adx_vcolor  = "#8892b0"

        # Build confluence checklist for signal card
        check_body = "✅ Body"
        check_vol  = "✅ Volume"
        check_di10 = "✅ DI≥10" if sr["DI≥10"] == "✅" else "❌ DI≥10"
        check_di20 = "✅ DI≥20" if sr["DI≥20"] == "✅" else "❌ DI≥20"

        # EMA checklist items
        _ema_s_ok  = sr["_ema_same_ok"]
        _ema_h_ok  = sr["_ema_htf_ok"]
        _ema_s_val = sr["_ema_s_val"]
        _ema_h_val = sr["_ema_h_val"]
        _ema_s_str = f"{_ema_s_val:.4f}" if pd.notna(_ema_s_val) else "N/A"
        _ema_h_str = f"{_ema_h_val:.4f}" if pd.notna(_ema_h_val) else "N/A"
        check_ema_same = (f"{'✅' if _ema_s_ok else '❌'} EMA{ema_period} ({tf_label})"
                          f" <span style='color:#8892b0;font-size:11px;'>= {_ema_s_str}</span>")
        if ema_h_col and _ema_h_ok is not None:
            check_ema_htf = (f"{'✅' if _ema_h_ok else '❌'} EMA{ema_htf_period} ({htf_label} HTF)"
                             f" <span style='color:#8892b0;font-size:11px;'>= {_ema_h_str}</span>")
        else:
            check_ema_htf = None

        if adx_filter_on and adx_di_gap_min > 0:
            req_label  = f"Required: Body + Volume + DI gap ≥ {adx_di_gap_min:.0f}"
            checklist  = f"{check_body} &nbsp; {check_vol} &nbsp; {'✅' if sr['DI≥10'] == '✅' and adx_di_gap_min <= 10 or sr['DI≥20'] == '✅' and adx_di_gap_min <= 20 else '✅'} DI gap ≥ {adx_di_gap_min:.0f}"
        else:
            req_label  = "Required: Body + Volume &nbsp;|&nbsp; DI columns = extra confluence"
            checklist  = f"{check_body} &nbsp; {check_vol} &nbsp; {check_di10} &nbsp; {check_di20}"

        # Append EMA checklist items
        _ema_status = "ON — must pass to qualify" if ema_filter else "OFF (informational only)"
        ema_checklist_html = (
            f'<div class="signal-line" style="margin-top:6px;font-size:12px;color:#8892b0;">'
            f'EMA Trend Filter: <span style="color:#64ffda;">{_ema_status}</span></div>'
            f'<div class="signal-line" style="letter-spacing:1px;">{check_ema_same}'
            + (f' &nbsp; {check_ema_htf}' if check_ema_htf else '') +
            '</div>'
        )

        # ── Derivatives & sentiment context for signal card ──────────────────
        _sc_fr_rate = _fr_data.get("rate", 0.0)
        _sc_oi_chg  = _oi_data.get("oi_change_pct", 0.0)
        _sc_fg_val  = _fg_data.get("value", 50)
        _sc_fg_clf  = _fg_data.get("classification", "Neutral")

        # Funding rate signal interpretation
        if _fr_data.get("ok"):
            _fr_color = "#64ffda" if _sc_fr_rate < -0.01 else "#f85149" if _sc_fr_rate > 0.03 else "#8892b0"
            if direction == "long":
                _fr_verdict = ("✅ CONFIRMS — shorts crowded, squeeze potential"
                               if _sc_fr_rate < -0.01 else
                               "⚠️ WARNING — longs crowded, squeeze risk"
                               if _sc_fr_rate > 0.03 else "Neutral")
            else:
                _fr_verdict = ("✅ CONFIRMS — longs crowded, flush potential"
                               if _sc_fr_rate > 0.03 else
                               "⚠️ WARNING — shorts crowded, squeeze risk"
                               if _sc_fr_rate < -0.01 else "Neutral")
            _fr_html = (f'<div class="signal-line">Funding rate: '
                        f'<span style="color:{_fr_color};">{_sc_fr_rate*100:.4f}%</span>'
                        f' — <span style="color:{_fr_color};">{_fr_verdict}</span></div>')
        else:
            _fr_html = ""

        # OI interpretation
        if _oi_data.get("ok"):
            _oi_color = "#64ffda" if _sc_oi_chg > 5 else "#f85149" if _sc_oi_chg < -5 else "#8892b0"
            _oi_dir   = "▲ Rising" if _sc_oi_chg > 2 else "▼ Falling" if _sc_oi_chg < -2 else "Flat"
            if direction == "long":
                _oi_verdict = ("✅ New money entering — conviction" if _sc_oi_chg > 5 else
                               "⚠️ OI falling — possible short cover, not fresh longs" if _sc_oi_chg < -5 else "Neutral")
            else:
                _oi_verdict = ("✅ New shorts entering — conviction" if _sc_oi_chg > 5 else
                               "⚠️ OI falling — squeeze risk" if _sc_oi_chg < -5 else "Neutral")
            _oi_html = (f'<div class="signal-line">Open interest 24h: '
                        f'<span style="color:{_oi_color};">{_oi_dir} ({_sc_oi_chg:+.1f}%)</span>'
                        f' — <span style="color:{_oi_color};">{_oi_verdict}</span></div>')
        else:
            _oi_html = ""

        # vol_delta_20 flow proxy
        _vd20 = float(row.get("vol_delta_20", 0.0) or 0.0)
        _vd20_color = "#64ffda" if _vd20 > 0 else "#f85149" if _vd20 < 0 else "#8892b0"
        _vd20_label = "Accumulation pattern" if _vd20 > 0 else "Distribution pattern" if _vd20 < 0 else "Neutral"
        _vd20_html  = (f'<div class="signal-line">Vol flow (20-bar): '
                       f'<span style="color:{_vd20_color};">{_vd20_label}</span>'
                       f'<span style="color:#8892b0;font-size:11px;"> (vol_delta_20={_vd20:+.0f})</span></div>')

        # Derivatives block — only if at least one data point available
        _deriv_block = ""
        if _fr_html or _oi_html:
            _deriv_block = (
                '<hr style="border-color:#1e2d1e; margin:8px 0;">'
                '<div class="signal-line" style="font-size:12px;color:#8892b0;">📊 Derivatives & Sentiment</div>'
                + _fr_html + _oi_html + _vd20_html
            )

        st.markdown(f"""<div class="signal-card">
<h4>SIGNAL FOUND — {date_str}</h4>
<div class="signal-line">Trigger close: <span>${close:,.6f}</span> &nbsp;|&nbsp;
Body: <span>{sr['Body %']:.1f}%</span> &nbsp;|&nbsp;
Vol mult: <span>{sr['Vol Mult']:.2f}x</span></div>
<hr style="border-color:#1e2d1e; margin:8px 0;">
<div class="signal-line" style="font-size:12px;color:#8892b0;">{req_label}</div>
<div class="signal-line" style="letter-spacing:2px;">{checklist}</div>
{ema_checklist_html}<hr style="border-color:#1e2d1e; margin:8px 0;">
<div class="signal-line">ADX strength ({tf_label}): <span style="color:{adx_s_color};">{adx_s_disp}</span>
&nbsp;|&nbsp; ADX strength ({htf_label} HTF): <span style="color:{adx_h_color};">{adx_h_disp}</span>
&nbsp;|&nbsp; <span style="color:{adx_vcolor};">{adx_verdict}</span></div>
<div class="signal-line">Trend direction: <span style="color:{di_dir_color};">{di_dir_str}</span>
&nbsp;<span style="color:#8892b0;font-size:11px;">(DI− &gt; DI+ = bearish; DI+ &gt; DI− = bullish)</span></div>
{_deriv_block}
<hr style="border-color:#1e2d1e; margin:8px 0;">
<div class="signal-line">Based on historical best setup for <span>{ticker}</span>:</div>
<div class="signal-line">Entry: <span>${entry:,.6f}</span> &nbsp;({RETRACE_LABELS[best_ret]} retrace from close)</div>
<div class="signal-line">SL: &nbsp;&nbsp;<span>${sl:,.6f}</span> &nbsp;({sl_desc})</div>
{tp_lines}
<div class="signal-line" style="margin-top:8px;">Historical win rate: <span>{wr_str}</span> ({best_tp} TP)</div>
</div>""", unsafe_allow_html=True)

        # ── Quick Analyze button (Intelligence engine inline) ─────────────────
        _scan_btn_key = f"scan_analyze_{date_str.replace(' ','_').replace(':','')}"
        with st.expander(f"🧠 Analyze this signal — {date_str}", expanded=False):
            # Build enriched candle dict from live_df bar
            _scan_bar = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            _scan_bar["adx_value"] = adx_s_v if pd.notna(adx_s_v) else 0.0
            _scan_bar["di_plus"]   = dip_v   if pd.notna(dip_v)   else 0.0
            _scan_bar["di_minus"]  = dim_v   if pd.notna(dim_v)   else 0.0
            _scan_bar["_date_str"] = date_str
            # Inject live market-context features into bar for ML scoring
            _scan_bar["fear_greed_normalized"] = _sc_fg_val / 100.0
            _scan_bar["funding_rate"]          = _sc_fr_rate
            _scan_bar["oi_change_pct_24h"]     = _sc_oi_chg

            # EMA stack from live_df bar (if _clean_df ran on live_df)
            _scan_ml_model = st.session_state.get("ml_model_data")

            # Regime snapshot: use last regime from session or compute quickly
            _scan_regime = st.session_state.get("_regime_data", {})

            # ML score
            _scan_ml_result = score_candle_ml(_scan_bar, _scan_ml_model, direction) if _scan_ml_model else {"probability": 0.5, "available": False}
            _scan_ml_prob   = _scan_ml_result.get("probability", 0.5)

            # Rule-based verdict (instant, no API)
            _scan_rb = rule_based_fallback(_scan_regime, {}, _scan_ml_prob)

            col_ana_l, col_ana_r = st.columns(2)
            with col_ana_l:
                st.markdown(f"**ADX** {adx_s_disp} ({tf_label}) / {adx_h_disp} ({htf_label} HTF)")
                st.markdown(f"**Trend:** {di_dir_str}")
                _ema_line = f"EMA{ema_period}: {'✅ aligned' if sr['_ema_same_ok'] else '❌ not aligned'}"
                if sr["_ema_htf_ok"] is not None:
                    _ema_line += f" | HTF EMA{ema_htf_period}: {'✅' if sr['_ema_htf_ok'] else '❌'}"
                st.markdown(f"**{_ema_line}**")
                _scan_rb_vrd = _scan_rb.get("verdict", "WAIT")
                _vrdc = {"TRADE": "#3fb950", "WAIT": "#e3b341", "NO TRADE": "#f85149"}
                st.markdown(
                    f'<div style="margin-top:8px;padding:8px 12px;border-radius:6px;'
                    f'border:1px solid {_vrdc.get(_scan_rb_vrd,"#8b949e")};'
                    f'color:{_vrdc.get(_scan_rb_vrd,"#8b949e")};font-weight:700;">'
                    f'Rule-based: {_scan_rb_vrd}</div>',
                    unsafe_allow_html=True)
                st.caption(_scan_rb.get("full_text", "").split("\n")[-1])

            with col_ana_r:
                _scan_ai_key = st.session_state.get("anthropic_api_key", "")
                if _scan_ai_key:
                    if st.button("Run AI Analysis", key=_scan_btn_key, type="primary"):
                        with st.spinner("Calling AI…"):
                            _scan_ai_result = analyze_candle_ai(
                                _scan_bar, _scan_regime, "", _scan_ml_prob,
                                strategy_params={"retracement": best_ret,
                                                 "sl_dist": best_sl_dist,
                                                 "tp_mode": best_tp},
                                wfo_results=st.session_state.get("wfo_results"),
                                direction=direction,
                                ticker=ticker,
                                timeframe=timeframe,
                                risk_context={},
                            )
                        st.session_state[f"_scan_ai_{_scan_btn_key}"] = _scan_ai_result
                    _cached_ai = st.session_state.get(f"_scan_ai_{_scan_btn_key}")
                    if _cached_ai:
                        _ai_vrd  = _cached_ai.get("verdict", "WAIT")
                        _ai_col  = _vrdc.get(_ai_vrd, "#8b949e")
                        st.markdown(
                            f'<div style="padding:8px 12px;border-radius:6px;'
                            f'border:1px solid {_ai_col};color:{_ai_col};font-weight:700;">'
                            f'AI: {_ai_vrd}</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="font-size:11px;color:#ccd6f6;'
                            f'white-space:pre-wrap;margin-top:6px;">'
                            f'{_cached_ai.get("full_text","")}</div>',
                            unsafe_allow_html=True)
                else:
                    st.info("Add Anthropic API key in sidebar to enable AI analysis here.")
                    st.caption(f"ML probability: {_scan_ml_prob*100:.0f}%")

def _apply_strategy_to_sidebar(row: dict):
    """Store strategy as a pending dict. Applied at top of sidebar on next rerun
    BEFORE widgets are instantiated, avoiding the 'cannot modify after instantiation' error."""
    import re

    dirn      = row["direction"].upper()
    adx_label = row.get("adx_label", "No ADX filter")
    hold_label= row.get("hold_label", "No limit (TP/SL only)")

    pending = {
        "direction_radio": ("📈 Long  (Bullish candles)" if dirn == "LONG"
                            else "📉 Short (Bearish candles)"),
        "sb_timeframe":    row.get("timeframe", "1D"),
        "sb_body":         max(50, min(95, round(row["body_pct"] / 5) * 5)),
        "sb_vol":          max(1.0, min(4.0, round(float(row["vol_mult"]) * 10) / 10)),
        "adx_toggle":      adx_label != "No ADX filter",
        "di_filter_toggle": bool(re.search(r"DI\s*gap", adx_label)),
        "time_exit_toggle": "No limit" not in hold_label,
    }

    if adx_label != "No ADX filter":
        m = re.search(r"ADX\s*[≥>=]+\s*(\d+)", adx_label)
        if m:
            pending["_adx_threshold_val"] = max(15, min(40, int(m.group(1))))
        m2 = re.search(r"DI\s*gap\s*[≥>=]+\s*(\d+)", adx_label)
        if m2:
            gap = int(m2.group(1))
            pending["di_gap_slider"] = min([5,10,15,20,25], key=lambda x: abs(x-gap))

    if "No limit" not in hold_label:
        m3 = re.search(r"(\d+)", hold_label)
        if m3:
            pending["max_hold_slider"] = str(int(m3.group(1)))

    # Store as pending — applied at sidebar top on next render
    st.session_state["_pending_strategy"] = pending

    # Clear cached results so fresh Run Analysis is required
    for k in ["opt_results", "run_key", "hold_df", "hold_key", "hold_parent_run_key"]:
        st.session_state.pop(k, None)

    st.session_state["_applied_strategy_notice"] = (
        "✅ Strategy loaded into sidebar! Check settings, then click **Run Analysis**."
    )


def render_strategy_guide():
    """Comprehensive beginner guide covering every feature and the full execution workflow."""

    _BOX  = "background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px 18px;margin:8px 0;"
    _BLUE = "background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:14px 18px;margin:8px 0;"
    _GRN  = "background:#0d2818;border:1px solid #238636;border-radius:8px;padding:14px 18px;margin:8px 0;"
    _YLW  = "background:#2d2200;border:1px solid #d29922;border-radius:8px;padding:14px 18px;margin:8px 0;"
    _RED  = "background:#2d0d0d;border:1px solid #da3633;border-radius:8px;padding:14px 18px;margin:8px 0;"

    st.markdown("## 📖 Complete Beginner Guide — Momentum Candle Strategy")
    st.caption("Read top to bottom the first time. Use the section headers to navigate when you return.")

    # ─── Section 0: Quick Start ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Quick Start — 5 Steps to Your First Trade")
    st.markdown(f"""
<div style="{_BLUE}">
<ol style="color:#ccd6f6;line-height:2;margin:0;padding-left:20px;font-size:14px;">
<li><b>Strategy Finder tab</b> — Enter ticker (e.g. BTCUSDT), pick timeframe (1D for beginners), click <b>Run Analysis</b></li>
<li><b>Read the Best Setup box</b> — Optimizer shows best Body%, Volume, SL, TP with historical win rate and profit factor</li>
<li><b>Backtest Results tab</b> — Verify profit factor ≥ 1.5 and at least 20 trades before trusting the result</li>
<li><b>WFO Validation tab</b> — Run Walk-Forward Validation to confirm edge holds on unseen data (look for PASS)</li>
<li><b>Live Scanner tab</b> — Monitor for live signals; click Quick Analyze on any signal card for AI review before entering</li>
</ol>
</div>
""", unsafe_allow_html=True)

    # ─── Section 1: What Is This Strategy ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ Section 1 — What Is the Momentum Candle Strategy?")
    st.markdown("""
**Core idea:** Big institutions (banks, hedge funds) move markets. When they enter, they leave a
footprint: a candle with a **large body** AND **unusually high volume**. This tool finds those candles,
backtests entry rules on historical data, and gives you exact entry, stop loss, and take profit levels.
""")
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:8px 0;">
<div style="{_BLUE}">
<b style="color:#58a6ff;">Filter 1 — Body %</b><br>
<span style="color:#ccd6f6;font-size:13px;">
The candle body must be ≥ 70% of the total wick-to-wick range.<br>
Body = |close − open|. Range = high − low. Body% = Body ÷ Range × 100.<br><br>
<b style="color:#64ffda;">Why it matters:</b> A fat body with tiny wicks means conviction —
price went one direction and stayed there. No indecision.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Filter 2 — Volume ×</b><br>
<span style="color:#ccd6f6;font-size:13px;">
Volume must be ≥ 1.5× the 7-candle average.<br>
Vol Mult = Today's volume ÷ Avg(last 7 candles).<br><br>
<b style="color:#64ffda;">Why it matters:</b> High volume = big players participating.
A large candle on low volume is a fake-out. High volume confirms real institutional activity.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Direction</b><br>
<span style="color:#ccd6f6;font-size:13px;">
<b>Long:</b> Only bullish candles (close &gt; open, green body)<br>
<b>Short:</b> Only bearish candles (close &lt; open, red body)<br><br>
Choose in sidebar. Beginners: start with <b>Long only on 1D</b>
— crypto has long-term bullish bias and long setups are simpler to understand.
</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="{_YLW}">
<b>📐 Example:</b> BTC closes $50,000. Open $48,000. High $50,200. Low $47,900.<br>
Body = 50,000 − 48,000 = <b>$2,000</b>. Range = 50,200 − 47,900 = <b>$2,300</b>. Body% = <b>87%</b> → PASSES ✅<br>
Volume today = 15,000 BTC, 7-day avg = 8,000 BTC → Vol Mult = <b>1.87×</b> → PASSES ✅<br>
→ <b>This candle qualifies as a momentum signal.</b>
</div>
""", unsafe_allow_html=True)

    # ─── Section 2: Strategy Finder ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Section 2 — Strategy Finder Tab")
    st.markdown("""
The **Strategy Finder** sweeps through combinations of parameters and shows which settings
produced the best historical results for your ticker and timeframe. This is where you calibrate the strategy.
""")
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:8px 0;">
<div style="{_BOX}">
<b style="color:#ccd6f6;">Sidebar Inputs</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.9;">
<b>Ticker</b> — crypto pair on Binance (BTCUSDT, ETHUSDT...)<br>
<b>Timeframe</b> — 1D = daily, 4H = 4-hour, 1H = hourly<br>
<b>History</b> — how many past days to backtest<br>
<b>Min Body %</b> — body filter threshold (default 70%)<br>
<b>Min Vol ×</b> — volume multiplier (default 1.5×)<br>
<b>Direction</b> — Long (bullish) or Short (bearish)
</span>
</div>
<div style="{_BOX}">
<b style="color:#ccd6f6;">Best Setup Box (output)</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.9;">
<b>Retracement</b> — how far to wait for pullback before entering<br>
&nbsp;&nbsp;e.g. 38.2% = enter at 38.2% retrace of signal candle<br>
<b>SL Distance</b> — stop loss as % from entry<br>
<b>TP Mode</b> — 2R, 3R, or Partial (R = reward-to-risk)<br>
<b>Win Rate</b> — % of historical trades profitable<br>
<b>Profit Factor</b> — gross profit ÷ gross loss (≥ 1.5 = good)
</span>
</div>
<div style="{_BOX}">
<b style="color:#ccd6f6;">Retracement Explained</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.9;">
<b>0% (Immediate):</b> Enter next candle open after signal. Fast but less precise price.<br><br>
<b>23.6–61.8% Fib:</b> Wait for price to pull back X% of signal candle range before entering.
Better price but signals that don't retrace enough are skipped.
</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="{_GRN}">
<b style="color:#3fb950;">Win Rate + Profit Factor — read them together:</b><br>
<span style="color:#ccd6f6;font-size:13px;">
A <b>35% win rate with 3R target</b> is profitable: win $3 for every $1 lost → even winning 1 in 3 = net positive.
Never judge by win rate alone.<br><br>
<b>PF ≥ 1.5</b> = acceptable. &nbsp;<b>PF ≥ 2.0</b> = strong. &nbsp;<b>PF ≥ 3.0</b> = exceptional.<br>
Always require <b>at least 20 trades</b> — fewer trades = results could be lucky noise.
</span>
</div>
""", unsafe_allow_html=True)

    # ─── Section 3: Backtest Results ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Section 3 — Backtest Results Tab")
    st.markdown("""
After running, **Backtest Results** shows every trade the strategy would have taken historically.
Use this to understand the risk profile and verify the edge is real.
""")
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:8px 0;">
<div style="{_BOX}">
<b style="color:#58a6ff;">Key Metrics</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.9;">
📈 <b>Total Return %</b> — net profit over period<br>
🎯 <b>Win Rate</b> — wins ÷ total trades<br>
⚖️ <b>Profit Factor</b> — gross profit ÷ gross loss<br>
📉 <b>Max Drawdown</b> — worst peak-to-trough loss<br>
🔢 <b>Trade Count</b> — total qualifying signals taken
</span>
</div>
<div style="{_BOX}">
<b style="color:#58a6ff;">Trade Log Tab</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.9;">
Shows every individual trade: entry date, entry price, exit price, result (W/L), and R multiple.<br><br>
Use it to spot if losses cluster at certain times or market conditions (e.g. all losses in flat months).
</span>
</div>
<div style="{_BOX}">
<b style="color:#58a6ff;">Good Result Checklist</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.9;">
✅ PF ≥ 1.5 with ≥ 20 trades<br>
✅ Max drawdown &lt; 30%<br>
✅ Consistent across date ranges<br>
✅ Passes WFO Validation<br><br>
❌ PF 10+ with 5 trades = lucky noise
</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Section 4: ADX + EMA Filters ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📡 Section 4 — ADX Filter + EMA Trend Filter")
    st.markdown("""
Both filters are **optional** and sit in the sidebar. They reduce signal count while improving quality — fewer but higher-probability setups.
""")
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:8px 0;">
<div style="{_BLUE}">
<b style="color:#58a6ff;">ADX Filter — Is the market trending?</b><br>
<span style="color:#ccd6f6;font-size:13px;line-height:1.8;">
ADX measures <b>trend strength</b>, not direction.<br>
ADX ≥ 25 = strong trend → take signals<br>
ADX &lt; 25 = weak/ranging market → skip signals<br><br>
<b>DI Gap:</b> Requires DI+ and DI− to be separated by a minimum — confirms direction is clear, not a coin-flip.<br><br>
<b>HTF mode:</b> Check ADX on higher timeframe (4H ADX for 1H signals) for stronger confirmation.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">EMA Filter — Which direction?</b><br>
<span style="color:#ccd6f6;font-size:13px;line-height:1.8;">
EMA shows the <b>direction</b> of the trend.<br>
Close &gt; EMA 200 = bullish bias → LONG signals only<br>
Close &lt; EMA 200 = bearish bias → SHORT signals only<br><br>
<b>HTF EMA:</b> 1H signal + check 4H EMA 200 = even stronger.
Trade 1H candles only when 4H trend agrees.<br><br>
<b>Weekly EMA 50</b> is the standard for daily chart signals.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Combined = best quality signals</b><br>
<span style="color:#ccd6f6;font-size:13px;line-height:1.8;">
ADX says: <i>"Is it trending?"</i><br>
EMA says: <i>"Which direction?"</i><br><br>
Together: ADX ≥ 25 + close &gt; EMA 200 + momentum candle<br>
= strong trend + right direction + institutional move<br>
= <b style="color:#64ffda;">highest probability signals</b><br><br>
Always test with/without filters and compare PF + WFO.
</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Section 5: Market Regime Gate ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌡️ Section 5 — Market Regime Gate")
    st.markdown("""
The **Regime Gate** scores the current market environment from 0–100 and shows a colored banner (GREEN/YELLOW/RED).
It does not block signals — it tells you whether conditions favor your strategy right now.
""")
    st.markdown(f"""
<div style="{_BOX}">
<table style="width:100%;border-collapse:collapse;font-size:13px;color:#ccd6f6;">
<tr style="border-bottom:1px solid #21262d;">
  <th style="text-align:left;padding:6px 10px;color:#58a6ff;">Component</th>
  <th style="text-align:left;padding:6px 10px;color:#58a6ff;">What it measures</th>
  <th style="text-align:left;padding:6px 10px;color:#58a6ff;">Score</th>
</tr>
<tr style="border-bottom:1px solid #21262d;">
  <td style="padding:6px 10px;">ADX ≥ 25</td>
  <td style="padding:6px 10px;">Trend strength (market is directional, not choppy)</td>
  <td style="padding:6px 10px;">+25 pts</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
  <td style="padding:6px 10px;">DI+ &gt; DI−</td>
  <td style="padding:6px 10px;">Bulls outpacing bears (for long signals)</td>
  <td style="padding:6px 10px;">+25 pts</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
  <td style="padding:6px 10px;">Vol Rank &gt; 50%</td>
  <td style="padding:6px 10px;">Volume above 20-day median (active participation)</td>
  <td style="padding:6px 10px;">+25 pts</td>
</tr>
<tr>
  <td style="padding:6px 10px;">Close &gt; EMA 50</td>
  <td style="padding:6px 10px;">Price above medium-term trend line</td>
  <td style="padding:6px 10px;">+25 pts</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0;">
<div style="background:#0d2818;border:1px solid #238636;border-radius:8px;padding:12px;text-align:center;">
<b style="color:#3fb950;font-size:18px;">GREEN</b><br>
<span style="color:#ccd6f6;font-size:13px;">Score 75–100<br>All systems go.<br>Best conditions for signals.</span>
</div>
<div style="background:#2d2200;border:1px solid #d29922;border-radius:8px;padding:12px;text-align:center;">
<b style="color:#d29922;font-size:18px;">YELLOW</b><br>
<span style="color:#ccd6f6;font-size:13px;">Score 50–74<br>Moderate conditions.<br>Reduce size 50%.</span>
</div>
<div style="background:#2d0d0d;border:1px solid #da3633;border-radius:8px;padding:12px;text-align:center;">
<b style="color:#da3633;font-size:18px;">RED</b><br>
<span style="color:#ccd6f6;font-size:13px;">Score 0–49<br>Poor conditions.<br>Skip or sit out.</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Section 6: WFO Validation ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✅ Section 6 — WFO Validation Tab")
    st.markdown("""
**Walk-Forward Optimisation (WFO)** is the most important test. It proves your strategy works on
**data it has never seen before** — not just data it was optimized on.
""")
    st.markdown(f"""
<div style="{_BOX}">
<b style="color:#ccd6f6;">How WFO works:</b><br>
<ol style="color:#8892b0;font-size:13px;line-height:1.9;margin:8px 0;padding-left:20px;">
<li>Split historical data into N windows (default 5)</li>
<li>For each window: optimize on first 70% (“in-sample”) → test on remaining 30% (“out-of-sample”)</li>
<li>Report PF and win rate for each out-of-sample period</li>
<li>Consistent results across windows → <b style="color:#3fb950;">PASS</b></li>
<li>Wildly varying or mostly failing → <b style="color:#da3633;">FAIL</b> = overfit, do not trade live</li>
</ol>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0;">
<div style="{_GRN}">
<b style="color:#3fb950;">PASS signs</b><br>
<span style="color:#ccd6f6;font-size:13px;">
✅ ≥ 60% of windows profitable<br>
✅ Avg out-of-sample PF ≥ 1.2<br>
✅ No single catastrophic window<br>
✅ Consistent across timeframes
</span>
</div>
<div style="{_RED}">
<b style="color:#da3633;">FAIL warning signs</b><br>
<span style="color:#ccd6f6;font-size:13px;">
❌ &lt; 40% of windows profitable<br>
❌ Great in-sample, terrible out-of-sample<br>
❌ One window carries all profit<br>
❌ PF drops from 3.0 to 0.8 out-of-sample
</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="{_YLW}">
<b>💡 If WFO fails:</b> the strategy is overfit. Fix it by reducing complexity — fewer Fib levels,
wider SL ranges, or try a different ticker/timeframe. Do not trade an overfit strategy live.
</div>
""", unsafe_allow_html=True)

    # ─── Section 7: ML Classifier ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Section 7 — ML Classifier Tab")
    st.markdown("""
The **ML Classifier** uses machine learning (Logistic Regression with TimeSeriesSplit) to predict
whether a new momentum signal is likely to be a winner or loser before you take the trade.
""")
    st.markdown(f"""
<div style="{_BOX}">
<b style="color:#ccd6f6;">The 7 features the model learns from:</b><br>
<table style="width:100%;border-collapse:collapse;font-size:13px;color:#ccd6f6;margin-top:8px;">
<tr style="border-bottom:1px solid #21262d;">
  <th style="text-align:left;padding:5px 8px;color:#58a6ff;">Feature</th>
  <th style="text-align:left;padding:5px 8px;color:#58a6ff;">Why it predicts trade outcome</th>
</tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 8px;">Body %</td><td style="padding:5px 8px;">Stronger body = more conviction, higher follow-through probability</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 8px;">Volume Multiplier</td><td style="padding:5px 8px;">Higher volume = institutional participation = move more likely to continue</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 8px;">ATR (volatility)</td><td style="padding:5px 8px;">Market context — candle strength relative to current market activity</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 8px;">ADX</td><td style="padding:5px 8px;">Trend strength at signal time — signals in strong trends win more often</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 8px;">DI+ − DI−</td><td style="padding:5px 8px;">Directional gap — how clear the trend direction is</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 8px;">Volume Rank (20)</td><td style="padding:5px 8px;">Volume percentile over 20 periods — is today unusually active?</td></tr>
<tr><td style="padding:5px 8px;">Candle Range</td><td style="padding:5px 8px;">Absolute size — larger candle = more energy behind the move</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0;">
<div style="{_BLUE}">
<b style="color:#58a6ff;">How to use the ML score</b><br>
<span style="color:#ccd6f6;font-size:13px;line-height:1.8;">
Score &gt; 0.65 → High-probability → full position<br>
Score 0.50–0.65 → Moderate → half size or skip<br>
Score &lt; 0.50 → Low probability → skip<br><br>
ML score is a <b>second opinion</b>. Never override a WFO-validated
setup purely based on ML score.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">No lookahead bias</b><br>
<span style="color:#ccd6f6;font-size:13px;line-height:1.8;">
Uses TimeSeriesSplit: trained on past, tested on future only.
Cross-validation accuracy shown in the tab.<br><br>
<b>Accuracy &gt; 60%</b> on test set = useful predictive signal.<br>
Below 55% = near-random, don't rely on it.
</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Section 8: Intelligence Tab ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 Section 8 — Intelligence Tab")
    st.markdown("""
The **Intelligence Tab** gives a deep analysis of any historical signal candle using three engines:
a similarity engine, a regime check, and an AI assistant (Groq LLM — free, no API cost for basic use).
""")
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:8px 0;">
<div style="{_BOX}">
<b style="color:#ccd6f6;">Engine 1 — Similarity Search</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.8;">
Finds past candles with similar body%, vol, ADX, and size.
Shows: how many similar candles hit TP vs SL, and average R multiple.<br><br>
<b style="color:#64ffda;">Use it as:</b> Historical win rate for this specific type of candle.
</span>
</div>
<div style="{_BOX}">
<b style="color:#ccd6f6;">Engine 2 — Regime Score</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.8;">
Scores the market conditions at the time of that signal (same 0–100 regime scoring).<br><br>
<b style="color:#64ffda;">Use it as:</b> Signals fired in GREEN regime historically outperform those fired in RED.
</span>
</div>
<div style="{_BOX}">
<b style="color:#ccd6f6;">Engine 3 — AI Analysis (Groq)</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.8;">
Sends candle data, ADX, EMA stack, vol delta, regime score, and best backtested params to Groq LLM.<br><br>
Returns: quality score, risk notes, exact entry/SL/TP prices, hold time estimate.<br><br>
<b style="color:#64ffda;">Use it as:</b> Expert synthesis of all data into one trade decision.
</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="{_YLW}">
<b>⚠️ Note:</b> Intelligence tab analyzes <b>historical signal candles</b> from past data.
To analyze the <b>most recent live signal</b>, go to <b>Live Scanner → Quick Analyze button</b>.
This sends the live candle directly to the AI engine.
</div>
""", unsafe_allow_html=True)

    # ─── Section 9: Live Scanner ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📡 Section 9 — Live Scanner Tab")
    st.markdown("""
The **Live Scanner** checks current candle data in real time and alerts you when a momentum signal
is forming or has just completed. It runs automatically when you open the tab.
""")
    st.markdown(f"""
<div style="{_BOX}">
<b style="color:#ccd6f6;">Signal Card explained:</b><br>
<span style="color:#8892b0;font-size:13px;line-height:1.9;">
🟢 <b>Signal type</b> — LONG or SHORT<br>
📅 <b>Date/Time</b> — when candle closed (WIB timezone for 1H/4H)<br>
📐 <b>Body %</b> — how strong the candle body is<br>
📊 <b>Vol ×</b> — volume multiplier vs 7-bar average<br>
🎯 <b>Entry</b> — suggested entry price from best backtested params<br>
🛑 <b>SL</b> — stop loss price<br>
🟢 <b>TP1 / TP2 / TP3</b> — take profit targets (2R, 3R, or Partial)<br>
🤖 <b>Quick Analyze</b> — sends this signal to AI for full Intelligence analysis
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0;">
<div style="{_GRN}">
<b style="color:#3fb950;">When to act</b><br>
<span style="color:#ccd6f6;font-size:13px;line-height:1.8;">
✅ Regime banner is GREEN<br>
✅ ML score &gt; 0.60<br>
✅ AI says HIGH quality<br>
✅ Signal matches HTF EMA direction<br>
✅ ADX ≥ 25 at signal time
</span>
</div>
<div style="{_RED}">
<b style="color:#da3633;">When to skip</b><br>
<span style="color:#ccd6f6;font-size:13px;line-height:1.8;">
❌ Regime banner is RED<br>
❌ ML score &lt; 0.50<br>
❌ AI says LOW quality or avoid<br>
❌ Signal against HTF trend<br>
❌ Major news / high uncertainty
</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Section 10: Full Execution Workflow ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ Section 10 — Full Execution Workflow")
    st.markdown("The complete decision tree from initial setup to live trade:")
    st.markdown(f"""
<div style="{_BOX}">
<ol style="color:#ccd6f6;font-size:13px;line-height:2.2;margin:0;padding-left:20px;">
<li><b style="color:#58a6ff;">Choose ticker + timeframe</b> — Beginners: BTCUSDT + 1D. More advanced: ETH + 4H.</li>
<li><b style="color:#58a6ff;">Run Strategy Finder</b> — Get Best Setup. Require PF ≥ 1.5 and ≥ 20 trades.</li>
<li><b style="color:#58a6ff;">Test filters</b> — Enable ADX, re-run. If PF improves, keep it. Same for EMA. Re-run each change.</li>
<li><b style="color:#58a6ff;">Run WFO Validation</b> — Must PASS. If FAIL: simplify params or change ticker/timeframe.</li>
<li><b style="color:#58a6ff;">Check Regime Banner</b> — GREEN = full size. YELLOW = half size. RED = wait.</li>
<li><b style="color:#58a6ff;">Open Live Scanner</b> — Wait for signal card. Note entry/SL/TP from card.</li>
<li><b style="color:#58a6ff;">Quick Analyze</b> — Check AI quality score, regime score, similarity stats.</li>
<li><b style="color:#58a6ff;">Check ML score</b> — ML Classifier tab. Score &gt; 0.60 = green light.</li>
<li><b style="color:#58a6ff;">Enter trade</b> — Use exactly the entry/SL/TP from signal card. No emotional changes.</li>
<li><b style="color:#58a6ff;">Manage trade</b> — Partial TP: 50% at TP1, trail the rest. 2R/3R: full exit at target.</li>
<li><b style="color:#58a6ff;">Review monthly</b> — Check Trade Log. Identify patterns in wins and losses.</li>
</ol>
</div>
""", unsafe_allow_html=True)

    # ─── Section 11: Tips ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Section 11 — Tips to Maximize Profit")
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:8px 0;">
<div style="{_BLUE}">
<b style="color:#58a6ff;">More history = better stats</b><br>
<span style="color:#ccd6f6;font-size:13px;">
≥ 365 days for 1D. ≥ 180 days for 4H. ≥ 90 days for 1H.
More data = more signals = more reliable PF and win rate numbers.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Add filters gradually</b><br>
<span style="color:#ccd6f6;font-size:13px;">
Start with no filters. Note baseline PF. Add ADX — if PF improves, keep.
Add EMA — check again. Only keep what improves PF AND passes WFO.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Hold Time Analysis</b><br>
<span style="color:#ccd6f6;font-size:13px;">
Check Advanced Stats tab for optimal hold period.
Exiting after 3–5 candles sometimes beats waiting for SL/TP,
especially in choppy or reversing markets.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Scale with regime</b><br>
<span style="color:#ccd6f6;font-size:13px;">
GREEN → 100% size. YELLOW → 50–75%. RED → skip or 25%.
This is regime-adjusted position sizing — the same way professional funds allocate risk.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Partial TP</b><br>
<span style="color:#ccd6f6;font-size:13px;">
Partial exit (50% at 2R, trail rest) often beats single full exits in trending markets.
Locks in profit while letting winners run. Test both and use what WFO validates.
</span>
</div>
<div style="{_BLUE}">
<b style="color:#58a6ff;">Paper trade first</b><br>
<span style="color:#ccd6f6;font-size:13px;">
Before risking real money, paper trade 10–15 signals.
Check that live execution matches backtest. The Transaction Cost setting (default 0.1%) simulates fees.
</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Closing ─────────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
<div style="{_BOX}">
<h4 style="color:#da3633;margin-top:0;">❌ Common Mistakes to Avoid</h4>
<ul style="color:#ccd6f6;font-size:13px;line-height:1.9;margin:0;padding-left:20px;">
<li><b>Cherry-picking results</b> — use what the optimizer recommends, not what looks best to you</li>
<li><b>Over-optimizing</b> — PF 99 with 5 trades is lucky noise, not a real edge</li>
<li><b>Ignoring WFO</b> — a FAIL means overfit; do not trade it live no matter how good backtest looks</li>
<li><b>Trading RED regime</b> — choppy, trendless markets erode your edge fast</li>
<li><b>Moving the SL away from entry</b> — SL defines your max loss; it is non-negotiable</li>
<li><b>Too little 1H history</b> — 1H needs 180+ days; 3 months gives too few clean signals</li>
<li><b>Expecting 80% win rates</b> — 40% win rate with 3R is highly profitable; think in expectancy</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#8892b0;font-size:13px;padding:12px;">
📌 Ready to start? Go to <b>Strategy Finder</b> tab → enter your ticker → click <b>Run Analysis</b>
</div>""", unsafe_allow_html=True)


# ─── Advanced Statistics ───────────────────────────────────────────────────────

def render_advanced_stats(trades: list, ticker: str, timeframe: str):
    """Render 8 advanced statistical panels about a trade list."""
    if not trades:
        st.info("Run a backtest first to see advanced statistics.")
        return

    r_arr  = np.array([t["r_mult"] for t in trades])
    wins   = r_arr[r_arr > 0]
    losses = r_arr[r_arr <= 0]
    wr     = len(wins) / len(r_arr) if len(r_arr) else 0
    avg_rr = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 1.0
    expectancy = r_arr.mean()
    edge       = wr - (1 - wr) / avg_rr if avg_rr > 0 else -1

    # ── Summary banner ─────────────────────────────────────────────────────────
    _edge_color  = "#64ffda" if edge > 0 else "#ff6b6b"
    _edge_label  = f"+{edge:.3f}" if edge > 0 else f"{edge:.3f}"
    _exp_color   = "#64ffda" if expectancy > 0 else "#ff6b6b"
    _exp_label   = f"+{expectancy:.3f}R" if expectancy > 0 else f"{expectancy:.3f}R"
    _wr_color    = "#64ffda" if wr >= 0.5 else "#ffd700" if wr >= 0.35 else "#ff6b6b"
    _rr_color    = "#64ffda" if avg_rr >= 2.0 else "#ffd700" if avg_rr >= 1.0 else "#ff6b6b"
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
            gap:10px;margin:0 0 16px 0;">
  <div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;
              padding:14px;text-align:center;">
    <div style="color:#8892b0;font-size:11px;text-transform:uppercase;
                letter-spacing:1px;">Win Rate</div>
    <div style="color:{_wr_color};font-size:22px;font-weight:700;
                margin-top:4px;">{wr*100:.1f}%</div>
    <div style="color:#8892b0;font-size:11px;">{len(wins)}W / {len(losses)}L</div>
  </div>
  <div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;
              padding:14px;text-align:center;">
    <div style="color:#8892b0;font-size:11px;text-transform:uppercase;
                letter-spacing:1px;">Avg Win:Loss</div>
    <div style="color:{_rr_color};font-size:22px;font-weight:700;
                margin-top:4px;">{avg_rr:.2f}×</div>
    <div style="color:#8892b0;font-size:11px;">avg win / avg loss</div>
  </div>
  <div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;
              padding:14px;text-align:center;">
    <div style="color:#8892b0;font-size:11px;text-transform:uppercase;
                letter-spacing:1px;">Expectancy</div>
    <div style="color:{_exp_color};font-size:22px;font-weight:700;
                margin-top:4px;">{_exp_label}</div>
    <div style="color:#8892b0;font-size:11px;">avg R per trade</div>
  </div>
  <div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;
              padding:14px;text-align:center;">
    <div style="color:#8892b0;font-size:11px;text-transform:uppercase;
                letter-spacing:1px;">Edge</div>
    <div style="color:{_edge_color};font-size:22px;font-weight:700;
                margin-top:4px;">{_edge_label}</div>
    <div style="color:#8892b0;font-size:11px;">{'positive ✅' if edge > 0 else 'negative ❌'}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    if edge <= 0:
        st.error(
            "❌ **This setup has negative edge** — wins don't cover losses over time. "
            "Risk of Ruin (panel F) will be 100% and Kelly (panel G) will be 0%. "
            "Consider switching to a different ticker, timeframe, or retracement level "
            "using the **Strategy Finder** tab.")

    DAY_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    MON_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── A: Session Performance ─────────────────────────────────────────────────
    st.markdown("### A — Session Performance (WIB)")
    if timeframe in ("4H", "1H"):
        sess_data: dict = {}
        for t in trades:
            s = t.get("session", "") or "Unknown"
            sess_data.setdefault(s, {"wins": 0, "total": 0})
            sess_data[s]["total"] += 1
            if t["r_mult"] > 0:
                sess_data[s]["wins"] += 1
        if sess_data:
            sess_order = ["NY+London", "London", "Asian", "Dead Zone", "Unknown"]
            s_labels = [s for s in sess_order if s in sess_data]
            s_wrs    = [sess_data[s]["wins"] / sess_data[s]["total"] * 100 for s in s_labels]
            s_counts = [sess_data[s]["total"] for s in s_labels]
            sfig = go.Figure(go.Bar(
                x=s_labels, y=s_wrs,
                marker_color=["#64ffda" if w >= 50 else "#ff6b6b" for w in s_wrs],
                text=[f"{w:.1f}% ({c} trades)" for w, c in zip(s_wrs, s_counts)],
                textposition="outside"))
            _dark(sfig, "Win Rate by Session (WIB)", yaxis=dict(range=[0, 110]),
                  yaxis_title="Win Rate %", xaxis_title="Session")
            st.plotly_chart(sfig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No session data — re-run analysis on 4H or 1H timeframe.")
    else:
        st.info("Session analysis applies to **4H and 1H** timeframes only. "
                "Switch timeframe and re-run to see session breakdowns.")

    # ── B: Day of Week ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### B — Day of Week Analysis")
    dow_data: dict = {i: {"wins": 0, "total": 0} for i in range(7)}
    for t in trades:
        ed = t.get("exit_date")
        if ed:
            d = pd.Timestamp(ed).dayofweek
            dow_data[d]["total"] += 1
            if t["r_mult"] > 0:
                dow_data[d]["wins"] += 1
    dow_wrs    = [dow_data[i]["wins"] / dow_data[i]["total"] * 100
                  if dow_data[i]["total"] else 0 for i in range(7)]
    dow_counts = [dow_data[i]["total"] for i in range(7)]
    dfig = go.Figure(go.Bar(
        x=DAY_NAMES, y=dow_wrs,
        marker_color=["#64ffda" if w >= 50 else "#ff6b6b" for w in dow_wrs],
        text=[f"{w:.1f}%<br>({c})" for w, c in zip(dow_wrs, dow_counts)],
        textposition="outside"))
    _dark(dfig, "Win Rate by Day of Week", yaxis=dict(range=[0, 110]),
          yaxis_title="Win Rate %", xaxis_title="Day")
    st.plotly_chart(dfig, use_container_width=True, config={"displayModeBar": False})

    # ── C: Monthly Seasonality ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### C — Monthly Seasonality")
    mon_data: dict = {i: {"wins": 0, "total": 0} for i in range(1, 13)}
    for t in trades:
        ed = t.get("exit_date")
        if ed:
            m = pd.Timestamp(ed).month
            mon_data[m]["total"] += 1
            if t["r_mult"] > 0:
                mon_data[m]["wins"] += 1
    mon_wrs    = [mon_data[i]["wins"] / mon_data[i]["total"] * 100
                  if mon_data[i]["total"] else 0 for i in range(1, 13)]
    mon_counts = [mon_data[i]["total"] for i in range(1, 13)]
    mfig = go.Figure(go.Bar(
        x=MON_NAMES, y=mon_wrs,
        marker_color=["#64ffda" if w >= 50 else "#ff6b6b" for w in mon_wrs],
        text=[f"{w:.1f}%<br>({c})" for w, c in zip(mon_wrs, mon_counts)],
        textposition="outside"))
    _dark(mfig, "Win Rate by Month", yaxis=dict(range=[0, 110]),
          yaxis_title="Win Rate %", xaxis_title="Month")
    st.plotly_chart(mfig, use_container_width=True, config={"displayModeBar": False})

    # ── D: Consecutive Trade Analysis ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### D — Consecutive Trade Analysis")
    st.caption("Are wins/losses streaky, or are trades independent?")
    sorted_t = sorted(trades,
                      key=lambda t: t["entry_date"] if t["entry_date"] else datetime.min)
    _after_w, _after_l, _after_2l, _after_3l = [], [], [], []
    for i in range(1, len(sorted_t)):
        prev     = sorted_t[i - 1]["r_mult"]
        curr     = sorted_t[i]["r_mult"]
        win_now  = curr > 0
        if prev > 0:
            _after_w.append(win_now)
        else:
            _after_l.append(win_now)
            if i >= 2 and sorted_t[i - 2]["r_mult"] <= 0:
                _after_2l.append(win_now)
            if i >= 3 and sorted_t[i - 2]["r_mult"] <= 0 and sorted_t[i - 3]["r_mult"] <= 0:
                _after_3l.append(win_now)

    def _pct(lst):
        return f"{np.mean(lst)*100:.1f}% ({len(lst)} trades)" if lst else "N/A"

    con_rows = [
        {"Scenario": "After a WIN → next trade WR",     "Win Rate": _pct(_after_w)},
        {"Scenario": "After a LOSS → next trade WR",    "Win Rate": _pct(_after_l)},
        {"Scenario": "After 2 consecutive losses → WR", "Win Rate": _pct(_after_2l)},
        {"Scenario": "After 3 consecutive losses → WR", "Win Rate": _pct(_after_3l)},
    ]
    st.dataframe(pd.DataFrame(con_rows), use_container_width=True)
    if _after_w and _after_l:
        diff = abs(np.mean(_after_w) - np.mean(_after_l))
        if diff < 0.05:
            st.success("✅ Trades appear statistically independent — no strong streak bias.")
        else:
            st.warning(f"⚠️ Streak bias detected ({diff*100:.1f}% difference). "
                       "Results may cluster — consider trade spacing.")

    # ── E: Volatility Regime Performance ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### E — Volatility Regime Performance")
    adx_vals = np.array([t.get("adx_value", 0) or 0 for t in trades])
    if adx_vals.max() > 1:
        p33, p66 = np.percentile(adx_vals, 33), np.percentile(adx_vals, 66)
        vol_buckets = {
            "Low ADX (< 33rd pct)":     [],
            "Medium ADX":               [],
            "High ADX (> 66th pct)":    [],
        }
        for t, av in zip(trades, adx_vals):
            if av <= p33:
                vol_buckets["Low ADX (< 33rd pct)"].append(t["r_mult"])
            elif av <= p66:
                vol_buckets["Medium ADX"].append(t["r_mult"])
            else:
                vol_buckets["High ADX (> 66th pct)"].append(t["r_mult"])
        vol_rows = []
        for label, rmults in vol_buckets.items():
            if not rmults:
                continue
            ra   = np.array(rmults)
            _w   = ra[ra > 0]; _l = ra[ra <= 0]
            _pf  = _w.sum() / abs(_l.sum()) if len(_l) else 99.0
            vol_rows.append({
                "Volatility Regime": label,
                "Trades":            len(ra),
                "Win Rate":          f"{len(_w)/len(ra)*100:.1f}%",
                "Profit Factor":     f"{min(_pf, 99.0):.2f}",
            })
        if vol_rows:
            st.dataframe(pd.DataFrame(vol_rows), use_container_width=True)
    else:
        st.info("Enable ADX Filter in the sidebar and re-run to see volatility regime analysis.")

    # ── F: Risk of Ruin Calculator ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### F — Risk of Ruin Calculator")
    rf1, rf2, rf3 = st.columns(3)
    with rf1:
        rr_risk_pct = st.number_input("Risk per trade (%)", 0.5, 20.0, 2.0, 0.5,
                                       key="adv_ror_risk") / 100
    with rf2:
        rr_account  = st.number_input("Account size ($)", 100, 1_000_000, 10_000, 100,
                                       key="adv_ror_account")
    with rf3:
        st.metric("Historical Win Rate", f"{wr*100:.1f}%")

    edge = wr - (1 - wr) / avg_rr if avg_rr > 0 else -1  # already computed above; re-assign for clarity

    # Show edge explanation before the metric
    if edge <= 0:
        st.markdown("""<div style="background:#2d1a1a;border:1px solid #ff6b6b;border-radius:8px;
padding:12px 16px;margin:8px 0;">
<b style="color:#ff6b6b;">⚠️ Negative edge detected</b><br>
<span style="color:#ccd6f6;font-size:13px;">
The current setup has a <b>negative mathematical expectancy</b>:
wins × win-rate &lt; losses × loss-rate.<br>
This means the strategy is unprofitable at these parameters — Risk of Ruin is 100%
because over enough trades the account will eventually reach zero.<br>
<b>Action:</b> try a different retracement level, tighter stop, or longer history to find
a setup where avg win &gt; avg loss.
</span></div>""", unsafe_allow_html=True)
        ror = 100.0
    else:
        try:
            ror = ((1 - edge) / (1 + edge)) ** (1 / rr_risk_pct) * 100
        except Exception:
            ror = 100.0

    ror_color = "green" if ror < 1 else ("red" if ror > 5 else "")
    st.markdown(metric_card("Risk of Ruin", f"{ror:.2f}%", ror_color), unsafe_allow_html=True)

    _exp_double   = (np.log(2) / max(edge, 1e-9) / rr_risk_pct) if edge > 0 else float("inf")
    _max_streak   = int(np.log(0.05) / np.log(1 - wr)) if 0 < wr < 1 else 0
    c_ror1, c_ror2 = st.columns(2)
    c_ror1.metric("Expected trades to double account",
                   f"{_exp_double:.0f}" if _exp_double < 1e6 else "∞")
    c_ror2.metric("Max losing streak (95% confidence)", str(_max_streak))

    # ── G: Kelly Criterion ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### G — Optimal Position Sizing (Kelly Criterion)")
    kelly     = wr - (1 - wr) / avg_rr if avg_rr > 0 else 0
    half_k    = max(kelly / 2, 0)
    quarter_k = max(kelly / 4, 0)
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Full Kelly (aggressive)",      f"{kelly*100:.1f}%")
    kc2.metric("Half Kelly (recommended)",     f"{half_k*100:.1f}%")
    kc3.metric("Quarter Kelly (conservative)", f"{quarter_k*100:.1f}%")
    if kelly <= 0:
        st.error("❌ Kelly = 0% — the strategy has no positive edge at current parameters. "
                 "Do not trade this setup. Look for a higher win rate or better R:R ratio.")
    else:
        st.warning("⚠️ Never exceed Half Kelly — accounts blow up with Full Kelly due to variance.")

    # ── H: Profit Distribution ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### H — Profit Distribution (R Multiples)")
    st.caption("Where do your wins and losses cluster?")
    hfig = go.Figure()
    hfig.add_trace(go.Histogram(
        x=r_arr, nbinsx=40,
        marker_color=["#64ffda" if v > 0 else "#ff6b6b" for v in r_arr],
        marker_line_color="#0d1117", marker_line_width=0.5,
        name="R Multiple",
        hovertemplate="R: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    hfig.add_vline(x=0, line_dash="dash", line_color="#8892b0", line_width=1)
    hfig.add_vline(x=r_arr.mean(), line_dash="dot", line_color="#ffd700", line_width=1,
                   annotation_text=f"Avg: {r_arr.mean():.2f}R",
                   annotation_position="top right", annotation_font_color="#ffd700")
    _dark(hfig, "R Multiple Distribution", xaxis_title="R Multiple", yaxis_title="Frequency")
    st.plotly_chart(hfig, use_container_width=True, config={"displayModeBar": False})

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
                "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore",
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

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
        # Never let SL be tighter than 0.8% or wider than 6% of close
        sl_dist   = max(0.008, min(0.06, (close_px - struct_sl) / close_px))
    else:
        struct_sl = high_px + atr_buffer * 0.5
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
        df = _scanner_fetch_candles(symbol, interval, limit=120)
        if df.empty or len(df) < 22:
            continue

        try:
            adx_df = calculate_adx(df)
        except Exception:
            adx_df = pd.DataFrame()

        # Check last 3 CLOSED candles (skip index -1 = current open candle)
        for bar_offset in [1, 2, 3]:
            bar_idx = len(df) - bar_offset - 1   # -1 skips the live candle
            if bar_idx < 14:   # need enough bars for indicators to warm up
                continue

            for direction in directions:
                sig = _scanner_score_signal(
                    df, adx_df, bar_idx, direction,
                    tf, symbol, min_body_pct, min_vol_mult,
                )
                if sig is None:
                    continue

                recency_pts        = _RECENCY_PTS.get(bar_offset, 0)
                sig["bar_offset"]  = bar_offset
                sig["score"]       = round(sig["base_score"] + recency_pts, 2)
                # Convert UTC → WIB (UTC+7) for display
                _ts_utc = pd.Timestamp(df.index[bar_idx])
                _ts_wib = _ts_utc + pd.Timedelta(hours=7)
                sig["candle_date"] = _ts_wib.strftime("%Y-%m-%d %H:%M WIB")
                results.append(sig)

    return results


def _scanner_ai_verdict(sig: dict, ml: dict = None, bt: dict = None) -> dict:
    """
    Call Groq LLM to synthesize all evidence: signal features, ML probability,
    full multi-method backtest comparison, and the enhanced trade plan prices.
    Returns dict with: verdict, confidence, rationale, risk, execution, source.
    """
    api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        return {
            "verdict": "NO KEY",
            "confidence": "",
            "rationale": "Add a free Groq API key in the sidebar (AI Analysis section) to enable AI confirmation.",
            "risk": "",
            "execution": "",
            "source": "",
        }

    ema_status = (
        "fully aligned"   if sig.get("ema_full")    else
        "partially aligned" if sig.get("ema_partial") else
        "not aligned"
    )
    direction     = sig["direction"].upper()
    reasons_text  = "\n".join(f"- {r}" for r in sig.get("reasons", []))
    _etp          = sig.get("_trade_plan", {})

    # ── ML section ─────────────────────────────────────────────────────────
    ml_section = f"ML Probability: {ml['pct']:.1f}% ({ml['label']})" if ml else "ML: not computed"

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
            f"Management for best: {best.get('mgmt','Simple')} with {best.get('sl_label','Fixed SL')}"
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

    prompt = f"""You are a professional momentum trading analyst synthesizing ALL evidence for a trade decision. Be decisive and data-driven.

=== SIGNAL ===
Symbol: {sig['symbol']} | Timeframe: {sig['timeframe']} | Direction: {direction}
Composite Score: {sig['score']:.1f}/100 | Recency: {sig.get('bar_offset',1)} candle(s) ago
Body: {sig['body_pct']:.1f}% | Volume: {sig['vol_mult']:.2f}x | ADX: {sig['adx']:.1f}
DI+: {sig['di_plus']:.1f} vs DI-: {sig['di_minus']:.1f} | ATR Ratio: {sig['atr_ratio']:.2f}
EMA Stack: {ema_status} | Regime: {sig['regime']} ({sig['regime_score']}/100)

=== ML ENGINE ===
{ml_section}

=== BACKTEST (500-bar historical similar setups) ===
{bt_section}

=== ALL ENTRY PRICE LEVELS ===
{price_ref}

=== SELECTION CRITERIA ===
{reasons_text}

DECISION RULES (apply strictly):
- If best backtest EV < 0 with n >= 8: NO TRADE
- If best WR < 40% with n >= 8: NO TRADE  
- If ML < 50%: lean WAIT or NO TRADE
- If all three engines (score, ML, backtest) agree: HIGH confidence
- If signal is >1 candle old and best zone = Aggressive: WAIT for retrace

Respond in EXACTLY this format (no markdown, no extra text):
VERDICT: [TRADE / WAIT / NO TRADE]
CONFIDENCE: [HIGH / MEDIUM / LOW]
RATIONALE: [3 sentences max. Cite the specific win rate, EV, and ML % numbers. State what the strongest and weakest factor is.]
EXECUTION: [If TRADE: state exact entry zone, entry price, SL price, TP1 and TP2 prices, and management (Simple/Partial/Trailing). If WAIT or NO TRADE: state what condition must change.]
RISK: [1 sentence on the main failure mode for this specific setup.]"""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model":       "llama-3.3-70b-versatile",
                "max_tokens":  350,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a systematic momentum trading analyst. "
                            "Be decisive and concise. Follow the output format exactly. "
                            "Always cite specific numbers from the data provided. "
                            "Never add extra commentary or markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

        verdict    = "WAIT"
        confidence = "MEDIUM"
        rationale  = ""
        execution  = ""
        risk       = ""

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                verdict = "NO TRADE" if "NO TRADE" in v else ("TRADE" if "TRADE" in v else "WAIT")
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().upper()
            elif line.startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()
            elif line.startswith("EXECUTION:"):
                execution = line.split(":", 1)[1].strip()
            elif line.startswith("RISK:"):
                risk = line.split(":", 1)[1].strip()

        if not rationale:
            rationale = raw[:300]

        return {
            "verdict":    verdict,
            "confidence": confidence,
            "rationale":  rationale,
            "execution":  execution,
            "risk":       risk,
            "source":     "groq/llama-3.3-70b",
        }

    except Exception as exc:
        return {
            "verdict":    "ERROR",
            "confidence": "",
            "rationale":  f"API error: {str(exc)[:100]}",
            "execution":  "",
            "risk":       "",
            "source":     "error",
        }
    api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        return {
            "verdict": "NO KEY",
            "confidence": "",
            "rationale": "Add a free Groq API key in the sidebar (AI Analysis section) to enable AI confirmation.",
            "risk": "",
            "source": "",
        }

    ema_status = (
        "fully aligned"   if sig.get("ema_full")    else
        "partially aligned" if sig.get("ema_partial") else
        "not aligned"
    )
    direction = sig["direction"].upper()
    reasons_text = "\n".join(f"- {r}" for r in sig.get("reasons", []))

    # Build ML section
    if ml:
        ml_section = (
            f"ML Heuristic Probability: {ml['pct']:.1f}% ({ml['label']})"
        )
    else:
        ml_section = "ML: not computed"

    # Build backtest section
    if bt and bt.get("error") is None and bt.get("n", 0) >= 3:
        bt_section = (
            f"Historical setups found: {bt['n']}\n"
            f"Win Rate 2R (3%): {bt['win_2r']:.1f}%\n"
            f"Win Rate 3R (4.5%): {bt['win_3r']:.1f}%\n"
            f"Expected Value 2R: {bt['ev_2r']:+.2f}R\n"
            f"Avg hold: {bt['avg_bars']:.1f} bars"
        )
    elif bt and bt.get("error"):
        bt_section = f"Backtest: {bt['error']}"
    else:
        bt_section = "Backtest: not computed"

    prompt = f"""You are synthesizing ALL available evidence for a momentum trading signal. Give a final structured verdict.

=== SIGNAL ===
Symbol: {sig['symbol']} | Timeframe: {sig['timeframe']} | Direction: {direction}
Composite Score: {sig['score']:.1f}/100
Body %: {sig['body_pct']:.1f}% | Volume: {sig['vol_mult']:.2f}x | ADX: {sig['adx']:.1f}
DI+: {sig['di_plus']:.1f} | DI-: {sig['di_minus']:.1f} | ATR Ratio: {sig['atr_ratio']:.2f}
EMA Stack (5/15/21): {ema_status}
Candle Rank: top {(1 - sig['candle_rank']) * 100:.0f}% of last 20 bars
Market Regime: {sig['regime']} ({sig['regime_score']}/100)
Signal recency: {sig.get('bar_offset', 1)} closed candle(s) ago

=== ML ENGINE ===
{ml_section}

=== BACKTEST (historical similar setups) ===
{bt_section}

=== WHY SELECTED ===
{reasons_text}

Synthesize all three engines. If backtest win rate < 45% with n >= 10, lean NO TRADE regardless of signal score.
If ML < 50%, lean WAIT or NO TRADE. If all engines agree, use HIGH confidence.

Respond in EXACTLY this format (no extra text):
VERDICT: [TRADE / WAIT / NO TRADE]
CONFIDENCE: [HIGH / MEDIUM / LOW]
RATIONALE: [2 sentences max. Reference the weakest and strongest evidence.]
RISK: [1 sentence on the main risk.]"""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model":       "llama-3.3-70b-versatile",
                "max_tokens":  200,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a systematic momentum trading analyst. "
                            "Be decisive and concise. Follow the output format exactly. "
                            "Never add extra commentary or markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

        verdict    = "WAIT"
        confidence = "MEDIUM"
        rationale  = raw
        risk       = ""

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                verdict = "NO TRADE" if "NO TRADE" in v else ("TRADE" if "TRADE" in v else "WAIT")
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().upper()
            elif line.startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()
            elif line.startswith("RISK:"):
                risk = line.split(":", 1)[1].strip()

        return {
            "verdict":    verdict,
            "confidence": confidence,
            "rationale":  rationale,
            "risk":       risk,
            "source":     "groq/llama-3.3-70b",
        }

    except Exception as exc:
        return {
            "verdict":    "ERROR",
            "confidence": "",
            "rationale":  f"API error: {str(exc)[:100]}",
            "risk":       "",
            "source":     "error",
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
    df = _scanner_fetch_candles(symbol, interval, limit=500)

    if df.empty or len(df) < 30:
        return {"error": "Not enough data", "n": 0}

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
    MAX_HOLD   = 20
    n_df       = len(df)
    method_results = {}

    for zone_name, zone_cfg in ENTRY_ZONES.items():
        ret_frac = zone_cfg["retrace"]
        expiry   = zone_cfg["expiry_bars"]

        for sl_label, sl_pct_val in [("Fixed SL", FIXED_SL), ("ATR SL", atr_sl_pct)]:
            for mgmt in MGMT_MODES:
                key        = f"{zone_name} / {sl_label} / {mgmt}"
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
                            _sl_pct      = max(0.008, min(0.06, (entry_target - _struct_sl) / entry_target))
                            sl_px        = round(entry_target - entry_target * _sl_pct, 8)
                        else:
                            sl_px        = round(entry_target * (1 - sl_pct_val), 8)
                    else:
                        entry_target = min(round(close_v + body_abs * ret_frac, 8), open_v * 0.999)
                        if sl_label == "ATR SL":
                            bar_high     = float(bar.get("high", close_v))
                            _struct_sl   = bar_high + atr14 * 0.5
                            _sl_pct      = max(0.008, min(0.06, (_struct_sl - entry_target) / entry_target))
                            sl_px        = round(entry_target + entry_target * _sl_pct, 8)
                        else:
                            sl_px        = round(entry_target * (1 + sl_pct_val), 8)

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

                        # TP2 full exit
                        tp2_hit = (hi >= tp2_px if direction == "long" else lo <= tp2_px)
                        if tp2_hit:
                            r_mult = ((1.0 * 0.5 + 2.0 * 0.5) if partial_done else 2.0) - 0.002
                            result = "WIN"; break

                    if result in ("WIN", "LOSS"):
                        trades_raw.append({"result": result, "r_mult": r_mult, "bars_held": bars_held})

                if len(trades_raw) < 3:
                    method_results[key] = {
                        "zone": zone_name, "sl_label": sl_label, "mgmt": mgmt,
                        "n": len(trades_raw), "win_rate": 0, "ev": 0,
                        "avg_r": 0, "avg_bars": 0, "insufficient": True,
                    }
                    continue

                rs    = [t["r_mult"] for t in trades_raw]
                wins  = [r for r in rs if r > 0]
                wr    = len(wins) / len(rs)
                avg_r = float(np.mean(rs))
                avg_b = float(np.mean([t["bars_held"] for t in trades_raw]))
                method_results[key] = {
                    "zone": zone_name, "sl_label": sl_label, "mgmt": mgmt,
                    "n": len(trades_raw), "win_rate": round(wr * 100, 1),
                    "ev": round(avg_r, 3), "avg_r": round(avg_r, 3),
                    "avg_bars": round(avg_b, 1), "insufficient": False,
                }

    # Best overall method
    valid    = {k: v for k, v in method_results.items()
                if not v.get("insufficient") and v["n"] >= 4 and v["win_rate"] >= 35}
    best_key = max(valid, key=lambda k: valid[k]["ev"]) if valid else None
    best     = method_results.get(best_key, {}) if best_key else {}

    # Best per zone
    zone_best = {}
    for zn in ("Aggressive", "Standard", "Sniper"):
        zm = {k: v for k, v in method_results.items()
              if v.get("zone") == zn and not v.get("insufficient") and v["n"] >= 4}
        if zm:
            bk = max(zm, key=lambda k: zm[k]["ev"])
            zone_best[zn] = {**zm[bk], "key": bk}

    # Legacy compat fields
    leg = method_results.get("Aggressive / Fixed SL / Simple", {})
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
    for live momentum signals. Ranks top 20 by composite score with point-by-point reasons.
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
        'then shows you the <b>top 20 highest-quality setups available right now</b> with '
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
        top_20 = []
        for s in all_signals:
            key = (s["symbol"], s["direction"])
            if key not in seen:
                seen[key] = True
                top_20.append(s)
            if len(top_20) >= 20:
                break

        st.session_state["mscanner_results"]    = top_20
        st.session_state["mscanner_all"]        = all_signals[:100]
        st.session_state["mscanner_key"]        = scan_key
        st.session_state["mscanner_scanned_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        st.session_state["mscanner_total_found"] = len(all_signals)

    # ── Render results ─────────────────────────────────────────────────────────
    top_20      = st.session_state.get("mscanner_results", [])
    scanned_at  = st.session_state.get("mscanner_scanned_at", "")
    total_found = st.session_state.get("mscanner_total_found", 0)

    if not top_20:
        st.warning(
            "No qualifying signals found with current settings. "
            "Try lowering Min Body % or Min Volume ×, "
            "or expand the coin universe.")
        return

    # Summary banner
    regime_counts = {}
    for s in top_20:
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
        f'Showing top {len(top_20)} &nbsp;|&nbsp; {regime_summary}</div>',
        unsafe_allow_html=True,
    )

    # Quick summary table
    _dir_icon = {"long": "📈", "short": "📉"}
    _reg_color = {"GREEN": "#3fb950", "YELLOW": "#e3b341", "RED": "#f85149"}

    summary_rows = []
    for i, s in enumerate(top_20):
        _etp_s = s.get("_trade_plan", {})
        summary_rows.append({
            "Rank":         f"#{i+1}",
            "Coin":         s["symbol"].replace("USDT", ""),
            "TF":           s["timeframe"],
            "Dir":          ("LONG" if s["direction"] == "long" else "SHORT"),
            "Score":        s["score"],
            "Regime":       s["regime"],
            "Body%":        s["body_pct"],
            "Vol×":         s["vol_mult"],
            "ADX":          s["adx"],
            "Agg Entry":    s["entry"],
            "Std Entry":    _etp_s.get("std_entry", s["entry"]),
            "Sniper Entry": _etp_s.get("sniper_entry", s["entry"]),
            "SL%":          _etp_s.get("sl_dist_pct", 1.5),
            "TP2 (Std)":    _etp_s.get("std_tp2", s["tp2r"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        height=min(40 + len(top_20) * 35, 750),
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
    for i, sig in enumerate(top_20):
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

                    def _fmt(v):
                        return f"{v:.6g}" if v else "—"

                    # Aggressive zone (enter at close)
                    _agg_rr1  = abs(_etp['agg_tp1'] - _etp['agg_entry']) / max(abs(_etp['agg_entry'] - _etp['agg_sl']), 1e-10)
                    _std_rr2  = 2.0  # always 2R by construction
                    _snp_rr3  = 3.0

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
  </div>

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
  </div>

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

                # ── OI + Funding Rate block (fetched once per symbol, cached) ─
                _deriv_cache_key = f"deriv_{sig['symbol']}"
                if _deriv_cache_key not in st.session_state:
                    _fr = fetch_funding_rate(sig["symbol"])
                    _oi = fetch_open_interest(sig["symbol"])
                    st.session_state[_deriv_cache_key] = {"fr": _fr, "oi": _oi}
                _deriv = st.session_state[_deriv_cache_key]
                _af_fr  = _deriv["fr"]
                _af_oi  = _deriv["oi"]

                # Build OI + Funding html rows
                _af_deriv_rows = []
                if _af_fr.get("ok"):
                    _fr_rate  = _af_fr["rate"]
                    _fr_col   = "#64ffda" if _fr_rate < -0.01 else "#f85149" if _fr_rate > 0.03 else "#8892b0"
                    if sig["direction"] == "long":
                        _fr_note = ("✅ shorts crowded — squeeze potential"
                                    if _fr_rate < -0.01 else
                                    "⚠️ longs crowded — squeeze risk"
                                    if _fr_rate > 0.03 else "neutral")
                    else:
                        _fr_note = ("✅ longs crowded — flush potential"
                                    if _fr_rate > 0.03 else
                                    "⚠️ shorts crowded — squeeze risk"
                                    if _fr_rate < -0.01 else "neutral")
                    _af_deriv_rows.append(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                        f'<span style="color:#8892b0;font-size:12px;">Funding Rate</span>'
                        f'<span style="color:{_fr_col};font-size:12px;font-weight:600;">'
                        f'{_fr_rate*100:.4f}% — {_fr_note}</span></div>'
                    )
                if _af_oi.get("ok"):
                    _oi_chg   = _af_oi["oi_change_pct"]
                    _oi_col   = "#64ffda" if _oi_chg > 5 else "#f85149" if _oi_chg < -5 else "#8892b0"
                    _oi_arrow = "▲" if _oi_chg >= 0 else "▼"
                    if sig["direction"] == "long":
                        _oi_note = ("✅ new money in — conviction" if _oi_chg > 5 else
                                    "⚠️ OI falling — possible short cover" if _oi_chg < -5 else "neutral")
                    else:
                        _oi_note = ("✅ new shorts entering — conviction" if _oi_chg > 5 else
                                    "⚠️ OI falling — squeeze risk" if _oi_chg < -5 else "neutral")
                    _af_deriv_rows.append(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                        f'<span style="color:#8892b0;font-size:12px;">OI 24h Δ</span>'
                        f'<span style="color:{_oi_col};font-size:12px;font-weight:600;">'
                        f'{_oi_arrow} {abs(_oi_chg):.1f}% — {_oi_note}</span></div>'
                    )

                if _af_deriv_rows:
                    st.markdown(
                        f'<div style="background:#0d1117;border:1px solid #2d3250;'
                        f'border-radius:8px;padding:12px 16px;margin-top:10px;">'
                        f'<div style="color:#8892b0;font-size:11px;text-transform:uppercase;'
                        f'letter-spacing:1px;margin-bottom:6px;">📊 Derivatives</div>'
                        + "".join(_af_deriv_rows) +
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Confluence Panel (full-width, below both columns) ────────────
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

            _sym_key      = f"{sig['symbol']}_{sig['timeframe']}_{sig['direction']}"
            _bt_cache_key = f"bt_{_sym_key}"
            _ml_cache_key = f"ml_{_sym_key}"
            _ai_key       = f"ai_result_{_sym_key}"
            _has_ai_key   = bool(st.session_state.get("groq_api_key", ""))

            # ── Step 1: Backtest + ML ─────────────────────────────────────────
            _step1_col, _step2_col = st.columns([1, 1])
            with _step1_col:
                if st.button("📊 Step 1 — Backtest + ML Score",
                             key=f"step1_{_sym_key}_{i}",
                             use_container_width=True,
                             help="Scan 500 bars historically + compute ML probability. Run this first."):
                    with st.spinner("Backtesting + ML…"):
                        _bt = _scanner_quick_backtest(sig)
                        _ml = _scanner_heuristic_ml(sig)
                    st.session_state[_bt_cache_key] = _bt
                    st.session_state[_ml_cache_key] = _ml

            # ── Step 2: AI Summary ────────────────────────────────────────────
            _bt_ready = _bt_cache_key in st.session_state
            _ml_ready = _ml_cache_key in st.session_state
            with _step2_col:
                _ai_disabled = not _has_ai_key or not (_bt_ready and _ml_ready)
                _ai_tip = (
                    "Run Step 1 first, then AI will synthesize all evidence."
                    if not (_bt_ready and _ml_ready) else
                    "Ask Groq LLM to synthesize signal + ML + backtest into a final verdict."
                    if _has_ai_key else
                    "Add Groq API key in sidebar to enable."
                )
                if st.button("🤖 Step 2 — AI Final Verdict",
                             key=f"step2_{_sym_key}_{i}",
                             use_container_width=True,
                             type="primary",
                             disabled=_ai_disabled,
                             help=_ai_tip):
                    with st.spinner("AI synthesizing all evidence…"):
                        _ai_res = _scanner_ai_verdict(
                            sig,
                            ml=st.session_state.get(_ml_cache_key),
                            bt=st.session_state.get(_bt_cache_key),
                        )
                    st.session_state[_ai_key] = _ai_res

            _bt_res = st.session_state.get(_bt_cache_key)
            _ml_res = st.session_state.get(_ml_cache_key)
            _ai_res = st.session_state.get(_ai_key)

            if _bt_res or _ml_res:
                _ml_res = _ml_res or _scanner_heuristic_ml(sig)
                _bt_res = _bt_res or {}
                _grade, _grade_color, _grade_desc = _scanner_setup_grade(sig, _ml_res, _bt_res)


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

                    # Best config for this zone
                    _best_sl_label = _zd.get("sl_label", "Fixed SL") if _zd else "Fixed SL"
                    _best_mgmt     = _zd.get("mgmt", "Simple") if _zd else "Simple"
                    _use_atr       = "ATR" in _best_sl_label
                    _sl_show       = _atr_sl_p if (_use_atr and _atr_sl_p) else _fix_sl_p
                    # TPs must match the SL method being displayed:
                    # Fixed SL → use fixed TPs; ATR SL → use ATR TPs from trade plan.
                    _tp1_show      = (_tp1_p if (_use_atr and _ep) else _fix_tp1)
                    _tp2_show      = (_tp2_p if (_use_atr and _ep) else _fix_tp2)
                    _tp3_show      = (_tp3_p if (_use_atr and _ep) else _fix_tp3)
                    _sl_pct_show   = _etp_card.get("sl_dist_pct", FIXED_SL_PCT * 100) if _use_atr else FIXED_SL_PCT * 100

                    if _zd and not _zd.get("insufficient") and _zd.get("n", 0) >= 4:
                        _expiry_note = "" if _zn == "Aggressive" else (
                            f' <span style="color:#e3b341;font-size:10px;">· Expires in 3 bars if not filled</span>'
                        )
                        _mgmt_html = _mgmt_detail_html(_ep, _sl_show, _tp1_show, _tp2_show, _best_mgmt, _best_sl_label)

                        _zone_table_rows += (
                            f'<div style="{_bg}{_border}border-radius:8px;padding:12px 14px;margin-bottom:8px;">'

                            # Header row
                            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                            f'<div>'
                            f'<span style="color:#ccd6f6;font-size:13px;font-weight:700;">{_zone_icons.get(_zn,"•")} {_zn}'
                            f'<span style="color:#3fb950;font-size:12px;">{_crown}</span></span>'
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
                            f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP2 (2R) / TP3</div>'
                            f'<div style="color:#64ffda;font-size:12px;font-weight:700;">{_fmt_px(_tp2_show)}</div>'
                            f'<div style="color:#3fb950;font-size:10px;">{_fmt_px(_tp3_show)}</div>'
                            f'</div>'
                            f'</div>'

                            # Management instructions
                            + _mgmt_html
                            + f'</div>'
                        )
                    else:
                        _zone_table_rows += (
                            f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;'
                            f'padding:8px 10px;margin-bottom:6px;opacity:0.5;">'
                            f'<span style="color:#8892b0;font-size:12px;">{_zone_icons.get(_zn,"•")} {_zn} — insufficient data (&lt;4 setups)</span>'
                            f'</div>'
                        )

                # ── Best method recommendation with full execution plan ────────
                if _best and _best_key:
                    _bev    = _best.get("ev", 0)
                    _bwr    = _best.get("win_rate", 0)
                    _bn     = _best.get("n", 0)
                    _bzone  = _best.get("zone", "Aggressive")
                    _bsl    = _best.get("sl_label", "Fixed SL")
                    _bmgmt  = _best.get("mgmt", "Simple")
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
                    # TPs must match the chosen SL method (Fixed vs ATR)
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
                        f'<div style="color:#ccd6f6;font-size:13px;font-weight:700;margin-bottom:8px;">{_best_key}</div>'
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
                        f'<div style="color:#8892b0;font-size:9px;text-transform:uppercase;">TP2 (2R)</div>'
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

                # ── Full management breakdown (expandable) ────────────────────
                _mgmt_table = ""
                if _per_method:
                    _mgmt_rows_html = ""
                    for _mk, _mv in sorted(_per_method.items(), key=lambda x: -x[1].get("ev", -99)):
                        if _mv.get("insufficient") or _mv.get("n", 0) < 4:
                            continue
                        _is_best = (_mk == _best_key)
                        _row_bg  = "background:#091a0d;" if _is_best else ""
                        _crown2  = " 👑" if _is_best else ""
                        _mgmt_rows_html += (
                            f'<div style="{_row_bg}display:grid;grid-template-columns:3fr 1fr 1fr 1fr 1fr;'
                            f'gap:4px;padding:5px 6px;border-bottom:1px solid #1a1f2e;font-size:11px;">'
                            f'<div style="color:#ccd6f6;">{_mk}{_crown2}</div>'
                            f'<div style="color:{_wr_color(_mv["win_rate"])};text-align:right;font-weight:700;">{_mv["win_rate"]:.0f}%</div>'
                            f'<div style="color:{_ev_color(_mv["ev"])};text-align:right;font-weight:700;">{_mv["ev"]:+.2f}R</div>'
                            f'<div style="color:#8892b0;text-align:right;">{_mv["avg_r"]:+.2f}R</div>'
                            f'<div style="color:#8892b0;text-align:right;">{_mv["n"]}n/{_mv["avg_bars"]:.0f}b</div>'
                            f'</div>'
                        )
                    if _mgmt_rows_html:
                        _mgmt_table = (
                            f'<div style="margin-top:10px;border:1px solid #21262d;border-radius:6px;overflow:hidden;">'
                            f'<div style="background:#161b22;display:grid;grid-template-columns:3fr 1fr 1fr 1fr 1fr;'
                            f'gap:4px;padding:5px 6px;border-bottom:1px solid #30363d;">'
                            f'<div style="color:#8892b0;font-size:10px;text-transform:uppercase;">Method</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">WR%</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">EV</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">Avg R</div>'
                            f'<div style="color:#8892b0;font-size:10px;text-align:right;">n/bars</div>'
                            f'</div>'
                            f'{_mgmt_rows_html}'
                            f'</div>'
                        )

                _bt_rows = (
                    (
                        f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #21262d;">'
                        f'<div style="color:#58a6ff;font-size:11px;text-transform:uppercase;'
                        f'letter-spacing:1px;font-weight:700;margin-bottom:8px;">📊 Entry Method Comparison</div>'
                        + _zone_table_rows
                        + _recommendation_html
                        + _mgmt_table
                        + f'</div>'
                    ) if _bt_valid else
                    f'<div style="color:#8892b0;font-size:12px;padding:5px 0;">📊 Backtest: {_bt_res.get("error","No matching setups")}</div>'
                )

                # AI verdict block
                _ai_block = ""
                if _ai_res:
                    _v = _ai_res.get("verdict", "WAIT")
                    _c = _ai_res.get("confidence", "")
                    _v_color  = "#3fb950" if _v == "TRADE" else "#e3b341" if _v == "WAIT" else "#f85149"
                    _v_bg     = "#091a0d" if _v == "TRADE" else "#1a1500" if _v == "WAIT" else "#1a0505"
                    _c_badge  = (f'<span style="background:#1f2b1f;color:#3fb950;font-size:10px;'
                                 f'border-radius:3px;padding:1px 6px;margin-left:6px;">{_c}</span>'
                                 if _c in ("HIGH","MEDIUM","LOW") else "")
                    _exec_str = _ai_res.get("execution", "")
                    _exec_row = (
                        f'<div style="background:#0a1628;border:1px solid #1f6feb;border-radius:6px;'
                        f'padding:8px 10px;margin-top:8px;">'
                        f'<div style="color:#58a6ff;font-size:10px;text-transform:uppercase;'
                        f'letter-spacing:1px;margin-bottom:4px;">📋 Execution Plan</div>'
                        f'<div style="color:#ccd6f6;font-size:12px;line-height:1.6;">{_exec_str}</div>'
                        f'</div>'
                    ) if _exec_str else ""
                    _risk_row = (
                        f'<div style="background:#1a0a0a;border-radius:4px;padding:6px 10px;margin-top:6px;">'
                        f'<span style="color:#e3b341;font-size:10px;text-transform:uppercase;'
                        f'letter-spacing:1px;">⚠️ Risk: </span>'
                        f'<span style="color:#ccd6f6;font-size:11px;">{_ai_res["risk"]}</span>'
                        f'</div>'
                    ) if _ai_res.get("risk") else ""
                    _src = _ai_res.get("source","")
                    _ai_block = (
                        f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #21262d;">'
                        f'<div style="color:#8892b0;font-size:10px;text-transform:uppercase;'
                        f'letter-spacing:1px;margin-bottom:8px;">🤖 AI Final Verdict{_c_badge}</div>'
                        f'<div style="background:{_v_bg};border:1px solid {_v_color};border-radius:6px;'
                        f'padding:10px 14px;margin-bottom:8px;">'
                        f'<div style="color:{_v_color};font-size:24px;font-weight:900;margin-bottom:4px;">{_v}</div>'
                        f'<div style="color:#ccd6f6;font-size:12px;line-height:1.6;">{_ai_res.get("rationale","")}</div>'
                        f'</div>'
                        + _exec_row
                        + _risk_row
                        + (f'<div style="color:#3a3f4b;font-size:10px;margin-top:4px;">{_src}</div>'
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
                    + f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #21262d;">'
                    f'<div style="color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">Edge Summary</div>'
                    f'<div style="color:#ccd6f6;font-size:12px;">ML {_ml_res["pct"]:.0f}%'
                    + _edge_bt
                    + f' · Score {score_pct}/100 · {sig["regime"]} regime</div></div>'
                    + _ai_block
                    + f'</div>'
                )
                st.markdown(_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#8892b0;font-size:12px;padding:8px 0;">'
                    '▸ Click <b>Step 1</b> to run backtest + ML, then <b>Step 2</b> for AI final verdict.</div>',
                    unsafe_allow_html=True,
                )

    # Download button
    st.markdown("---")
    _dl_rows = []
    for s in top_20:
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
        "⬇ Download Top 20 as CSV",
        _dl_df.to_csv(index=False).encode("utf-8"),
        f"market_scanner_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
    )


# ─── Intelligence / WFO / ML Tab Renderers ────────────────────────────────────

def render_intelligence_tab(df, qualifying, all_trades, direction,
                             ticker, timeframe, regime_data, best_params=None):
    """Tab 10 — Candle Intelligence: similarity engine + AI analysis per bar."""
    import math

    adx_df = st.session_state.get("_intel_adx_df")
    if adx_df is None:
        with st.spinner("Computing ADX for intelligence tab…"):
            adx_df = calculate_adx(df)
        st.session_state["_intel_adx_df"] = adx_df

    # ── Build last-20-candles table ───────────────────────────────────────────
    n_rows   = min(20, len(df))
    view_df  = df.iloc[-n_rows:].copy()
    # For intraday timeframes use exact timestamps; for daily normalize to date
    if timeframe in ("1H", "4H"):
        qual_idx = set(qualifying.index) if not qualifying.empty else set()
    else:
        qual_idx = set(qualifying.index.normalize()) if not qualifying.empty else set()

    ml_model = st.session_state.get("ml_model_data")

    rows = []
    for i, (ts, bar) in enumerate(view_df.iterrows()):
        bar_i    = len(df) - n_rows + i
        if timeframe in ("1H", "4H"):
            ts_wib   = pd.Timestamp(ts) + pd.Timedelta(hours=7)
            date_str = ts_wib.strftime("%m-%d %H:%M")
        else:
            date_str = str(ts)[:10]
        is_long  = bar.get("body", 0) > 0
        dir_sym  = "▲" if is_long else "▼"

        body_pct = float(bar.get("body_pct", 0) or 0)
        vol_m    = float(bar.get("vol_mult",  0) or 0)
        atr_r    = float(bar.get("atr_ratio", 1) or 1)

        adx_val  = 0.0
        try:
            adx_val = float(adx_df["adx"].iloc[bar_i])
        except Exception:
            pass

        ema5  = float(bar.get("ema5",  bar.get("close", 0)) or 0)
        ema15 = float(bar.get("ema15", bar.get("close", 0)) or 0)
        ema21 = float(bar.get("ema21", bar.get("close", 0)) or 0)
        if direction == "long":
            ema_ok = ema5 > ema15 > ema21
            ema_p  = (ema5 > ema15) or (ema15 > ema21)
        else:
            ema_ok = ema5 < ema15 < ema21
            ema_p  = (ema5 < ema15) or (ema15 < ema21)
        ema_sym = "OK" if ema_ok else ("~" if ema_p else "X")

        di_plus  = float(adx_df["di_plus"].iloc[bar_i])  if "di_plus"  in adx_df.columns else 0
        di_minus = float(adx_df["di_minus"].iloc[bar_i]) if "di_minus" in adx_df.columns else 0
        di_gap   = round((di_plus - di_minus) if direction == "long" else (di_minus - di_plus), 1)

        reg = calculate_regime_score(df, bar_i, direction, adx_df,
                                      htf_ema_series=None,
                                      timeframe=timeframe, ticker=ticker)
        reg_score = reg.get("score", 0)
        reg_v     = reg.get("verdict", "RED")[0]   # G/Y/R

        if timeframe in ("1H", "4H"):
            is_signal = ts in qual_idx
        else:
            is_signal = ts.normalize() in qual_idx if hasattr(ts, "normalize") else False
        sig_sym   = "SIGNAL" if is_signal else ""

        ml_pct = ""
        if ml_model and "model" in ml_model:
            sc = score_candle_ml(bar, ml_model, direction)
            if sc.get("available"):
                ml_pct = f"{sc['probability']*100:.0f}"

        rows.append({
            "#":       n_rows - i,
            "Date":    date_str,
            "Dir":     dir_sym,
            "Body%":   f"{abs(body_pct)*100:.1f}",
            "Vol x":   f"{vol_m:.2f}",
            "ATR":     f"{atr_r:.2f}",
            "ADX":     f"{adx_val:.1f}",
            "EMA":     ema_sym,
            "DI Gap":  f"{di_gap:+.1f}",
            "ML%":     ml_pct,
            "Regime":  f"{reg_v}{reg_score}",
            "Signal":  sig_sym,
            "_bar_i":  bar_i,
            "_ts":     ts,
        })

    display_cols = ["#", "Date", "Dir", "Body%", "Vol x", "ATR", "ADX",
                    "EMA", "DI Gap", "ML%", "Regime", "Signal"]
    table_df = pd.DataFrame(rows)

    def _style_row(row):
        sig = row.get("Signal", "")
        if sig == "SIGNAL":
            return ["background-color:#0d2818;color:#3fb950"] * len(display_cols)
        reg_str = str(row.get("Regime", ""))
        if reg_str.startswith("G"):
            return ["background-color:#0d1f14;color:#ccd6f6"] * len(display_cols)
        if reg_str.startswith("Y"):
            return ["background-color:#1f1a00;color:#ccd6f6"] * len(display_cols)
        return ["color:#8b949e"] * len(display_cols)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Last 20 Candles")
        st.caption("Click a row to analyze. Green = signal candle, yellow = GREEN regime.")

        styled = (table_df[display_cols]
                  .style.apply(_style_row, axis=1))

        sel = st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="intel_candle_sel",
        )

        sel_rows = sel.selection.get("rows", []) if hasattr(sel, "selection") else []
        if sel_rows:
            st.session_state["_intel_sel_row"] = sel_rows[0]

    with col_right:
        sel_idx = st.session_state.get("_intel_sel_row")

        if sel_idx is None or sel_idx >= len(rows):
            st.info("Click a candle row to analyze")
        else:
            row_meta  = rows[sel_idx]
            bar_i_sel = row_meta["_bar_i"]
            candle    = df.iloc[bar_i_sel]

            # Enrich candle with ADX/DI values (computed separately from adx_df)
            candle_dict = candle.to_dict()
            candle_dict["_date_str"] = row_meta["Date"]
            try:
                candle_dict["adx_value"] = float(adx_df["adx"].iloc[bar_i_sel])
                candle_dict["di_plus"]   = float(adx_df["di_plus"].iloc[bar_i_sel])
                candle_dict["di_minus"]  = float(adx_df["di_minus"].iloc[bar_i_sel])
            except Exception:
                pass
            candle = candle_dict

            st.markdown(f"#### Analysis — {row_meta['Date']}")

            # ── Similarity ────────────────────────────────────────────────────
            hq = qualifying.copy() if not qualifying.empty else pd.DataFrame()
            if not hq.empty:
                if "body_pct_abs" not in hq.columns:
                    hq["body_pct_abs"] = hq["body_pct"].abs()
                if "adx_value" not in hq.columns:
                    hq["adx_value"] = 25.0

                target_dict = {
                    "body_pct_abs": abs(float(candle.get("body_pct", 0.7) or 0.7)),
                    "vol_mult":     float(candle.get("vol_mult",  1.5) or 1.5),
                    "atr_ratio":    float(candle.get("atr_ratio", 1.0) or 1.0),
                    "adx_value":    float(candle.get("adx_value", 20.0) or 20.0),
                }

                matched, meta = find_similar_candles(
                    target_dict, hq, direction,
                    regime_zone=None, session=None,
                    min_matches=10, ticker=ticker,
                )

                # Flatten all_trades to a single list
                if isinstance(all_trades, dict):
                    flat_trades = [t for tl in all_trades.values() for t in tl]
                else:
                    flat_trades = list(all_trades)

                # Build trades DataFrame for aggregate_outcomes
                if flat_trades:
                    trades_df = pd.DataFrame(flat_trades)
                    if "entry_date" in trades_df.columns:
                        trades_df.index = pd.to_datetime(trades_df["entry_date"])
                    if "r_mult" in trades_df.columns and "r_multiple" not in trades_df.columns:
                        trades_df["r_multiple"] = trades_df["r_mult"]
                    if "exit_type" not in trades_df.columns:
                        trades_df["exit_type"] = "TP"
                else:
                    trades_df = pd.DataFrame()

                outcomes = aggregate_outcomes(matched, trades_df) if not matched.empty and not trades_df.empty else {"n": 0, "confidence": "INSUFFICIENT"}

                ml_result = score_candle_ml(candle, ml_model, direction) if ml_model else {"probability": 0.5, "available": False}
                ml_prob   = ml_result.get("probability", 0.5)

                anchor = generate_probability_anchor(
                    outcomes,
                    target_buckets=meta.get("target_buckets", {}),
                    regime_zone=regime_data.get("verdict") if regime_data else None,
                    session=None,
                    ml_score=ml_prob,
                    ml_meta={},
                )

                # Quality badge
                quality = meta.get("quality_pct", 0)
                q_color = "#3fb950" if quality >= 50 else ("#e3b341" if quality >= 20 else "#f85149")
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;'
                    f'border-radius:6px;padding:10px 14px;margin-bottom:8px;">'
                    f'<span style="color:#8b949e;font-size:11px;">MATCH QUALITY</span> '
                    f'<span style="color:{q_color};font-weight:700;">{quality}%</span> '
                    f'<span style="color:#8b949e;font-size:11px;">({meta.get("n",0)} matches'
                    f'{", relaxed: " + meta["relaxed_dim"] if meta.get("relaxed_dim") else ""})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Anchor text
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #21262d;'
                    f'border-radius:6px;padding:12px 14px;'
                    f'font-family:monospace;font-size:12px;color:#ccd6f6;'
                    f'white-space:pre-wrap;">{anchor}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Run Analysis first to enable similarity matching.")
                outcomes = {"n": 0, "confidence": "INSUFFICIENT"}
                ml_prob  = 0.5
                anchor   = ""

            st.markdown("---")

            # ── AI / Rule-based verdict ───────────────────────────────────────
            _ai_provider  = st.session_state.get("ai_provider", "Groq (Free)")
            _use_groq     = "Groq" in _ai_provider
            _has_groq_key = bool(st.session_state.get("groq_api_key", ""))
            _has_ant_key  = bool(st.session_state.get("anthropic_api_key", ""))
            _has_any_key  = (_has_groq_key if _use_groq else _has_ant_key)

            if _has_any_key:
                _provider_label = f"Groq / Llama 3.3 70B (free)" if _use_groq else "Anthropic / Claude 3.5 Sonnet"
                st.markdown(
                    f'<div style="font-size:11px;color:#58a6ff;margin-bottom:4px;">'
                    f'AI provider: <b>{_provider_label}</b></div>',
                    unsafe_allow_html=True)
                if st.button("Analyze this candle", key="ai_analyze_btn",
                             type="primary", use_container_width=True):
                    # Clear previous result so stale data isn't shown
                    st.session_state.pop("_intel_ai_result", None)
                    with st.spinner(f"Calling {_provider_label}…"):
                        result = analyze_candle_ai(
                            candle,
                            regime_data,
                            anchor,
                            ml_prob,
                            strategy_params=best_params,
                            wfo_results=st.session_state.get("wfo_results"),
                            direction=direction,
                            ticker=ticker,
                            timeframe=timeframe,
                            risk_context={},
                        )
                    st.session_state["_intel_ai_result"] = result

                ai_result = st.session_state.get("_intel_ai_result")
                if ai_result:
                    vrd = ai_result.get("verdict", "WAIT")
                    src = ai_result.get("source", "")
                    err = ai_result.get("error", "")
                    v_colors = {"TRADE": "#3fb950", "WAIT": "#e3b341", "NO TRADE": "#f85149"}
                    v_col = v_colors.get(vrd, "#8b949e")
                    if err:
                        # Show error detail so user can diagnose
                        st.error(f"AI call failed — using rule-based fallback\n\n`{err}`")
                    st.markdown(
                        f'<div style="background:#0d1117;border:1px solid {v_col};'
                        f'border-radius:6px;padding:12px 14px;margin-top:8px;">'
                        f'<div style="color:{v_col};font-weight:700;font-size:14px;'
                        f'margin-bottom:8px;">{vrd}'
                        f'<span style="color:#8b949e;font-size:10px;margin-left:8px;">'
                        f'via {src if src else "fallback"}</span></div>'
                        f'<div style="color:#ccd6f6;font-size:12px;white-space:pre-wrap;">'
                        f'{ai_result.get("full_text","")}</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                # No key — show rule-based and prompt to add key
                rb = rule_based_fallback(regime_data, outcomes, ml_prob)
                vrd = rb.get("verdict", "WAIT")
                v_colors = {"TRADE": "#3fb950", "WAIT": "#e3b341", "NO TRADE": "#f85149"}
                v_col = v_colors.get(vrd, "#8b949e")
                st.info("Add a free Groq API key in the sidebar (🤖 AI Analysis) to enable AI analysis. "
                        "Rule-based verdict shown below.")
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid {v_col};'
                    f'border-radius:6px;padding:12px 14px;margin-top:8px;">'
                    f'<div style="color:{v_col};font-weight:700;font-size:14px;'
                    f'margin-bottom:8px;">{vrd} <span style="color:#8b949e;font-size:10px;">'
                    f'(rule-based)</span></div>'
                    f'<div style="color:#ccd6f6;font-size:12px;white-space:pre-wrap;">'
                    f'{rb.get("full_text","")}</div></div>',
                    unsafe_allow_html=True,
                )


def render_wfo_tab(df, qualifying_func, ticker, timeframe, direction):
    """Tab 11 — Walk-Forward Optimisation Validation."""
    st.markdown("### 📈 Walk-Forward Validation")
    st.markdown(
        '<div class="info-box">'
        '<h4>What is Walk-Forward Validation?</h4>'
        '<p>Walk-Forward Optimisation (WFO) tests whether your parameters actually generalise '
        'to unseen data. It repeatedly optimises on an In-Sample (IS) window, then measures '
        'performance on the Out-of-Sample (OOS) window that immediately follows — simulating '
        'real-world deployment. A PASS verdict means your strategy is not curve-fitted: '
        'the same logic that worked historically continues to work on data the optimiser never saw.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("Run WFO Validation", type="primary", key="wfo_run_btn",
                 use_container_width=True):
        with st.spinner("Running Walk-Forward Optimisation — this may take a few minutes…"):
            result = run_wfo_pipeline(
                df, qualifying_func, adx_df=None,
                timeframe=timeframe, direction=direction,
            )
        st.session_state["wfo_results"] = result
        st.rerun()

    wfo = st.session_state.get("wfo_results")
    if not wfo:
        st.info("Click **Run WFO Validation** to test parameter robustness.")
        return

    summary = wfo.get("summary", {})
    verdict = summary.get("verdict", "FAIL")
    v_color = "#3fb950" if verdict == "PASS" else "#f85149"

    st.markdown(
        f'<div style="background:#0d1117;border:2px solid {v_color};border-radius:8px;'
        f'padding:16px 20px;margin:12px 0;text-align:center;">'
        f'<span style="color:{v_color};font-size:28px;font-weight:800;">{verdict}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg OOS PF",     f"{summary.get('avg_oos_pf', 0):.2f}")
    with c2:
        st.metric("% Profitable",   f"{summary.get('pct_profitable_cycles', 0):.0f}%")
    with c3:
        st.metric("OOS/IS Ratio",   f"{summary.get('oos_is_ratio_avg', 0):.2f}")
    with c4:
        st.metric("Cycles",         str(summary.get("n_cycles", 0)))

    cycles = wfo.get("cycles", [])
    if cycles:
        cyc_df = pd.DataFrame(cycles).drop(columns=["best_params"], errors="ignore")
        st.dataframe(cyc_df, use_container_width=True, hide_index=True)

        # Bar chart — OOS PF per cycle
        x_vals  = [f"C{c['cycle']}" for c in cycles]
        y_vals  = [c["oos_pf"] for c in cycles]
        colors  = ["#3fb950" if v > 1.0 else "#f85149" for v in y_vals]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_vals, y=y_vals, marker_color=colors,
                             name="OOS PF", showlegend=False))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#e3b341",
                      annotation_text="PF = 1.0", annotation_position="right")
        _dark(fig, title="Out-of-Sample Profit Factor per Cycle",
              height=280, yaxis_title="PF")
        st.plotly_chart(fig, use_container_width=True)

    bp = wfo.get("best_params")
    if bp:
        st.markdown("**Best Validated Parameters:**")
        bp_cols = st.columns(5)
        labels  = ["Body %", "Vol ×", "Retracement", "SL dist", "TP mode"]
        vals    = [str(bp.get("body_pct","—")), str(bp.get("vol_mult","—")),
                   str(bp.get("retracement","—")), str(bp.get("sl_dist","—")),
                   str(bp.get("tp_mode","—"))]
        for col, lbl, val in zip(bp_cols, labels, vals):
            col.metric(lbl, val)

    vt = wfo.get("validated_at")
    if vt:
        st.caption(f"Validated at: {str(vt)[:16]}")


def render_ml_tab(qualifying, all_trades, direction, ticker, timeframe, adx_df):
    """Tab 12 — ML Signal Classifier training and inspection."""
    st.markdown("### 🤖 ML Signal Classifier")

    validated_params = None
    wfo = st.session_state.get("wfo_results")
    if wfo and wfo.get("best_params"):
        bp = wfo["best_params"]
        validated_params = (
            bp.get("retracement"),
            bp.get("sl_dist"),
            bp.get("tp_mode"),
        )

    if st.button("Train / Retrain Model", type="primary", key="ml_train_btn",
                 use_container_width=True):
        with st.spinner("Training classifier…"):
            result = train_signal_classifier(
                qualifying, all_trades, direction, validated_params
            )
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state["ml_model_data"] = result
            st.session_state["ml_meta"] = {
                "cv_accuracy":      result["cv_accuracy"],
                "holdout_accuracy": result["holdout_accuracy"],
                "n_samples":        result["n_samples"],
                "trained_at":       result.get("trained_at"),
            }
            st.rerun()

    model_data = st.session_state.get("ml_model_data")
    if not model_data:
        st.info("Click **Train / Retrain Model** to build the classifier.")
        return

    # ── Status ────────────────────────────────────────────────────────────────
    ho_acc = model_data.get("holdout_accuracy", 0)
    cv_acc = model_data.get("cv_accuracy",      0)
    if ho_acc >= 0.55:
        health_label = "HEALTHY"
        h_color      = "#3fb950"
    elif ho_acc >= 0.50:
        health_label = "MARGINAL"
        h_color      = "#e3b341"
    else:
        health_label = "UNRELIABLE — retrain needed"
        h_color      = "#f85149"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples",        str(model_data.get("n_samples", 0)))
    c2.metric("CV Accuracy",    f"{cv_acc*100:.1f}%")
    c3.metric("Holdout Acc.",   f"{ho_acc*100:.1f}%")
    c4.metric("Class Balance",  model_data.get("class_balance", "—"))

    st.markdown(
        f'<div style="background:#0d1117;border:1px solid {h_color};'
        f'border-radius:6px;padding:10px 16px;margin:8px 0;">'
        f'<span style="color:{h_color};font-weight:700;">Model health: {health_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Feature importance chart ───────────────────────────────────────────────
    fi = model_data.get("feature_importance", {})
    if fi:
        feats  = list(fi.keys())
        coeffs = [fi[f] for f in feats]
        colors = ["#3fb950" if c > 0 else "#f85149" for c in coeffs]

        fig = go.Figure(go.Bar(
            x=coeffs, y=feats, orientation="h",
            marker_color=colors, showlegend=False,
        ))
        _dark(fig, title="Feature Importance (Logistic Regression Coefficients)",
              height=280, xaxis_title="Coefficient", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    trained_at = model_data.get("trained_at")
    if trained_at:
        st.caption(f"Trained: {str(trained_at)[:16]}")


# ─── Main App ──────────────────────────────────────────────────────────────────

def main():
    # ── Default AI provider: Groq (free) — pre-fill on first load ────────────
    if "ai_provider" not in st.session_state:
        st.session_state["ai_provider"] = "Groq (Free)"
    if "groq_api_key" not in st.session_state:
        st.session_state["groq_api_key"] = "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"

    with st.sidebar:
        st.markdown("## Momentum Candle")
        st.markdown("---")

        # ── Apply pending strategy BEFORE widgets are instantiated ────────────
        if "_pending_strategy" in st.session_state:
            _ps = st.session_state.pop("_pending_strategy")
            for _k, _v in _ps.items():
                if not _k.startswith("_"):
                    st.session_state[_k] = _v
            # ADX threshold stored separately (slider needs it as default)
            if "_adx_threshold_val" in _ps:
                st.session_state["_pending_adx_threshold"] = _ps["_adx_threshold_val"]

        _dir_choice = st.radio(
            "Trade Direction",
            ["📈 Long  (Bullish candles)", "📉 Short (Bearish candles)"],
            index=0, horizontal=False, key="direction_radio",
        )
        direction = "short" if "Short" in _dir_choice else "long"
        if direction == "short":
            st.markdown(
                '<div style="background:#2d1a1a;border:1px solid #ff6b6b;border-radius:6px;'
                'padding:8px 12px;margin-bottom:4px;color:#ff6b6b;font-size:12px;">'
                '📉 SHORT mode — looks for large bearish candles with high volume. '
                'Entry waits for a retrace UP from the close.</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        ticker = normalise_ticker(
            st.text_input("Ticker Symbol", value="BTC",
                          placeholder="BTC, ETH, SOL, MNT, HYPE, BTCUSDT...",
                          key="sb_ticker"))

        timeframe = st.selectbox("Timeframe", ["1D","4H","1H"], index=0, key="sb_timeframe")
        if timeframe in ("4H","1H"):
            st.warning("Shorter timeframes = more signals but less reliable. "
                       "Recommend minimum 30 qualifying candles.")

        history_label = st.select_slider(
            "Data history to analyze",
            options=list(HISTORY_OPTIONS.keys()), value="5 years",
            key="sb_history")
        history_days = HISTORY_OPTIONS[history_label]

        st.markdown("---")
        min_body_pct     = st.slider("Min Body %", 50, 95, 70, 5, key="sb_body") / 100
        min_vol_mult     = st.slider("Min Volume Multiplier", 1.0, 4.0, 1.5, 0.1, key="sb_vol")
        transaction_cost = st.number_input(
            "Transaction Cost (%)", 0.0, 1.0, 0.10, 0.01, key="sb_tc") / 100

        st.markdown("---")
        st.markdown("**Market Regime Filter (ADX)**")
        adx_filter     = st.toggle("Enable ADX Filter", value=False, key="adx_toggle")
        adx_threshold  = 25
        adx_tf_mode    = "Same as signal timeframe"
        di_gap_min     = 0.0
        if adx_filter:
            _adx_thr_default = st.session_state.pop("_pending_adx_threshold", 25)
            adx_threshold = st.slider("ADX Threshold", 15, 40, _adx_thr_default, 1,
                                      help="Only take trades when ADX ≥ this value (trending market)")
            adx_tf_mode = st.selectbox(
                "Apply ADX on which timeframe?",
                ["Same as signal timeframe", "One timeframe higher", "Both"],
                index=0, key="adx_tf_mode",
                help="'Both' = ADX must be ≥ threshold on BOTH timeframes")

            st.markdown("**DI Alignment Filter**")
            use_di_filter = st.toggle(
                "Require DI aligned + meaningful gap", value=False, key="di_filter_toggle",
                help="When ON: signal only qualifies if DI is aligned with direction "
                     "AND the DI gap is large enough. Adds confluence but reduces trade count.")
            if use_di_filter:
                di_gap_min = st.select_slider(
                    "Minimum DI gap",
                    options=[5, 10, 15, 20, 25],
                    value=10, key="di_gap_slider",
                    help="DI gap = |DI+ − DI−|. "
                         "< 5 = noise, 10 = moderate, ≥ 20 = strong. "
                         "Long: requires DI+ > DI− by this gap. "
                         "Short: requires DI− > DI+ by this gap.")
                st.caption(
                    f"Signal requires: body ✓ + volume ✓ + ADX ≥ {adx_threshold} ✓ "
                    f"+ DI aligned + gap ≥ {di_gap_min:.0f} ✓"
                )
            else:
                st.caption(
                    "ADX measures trend strength (not direction). "
                    "ADX ≥ 25 = trending, < 25 = ranging. "
                    "Enable DI Alignment to also filter by trend direction."
                )

        st.markdown("---")
        st.markdown("**EMA Trend Filter** *(optional)*")
        ema_filter = st.toggle(
            "Enable EMA Filter", value=False, key="ema_toggle",
            help="Only take LONG signals when price is above EMA. "
                 "Only SHORT signals when price is below EMA. "
                 "Trades WITH the macro trend — reduces signals but improves quality.")

        # Defaults for when filter is off
        ema_period      = 200
        ema_tf_mode     = "Same as signal timeframe"
        ema_htf_period  = 200

        if ema_filter:
            st.markdown("**Same-Timeframe EMA**")
            ema_period = st.select_slider(
                "EMA period (signal timeframe)",
                options=[50, 100, 200],
                value=200,
                key="ema_period",
                help="EMA 200 = ~10 months on daily. "
                     "EMA 100 = ~5 months. EMA 50 = ~2.5 months.")

            st.markdown("**Higher Timeframe EMA**")
            _htf_label_map = {"1H": "4H", "4H": "Daily", "1D": "Weekly"}
            _htf_name      = _htf_label_map.get(timeframe, "HTF")
            _htf_default   = 50 if timeframe == "1D" else 200  # Weekly → EMA 50

            use_htf_ema = st.toggle(
                f"Also require {_htf_name} EMA alignment",
                value=False,
                key="ema_htf_toggle",
                help=f"Checks EMA on {_htf_name} chart.\n"
                     f"1H → 4H EMA | 4H → Daily EMA | Daily → Weekly EMA.\n"
                     f"Both EMAs must agree with direction for a signal to trigger.")

            if use_htf_ema:
                ema_htf_period = st.select_slider(
                    f"EMA period ({_htf_name} chart)",
                    options=[50, 100, 200],
                    value=_htf_default,
                    key="ema_htf_period",
                    help="Weekly chart: EMA 50 is the institutional standard.\n"
                         "Daily chart: EMA 200 is most widely watched.\n"
                         "4H chart: EMA 200 covers ~33 days of 4H history.")
                ema_tf_mode = "Higher timeframe"
            else:
                ema_tf_mode     = "Same as signal timeframe"
                ema_htf_period  = ema_period

            # Show what the filter will do
            _op       = "above" if direction == "long" else "below"
            _htf_str  = (f" + {_htf_name} EMA {ema_htf_period}"
                         if ema_tf_mode == "Higher timeframe" else "")
            st.caption(
                f"Signal requires: close {_op} {timeframe} EMA {ema_period}"
                f"{_htf_str}")

        st.markdown("---")
        st.markdown("**Stop Loss Management**")
        sl_mode    = st.radio("SL Method",
                              ["Fixed","Breakeven","Trailing ATR","Breakeven+Trail"],
                              index=0, key="sb_sl_mode")
        trail_mult = 1.5
        if sl_mode in ("Trailing ATR","Breakeven+Trail"):
            trail_mult = st.slider("ATR Trail Multiplier", 1.0, 3.0, 1.5, 0.1)

        st.markdown("---")
        st.markdown("**Time-Based Exit (Optional)**")
        use_time_exit = st.toggle("Enable Max Hold Time", value=False,
                                  key="time_exit_toggle")
        max_hold_bars = 500  # default = no limit
        if use_time_exit:
            hold_options = [str(i) for i in range(1, 21)] + ["No Limit"]
            hold_choice  = st.select_slider(
                "Max candles to hold",
                options=hold_options,
                value="5",
                key="max_hold_slider",
            )
            if hold_choice == "No Limit":
                max_hold_bars = 500
                use_time_exit = False   # No Limit = run until TP/SL, no time exit fires
                st.caption(
                    "**No Limit** — trades run until TP or SL is hit. "
                    "No time-based exits will appear in the Trade Log."
                )
            else:
                max_hold_bars = int(hold_choice)
                st.caption(
                    "Candle 1 = first candle after your entry fills. "
                    "If still open after (N-1) candles, exits at candle N's open price. "
                    "TP/SL hit earlier always takes priority. "
                    "Note: parameter sweep always uses pure TP/SL — "
                    "time exit only applies to best-setup display and hold time analysis."
                )

        st.markdown("---")
        st.markdown("**Session Filter** *(4H & 1H only, WIB = UTC+7)*")
        use_session = st.toggle("Enable Session Filter", value=False, key="session_toggle")
        allowed_sessions = ["London", "NY+London", "Asian", "Dead Zone"]  # default = all
        if use_session:
            _sc1, _sc2 = st.columns(2)
            with _sc1:
                _s_ny  = st.checkbox("NY+London (20:00-00:00)", value=True,  key="sess_ny")
                _s_lon = st.checkbox("London (15:00-20:00)",    value=True,  key="sess_london")
            with _sc2:
                _s_asi = st.checkbox("Asian (07:00-15:00)",     value=False, key="sess_asian")
                _s_dz  = st.checkbox("Dead Zone (00:00-07:00)", value=False, key="sess_dead")
            allowed_sessions = []
            if _s_ny:  allowed_sessions.append("NY+London")
            if _s_lon: allowed_sessions.append("London")
            if _s_asi: allowed_sessions.append("Asian")
            if _s_dz:  allowed_sessions.append("Dead Zone")
            if not allowed_sessions:
                st.warning("Select at least one session.")
                use_session = False
                allowed_sessions = ["London", "NY+London", "Asian", "Dead Zone"]
            else:
                st.caption(f"Active: {', '.join(allowed_sessions)}")
        else:
            use_session = False

        # ── WFO / ML status badges ────────────────────────────────────────────
        _wfo_s = st.session_state.get("wfo_results")
        _ml_s  = st.session_state.get("ml_model_data")
        if _wfo_s or _ml_s:
            _badge_parts = []
            if _wfo_s:
                _wv  = _wfo_s.get("summary", {}).get("verdict", "?")
                _wdt = str(_wfo_s.get("validated_at", ""))[:10]
                _wc  = "#3fb950" if _wv == "PASS" else "#f85149"
                _badge_parts.append(
                    f'<span style="color:{_wc};font-size:11px;">WFO: {_wv} ({_wdt})</span>')
            if _ml_s:
                _mh  = _ml_s.get("holdout_accuracy", 0)
                _mn  = _ml_s.get("n_samples", 0)
                _mc  = "#3fb950" if _mh >= 0.55 else ("#e3b341" if _mh >= 0.50 else "#f85149")
                _badge_parts.append(
                    f'<span style="color:{_mc};font-size:11px;">ML: {_mh*100:.0f}% acc / {_mn} samples</span>')
            st.markdown(" &nbsp;|&nbsp; ".join(_badge_parts), unsafe_allow_html=True)

        st.markdown("---")
        # ── AI Provider + Key (for AI candle analysis) ───────────────────────
        with st.expander("🤖 AI Analysis (optional)", expanded=False):
            st.markdown(
                '<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:6px;'
                'padding:8px 10px;font-size:12px;color:#ccd6f6;margin-bottom:8px;">'
                '<b style="color:#58a6ff;">Groq is FREE</b> — sign up at '
                '<b>console.groq.com</b>, no credit card needed.<br>'
                'Anthropic requires paid credits at console.anthropic.com.</div>',
                unsafe_allow_html=True)

            _ai_provider = st.selectbox(
                "AI Provider",
                ["Groq (Free)", "Anthropic (Claude)"],
                key="ai_provider",
                help="Groq is free with Llama 3.3 70B. Anthropic requires paid credits.",
            )

            if "Groq" in _ai_provider:
                _groq_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    key="groq_api_key",
                    placeholder="gsk_...",
                    help="Get free key at console.groq.com → API Keys",
                )
                if _groq_key:
                    st.caption("✅ Groq key set — free AI analysis ready (Llama 3.3 70B)")
                else:
                    st.caption("Paste Groq key above. Free at console.groq.com")
            else:
                _ant_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    key="anthropic_api_key",
                    placeholder="sk-ant-...",
                    help="Get key at console.anthropic.com. Requires paid credits.",
                )
                if _ant_key:
                    st.caption("✅ Anthropic key set (Claude 3.5 Sonnet)")
                else:
                    st.caption("Paste Anthropic key above")

        st.markdown("---")
        run_btn = st.button("Run Analysis", use_container_width=True, type="primary")
        if st.button("⚡ Auto Finder", use_container_width=True, key="sidebar_autofinder_btn",
                     help="Scan ALL Binance altcoins for live momentum signals — no analysis run needed."):
            st.session_state["_show_autofinder"] = True
            st.rerun()
        st.caption("Data: Binance (data-api.binance.vision) | Risk model: 2% fixed")

        # Show notice if a strategy was just applied from Auto Analyzer
        if "_applied_strategy_notice" in st.session_state:
            st.success(st.session_state.pop("_applied_strategy_notice"))

    # ── All heavy computation gated behind Run Analysis button ────────────────
    if not ticker:
        st.info("Enter a ticker symbol in the sidebar.")
        return

    # Show stale warning if settings changed since last run
    _sess_key = "-".join(sorted(allowed_sessions)) if use_session else "all"
    _ema_key = f"ema{ema_filter}_{ema_period}_{ema_tf_mode}_{ema_htf_period}" if ema_filter else "ema_off"
    run_key = (f"{ticker}_{timeframe}_{history_label}_{min_body_pct:.2f}_"
               f"{min_vol_mult}_{sl_mode}_{trail_mult}_{transaction_cost}_"
               f"{use_time_exit}_{max_hold_bars}_"
               f"adx{adx_filter}_{adx_threshold}_{adx_tf_mode}_gap{di_gap_min}_{direction}_"
               f"sess{_sess_key}_{_ema_key}")
    _last_key   = st.session_state.get("run_key")
    _has_results = "opt_results" in st.session_state

    if _has_results and _last_key != run_key:
        st.sidebar.warning("⚠️ Settings changed — click **Run Analysis** to update.")

    # ── Auto Finder shortcut — always works, even without analysis ───────────
    if st.session_state.get("_show_autofinder"):
        _af_df = st.session_state.get("df_full", pd.DataFrame())
        _af_tc = st.session_state.get("transaction_cost", transaction_cost)
        if st.button("← Back to Analysis", key="autofinder_back_btn"):
            st.session_state.pop("_show_autofinder", None)
            st.rerun()
        render_auto_analyzer(ticker, _af_df, _af_tc, timeframe)
        return

    # Nothing computed yet — show guide and prompt, don't show analysis tabs
    if not _has_results and not run_btn:
        render_strategy_guide()
        return

    # On button click: fetch, compute, optimize everything fresh
    if run_btn:
        with st.spinner(f"Fetching {ticker} ({timeframe}) from Binance..."):
            df_full = _binance_fetch(ticker, timeframe, history_days)

        if df_full.empty:
            st.error(
                f"❌ No data found for **{ticker}** on Binance.\n\n"
                "Enter a valid Binance spot symbol: `BTC`, `ETH`, `SOL`, `MNT`, `HYPE`, `BTCUSDT`...")
            return

        df = trim_by_days(df_full, history_days)
        if df.empty:
            st.error("No data in selected time range.")
            return

        years      = (df.index[-1] - df.index[0]).days / 365
        start_date = df.index[0].date()

        # ADX computation — cached in session_state by run_key to avoid recomputing
        _HIGHER_TF = {"1H": "4H", "4H": "1D", "1D": "1D"}
        _adx_cache_key = f"adxcache_{run_key}"
        if st.session_state.get("_adx_cache_key") == _adx_cache_key and                 "adx_df_signal" in st.session_state:
            adx_df_signal     = st.session_state["adx_df_signal"]
            adx_series_higher = st.session_state.get("adx_series_higher")
        else:
            adx_df_signal = calculate_adx(df) if adx_filter else None
            adx_series_higher = None
            if adx_filter and adx_tf_mode in ("One timeframe higher", "Both"):
                higher_tf = _HIGHER_TF.get(timeframe, "1D")
                if higher_tf != timeframe:
                    with st.spinner(f"Fetching {higher_tf} data for ADX filter..."):
                        df_higher = _binance_fetch(ticker, higher_tf, history_days)
                    if not df_higher.empty:
                        adx_series_higher = calculate_adx(df_higher)["adx"]
                else:
                    adx_series_higher = adx_df_signal["adx"] if adx_df_signal is not None else None
            st.session_state["adx_df_signal"]     = adx_df_signal
            st.session_state["adx_series_higher"] = adx_series_higher
            st.session_state["_adx_cache_key"]    = _adx_cache_key
        adx_series_signal = adx_df_signal["adx"]      if adx_df_signal is not None else None
        di_plus_signal    = adx_df_signal["di_plus"]  if adx_df_signal is not None else None
        di_minus_signal   = adx_df_signal["di_minus"] if adx_df_signal is not None else None

        if not adx_filter:
            adx_for_filter = None
        elif adx_tf_mode == "Same as signal timeframe":
            adx_for_filter = adx_series_signal
        elif adx_tf_mode == "One timeframe higher":
            adx_for_filter = adx_series_higher if adx_series_higher is not None else adx_series_signal
        else:  # "Both"
            if adx_series_higher is not None and adx_series_signal is not None:
                sig_aligned    = adx_series_signal.reindex(df.index, method="ffill")
                high_aligned   = adx_series_higher.reindex(df.index, method="ffill")
                adx_for_filter = pd.concat([sig_aligned, high_aligned], axis=1).min(axis=1)
            else:
                adx_for_filter = adx_series_signal

        # ── EMA computation ──────────────────────────────────────────────────
        ema_series_signal = None
        ema_series_htf    = None

        if ema_filter:
            # Same-TF EMA — computed from signal timeframe data (shift(1) inside calculate_ema)
            ema_series_signal = calculate_ema(df, ema_period)

            # HTF EMA — only when "Higher timeframe" mode is selected
            # Use _HTF_MAP (not _HIGHER_TF): 1D → 1W, not 1D → 1D
            if ema_tf_mode == "Higher timeframe":
                _ema_htf_tf = _HTF_MAP.get(timeframe, "1D")
                if _ema_htf_tf != timeframe:
                    with st.spinner(f"Fetching {_ema_htf_tf} data for EMA filter..."):
                        _df_ema_htf = _binance_fetch(ticker, _ema_htf_tf, history_days + 365)
                    if not _df_ema_htf.empty:
                        # Compute HTF EMA and forward-fill to signal TF index
                        _ema_htf_raw      = calculate_ema(_df_ema_htf, ema_htf_period)
                        ema_series_htf    = _ema_htf_raw.reindex(df.index, method="ffill")
                else:
                    ema_series_htf = ema_series_signal

        # Combine: if both TF EMAs are active, price must satisfy BOTH
        # Use the more restrictive (HTF) when available, otherwise same-TF
        if ema_filter:
            if ema_series_htf is not None and ema_tf_mode == "Higher timeframe":
                # Both EMAs must be satisfied — use HTF (more restrictive)
                # The same-TF check is also applied inside detect_qualifying_candles
                ema_for_filter     = ema_series_htf
                ema_same_tf_check  = ema_series_signal
            else:
                ema_for_filter     = ema_series_signal
                ema_same_tf_check  = None
        else:
            ema_for_filter     = None
            ema_same_tf_check  = None

        qualifying = detect_qualifying_candles(
            df, min_body_pct, min_vol_mult,
            adx_filter=adx_filter, adx_threshold=adx_threshold,
            adx_series=adx_for_filter,
            di_plus_series=di_plus_signal, di_minus_series=di_minus_signal,
            di_gap_min=di_gap_min, direction=direction,
            ema_filter=ema_filter,
            ema_series=ema_for_filter,
        )

        # If both same-TF and HTF EMA are active, apply same-TF filter too
        if ema_filter and ema_same_tf_check is not None and not qualifying.empty:
            _ema_st_aligned = ema_same_tf_check.reindex(qualifying.index, method="ffill")
            if direction == "long":
                qualifying = qualifying[qualifying["close"] > _ema_st_aligned]
            else:
                qualifying = qualifying[qualifying["close"] < _ema_st_aligned]

        # ── Session filter (4H / 1H only) ────────────────────────────────────
        if use_session and timeframe in ("4H", "1H") and allowed_sessions:
            def _sess_ok(ts):
                return get_session((ts + WIB_OFFSET).hour) in allowed_sessions
            qualifying = qualifying[qualifying.index.to_series().apply(_sess_ok)]
            if len(qualifying) < 5:
                st.warning(
                    f"Session filter left only {len(qualifying)} qualifying candles — "
                    "need at least 5. Try enabling more sessions or switching to 1D timeframe.")
                return

        if len(qualifying) < 5:
            st.warning(f"Only {len(qualifying)} qualifying candles — need at least 5. "
                       "Relax the body %, volume, ADX, or session filters.")
            return

        try:
            opt_df, all_trades = optimize(df, qualifying, sl_mode, trail_mult,
                                          transaction_cost, max_hold_bars, use_time_exit,
                                          direction=direction)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)
            return

        # Clear all derivative caches so they rebuild after a fresh run
        for _ck in ("best_trades","best_m","_bt_key","_sl_comp_cache",
                    "_eq_fig","_heatmap_fig","_eq_fig_key","hold_df","hold_key",
                    "adx_df_signal","adx_series_higher","_adx_cache_key",
                    "_res_fig","_res_fig_key","_t4_fig_key",
                    "_mae_key","_mae_data","_mae_fig","_mae_fig_key",
                    "_adv_stats_key","hold_parent_run_key",
                    "ema_series_signal","ema_series_htf","_ema_htf_df"):
            st.session_state.pop(_ck, None)
        st.session_state.update({
            "opt_results":    opt_df,
            "ticker":         ticker,
            "df":             df,
            "df_full":        df_full,
            "qualifying":     qualifying,
            "all_trades":     all_trades,
            "transaction_cost": transaction_cost,
            "sl_mode":        sl_mode,
            "trail_mult":     trail_mult,
            "run_key":        run_key,
            "years":          years,
            "start_date":     start_date,
            "max_hold_bars":  max_hold_bars,
            "use_time_exit":  use_time_exit,
            "direction":      direction,
            "timeframe":      timeframe,
            "adx_filter":     adx_filter,
            "di_gap_min":     di_gap_min,
            "adx_threshold":  adx_threshold,
            "use_session":    use_session,
            "allowed_sessions": allowed_sessions,
            "ema_filter":     ema_filter,
            "ema_tf_mode":    ema_tf_mode,
        })

    # ── Read from session_state (works for both fresh run and cached) ──────────

    opt_df        = st.session_state["opt_results"]
    df            = st.session_state["df"]
    df_full       = st.session_state.get("df_full", df)
    qualifying    = st.session_state["qualifying"]
    all_trades    = st.session_state["all_trades"]
    tc            = st.session_state["transaction_cost"]
    _sl_mode      = st.session_state["sl_mode"]
    _trail        = st.session_state["trail_mult"]
    years         = st.session_state["years"]
    start_date    = st.session_state["start_date"]
    _max_hold     = st.session_state.get("max_hold_bars", 5)
    _time_exit    = st.session_state.get("use_time_exit", False)
    _direction    = st.session_state.get("direction", "long")
    _timeframe    = st.session_state.get("timeframe", timeframe)
    # Use run-time values (not current sidebar values) for display consistency
    adx_filter    = st.session_state.get("adx_filter",    adx_filter)
    di_gap_min    = st.session_state.get("di_gap_min",    di_gap_min)
    adx_threshold = st.session_state.get("adx_threshold", adx_threshold)
    adx_tf_mode   = st.session_state.get("adx_tf_mode",   adx_tf_mode)
    ema_filter     = st.session_state.get("ema_filter",     False)
    ema_period     = st.session_state.get("ema_period",     200)   # widget key — auto-tracked
    ema_tf_mode    = st.session_state.get("ema_tf_mode",    "Same as signal timeframe")
    ema_htf_period = st.session_state.get("ema_htf_period", 200)   # widget key — auto-tracked
    # Session values from last run
    _run_use_session     = st.session_state.get("use_session", False)
    _run_allowed_sessions = st.session_state.get("allowed_sessions", [])

    # Sidebar info badges (show cached values, no recompute)
    _run_use_sess  = st.session_state.get("use_session", False)
    _run_sess_list = st.session_state.get("allowed_sessions", [])
    _sess_badge    = (f" | 🕐 {', '.join(_run_sess_list)}"
                      if _run_use_sess and _run_sess_list else "")
    st.sidebar.markdown(
        f'<span style="color:#3fb950;font-size:12px;">&#x2705; Last run: {ticker} ({_timeframe}) — {len(df)} bars{_sess_badge}</span>',
        unsafe_allow_html=True)
    st.sidebar.caption(f"Data: {start_date} to {df.index[-1].date()}")

    # Title
    _dir_emoji = "📉 SHORT" if _direction == "short" else "📈 LONG"
    st.title(f"{st.session_state.get('ticker', ticker)} ({_timeframe}) — Momentum Candle Strategy [{_dir_emoji}]")
    st.caption(f"Data: {start_date} to {df.index[-1].date()} "
               f"({years:.1f} years) | {len(df)} bars | {len(qualifying)} qualifying triggers")

    # ── Regime Banner ────────────────────────────────────────────────────────
    _regime_cache_key = f"regime_{df.index[-1]}_{_timeframe}_{_direction}"
    if st.session_state.get("_regime_cache_key") != _regime_cache_key:
        _adx_df_for_regime = calculate_adx(df)
        _last_bar_i        = len(df) - 1
        # Fetch market context data (cached, low overhead)
        _fg_data  = fetch_fear_greed()
        _btcd_raw = fetch_btc_dominance()
        # Compute BTC.D rising flag using session-cached previous value
        _prev_btcd = st.session_state.get("_btcd_prev", _btcd_raw.get("btc_d", 50.0))
        _btcd_rising = _btcd_raw.get("btc_d", 50.0) > _prev_btcd
        st.session_state["_btcd_prev"] = _btcd_raw.get("btc_d", 50.0)
        _btcd_data = {**_btcd_raw, "rising": _btcd_rising}
        # Wyckoff phase from last bar
        _wyckoff = detect_wyckoff_phase(df, _last_bar_i, _adx_df_for_regime)
        _regime_result     = calculate_regime_score(
            df, _last_bar_i, _direction, _adx_df_for_regime,
            htf_ema_series=None, timeframe=_timeframe, ticker=ticker,
            fear_greed_data=_fg_data, btc_dom_data=_btcd_data,
        )
        _regime_result["wyckoff"] = _wyckoff
        st.session_state["_regime_result"]    = _regime_result
        st.session_state["_regime_cache_key"] = _regime_cache_key
        st.session_state["_intel_adx_df"]     = _adx_df_for_regime
    _regime_result = st.session_state.get("_regime_result", {})
    render_regime_banner(_regime_result)

    if opt_df.empty:
        st.warning("No parameter combination produced >=3 trades. Try different criteria.")
        return

    best    = opt_df.iloc[0]
    best_wr = best["win_rate"]

    # ── Cache best_trades / best_m — only recompute when Run is clicked ───────
    _bt_key = (f"bt_{st.session_state.get('run_key','')}_"
               f"{_max_hold}_{_time_exit}")
    if run_btn or "best_trades" not in st.session_state or             st.session_state.get("_bt_key") != _bt_key:
        _bt = run_backtest(
            df, qualifying,
            best["retracement"], best["sl_dist"], best["tp_mode"],
            _sl_mode, _trail, tc, _max_hold, _time_exit,
            direction=_direction)
        st.session_state["best_trades"] = _bt
        st.session_state["best_m"]      = calc_metrics(_bt)
        st.session_state["_bt_key"]     = _bt_key
    best_trades = st.session_state["best_trades"]
    best_m      = st.session_state["best_m"]

    # ── Tabs — reliable tab persistence via fast polling ─────────────────────
    st.markdown("""
<script>
(function(){
  const KEY="mcs_tab";
  let _attached=false;
  function tabs(){return Array.from(document.querySelectorAll('[data-testid="stTabs"] button[role="tab"]'));}
  function attach(btns){
    if(_attached) return;
    btns.forEach((b,i)=>{
      b.addEventListener("click",()=>sessionStorage.setItem(KEY,String(i)),{once:false});
    });
    _attached=true;
  }
  function restore(btns){
    const s=sessionStorage.getItem(KEY);
    if(s===null) return;
    const i=parseInt(s,10);
    const active=btns.findIndex(b=>b.getAttribute("aria-selected")==="true");
    if(btns[i] && active!==i) btns[i].click();
  }
  function tick(){
    const btns=tabs();
    if(!btns.length) return;
    attach(btns);
    restore(btns);
  }
  // Run immediately then poll until settled
  tick();
  const t=setInterval(()=>{tick();},80);
  setTimeout(()=>clearInterval(t),4000);
})();
</script>
""", unsafe_allow_html=True)

    # Analysis results are available — show all tabs
    # Add a small guide link at top so users can always get back to it
    st.markdown(
        '<div style="color:#8892b0;font-size:12px;margin-bottom:4px;">'
        '📖 New here? The <b>Strategy Guide</b> tab explains how everything works.</div>',
        unsafe_allow_html=True)

    # ── Tab variable names describe content, not position ────────────────────
    # Order: Learn → Build → Validate → Train → Analyze → Live
    (t_guide,       # 1. 📖 How the strategy works
     t_finder,      # 2. 🔍 Optimize parameters
     t_backtest,    # 3. 📊 Performance metrics & equity curve
     t_tradelog,    # 4. 📋 Every individual trade
     t_wfo,         # 5. ✅ Walk-Forward robustness test
     t_advstats,    # 6. 📈 Deep statistics & risk metrics
     t_candlestats, # 7. 🕯️ Candle pattern distribution & MAE
     t_ml,          # 8. 🤖 ML classifier — train BEFORE Intelligence
     t_intel,       # 9. 🧠 Per-candle similarity + AI verdict
     t_scanner,     # 10. 📡 Real-time signal scanner
     t_portfolio,   # 11. 💼 Portfolio & position sizing simulator
    ) = st.tabs([
        "📖 Guide",
        "🔍 Strategy Finder",
        "📊 Backtest Results",
        "📋 Trade Log",
        "✅ WFO Validation",
        "📈 Advanced Stats",
        "🕯️ Candle Stats",
        "🤖 ML Classifier",
        "🧠 Intelligence",
        "📡 Live Scanner",
        "💼 Portfolio",
    ])


    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — 📖 Guide
    # ═══════════════════════════════════════════════════════════════════════
    with t_guide:
        render_strategy_guide()

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — 🔍 Strategy Finder
    # ═══════════════════════════════════════════════════════════════════════
    with t_finder:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            adx_badge  = ""
            if adx_filter:
                di_info = f" + DI gap ≥ {di_gap_min:.0f}" if di_gap_min > 0 else ""
                adx_badge = (f'<div style="color:#ffd700;font-size:13px;margin-top:6px;">'
                             f'⚡ ADX filter ON — threshold ≥ {adx_threshold}'
                             f'{di_info} ({adx_tf_mode})</div>')
            dir_color  = "#ff6b6b" if _direction == "short" else "#64ffda"
            dir_badge  = ("📉 SHORT — bearish momentum candles" if _direction == "short"
                          else "📈 LONG — bullish momentum candles")
            _sess_active   = st.session_state.get("use_session", False)
            _sess_active_l = st.session_state.get("allowed_sessions", [])
            sess_badge = (
                f'<div style="color:#7c83fd;font-size:13px;margin-top:4px;">'
                f'🕐 Session filter: {", ".join(_sess_active_l)} (WIB)</div>'
                if _sess_active and _sess_active_l and _timeframe in ("4H", "1H") else ""
            )
            ema_badge = ""
            if ema_filter:
                _htf_lbl  = _HTF_LABEL.get(_timeframe, "HTF")
                _ema_desc = f"{_timeframe} EMA {ema_period}"
                if ema_tf_mode == "Higher timeframe":
                    _ema_desc += f" + {_htf_lbl} EMA {ema_htf_period}"
                ema_badge = (f'<div style="color:#64ffda;font-size:13px;margin-top:4px;">'
                             f'📈 EMA filter ON — {_ema_desc}</div>')
            st.markdown(f"""<div style="font-size:36px;font-weight:800;color:{dir_color};margin-bottom:4px;">
{len(qualifying)} qualifying candles
</div>
<div style="color:#8892b0;font-size:15px;">
found in {years:.1f} years of {ticker} data
&nbsp;|&nbsp; {len(qualifying)/max(years,0.1):.1f} per year average
</div>
<div style="color:{dir_color};font-size:13px;margin-top:4px;">{dir_badge}</div>
{adx_badge}{sess_badge}{ema_badge}""", unsafe_allow_html=True)

        with col_r:
            best_label = RETRACE_LABELS[best["retracement"]]
            best_sl    = SL_LABELS[best["sl_dist"]]
            pf_disp    = f"{best['profit_factor']:.2f}" if best["profit_factor"] < 90 else "inf"
            dir_row    = (f'<div class="best-row"><span class="best-key">Direction</span>'
                          f'<span class="best-val" style="color:{"#ff6b6b" if _direction == "short" else "#64ffda"};">'
                          f'{"📉 SHORT" if _direction == "short" else "📈 LONG"}</span></div>')
            sl_label   = ("above entry" if _direction == "short" else "below entry")
            adx_row    = ""
            if adx_filter:
                di_row_info = f" + DI gap ≥ {di_gap_min:.0f}" if di_gap_min > 0 else ""
                adx_row = (f'<div class="best-row"><span class="best-key">ADX Filter</span>'
                           f'<span class="best-val" style="color:#ffd700;">ON — ≥{adx_threshold}'
                           f'{di_row_info} ({adx_tf_mode})</span></div>')
            retrace_desc = ("retrace UP from close" if _direction == "short"
                            else "retrace DOWN from close")
            _sess_row = ""
            if st.session_state.get("use_session") and st.session_state.get("allowed_sessions"):
                _sa = st.session_state["allowed_sessions"]
                _sess_row = (f'<div class="best-row"><span class="best-key">Session Filter</span>'
                             f'<span class="best-val" style="color:#7c83fd;">'
                             f'🕐 {", ".join(_sa)} (WIB)</span></div>')
            _ema_row = ""
            if ema_filter:
                _htf_lbl   = _HTF_LABEL.get(_timeframe, "HTF")
                _ema_same  = f"{_timeframe} EMA {ema_period}"
                _ema_htf   = (f" + {_htf_lbl} EMA {ema_htf_period}"
                               if ema_tf_mode == "Higher timeframe" else "")
                _ema_row   = (f'<div class="best-row">'
                              f'<span class="best-key">EMA Trend Filter</span>'
                              f'<span class="best-val" style="color:#64ffda;">'
                              f'✅ {_ema_same}{_ema_htf}</span></div>')
            st.markdown(f"""<div class="best-box">
<h3>Best Setup Found</h3>
{dir_row}
<div class="best-row"><span class="best-key">Entry Retracement</span><span class="best-val">{best_label} — {retrace_desc}</span></div>
<div class="best-row"><span class="best-key">Stop Loss</span><span class="best-val">{best_sl} {sl_label}</span></div>
<div class="best-row"><span class="best-key">Take Profit Mode</span><span class="best-val">{best['tp_mode']}</span></div>
<div class="best-row"><span class="best-key">SL Management</span><span class="best-val">{_sl_mode}</span></div>
{adx_row}{_sess_row}{_ema_row}
<div class="best-row"><span class="best-key">Expected Win Rate ★</span><span class="best-val">{best['win_rate']:.1f}%</span></div>
<div class="best-row"><span class="best-key">Profit Factor</span><span class="best-val">{pf_disp}</span></div>
<div class="best-row"><span class="best-key">Edge Score</span><span class="best-val">{best['sharpe']:.2f}</span></div>
<div class="best-row"><span class="best-key">Total Trades</span><span class="best-val">{int(best['total_trades'])}</span></div>
</div>
<div style="color:#8892b0;font-size:11px;margin:4px 0 12px 0;padding:0 4px;">
★ Win Rate shown is from parameter sweep (no time exit applied). Open <b>Backtest Results</b> tab to see performance with your current time-exit settings.
</div>""", unsafe_allow_html=True)

        # (1R removed — only 2R/3R/Partial are supported TP modes)

        if best["tp_mode"] == "Partial":
            st.markdown("---")
            render_partial_explanation(all_trades)

        st.markdown("---")
        st.markdown("### Top 5 Parameter Combinations")
        top5 = opt_df.head(5).copy()
        top5.index = range(1, len(top5) + 1)
        top5["retracement"] = top5["retracement"].map(RETRACE_LABELS)
        top5["sl_dist"]     = top5["sl_dist"].map(SL_LABELS)
        top5.rename(columns={
            "retracement":"Entry Retrace","sl_dist":"Stop Loss","tp_mode":"TP Mode",
            "total_trades":"Trades","win_rate":"Win Rate %","avg_r":"Avg R",
            "profit_factor":"Profit Factor","max_drawdown":"Max DD %","sharpe":"Edge Score",
            "longest_losing_streak":"Max Loss Streak","avg_hold_days":"Avg Hold Days",
        }, inplace=True)
        st.dataframe(top5, use_container_width=True)

        st.markdown("---")
        render_sl_comparison(df, qualifying, best["retracement"],
                             best["sl_dist"], best["tp_mode"], _trail, tc, direction=_direction)

        render_hold_time_analysis(
            df, qualifying,
            best["retracement"], best["sl_dist"], best["tp_mode"],
            _sl_mode, _trail, tc, ticker,
            default_hold_bars=_max_hold,
            direction=_direction,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — 📊 Backtest Results
    # ═══════════════════════════════════════════════════════════════════════
    with t_backtest:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">📊 Backtest Results</b> — Did the strategy actually make money historically? '
            'This tab shows the equity curve, win rate, profit factor, and monthly heatmap for the best parameter set found by Strategy Finder. '
            'A rising equity curve + profit factor above 1.0 = genuine edge. '
            '<b>Read this before going live.</b></div>', unsafe_allow_html=True)
        st.markdown("### Performance Metrics (Best Setup)")
        render_metrics_grid(best_m, cols=5)
        st.markdown("---")
        # Cache plotly figures — only rebuild when best_trades changes
        _fig_key = st.session_state.get("_bt_key", "")
        if st.session_state.get("_eq_fig_key") != _fig_key:
            st.session_state["_eq_fig"]      = plot_equity_curve(best_trades, ticker)
            st.session_state["_heatmap_fig"] = plot_monthly_heatmap(best_trades)
            st.session_state["_eq_fig_key"]  = _fig_key
        st.plotly_chart(st.session_state["_eq_fig"],      use_container_width=True)
        st.plotly_chart(st.session_state["_heatmap_fig"], use_container_width=True, config={"displayModeBar":False})

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 — 📋 Trade Log
    # ═══════════════════════════════════════════════════════════════════════
    with t_tradelog:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">📋 Trade Log</b> — Every individual trade the strategy would have taken historically. '
            'Use this to understand WHEN trades win vs lose — are losses clustered in certain months? '
            'After a drawdown? Filter and study the patterns to understand your edge\'s weaknesses.</div>', unsafe_allow_html=True)
        st.markdown("### All Historical Trades (Best Setup)")
        st.markdown("""<div class="info-box" style="padding:12px 16px;margin-bottom:12px;">
<span style="color:#58a6ff;font-weight:600;">📌 Candle count convention:</span>
&nbsp; Candle 1 = the first full candle after your entry fills.
A <b>Time Exit at candle N</b> means the trade was still open after <b>(N-1) candles</b>
and closed at the <b>open price of candle N</b> (not its close).
TP / SL always takes priority if hit earlier.
</div>""", unsafe_allow_html=True)
        if not best_trades:
            st.info("No trades to display.")
        else:
            base_cols = [
                "trigger_date","trigger_body_pct","trigger_vol_mult",
                "entry_price","entry_date","sl_price","tp_price",
                "exit_price","exit_date","result","r_mult",
                "candles_held","exit_type",
            ]
            base_names = [
                "Trigger Date","Body %","Vol Mult",
                "Entry Price","Entry Date","SL Price","TP Price",
                "Exit Price","Exit Date","Result","R Mult",
                "Candles Held","Exit Type",
            ]
            # Include ADX column only when filter was active
            show_adx = adx_filter and any(
                not np.isnan(t.get("adx_value", float("nan"))) for t in best_trades)
            if show_adx:
                base_cols.append("adx_value")
                base_names.append("ADX")

            log = pd.DataFrame(best_trades)[base_cols].copy()
            log.columns = base_names

            # Format Exit Type — short label, candle number as suffix
            def _fmt_exit_type(row):
                et = row["Exit Type"]
                if et == "Time Exit":
                    n = int(row["Candles Held"])
                    return f"Time Exit (c{n})"
                if et == "TP":
                    return "Take Profit"
                if et == "SL":
                    return "Stop Loss"
                return et
            log["Exit Type"] = log.apply(_fmt_exit_type, axis=1)

            # Short date format — drop seconds, keep WIB only for intraday
            def _fmt_date(ts):
                if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                    return ""
                ts = pd.Timestamp(ts) + pd.Timedelta(hours=7)
                if timeframe in ("4H", "1H"):
                    return ts.strftime("%Y-%m-%d %H:%M")
                return ts.strftime("%Y-%m-%d")

            for c in ["Trigger Date", "Entry Date", "Exit Date"]:
                log[c] = pd.to_datetime(log[c]).apply(_fmt_date)

            # Round numeric columns
            log["Body %"]   = log["Body %"].round(1)
            log["Vol Mult"] = log["Vol Mult"].round(2)
            log["R Mult"]   = log["R Mult"].round(3)

            # Drop redundant raw candles column — info is in Exit Type
            display_log = log.drop(columns=["Candles Held"])

            def color_result(val):
                if val == "Win":   return "color:#64ffda"
                if val == "Loss":  return "color:#ff6b6b"
                return "color:#ffd700"

            def color_exit(val):
                if val == "Take Profit":        return "color:#64ffda"
                if val == "Stop Loss":          return "color:#ff6b6b"
                if val.startswith("Time Exit"): return "color:#ffd700"
                return "color:#8892b0"

            col_cfg = {
                "Trigger Date": st.column_config.TextColumn("Trigger",    width=110),
                "Body %":       st.column_config.NumberColumn("Body %",   width=70,  format="%.1f"),
                "Vol Mult":     st.column_config.NumberColumn("Vol×",      width=60,  format="%.2f"),
                "Entry Price":  st.column_config.NumberColumn("Entry",     width=105, format="%.4f"),
                "Entry Date":   st.column_config.TextColumn("Entry Date",  width=115),
                "SL Price":     st.column_config.NumberColumn("SL",        width=105, format="%.4f"),
                "TP Price":     st.column_config.NumberColumn("TP",        width=105, format="%.4f"),
                "Exit Price":   st.column_config.NumberColumn("Exit",      width=105, format="%.4f"),
                "Exit Date":    st.column_config.TextColumn("Exit Date",   width=115),
                "Result":       st.column_config.TextColumn("Result",      width=70),
                "R Mult":       st.column_config.NumberColumn("R",         width=70,  format="%.3f"),
                "Exit Type":    st.column_config.TextColumn("Exit Type",   width=130),
            }
            if show_adx:
                col_cfg["ADX"] = st.column_config.NumberColumn(
                    "ADX", width=65, format="%.1f",
                    help="ADX value at trigger candle")

            st.dataframe(
                display_log.style
                    .applymap(color_result, subset=["Result"])
                    .applymap(color_exit,   subset=["Exit Type"]),
                use_container_width=True,
                height=520,
                column_config=col_cfg,
            )
            st.caption(
                "Time Exit (cN) = exited at open of candle N after entry  |  "
                "Candle 1 = first full candle after entry fills"
            )
            st.download_button("⬇ Download Trade Log CSV",
                               log.to_csv(index=False).encode("utf-8"),
                               f"{ticker}_trades.csv", "text/csv")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 8 — 🕯️ Candle Stats  (placed in code after Trade Log; renders as tab 8)
    # ═══════════════════════════════════════════════════════════════════════
    with t_candlestats:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">🕯️ Candle Stats</b> — Deep analysis of the signal candles themselves. '
            'Tells you: what time of day do signals cluster? What body sizes perform best? '
            'What does price typically do BEFORE the signal closes (MAE = worst drawdown before TP/SL)? '
            'Use this to refine your entry timing and stop placement.</div>', unsafe_allow_html=True)
        # ── Period selector ─────────────────────────────────────────────────
        st.markdown("#### 📅 Analysis Period")
        _period_opts = {
            "All data":    None,
            "Last 3 months":  90,
            "Last 6 months": 180,
            "Last 1 year":   365,
            "Last 2 years":  730,
            "Last 3 years": 1095,
            "Last 5 years": 1825,
        }
        _sel_period = st.select_slider(
            "Filter charts to period:",
            options=list(_period_opts.keys()),
            value="All data",
            key="tab4_period",
            label_visibility="collapsed",
        )
        _period_days = _period_opts[_sel_period]

        # Slice df and qualifying to selected period
        if _period_days is not None:
            _cutoff      = df.index[-1] - pd.Timedelta(days=_period_days)
            df_tab4      = df[df.index >= _cutoff].copy()
            qual_tab4    = qualifying[qualifying.index >= _cutoff].copy()
            # Filter all_trades to only include trades whose trigger date is in window
            all_trades_tab4 = {
                k: [t for t in v if pd.Timestamp(t["trigger_date"]) >= _cutoff]
                for k, v in all_trades.items()
            }
        else:
            df_tab4         = df
            qual_tab4       = qualifying
            all_trades_tab4 = all_trades

        _n_q   = len(qual_tab4)
        _n_yrs = (df_tab4.index[-1] - df_tab4.index[0]).days / 365 if len(df_tab4) > 1 else 0
        st.caption(
            f"Showing: **{_sel_period}** — "
            f"{_n_q} qualifying candles over {_n_yrs:.1f} years "
            f"({df_tab4.index[0].date()} → {df_tab4.index[-1].date()})"
        )
        st.markdown("---")

        # ── Charts (cached by period + direction) ───────────────────────────
        _t4_fig_key = f"t4_{_sel_period}_{_direction}_{st.session_state.get('run_key','')}"
        if st.session_state.get("_t4_fig_key") != _t4_fig_key:
            st.session_state["_t4_reach"]   = plot_retrace_reach(df_tab4, qual_tab4, direction=_direction)
            st.session_state["_t4_winrate"] = plot_winrate_by_retrace(all_trades_tab4)
            st.session_state["_t4_seasonal"]= plot_seasonal(qual_tab4)
            st.session_state["_t4_fig_key"] = _t4_fig_key
        st.plotly_chart(st.session_state["_t4_reach"],    use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(st.session_state["_t4_winrate"],  use_container_width=True, config={"displayModeBar":False})
        with c2:
            st.plotly_chart(st.session_state["_t4_seasonal"], use_container_width=True, config={"displayModeBar":False})

        st.markdown("### Average Max Adverse Excursion by Retracement Level")
        st.caption(
            "MAE = how far price moved against your entry before the trade closed via TP or SL, "
            "measured in R (your risk distance). "
            "Lower MAE = cleaner entry, price moved in your favour quickly. "
            "Only TP/SL exits are included — Max Hold exits are excluded as they distort the average."
        )
        # Cache MAE computation — expensive loop, only rerun when period/direction/run changes
        _mae_key = f"mae_{_sel_period}_{_direction}_{st.session_state.get('run_key','')}"
        if st.session_state.get("_mae_key") != _mae_key:
            # Fast MAE: use DatetimeIndex directly, cache result by period+direction
            _mae_cache_key = f"mae_{_sel_period}_{_direction}_{st.session_state.get('run_key','')}"
            if st.session_state.get("_mae_key") == _mae_cache_key:
                _mae, _mae_counts = st.session_state["_mae_data"]
            else:
                _dt_idx = df_tab4.index
                _mae, _mae_counts = {}, {}
                for ret in RETRACEMENTS:
                    vals = []
                    for (r, sl, tp), trades in all_trades_tab4.items():
                        if r != ret:
                            continue
                        for t in trades:
                            if t.get("exit_type") not in ("TP", "SL"):
                                continue
                            risk = (t["sl_price"] - t["entry_price"] if _direction == "short"
                                    else t["entry_price"] - t["sl_price"])
                            if risk <= 0 or t["entry_date"] is None or t["exit_date"] is None:
                                continue
                            entry_ts = pd.Timestamp(t["entry_date"])
                            exit_ts  = pd.Timestamp(t["exit_date"])
                            trade_bars = df_tab4.loc[
                                (_dt_idx >= entry_ts) & (_dt_idx <= exit_ts)]
                            if trade_bars.empty:
                                continue
                            extreme_price = (trade_bars["high"].max() if _direction == "short"
                                             else trade_bars["low"].min())
                            mae_r = ((extreme_price - t["entry_price"]) / risk if _direction == "short"
                                     else (t["entry_price"] - extreme_price) / risk)
                            vals.append(max(0.0, min(mae_r, 1.5)))
                    _mae[ret]        = np.mean(vals) if vals else 0
                    _mae_counts[ret] = len(vals)
                st.session_state["_mae_data"] = (_mae, _mae_counts)
                st.session_state["_mae_key"]  = _mae_cache_key
            st.session_state["_mae"]      = _mae
            st.session_state["_mae_counts"] = _mae_counts
            st.session_state["_mae_key"]  = _mae_key
        mae        = st.session_state["_mae"]
        mae_counts = st.session_state["_mae_counts"]

        if st.session_state.get("_mae_fig_key") != _mae_key:
            colors = ["#64ffda" if v < 0.4 else "#ffd700" if v < 0.7 else "#ff6b6b"
                      for v in [mae[r] for r in RETRACEMENTS]]
            _mae_fig = go.Figure(go.Bar(
                x=[RETRACE_LABELS[r] for r in RETRACEMENTS],
                y=[mae[r] for r in RETRACEMENTS],
                marker_color=colors,
                text=[f"{mae[r]:.2f}R<br><span style='font-size:10px'>n={mae_counts[r]}</span>"
                      for r in RETRACEMENTS],
                textposition="outside"))
            _mae_fig.add_hline(y=1.0, line_dash="dash", line_color="#ff6b6b", line_width=1,
                               annotation_text="1.0R = stop level",
                               annotation_position="top right",
                               annotation_font_color="#ff6b6b")
            _dark(_mae_fig,
                  title="Average Max Adverse Excursion (MAE) by Entry Retracement — TP/SL exits only",
                  xaxis_title="Entry Retracement",
                  yaxis_title="Avg MAE (R)",
                  yaxis=dict(range=[0, max(max(mae.values()) * 1.3, 1.3)]))
            st.session_state["_mae_fig"]     = _mae_fig
            st.session_state["_mae_fig_key"] = _mae_key
        st.plotly_chart(st.session_state["_mae_fig"], use_container_width=True,
                        config={"displayModeBar": False})

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 12 — 💼 Portfolio  (placed in code here; renders as tab 12)
    # ═══════════════════════════════════════════════════════════════════════
    with t_portfolio:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">💼 Portfolio Simulator</b> — How would your account grow if you traded this strategy with real money? '
            'Set your starting capital and risk % per trade. The simulator compounds your results trade-by-trade. '
            'Shows max drawdown in dollars, peak equity, and realistic growth curve. '
            '<b>Never risk more than 1-2% per trade.</b></div>', unsafe_allow_html=True)
        render_portfolio_simulator(best_trades, df, ticker)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 11 — 📡 Live Scanner  (placed in code here; renders as tab 11)
    # ═══════════════════════════════════════════════════════════════════════
    with t_scanner:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">📡 Live Scanner</b> — Real-time signal detection using the SAME rules as your backtest. '
            'Scans the last 20 candles for qualifying signals right now. '
            'When a SIGNAL appears, it shows you the exact entry price, stop loss, and take profit levels based on your best backtested parameters. '
            '<b>This is where you find trades to execute today.</b> '
            'Click "🧠 Analyze this signal" on any signal card to get AI analysis instantly.</div>', unsafe_allow_html=True)
        render_live_scanner(
            ticker, timeframe, min_body_pct, min_vol_mult,
            best["retracement"], best["sl_dist"], best["tp_mode"], best_wr,
            direction=_direction,
            adx_filter_on=adx_filter,
            adx_di_gap_min=di_gap_min,
            adx_threshold=adx_threshold,
            ema_filter=ema_filter,
            ema_period=ema_period,
            ema_tf_mode=ema_tf_mode,
            ema_htf_period=ema_htf_period,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5 — ⚡ Auto Finder  (placed in code here; renders as tab 5)
    # ═══════════════════════════════════════════════════════════════════════
    # Auto Finder is rendered above the tabs (always accessible) — no tab block needed

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 7 — 📈 Advanced Stats  (placed in code here; renders as tab 7)
    # ═══════════════════════════════════════════════════════════════════════
    with t_advstats:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">📈 Advanced Stats</b> — Goes deeper than win rate and profit factor. '
            'Shows your <b>streak risk</b> (how many losses in a row to expect), <b>risk of ruin</b> (probability of blowing up at your position size), '
            '<b>volatility regime performance</b> (does the strategy work better in trending or ranging markets?), '
            'and <b>Kelly Criterion</b> (the mathematically optimal position size). '
            'Read this to size positions safely and understand the strategy\'s worst-case behaviour.</div>', unsafe_allow_html=True)
        st.markdown("### 📊 Advanced Statistics — Best Setup")
        st.caption(
            "Deep-dive analysis on trade timing, streak patterns, volatility regimes, "
            "risk of ruin, and optimal position sizing.")
        # Cache key: rebuild only when best_trades changes
        _adv_key = st.session_state.get("_bt_key", "")
        if st.session_state.get("_adv_stats_key") != _adv_key:
            # Pre-compute and store plots to avoid rebuilding on widget interaction
            st.session_state["_adv_stats_key"] = _adv_key
        render_advanced_stats(best_trades,
                              st.session_state.get("ticker", ticker),
                              st.session_state.get("timeframe", timeframe))

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 9 — 🧠 Intelligence
    # ═══════════════════════════════════════════════════════════════════════
    with t_intel:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">🧠 Intelligence</b> — Analyzes each specific candle using 3 engines working together: '
            '<b>Similarity</b> (finds historically similar candles and reports what happened next), '
            '<b>ML Model</b> (machine learning score from the trained classifier), and '
            '<b>AI Verdict</b> (Groq/Llama gives a structured TRADE / WAIT / NO TRADE decision). '
            'Click any row in the candle table → see the full analysis on the right. '
            '<b>Train the ML Classifier tab first</b> to get ML scores here.</div>', unsafe_allow_html=True)
        _best_row = opt_df.iloc[0] if not opt_df.empty else None
        _best_params = {
            "retracement": float(_best_row["retracement"]) if _best_row is not None else 0.0,
            "sl_dist":     float(_best_row["sl_dist"])     if _best_row is not None else 0.01,
            "tp_mode":     str(_best_row["tp_mode"])       if _best_row is not None else "2R",
            "win_rate":    float(_best_row["win_rate"])    if _best_row is not None else 0.0,
            "pf":          float(_best_row.get("pf", 0))   if _best_row is not None else 0.0,
        }
        render_intelligence_tab(
            df, qualifying, all_trades, _direction,
            ticker, _timeframe, _regime_result,
            best_params=_best_params,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 6 — ✅ WFO Validation
    # ═══════════════════════════════════════════════════════════════════════
    with t_wfo:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">✅ WFO Validation</b> — Proves your strategy isn\'t just "fit to past data." '
            'Walk-Forward Optimisation repeatedly optimises on old data, then tests on NEW data the optimizer never saw. '
            '<b>PASS</b> = the edge is real and generalises. <b>FAIL</b> = the parameters are curve-fitted and will likely fail live. '
            'Always run WFO before going live. A backtest without WFO is just an illusion of profit.</div>', unsafe_allow_html=True)
        render_wfo_tab(df, detect_qualifying_candles, ticker, _timeframe, _direction)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 10 — 🤖 ML Classifier
    # ═══════════════════════════════════════════════════════════════════════
    with t_ml:
        st.markdown('<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:13px;color:#ccd6f6;">'
            '<b style="color:#58a6ff;">🤖 ML Classifier</b> — Trains a machine learning model on YOUR historical signal candles. '
            'It learns which candle characteristics (body size, volume, ADX, EMA alignment, DI gap) predict winners vs losers. '
            '<b>Step 1: Click "Train Model" below.</b> Step 2: Go to Intelligence tab — every candle will now show an ML probability score. '
            'The model uses 5-fold cross-validation to prevent overfitting. Accuracy above 55% = useful signal.</div>', unsafe_allow_html=True)
        _ml_adx = st.session_state.get("_intel_adx_df")
        if _ml_adx is None:
            _ml_adx = calculate_adx(df)
        render_ml_tab(qualifying, all_trades, _direction, ticker, _timeframe, _ml_adx)


if __name__ == "__main__":
    main()
