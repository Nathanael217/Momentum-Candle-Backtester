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

SL_DISTANCES = [0.005, 0.010, 0.015, 0.020]
SL_LABELS    = {0.005: "0.5%", 0.010: "1.0%", 0.015: "1.5%", 0.020: "2.0%"}

TP_MODES = ["2R", "3R", "Partial"]

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
                               direction: str = "long"):
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
                         adx_threshold=25):
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

    # Re-fetch whenever ticker OR timeframe changes, or refresh pressed
    live_key = f"{ticker}_{timeframe}"
    if refresh or "live_df" not in st.session_state or \
            st.session_state.get("live_key") != live_key:
        with st.spinner("Fetching latest candles..."):
            live_df     = fetch_live(ticker, timeframe)
            # Fetch HTF for ADX context — always, regardless of ADX filter toggle
            htf_days    = {"1H": 14, "4H": 60, "1D": 365}.get(timeframe, 60)
            live_df_htf = _binance_klines(ticker, _BINANCE_INTERVAL[htf], htf_days)
            if live_df_htf.empty:
                live_df_htf = _gateio_klines(ticker, _BINANCE_INTERVAL[htf], htf_days)
        st.session_state["live_df"]     = live_df
        st.session_state["live_df_htf"] = live_df_htf
        st.session_state["live_ts"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["live_key"]    = live_key

    live_df     = st.session_state.get("live_df",     pd.DataFrame())
    live_df_htf = st.session_state.get("live_df_htf", pd.DataFrame())
    live_ts     = st.session_state.get("live_ts",     "unknown")

    with col_a:
        st.caption(f"Last updated: {live_ts}")

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

    # Take last 20 completed candles (exclude the current incomplete one)
    scan = live_df.iloc[-21:-1].copy() if len(live_df) > 20 else live_df.iloc[:-1].copy()
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
        })

    scan_df   = pd.DataFrame(table_rows)
    adx_s_col = f"ADX ({tf_label})"
    adx_h_col = f"ADX ({htf_label})"

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

    display_cols = ["#", "Date", "Body %", "Vol Mult", "Body OK", "Vol OK",
                    "SIGNAL", adx_s_col, "Trend Dir", "DI≥10", "DI≥20", adx_h_col, "ADX Align"]
    hidden_cols  = ["_body_ok", "_vol_ok", "_signal", "_adx_s", "_adx_h", "_dip", "_dim"]

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

        if adx_filter_on and adx_di_gap_min > 0:
            req_label  = f"Required: Body + Volume + DI gap ≥ {adx_di_gap_min:.0f}"
            checklist  = f"{check_body} &nbsp; {check_vol} &nbsp; {'✅' if sr['DI≥10'] == '✅' and adx_di_gap_min <= 10 or sr['DI≥20'] == '✅' and adx_di_gap_min <= 20 else '✅'} DI gap ≥ {adx_di_gap_min:.0f}"
        else:
            req_label  = "Required: Body + Volume &nbsp;|&nbsp; DI columns = extra confluence"
            checklist  = f"{check_body} &nbsp; {check_vol} &nbsp; {check_di10} &nbsp; {check_di20}"

        st.markdown(f"""<div class="signal-card">
<h4>SIGNAL FOUND — {date_str}</h4>
<div class="signal-line">Trigger close: <span>${close:,.6f}</span> &nbsp;|&nbsp;
Body: <span>{sr['Body %']:.1f}%</span> &nbsp;|&nbsp;
Vol mult: <span>{sr['Vol Mult']:.2f}x</span></div>
<hr style="border-color:#1e2d1e; margin:8px 0;">
<div class="signal-line" style="font-size:12px;color:#8892b0;">{req_label}</div>
<div class="signal-line" style="letter-spacing:2px;">{checklist}</div>
<hr style="border-color:#1e2d1e; margin:8px 0;">
<div class="signal-line">ADX strength ({tf_label}): <span style="color:{adx_s_color};">{adx_s_disp}</span>
&nbsp;|&nbsp; ADX strength ({htf_label} HTF): <span style="color:{adx_h_color};">{adx_h_disp}</span>
&nbsp;|&nbsp; <span style="color:{adx_vcolor};">{adx_verdict}</span></div>
<div class="signal-line">Trend direction: <span style="color:{di_dir_color};">{di_dir_str}</span>
&nbsp;<span style="color:#8892b0;font-size:11px;">(DI− &gt; DI+ = bearish; DI+ &gt; DI− = bullish)</span></div>
<hr style="border-color:#1e2d1e; margin:8px 0;">
<div class="signal-line">Based on historical best setup for <span>{ticker}</span>:</div>
<div class="signal-line">Entry: <span>${entry:,.6f}</span> &nbsp;({RETRACE_LABELS[best_ret]} retrace from close)</div>
<div class="signal-line">SL: &nbsp;&nbsp;<span>${sl:,.6f}</span> &nbsp;({sl_desc})</div>
{tp_lines}
<div class="signal-line" style="margin-top:8px;">Historical win rate: <span>{wr_str}</span> ({best_tp} TP)</div>
</div>""", unsafe_allow_html=True)

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
    """Interactive strategy guide with diagrams for newcomers."""

    st.markdown("## 📖 Momentum Candle Strategy — Complete Guide")
    st.caption("Everything you need to understand this strategy from zero. Read top to bottom.")

    # ─── Section 1: The Problem ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤔 What Problem Does This Solve?")

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:8px 0;">
<div style="color:#ccd6f6;font-size:14px;line-height:1.7;">
Most traders lose because they <b>enter trades randomly</b> — they see price moving and jump in,
or they buy/sell based on gut feeling.<br><br>
This strategy fixes that by giving you a <b>rule-based system</b> that only enters when 3 specific
things happen at the same time. No guessing, no emotion.<br><br>
<b>The core idea:</b> Big institutions make huge moves. When they do, they leave a
footprint — a candle with a large body AND unusually high volume. That's your signal.
</div>
<div>
<div style="margin-bottom:8px;color:#ff6b6b;font-weight:600;">Without a system:</div>
<div style="color:#ccd6f6;font-size:13px;line-height:1.9;">
❌ Random entries — sometimes right, often wrong<br>
❌ Emotional decisions — buy high, sell low<br>
❌ No way to measure if you're improving
</div>
<div style="margin:10px 0 8px;color:#64ffda;font-weight:600;">With this system:</div>
<div style="color:#ccd6f6;font-size:13px;line-height:1.9;">
✅ Defined entry rules — same criteria every time<br>
✅ Fixed stop loss — know max loss before entering<br>
✅ Backtested results — historical proof first<br>
✅ Measurable edge — win rate &amp; profit factor tracked
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Section 2: The Momentum Candle ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🕯️ What Is a Momentum Candle?")
    st.markdown("A **momentum candle** is a special candle that passes two filters simultaneously:")

    # FIX 1: equal-height cards using a single HTML block instead of st.columns
    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:12px 0;">

<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:20px;box-sizing:border-box;">
<h4 style="color:#58a6ff;margin:0 0 10px 0;font-size:15px;">Filter 1: Body %</h4>
<p style="font-size:32px;margin:8px 0;text-align:center;">📏</p>
<p style="color:#ccd6f6;">The candle body must be <b>≥ 70%</b> of the total candle range.</p>
<p style="color:#8892b0;font-size:12px;margin:8px 0 0 0;">Body = close − open<br>Range = high − low<br>Body % = Body ÷ Range</p>
<p style="color:#64ffda;margin:10px 0 0 0;font-size:13px;">Strong directional move — no indecision.</p>
</div>

<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:20px;box-sizing:border-box;">
<h4 style="color:#58a6ff;margin:0 0 10px 0;font-size:15px;">Filter 2: Volume ×</h4>
<p style="font-size:32px;margin:8px 0;text-align:center;">📊</p>
<p style="color:#ccd6f6;">Volume must be <b>≥ 1.5×</b> the 7-candle average volume.</p>
<p style="color:#8892b0;font-size:12px;margin:8px 0 0 0;">Vol Mult = Today's Vol ÷ Avg(last 7 days)<br>≥ 1.5 = above average activity</p>
<p style="color:#64ffda;margin:10px 0 0 0;font-size:13px;">Institutional participation — smart money moving.</p>
</div>

<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:20px;box-sizing:border-box;">
<h4 style="color:#58a6ff;margin:0 0 10px 0;font-size:15px;">Direction</h4>
<p style="font-size:32px;margin:8px 0;text-align:center;">🧭</p>
<p style="color:#ccd6f6;"><b>Long mode:</b> Bullish candle (close &gt; open) — green</p>
<p style="color:#ccd6f6;margin-top:8px;"><b>Short mode:</b> Bearish candle (close &lt; open) — red</p>
<p style="color:#8892b0;font-size:12px;margin:8px 0 0 0;">Choose direction in the sidebar.</p>
<p style="color:#64ffda;margin:10px 0 0 0;font-size:13px;">Trade WITH momentum, not against it.</p>
</div>

</div>
""", unsafe_allow_html=True)

    # Responsive two-panel candle anatomy (HTML flex — no SVG stretching)
    st.markdown("#### Anatomy of a Qualifying Candle (Long)")
    st.components.v1.html("""
<style>
  body{margin:0;padding:0;background:#0d1117;}
  .ca-wrap{display:flex;gap:0;background:#0d1117;border-radius:8px;overflow:hidden;font-family:sans-serif;}
  .ca-panel{flex:1;min-width:0;padding:18px 16px 14px;box-sizing:border-box;display:flex;flex-direction:column;align-items:center;}
  .ca-left{border-right:1px solid #2d3250;}
  .ca-title{font-size:13px;font-weight:700;margin-bottom:2px;text-align:center;}
  .ca-sub{font-size:11px;margin-bottom:12px;text-align:center;}
  .ca-candle-wrap{display:flex;align-items:flex-start;gap:10px;width:100%;justify-content:center;}
  .ca-svg-col{flex-shrink:0;}
  .ca-lbl-col{display:flex;flex-direction:column;justify-content:space-between;font-size:11px;line-height:1.5;padding-top:4px;}
  .ca-vol{margin-top:10px;font-size:10px;text-align:center;}
  .ca-volbar{height:10px;border-radius:3px;margin:3px auto 0;}
  @media(max-width:400px){.ca-wrap{flex-direction:column;} .ca-left{border-right:none;border-bottom:1px solid #2d3250;}}
</style>
<div class="ca-wrap">
  <!-- LEFT: Normal candle -->
  <div class="ca-panel ca-left">
    <div class="ca-title" style="color:#8892b0;">Normal candle</div>
    <div class="ca-sub" style="color:#ff6b6b;">❌ Does NOT qualify</div>
    <div class="ca-candle-wrap">
      <div class="ca-svg-col">
        <svg viewBox="0 0 40 180" width="40" height="180" xmlns="http://www.w3.org/2000/svg">
          <!-- full wick -->
          <line x1="20" y1="5" x2="20" y2="175" stroke="#8892b0" stroke-width="2.5"/>
          <!-- small body -->
          <rect x="5" y="88" width="30" height="38" fill="#64ffda" opacity="0.65" rx="2"/>
        </svg>
      </div>
      <div class="ca-lbl-col" style="height:180px;">
        <span style="color:#8892b0;">▲ Large top wick</span>
        <span>
          <span style="color:#ccd6f6;">Small body</span><br>
          <span style="color:#ff6b6b;">Body% ≈ 30% → FAIL</span>
        </span>
        <span style="color:#8892b0;">▼ Large bottom wick</span>
      </div>
    </div>
    <div class="ca-vol">
      <div class="ca-volbar" style="width:44px;background:#4a5568;"></div>
      <span style="color:#8892b0;">Vol = normal</span>
    </div>
  </div>
  <!-- RIGHT: Momentum candle -->
  <div class="ca-panel">
    <div class="ca-title" style="color:#3fb950;">Momentum candle</div>
    <div class="ca-sub" style="color:#64ffda;">✅ QUALIFIES</div>
    <div class="ca-candle-wrap">
      <div class="ca-svg-col">
        <svg viewBox="0 0 40 180" width="40" height="180" xmlns="http://www.w3.org/2000/svg">
          <!-- tiny top wick -->
          <line x1="20" y1="5" x2="20" y2="14" stroke="#8892b0" stroke-width="2.5"/>
          <!-- tiny bottom wick -->
          <line x1="20" y1="166" x2="20" y2="175" stroke="#8892b0" stroke-width="2.5"/>
          <!-- large body -->
          <rect x="5" y="14" width="30" height="152" fill="#64ffda" opacity="0.9" rx="2"/>
          <!-- bracket -->
          <line x1="35" y1="14"  x2="38" y2="14"  stroke="#64ffda" stroke-width="1.5" stroke-dasharray="2"/>
          <line x1="35" y1="166" x2="38" y2="166" stroke="#64ffda" stroke-width="1.5" stroke-dasharray="2"/>
          <line x1="37" y1="14"  x2="37" y2="166" stroke="#64ffda" stroke-width="1.5"/>
        </svg>
      </div>
      <div class="ca-lbl-col" style="height:180px;">
        <span style="color:#8892b0;">Close (high)</span>
        <span style="color:#64ffda;font-weight:700;">Body ≥ 70%<br>of range ✅</span>
        <span style="color:#8892b0;">Open (low)</span>
      </div>
    </div>
    <div class="ca-vol">
      <div class="ca-volbar" style="width:44px;background:#7c83fd;"></div>
      <span style="color:#7c83fd;">Vol ≥ 1.5× avg ✅</span>
    </div>
  </div>
</div>
""", height=280)

    # ─── Section 3: The Trade Setup ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 How the Trade is Set Up")
    st.markdown("""
Once a momentum candle fires, you **wait for a pullback** (retrace) before entering.
This gives you a better price and tighter risk. Then you set a fixed Stop Loss below and a Take Profit target above.
""")

    # Trade setup SVG: max-width 820px so height never overflows iframe
    st.components.v1.html("""
<style>body{margin:0;padding:0;background:#0d1117;}</style>
<div style="max-width:820px;margin:0 auto;">
<svg viewBox="0 0 960 360" xmlns="http://www.w3.org/2000/svg"
     style="background:#0d1117;border-radius:8px;width:100%;display:block;">
  <!-- axes -->
  <line x1="70" y1="20" x2="70"  y2="320" stroke="#2d3250" stroke-width="1.5"/>
  <line x1="70" y1="320" x2="810" y2="320" stroke="#2d3250" stroke-width="1.5"/>

  <!-- Previous candles -->
  <rect x="90"  y="210" width="28" height="40" fill="#4a5568" opacity="0.5" rx="1"/>
  <rect x="128" y="195" width="28" height="50" fill="#4a5568" opacity="0.5" rx="1"/>
  <rect x="166" y="205" width="28" height="38" fill="#4a5568" opacity="0.5" rx="1"/>
  <rect x="204" y="185" width="28" height="48" fill="#4a5568" opacity="0.5" rx="1"/>

  <!-- Trigger candle -->
  <line x1="258" y1="160" x2="258" y2="310" stroke="#8892b0" stroke-width="1.5"/>
  <rect x="242" y="165" width="32" height="130" fill="#64ffda" opacity="0.9" rx="2"/>
  <text x="258" y="148" fill="#3fb950" font-size="11" text-anchor="middle" font-family="sans-serif" font-weight="bold">Trigger</text>
  <text x="258" y="136" fill="#8892b0" font-size="10" text-anchor="middle" font-family="sans-serif">Momentum candle</text>

  <!-- Horizontal price levels — labels on RIGHT side with plenty of room -->
  <!-- TP -->
  <line x1="242" y1="80"  x2="730" y2="80"  stroke="#64ffda" stroke-width="1.5" stroke-dasharray="5"/>
  <text x="736" y="84"  fill="#64ffda" font-size="11" font-family="sans-serif">Take Profit (3R)</text>

  <!-- Close -->
  <line x1="242" y1="165" x2="730" y2="165" stroke="#ccd6f6" stroke-width="1" stroke-dasharray="4"/>
  <text x="736" y="169" fill="#ccd6f6" font-size="11" font-family="sans-serif">Close of trigger</text>

  <!-- Entry -->
  <line x1="242" y1="200" x2="730" y2="200" stroke="#ffd700" stroke-width="1.5" stroke-dasharray="5"/>
  <text x="736" y="204" fill="#ffd700" font-size="11" font-family="sans-serif">Entry (23.6% retrace)</text>

  <!-- SL -->
  <line x1="242" y1="235" x2="730" y2="235" stroke="#ff6b6b" stroke-width="1.5" stroke-dasharray="5"/>
  <text x="736" y="239" fill="#ff6b6b" font-size="11" font-family="sans-serif">Stop Loss (2% below)</text>

  <!-- Retrace + entry candle -->
  <rect x="284" y="188" width="30" height="55" fill="#4a5568" opacity="0.7" rx="1"/>
  <text x="299" y="260" fill="#8892b0" font-size="9" text-anchor="middle" font-family="sans-serif">Retrace</text>
  <circle cx="299" cy="200" r="5" fill="#ffd700"/>
  <text x="299" y="278" fill="#ffd700" font-size="10" text-anchor="middle" font-family="sans-serif">Entry fills</text>

  <!-- Candles moving up to TP -->
  <rect x="324" y="145" width="30" height="65" fill="#64ffda" opacity="0.6" rx="1"/>
  <rect x="364" y="120" width="30" height="78" fill="#64ffda" opacity="0.7" rx="1"/>
  <rect x="404" y="95"  width="30" height="88" fill="#64ffda" opacity="0.8" rx="1"/>
  <circle cx="419" cy="80" r="6" fill="#64ffda"/>
  <text x="419" y="68" fill="#64ffda" font-size="11" text-anchor="middle" font-family="sans-serif" font-weight="bold">TP Hit → WIN ✅</text>

  <!-- R annotations on left axis -->
  <line x1="75" y1="200" x2="75" y2="235" stroke="#ff6b6b" stroke-width="2"/>
  <text x="65" y="221" fill="#ff6b6b" font-size="10" text-anchor="end" font-family="sans-serif">1R</text>
  <line x1="75" y1="80"  x2="75" y2="200" stroke="#64ffda" stroke-width="2"/>
  <text x="65" y="143" fill="#64ffda" font-size="10" text-anchor="end" font-family="sans-serif">3R</text>
</svg>
</div>
""", height=310)

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px;">
<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:14px 16px;">
<b style="color:#58a6ff;">Entry retracement</b><br>
<span style="color:#ccd6f6;font-size:13px;">Wait for price to pull back before entering. At <b>23.6% Fib</b>:<br>
• Trigger body = $1,000 ($50,000 → $51,000)<br>
• 23.6% of $1,000 = $236<br>
• Entry = $51,000 − $236 = <b style="color:#ffd700;">$50,764</b><br>
Better price, tighter stop than entering at close.</span>
</div>
<div style="background:#0d1f2d;border:1px solid #1f6feb;border-radius:8px;padding:14px 16px;">
<b style="color:#58a6ff;">R = your risk unit</b><br>
<span style="color:#ccd6f6;font-size:13px;">Entry $50,764 / Stop $49,749 (2% below):<br>
• 1R = $50,764 − $49,749 = $1,015<br>
• 2R target = $52,794 &nbsp; 3R target = $53,809<br>
Risk $100 per trade → win 3R = <b style="color:#64ffda;">+$300</b>. Lose = <b style="color:#ff6b6b;">−$100</b>.</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ─── Section 4: Risk Math ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Why Low Win Rate Still Makes Money")

    # FIX 4: uniform box sizes, last box red, proper spacing
    st.components.v1.html("""
<style>body{margin:0;padding:0;background:#0d1117;}</style>
<div style="max-width:720px;margin:0 auto;">
<svg viewBox="0 0 740 195" xmlns="http://www.w3.org/2000/svg"
     style="background:#0d1117;border-radius:8px;width:100%;display:block;">
  <text x="370" y="22" fill="#ccd6f6" font-size="13" text-anchor="middle"
        font-family="sans-serif" font-weight="bold">
    10 trades — 3R target, 2% risk, $1,000 account — Win rate 30%
  </text>

  <!-- 7 LOSS boxes — all same size 60×60, spaced evenly -->
  <rect x="20"  y="38" width="60" height="60" fill="#ff6b6b" opacity="0.82" rx="5"/>
  <text x="50"  y="66" fill="white" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">LOSS</text>
  <text x="50"  y="82" fill="white" font-size="11" text-anchor="middle" font-family="sans-serif">−$20</text>

  <rect x="90"  y="38" width="60" height="60" fill="#ff6b6b" opacity="0.82" rx="5"/>
  <text x="120" y="66" fill="white" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">LOSS</text>
  <text x="120" y="82" fill="white" font-size="11" text-anchor="middle" font-family="sans-serif">−$20</text>

  <rect x="160" y="38" width="60" height="60" fill="#ff6b6b" opacity="0.82" rx="5"/>
  <text x="190" y="66" fill="white" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">LOSS</text>
  <text x="190" y="82" fill="white" font-size="11" text-anchor="middle" font-family="sans-serif">−$20</text>

  <rect x="230" y="38" width="60" height="60" fill="#ff6b6b" opacity="0.82" rx="5"/>
  <text x="260" y="66" fill="white" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">LOSS</text>
  <text x="260" y="82" fill="white" font-size="11" text-anchor="middle" font-family="sans-serif">−$20</text>

  <rect x="300" y="38" width="60" height="60" fill="#ff6b6b" opacity="0.82" rx="5"/>
  <text x="330" y="66" fill="white" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">LOSS</text>
  <text x="330" y="82" fill="white" font-size="11" text-anchor="middle" font-family="sans-serif">−$20</text>

  <rect x="370" y="38" width="60" height="60" fill="#ff6b6b" opacity="0.82" rx="5"/>
  <text x="400" y="66" fill="white" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">LOSS</text>
  <text x="400" y="82" fill="white" font-size="11" text-anchor="middle" font-family="sans-serif">−$20</text>

  <rect x="440" y="38" width="60" height="60" fill="#ff6b6b" opacity="0.82" rx="5"/>
  <text x="470" y="66" fill="white" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">LOSS</text>
  <text x="470" y="82" fill="white" font-size="11" text-anchor="middle" font-family="sans-serif">−$20</text>

  <!-- 3 WIN boxes — same size 60×60 -->
  <rect x="510" y="38" width="60" height="60" fill="#64ffda" opacity="0.88" rx="5"/>
  <text x="540" y="64" fill="#0d1117" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">WIN</text>
  <text x="540" y="80" fill="#0d1117" font-size="11" text-anchor="middle" font-family="sans-serif">+$60</text>
  <text x="540" y="111" fill="#64ffda" font-size="10" text-anchor="middle" font-family="sans-serif">3R</text>

  <rect x="580" y="38" width="60" height="60" fill="#64ffda" opacity="0.88" rx="5"/>
  <text x="610" y="64" fill="#0d1117" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">WIN</text>
  <text x="610" y="80" fill="#0d1117" font-size="11" text-anchor="middle" font-family="sans-serif">+$60</text>

  <rect x="650" y="38" width="60" height="60" fill="#64ffda" opacity="0.88" rx="5"/>
  <text x="680" y="64" fill="#0d1117" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">WIN</text>
  <text x="680" y="80" fill="#0d1117" font-size="11" text-anchor="middle" font-family="sans-serif">+$60</text>

  <!-- Summary bar -->
  <rect x="20" y="120" width="700" height="60" fill="#1e2130" rx="6"/>
  <text x="370" y="145" fill="#ccd6f6" font-size="12" text-anchor="middle" font-family="sans-serif">
    7 losses × $20 = −$140   |   3 wins × $60 = +$180
  </text>
  <text x="370" y="168" fill="#64ffda" font-size="14" text-anchor="middle"
        font-family="sans-serif" font-weight="bold">
    NET = +$40 profit — even at only 30% win rate
  </text>
</svg>
</div>
""", height=205)

    st.markdown("""
> **The key insight:** you don't need to win most of the time. You just need your wins to be
> bigger than your losses. At **3R target with 2% risk**, winning just **1 in 4 trades** breaks even.
> Win 35%+ and you're consistently profitable.
""")

    # ─── Section 5: ADX ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📡 The ADX Filter — Optional But Powerful")
    st.markdown("""
**ADX** tells you whether the market is **trending** or **ranging**.
Momentum candles work far better in trending markets.
""")

    # FIX 5: two side-by-side panels, text never overlaps chart, plenty of padding
    st.components.v1.html("""
<style>body{margin:0;padding:0;background:#0d1117;}</style>
<div style="max-width:740px;margin:0 auto;">
<svg viewBox="0 0 760 260" xmlns="http://www.w3.org/2000/svg"
     style="background:#0d1117;border-radius:8px;width:100%;display:block;">

  <!-- ── LEFT PANEL: Ranging ── -->
  <rect x="10" y="10" width="355" height="240" fill="#1a0d0d" rx="6" stroke="#ff6b6b" stroke-width="1" opacity="0.6"/>
  <text x="187" y="32" fill="#ff6b6b" font-size="13" text-anchor="middle" font-family="sans-serif" font-weight="bold">
    Ranging Market  (ADX &lt; 25)
  </text>
  <text x="187" y="50" fill="#8892b0" font-size="10" text-anchor="middle" font-family="sans-serif">
    Signal fires but price chops sideways
  </text>
  <!-- choppy line -->
  <polyline points="30,155 55,125 80,165 105,118 130,158 155,112 180,150 205,108 230,148 255,118 290,152 325,120"
            fill="none" stroke="#ff6b6b" stroke-width="2"/>
  <line x1="30" y1="135" x2="325" y2="135" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4" opacity="0.3"/>
  <!-- signal candle -->
  <rect x="97" y="107" width="16" height="36" fill="#64ffda" opacity="0.7" rx="1"/>
  <text x="105" y="100" fill="#ffd700" font-size="10" text-anchor="middle" font-family="sans-serif">signal</text>
  <!-- result -->
  <text x="187" y="178" fill="#ff6b6b" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">
    ❌ Price reverses — false signal
  </text>
  <!-- ADX bar -->
  <rect x="30" y="192" width="295" height="16" fill="#2d1a1a" rx="3"/>
  <rect x="30" y="192" width="74"  height="16" fill="#ff6b6b" opacity="0.7" rx="3"/>
  <text x="187" y="204" fill="#8892b0" font-size="10" text-anchor="middle" font-family="sans-serif">ADX ≈ 15 (ranging)</text>
  <text x="187" y="228" fill="#ff6b6b" font-size="11" text-anchor="middle" font-family="sans-serif">
    Strategy underperforms here
  </text>

  <!-- ── RIGHT PANEL: Trending ── -->
  <rect x="395" y="10" width="355" height="240" fill="#0d1a0d" rx="6" stroke="#64ffda" stroke-width="1" opacity="0.6"/>
  <text x="572" y="32" fill="#64ffda" font-size="13" text-anchor="middle" font-family="sans-serif" font-weight="bold">
    Trending Market  (ADX ≥ 25)
  </text>
  <text x="572" y="50" fill="#8892b0" font-size="10" text-anchor="middle" font-family="sans-serif">
    Signal fires — price follows through
  </text>
  <!-- trending line -->
  <polyline points="415,195 445,178 475,160 510,138 545,115 580,92 615,70 650,50 690,32 730,18"
            fill="none" stroke="#64ffda" stroke-width="2.5"/>
  <!-- signal candle -->
  <rect x="500" y="126" width="16" height="30" fill="#64ffda" opacity="0.9" rx="1"/>
  <text x="508" y="120" fill="#ffd700" font-size="10" text-anchor="middle" font-family="sans-serif">signal</text>
  <!-- arrow to TP -->
  <line x1="516" y1="118" x2="645" y2="52" stroke="#64ffda" stroke-width="1.5" stroke-dasharray="5"/>
  <text x="572" y="178" fill="#64ffda" font-size="12" text-anchor="middle" font-family="sans-serif" font-weight="bold">
    ✅ Price keeps moving — clean win
  </text>
  <!-- ADX bar -->
  <rect x="415" y="192" width="295" height="16" fill="#0d2d0d" rx="3"/>
  <rect x="415" y="192" width="207" height="16" fill="#64ffda" opacity="0.7" rx="3"/>
  <text x="572" y="204" fill="#64ffda" font-size="10" text-anchor="middle" font-family="sans-serif">ADX ≈ 32 (trending)</text>
  <text x="572" y="228" fill="#64ffda" font-size="11" text-anchor="middle" font-family="sans-serif">
    Strategy excels here
  </text>
</svg>
</div>
""", height=285)

    # ─── Section 6: Full workflow ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Full Workflow — Step by Step")

    steps = [
        ("1️⃣", "Set your ticker & timeframe", "Enter a crypto pair (e.g. BTC, ETH) and choose 1D, 4H, or 1H. Daily is most reliable for beginners."),
        ("2️⃣", "Click Run Analysis", "The app backtests ALL combinations of entry retrace, stop loss, take profit, and ADX filter — and finds the best one."),
        ("3️⃣", "Read the Best Setup", "The app shows the single best parameter combo based on historical Profit Factor. This is your trading blueprint."),
        ("4️⃣", "Check Hold Time Analysis", "Tells you whether to hold until TP/SL or cut trades early at a specific candle count. Cutting early sometimes helps."),
        ("5️⃣", "Monitor Live Scanner", "Checks the last 20 candles for fresh qualifying signals with entry, SL, and TP levels pre-calculated."),
        ("6️⃣", "Use Auto Analyzer", "Run the full sweep to discover if Short mode, a different timeframe, or ADX filtering improves results further."),
    ]

    for emoji, title, desc in steps:
        st.markdown(f"""<div style="display:flex;align-items:flex-start;padding:12px 0;border-bottom:1px solid #21262d;">
<span style="font-size:24px;margin-right:14px;flex-shrink:0;">{emoji}</span>
<div>
<div style="color:#ccd6f6;font-weight:700;font-size:15px;margin-bottom:4px;">{title}</div>
<div style="color:#8892b0;font-size:13px;">{desc}</div>
</div>
</div>""", unsafe_allow_html=True)

    # FIX 6: metric cards — fixed height with min-height so all equal
    st.markdown("---")
    st.markdown("### 📊 Understanding the Numbers")
    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:12px 0;">

<div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;padding:16px;min-height:130px;box-sizing:border-box;">
<div style="color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Profit Factor</div>
<div style="color:#64ffda;font-size:13px;margin-top:10px;">Total wins ÷ Total losses.<br><b>&gt; 1.0</b> = profitable.<br><b>1.5</b> = good. <b>2.0+</b> = excellent.</div>
</div>

<div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;padding:16px;min-height:130px;box-sizing:border-box;">
<div style="color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Win Rate</div>
<div style="color:#ffd700;font-size:13px;margin-top:10px;">% of trades that hit TP.<br>Low win rate (35%) is fine if your reward is 3× your risk.</div>
</div>

<div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;padding:16px;min-height:130px;box-sizing:border-box;">
<div style="color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Avg R</div>
<div style="color:#7c83fd;font-size:13px;margin-top:10px;">Average profit per trade in R units.<br>Positive = strategy makes money over time.</div>
</div>

<div style="background:#1e2130;border:1px solid #2d3250;border-radius:8px;padding:16px;min-height:130px;box-sizing:border-box;">
<div style="color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Edge Score</div>
<div style="color:#ff8c42;font-size:13px;margin-top:10px;">Return vs volatility.<br><b>&gt; 1.0</b> = good risk-adjusted returns.<br><b>&gt; 2.0</b> = excellent.</div>
</div>

</div>
""", unsafe_allow_html=True)

    # ─── Section 8: Quick tips ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Tips for Beginners")
    st.markdown("""
<div class="info-box">
<h4>Do's ✅</h4>
<ul style="color:#ccd6f6;margin:0;padding-left:20px;">
<li>Start with <b>Daily (1D)</b> timeframe — fewer signals but more reliable</li>
<li>Use at least <b>1 year of data</b> before trusting results</li>
<li>Aim for <b>≥ 20 qualifying signals</b> in your backtest period</li>
<li>Prefer setups with <b>Profit Factor ≥ 1.3</b> and <b>≥ 15 trades</b></li>
<li>Use <b>Hold Time Analysis</b> — cutting at 3–5 candles sometimes beats holding forever</li>
<li>Paper trade (demo) first before using real money</li>
</ul>
</div>

<div class="info-box" style="margin-top:12px;">
<h4>Don'ts ❌</h4>
<ul style="color:#ccd6f6;margin:0;padding-left:20px;">
<li>Don't use 1H with only 3 months of data — too few signals to trust</li>
<li>Don't cherry-pick results — use what the optimizer recommends</li>
<li>Don't over-optimize — PF of 99 with only 3 trades is meaningless</li>
<li>Don't ignore drawdown — losing 50% before recovering is not practical</li>
<li>Don't expect every trade to win — 35% win rate is profitable with 3R target</li>
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

def render_auto_analyzer(ticker: str, df_full_1d: pd.DataFrame, tc: float,
                          current_tf: str):
    """Sweep ALL parameter combinations across all timeframes and rank by score."""

    st.markdown("### 🤖 Auto Analyzer — Full Parameter Sweep")
    st.markdown("""<div class="info-box">
<h4>How it works</h4>
<p>
Sweeps <b>every combination</b> of:
Direction × Timeframe × Body % × Volume × ADX/DI config × Entry retrace × SL × TP × Hold mode.
</p>
<p>
<b>Score = Profit Factor × √(Trades) × Win Rate</b> — rewards both profitability AND frequency.<br>
Only strategies with <b>PF &gt; 1.0</b> and <b>≥ 8 trades</b> are shown.
Top 5 ranked by score.
</p>
<p style="color:#8892b0;font-size:12px;">⚠️ May take 2–5 minutes depending on period selected.</p>
</div>""", unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        _aa_period_opts = {
            "Last 3 months":  90,
            "Last 6 months": 180,
            "Last 1 year":   365,
            "Last 2 years":  730,
            "Last 3 years": 1095,
            "Last 5 years": 1825,
        }
        _aa_period_label = st.select_slider(
            "Analysis period", options=list(_aa_period_opts.keys()),
            value="Last 1 year", key="aa_period")

    with ctrl2:
        _aa_tf_opts = ["1D", "4H", "1H", "All timeframes"]
        _aa_tf = st.selectbox("Candle timeframe", _aa_tf_opts,
                              index=_aa_tf_opts.index(current_tf)
                              if current_tf in _aa_tf_opts else 0,
                              key="aa_tf")

    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_aa = st.button("🚀 Run", key="aa_run",
                           use_container_width=True, type="primary")

    # ── Speed / depth toggles ─────────────────────────────────────────────────
    st.markdown("**Sweep depth** — uncheck to go faster:")
    sp1, sp2 = st.columns(2)
    with sp1:
        sweep_adx  = st.checkbox("Include ADX / DI configs", value=True, key="aa_adx",
                                  help="7 ADX configs. Uncheck = No ADX filter only (~7× faster)")
        sweep_hold = st.checkbox("Include hold time variants", value=True, key="aa_hold",
                                  help="5 hold modes. Uncheck = No-limit only (~5× faster)")
    with sp2:
        sweep_dir  = st.checkbox("Include both Long & Short", value=True, key="aa_dir",
                                  help="Uncheck to test Long only (~2× faster)")
        sweep_body = st.checkbox("Include multiple Body % thresholds", value=True, key="aa_body",
                                  help="Uncheck = 70% only (~4× faster)")

    # Estimate and display combo count
    _n_dirs  = 2 if sweep_dir  else 1
    _n_body  = 4 if sweep_body else 1
    _n_adx   = 7 if sweep_adx  else 1
    _n_hold  = 5 if sweep_hold else 1
    _n_inner = len(RETRACEMENTS) * len(SL_DISTANCES) * len(TP_MODES)  # 10×4×3 = 120
    _est     = _n_dirs * _n_body * 4 * _n_adx * _n_hold * _n_inner
    _spd_label = "🐢 Full sweep" if _est > 50_000 else "🐇 Fast" if _est < 10_000 else "⚡ Medium"
    st.caption(f"{_spd_label} — estimated **{_est:,}** strategy runs "
               f"(≈ {max(1, _est // 8000)} min on average hardware)")

    _aa_days = _aa_period_opts[_aa_period_label]
    aa_key   = f"aa_{ticker}_{_aa_period_label}_{_aa_tf}_{sweep_adx}{sweep_hold}{sweep_dir}{sweep_body}"

    # Always show cached results if available — only re-run on button click
    _aa_has_results = "aa_results" in st.session_state
    _aa_key_changed = st.session_state.get("aa_key") != aa_key

    if _aa_has_results and _aa_key_changed:
        st.warning("⚠️ Settings changed since last run — click **🚀 Run** to refresh results.")

    if not run_aa:
        if _aa_has_results:
            _render_aa_results(st.session_state["aa_results"], ticker,
                               st.session_state.get("aa_period_label", _aa_period_label),
                               st.session_state.get("aa_tf_label", _aa_tf))
        else:
            st.info("👆 Configure settings then click **🚀 Run**.")
        return

    # ── Fetch data per timeframe ──────────────────────────────────────────────
    _tf_list = ["1D", "4H", "1H"] if _aa_tf == "All timeframes" else [_aa_tf]
    _tf_dfs  = {}
    for _tf in _tf_list:
        with st.spinner(f"Fetching {ticker} ({_tf})…"):
            _df = _binance_fetch(ticker, _tf, _aa_days)
        if not _df.empty:
            _aa_cutoff = _df.index[-1] - pd.Timedelta(days=_aa_days)
            _tf_dfs[_tf] = _df[_df.index >= _aa_cutoff].copy()

    if not _tf_dfs:
        st.error("No data available for selected timeframe(s).")
        return

    # ── Build parameter grid based on toggles ────────────────────────────────
    _aa_directions = ["long", "short"] if sweep_dir else ["long"]
    _aa_body_pcts  = [0.60, 0.70, 0.80, 0.90] if sweep_body else [0.70]
    _aa_vol_mults  = [1.5, 2.0, 2.5, 3.0]

    _aa_adx_configs = [{"on": False, "threshold": 25, "gap": 0.0, "label": "No ADX filter"}]
    if sweep_adx:
        _aa_adx_configs += [
            {"on": True,  "threshold": 20, "gap": 0.0,  "label": "ADX ≥ 20"},
            {"on": True,  "threshold": 25, "gap": 0.0,  "label": "ADX ≥ 25"},
            {"on": True,  "threshold": 25, "gap": 10.0, "label": "ADX ≥ 25 + DI gap ≥ 10"},
            {"on": True,  "threshold": 25, "gap": 20.0, "label": "ADX ≥ 25 + DI gap ≥ 20"},
            {"on": True,  "threshold": 30, "gap": 0.0,  "label": "ADX ≥ 30"},
            {"on": True,  "threshold": 30, "gap": 10.0, "label": "ADX ≥ 30 + DI gap ≥ 10"},
        ]

    _aa_hold_modes = [{"bars": 9999, "time_exit": False, "label": "No limit (TP/SL only)"}]
    if sweep_hold:
        _aa_hold_modes += [
            {"bars": 3,  "time_exit": True, "label": "Max 3 candles"},
            {"bars": 5,  "time_exit": True, "label": "Max 5 candles"},
            {"bars": 7,  "time_exit": True, "label": "Max 7 candles"},
            {"bars": 10, "time_exit": True, "label": "Max 10 candles"},
        ]
    _aa_inner = list(product(RETRACEMENTS, SL_DISTANCES, TP_MODES))

    outer_combos = list(product(
        _tf_list, _aa_directions, _aa_body_pcts,
        _aa_vol_mults, _aa_adx_configs, _aa_hold_modes
    ))
    total_outer = len(outer_combos)

    st.info(f"Testing {total_outer:,} signal configs × {len(_aa_inner)} parameter combos "
            f"= up to {total_outer * len(_aa_inner):,} strategy runs…")

    prog    = st.progress(0.0, text="Starting sweep…")
    results = []
    # Cache ADX per TF to avoid recomputing
    _adx_cache: dict = {}

    for i, (tf, dirn, body_pct, vol_mult, adx_cfg, hold_cfg) in enumerate(outer_combos):
        df_aa = _tf_dfs.get(tf)
        if df_aa is None or df_aa.empty:
            prog.progress((i + 1) / total_outer)
            continue

        if tf not in _adx_cache:
            _adx_cache[tf] = calculate_adx(df_aa)
        _adx_df  = _adx_cache[tf]
        _adx_ser = _adx_df["adx"]
        _dip_ser = _adx_df["di_plus"]
        _dim_ser = _adx_df["di_minus"]

        qual = detect_qualifying_candles(
            df_aa, body_pct, vol_mult,
            adx_filter      = adx_cfg["on"],
            adx_threshold   = adx_cfg["threshold"],
            adx_series      = _adx_ser if adx_cfg["on"] else None,
            di_plus_series  = _dip_ser if adx_cfg["on"] else None,
            di_minus_series = _dim_ser if adx_cfg["on"] else None,
            di_gap_min      = adx_cfg["gap"],
            direction       = dirn,
        )

        if len(qual) < 5:
            prog.progress((i + 1) / total_outer,
                          text=f"{i+1}/{total_outer} — {tf} {dirn.upper()} skipped (< 5 signals)")
            continue

        for ret, sl_d, tp in _aa_inner:
            trades = run_backtest(
                df_aa, qual, ret, sl_d, tp,
                sl_mode="Fixed", trail_atr_mult=1.5,
                transaction_cost=tc,
                max_hold_bars=hold_cfg["bars"],
                use_time_exit=hold_cfg["time_exit"],
                direction=dirn,
            )
            m = calc_metrics(trades)
            if not m or m["total_trades"] < 8:
                continue

            pf = m["profit_factor"]
            if np.isinf(pf):
                pf = 99.0

            # Only include profitable strategies
            if pf <= 1.0:
                continue

            score = pf * np.sqrt(m["total_trades"]) * m["win_rate"]

            results.append({
                "score":         round(score, 3),
                "timeframe":     tf,
                "direction":     dirn.upper(),
                "body_pct":      int(body_pct * 100),
                "vol_mult":      vol_mult,
                "adx_label":     adx_cfg["label"],
                "hold_label":    hold_cfg["label"],
                "retracement":   RETRACE_LABELS[ret],
                "sl_dist":       SL_LABELS[sl_d],
                "tp_mode":       tp,
                "total_trades":  m["total_trades"],
                "qualifying":    len(qual),
                "win_rate":      round(m["win_rate"] * 100, 1),
                "profit_factor": round(pf, 2),
                "avg_r":         round(m["avg_r"], 3),
                "max_drawdown":  round(m["max_drawdown"] * 100, 1),
                "sharpe":        round(m["sharpe"], 2),
            })

        prog.progress((i + 1) / total_outer,
                      text=f"{i+1}/{total_outer} — {tf} {dirn.upper()} "
                           f"body≥{int(body_pct*100)}% vol≥{vol_mult}× {adx_cfg['label']}")

    prog.empty()

    if not results:
        st.warning("No profitable strategies (PF > 1.0, ≥ 8 trades) found. Try a longer period.")
        return

    results_df = (pd.DataFrame(results)
                  .sort_values("score", ascending=False)
                  .reset_index(drop=True))

    st.session_state["aa_results"]      = results_df
    st.session_state["aa_key"]          = aa_key
    st.session_state["aa_period_label"] = _aa_period_label
    st.session_state["aa_tf_label"]     = _aa_tf

    _render_aa_results(results_df, ticker, _aa_period_label, _aa_tf)


def _render_aa_results(results_df: pd.DataFrame, ticker: str,
                        period_label: str, tf_label: str):
    """Display top-5 results from the auto analyzer using native Streamlit layout."""

    top5         = results_df.head(5).copy()
    total_tested = len(results_df)

    st.success(f"✅ {total_tested:,} profitable strategies found — "
               f"**{ticker} | {tf_label} | {period_label}**")
    st.caption("Score = PF × √(Trades) × Win Rate  |  Only PF > 1.0 and ≥ 8 trades shown")
    st.markdown("---")
    st.markdown("### 🏆 Top 5 Strategies")

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        medal     = medals[rank - 1]
        dir_color = "#ff6b6b" if row["direction"] == "SHORT" else "#64ffda"
        dir_emoji = "📉 SHORT" if row["direction"] == "SHORT" else "📈 LONG"
        pf_color  = "#64ffda" if row["profit_factor"] >= 1.5 else "#ffd700"
        sl_side   = "above" if row["direction"] == "SHORT" else "below"

        with st.container():
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0d1117,#161b22);'
                f'border:1px solid #238636;border-radius:10px;padding:18px 22px;margin-bottom:14px;">'
                f'<div style="font-size:17px;font-weight:700;margin-bottom:12px;">'
                f'{medal} Rank {rank} &nbsp;'
                f'<span style="color:{dir_color};">{dir_emoji}</span> &nbsp;'
                f'<span style="background:#1e2130;border-radius:4px;padding:2px 8px;'
                f'font-size:13px;color:#ccd6f6;">{row["timeframe"]} candles</span> &nbsp;'
                f'<span style="color:#8892b0;font-size:13px;font-weight:400;">'
                f'Score: {row["score"]:.2f}</span></div></div>',
                unsafe_allow_html=True)

            # Two columns: setup | metrics
            lc, rc = st.columns(2)
            with lc:
                st.markdown("**Setup Parameters**")
                st.markdown(f"- Body % min: **{row['body_pct']}%**")
                st.markdown(f"- Vol Multiplier min: **{row['vol_mult']}×**")
                st.markdown(f"- ADX / DI: **{row['adx_label']}**")
                st.markdown(f"- Hold Mode: **{row['hold_label']}**")
                st.markdown(f"- Entry Retrace: **{row['retracement']}**")
                st.markdown(f"- Stop Loss: **{row['sl_dist']} {sl_side} entry**")
                st.markdown(f"- Take Profit: **{row['tp_mode']}**")

                # Use Strategy button
                if st.button(f"⚡ Use this strategy", key=f"use_strat_{rank}",
                             type="primary", use_container_width=True):
                    _apply_strategy_to_sidebar(row.to_dict())
                    st.rerun()
            with rc:
                st.markdown("**Performance**")
                st.markdown(f"- Qualifying signals: **{row['qualifying']}**")
                st.markdown(f"- Trades taken: **{row['total_trades']}**")
                st.markdown(f"- Win Rate: **{row['win_rate']}%**")
                st.markdown(f"- Profit Factor: **{row['profit_factor']}**")
                st.markdown(f"- Avg R per trade: **{row['avg_r']}R**")
                st.markdown(f"- Max Drawdown: **{row['max_drawdown']}%**")
                st.markdown(f"- Edge Score: **{row['sharpe']}**")

            st.markdown("---")

    # Full table
    st.markdown("### Full Rankings (top 50)")
    display = results_df.head(50).rename(columns={
        "score": "Score", "timeframe": "TF", "direction": "Dir",
        "body_pct": "Body%", "vol_mult": "Vol×", "adx_label": "ADX Config",
        "hold_label": "Hold", "retracement": "Retrace", "sl_dist": "SL",
        "tp_mode": "TP", "total_trades": "Trades", "qualifying": "Signals",
        "win_rate": "Win%", "profit_factor": "PF", "avg_r": "Avg R",
        "max_drawdown": "Max DD%", "sharpe": "Edge Score",
    })
    st.dataframe(display, use_container_width=True, height=480,
                 column_config={
                     "Score":  st.column_config.NumberColumn(width=65,  format="%.2f"),
                     "TF":     st.column_config.TextColumn(width=45),
                     "Dir":    st.column_config.TextColumn(width=50),
                     "Body%":  st.column_config.NumberColumn(width=55),
                     "Vol×":   st.column_config.NumberColumn(width=48,  format="%.1f"),
                     "ADX Config": st.column_config.TextColumn(width=175),
                     "Hold":   st.column_config.TextColumn(width=140),
                     "Retrace":st.column_config.TextColumn(width=105),
                     "SL":     st.column_config.TextColumn(width=48),
                     "TP":     st.column_config.TextColumn(width=60),
                     "Trades": st.column_config.NumberColumn(width=55),
                     "Signals":st.column_config.NumberColumn(width=58),
                     "Win%":   st.column_config.NumberColumn(width=55,  format="%.1f"),
                     "PF":     st.column_config.NumberColumn(width=52,  format="%.2f"),
                     "Avg R":  st.column_config.NumberColumn(width=58,  format="%.3f"),
                     "Max DD%":st.column_config.NumberColumn(width=65,  format="%.1f"),
                     "Edge Score": st.column_config.NumberColumn(width=72, format="%.2f"),
                 })

    st.markdown("**Apply any ranked strategy directly:**")
    btn_cols = st.columns(5)
    for idx in range(min(10, len(results_df))):
        row = results_df.iloc[idx]
        col = btn_cols[idx % 5]
        with col:
            lbl = (f"#{idx+1} {row['direction']} {row['timeframe']} "
                   f"PF={row['profit_factor']:.2f}")
            if st.button(lbl, key=f"use_table_{idx}", use_container_width=True):
                _apply_strategy_to_sidebar(row.to_dict())
                st.rerun()

    st.download_button(
        "⬇ Download Full Rankings CSV",
        display.to_csv(index=False).encode("utf-8"),
        f"auto_analyzer_{ticker}.csv", "text/csv",
    )


# ─── Main App ──────────────────────────────────────────────────────────────────

def main():
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

        st.markdown("---")
        run_btn = st.button("Run Analysis", use_container_width=True, type="primary")
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
    run_key = (f"{ticker}_{timeframe}_{history_label}_{min_body_pct:.2f}_"
               f"{min_vol_mult}_{sl_mode}_{trail_mult}_{transaction_cost}_"
               f"{use_time_exit}_{max_hold_bars}_"
               f"adx{adx_filter}_{adx_threshold}_{adx_tf_mode}_gap{di_gap_min}_{direction}_"
               f"sess{_sess_key}")
    _last_key   = st.session_state.get("run_key")
    _has_results = "opt_results" in st.session_state

    if _has_results and _last_key != run_key:
        st.sidebar.warning("⚠️ Settings changed — click **Run Analysis** to update.")

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

        qualifying = detect_qualifying_candles(
            df, min_body_pct, min_vol_mult,
            adx_filter=adx_filter, adx_threshold=adx_threshold,
            adx_series=adx_for_filter,
            di_plus_series=di_plus_signal, di_minus_series=di_minus_signal,
            di_gap_min=di_gap_min, direction=direction,
        )

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
                    "_adv_stats_key","hold_parent_run_key"):
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
            "adx_tf_mode":    adx_tf_mode,
            "use_session":    use_session,
            "allowed_sessions": allowed_sessions,
        })

    # ── Read from session_state (works for both fresh run and cached) ──────────
    if "opt_results" not in st.session_state:
        st.info("Click **Run Analysis** to start.")
        return

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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📖 Strategy Guide",
        "Strategy Finder",
        "Backtest Results",
        "Trade Log",
        "Candle Statistics",
        "Portfolio Simulator",
        "Live Scanner",
        "🤖 Auto Analyzer",
        "📊 Advanced Stats",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 -- Strategy Guide
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        render_strategy_guide()

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 -- Strategy Finder (was tab1)
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
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
            st.markdown(f"""<div style="font-size:36px;font-weight:800;color:{dir_color};margin-bottom:4px;">
{len(qualifying)} qualifying candles
</div>
<div style="color:#8892b0;font-size:15px;">
found in {years:.1f} years of {ticker} data
&nbsp;|&nbsp; {len(qualifying)/max(years,0.1):.1f} per year average
</div>
<div style="color:{dir_color};font-size:13px;margin-top:4px;">{dir_badge}</div>
{adx_badge}{sess_badge}""", unsafe_allow_html=True)

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
            st.markdown(f"""<div class="best-box">
<h3>Best Setup Found</h3>
{dir_row}
<div class="best-row"><span class="best-key">Entry Retracement</span><span class="best-val">{best_label} — {retrace_desc}</span></div>
<div class="best-row"><span class="best-key">Stop Loss</span><span class="best-val">{best_sl} {sl_label}</span></div>
<div class="best-row"><span class="best-key">Take Profit Mode</span><span class="best-val">{best['tp_mode']}</span></div>
<div class="best-row"><span class="best-key">SL Management</span><span class="best-val">{_sl_mode}</span></div>
{adx_row}{_sess_row}
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
    # TAB 2 -- Backtest Results
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
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
    # TAB 3 -- Trade Log
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
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
    # TAB 4 -- Candle Statistics
    # ═══════════════════════════════════════════════════════════════════════
    with tab5:
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
    # TAB 5 -- Portfolio Simulator
    # ═══════════════════════════════════════════════════════════════════════
    with tab6:
        render_portfolio_simulator(best_trades, df, ticker)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 6 -- Live Scanner
    # ═══════════════════════════════════════════════════════════════════════
    with tab7:
        render_live_scanner(
            ticker, timeframe, min_body_pct, min_vol_mult,
            best["retracement"], best["sl_dist"], best["tp_mode"], best_wr,
            direction=_direction,
            adx_filter_on=adx_filter,
            adx_di_gap_min=di_gap_min,
            adx_threshold=adx_threshold,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 7 -- Auto Analyzer
    # ═══════════════════════════════════════════════════════════════════════
    with tab8:
        render_auto_analyzer(ticker, df_full, tc, timeframe)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 9 -- Advanced Statistics
    # ═══════════════════════════════════════════════════════════════════════
    with tab9:
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


if __name__ == "__main__":
    main()
