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
    df["candle_range"] = df["high"] - df["low"]
    df["body_pct"]     = df["body"] / df["candle_range"].replace(0, float("nan"))
    df["vol_avg_7"]    = df["volume"].shift(1).rolling(7).mean()
    df["vol_mult"]     = df["volume"] / df["vol_avg_7"]
    # ATR(14) for trailing stop
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    return df


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


@st.cache_data(ttl=3600, show_spinner=False)
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

    result = df[mask].copy()
    if adx_filter and adx_series is not None:
        adx_aligned_full = adx_series.reindex(df.index, method="ffill")
        result["adx_value"] = adx_aligned_full[mask].round(1)
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
                    r = -(1 + transaction_cost * 2)
                    return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                       entry_date, sl_price, tp1, current_sl,
                                       bar_date, "Loss", r, retracement,
                                       sl_dist, tp_mode, sl_mode,
                                       exit_type="SL", candles_held=0)
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
                r = -(1 + transaction_cost * 2)
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp1, current_sl,
                                   bar_date, "Loss", r, retracement,
                                   sl_dist, tp_mode, sl_mode,
                                   exit_type="SL", candles_held=fwd_i - entry_bar_i)
            if tp_gapped:
                r = (tp1 - entry_price) / risk * (-1 if direction == "short" else 1) - transaction_cost * 2
                r = abs(r)  # TP always positive
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp1, tp1,
                                   bar_date, "Win", r, retracement,
                                   sl_dist, tp_mode, sl_mode,
                                   exit_type="TP", candles_held=fwd_i - entry_bar_i)

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
                               exit_type="Time Exit", candles_held=fwd_i - entry_bar_i)

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
            r      = -(1 + transaction_cost * 2)
            return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                               entry_date, sl_price, tp1, current_sl,
                               bar_date, "Loss", r, retracement,
                               sl_dist, tp_mode, sl_mode,
                               exit_type="SL", candles_held=fwd_i - entry_bar_i)

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
                                   exit_type="TP", candles_held=fwd_i - entry_bar_i)
        else:
            tp1_hit = bar["low"] <= tp1 if direction == "short" else bar["high"] >= tp1
            if tp1_hit:
                r = abs((tp1 - entry_price) / risk) - transaction_cost * 2
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp1, tp1,
                                   bar_date, "Win", r, retracement,
                                   sl_dist, tp_mode, sl_mode,
                                   exit_type="TP", candles_held=fwd_i - entry_bar_i)

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
                       exit_type=exit_type, candles_held=candles_held)


def _trade_dict(trigger, trigger_date, entry_price, entry_date,
                sl_price, tp_price, exit_price, exit_date,
                result, r_mult, retracement, sl_dist, tp_mode, sl_mode,
                exit_type: str = "Max Hold", candles_held: int = 0) -> Dict:
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
        "adx_value":        round(float(trigger.get("adx_value", float("nan"))), 1)
                            if "adx_value" in trigger.index else float("nan"),
    }


def run_backtest(df, qualifying, retracement, sl_dist, tp_mode,
                 sl_mode="Fixed", trail_atr_mult=1.5,
                 transaction_cost=0.001,
                 max_hold_bars=5, use_time_exit=False,
                 direction="long") -> List[Dict]:
    trades   = []
    df_reset = df.reset_index()
    q_dates  = set(qualifying.index)
    for date in q_dates:
        loc = df_reset[df_reset.iloc[:, 0] == date]
        if loc.empty:
            continue
        idx   = loc.index[0]
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
    streak = max_s = 0
    for v in r:
        streak = streak + 1 if v <= 0 else 0
        max_s  = max(max_s, streak)
    holds      = [t["hold_days"] for t in trades if t["hold_days"] > 0]
    time_exits = sum(1 for t in trades if t.get("exit_type") == "Time Exit")
    return {
        "total_trades": total, "win_rate": len(wins) / total,
        "avg_r": r.mean(), "profit_factor": pf,
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
        if m and m["total_trades"] >= 3:
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
    r  = np.array([t["r_mult"] for t in trades])
    eq = np.cumprod(1 + r * 0.02) * 10000
    dt = [t["exit_date"] for t in trades]
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
    dr    = df.reset_index()
    for date in qualifying.index:
        loc = dr[dr.iloc[:, 0] == date]
        if loc.empty:
            continue
        idx     = loc.index[0]
        trigger = df.iloc[idx]
        future  = df.iloc[idx + 1: min(idx + 6, len(df))]
        if future.empty:
            continue
        body_abs = abs(float(trigger["body"]))
        total += 1
        for r in RETRACEMENTS:
            if direction == "short":
                # For short: does price bounce UP to the retrace level?
                level = float(trigger["close"]) + body_abs * r
                if future["high"].max() >= level:
                    hits[r] += 1
            else:
                # For long: does price dip DOWN to the retrace level?
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
        ("Total Trades",          str(m["total_trades"]),           ""),
        ("Win Rate",              f"{m['win_rate']*100:.1f}%",      "green" if m["win_rate"] >= 0.5 else "red"),
        ("Avg R Multiple",        f"{m['avg_r']:.3f}R",             "green" if m["avg_r"] > 0 else "red"),
        ("Profit Factor",         pf_str,                           "green" if pf >= 1.5 else ("red" if pf < 1 else "")),
        ("Max Drawdown",          f"{m['max_drawdown']*100:.1f}%",  "red" if m["max_drawdown"] < -0.15 else ""),
        ("Sharpe Ratio",          f"{m['sharpe']:.2f}",             "green" if m["sharpe"] > 1 else ""),
        ("Longest Losing Streak", str(m["longest_losing_streak"]),  ""),
        ("Avg Hold Time",         f"{m['avg_hold_days']:.1f} days", ""),
    ]
    gc = st.columns(cols)
    for i, (lbl, val, col) in enumerate(cards):
        with gc[i % cols]:
            st.markdown(metric_card(lbl, val, col), unsafe_allow_html=True)

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

    for t in trades:
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

    # Cache per unique analysis params to avoid re-running on every interaction
    hold_key = (f"hold_{ticker}_{best_ret}_{best_sl_dist}_{best_tp}_"
                f"{sl_mode}_{trail_mult}_{tc}_{direction}")
    if st.session_state.get("hold_key") != hold_key:
        with st.spinner("Testing hold times 1–20 candles..."):
            hold_df = analyze_hold_times(df, qualifying, best_ret, best_sl_dist,
                                         best_tp, sl_mode, trail_mult, tc, direction=direction)
        st.session_state["hold_df"]  = hold_df
        st.session_state["hold_key"] = hold_key

    hold_df = st.session_state.get("hold_df", pd.DataFrame())
    if hold_df.empty:
        st.info("Not enough trades for hold time analysis.")
        return

    # ── Line chart ───────────────────────────────────────────────────────────
    st.plotly_chart(plot_hold_time_chart(hold_df), use_container_width=True)

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

    # ── Resolution chart ─────────────────────────────────────────────────────
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
This chart runs with <em>no time cap</em> — every trade is allowed to run until it genuinely
hits TP or SL (up to 500 candles). "Still Open" at candle 20 means the trade truly hasn't
resolved yet, not that it was artificially cut off. This ensures the TP/SL distribution
is accurate and consistent with the "no time limit" Profit Factor baseline.
</p>
</div>""", unsafe_allow_html=True)
    with st.spinner("Building resolution chart..."):
        res_fig = plot_candle_resolution(df, qualifying, best_ret, best_sl_dist,
                                         best_tp, sl_mode, trail_mult, tc, direction=direction)
    st.plotly_chart(res_fig, use_container_width=True)


# ─── Feature 3: Live Scanner ───────────────────────────────────────────────────

def render_live_scanner(ticker, timeframe, min_body_pct, min_vol_mult,
                         best_ret, best_sl_dist, best_tp, best_wr,
                         direction="long", adx_filter_on=False, adx_di_gap_min=0.0):
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

        # When ADX filter is ON, signal requires body + volume + DI gap
        # When ADX filter is OFF, signal requires only body + volume
        if adx_filter_on and adx_di_gap_min > 0:
            req_di_gap = di_aligned and di_gap_val >= adx_di_gap_min
            is_signal  = bp_ok and vol_ok and req_di_gap
        else:
            is_signal  = bp_ok and vol_ok

        sig = "SIGNAL" if is_signal else ("partial" if (bp_ok or vol_ok) else "")
        bp_val = abs(row["body_pct"]) * 100 if pd.notna(row["body_pct"]) else 0
        vm_val = row["vol_mult"]       if pd.notna(row["vol_mult"])  else 0

        # ADX values at this candle
        adx_s_val = adx_signal.get(idx, float("nan"))
        adx_h_val = adx_htf.get(idx,    float("nan"))
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

# ─── Main App ──────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("## Momentum Candle")
        st.markdown("---")

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
                          placeholder="BTC, ETH, SOL, MNT, HYPE, BTCUSDT..."))

        timeframe = st.selectbox("Timeframe", ["1D","4H","1H"], index=0)
        if timeframe in ("4H","1H"):
            st.warning("Shorter timeframes = more signals but less reliable. "
                       "Recommend minimum 30 qualifying candles.")

        history_label = st.select_slider(
            "Data history to analyze",
            options=list(HISTORY_OPTIONS.keys()), value="5 years")
        history_days = HISTORY_OPTIONS[history_label]

        st.markdown("---")
        min_body_pct     = st.slider("Min Body %", 50, 95, 70, 5) / 100
        min_vol_mult     = st.slider("Min Volume Multiplier", 1.0, 4.0, 1.5, 0.1)
        transaction_cost = st.number_input(
            "Transaction Cost (%)", 0.0, 1.0, 0.10, 0.01) / 100

        st.markdown("---")
        st.markdown("**Market Regime Filter (ADX)**")
        adx_filter     = st.toggle("Enable ADX Filter", value=False, key="adx_toggle")
        adx_threshold  = 25
        adx_tf_mode    = "Same as signal timeframe"
        di_gap_min     = 0.0
        if adx_filter:
            adx_threshold = st.slider("ADX Threshold", 15, 40, 25, 1,
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
                              index=0)
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
        run_btn = st.button("Run Analysis", use_container_width=True, type="primary")
        st.caption("Data: Binance (data-api.binance.vision) | Risk model: 2% fixed")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not ticker:
        st.info("Enter a ticker symbol in the sidebar.")
        return

    # ── Fetch data from Binance ────────────────────────────────────────────────
    with st.spinner(f"Fetching {ticker} ({timeframe}) from Binance..."):
        df_full = _binance_fetch(ticker, timeframe, history_days)

    if df_full.empty:
        st.error(
            f"❌ No data found for **{ticker}** on Binance.\n\n"
            "Enter a valid Binance spot symbol: `BTC`, `ETH`, `SOL`, `MNT`, `HYPE`, `BTCUSDT`...")
        return

    # Data source badge
    st.sidebar.markdown(
        f'<span style="color:#3fb950;font-size:12px;">&#x2705; Data source: Binance ({len(df_full)} candles)</span>',
        unsafe_allow_html=True)

    df = trim_by_days(df_full, history_days)
    if df.empty:
        st.error("No data in selected time range.")
        return

    start_date = df.index[0].date()
    st.sidebar.caption(f"Using data from {start_date} to {df.index[-1].date()}")

    years      = (df.index[-1] - df.index[0]).days / 365

    # ── ADX computation ──────────────────────────────────────────────────────
    _HIGHER_TF = {"1H": "4H", "4H": "1D", "1D": "1D"}  # 1D has no higher
    adx_df_signal  = calculate_adx(df) if adx_filter else None
    adx_series_signal  = adx_df_signal["adx"]      if adx_df_signal is not None else None
    di_plus_signal     = adx_df_signal["di_plus"]   if adx_df_signal is not None else None
    di_minus_signal    = adx_df_signal["di_minus"]  if adx_df_signal is not None else None
    adx_series_higher  = None

    if adx_filter and adx_tf_mode in ("One timeframe higher", "Both"):
        higher_tf = _HIGHER_TF.get(timeframe, "1D")
        if higher_tf != timeframe:
            with st.spinner(f"Fetching {higher_tf} data for ADX filter..."):
                df_higher = _binance_fetch(ticker, higher_tf, history_days)
            if not df_higher.empty:
                adx_series_higher = calculate_adx(df_higher)["adx"]
        else:
            st.sidebar.caption("⚠️ Already on highest timeframe (1D) — ADX uses same TF.")
            adx_series_higher = adx_series_signal

    # Pick which ADX series to use for filtering
    if not adx_filter:
        adx_for_filter = None
    elif adx_tf_mode == "Same as signal timeframe":
        adx_for_filter = adx_series_signal
    elif adx_tf_mode == "One timeframe higher":
        adx_for_filter = adx_series_higher if adx_series_higher is not None else adx_series_signal
    else:  # Both — combine: require both >= threshold → use min of the two aligned
        if adx_series_higher is not None:
            sig_aligned   = adx_series_signal.reindex(df.index, method="ffill")
            high_aligned  = adx_series_higher.reindex(df.index, method="ffill")
            adx_for_filter = pd.concat([sig_aligned, high_aligned], axis=1).min(axis=1)
            adx_for_filter.index = df.index
        else:
            adx_for_filter = adx_series_signal

    qualifying = detect_qualifying_candles(
        df, min_body_pct, min_vol_mult,
        adx_filter=adx_filter,
        adx_threshold=adx_threshold,
        adx_series=adx_for_filter,
        di_plus_series=di_plus_signal,
        di_minus_series=di_minus_signal,
        di_gap_min=di_gap_min,
        direction=direction,
    )

    if len(qualifying) < 20:
        st.warning(f"Only {len(qualifying)} qualifying candles found -- "
                   "increase timespan for more reliable results.")

    dir_emoji = "📉 SHORT" if direction == "short" else "📈 LONG"
    st.title(f"{ticker} ({timeframe}) — Momentum Candle Strategy [{dir_emoji}]")
    st.caption(f"Data: {start_date} to {df.index[-1].date()} "
               f"({years:.1f} years) | {len(df)} bars | {len(qualifying)} qualifying triggers")

    # ── Run optimization ─────────────────────────────────────────────────────
    run_key = (f"{ticker}_{timeframe}_{history_label}_{min_body_pct:.2f}_"
               f"{min_vol_mult}_{sl_mode}_{trail_mult}_{transaction_cost}_"
               f"{use_time_exit}_{max_hold_bars}_"
               f"adx{adx_filter}_{adx_threshold}_{adx_tf_mode}_gap{di_gap_min}_{direction}")
    if ("opt_results" not in st.session_state or run_btn or
            st.session_state.get("run_key") != run_key):
        if len(qualifying) < 3:
            st.warning("Too few qualifying candles -- relax the criteria.")
            return
        try:
            opt_df, all_trades = optimize(df, qualifying, sl_mode, trail_mult,
                                          transaction_cost, max_hold_bars, use_time_exit,
                                          direction=direction)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)
            return
        st.session_state.update({
            "opt_results": opt_df, "ticker": ticker, "df": df,
            "qualifying": qualifying, "all_trades": all_trades,
            "transaction_cost": transaction_cost, "sl_mode": sl_mode,
            "trail_mult": trail_mult, "run_key": run_key, "years": years,
            "max_hold_bars": max_hold_bars, "use_time_exit": use_time_exit,
            "direction": direction,
        })

    if "opt_results" not in st.session_state:
        st.info("Click **Run Analysis** to start.")
        return

    opt_df        = st.session_state["opt_results"]
    df            = st.session_state["df"]
    qualifying    = st.session_state["qualifying"]
    all_trades    = st.session_state["all_trades"]
    tc            = st.session_state["transaction_cost"]
    _sl_mode      = st.session_state["sl_mode"]
    _trail        = st.session_state["trail_mult"]
    years         = st.session_state["years"]
    _max_hold     = st.session_state.get("max_hold_bars", 5)
    _time_exit    = st.session_state.get("use_time_exit", False)
    _direction    = st.session_state.get("direction", "long")

    if opt_df.empty:
        st.warning("No parameter combination produced >=3 trades. Try different criteria.")
        return

    best        = opt_df.iloc[0]
    # Re-run best combo with the sidebar's actual time-exit settings.
    # The all_trades dict from optimize() was built with use_time_exit=False /
    # unlimited hold, so we can't reuse it directly when the user has a time exit on.
    best_trades = run_backtest(
        df, qualifying,
        best["retracement"], best["sl_dist"], best["tp_mode"],
        _sl_mode, _trail, tc, _max_hold, _time_exit,
        direction=_direction)
    best_m      = calc_metrics(best_trades)
    best_wr     = best["win_rate"]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Strategy Finder",
        "Backtest Results",
        "Trade Log",
        "Candle Statistics",
        "Portfolio Simulator",
        "Live Scanner",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 -- Strategy Finder
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
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
            st.markdown(f"""<div style="font-size:36px;font-weight:800;color:{dir_color};margin-bottom:4px;">
{len(qualifying)} qualifying candles
</div>
<div style="color:#8892b0;font-size:15px;">
found in {years:.1f} years of {ticker} data
&nbsp;|&nbsp; {len(qualifying)/max(years,0.1):.1f} per year average
</div>
<div style="color:{dir_color};font-size:13px;margin-top:4px;">{dir_badge}</div>
{adx_badge}""", unsafe_allow_html=True)

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
            st.markdown(f"""<div class="best-box">
<h3>Best Setup Found</h3>
{dir_row}
<div class="best-row"><span class="best-key">Entry Retracement</span><span class="best-val">{best_label} — {retrace_desc}</span></div>
<div class="best-row"><span class="best-key">Stop Loss</span><span class="best-val">{best_sl} {sl_label}</span></div>
<div class="best-row"><span class="best-key">Take Profit Mode</span><span class="best-val">{best['tp_mode']}</span></div>
<div class="best-row"><span class="best-key">SL Management</span><span class="best-val">{_sl_mode}</span></div>
{adx_row}
<div class="best-row"><span class="best-key">Expected Win Rate</span><span class="best-val">{best['win_rate']:.1f}%</span></div>
<div class="best-row"><span class="best-key">Profit Factor</span><span class="best-val">{pf_disp}</span></div>
<div class="best-row"><span class="best-key">Sharpe Ratio</span><span class="best-val">{best['sharpe']:.2f}</span></div>
<div class="best-row"><span class="best-key">Total Trades</span><span class="best-val">{int(best['total_trades'])}</span></div>
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
            "profit_factor":"Profit Factor","max_drawdown":"Max DD %","sharpe":"Sharpe",
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
    with tab2:
        st.markdown("### Performance Metrics (Best Setup)")
        render_metrics_grid(best_m, cols=4)
        st.markdown("---")
        st.plotly_chart(plot_equity_curve(best_trades, ticker), use_container_width=True)
        st.plotly_chart(plot_monthly_heatmap(best_trades), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 -- Trade Log
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
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
    with tab4:
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

        # ── Charts (all use filtered data) ─────────────────────────────────
        st.plotly_chart(plot_retrace_reach(df_tab4, qual_tab4, direction=_direction),
                        use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_winrate_by_retrace(all_trades_tab4), use_container_width=True)
        with c2:
            st.plotly_chart(plot_seasonal(qual_tab4), use_container_width=True)

        st.markdown("### Average Max Adverse Excursion by Retracement Level")
        st.caption(
            "MAE = how far price moved against your entry before the trade closed via TP or SL, "
            "measured in R (your risk distance). "
            "Lower MAE = cleaner entry, price moved in your favour quickly. "
            "Only TP/SL exits are included — Max Hold exits are excluded as they distort the average."
        )
        df_reset = df_tab4.reset_index()
        mae = {}
        mae_counts = {}
        for ret in RETRACEMENTS:
            vals = []
            for (r, sl, tp), trades in all_trades_tab4.items():
                if r != ret:
                    continue
                for t in trades:
                    if t.get("exit_type") not in ("TP", "SL"):
                        continue
                    if _direction == "short":
                        risk = t["sl_price"] - t["entry_price"]
                    else:
                        risk = t["entry_price"] - t["sl_price"]
                    if risk <= 0 or t["entry_date"] is None or t["exit_date"] is None:
                        continue
                    entry_ts = pd.Timestamp(t["entry_date"])
                    exit_ts  = pd.Timestamp(t["exit_date"])
                    mask = (df_reset.iloc[:, 0] >= entry_ts) & (df_reset.iloc[:, 0] <= exit_ts)
                    trade_bars = df_tab4.loc[df_reset[mask].iloc[:, 0].values] if mask.any() else pd.DataFrame()
                    if trade_bars.empty:
                        continue
                    if _direction == "short":
                        extreme_price = trade_bars["high"].max()
                        mae_r = (extreme_price - t["entry_price"]) / risk
                    else:
                        extreme_price = trade_bars["low"].min()
                        mae_r = (t["entry_price"] - extreme_price) / risk
                    vals.append(max(0.0, min(mae_r, 1.5)))
            mae[ret]        = np.mean(vals) if vals else 0
            mae_counts[ret] = len(vals)

        colors = ["#64ffda" if v < 0.4 else "#ffd700" if v < 0.7 else "#ff6b6b"
                  for v in [mae[r] for r in RETRACEMENTS]]
        mae_fig = go.Figure(go.Bar(
            x=[RETRACE_LABELS[r] for r in RETRACEMENTS],
            y=[mae[r] for r in RETRACEMENTS],
            marker_color=colors,
            text=[f"{mae[r]:.2f}R<br><span style='font-size:10px'>n={mae_counts[r]}</span>"
                  for r in RETRACEMENTS],
            textposition="outside"))
        mae_fig.add_hline(y=1.0, line_dash="dash", line_color="#ff6b6b", line_width=1,
                          annotation_text="1.0R = stop level",
                          annotation_position="top right",
                          annotation_font_color="#ff6b6b")
        _dark(mae_fig,
              title="Average Max Adverse Excursion (MAE) by Entry Retracement — TP/SL exits only",
              xaxis_title="Entry Retracement",
              yaxis_title="Avg MAE (R)",
              yaxis=dict(range=[0, max(max(mae.values()) * 1.3, 1.3)]))
        st.plotly_chart(mae_fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5 -- Portfolio Simulator
    # ═══════════════════════════════════════════════════════════════════════
    with tab5:
        render_portfolio_simulator(best_trades, df, ticker)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 6 -- Live Scanner
    # ═══════════════════════════════════════════════════════════════════════
    with tab6:
        render_live_scanner(
            ticker, timeframe, min_body_pct, min_vol_mult,
            best["retracement"], best["sl_dist"], best["tp_mode"], best_wr,
            direction=_direction,
            adx_filter_on=adx_filter,
            adx_di_gap_min=di_gap_min,
        )


if __name__ == "__main__":
    main()
