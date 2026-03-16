"""
Momentum Candle Strategy Backtester
Daily candle pattern detection, backtesting, and optimization dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
from itertools import product
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

# Feature 1: 1R added to TP modes
TP_MODES = ["1R", "2R", "3R", "Partial"]

HISTORY_OPTIONS = {
    "3 months": 90, "6 months": 180, "1 year": 365,
    "2 years": 730, "3 years": 1095, "5 years": 1825,
}

# ─── Data Layer ────────────────────────────────────────────────────────────────

def normalise_ticker(raw: str) -> str:
    t = raw.upper().strip()
    if "-" in t:
        return t
    if t.endswith("USDT"):
        return f"{t[:-4]}-USD"
    if t.endswith("USD"):
        return f"{t[:-3]}-USD"
    return t


# ─── CoinGecko Helpers ────────────────────────────────────────────────────────

def _cg_search(query: str) -> List[Dict]:
    """Search CoinGecko for coins matching query. Returns list of {id, symbol, name}."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": query}, timeout=10)
        coins = resp.json().get("coins", [])
        return [{"id": c["id"], "symbol": c["symbol"].upper(), "name": c["name"]}
                for c in coins[:5]]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _cg_fetch_ohlcv(coin_id: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch daily OHLCV from CoinGecko for a known coin ID.
    Uses OHLC endpoint for price bars + market_chart for volume.
    """
    base = "https://api.coingecko.com/api/v3/coins"
    try:
        # OHLC bars: [[ts_ms, open, high, low, close], ...]
        time.sleep(2)
        ohlc_resp = requests.get(
            f"{base}/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days}, timeout=15)
        ohlc_data = ohlc_resp.json()
        if not isinstance(ohlc_data, list) or not ohlc_data:
            return pd.DataFrame()

        ohlc_df = pd.DataFrame(ohlc_data, columns=["ts", "open", "high", "low", "close"])
        ohlc_df["date"] = pd.to_datetime(ohlc_df["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
        ohlc_df = ohlc_df.groupby("date").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"),    close=("close", "last")).reset_index()

        # Volume: [[ts_ms, vol], ...]
        time.sleep(2)
        vol_resp = requests.get(
            f"{base}/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=15)
        vol_data = vol_resp.json().get("total_volumes", [])
        if vol_data:
            vol_df = pd.DataFrame(vol_data, columns=["ts", "volume"])
            vol_df["date"] = pd.to_datetime(vol_df["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
            vol_df = vol_df.groupby("date")["volume"].last().reset_index()
            ohlc_df = ohlc_df.merge(vol_df, on="date", how="left")
        else:
            ohlc_df["volume"] = 0.0

        ohlc_df.set_index("date", inplace=True)
        ohlc_df.sort_index(inplace=True)
        return _clean_df(ohlc_df)
    except Exception as e:
        print(f"[CoinGecko] Error fetching {coin_id}: {e}")
        return pd.DataFrame()


def _cg_resolve(ticker: str) -> tuple:
    """
    Try to find the best CoinGecko coin ID for a given ticker string.
    Returns (coin_id or None, list_of_suggestions).
    Strips -USD suffix before searching.
    """
    query = ticker.replace("-USD", "").replace("-USDT", "")
    results = _cg_search(query)
    if not results:
        return None, []
    # Prefer exact symbol match
    for r in results:
        if r["symbol"] == query.upper():
            return r["id"], results
    # Fall back to first result
    return results[0]["id"], results


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


@st.cache_data(ttl=3600, show_spinner=False)
def _yf_fetch(ticker: str, timeframe: str) -> pd.DataFrame:
    """yfinance fetch only — cached."""
    print(f"\n[yfinance] {ticker} @ {timeframe}")
    if timeframe == "1D":
        raw = yf.download(ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        return _clean_df(raw)
    else:
        raw = yf.download(ticker, period="2y", interval="1h",
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = _clean_df(raw)
        if raw.empty:
            return pd.DataFrame()
        if timeframe == "4H":
            df4 = raw[["open","high","low","close","volume"]].resample("4h").agg(
                {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
            ).dropna()
            return _clean_df(df4)
        return raw


def fetch_data(ticker: str, timeframe: str,
               cg_coin_id: str = "") -> tuple:
    """
    Fetch OHLCV data.
    Priority: yfinance → CoinGecko fallback.
    Returns (df, source_label, cg_suggestions).
    """
    # 1. Try yfinance
    df = _yf_fetch(ticker, timeframe)
    if not df.empty:
        return df, "yfinance", []

    # 2. CoinGecko fallback
    days = {
        "1D": 1825,   # 5y daily
        "4H": 730,    # 2y (no sub-daily on free tier, use daily)
        "1H": 730,
    }.get(timeframe, 1825)

    coin_id = cg_coin_id
    suggestions = []
    if not coin_id:
        coin_id, suggestions = _cg_resolve(ticker)

    if coin_id:
        df = _cg_fetch_ohlcv(coin_id, days)
        if not df.empty:
            return df, f"CoinGecko ({coin_id})", suggestions

    return pd.DataFrame(), "none", suggestions


def fetch_live(ticker: str, timeframe: str,
               cg_coin_id: str = "") -> pd.DataFrame:
    """
    Fetch fresh (uncached) data for the live scanner.
    Uses same timeframe logic as the main backtest.
    Falls back to CoinGecko daily data if yfinance fails.
    """
    period_map = {"1D": "60d", "4H": "14d", "1H": "5d"}
    interval   = "1d" if timeframe == "1D" else "1h"
    period     = period_map[timeframe]

    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is not None and not raw.empty:
        df = _clean_df(raw)
        if not df.empty:
            if timeframe == "4H":
                df4 = df[["open","high","low","close","volume"]].resample("4h").agg(
                    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                ).dropna()
                df = _clean_df(df4)
            return df

    # CoinGecko fallback (daily only for free tier)
    coin_id = cg_coin_id
    if not coin_id:
        coin_id, _ = _cg_resolve(ticker)
    if coin_id:
        return _cg_fetch_ohlcv(coin_id, days=60)
    return pd.DataFrame()


def trim_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = df.index[-1] - timedelta(days=days)
    return df[df.index >= cutoff].copy()

# ─── Candle Detection ──────────────────────────────────────────────────────────

def detect_qualifying_candles(df, min_body_pct=0.70, min_vol_mult=1.5):
    mask = (
        (df["body"] > 0) &
        (df["body_pct"] >= min_body_pct) &
        (df["vol_mult"] >= min_vol_mult)
    )
    return df[mask].copy()

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
) -> Optional[Dict]:
    trigger     = df.iloc[trigger_idx]
    entry_price = trigger["close"] - trigger["body"] * retracement
    sl_price    = entry_price * (1 - sl_dist)
    risk        = entry_price - sl_price

    if risk <= 0:
        return None

    # Feature 1: 1R TP
    if tp_mode == "1R":
        tp1 = entry_price + 1 * risk
        tp2 = None
    elif tp_mode == "2R":
        tp1 = entry_price + 2 * risk
        tp2 = None
    elif tp_mode == "3R":
        tp1 = entry_price + 3 * risk
        tp2 = None
    else:  # Partial
        tp1 = entry_price + 1 * risk
        tp2 = entry_price + 2 * risk

    future_start = trigger_idx + 1
    future_end   = min(future_start + max_hold_bars, len(df))
    if future_start >= len(df):
        return None

    # For 0% retracement (immediate entry), assume filled at open of next bar
    if retracement == 0.0:
        entry_filled = True
        entry_date   = df.index[future_start]
        current_sl   = sl_price
        be_moved     = False
        scan_start   = future_start
    else:
        entry_filled      = False
        entry_date        = None
        current_sl        = sl_price
        be_moved          = False
        scan_start        = future_start

    partial_exited     = False
    partial_exit_price = None

    for fwd_i in range(scan_start, future_end):
        bar      = df.iloc[fwd_i]
        bar_date = df.index[fwd_i]

        if not entry_filled:
            if bar["low"] <= entry_price:
                entry_filled = True
                entry_date   = bar_date
                if bar["low"] <= current_sl:
                    r = -(1 + transaction_cost * 2)
                    return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                       entry_date, sl_price, tp1, current_sl,
                                       bar_date, "Loss", r, retracement,
                                       sl_dist, tp_mode, sl_mode)
            continue

        # Trailing / breakeven SL updates
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

        # SL check
        if bar["low"] <= current_sl:
            r      = (current_sl - entry_price) / risk - transaction_cost * 2
            result = "Win" if r > 0 else "Loss"
            return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                               entry_date, sl_price, tp1, current_sl,
                               bar_date, result, r, retracement,
                               sl_dist, tp_mode, sl_mode)

        # TP checks
        if tp_mode == "Partial":
            if not partial_exited and bar["high"] >= tp1:
                partial_exited     = True
                partial_exit_price = tp1
                if sl_mode in ("Breakeven", "Breakeven+Trail"):
                    current_sl = entry_price
            if partial_exited and bar["high"] >= tp2:
                avg = (partial_exit_price + tp2) / 2
                r   = (avg - entry_price) / risk - transaction_cost * 2
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp2, avg,
                                   bar_date, "Partial Win", r, retracement,
                                   sl_dist, tp_mode, sl_mode)
        else:
            if bar["high"] >= tp1:
                r = (tp1 - entry_price) / risk - transaction_cost * 2
                return _trade_dict(trigger, df.index[trigger_idx], entry_price,
                                   entry_date, sl_price, tp1, tp1,
                                   bar_date, "Win", r, retracement,
                                   sl_dist, tp_mode, sl_mode)

    if not entry_filled:
        return None

    last_i     = min(future_end - 1, len(df) - 1)
    last_bar   = df.iloc[last_i]
    last_date  = df.index[last_i]
    exit_price = last_bar["close"]

    if tp_mode == "Partial" and partial_exited:
        avg        = (partial_exit_price + exit_price) / 2
        r          = (avg - entry_price) / risk - transaction_cost * 2
        exit_price = avg
        result     = "Partial Win" if r > 0 else "Loss"
    else:
        r      = (exit_price - entry_price) / risk - transaction_cost * 2
        result = "Win" if r > 0 else "Loss"

    return _trade_dict(trigger, df.index[trigger_idx], entry_price, entry_date,
                       sl_price, tp1, exit_price, last_date, result, r,
                       retracement, sl_dist, tp_mode, sl_mode)


def _trade_dict(trigger, trigger_date, entry_price, entry_date,
                sl_price, tp_price, exit_price, exit_date,
                result, r_mult, retracement, sl_dist, tp_mode, sl_mode) -> Dict:
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
    }


def run_backtest(df, qualifying, retracement, sl_dist, tp_mode,
                 sl_mode="Fixed", trail_atr_mult=1.5,
                 transaction_cost=0.001) -> List[Dict]:
    trades   = []
    df_reset = df.reset_index()
    q_dates  = set(qualifying.index)
    for date in q_dates:
        loc = df_reset[df_reset.iloc[:, 0] == date]
        if loc.empty:
            continue
        idx   = loc.index[0]
        trade = simulate_trade(idx, df, retracement, sl_dist, tp_mode,
                               sl_mode, trail_atr_mult, transaction_cost)
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
    holds = [t["hold_days"] for t in trades if t["hold_days"] > 0]
    return {
        "total_trades": total, "win_rate": len(wins) / total,
        "avg_r": r.mean(), "profit_factor": pf,
        "max_drawdown": dd.min(), "sharpe": sharpe,
        "longest_losing_streak": max_s,
        "avg_hold_days": np.mean(holds) if holds else 0,
        "gross_profit": gp, "gross_loss": gl,
        "equity": equity, "r_mults": r,
    }

# ─── Optimization ──────────────────────────────────────────────────────────────

def optimize(df, qualifying, sl_mode="Fixed", trail_atr_mult=1.5,
             transaction_cost=0.001):
    """Returns (results_df, all_trades_dict) — runs each combo only once."""
    results    = []
    all_trades = {}
    combos     = list(product(RETRACEMENTS, SL_DISTANCES, TP_MODES))
    n          = len(combos)
    prog       = st.progress(0, text=f"Testing {n} parameter combinations...")
    for i, (ret, sl, tp) in enumerate(combos):
        trades = run_backtest(df, qualifying, ret, sl, tp,
                              sl_mode, trail_atr_mult, transaction_cost)
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


def plot_retrace_reach(df, qualifying):
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
        ml = future["low"].min()
        total += 1
        for r in RETRACEMENTS:
            level = trigger["close"] - trigger["body"] * r
            if ml <= level:
                hits[r] += 1
    if not total:
        return go.Figure()
    pcts  = [hits[r] / total * 100 for r in RETRACEMENTS]
    lbls  = [RETRACE_LABELS[r] for r in RETRACEMENTS]
    colors = ["#64ffda" if p >= 50 else "#ffd700" if p >= 30 else "#ff6b6b"
              for p in pcts]
    fig = go.Figure(go.Bar(x=lbls, y=pcts, marker_color=colors,
                           text=[f"{p:.1f}%" for p in pcts],
                           textposition="outside"))
    return _dark(fig, "How Deep Did Price Retrace After Trigger Candle? (5-bar window)",
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

    stats = {tp: [] for tp in TP_MODES}
    for (ret, sl, tp), trades in all_trades.items():
        if not trades or tp not in stats:
            continue
        wr = sum(1 for t in trades if t["r_mult"] > 0) / len(trades) * 100
        stats[tp].append(wr)

    st.markdown("#### TP Mode Win Rate Comparison")
    labels = {"1R":"1R Fixed","2R":"Full 2R","3R":"Full 3R","Partial":"Partial (1R+2R)"}
    cols   = st.columns(len(TP_MODES))
    for col, tp in zip(cols, TP_MODES):
        avg = np.mean(stats[tp]) if stats[tp] else 0
        with col:
            st.markdown(metric_card(labels[tp], f"{avg:.1f}%"),
                        unsafe_allow_html=True)

# ─── SL Comparison ────────────────────────────────────────────────────────────

def render_sl_comparison(df, qualifying, best_ret, best_sl_dist,
                          best_tp, trail_mult, tc):
    st.markdown("### Stop Loss Method Comparison")

    # Feature 1: also include a dedicated 1R Fixed row
    tests  = [
        ("Fixed",          best_tp),
        ("Fixed",          "1R"),       # 1R Fixed dedicated row
        ("Breakeven",      best_tp),
        ("Trailing ATR",   best_tp),
        ("Breakeven+Trail",best_tp),
    ]
    labels = [
        f"Fixed SL ({best_tp})",
        "1R Fixed -- highest win rate, lowest R per trade",
        f"Breakeven SL ({best_tp})",
        f"Trailing ATR ({best_tp})",
        f"Breakeven+Trail ({best_tp})",
    ]
    rows = []
    for (mode, tp), lbl in zip(tests, labels):
        trades = run_backtest(df, qualifying, best_ret, best_sl_dist,
                              tp, mode, trail_mult, tc)
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

# ─── Feature 3: Live Scanner ───────────────────────────────────────────────────

def render_live_scanner(ticker, timeframe, min_body_pct, min_vol_mult,
                         best_ret, best_sl_dist, best_tp, best_wr,
                         cg_coin_id=""):
    tf_label = {"1D": "Daily", "4H": "4H", "1H": "1H"}[timeframe]
    st.markdown(f"### Live Scanner -- {tf_label} Candles")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        refresh = st.button("Refresh Scanner", key="scanner_refresh",
                            use_container_width=True)

    # Re-fetch whenever ticker OR timeframe changes, or refresh pressed
    live_key = f"{ticker}_{timeframe}"
    if refresh or "live_df" not in st.session_state or \
            st.session_state.get("live_key") != live_key:
        spinner_msg = "Fetching latest candles..."
        with st.spinner(spinner_msg):
            live_df = fetch_live(ticker, timeframe, cg_coin_id)
        st.session_state["live_df"]  = live_df
        st.session_state["live_ts"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["live_key"] = live_key

    live_df = st.session_state.get("live_df", pd.DataFrame())
    live_ts = st.session_state.get("live_ts", "unknown")

    with col_a:
        st.caption(f"Last updated: {live_ts}")

    if live_df.empty:
        st.error("Could not fetch live data.")
        return

    # Take last 20 completed candles (exclude the current incomplete one)
    scan = live_df.iloc[-21:-1].copy() if len(live_df) > 20 else live_df.iloc[:-1].copy()
    if scan.empty:
        st.warning("Not enough candle data for scanner.")
        return

    st.markdown(f"**Last 20 {tf_label} candles for {ticker}**")

    # Build scanner table
    table_rows = []
    for i, (idx, row) in enumerate(scan.iterrows()):
        bp_ok  = (row["body"] > 0) and (row["body_pct"] >= min_body_pct)
        vol_ok = (row["vol_mult"] >= min_vol_mult)
        sig    = "SIGNAL" if (bp_ok and vol_ok) else ("partial" if (bp_ok or vol_ok) else "")
        bp_val = row["body_pct"] * 100 if pd.notna(row["body_pct"]) else 0
        vm_val = row["vol_mult"] if pd.notna(row["vol_mult"]) else 0
        table_rows.append({
            "#":        len(scan) - i,
            "Date":     str(idx.date()) if hasattr(idx, "date") else str(idx),
            "Body %":   round(bp_val, 1),
            "Vol Mult": round(vm_val, 2),
            "Body OK":  "YES" if bp_ok  else "no",
            "Vol OK":   "YES" if vol_ok else "no",
            "SIGNAL":   sig,
            "_body_ok": bp_ok,
            "_vol_ok":  vol_ok,
            "_signal":  bp_ok and vol_ok,
        })

    scan_df = pd.DataFrame(table_rows)

    def style_row(row):
        if row["_signal"]:
            return ["background-color: rgba(63,185,80,0.18); color:#3fb950"] * len(row)
        if row["_body_ok"] or row["_vol_ok"]:
            return ["background-color: rgba(255,215,0,0.10); color:#ccd6f6"] * len(row)
        return ["color:#6e7681"] * len(row)

    display_cols = ["#","Date","Body %","Vol Mult","Body OK","Vol OK","SIGNAL"]
    st.dataframe(
        scan_df[display_cols + ["_body_ok","_vol_ok","_signal"]].style.apply(
            style_row, axis=1).hide(axis="columns",
            subset=["_body_ok","_vol_ok","_signal"]),
        use_container_width=True, height=440,
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
        # Find the row in live_df
        match = live_reset[live_reset.iloc[:, 0].astype(str).str.startswith(date_str)]
        if match.empty:
            continue
        row     = live_df.iloc[match.index[0]]
        close   = row["close"]
        body    = row["body"]
        sl_dist = best_sl_dist

        entry    = round(close - body * best_ret, 6)
        sl       = round(entry * (1 - sl_dist), 6)
        risk_amt = entry - sl
        tp1r     = round(entry + 1 * risk_amt, 6)
        tp2r     = round(entry + 2 * risk_amt, 6)
        tp3r     = round(entry + 3 * risk_amt, 6)

        wr_str = f"{best_wr:.1f}%" if best_wr else "N/A"

        st.markdown(f"""<div class="signal-card">
<h4>SIGNAL FOUND -- {date_str} candle</h4>
<div class="signal-line">Trigger close: <span>${close:,.6f}</span> &nbsp;|&nbsp;
Body: <span>{sr['Body %']:.1f}%</span> &nbsp;|&nbsp;
Vol mult: <span>{sr['Vol Mult']:.2f}x</span></div>
<hr style="border-color:#1e2d1e; margin:8px 0;">
<div class="signal-line">Based on historical best setup for <span>{ticker}</span>:</div>
<div class="signal-line">Entry:  <span>${entry:,.6f}</span> &nbsp; ({RETRACE_LABELS[best_ret]} retrace from close)</div>
<div class="signal-line">SL:     <span>${sl:,.6f}</span> &nbsp; ({SL_LABELS[sl_dist]} below entry)</div>
<div class="signal-line">TP1 (1R): <span>${tp1r:,.6f}</span></div>
<div class="signal-line">TP2 (2R): <span>${tp2r:,.6f}</span></div>
<div class="signal-line">TP3 (3R): <span>${tp3r:,.6f}</span></div>
<div class="signal-line">Historical win rate at this setup: <span>{wr_str}</span> ({best_tp} TP)</div>
</div>""", unsafe_allow_html=True)

# ─── Main App ──────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("## Momentum Candle")
        st.markdown("---")

        ticker = normalise_ticker(
            st.text_input("Ticker Symbol", value="BTC-USD",
                          placeholder="BTC-USD, ETH-USD, MNT, HYPE..."))
        coin_name_search = st.text_input(
            "Or search by full name (CoinGecko):",
            placeholder="e.g. mantle, hyperliquid, pepe...",
            key="cg_name_search")

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
        st.markdown("**Stop Loss Management**")
        sl_mode    = st.radio("SL Method",
                              ["Fixed","Breakeven","Trailing ATR","Breakeven+Trail"],
                              index=0)
        trail_mult = 1.5
        if sl_mode in ("Trailing ATR","Breakeven+Trail"):
            trail_mult = st.slider("ATR Trail Multiplier", 1.0, 3.0, 1.5, 0.1)

        st.markdown("---")
        run_btn = st.button("Run Analysis", use_container_width=True, type="primary")
        st.caption("Data: yfinance / CoinGecko | Risk model: 2% fixed")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not ticker:
        st.info("Enter a ticker symbol in the sidebar.")
        return

    # Resolve CoinGecko ID from full-name search if provided
    cg_coin_id   = ""
    cg_name_hit  = ""
    if coin_name_search:
        cg_results = _cg_search(coin_name_search)
        if cg_results:
            cg_coin_id  = cg_results[0]["id"]
            cg_name_hit = f"{cg_results[0]['name']} ({cg_results[0]['symbol']})"
            st.sidebar.caption(f"CoinGecko match: {cg_name_hit}")
        else:
            st.sidebar.warning(f"No CoinGecko match for '{coin_name_search}'")

    # ── Fetch data with status messages ───────────────────────────────────────
    df_full        = pd.DataFrame()
    data_source    = "none"
    cg_suggestions = []

    with st.spinner(f"Fetching {ticker} ({timeframe}) data..."):
        df_full = _yf_fetch(ticker, timeframe)
        if not df_full.empty:
            data_source = "yfinance"

    if df_full.empty:
        # yfinance failed — try CoinGecko
        query     = ticker.replace("-USD", "").replace("-USDT", "")
        cg_status = st.empty()
        cg_status.info(f"🔍 Searching CoinGecko for **{query}**...")

        resolved_id = cg_coin_id
        if not resolved_id:
            resolved_id, cg_suggestions = _cg_resolve(ticker)

        if resolved_id:
            search_hits = _cg_search(query)
            coin_name   = next((r["name"] for r in search_hits if r["id"] == resolved_id),
                               resolved_id)
            cg_status.info(f"✅ Found: **{coin_name}** — fetching 365 days of data...")
            days = {"1D": 1825, "4H": 730, "1H": 730}.get(timeframe, 1825)
            with st.spinner(f"Downloading {coin_name} from CoinGecko..."):
                df_full = _cg_fetch_ohlcv(resolved_id, days)
            if not df_full.empty:
                data_source = f"CoinGecko ({resolved_id})"
                cg_status.success(
                    f"✅ Data source: CoinGecko — **{coin_name}** ({len(df_full)} candles loaded)")
            else:
                cg_status.error(f"❌ CoinGecko returned no data for {coin_name}.")
        else:
            cg_status.empty()

    if df_full.empty:
        query = ticker.replace("-USD", "").replace("-USDT", "")
        if cg_suggestions:
            names = ", ".join(f"{s['name']} ({s['symbol']})" for s in cg_suggestions[:3])
            st.error(
                f"❌ Could not find **{ticker}** on yfinance or CoinGecko.\n\n"
                f"Did you mean: {names}?\n\n"
                "Try typing the full name in the **search box** below the ticker, "
                "e.g. `mantle` instead of `MNT`.")
        else:
            st.error(
                f"❌ Could not find **{ticker}** on yfinance or CoinGecko.\n\n"
                "Try: `BTC-USD`, `ETH-USD`, `SOL-USD` format — or use the "
                "**full-name search** box in the sidebar (e.g. type `mantle` instead of `MNT`).")
        return

    # Data source badge
    badge_color = "#3fb950" if "CoinGecko" not in data_source else "#58a6ff"
    st.sidebar.markdown(
        f'<span style="color:{badge_color};font-size:12px;">&#x2705; Data source: {data_source}</span>',
        unsafe_allow_html=True)

    df = trim_by_days(df_full, history_days)
    if df.empty:
        st.error("No data in selected time range.")
        return

    start_date = df.index[0].date()
    st.sidebar.caption(f"Using data from {start_date} to {df.index[-1].date()}")

    years      = (df.index[-1] - df.index[0]).days / 365
    qualifying = detect_qualifying_candles(df, min_body_pct, min_vol_mult)

    if len(qualifying) < 20:
        st.warning(f"Only {len(qualifying)} qualifying candles found -- "
                   "increase timespan for more reliable results.")

    st.title(f"{ticker} ({timeframe}) -- Momentum Candle Strategy")
    st.caption(f"Data: {start_date} to {df.index[-1].date()} "
               f"({years:.1f} years) | {len(df)} bars | {len(qualifying)} qualifying triggers")

    # ── Run optimization ─────────────────────────────────────────────────────
    run_key = (f"{ticker}_{timeframe}_{history_label}_{min_body_pct:.2f}_"
               f"{min_vol_mult}_{sl_mode}_{trail_mult}_{transaction_cost}")
    if ("opt_results" not in st.session_state or run_btn or
            st.session_state.get("run_key") != run_key):
        if len(qualifying) < 3:
            st.warning("Too few qualifying candles -- relax the criteria.")
            return
        try:
            opt_df, all_trades = optimize(df, qualifying, sl_mode, trail_mult, transaction_cost)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)
            return
        st.session_state.update({
            "opt_results": opt_df, "ticker": ticker, "df": df,
            "qualifying": qualifying, "all_trades": all_trades,
            "transaction_cost": transaction_cost, "sl_mode": sl_mode,
            "trail_mult": trail_mult, "run_key": run_key, "years": years,
        })

    if "opt_results" not in st.session_state:
        st.info("Click **Run Analysis** to start.")
        return

    opt_df     = st.session_state["opt_results"]
    df         = st.session_state["df"]
    qualifying = st.session_state["qualifying"]
    all_trades = st.session_state["all_trades"]
    tc         = st.session_state["transaction_cost"]
    _sl_mode   = st.session_state["sl_mode"]
    _trail     = st.session_state["trail_mult"]
    years      = st.session_state["years"]

    if opt_df.empty:
        st.warning("No parameter combination produced >=3 trades. Try different criteria.")
        return

    best        = opt_df.iloc[0]
    # Reuse trades already computed in optimize() — avoids an extra backtest run.
    # Note: all_trades uses the same sl_mode/trail_mult from the same run.
    best_trades = all_trades.get(
        (best["retracement"], best["sl_dist"], best["tp_mode"]),
        run_backtest(df, qualifying, best["retracement"],
                     best["sl_dist"], best["tp_mode"],
                     _sl_mode, _trail, tc))
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
            st.markdown(f"""
            <div style="font-size:36px;font-weight:800;color:#64ffda;margin-bottom:4px;">
                {len(qualifying)} qualifying candles
            </div>
            <div style="color:#8892b0;font-size:15px;">
                found in {years:.1f} years of {ticker} data
                &nbsp;|&nbsp; {len(qualifying)/max(years,0.1):.1f} per year average
            </div>""", unsafe_allow_html=True)

        with col_r:
            best_label = RETRACE_LABELS[best["retracement"]]
            best_sl    = SL_LABELS[best["sl_dist"]]
            pf_disp    = f"{best['profit_factor']:.2f}" if best["profit_factor"] < 90 else "inf"
            st.markdown(f"""<div class="best-box">
            <h3>Best Setup Found</h3>
            <div class="best-row"><span class="best-key">Entry Retracement</span><span class="best-val">{best_label} of candle body</span></div>
            <div class="best-row"><span class="best-key">Stop Loss</span><span class="best-val">{best_sl} below entry</span></div>
            <div class="best-row"><span class="best-key">Take Profit Mode</span><span class="best-val">{best['tp_mode']}</span></div>
            <div class="best-row"><span class="best-key">SL Management</span><span class="best-val">{_sl_mode}</span></div>
            <div class="best-row"><span class="best-key">Expected Win Rate</span><span class="best-val">{best['win_rate']:.1f}%</span></div>
            <div class="best-row"><span class="best-key">Profit Factor</span><span class="best-val">{pf_disp}</span></div>
            <div class="best-row"><span class="best-key">Sharpe Ratio</span><span class="best-val">{best['sharpe']:.2f}</span></div>
            <div class="best-row"><span class="best-key">Total Trades</span><span class="best-val">{int(best['total_trades'])}</span></div>
            </div>""", unsafe_allow_html=True)

        # Feature 1 -- 1R vs 2R comparison note
        one_r_trades = run_backtest(df, qualifying, best["retracement"],
                                    best["sl_dist"], "1R", "Fixed", 1.5, tc)
        two_r_trades = run_backtest(df, qualifying, best["retracement"],
                                    best["sl_dist"], "2R", "Fixed", 1.5, tc)
        one_r_m = calc_metrics(one_r_trades)
        two_r_m = calc_metrics(two_r_trades)
        if one_r_m and two_r_m:
            st.markdown("---")
            st.markdown("#### 1R Fixed vs 2R Comparison (same entry/SL)")
            c1r, c2r = st.columns(2)
            with c1r:
                st.markdown(metric_card(
                    "1R Fixed Win Rate",
                    f"{one_r_m['win_rate']*100:.1f}%",
                    "green" if one_r_m["win_rate"] >= 0.5 else "red"),
                    unsafe_allow_html=True)
                st.markdown(metric_card(
                    "1R Fixed Profit Factor",
                    f"{one_r_m['profit_factor']:.2f}" if not np.isinf(one_r_m['profit_factor']) else "inf"),
                    unsafe_allow_html=True)
            with c2r:
                st.markdown(metric_card(
                    "2R Win Rate",
                    f"{two_r_m['win_rate']*100:.1f}%",
                    "green" if two_r_m["win_rate"] >= 0.5 else "red"),
                    unsafe_allow_html=True)
                st.markdown(metric_card(
                    "2R Profit Factor",
                    f"{two_r_m['profit_factor']:.2f}" if not np.isinf(two_r_m['profit_factor']) else "inf"),
                    unsafe_allow_html=True)
            # Interpretation line
            if one_r_m["win_rate"] > two_r_m["win_rate"]:
                higher_wr = "1R wins more often"
            else:
                higher_wr = "2R wins more often"
            if two_r_m["profit_factor"] > one_r_m["profit_factor"]:
                better_pf = "2R makes more money overall"
            else:
                better_pf = "1R makes more money overall"
            st.caption(f"{higher_wr}, but {better_pf}.")

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
                             best["sl_dist"], best["tp_mode"], _trail, tc)

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
        if not best_trades:
            st.info("No trades to display.")
        else:
            log = pd.DataFrame(best_trades)[[
                "trigger_date","trigger_body_pct","trigger_vol_mult",
                "entry_price","entry_date","sl_price","tp_price",
                "exit_price","exit_date","result","r_mult","hold_days",
            ]].copy()
            log.columns = [
                "Trigger Date","Body %","Vol Mult",
                "Entry Price","Entry Date","SL Price","TP Price",
                "Exit Price","Exit Date","Result","R Multiple","Hold Days",
            ]
            for c in ["Trigger Date","Entry Date","Exit Date"]:
                log[c] = pd.to_datetime(log[c]).dt.date

            def color_result(val):
                if val == "Win":      return "color:#64ffda"
                if val == "Loss":     return "color:#ff6b6b"
                return "color:#ffd700"

            st.dataframe(log.style.applymap(color_result, subset=["Result"]),
                         use_container_width=True, height=500)
            st.download_button("Download Trade Log CSV",
                               log.to_csv(index=False).encode("utf-8"),
                               f"{ticker}_trades.csv", "text/csv")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 -- Candle Statistics
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
        # Feature 2: full Fib depth chart
        st.plotly_chart(plot_retrace_reach(df, qualifying), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_winrate_by_retrace(all_trades), use_container_width=True)
        with c2:
            st.plotly_chart(plot_seasonal(qualifying), use_container_width=True)

        st.markdown("### Average Max Adverse Excursion by Retracement Level")
        mae = {}
        for ret in RETRACEMENTS:
            vals = []
            for (r, sl, tp), trades in all_trades.items():
                if r != ret:
                    continue
                for t in trades:
                    risk = t["entry_price"] - t["sl_price"]
                    if risk > 0:
                        vals.append((t["entry_price"] - t["sl_price"]) / risk)
            mae[ret] = np.mean(vals) if vals else 0
        mae_fig = go.Figure(go.Bar(
            x=[RETRACE_LABELS[r] for r in RETRACEMENTS],
            y=[mae[r] for r in RETRACEMENTS],
            marker_color="#7c83fd",
            text=[f"{mae[r]:.2f}R" for r in RETRACEMENTS],
            textposition="outside"))
        _dark(mae_fig, xaxis_title="Entry Retracement",
              yaxis_title="Avg Adverse Excursion (R)")
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
            cg_coin_id=cg_coin_id,
        )


if __name__ == "__main__":
    main()
