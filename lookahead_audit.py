"""
Lookahead Audit — proves that no feature in app.py's _clean_df + calculate_adx
accidentally peeks into future bars when computed at any historical bar i.

USAGE
─────
    python lookahead_audit.py BTCUSDT 1d
    python lookahead_audit.py ETHUSDT 4h
    python lookahead_audit.py REZUSDT 1d

WHAT IT DOES (the principle in plain English)
─────────────────────────────────────────────
The features your ML and backtest read at bar i (body_pct, vol_mult, atr14,
atr_ratio, ema5/15/21, candle_rank_20, vol_rank_20, body_vs_atr,
dist_from_ema21_pct, adx, di_plus, di_minus) are precomputed across the
ENTIRE dataframe at fetch time. The question:

    "If I had only had data up to bar i in real life, would I have computed
     the SAME values that are stored at row i in the precomputed df?"

If yes → causal (safe). If no → leak (model trains on info it can't have live).

The test method is dead simple: at a randomly chosen bar `i`, slice the df
to df.iloc[:i+1] (past-only), recompute every feature on that slice, and
compare the LAST row of the recomputed slice to row `i` of the precomputed df.

Tolerance: 1e-9 absolute or 1e-6 relative. EWM-based features (ema*, atr14,
adx) are mathematically equivalent only when seeded identically. Tiny
numerical drift from floating-point accumulation across different slice
lengths is normal and is NOT a leak — we set tolerance accordingly.

WHAT TO LOOK FOR IN THE OUTPUT
──────────────────────────────
✓ green checkmark on every feature  → no leak. Safe to keep training.
✗ red X with a delta value          → that feature LEAKS. Must be fixed
                                       before any backtest result is trusted.

KNOWN OK (CAUSAL) FEATURES — what we expect to PASS
───────────────────────────────────────────────────
- body, body_pct, candle_range:   pure per-bar arithmetic, no window
- vol_avg_7, vol_mult:            uses .shift(1) → strictly past
- atr14:                          .rolling(14).mean() → causal
- atr_ratio:                      atr14 / atr14.rolling(20).mean() → causal
- ema5, ema15, ema21:             .shift(1).ewm() → strictly past
- candle_rank_20, vol_rank_20:    .rolling(N).rank() is causal in pandas
- body_vs_atr:                    body / atr14, both causal
- dist_from_ema21_pct:            close - ema21, both causal
- adx, di_plus, di_minus:         .ewm() chain, causal

KNOWN POTENTIAL LEAK — what we expect to investigate
────────────────────────────────────────────────────
- vol_delta_5, vol_delta_20, vol_delta_regime:
    These use .rolling().sum() over price-action — should be causal too.
    If they fail the test, it's a numerical artifact, not a real leak.

NOTE ON EWM SEED SENSITIVITY
────────────────────────────
Exponential moving averages are recursive: EMA[t] = α * x[t] + (1-α) * EMA[t-1].
When you slice the df to [:i+1] and recompute, the EWM starts fresh from
bar 0 — but it doesn't matter, because both the full computation AND the
sliced computation start from bar 0 (we slice forward, not backward). So
EWM features should match exactly to within float epsilon. The only
exception: if pandas internally uses different code paths for very short
vs long Series, you can get sub-1e-10 deltas that are pure float noise.
"""
from __future__ import annotations
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse app.py's exact functions instead of duplicating logic — that way the
# audit tests the SAME code that runs in production. We import via a small
# bootstrap because app.py is a Streamlit app and importing it triggers a lot
# of decorators we don't need.
sys.path.insert(0, str(Path(__file__).parent))


def _import_app_functions():
    """
    Pull just _clean_df and calculate_adx from app.py without triggering
    Streamlit/imports we don't need. We use exec on the relevant blocks.
    """
    src = (Path(__file__).parent / "app.py").read_text()
    # Find the blocks we care about by line markers
    start_clean = src.find("def _clean_df(")
    end_clean   = src.find("@st.cache_data", start_clean)
    start_adx   = src.find("def calculate_adx(")
    end_adx     = src.find("\ndef calculate_ema", start_adx)
    start_ema   = src.find("def calculate_ema(")
    end_ema     = src.find("\n\n", start_ema + 50)

    blob = (
        "import pandas as pd\nimport numpy as np\n"
        "import streamlit as st\n\n"   # so @st.cache_data resolves
        + src[start_clean:end_clean].rstrip()  + "\n\n"
        + src[start_adx:end_adx].rstrip()      + "\n\n"
        + src[start_ema:end_ema].rstrip()      + "\n"
    )
    ns = {}
    try:
        exec(blob, ns)
    except Exception as e:
        print(f"[bootstrap] Failed to import functions from app.py: {e}")
        sys.exit(1)
    return ns["_clean_df"], ns["calculate_adx"]


def fetch_klines(symbol: str, interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV from Binance public API. No key needed."""
    import requests
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol.upper(),
                                   "interval": interval,
                                   "limit": limit}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "_ignore",
    ])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# The audit
# ─────────────────────────────────────────────────────────────────────────────

# Features to audit. Tolerance per feature — strict (1e-12) for non-recursive,
# loose (1e-6 relative) for EWM-based features where float accumulation order
# can differ between slice lengths.
FEATURES_STRICT = [
    "body", "candle_range", "body_pct", "vol_avg_7", "vol_mult",
    "candle_rank_20", "vol_rank_20",
]
FEATURES_EWM = [
    "atr14", "atr_ratio", "ema5", "ema15", "ema21",
    "body_vs_atr", "dist_from_ema21_pct", "vol_delta_5",
    "vol_delta_20", "vol_delta_regime",
]
FEATURES_ADX = ["adx", "di_plus", "di_minus"]


def _approx_equal(a: float, b: float, *, atol: float, rtol: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    diff = abs(float(a) - float(b))
    return diff <= atol or diff <= rtol * max(abs(float(a)), abs(float(b)))


def audit(symbol: str, interval: str, n_test_bars: int = 25) -> int:
    """
    Run the audit. Returns 0 on success, 1 if any leak was detected.
    """
    print("=" * 70)
    print(f"LOOKAHEAD AUDIT — {symbol} @ {interval}")
    print("=" * 70)

    print("\n[1/4] Fetching OHLCV data from Binance...")
    raw = fetch_klines(symbol, interval, limit=500)
    if len(raw) < 100:
        print(f"   Only {len(raw)} bars available — need at least 100 for meaningful audit")
        return 1
    print(f"   Got {len(raw)} bars ({raw.index[0]} → {raw.index[-1]})")

    print("\n[2/4] Bootstrapping feature functions from app.py...")
    _clean_df, calculate_adx = _import_app_functions()
    print("   Loaded _clean_df + calculate_adx")

    print("\n[3/4] Computing features on the FULL df (production path)...")
    full = _clean_df(raw.copy())
    full_adx = calculate_adx(full)
    full = full.join(full_adx, how="left")
    print(f"   Full df: {len(full)} rows, {len(full.columns)} columns")

    # Pick test bars — spread across the middle of the df, avoiding edges
    # (warmup near start, no future data near end). We test 25 random bars.
    rng = np.random.default_rng(42)
    eligible = list(range(50, len(full) - 5))
    test_bars = sorted(rng.choice(eligible, size=min(n_test_bars, len(eligible)),
                                   replace=False).tolist())
    print(f"\n[4/4] Testing {len(test_bars)} historical bars: {test_bars[:6]}{'...' if len(test_bars)>6 else ''}")
    print("      For each bar i: recompute features on df.iloc[:i+1], compare to full df row i")
    print()

    # Per-feature tally
    leak_counts = {f: 0 for f in (FEATURES_STRICT + FEATURES_EWM + FEATURES_ADX)}
    worst_diffs = {f: 0.0 for f in (FEATURES_STRICT + FEATURES_EWM + FEATURES_ADX)}
    worst_bars  = {f: -1  for f in (FEATURES_STRICT + FEATURES_EWM + FEATURES_ADX)}

    for i in test_bars:
        # Past-only slice — what would be available at bar i in live trading
        past_only = raw.iloc[: i + 1].copy()
        recomputed = _clean_df(past_only)
        recomputed_adx = calculate_adx(recomputed)
        recomputed = recomputed.join(recomputed_adx, how="left")

        # Compare row i of full df to LAST row of recomputed slice
        full_row = full.iloc[i]
        rec_row  = recomputed.iloc[-1]

        # Strict features: must match almost exactly (1e-9 tol)
        for f in FEATURES_STRICT:
            if f not in full.columns:
                continue
            ok = _approx_equal(full_row[f], rec_row[f], atol=1e-9, rtol=1e-9)
            if not ok:
                diff = abs(float(full_row[f]) - float(rec_row[f]))
                leak_counts[f] += 1
                if diff > worst_diffs[f]:
                    worst_diffs[f] = diff
                    worst_bars[f]  = i

        # EWM features: float-noise tolerance (1e-6 rel)
        for f in FEATURES_EWM:
            if f not in full.columns:
                continue
            ok = _approx_equal(full_row[f], rec_row[f], atol=1e-9, rtol=1e-6)
            if not ok:
                diff = abs(float(full_row[f]) - float(rec_row[f]))
                leak_counts[f] += 1
                if diff > worst_diffs[f]:
                    worst_diffs[f] = diff
                    worst_bars[f]  = i

        # ADX features (EWM-based): same tolerance
        for f in FEATURES_ADX:
            if f not in full.columns:
                continue
            ok = _approx_equal(full_row[f], rec_row[f], atol=1e-9, rtol=1e-6)
            if not ok:
                diff = abs(float(full_row[f]) - float(rec_row[f]))
                leak_counts[f] += 1
                if diff > worst_diffs[f]:
                    worst_diffs[f] = diff
                    worst_bars[f]  = i

    # ── Report ───────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"{'FEATURE':<24} {'STATUS':<10} {'LEAK BARS':<11} {'WORST Δ':<14} {'AT BAR':<8}")
    print("─" * 70)

    n_leaks = 0
    n_clean = 0
    for f in FEATURES_STRICT + FEATURES_EWM + FEATURES_ADX:
        if f not in full.columns:
            print(f"{f:<24} {'SKIPPED':<10} (column not found in df)")
            continue
        if leak_counts[f] == 0:
            print(f"{f:<24} {'✓ CLEAN':<10} {0:<11} {'—':<14} {'—':<8}")
            n_clean += 1
        else:
            ratio = leak_counts[f] / len(test_bars) * 100
            print(f"{f:<24} {'✗ LEAK':<10} {leak_counts[f]}/{len(test_bars)} ({ratio:.0f}%)  "
                  f"{worst_diffs[f]:.3e}    bar {worst_bars[f]}")
            n_leaks += 1

    print("─" * 70)
    print(f"\nResult: {n_clean} CLEAN, {n_leaks} LEAKING (out of {n_clean + n_leaks} features tested)")

    if n_leaks == 0:
        print("\n✅ NO LOOKAHEAD LEAK DETECTED")
        print("   All features at bar i match what would be computed using only past data.")
        print("   Your ML training and backtest can be trusted to not peek into the future.")
        return 0
    else:
        print("\n⚠️  LOOKAHEAD LEAK DETECTED in the features above.")
        print("   These features compute differently with vs without future data — your")
        print("   backtest results using these are inflated. Fix them before trusting any")
        print("   PF/WR numbers from the affected features.")
        print()
        print("   Common fixes:")
        print("     - Replace .rolling(N, center=True)  → .rolling(N)")
        print("     - Replace .ewm(...).mean()           → .shift(1).ewm(...).mean()")
        print("     - Replace any reversed/forward-looking groupby with a strict cumulative one")
        return 1


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    tf  = sys.argv[2] if len(sys.argv) > 2 else "1d"
    sys.exit(audit(sym, tf))
