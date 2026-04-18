# CLAUDE.md — AutoFinder + Pulse Intel Project Brain
# Load this at the start of EVERY Claude Code session.
# This file is the single source of truth for the project.

---

## What this system is

A Streamlit-based **momentum candle** backtester + scanner for Binance altcoins.
Three tabs, three concerns:

| Tab | File | Purpose |
|-----|------|---------|
| 🔭 Scanner | app.py | Auto-scans 300+ USDT pairs for momentum breakout candles |
| 🔍 Manual | app.py | Analyze any coin + any date + direction — full pipeline |
| 🫀 Pulse | pulse_intel.py | Free on-chain intelligence (Nansen-lite) |

Support files:
- `lookahead_audit.py` — run `python lookahead_audit.py BTCUSDT 1d` to prove no feature leak
- `requirements.txt` — includes plotly, scikit-learn, streamlit

---

## File sizes (reference)
- `app.py` — 8,264 lines (grew from 7,817 — Manual tab enrichment)
- `pulse_intel.py` — 1,682 lines
- `lookahead_audit.py` — 290 lines

---

## Architecture: all top-level functions in app.py

```
# ── Data layer ──────────────────────────────────────────────────────────────
_clean_df(df)                          → 22 features (all causal, no leak)
calculate_adx(df, period=14)           → di_plus, di_minus, adx
calculate_ema(df, period)              → EMA series with shift(1)
_scanner_fetch_candles(symbol, interval, limit)
_binance_klines / _gateio_klines / _binance_fetch / fetch_live

# ── Key constants ────────────────────────────────────────────────────────────
_BINANCE_INTERVAL = {"1D":"1d", "4H":"4h", "1H":"1h", "1W":"1w"}
_DEEP_FETCH_LIMITS = {"1h":1000, "2h":1000, "4h":1000, "6h":1000, "12h":1000, "1d":1000}
NEUTRAL_R_THRESHOLD = 0.30            # ±0.30R band → NEUTRAL (excluded from ML labels)
MAX_HOLD = 20                         # max bars per trade (used everywhere)

# ── Validation layer (de Prado Ch.7) ────────────────────────────────────────
PurgedTimeSeriesSplit                  → class, replaces sklearn TimeSeriesSplit
_purge_is_oos(trades, is_end_bar, total_bars, embargo_pct=0.01)

# ── Helpers ──────────────────────────────────────────────────────────────────
_classify_outcome(r_mult)              → "WIN" / "LOSS" / "NEUTRAL"
_deep_limit_for(timeframe)             → int (Binance bar limit for timeframe)
_compute_decay_buckets(n_df)           → dict {count, weights, edges, labels}
_bucket_stats_for_trades(trades_raw, n_df, buckets, current_regime_score=None)
_regime_similarity_weight(current, historical)  → float 0.15–1.00
_compute_candidate_prices(cand, sig)   → dict with entry/SL/TP1/TP2 prices
_compute_enhanced_trade_plan(...)      → full trade plan dict
_classify_outcome(r_mult)             → WIN / LOSS / NEUTRAL

# ── Regime ──────────────────────────────────────────────────────────────────
calculate_regime_score(df, bar_index, direction, adx_df, timeframe, ticker)
fetch_fear_greed() / fetch_btc_dominance() / fetch_funding_rate() / fetch_open_interest()

# ── Scanner core ─────────────────────────────────────────────────────────────
_scanner_get_universe(min_volume_usdt)
_scanner_score_signal(df, adx_df, bar_idx, direction, timeframe, symbol, min_body_pct, min_vol_mult)
_scan_one_symbol(args)                 → list of sig dicts (scans last 3 closed candles)

# ── Main pipeline (run in order) ─────────────────────────────────────────────
_scanner_quick_backtest(sig)           → bt_results
_scanner_mini_wfo(sig, bt_results)     → wfo_results
_scanner_heuristic_ml(sig)             → ml fallback (no training needed)
_scanner_train_ml(sig, method_cfg)     → ml_results (real trained model)
_scanner_ai_verdict(sig, ml_a, ml_b, bt, wfo, cand_a, cand_b)  → dual-candidate verdict
_scanner_setup_grade(sig, ml, bt)      → (grade, color, description)

# ── UI render functions ──────────────────────────────────────────────────────
_render_enhanced_trade_plan_html(sig)              → HTML string: 3-zone card (line 7299)
_render_method_breakdown_table(bt_result)          → HTML string: 72-method table (line 7474)
render_auto_analyzer(ticker, df_full_1d, tc, current_tf, manual_sig=None)  → Scanner tab
render_pulse_tab()                                 → Pulse tab
render_manual_analyzer_tab()                       → Manual tab (now matches Scanner depth)
main()                                             → tab setup + session state
```

---

## The 3-Step Signal Flow (per coin, per signal)

```
Step 1 (button) → _scanner_quick_backtest(sig)
                  _scanner_mini_wfo(sig, bt_results)
                  → Candidate A (best newest-bucket EVw) stored in bt["candidate_newest"]
                  → Candidate B (best weighted all-time EVw) stored in bt["candidate_weighted"]

Step 2 (button) → If A == B: one _scanner_train_ml call (unanimous)
                  If A != B: two _scanner_train_ml calls (one per candidate)
                  → ml_a and ml_b stored in session state

Step 3 (button) → _scanner_ai_verdict(sig, ml_a, ml_b, bt, wfo, cand_a, cand_b)
                  → dual-verdict dict: {candidate_a, candidate_b, winner, winner_rationale}
```

---

## Backtest method space (72 combinations)

```
ENTRY_ZONES  = ["Aggressive", "Standard", "Sniper"]         # 3
SL_METHODS   = ["Fixed SL", "ATR SL"]                       # 2
MGMT_MODES   = ["Simple", "Partial", "Partial-NoBE", "Trailing"]  # 4
TP_MULTS     = [2.0, 2.5, 3.0]                              # 3
→ 3 × 2 × 4 × 3 = 72 combinations
```

**Zone validity rule**: Standard/Sniper become structurally invalid when the retrace entry
price falls below the structural SL. When this happens the zone is added to `_invalid_zones`
and excluded from candidate selection — this is CORRECT behavior, not a bug.
The retrace entry math: `entry = close - retrace_frac × body`. On a 100%-body candle with
large ATR the 38.2%/61.8% retrace can land below the SL. Cannot be fixed by widening SL
(design decision: keep structural SL purity).

**MGMT mode behavior**:
- `Simple`: full size, hold to TP2 or original SL
- `Partial`: TP 50% at 1R + auto-move SL to BE on remaining
- `Partial-NoBE`: TP 50% at 1R + keep original SL (real downside remains)
- `Trailing`: full size, BE at 1R, then trail 0.5×ATR from close

**NEUTRAL label problem (Partial)**: Partial+BE trades that hit TP1 then reverse to BE
produce r_mult ≈ +0.498R → labeled WIN by PnL but NEUTRAL by ML (|r_mult| ≤ 0.30R).
This was causing 100% WR on trending coins (e.g. REZ). `_classify_outcome()` fixes this.

---

## Adaptive ratchet filter (both backtest and ML)

When a high-volatility signal is scanned, requiring analogs with 70% of its own
body/vol creates a sample starvation problem. Both systems relax progressively:

**Backtest ratchet** (target: 50 bars):
```python
_BT_RATCHET_RATIOS = [0.70, 0.55, 0.45, 0.35, 0.25, 0.20]
_BT_MIN_BODY_FLOOR = 0.20
_BT_MIN_VOL_FLOOR  = 1.10
```

**ML ratchet** (target: 80 samples):
```python
_RATCHET_RATIOS = [0.70, 0.55, 0.45, 0.35, 0.25, 0.20]
_MIN_BODY_FLOOR = 0.20
_MIN_VOL_FLOOR  = 1.10
```

UI badge: STRICT 70% (green) → RELAXED 45% (yellow) → LOOSE 20% (orange).
A LOOSE badge means analogs are broad — treat ML probability as directional only.

---

## ML model selection

```python
if n_samples < 50:
    model = Pipeline([StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced")])
elif n_samples < 150:
    model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=5)
else:
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, lr=0.05, subsample=0.8)

# Wrap with isotonic calibration when n >= 60
if n_samples >= 60:
    model = CalibratedClassifierCV(model, method="isotonic", cv=3)
```

**11 features** (as of current code):
```
["body_pct", "vol_mult", "adx", "di_gap", "atr_ratio",
 "ema_score", "regime_score", "candle_rank", "vol_rank",
 "body_vs_atr", "dist_from_ema21"]
```

**CV**: `PurgedTimeSeriesSplit(n_splits=min(5, n//15), embargo_pct=0.01)` — NOT sklearn default

**Sample weights**: `time_decay_bucket_weight × regime_similarity_weight`

**Minimum samples**: n < 20 → heuristic fallback

---

## Time-decay bucket scheme

```python
n_df >= 400  → 4 buckets, weights [0.40, 0.60, 0.80, 1.00]  (oldest → newest)
n_df >= 200  → 3 buckets, weights [0.50, 0.75, 1.00]
n_df >= 80   → 2 buckets, weights [0.60, 1.00]
n_df < 80    → 1 bucket,  weights [1.00]
```

`EVw` = weighted EV using these weights × regime_similarity_weight.
`best_key` is now selected by `ev_weighted` (NOT raw `ev`) — this was a bug that got fixed.

---

## Soft regime filtering

```python
def _regime_similarity_weight(current, historical):
    return max(0.15, 1 - abs(current - historical) / 100)
```

Applied to BOTH backtest `EVw`/`WRw` AND ML `sample_weights`.
Floor = 0.15 (never zero — prevents sample-size cliff on illiquid coins).
WFO is intentionally NOT regime-filtered (WFO tests generalization, not regime-match).

---

## Purge + Embargo (de Prado Ch.7)

Every trade stores `bar_index` (entry) and `label_end_bar` (= `j`, the bar where WIN/LOSS resolved).
`PurgedTimeSeriesSplit` drops training samples whose label period overlaps the test fold.
Embargo = `ceil(0.01 × total_bars)` after each test fold.
Applied to: ML CV, WFO IS/OOS split.
**NOT** in sklearn's default `TimeSeriesSplit` — that's why we replaced it.

---

## WFO return dict (key fields)

```python
{
  "ok": bool,
  "verdict": "PASS" | "BORDERLINE" | "FAIL" | "INSUFFICIENT",
  "is_pf": float, "oos_pf": float, "oos_wr": float,
  "is_pf_clean": float, "oos_pf_clean": float,   # honest PF excluding NEUTRAL trades
  "oos_n": int,
  "purge_diag": {n_purged, n_embargoed, embargo_bars},
  "label_diag": {n_neutral, raw_pf, honest_pf, pf_inflation_pct},
  "oos_pf_ci": {"lo": float, "hi": float},        # bootstrap 95% CI
  "rolling_wfo": {"ok": bool, "edge_hit_rate": float, "windows": [...]},
  "regime_breakdown": {"STRONG": {...}, "MID": {...}, "WEAK": {...}},
  "method_used": str,
  "tier_label": "PURGED IS/OOS split (70%/30%, embargo 1%)",
}
```

WFO PASS requires `n_oos >= 8`. BORDERLINE for 5-7. INSUFFICIENT for < 5.

---

## Backtest return dict (key fields)

```python
{
  "best_key": str,           # key into method_results with highest ev_weighted
  "best": dict,              # best method stats
  "candidate_newest":  dict, # best method in newest bucket — Candidate A
  "candidate_weighted": dict,# best method by decay-weighted EVw — Candidate B
  "method_results": dict,    # all 72 combos
  "meta": {
    "bars_used": int, "bars_requested": int,
    "coverage_start": str, "coverage_end": str,
    "bucket_count": int, "bucket_scheme": str,
    "filter_ratio": float,   # ratchet level used (0.20–0.70)
    "filter_min_body": float, "filter_min_vol": float,
    "regime_score": float,   # current signal regime used for soft filter
  },
}
```

Per-method entry (in `method_results[key]`):
```python
{
  "zone", "sl_label", "mgmt", "tp_mult",
  "n", "win_rate", "ev", "ev_weighted", "wr_weighted", "pf",
  "buckets": [{"label", "weight", "n", "wr", "ev"}],  # time-decay breakdown
  "newest_bucket": dict,
  "n_qualifying": int,   # signals that passed the filter (before zone fill)
  "n_filled": int,       # trades that actually entered (fill_rate = n_filled/n_qualifying)
  "fill_rate": float,    # IMPORTANT: low fill_rate on Standard/Sniper = survivor bias
  "outcome_class": str,  # WIN / LOSS / NEUTRAL per trade
}
```

---

## AI verdict (Groq)

```python
# Default model (sidebar selectable):
"openai/gpt-oss-120b"   # strongest free reasoning, default
"openai/gpt-oss-20b"    # faster
"qwen/qwen3-32b"
"llama-3.3-70b-versatile"
"meta-llama/llama-4-scout-17b-16e-instruct"

# reasoning_effort = "medium" only for gpt-oss and qwen models
# max_tokens = 2500  (reasoning models need token budget)
# timeout = 60s
```

AI receives canonical prices from `_compute_candidate_prices(cand, sig)` — the single source
of truth. The AI MUST copy prices verbatim (strict instruction in prompt to prevent hallucination).

---

## Lookahead audit (confirmed 20/20 CLEAN)

Run: `python lookahead_audit.py BTCUSDT 1d`

All 22 features confirmed causal (no forward-looking computation):
- EMA uses `.shift(1).ewm()` — strictly past
- `vol_avg_7` uses `.shift(1).rolling(7).mean()` — strictly past
- `candle_rank_20`, `vol_rank_20` use `.rolling().rank(pct=True)` — causal in pandas
- `body_vs_atr`, `dist_from_ema21_pct` derived from causal features

---

## Pulse Intel module (pulse_intel.py)

```python
get_pulse_intel(symbol, etherscan_api_key, lunarcrush_api_key, solscan_api_key)
  → composite_score (-15 to +15)
  → composite_label: STRONGLY BULLISH / BULLISH / NEUTRAL / BEARISH / STRONGLY BEARISH
  → tvl, exchange_flow (ETH), solana_flow (SOL), social, macro sub-dicts

# Composite weights (per-token, before ±3 macro modifier):
  Flow (ETH or SOL, whichever applies): 40%
  TVL (DefiLlama):                      35%
  Social (LunarCrush):                  25%
# Missing modules: weight redistributed proportionally
# Per-token scaled to ±12, then macro modifier (±3) → final ±15
```

API keys needed: Etherscan (free, etherscan.io/apis), LunarCrush (free), Solscan (free Pro v2).
DefiLlama and macro backdrop: NO key needed.
Cache TTLs: TVL=3600s, flow=900s, social=1800s, macro=14400s.

---

## Known issues / things NOT to break

1. **Do not use sklearn's TimeSeriesSplit** — always `PurgedTimeSeriesSplit`
2. **Do not change `best_key` to use raw `ev`** — it must use `ev_weighted` (bug was fixed)
3. **Do not remove `label_end_bar` from trades_raw** — purge logic depends on it
4. **Do not remove the ratchet** — without it, high-vol signals get 0-6 training samples
5. **Do not remove NEUTRAL classification** — it prevents single-class ML collapse on trending coins
6. **Zone validity (Standard/Sniper)** is mechanically correct — do NOT try to "fix" by widening SL
7. **WFO is NOT regime-filtered** by design — keep it that way
8. **`_compute_candidate_prices` is the single source of truth** for prices in AI prompt and UI cards
9. **`render_auto_analyzer` has dead code** — `manual_sig=None` parameter and
   `_manual_render_signals` short-circuit block (lines ~4394-4430) are unused leftovers
   from a failed refactor attempt. Harmless but should be cleaned up. Do NOT activate
   them without a full session dedicated to that refactor.
10. **`main()` must be defined** — a previous edit accidentally removed the `def main():` line causing NameError on deploy. Always verify `def main():` exists before shipping.

---

## Current working status

- Lookahead audit: ✅ 20/20 CLEAN (synthetic test + confirmed clean)
- Streamlit Cloud deployment: ✅ working (requirements.txt present)
- `def main():` bug: ✅ fixed
- best_key EVw bug: ✅ fixed (was selecting by raw `ev`, now `ev_weighted`)
- Partial+BE 100% WR artifact: ✅ fixed via NEUTRAL labeling + `_classify_outcome`
- Pipeline sample_weight error (Candidate B heuristic fallback): ✅ fixed (broad except)
- Purge + embargo: ✅ implemented (PurgedTimeSeriesSplit, label_end_bar)
- Soft regime filtering: ✅ implemented (_regime_similarity_weight)
- Rolling WFO + bootstrap CI + regime breakdown: ✅ implemented
- Fill-rate / survivor-bias diagnostic: ✅ implemented (n_qualifying, fill_rate)
- 4 MGMT modes (added Partial-NoBE): ✅ implemented
- NEUTRAL label option A: ✅ implemented
- Manual Analyzer enriched to match Scanner depth: ✅ shipped (Apr 18)
  - `_render_enhanced_trade_plan_html(sig)` — reusable 3-zone card helper
  - `_render_method_breakdown_table(bt_result)` — reusable 72-method table helper
  - Manual tab now shows: Enhanced Trade Plan, 6 Intelligence Layers table,
    Best method card, WFO verdict, Honest PF, OOS CI, Rolling WFO, Regime breakdown,
    Full Method Breakdown expander (👑 on best EVw row), ML cards, AI verdict
  - `render_auto_analyzer` has unused `manual_sig=None` param + dead short-circuit block
    (partial refactor attempt, harmless, clean up in future session)

---

## What's next (backlog, not yet built)

1. **Clean up dead code in render_auto_analyzer** — remove unused `manual_sig=None` param
   and `_manual_render_signals` short-circuit block (left from failed refactor attempt)
2. **Missing Manual tab vs Scanner (minor gaps)**:
   - Zone-summary table with all 3 zones' R:R stats side-by-side
   - "Why this coin was selected" confluence reasoning section
   - Confluence Grade (A/B/C) with point count breakdown
3. **Meta-labeling** — second classifier predicting "does this specific signal win?"
   Features: regime, F&G, onchain composite, funding rate, BTC.D delta
4. **Pulse as ML feature** — add pulse_score to the 11-feature vector (forward-only)
5. **CPCV + PBO** — for QUANTFLOW product launch proof-of-edge
6. **IDX BSJP port** — fork app.py, swap Binance → yfinance (.JK suffix), add broker summary
7. **Wide-SL toggle** — allow Standard/Sniper on big-body candles with widened SL
   (DECISION: skipped for now — mechanically correct to reject; revisit after journal data)
8. **Cross-coin feature pooling** — train master model across all coins (skipped: ratchet fix
   solved most sample-starvation; revisit if illiquid alts still starve)
