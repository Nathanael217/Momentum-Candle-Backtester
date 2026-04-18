# CONTEXT.md — Quick Architecture Reference
# Paste this at the start of a new chat alongside app.py when a chat goes stale.
# For full detail see CLAUDE.md.

---

## System: Momentum Candle Scanner + Backtester (Binance altcoins)
**Stack**: Streamlit + pandas + sklearn + Binance API + Groq AI
**app.py**: 8,264 lines

## 3 Tabs
- **🔭 Scanner** — auto-scans 300+ USDT pairs for momentum breakout candles
- **🔍 Manual** — analyze any coin + date + direction manually
- **🫀 Pulse** — free on-chain intelligence (Nansen-lite), pulse_intel.py

## Signal Flow Per Coin
```
[Scan] _scanner_score_signal → sig dict
   ↓
[Step 1] _scanner_quick_backtest(sig)    → bt_results (72 method combos, 2 candidates)
         _scanner_mini_wfo(sig, bt)      → wfo_results (rolling 5-window, purged IS/OOS)
   ↓
[Step 2] _scanner_train_ml(sig, method) → ml_a, ml_b  (adaptive: LR/RF/GB + calibration)
   ↓
[Step 3] _scanner_ai_verdict(...)        → dual-candidate AI analysis + winner pick
```

## Two Candidates System
- **Candidate A** = `bt["candidate_newest"]`  — best method in NEWEST time-decay bucket
- **Candidate B** = `bt["candidate_weighted"]` — best method by weighted all-time EVw
- `best_key` selected by `ev_weighted` (NOT raw `ev`)
- When A == B (same method): shown as "unanimous", ML trained once

## 4 Management Modes (MGMT_MODES)
1. `Simple` — full size, hold to TP2 or original SL
2. `Partial` — TP 50% at 1R + auto-move SL to breakeven
3. `Partial-NoBE` — TP 50% at 1R + keep original SL (real downside remains)
4. `Trailing` — full size, BE at 1R, then trail 0.5×ATR from close

## Backtest Constants
```python
ENTRY_ZONES = ["Aggressive" (0%), "Standard" (38.2% retrace), "Sniper" (61.8% retrace)]
TP_MULTS    = [2.0, 2.5, 3.0]    → 72 combinations total
MAX_HOLD    = 20 bars
FIXED_SL    = 1.5%
_DEEP_FETCH_LIMITS = 1000 bars (all timeframes)
NEUTRAL_R_THRESHOLD = 0.30R      (|r_mult| ≤ 0.30 → NEUTRAL, excluded from ML)
```

## ML Constants
```python
n < 50   → Logistic Regression (StandardScaler pipeline)
50-149   → Random Forest (n_estimators=150, max_depth=5)
≥ 150    → Gradient Boosting (n_estimators=150, max_depth=3, lr=0.05)
n ≥ 60   → +CalibratedClassifierCV(isotonic, cv=3)
Features (11): body_pct, vol_mult, adx, di_gap, atr_ratio, ema_score,
               regime_score, candle_rank, vol_rank, body_vs_atr, dist_from_ema21
CV: PurgedTimeSeriesSplit(n_splits=min(5, n//15), embargo_pct=0.01)
Sample weights: time_decay_bucket_weight × regime_similarity_weight
```

## Ratchet Filter (both backtest and ML)
Both systems start at 70% of signal's body/vol, relax until enough samples found:
`[0.70, 0.55, 0.45, 0.35, 0.25, 0.20]`
Backtest target: 50 bars. ML target: 80 samples.
Hard floors: body ≥ 0.20, vol ≥ 1.10× regardless of ratchet level.

## Time-Decay Buckets (adaptive)
```
n ≥ 400 → 4 buckets: [0.40, 0.60, 0.80, 1.00]   oldest→newest
n ≥ 200 → 3 buckets: [0.50, 0.75, 1.00]
n ≥ 80  → 2 buckets: [0.60, 1.00]
n < 80  → 1 bucket:  [1.00]
```

## Soft Regime Filter
`weight = max(0.15, 1 - abs(current_score - historical_score) / 100)`
Applied to: EVw calculation, ML sample_weights. NOT applied to WFO.

## Purge + Embargo (de Prado Ch.7)
- Every trade stores `bar_index` AND `label_end_bar`
- `PurgedTimeSeriesSplit` drops train samples whose label overlaps test fold
- Embargo = `ceil(0.01 × total_bars)` after each fold
- Applied to ML CV and WFO IS/OOS split

## Zone Validity
Standard/Sniper become invalid when retrace entry < structural SL.
This is CORRECT — not a bug. Do not "fix" by widening SL.

## AI Verdict
- Default model: `openai/gpt-oss-120b` (best free reasoning on Groq)
- Prices come from `_compute_candidate_prices(cand, sig)` — single source of truth
- AI must copy prices verbatim (hardcoded in prompt — prevents hallucination)

## Lookahead Audit
`python lookahead_audit.py BTCUSDT 1d`
20/20 features CLEAN. Safe to trust backtest + ML results.

## Pulse Intel Composite
```
Flow (ETH/SOL): 40%  |  TVL (DefiLlama): 35%  |  Social (LunarCrush): 25%
Per-token score → ×1.2 → ±12, plus macro modifier ±3 → final ±15
Verdicts: ≥10 STRONGLY BULLISH, ≥4 BULLISH, ≥-3 NEUTRAL, ≥-9 BEARISH, else STRONGLY BEARISH
```

## New helpers (Apr 18)
- `_render_enhanced_trade_plan_html(sig)` → line 7299, reusable 3-zone card HTML
- `_render_method_breakdown_table(bt_result)` → line 7474, reusable 72-method table HTML
- `render_auto_analyzer` has unused `manual_sig=None` param — dead code, do not activate

## Current Issues / What I'm Working On
- Manual tab minor gaps vs Scanner: zone R:R summary table, confluence grade, "why selected" block
- Dead code cleanup: manual_sig param + _manual_render_signals block in render_auto_analyzer

## Rules Already Decided
- Always PurgedTimeSeriesSplit, never sklearn TimeSeriesSplit
- embargo_pct = 0.01
- EMA uses .shift(1) — no exceptions
- best_key by ev_weighted, not raw ev
- Zone validity rejection is intentional — no wide-SL workaround
- Partial-NoBE is the 4th mgmt mode (added Apr 17)
- NEUTRAL_R_THRESHOLD = 0.30R (Option A labeling)
- WFO intentionally NOT regime-filtered
- per_method is the correct backtest return key (not method_results)
