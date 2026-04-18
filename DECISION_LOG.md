# DECISION_LOG.md — All Key Decisions and Why
# When a chat goes stale, paste this log in a new chat.
# Claude reconstructs context fast from a decision log.
# Update this file EVERY time a new decision is made.

---

## Format
`[Date] Decision — Reason`

---

## Apr 13 — Core Architecture

`[Apr 13] Added sklearn imports with try/except fallback`
— so the app degrades gracefully to heuristic ML if sklearn is missing on Streamlit Cloud.

`[Apr 13] Deep fetch: _DEEP_FETCH_LIMITS = 1000 bars for all timeframes`
— Binance /api/v3/klines hard cap. Use this everywhere (backtest, WFO, ML training).

`[Apr 13] Adaptive time-decay buckets`
— n≥400→4 buckets [0.40,0.60,0.80,1.00], n≥200→3 [0.50,0.75,1.00], n≥80→2 [0.60,1.00], else 1
— Reason: newer trades carry more signal about current market regime; old trades from 2021 bull market
  can corrupt EVw on coins that have changed character.

`[Apr 13] bar_index stored per trade in trades_raw`
— Required for bucket assignment and purge logic. Always include it.

`[Apr 13] Profit Factor (PF) added to method stats`
— Was missing from initial backtest output. PF = gross_wins / gross_losses.

`[Apr 13] Two candidates: candidate_newest (best newest-bucket EVw) + candidate_weighted (best all-time EVw)`
— Reason: gives a concrete decision choice between "what's working NOW" vs "what has worked historically."

`[Apr 13] Split into 3 Steps (was 2)`
— Step 1: Backtest+WFO. Step 2: Train ML. Step 3: AI Verdict.
— Reason: ML training needs to know WHICH method to label by, which requires the user to
  see backtest results first and make a judgment.

`[Apr 13] Single ML button for both candidates (not two separate buttons)`
— Trains A and B in sequence. If A==B, trains once and mirrors.
— Reason: reduces friction; if A==B there's no point in two separate models.

`[Apr 13] ML: adaptive model selection LR/RF/GB based on n`
— LR<50, RF 50-149, GB≥150. Reason: small n → LR least likely to overfit, large n → GB best generalization.

`[Apr 13] _scanner_train_ml labels historical candles by chosen method outcome`
— NOT a fixed Aggressive/Simple baseline. Reason: the ML should learn what works for the
  specific method you'll actually trade, not a generic proxy method.

`[Apr 13] Groq default model → openai/gpt-oss-120b`
— Best free reasoning model on Groq as of Apr 2026 (verified via web search).
— reasoning_effort="medium" only for gpt-oss/qwen models (not llama).
— max_tokens bumped to 2500, timeout to 60s.

---

## Apr 14 — ML Improvements + UI Refinements

`[Apr 14] Added 4 new ML features: body_vs_atr, dist_from_ema21_pct (Improvements #3 and #4)`
— body_vs_atr: body/atr14 → measures absolute explosiveness (2% body in 0.5% ATR regime = real breakout)
— dist_from_ema21_pct: signed % distance from EMA21 → captures mean-reversion risk
— Both added to _clean_df AND extracted into sig dict for current-bar prediction.
— For SHORT trades, dist_from_ema21 sign is FLIPPED so "stretched wrong way" = negative consistently.

`[Apr 14] Added isotonic probability calibration (Improvement #2)`
— CalibratedClassifierCV(method="isotonic", cv=3) when n≥60.
— Reason: uncalibrated RF/GB are systematically overconfident. Without calibration,
  "68% probability" might really mean 53%. Calibration makes thresholds meaningful.
— Only when n≥60 to avoid overfitting the calibration itself on tiny samples.

`[Apr 14] Added time-decay sample weights (Improvement #1)`
— sample_weight = time_decay_bucket_weight (same scheme as backtest EVw).
— Reason: newer trades should pull the ML fit harder. Aligns ML with backtest's EVw metric.

`[Apr 14] Adaptive ratchet filter added to _scanner_train_ml`
— Problem: filter requiring 70% of signal's own body/vol caused sample starvation.
  A "strong" 85%/5x signal found only 4 historical analogs. An "extreme" signal found zero.
— Fix: ratchet [0.70, 0.55, 0.45, 0.35, 0.25, 0.20], target 80 samples.
— Hard floors: body≥0.20, vol≥1.10× to prevent pure-noise analogs.
— Result: ETH 4H signal improved from 4→80+ samples.

`[Apr 14] UI: Two candidate execution cards replace the old 3-zone cards in main view`
— Old: showed Aggressive/Standard/Sniper cards + EXECUTE THIS banner.
— New: Card A (🟢 newest-bucket best) + Card B (🔵 weighted all-time best) with full execution
  detail + time-decay bucket breakdown per card.
— Old 3-zone cards moved into expander "▸ View Full 3-Zone Comparison".
— Full method table moved into expander "▸ Full Method Breakdown (all 54 combinations sorted by EVw)".

`[Apr 14] AI dual-candidate verdict`
— Old: single-candidate analysis.
— New: _scanner_ai_verdict(sig, ml_a, ml_b, bt, wfo, cand_a, cand_b) → {candidate_a, candidate_b, winner}.
— When A==B: single analysis mirrored to both sides ("unanimous" flag).
— If both TRADE → AI picks stronger one. If only one TRADE → that one wins. Neither → NONE.

`[Apr 14] Added regime similarity weight to ML sample weights`
— sample_weight = time_decay_weight × regime_similarity_weight.
— Reason: a signal in GREEN regime shouldn't learn equally from RED-regime historical trades.
— Soft filter (floor 0.15) — never zeros out off-regime samples.

`[Apr 14] Soft regime filtering added to backtest EVw`
— _bucket_stats_for_trades(trades_raw, n_df, buckets, current_regime_score=None)
— When regime_score provided: each trade's contribution multiplied by _regime_similarity_weight.
— ONLY EVw/WRw are regime-filtered. Per-bucket raw stats remain unfiltered (user sees truth).

`[Apr 14] Pre-compute regime score cache in backtest`
— _bar_regime_cache computed once per backtest run before the 72-method loop.
— Reason: calling calculate_regime_score() 72× per bar = expensive. Cache it once per bar.

`[Apr 14] _REGIME_BADGE shown in provenance strip + ML card note`
— Shows current regime score and whether analogs were weighted by regime similarity.
— avg_regime_weight shown in ML note (e.g. "regime-weighted (avg w=0.78, current=83)").

---

## Apr 15 — Validation Layer + On-chain

`[Apr 15] def main(): bug fixed`
— A str_replace edit accidentally replaced def main(): with the new render_manual_analyzer_tab
  function, leaving the main body as module-level code. Streamlit loaded fine but NameError
  on the final main() call. Fixed by restoring def main():
— LESSON: always run module-load simulation (not just py_compile) after structural edits.

`[Apr 15] AI price hallucination bug fixed`
— Bug: AI was generating plausible-sounding but wrong prices (mixed between candidates).
— Fix: _compute_candidate_prices(cand, sig) as single source of truth, injected into AI prompt
  as explicit EXECUTION PRICES block with strict "copy verbatim" instruction.
— Single source of truth means UI cards and AI prompt always show identical numbers.

`[Apr 15] Grade (A+/A/B/C) made dual-candidate aware`
— Bug: grade was reading from legacy aggregate bt["win_2r"]/bt["ev_2r"].
  When Candidate A was excellent but aggregate was bad → false "C — Backtest negative".
— Fix: _scanner_setup_grade reads candidate_newest and candidate_weighted, grades by best-of.
— New "B rescue": if any candidate tradeable AND ml_pct≥60, won't drop below B.

`[Apr 15] Pulse Intel Phase 1: DefiLlama TVL module`
— No API key needed. Covers ~30 DeFi tokens + ~20 L1 chains.
— TVL 24h + 7d delta → score -10 to +10.
— Cache: 3600s (TVL updates slowly).

`[Apr 15] Pulse Intel Phase 2: Etherscan exchange flow (ERC-20 tokens)`
— Tracks large token transfers to/from known CEX hot wallets (~30 wallets, 9 exchanges).
— Net inflow → distribution pressure → bearish. Net outflow → accumulation → bullish.
— Requires free Etherscan V2 API key (etherscan.io/apis, 5 calls/sec, 100k/day).
— Cache: 900s.

`[Apr 15] Pulse Intel Phase 3: Solscan SPL flow + LunarCrush social + macro backdrop`
— Solscan: SPL token CEX flow (free Pro v2 key), 13 mints, 14 SOL CEX wallets.
— LunarCrush: Galaxy Score + sentiment + alt rank (free v4 key). Cache: 1800s.
— Macro: BTC dominance delta + stablecoin supply 7d delta (no key, free). Cache: 14400s.
— Macro provides ±3 modifier (not blended into per-token composite — applied additionally).

`[Apr 15] Pulse composite weighting`
— Flow 40%, TVL 35%, Social 25%. Missing modules → weight redistributed proportionally.
— Per-token scaled ×1.2 → ±12, then macro modifier ±3 → final ±15.

`[Apr 15] New tab: 🔍 Manual Analyzer`
— Allows analysis of any Binance ticker at any historical date/time (WIB timezone).
— No body/vol filter (analyzes any candle the user picks).
— Same 3-step pipeline (backtest → ML → AI verdict) as scanner.
— WIB → UTC conversion handled automatically.

---

## Apr 17 — Purge/Embargo + Advanced WFO

`[Apr 17] PurgedTimeSeriesSplit implemented (de Prado, AFML Ch.7)`
— Problem: sklearn TimeSeriesSplit doesn't know that labels span MAX_HOLD=20 bars.
  A training sample at fold-boundary bar i has its label resolved inside the test fold → leak.
  On 4H chart, this leak spans ~3.3 days; on daily, 20 days.
— Fix: PurgedTimeSeriesSplit class. Drops train samples whose label_end_bar overlaps test fold.
  Embargo E = ceil(0.01 × n_total) after each fold.
— Applied to: ML CV, WFO IS/OOS split.
— Every trade now carries label_end_bar = j (bar where WIN/LOSS resolved).

`[Apr 17] label_end_bar stored on every trade in trades_raw`
— backtest: label_end_bar = j (the resolution bar).
— WFO: same.
— ML training: label_end_list tracked alongside bar_idx_list for purge computation.

`[Apr 17] WFO refactored: simulates once on full df, partitions via _purge_is_oos`
— Old: separate simulation for IS and OOS.
— New: one _run_full() call, then purge/embargo logic partitions into IS and OOS.
— Returns purge_diag: {n_purged, n_embargoed, embargo_bars}.
— Tier label updated to "PURGED IS/OOS split (70%/30%, embargo 1%)".

`[Apr 17] Rolling WFO: 5 anchored windows instead of 1 cut`
— Cuts at 50/60/70/80/90% of total data.
— Reports: edge_hit_rate (fraction of windows where OOS PF ≥ 1.0), PF distribution.
— Reason: a single 70/30 cut = one point estimate. 5 windows = a distribution.
  "5/5 windows with OOS PF > 1.0" is real edge. "1 good cut" could be luck.

`[Apr 17] Bootstrap 95% CI on OOS PF`
— 1000-resample block bootstrap on OOS r_mult list.
— Returns {lo, hi} confidence interval.
— Shown in WFO card. Context: n=8 trades with PF=1.3 has CI roughly [0.7, 2.8].

`[Apr 17] Regime-conditional WFO breakdown`
— Splits OOS trades by ATR-ratio proxy into STRONG/MID/WEAK regimes.
— Reports PF/WR per regime. Reason: aggregate PF=1.4 can hide PF=2.8 in STRONG and PF=0.9 in WEAK.

`[Apr 17] WFO PASS threshold tightened`
— Old: required n_oos ≥ 3 → was declaring PASS on n=3 (statistical noise).
— New: PASS requires n_oos ≥ 8, BORDERLINE for 5-7, INSUFFICIENT for < 5. IS min = 5.

`[Apr 17] Ratchet filter added to _scanner_quick_backtest`
— Same logic as ML ratchet. Target: 50 bars. Ratios: [0.70, 0.55, 0.45, 0.35, 0.25, 0.20].
— Two-pass: cheap _count_passing() scan first, then full 72-method loop once at chosen threshold.
— Result: "good" 70%/3x ETH 4H signal improved from 18→80+ analogs.

`[Apr 17] WFO uses backtest's filter ratios`
— After backtest runs with a chosen ratchet level, WFO reads filter_min_body/filter_min_vol
  from bt_results.meta. Ensures WFO and backtest evaluate the same population of analogs.

`[Apr 17] Fill-rate / survivor-bias diagnostic added`
— Each method combo tracks: n_qualifying (signals passing filter), n_filled (actually entered zone),
  n_expired (never retraced to fill), fill_rate = n_filled/n_qualifying.
— AI prompt warns when Standard/Sniper have fill_rate < 40% on ≥20 qualifying signals.
— Context: Standard/Sniper zones on trending coins (like REZ) mostly don't fill (price never retraces).
  The backtest only sees the rare pullback+continuation subset → survivor bias inflates WR.
  fill_rate makes this visible.

`[Apr 17] Lookahead audit: 20/20 features CLEAN (confirmed)`
— Run on synthetic 500-bar data. 25 random test bars. All 22 features matched exactly.
— Deliberately injected forward-looking feature correctly flagged.
— Key confirmation: candle_rank_20 and vol_rank_20 use .rolling().rank(pct=True) — causal in pandas.

---

## Apr 17 — NEUTRAL labeling + 4th MGMT mode

`[Apr 17] NEUTRAL_R_THRESHOLD = 0.30R (Option A labeling)`
— Problem: Partial+BE trades that hit TP1 then reverse to BE produce r_mult ≈ +0.498R.
  PnL accounting correctly counts this as profitable. But labeling it WIN for ML causes:
  - ML sees only wins on trending coins → single-class collapse → can't train
  - Backtest looks invincible (PF=∞, WR=100% on REZ)
— Fix: _classify_outcome(r_mult) → WIN/LOSS/NEUTRAL. |r_mult| ≤ 0.30R = NEUTRAL.
— NEUTRAL trades: still count in PF/WR accounting. Excluded from ML training labels.
— Also: WFO now reports honest_pf (excluding NEUTRAL trades) alongside raw PF.

`[Apr 17] Partial-NoBE added as 4th management mode`
— MGMT_MODES = ["Simple", "Partial", "Partial-NoBE", "Trailing"]
— Partial-NoBE: TP 50% at 1R, keep original SL on remaining (real downside on the other half).
— Reason: user believed they were trading this style. Partial was moving SL to BE automatically
  which is a different strategy. Now both are available for comparison.
— Impact: backtest method count went from 54 → 72. 15-20% slower per Step 1 run.
— Backtest, WFO, ML all updated to handle Partial-NoBE correctly.

`[Apr 17] Pipeline sample_weight error fixed`
— Bug: CalibratedClassifierCV wrapping Pipeline raises various exception types (not just TypeError)
  when sample_weight is passed. Old code only caught TypeError → jumped to heuristic fallback.
— Fix: broadened to catch any Exception on weighted fit. Falls back to unweighted fit before
  giving up and going to heuristic. Applied to both main fit and CV loop.

`[Apr 17] Pulse Phase 3 wired into app.py Pulse tab`
— Three new API key inputs: Etherscan + LunarCrush + Solscan (password-masked, session state).
— Three new live cards: Solana Flow, Social Pulse, Macro Backdrop.
— Composite updated: ETH flow or SOL flow (whichever applies, never double-count).

---

## Decisions NOT to Build (and why)

`Wide-SL toggle for big-body candles`
— Zone validity rejection (Standard/Sniper) is mechanically correct.
  Retrace entry below SL = R:R math breaks. Cannot trade it.
  Alternative (widen SL to accommodate) changes the strategy semantics.
  Decision: leave out. Revisit after 30+ days of journal data if "zone unavailable"
  pattern costs meaningful edge.

`Cross-coin feature pooling (#10)`
— Would train one master model across all 300 coins.
— Reason skipped: ratchet fix solved most sample-starvation (liquid coins now get 80+ samples).
  Remaining problem (illiquid alts) = probably shouldn't be traded anyway if no historical analog exists.
— Revisit if: LOOSE-badge signals on coins you want to trade persist after journal data.

`Regime-conditional ML (separate models per regime)`
— Would train separate LR/RF/GB for GREEN vs YELLOW vs RED regimes.
— Reason skipped: soft regime weighting in sample_weights already addresses this with less complexity.
  Needs ≥30 samples per regime to be reliable — most coins won't hit that.
— Revisit if: regime-conditional breakdown in WFO shows extreme divergence between regimes.

`LightGBM swap for n≥200`
— Would replace GradientBoostingClassifier with LGBMClassifier for 5-10× speed + slight accuracy gain.
— Reason skipped: acceptable performance for current use case. 
  Add when training time becomes a bottleneck (currently ~8-12s per ML click).

`ICT / S&R / trendline / volume profile strategy modules`
— Decision: do NOT build additional strategies until momentum candle system has
  30+ days of live trade journal data proving edge. Building more untested systems
  on top of an unvalidated primary strategy is premature optimization.

---

## Live Trading Notes (to fill in)

```
[DATE] Started live trading with small capital (0.5-1% risk per trade)
[DATE] First 10 trades: [results]
[DATE] ML probability calibration check: 
  - When ML says >70%, actual win rate was ___% (n=___)
  - STRICT filter signals vs LOOSE filter signals: ___% vs ___%
[DATE] Regime filter impact: GREEN regime trades WR=___%, non-GREEN WR=___%
[DATE] Candidate A vs Candidate B outcomes: A wins=___, B wins=___
```

---

## Next Planned Session

Priority 1 (when journal data available):
- Review 30-day journal
- Identify specific weaknesses (e.g. "ML overpredicts when filter is RELAXED")
- Build targeted fixes, not guesswork improvements

Priority 2 (can start now in parallel):
- IDX BSJP port: fork app.py, swap Binance→yfinance(.JK), add broker summary scraper
- 2-3 focused sessions, most code reuses existing pipeline

Priority 3 (longer term):
- Meta-labeling (second classifier predicting "does this specific signal win?")
- Pulse as ML feature (add pulse_score to 11-feature vector, forward-only)
- CPCV + PBO for QUANTFLOW product launch

---

## Apr 18 — Manual Analyzer enriched to match Scanner depth

`[Apr 18] _render_enhanced_trade_plan_html(sig) helper added (line 7299)`
— Extracts the Scanner's 3-zone trade plan card into a standalone reusable function.
— Shows Aggressive / Standard / Sniper entry zones with entry/SL/TP1/2/3, zone validity
  warnings (when retrace entry < structural SL), freshness banner, and Trade Management
  Plan card with all 4 mgmt modes (Simple / Partial / Partial-NoBE / Trailing).
— Called from Manual tab immediately on "Analyze This Candle" — before Step 1.

`[Apr 18] _render_method_breakdown_table(bt_result) helper added (line 7474)`
— Extracts the Scanner's full 72-method breakdown table into a reusable function.
— Uses correct key: per_method (NOT method_results — that's a local var, not returned).
— Shows 👑 crown on best EVw row. Called from Manual tab after Step 1.

`[Apr 18] Manual tab now shows same depth as Scanner`
— After "Analyze This Candle": Enhanced Trade Plan card + 6 Intelligence Layers table
— After Step 1: Best method card, WFO verdict, Honest PF diagnostic, OOS PF 95% CI,
  Rolling WFO 5-window table, OOS by Regime breakdown, Full Method Breakdown expander
— After Step 2: ML training cards (same as Scanner)
— After Step 3: AI dual-candidate verdict with winner (same as Scanner)

`[Apr 18] Failed refactor attempt: manual_sig=None on render_auto_analyzer`
— Attempted to route Manual tab's single sig through Scanner's render block via a
  manual_sig= parameter. Hit indentation complexity with 2,150-line nested block.
— Left dead code: manual_sig=None param + _manual_render_signals short-circuit (~lines 4394-4430).
— DECISION: do NOT attempt to activate this in a single session. Either:
  Option A (recommended): continue additive approach (add missing cards one by one)
  Option B: dedicate a full session to extracting render block into _render_signals_results()

`[Apr 18] app.py grew from 7,817 → 8,264 lines (+447 lines)`
— Growth from: 2 new helper functions + Manual tab enrichment cards

`[Apr 18] Confirmed: per_method is the correct backtest return key (not method_results)`
— method_results is a local variable inside _scanner_quick_backtest. Not returned.
— bt_result["per_method"] is the dict with all 72 combos. Always use this key.

---

## Still missing in Manual tab vs Scanner (minor, future sessions)
- Zone-summary table: all 3 zones' R:R side-by-side
- "Why selected" confluence reasoning block
- Confluence Grade (A/B/C) with point breakdown
