# SKILL.md — Trading Domain Knowledge + System Expertise
# Claude Code should read this before making any change touching:
# - signal scoring, backtest logic, ML features, regime scoring, trade management
# Reference this when brainstorming improvements or debugging edge cases.

---

## The Strategy: Momentum Candle

**Core thesis**: A candle with unusually large body (conviction) + unusually high volume
(participation) in a trending direction = an impulsive move with statistical follow-through.
Enter at or after that candle closes, ride the continuation.

This is NOT a mean-reversion strategy. It is NOT a support/resistance play. It does NOT
look for patterns across multiple candles (single-bar detection). It is NOT ICT, SMC, or
harmonic patterns.

**What makes a qualifying candle**:
1. `|body_pct|` above threshold (large body = conviction, body is most of candle range)
2. `vol_mult` above threshold (volume spike = unusual participation)
3. Direction alignment (long = bullish body, short = bearish body)

---

## Features and What They Mean

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `body_pct` | body / candle_range | Conviction (how much of the range is body) |
| `vol_mult` | volume / vol_avg_7_shift1 | Volume surge vs recent baseline (shift(1) = no leak) |
| `atr14` | rolling(14).mean() of TR | Current volatility |
| `atr_ratio` | atr14 / atr14.rolling(20).mean() | Volatility expansion (1.4x = expanding, good for momentum) |
| `ema5/15/21` | shift(1).ewm() | Trend structure (shift(1) = strictly past, no leak) |
| `ema_full` | ema5>ema15>ema21 (long) or reverse (short) | Full EMA stack = strong trend |
| `ema_partial` | partial alignment | Developing trend |
| `candle_rank_20` | rolling(20).rank(pct=True) of \|body_pct\| | Percentile size vs recent bars |
| `vol_rank_20` | rolling(20).rank(pct=True) of volume | Percentile volume vs recent bars |
| `body_vs_atr` | \|body\| / atr14 | Absolute explosiveness (2% body in 0.5% ATR = massive) |
| `dist_from_ema21_pct` | (close - ema21) / ema21 × 100 | Mean-reversion risk; flip sign for shorts |
| `adx` | EWM-based directional movement | Trend strength (>25 = trending, >40 = strong trend) |
| `di_plus` | DI+ from ADX calculation | Upward directional pressure |
| `di_minus` | DI- from ADX calculation | Downward directional pressure |
| `vol_delta_5/20` | sum of volume × (2×close_position-1) | Buying vs selling pressure proxy |
| `vol_delta_regime` | (vol_delta_5 - 20bar_mean) / 20bar_std | Normalized flow |

**key insight on body_vs_atr vs body_pct**:
- `body_pct` = 0.85 tells you the candle is 85% body. But a 2% body candle in a 0.5% ATR
  environment and a 2% body in a 3% ATR environment look identical by body_pct.
- `body_vs_atr` tells the model: first example = 4x the typical range (massive), second = 0.67x (noise).
- This is the single most important feature for distinguishing "real breakout" from "chop candle".

**key insight on dist_from_ema21**:
- A long signal +8% above EMA21 = buying a stretched market (mean-reversion risk).
- Same signal at +1% above EMA21 = buying early strength.
- For shorts: flip the sign so "stretched in wrong direction" = negative consistently.
- The model learns per-coin thresholds automatically from historical win/loss data.

---

## Entry Zones

| Zone | Entry Type | Retrace | Expiry |
|------|-----------|---------|--------|
| Aggressive | Enter at close of signal candle | 0% | No expiry |
| Standard | Wait for 38.2% retrace into body | 38.2% | 3 bars |
| Sniper | Wait for 61.8% retrace into body | 61.8% | 3 bars |

**Zone validity rule**: If the retrace entry price < structural SL, the zone is invalid.
This happens on big-body candles with large ATR (SL is below the candle low, retrace
is deep enough to land below that SL). The system correctly flags these as unavailable.
This is NOT a bug — the R:R math literally breaks if entry < SL on a long.

**Fill-rate / survivor bias**: Standard/Sniper on strongly trending coins often
never fill because price never retraces. The backtest only sees the rare pullback-then-
continuation subset → inflated WR. Always check `fill_rate` in the method table.
Low fill_rate (<40%) on Standard/Sniper = probably a survivor-bias artifact.

---

## Stop Loss Methods

| SL Method | Level | Logic |
|-----------|-------|-------|
| Fixed SL | 1.5% below entry | Simple, universal |
| ATR SL | low - 0.5×ATR14 | Structural (below candle low + buffer), clamped to [0.8%, 6.0%] |

ATR SL is always clamped: `max(0.008, min(0.06, (entry - sl_px) / entry))`.
The 6% cap prevents absurd stops on volatile coins. The 0.8% floor prevents stops
so tight they get hunted by normal noise.

---

## Trade Management Modes

| Mode | TP1 action | SL after TP1 | Best for |
|------|-----------|-------------|---------|
| Simple | — | Original SL | Quick reversal markets |
| Partial | Take 50% off at 1R | Move to breakeven | Mean-revert markets |
| Partial-NoBE | Take 50% off at 1R | Keep original SL | "Let it breathe" |
| Trailing | — | Move to BE, then trail 0.5×ATR | Strong trends |

**Critical point about Partial+BE**: Once TP1 hits, the remaining 50% can only:
- Hit TP2 (full win, r_mult ≈ +1.498R)
- Hit BE stop (partial win, r_mult ≈ +0.498R)
- Neither → time out at close (usually positive or neutral)

This means a Partial trade that hits 1R can NEVER produce a real loss. On trending coins,
this creates 100% WR and PF=∞ — looks invincible, but it's an accounting artifact.
`NEUTRAL_R_THRESHOLD = 0.30R` catches the +0.498R breakeven trades as NEUTRAL
so ML isn't contaminated by single-class labels.

**Trailing vs Partial comparison**:
- Trailing DOES keep full size (no partial exit)
- Trailing DOES move SL to BE at 1R (same as Partial on that point)
- Trailing does NOT take partial profits
- Trailing then trails at 0.5×ATR from close → can capture much larger moves
- Trailing r_mult range: -0.002R (trail back to BE exactly) to unlimited upside

---

## Regime Score (0–100)

Regime score integrates multiple market context signals:

| Score | Label | Color |
|-------|-------|-------|
| ≥67 | GREEN | Trending/bullish |
| 50-66 | YELLOW | Mixed/developing |
| <50 | RED | Choppy/bearish |

**Components** (approximate): ADX trend strength, EMA alignment, ATR regime, 
BTC dominance, Fear & Greed index, funding rate.

**Soft regime filter**: Instead of hard-filtering analogs to current regime only,
weight = max(0.15, 1 - abs(current - historical) / 100).
This prevents sample-size cliff on illiquid coins while still prioritizing
same-regime analogs in EVw and ML training.
WFO is intentionally NOT regime-filtered (WFO tests generalization).

---

## R-multiple Math

```
risk (R) = entry - SL  (for longs; SL - entry for shorts)

Simple/Trailing: r_mult = (exit - entry) / risk - fees
Partial: at TP1: r_mult_partial = 1.0 * 0.5 + r_mult_remainder * 0.5 - fees
         where r_mult_remainder = (exit - entry) / risk  (with potentially moved SL)

Fees: 0.002 (0.2%) deducted from all r_mult calculations
MAX_HOLD: 20 bars (if no TP/SL hit, exit at close of bar 20)
```

---

## Validation Layer (de Prado Ch.7)

**Why standard TimeSeriesSplit leaks**:
- Every trade label spans MAX_HOLD=20 bars (entry at bar i, resolution at bar j ≤ i+20)
- sklearn's TSS puts a hard fold boundary but doesn't know about label overlap
- A training sample at fold-boundary bar i has its outcome determined by bars inside the test fold
- This inflates CV accuracy by ~10-30% depending on fold size vs MAX_HOLD

**PurgedTimeSeriesSplit**:
- For each test fold [t_min, t_max]: keep training sample only if label_end_bar < t_min
  OR entry_bar > test_label_max + embargo_E
- Embargo E = ceil(0.01 × total_bars) — kills serial autocorrelation at fold boundary
- De Prado's standard choice: 1% of total bars

**Applied to**:
1. ML CV (PurgedTimeSeriesSplit(n_splits=min(5, n//15), embargo_pct=0.01))
2. WFO IS/OOS split (_purge_is_oos helper function)

**Data stored per trade** (required for purge):
```python
trades_raw.append({
    "bar_index": i,        # entry bar
    "label_end_bar": j,    # resolution bar (where WIN/LOSS determined)
    "r_mult": float,       # actual r-multiple
    "outcome_class": str,  # WIN / LOSS / NEUTRAL
    "regime_score": float, # for soft regime filter
    ...
})
```

---

## ML Probability Calibration

**Problem**: Uncalibrated RF/GB are systematically overconfident.
When RF says "80%", the actual historical win rate in that probability bin is often 62-68%.
This makes the probability thresholds in your decision rules meaningless.

**Fix**: `CalibratedClassifierCV(method="isotonic", cv=3)` when n_samples ≥ 60.
Isotonic regression fits a monotonically increasing step function from raw score → true probability.
After calibration, "68% probability" actually means ~68% of similar historical setups won.

**Effect you'll observe**:
- Probabilities pull toward middle (less extreme values, more honest)
- High-conviction setups still show 70-80%+ but it means more
- The WAIT threshold (e.g. ML < 45% = lean WAIT) triggers at the right time

**When calibration is NOT applied** (n < 60):
- LR doesn't need it (sigmoid output is already calibrated by construction)
- Small samples would cause the calibration to overfit
- Raw probabilities are used with a note in the ML card

---

## Selection Bias Warning: fill_rate

Standard and Sniper zones have `expiry_bars = 3`. If price doesn't retrace to the
entry zone within 3 bars, the trade is DROPPED from the backtest dataset (EXPIRED).

On strongly trending coins (like REZ, HIGH at 100% body), most signals just blast
through without retracing. The backtest only sees the rare pullback-then-continuation
subset → those trades continue almost by definition → inflated WR.

**How to detect**: check `fill_rate` in the method breakdown table.
- fill_rate = n_filled / n_qualifying × 100%
- If Standard/Sniper show fill_rate < 40% on ≥20 qualifying signals → survivor bias
- The AI prompt now warns on this automatically

**What it means for you**: a "Standard zone WR=65%" with fill_rate=15% is not really
"65% of the time I'd make money entering at the 38.2% retrace." It means "of the rare
cases where price pulled back 38.2% on this coin, 65% continued." Very different edge.

---

## Pulse Intel: What Each Module Measures

| Module | API | What it measures | Cache TTL |
|--------|-----|-----------------|----------|
| TVL | DefiLlama (free) | Protocol/chain total value locked 7d delta | 3600s |
| ETH Flow | Etherscan V2 (free key) | Net token transfer to/from CEX hot wallets (24h) | 900s |
| SOL Flow | Solscan Pro v2 (free key) | Same for Solana SPL tokens | 900s |
| Social | LunarCrush v4 (free key) | Galaxy Score, sentiment, mention trend | 1800s |
| Macro | DefiLlama + CoinGecko (free) | Stablecoin supply 7d delta + BTC dominance proxy | 14400s |

**Interpretation**:
- CEX outflow = whales withdrawing from exchanges → likely HODLing → bullish
- CEX inflow = whales depositing to exchanges → likely selling → bearish
- TVL rising = real demand for the protocol → fundamental buying pressure
- Macro positive = stablecoin supply growing (dry powder entering) + BTC.D falling (alt season)

**Important caveat**: Pulse is CONFLUENCE, not a trade signal on its own.
Use it to confirm or question what the scanner found. Don't trade based on Pulse alone.

---

## IDX BSJP Strategy (planned, not yet built)

**BSJP = "Beli Sore Jual Pagi"** (Buy Afternoon, Sell Morning)
- Entry: last 15-30 minutes of IDX Session 2 (14:45-15:00 WIB)
- Exit: first 15-30 minutes of next day Session 1 (09:00-09:15 WIB)
- Total screen time: ~30 minutes/day
- Thesis: same momentum candle thesis, but exploiting overnight continuation in IDX
  (a retail-dominated, less efficient market than BTC/ETH)

**Data source**: yfinance with .JK suffix (BBCA.JK, BBRI.JK, TLKM.JK, etc.)
- Supports `period="max"` → 10-15+ years of daily OHLCV for major IDX stocks
- No API key needed
- Limitation: daily only (no intraday below daily without premium)

**IDX-specific features to add** (not in current system):
1. Broker summary (net buy/sell per sekuritas from IDX website) — the "smart money" signal
2. Foreign net buy/sell (IDX publishes daily) — institutional vs retail flow
3. JCI (IHSG, ^JKSE) as regime indicator instead of BTC
4. Gap risk handling — stocks gap overnight, 3% gap-down can blow through SL
5. IDX tick size rounding — entry/SL/TP must be valid tick sizes (Rp1/5/25 depending on price band)
6. Session timing filter — only fire BSJP signals for end-of-day candles

**What stays identical from current system**: _clean_df features, backtest engine,
ML pipeline, WFO, regime similarity, Pulse concept, AI verdict, all UI rendering.

---

## Common Mistakes to Avoid

1. **Using raw `ev` instead of `ev_weighted` for best_key selection** — was a bug, now fixed.
   Always sort/rank by `ev_weighted` to match what the user sees in the UI table.

2. **Assuming NEUTRAL trades are losses** — they're NOT losses. They're breakeven outcomes
   excluded from ML to prevent single-class collapse. They still count in PF accounting.

3. **Interpreting 100% WR as edge on new/trending coins** — check fill_rate and check whether
   Partial+BE is the selected mode. REZ case: fill_rate=low (survivor bias) + Partial+BE = artifact.

4. **Treating WFO PASS on n_oos < 5 as meaningful** — the tightened threshold requires ≥8 OOS
   trades. 3-4 trades is statistical noise. Look at the rolling WFO edge_hit_rate instead.

5. **Changing the broad except on the ML fit** — the Pipeline sample_weight kwarg routing
   raises multiple exception types (TypeError, ValueError, etc.) across sklearn versions.
   The broad except catches all of them and falls back to unweighted fit. Don't narrow it.

6. **Removing .shift(1) from EMA** — this would create lookahead. The audit proves the
   current EMA computation is causal. Keep .shift(1) on ALL EMA calculations.

7. **Adding regime filtering to WFO** — WFO is intentionally not regime-filtered.
   WFO tests "does this pattern generalize to unseen data?" which is a different question.
   Regime filtering WFO would defeat its purpose as an independent validation signal.

8. **Building new features before validating live trading edge** — 30 days of journal data
   should drive the next feature priority, not intuition or "it sounds good."
