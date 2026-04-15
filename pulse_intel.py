"""
Pulse — On-chain Intelligence Module

A Nansen-lite intelligence layer that pulls free on-chain data and condenses
it into a "Pulse Score" (-15 to +15) plus per-source breakdowns.

Designed to be consumed by ANY trading strategy module — the momentum scanner
calls get_pulse_intel(symbol) and so will future ICT/S/R/volume-profile modules.

Phase 1 (this file):
  ✓ DefiLlama TVL delta tracker — works for DeFi tokens, completely free
  ⏳ Etherscan exchange flow tracker (next session)
  ⏳ LunarCrush social pulse (next session)
  ⏳ Solscan exchange flow for SOL tokens (next session)
  ⏳ Composite Pulse Score (next session, after all sources are in)

Architecture notes:
  - All HTTP calls use a global cached requests session
  - Each module returns a standardized dict with keys: ok, score, label, detail
  - Score range per module: -10 to +10 (then weighted into composite -15 to +15)
  - Cache TTLs are aggressive — on-chain data doesn't change minute-to-minute
  - All modules are independent — failures in one source don't break others
  - Module returns degrade gracefully: if data is unavailable, score=0 (neutral)
    rather than throwing or returning a fake bullish/bearish reading
"""
from __future__ import annotations
import time
import json
from typing import Optional

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Symbol → DefiLlama protocol slug mapping
# ─────────────────────────────────────────────────────────────────────────────
# DefiLlama identifies protocols by SLUG (lowercase short name in URL).
# A token symbol like "ETHUSDT" doesn't directly map — Ethereum the chain
# isn't a "protocol" in DefiLlama's TVL sense. But many Binance-listed
# tokens ARE DeFi protocol tokens whose TVL is the right signal.
#
# We maintain a curated mapping of common tokens. Extend as needed.
# When a symbol isn't in this map, the module returns ok=False neutrally.
# ─────────────────────────────────────────────────────────────────────────────
_SYMBOL_TO_DEFILLAMA_SLUG: dict = {
    # Major DeFi tokens — these have TVL as a direct fundamental signal
    "AAVE":   "aave",
    "UNI":    "uniswap",
    "LDO":    "lido",
    "MKR":    "makerdao",
    "CRV":    "curve-dex",
    "SUSHI":  "sushi",
    "COMP":   "compound-finance",
    "SNX":    "synthetix",
    "BAL":    "balancer",
    "1INCH":  "1inch-network",
    "DYDX":   "dydx",
    "GMX":    "gmx",
    "PENDLE": "pendle",
    "JUP":    "jupiter",
    "RAY":    "raydium",
    "ENA":    "ethena",
    "ETHFI":  "ether.fi",
    "ONDO":   "ondo-finance",
    "JTO":    "jito",
    "FXS":    "frax-finance",
    "RUNE":   "thorchain",
    "CAKE":   "pancakeswap",
    "STG":    "stargate-finance",
    "RPL":    "rocket-pool",
    "FRAX":   "frax-finance",
    "MORPHO": "morpho",
    "EIGEN":  "eigenlayer",
    "ZRO":    "layerzero",

    # Layer 1s — DefiLlama tracks them as CHAINS (different endpoint)
    # We mark them as "chain:" prefix so the resolver knows to use the chain endpoint
    "ETH":    "chain:Ethereum",
    "SOL":    "chain:Solana",
    "AVAX":   "chain:Avalanche",
    "MATIC":  "chain:Polygon",
    "POL":    "chain:Polygon",
    "ARB":    "chain:Arbitrum",
    "OP":     "chain:Optimism",
    "BNB":    "chain:BSC",
    "FTM":    "chain:Fantom",
    "NEAR":   "chain:Near",
    "ATOM":   "chain:Cosmos",
    "DOT":    "chain:Polkadot",
    "ADA":    "chain:Cardano",
    "TRX":    "chain:Tron",
    "TON":    "chain:Ton",
    "APT":    "chain:Aptos",
    "SUI":    "chain:Sui",
    "INJ":    "chain:Injective",
    "SEI":    "chain:Sei",

    # Tokens we explicitly DON'T track (no meaningful DefiLlama signal):
    # BTC, DOGE, SHIB, XRP, LTC, BCH, PEPE, WIF, BONK, FLOKI, etc.
    # These are non-DeFi tokens — TVL doesn't apply.
}


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize a Binance-style symbol to its base token.
    'ETHUSDT' -> 'ETH', 'AAVE-USDT' -> 'AAVE', 'BTCUSDC' -> 'BTC'
    """
    if not symbol:
        return ""
    s = symbol.upper().strip()
    # Strip common quote suffixes
    for q in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD"):
        if s.endswith(q):
            s = s[:-len(q)]
            break
    # Strip separators
    s = s.replace("-", "").replace("_", "").replace("/", "")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Caching layer — in-process TTL cache. Persists across Streamlit reruns
# in the same Python process but resets on app restart.
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: dict = {}
_CACHE_TTL: dict = {
    "tvl_protocol":   3600,   # 1hr — TVL updates slowly
    "tvl_chain":      3600,
    "exchange_flow":   900,   # 15min — flow shifts can be fast (Phase 2)
    "social":         1800,   # 30min — sentiment swings (Phase 2)
    "holder_data":   86400,   # 24hr — holder distribution barely moves (Phase 2)
}


def _cache_get(key: str, kind: str):
    """Return cached value if still valid, else None."""
    rec = _CACHE.get(key)
    if not rec:
        return None
    age = time.time() - rec["ts"]
    if age > _CACHE_TTL.get(kind, 600):
        return None
    return rec["val"]


def _cache_set(key: str, val):
    """Store value with current timestamp."""
    _CACHE[key] = {"ts": time.time(), "val": val}


def cache_clear():
    """Manually clear the cache. Exposed for the UI 'Refresh' button."""
    global _CACHE
    _CACHE = {}


# ─────────────────────────────────────────────────────────────────────────────
# DefiLlama TVL module
# ─────────────────────────────────────────────────────────────────────────────
_DEFILLAMA_BASE = "https://api.llama.fi"


def _fetch_protocol_tvl(slug: str) -> Optional[dict]:
    """
    Fetch protocol-level TVL data from DefiLlama.
    Returns the raw protocol dict (with current_tvl, change_1d, change_7d, etc.)
    or None on failure.
    """
    cache_key = f"prot:{slug}"
    cached = _cache_get(cache_key, "tvl_protocol")
    if cached is not None:
        return cached

    try:
        # The /protocol/{slug} endpoint returns historical TVL series
        # We compute deltas from this since /protocols list endpoint doesn't always
        # have accurate change_1d/7d on small protocols
        resp = requests.get(
            f"{_DEFILLAMA_BASE}/protocol/{slug}",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)   # cache the failure too — don't retry hot
            return None
        data = resp.json()
        _cache_set(cache_key, data)
        return data
    except Exception:
        return None


def _fetch_chain_tvl(chain_name: str) -> Optional[dict]:
    """
    Fetch chain-level TVL (e.g., total TVL across all protocols on Ethereum).
    Used for L1 tokens like ETH, SOL, AVAX where we want the chain's overall
    DeFi health rather than a single protocol.

    Returns dict with: chain_name, current_tvl, tvl_24h_ago, tvl_7d_ago,
    delta_24h_pct, delta_7d_pct  — or None.
    """
    cache_key = f"chain:{chain_name}"
    cached = _cache_get(cache_key, "tvl_chain")
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            f"{_DEFILLAMA_BASE}/v2/historicalChainTvl/{chain_name}",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        series = resp.json()
        if not series or not isinstance(series, list) or len(series) < 8:
            _cache_set(cache_key, None)
            return None

        # series is a list of {date: timestamp, tvl: usd} sorted ascending
        current = float(series[-1].get("tvl", 0) or 0)
        tvl_24h_ago = float(series[-2].get("tvl", 0) or 0) if len(series) >= 2 else current
        tvl_7d_ago  = float(series[-8].get("tvl", 0) or 0) if len(series) >= 8 else current

        delta_24h_pct = ((current - tvl_24h_ago) / tvl_24h_ago * 100) if tvl_24h_ago > 0 else 0.0
        delta_7d_pct  = ((current - tvl_7d_ago)  / tvl_7d_ago  * 100) if tvl_7d_ago  > 0 else 0.0

        out = {
            "chain_name":     chain_name,
            "current_tvl":    current,
            "tvl_24h_ago":    tvl_24h_ago,
            "tvl_7d_ago":     tvl_7d_ago,
            "delta_24h_pct":  delta_24h_pct,
            "delta_7d_pct":   delta_7d_pct,
            "n_data_points":  len(series),
        }
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


def _summarize_protocol_tvl(prot_data: dict) -> dict:
    """
    Reduce raw /protocol/{slug} response to the same shape as chain TVL.
    Computes 24h and 7d delta from the historical TVL series.
    """
    # The protocol endpoint returns a 'tvl' array of {date, totalLiquidityUSD}
    series = prot_data.get("tvl", []) or []
    if not series or len(series) < 2:
        return {
            "current_tvl": 0, "tvl_24h_ago": 0, "tvl_7d_ago": 0,
            "delta_24h_pct": 0.0, "delta_7d_pct": 0.0, "n_data_points": 0,
        }

    current      = float(series[-1].get("totalLiquidityUSD", 0) or 0)
    tvl_24h_ago  = float(series[-2].get("totalLiquidityUSD", 0) or 0) if len(series) >= 2 else current
    tvl_7d_ago   = float(series[-8].get("totalLiquidityUSD", 0) or 0) if len(series) >= 8 else current

    delta_24h_pct = ((current - tvl_24h_ago) / tvl_24h_ago * 100) if tvl_24h_ago > 0 else 0.0
    delta_7d_pct  = ((current - tvl_7d_ago)  / tvl_7d_ago  * 100) if tvl_7d_ago  > 0 else 0.0

    return {
        "current_tvl":   current,
        "tvl_24h_ago":   tvl_24h_ago,
        "tvl_7d_ago":    tvl_7d_ago,
        "delta_24h_pct": delta_24h_pct,
        "delta_7d_pct":  delta_7d_pct,
        "n_data_points": len(series),
    }


def get_tvl_intel(symbol: str) -> dict:
    """
    Get DefiLlama TVL intelligence for a symbol.

    Returns a standardized dict:
      {
        "ok": bool,                # True if data was fetched successfully
        "supported": bool,         # True if symbol is in our DefiLlama mapping
        "score": int,              # -10 to +10 (signed sub-score for composite)
        "label": str,              # "ACCUMULATING" | "STEADY" | "BLEEDING" | "N/A"
        "color": str,              # hex color for UI badge
        "detail": str,             # one-line human-readable description
        "data": {                  # raw numbers for the UI to render in detail card
          "source_type": "protocol" | "chain",
          "source_name": str,      # the slug or chain name
          "current_tvl": float,
          "delta_24h_pct": float,
          "delta_7d_pct": float,
        }
      }

    Score logic (the heart of the TVL signal):
      - +10 to +6: TVL strongly growing (>= +5% 7d, +1% 24h)  → bullish foundation
      - +5 to +2:  TVL modestly growing (+1% to +5% 7d)        → mild bullish
      - +1 to -1:  TVL flat (-1% to +1% 7d)                    → neutral
      - -2 to -5:  TVL bleeding (-5% to -1% 7d)                → mild bearish
      - -6 to -10: TVL collapsing (< -5% 7d)                   → strong bearish

    Why TVL matters: TVL = real capital staying in the protocol/chain. If price
    is up but TVL is bleeding, the rally is speculative and likely to fade. If
    price is up AND TVL is growing, it's genuine demand.
    """
    base_token = _normalize_symbol(symbol)
    slug = _SYMBOL_TO_DEFILLAMA_SLUG.get(base_token)

    # Symbol not in our map — return neutral
    if not slug:
        return {
            "ok": False, "supported": False,
            "score": 0, "label": "N/A", "color": "#8892b0",
            "detail": f"{base_token} not in DefiLlama tracking map (non-DeFi token)",
            "data": {"source_type": "none", "source_name": "", "current_tvl": 0,
                     "delta_24h_pct": 0, "delta_7d_pct": 0},
        }

    # Resolve chain vs protocol
    if slug.startswith("chain:"):
        chain_name = slug.split(":", 1)[1]
        chain_data = _fetch_chain_tvl(chain_name)
        if not chain_data:
            return {
                "ok": False, "supported": True,
                "score": 0, "label": "N/A", "color": "#8892b0",
                "detail": f"Could not fetch chain TVL for {chain_name} (API down or rate-limited)",
                "data": {"source_type": "chain", "source_name": chain_name,
                         "current_tvl": 0, "delta_24h_pct": 0, "delta_7d_pct": 0},
            }
        source_type, source_name = "chain", chain_name
        d24, d7 = chain_data["delta_24h_pct"], chain_data["delta_7d_pct"]
        cur_tvl = chain_data["current_tvl"]
    else:
        prot_raw = _fetch_protocol_tvl(slug)
        if not prot_raw:
            return {
                "ok": False, "supported": True,
                "score": 0, "label": "N/A", "color": "#8892b0",
                "detail": f"Could not fetch protocol TVL for {slug}",
                "data": {"source_type": "protocol", "source_name": slug,
                         "current_tvl": 0, "delta_24h_pct": 0, "delta_7d_pct": 0},
            }
        prot_summary = _summarize_protocol_tvl(prot_raw)
        source_type, source_name = "protocol", slug
        d24, d7 = prot_summary["delta_24h_pct"], prot_summary["delta_7d_pct"]
        cur_tvl = prot_summary["current_tvl"]

    # ─── Score the TVL movement ─────────────────────────────────────────
    # Weight 7d delta more than 24h (smoothed, less noisy)
    # Combined indicator: 0.7 * d7 + 0.3 * d24
    blended = 0.7 * d7 + 0.3 * d24

    if blended >= 8.0:
        score, label, color = 10, "STRONGLY ACCUMULATING", "#3fb950"
    elif blended >= 4.0:
        score, label, color = 7, "ACCUMULATING", "#3fb950"
    elif blended >= 1.5:
        score, label, color = 4, "GROWING", "#64ffda"
    elif blended >= -1.5:
        score, label, color = 0, "STEADY", "#8892b0"
    elif blended >= -4.0:
        score, label, color = -4, "DECLINING", "#e3b341"
    elif blended >= -8.0:
        score, label, color = -7, "BLEEDING", "#f0883e"
    else:
        score, label, color = -10, "COLLAPSING", "#f85149"

    # Human-readable detail
    _tvl_str = (
        f"${cur_tvl/1e9:.2f}B"  if cur_tvl >= 1e9  else
        f"${cur_tvl/1e6:.1f}M"  if cur_tvl >= 1e6  else
        f"${cur_tvl/1e3:.0f}K"  if cur_tvl >= 1e3  else
        f"${cur_tvl:.0f}"
    )
    detail = (
        f"{source_name} TVL: {_tvl_str} · "
        f"24h {d24:+.2f}% · 7d {d7:+.2f}% → {label}"
    )

    return {
        "ok": True, "supported": True,
        "score": score, "label": label, "color": color, "detail": detail,
        "data": {
            "source_type":   source_type,
            "source_name":   source_name,
            "current_tvl":   cur_tvl,
            "delta_24h_pct": d24,
            "delta_7d_pct":  d7,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite — Phase 1 only includes TVL. Will be updated as more sources are added.
# ─────────────────────────────────────────────────────────────────────────────
def get_pulse_intel(symbol: str) -> dict:
    """
    Main entry point. Returns the full Pulse intelligence dict for a symbol.

    In Phase 1 this only contains TVL. As more sources are added (exchange
    flow, social, smart money), they'll be merged into the composite_score.

    Returns:
      {
        "symbol": str,                    # the original symbol passed in
        "base_token": str,                # normalized (ETH from ETHUSDT)
        "tvl": {...},                     # output of get_tvl_intel()
        "exchange_flow": None,            # Phase 2
        "social": None,                   # Phase 2
        "smart_money": None,              # Phase 2
        "composite_score": int,           # -15 to +15, weighted aggregate
        "composite_label": str,           # BULLISH | BEARISH | NEUTRAL etc.
        "composite_color": str,           # hex for UI
        "verdict_summary": str,           # one-line takeaway
        "phase": "1 (TVL only)",          # so the UI can label it as preview
      }
    """
    base_token = _normalize_symbol(symbol)
    tvl = get_tvl_intel(symbol)

    # ─── Phase 1 composite logic ───────────────────────────────────────
    # With only TVL available, the composite IS the TVL score (capped to ±15
    # since each module is ±10 but composite range is wider).
    # When other modules come online, weights become:
    #   composite = 0.35 * exchange_flow + 0.30 * tvl + 0.20 * social + 0.15 * smart_money
    if tvl["ok"]:
        composite = tvl["score"]
    else:
        composite = 0

    if   composite >= 7:  c_label, c_color = "STRONGLY BULLISH", "#3fb950"
    elif composite >= 3:  c_label, c_color = "BULLISH",          "#64ffda"
    elif composite >= -2: c_label, c_color = "NEUTRAL",          "#8892b0"
    elif composite >= -6: c_label, c_color = "BEARISH",          "#f0883e"
    else:                 c_label, c_color = "STRONGLY BEARISH", "#f85149"

    # Build a one-line verdict the UI can show prominently
    if not tvl["supported"]:
        summary = (
            f"Pulse can't track {base_token} — it's not a DeFi protocol or L1 we monitor. "
            f"Phase 2 (exchange flow + social) will cover most majors."
        )
    elif not tvl["ok"]:
        summary = "TVL data unavailable right now. Treat as neutral until next refresh."
    elif composite >= 7:
        summary = (
            f"Strong on-chain accumulation in {tvl['data']['source_name']}. "
            f"This is a structural bullish backdrop — momentum signals here have stronger conviction."
        )
    elif composite >= 3:
        summary = (
            f"Moderate TVL growth in {tvl['data']['source_name']}. Decent backdrop for momentum."
        )
    elif composite >= -2:
        summary = f"TVL is steady. No on-chain tailwind, but no headwind either."
    elif composite >= -6:
        summary = (
            f"TVL declining in {tvl['data']['source_name']}. Capital is leaving — "
            f"momentum signals here may lack follow-through."
        )
    else:
        summary = (
            f"TVL collapsing in {tvl['data']['source_name']}. Avoid going long — "
            f"price strength here is likely speculative without on-chain support."
        )

    return {
        "symbol":          symbol,
        "base_token":      base_token,
        "tvl":             tvl,
        "exchange_flow":   None,    # Phase 2
        "social":          None,    # Phase 2
        "smart_money":     None,    # Phase 2
        "composite_score":  composite,
        "composite_label":  c_label,
        "composite_color":  c_color,
        "verdict_summary":  summary,
        "phase":           "1 (TVL only)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic / smoke test entry — run `python pulse_intel.py ETHUSDT`
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    test_symbols = sys.argv[1:] if len(sys.argv) > 1 else ["ETHUSDT", "AAVEUSDT", "BTCUSDT", "ONDOUSDT"]
    for s in test_symbols:
        print(f"\n{'='*60}")
        print(f"Pulse Intel: {s}")
        print('='*60)
        res = get_pulse_intel(s)
        print(json.dumps(res, indent=2, default=str))
