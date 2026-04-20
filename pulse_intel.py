"""
Pulse — On-chain Intelligence Module

A Nansen-lite intelligence layer that pulls free on-chain data and condenses
it into a "Pulse Score" (-15 to +15) plus per-source breakdowns.

Designed to be consumed by ANY trading strategy module — the momentum scanner
calls get_pulse_intel(symbol) and so will future ICT/S/R/volume-profile modules.

Phase 4 (current):
  ✓ DefiLlama TVL delta tracker — works for DeFi tokens, completely free
  ✓ Etherscan exchange flow tracker — ERC-20 CEX flow + top whale txs
  ✓ Solscan exchange flow tracker — SPL CEX flow + top whale txs
  ✓ LunarCrush social pulse — galaxy score + sentiment
  ✓ Macro backdrop modifier — BTC.D proxy + stablecoin supply delta
  ✓ Derivatives Pulse — Binance futures funding + OI 24h + L/S ratio (no key)
  ⏳ Smart-money wallet tracker (reserved)

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
    "solana_flow":     900,   # 15min — same for SPL tokens on Solana (Phase 3)
    "social":         1800,   # 30min — sentiment swings (Phase 3)
    "holder_data":   86400,   # 24hr — holder distribution barely moves (Phase 2)
    "macro_btc_dom":  1800,   # 30min — BTC dominance
    "macro_stables":  14400,  # 4hr — stablecoin supply moves slowly
    "derivatives":     300,   # 5min — funding/OI/L:S move fast, keep fresh
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
# Etherscan exchange flow module (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
# How it works:
#   1. Resolve symbol → ERC-20 contract address on Ethereum mainnet
#   2. Pull recent token transfers using Etherscan V2 API (free tier: 5/sec, 100k/day)
#   3. Classify each transfer:
#        - to a known CEX hot wallet  → INFLOW (selling pressure)
#        - from a known CEX hot wallet → OUTFLOW (accumulation/withdrawal)
#        - between non-CEX addresses  → ignored
#   4. Sum 24h net flow in USD (using DefiLlama price endpoint for valuation)
#   5. Score: positive net outflow = bullish (-10 to +10)
#
# Why this matters: large net inflows to exchanges typically precede sell
# pressure (whales depositing to dump). Large net outflows typically signal
# accumulation (whales withdrawing to cold storage or staking).
#
# Limitations to be honest about:
#   - Only Ethereum mainnet ERC-20 tokens (not Solana, BSC, Arbitrum, etc.)
#   - "Known CEX" list is static — new exchanges or wallet rotations miss
#   - Free tier limits us to ~100 txs per call, so very high-volume tokens
#     (USDT/USDC) won't get a full 24h window — we cap at 200 most-recent txs
#   - Centralized exchanges sometimes batch deposits — a single large
#     inflow tx can be 100 users depositing, not 1 whale dumping
# ─────────────────────────────────────────────────────────────────────────────

_ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
_ETH_CHAIN_ID   = 1   # Ethereum mainnet

# Symbol → ERC-20 contract address mapping. All addresses are LOWERCASE
# (Etherscan returns lowercase in transfer responses; case-sensitive comparison
# is unreliable). Curated for the major tokens — extend as needed.
_SYMBOL_TO_ERC20: dict = {
    # Stables (omitted from flow analysis — flow signal is meaningless on stables)
    # "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    # "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",

    # Major tokens with active CEX flow
    "WBTC":   "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    "WETH":   "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "LINK":   "0x514910771af9ca656af840dff83e8264ecf986ca",

    # DeFi blue-chips
    "AAVE":   "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",
    "UNI":    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
    "LDO":    "0x5a98fcbea516cf06857215779fd812ca3bef1b32",
    "MKR":    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2",
    "CRV":    "0xd533a949740bb3306d119cc777fa900ba034cd52",
    "SUSHI":  "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2",
    "COMP":   "0xc00e94cb662c3520282e6f5717214004a7f26888",
    "SNX":    "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f",
    "BAL":    "0xba100000625a3754423978a60c9317c58a424e3d",
    "1INCH":  "0x111111111117dc0aa78b770fa6a738034120c302",
    "GMX":    "0xfc5a1a6eb076a2c7ad06ed22c90d7e710e35ad0a",
    "PENDLE": "0x808507121b80c02388fad14726482e061b8da827",
    "ENA":    "0x57e114b691db790c35207b2e685d4a43181e6061",
    "ETHFI":  "0xfe0c30065b384f05761f15d0cc899d4f9f9cc0eb",
    "ONDO":   "0xfaba6f8e4a5e8ab82f62fe7c39859fa577269be3",
    "FXS":    "0x3432b6a60d23ca0dfca7761b7ab56459d9c964d0",
    "MORPHO": "0x9994e35db50125e0df82e4c2dde62496ce330999",
    "EIGEN":  "0xec53bf9167f50cdeb3ae105f56099aaab9061f83",

    # Layer-token bridge wrappers (the L1s themselves like ETH/SOL aren't ERC-20
    # so we skip them — for ETH itself, exchange flow is computed via the
    # `address` filter rather than `tokentx`. We mark them "native" below.)
}

_NATIVE_ETH_TOKENS: set = {"ETH"}   # use account txlist for these, not tokentx

# Known CEX hot wallets. These are the deposit/withdrawal addresses publicly
# labeled by Etherscan and used by the major exchanges. NOT exhaustive — each
# CEX has dozens of rotation wallets — but covers the bulk of flow for the
# top 5 exchanges. All lowercase for case-insensitive matching.
#
# Source: Etherscan public labels. Update periodically — addresses can rotate.
_KNOWN_CEX_WALLETS: dict = {
    # Binance hot wallets
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance 14",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance 15",
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f": "Binance 16",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance 17",
    "0x9696f59e4d72e237be84ffd425dcad154bf96976": "Binance 18",
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance Cold",
    "0xf977814e90da44bfa03b6295a0616a897441acec": "Binance Cold 2",
    "0x5a52e96bacdabb82fd05763e25335261b270efcb": "Binance Hot 4",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase 1",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase 2",
    "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740": "Coinbase 3",
    "0x3cd751e6b0078be393132286c442345e5dc49699": "Coinbase 4",
    "0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511": "Coinbase 5",
    "0xeb2629a2734e272bcc07bda959863f316f4bd4cf": "Coinbase 6",
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e": "Coinbase 7",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKEx 1",
    "0x236f9f97e0e62388479bf9e5ba4889e46b0273c3": "OKEx 2",
    "0xa7efae728d2936e78bda97dc267687568dd593f3": "OKEx 3",
    "0x2c8fbb630289363ac80705a1a61273f76fd5a161": "OKX 4",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "Bybit",
    "0xee5b5b923ffce93a870b3104b7ca09c3db80047a": "Bybit 2",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken 1",
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13": "Kraken 2",
    "0xe853c56864a2ebe4576a807d26fdc4a0ada51919": "Kraken 3",
    "0xae2d4617c862309a3d75a0ffb358c7a5009c673f": "Kraken 4",
    # KuCoin
    "0x2b5634c42055806a59e9107ed44d43c426e58258": "KuCoin 1",
    "0xd6216fc19db775df9774a6e33526131da7d19a2c": "KuCoin 2",
    # Bitfinex
    "0x1151314c646ce4e0efd76d1af4760ae66a9fe30f": "Bitfinex 1",
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa": "Bitfinex Multisig",
    # Crypto.com
    "0x6262998ced04146fa42253a5c0af90ca02dfd2a3": "Crypto.com",
    "0x46340b20830761efd32832a74d7169b29feb9758": "Crypto.com 2",
    # Gate.io
    "0x0d0707963952f2fba59dd06f2b425ace40b492fe": "Gate.io",
    "0x1c4b70a3968436b9a0a9cf5205c787eb81bb558c": "Gate.io 2",
}


def _is_cex(addr: str) -> bool:
    """Case-insensitive CEX check."""
    return (addr or "").lower() in _KNOWN_CEX_WALLETS


def _cex_label(addr: str) -> str:
    return _KNOWN_CEX_WALLETS.get((addr or "").lower(), "Unknown")


def _fetch_token_transfers(contract_addr: str, api_key: str,
                            n_recent: int = 200) -> Optional[list]:
    """
    Fetch the most recent ERC-20 token transfers for a contract.

    Etherscan returns up to 10,000 records but free tier rate-limits make
    that impractical. We pull the 200 most recent (single page, sort=desc).
    For high-volume tokens this only covers a few hours; for mid-volume
    tokens it covers a day or more. The 'window_hours_actual' field in the
    return tells you how much real-time the sample represents.
    """
    cache_key = f"flow:{contract_addr}"
    cached = _cache_get(cache_key, "exchange_flow")
    if cached is not None:
        return cached

    try:
        params = {
            "chainid":         _ETH_CHAIN_ID,
            "module":          "account",
            "action":          "tokentx",
            "contractaddress": contract_addr,
            "page":            1,
            "offset":          n_recent,
            "sort":            "desc",
            "apikey":          api_key,
        }
        resp = requests.get(_ETHERSCAN_BASE, params=params, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"})
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        data = resp.json()
        # Etherscan returns status="1" on success, status="0" on no-result OR error
        if str(data.get("status")) != "1":
            # Distinguish "no transactions" from real errors
            msg = (data.get("message") or "").lower()
            if "no transactions" in msg:
                _cache_set(cache_key, [])
                return []
            # Real API error (rate limit, bad key, etc.) — don't cache
            return None
        result = data.get("result", []) or []
        _cache_set(cache_key, result)
        return result
    except Exception:
        return None


def _fetch_token_price_usd(contract_addr: str) -> Optional[float]:
    """
    Get current USD price for a token using DefiLlama's free price API.
    No API key needed. Used to convert raw token amounts into USD flow.
    """
    cache_key = f"price:{contract_addr.lower()}"
    cached = _cache_get(cache_key, "tvl_chain")   # reuse 1hr TTL
    if cached is not None:
        return cached
    try:
        url = f"https://coins.llama.fi/prices/current/ethereum:{contract_addr}"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"})
        if resp.status_code != 200:
            return None
        coins = (resp.json() or {}).get("coins", {})
        for k, v in coins.items():
            price = float(v.get("price", 0) or 0)
            if price > 0:
                _cache_set(cache_key, price)
                return price
        return None
    except Exception:
        return None


def _build_top_whale_txs(whale_txs: list, price_usd: float, top_n: int = 5) -> dict:
    """
    Sort the collected CEX-touching transactions and return the top N biggest
    in each direction. Used by both the ETH (Etherscan) and SOL (Solscan) flow
    modules so the UI can show "who moved what recently".

    Shape:
      {
        "outflows": [{"amt_usd", "amount", "cex", "hash", "ts", "age_min"}, ...],
        "inflows":  [same shape ordered biggest → smallest],
        "sort_by":  "amt_usd" | "amount",   # which metric was used to rank
      }

    When `price_usd` is 0 (price unavailable) we fall back to ranking by token
    count. In that case `amt_usd` in each row will be 0 and the UI should
    display `amount` (+ token symbol) instead.
    """
    if not whale_txs:
        return {"outflows": [], "inflows": [], "sort_by": "amt_usd"}

    sort_key = "amt_usd" if price_usd and price_usd > 0 else "amount"

    # Rank within each direction. Ties broken by timestamp descending (fresher first).
    outflows = sorted(
        [t for t in whale_txs if t.get("direction") == "outflow"],
        key=lambda t: (float(t.get(sort_key, 0) or 0), int(t.get("ts", 0) or 0)),
        reverse=True,
    )[:top_n]
    inflows = sorted(
        [t for t in whale_txs if t.get("direction") == "inflow"],
        key=lambda t: (float(t.get(sort_key, 0) or 0), int(t.get("ts", 0) or 0)),
        reverse=True,
    )[:top_n]

    # Attach "minutes ago" for display convenience. Backend can always re-derive
    # from ts; doing it here keeps the UI layer dumb.
    now_ts = int(time.time())
    def _enrich(lst):
        out = []
        for t in lst:
            _ts = int(t.get("ts", 0) or 0)
            age_min = max(0, (now_ts - _ts) // 60) if _ts > 0 else 0
            out.append({
                "amt_usd": float(t.get("amt_usd", 0) or 0),
                "amount":  float(t.get("amount",  0) or 0),
                "cex":     t.get("cex", "") or "",
                "hash":    t.get("hash", "") or "",
                "ts":      _ts,
                "age_min": int(age_min),
            })
        return out

    return {
        "outflows": _enrich(outflows),
        "inflows":  _enrich(inflows),
        "sort_by":  sort_key,
    }


def get_exchange_flow_intel(symbol: str, api_key: str = "") -> dict:
    """
    Get net 24h CEX exchange flow for a symbol.

    Returns:
      {
        "ok": bool,
        "supported": bool,         # token in our ERC-20 map
        "score": int,              # -10 to +10
        "label": str,              # "STRONG OUTFLOW" | "OUTFLOW" | "NEUTRAL" | "INFLOW" | "STRONG INFLOW"
        "color": str,
        "detail": str,             # human-readable summary
        "data": {
          "contract":       str,   # ERC-20 contract address
          "n_transfers":    int,   # total transfers in our sample
          "n_cex_inflows":  int,   # transfers TO CEX (selling pressure)
          "n_cex_outflows": int,   # transfers FROM CEX (accumulation)
          "inflow_usd":     float, # USD value of inflows
          "outflow_usd":    float, # USD value of outflows
          "net_flow_usd":   float, # outflow - inflow (positive = bullish)
          "window_hours":   float, # actual time window the sample covers
          "top_inflow":     dict,  # largest single inflow {amt_usd, cex, hash}
          "top_outflow":    dict,  # largest single outflow
          "top_transactions": dict,# {outflows: [top 5], inflows: [top 5], sort_by: str}
          "needs_api_key":  bool,  # True if we couldn't run because no key
        }
      }
    """
    base = _normalize_symbol(symbol)

    # Check support
    if base not in _SYMBOL_TO_ERC20:
        return {
            "ok": False, "supported": False,
            "score": 0, "label": "N/A", "color": "#8892b0",
            "detail": f"{base} not in Etherscan ERC-20 tracking map",
            "data": {"contract": "", "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {}, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"},
                     "needs_api_key": False},
        }

    # Need an Etherscan API key
    if not api_key:
        return {
            "ok": False, "supported": True,
            "score": 0, "label": "NO KEY", "color": "#e3b341",
            "detail": (
                f"Etherscan API key required. Get a free one at "
                f"etherscan.io/apis (5 calls/sec, 100k/day on free tier) "
                f"and paste it in the Pulse sidebar."
            ),
            "data": {"contract": _SYMBOL_TO_ERC20[base], "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {}, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"},
                     "needs_api_key": True},
        }

    contract = _SYMBOL_TO_ERC20[base]
    transfers = _fetch_token_transfers(contract, api_key, n_recent=200)

    if transfers is None:
        return {
            "ok": False, "supported": True,
            "score": 0, "label": "API ERROR", "color": "#f85149",
            "detail": "Etherscan API call failed (rate limit, network, or bad key)",
            "data": {"contract": contract, "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {}, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"},
                     "needs_api_key": False},
        }
    if not transfers:
        return {
            "ok": True, "supported": True,
            "score": 0, "label": "NO ACTIVITY", "color": "#8892b0",
            "detail": "No recent ERC-20 transfers found for this token",
            "data": {"contract": contract, "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {}, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"},
                     "needs_api_key": False},
        }

    # Get token price for USD conversion (best-effort; if we can't get price
    # we still report transfer counts and use those for scoring).
    price_usd = _fetch_token_price_usd(contract) or 0.0

    # ── Classify and aggregate ──────────────────────────────────────────
    n_inflows = n_outflows = 0
    inflow_usd = outflow_usd = 0.0
    top_inflow  = {"amt_usd": 0, "cex": "", "hash": ""}
    top_outflow = {"amt_usd": 0, "cex": "", "hash": ""}
    # NEW: detailed whale list — collect every CEX-touching transaction with
    # amount + direction + exchange label so the UI can show top N movers.
    # Keeps it lightweight (ts + amt + direction + cex + hash). Sorted later.
    whale_txs: list = []

    # Track the time window we actually sampled
    timestamps = []

    for tx in transfers:
        try:
            ts = int(tx.get("timeStamp", 0) or 0)
            timestamps.append(ts)
            from_addr = (tx.get("from") or "").lower()
            to_addr   = (tx.get("to")   or "").lower()
            value_raw = float(tx.get("value", 0) or 0)
            decimals  = int(tx.get("tokenDecimal", 18) or 18)
            amount    = value_raw / (10 ** decimals)
            amt_usd   = amount * price_usd if price_usd > 0 else 0
            tx_hash   = tx.get("hash", "")

            from_is_cex = _is_cex(from_addr)
            to_is_cex   = _is_cex(to_addr)

            # Classify — ignore CEX-to-CEX (internal rebalancing, no signal)
            if to_is_cex and not from_is_cex:
                # Deposit into exchange = INFLOW = bearish (selling pressure)
                n_inflows += 1
                inflow_usd += amt_usd
                if amt_usd > top_inflow["amt_usd"]:
                    top_inflow = {"amt_usd": amt_usd,
                                  "cex": _cex_label(to_addr),
                                  "hash": tx_hash}
                whale_txs.append({
                    "direction": "inflow",   # to CEX — selling pressure
                    "amt_usd":   amt_usd,
                    "amount":    amount,
                    "cex":       _cex_label(to_addr),
                    "hash":      tx_hash,
                    "ts":        ts,
                })
            elif from_is_cex and not to_is_cex:
                # Withdrawal from exchange = OUTFLOW = bullish (accumulation)
                n_outflows += 1
                outflow_usd += amt_usd
                if amt_usd > top_outflow["amt_usd"]:
                    top_outflow = {"amt_usd": amt_usd,
                                   "cex": _cex_label(from_addr),
                                   "hash": tx_hash}
                whale_txs.append({
                    "direction": "outflow",  # from CEX — accumulation
                    "amt_usd":   amt_usd,
                    "amount":    amount,
                    "cex":       _cex_label(from_addr),
                    "hash":      tx_hash,
                    "ts":        ts,
                })
        except Exception:
            continue

    # Time window covered (oldest to newest in our sample)
    if len(timestamps) >= 2:
        window_seconds = max(timestamps) - min(timestamps)
        window_hours   = round(window_seconds / 3600.0, 2)
    else:
        window_hours = 0.0

    net_flow_usd = outflow_usd - inflow_usd
    cex_total    = n_inflows + n_outflows

    # ── Scoring ─────────────────────────────────────────────────────────
    # Two-axis scoring: net flow magnitude AND count balance
    #
    # Magnitude: score net_flow_usd against thresholds chosen for typical
    # mid-cap token flow. These are calibration choices — tune after data.
    #
    # Count balance: a 60/40 outflow/inflow split is a stronger signal
    # than 51/49. We use the relative imbalance as a confidence multiplier.

    if cex_total == 0:
        # No CEX-related transfers in our sample — return neutral
        return {
            "ok": True, "supported": True,
            "score": 0, "label": "NO CEX FLOW", "color": "#8892b0",
            "detail": (
                f"{len(transfers)} recent transfers but none touched known CEX "
                f"wallets. Either low CEX activity or the CEX list misses this "
                f"token's primary exchange."
            ),
            "data": {"contract": contract, "n_transfers": len(transfers),
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": window_hours,
                     "top_inflow": top_inflow, "top_outflow": top_outflow,
                     "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"},
                     "needs_api_key": False},
        }

    # Score the USD net flow (in millions, since CEX flows are large)
    # Note: these thresholds are aggressive by design — small flows shouldn't
    # move the needle. A $5M+ net outflow is a real whale-level signal.
    flow_m = net_flow_usd / 1_000_000.0

    if   flow_m >=  10.0: mag_score = 10
    elif flow_m >=   5.0: mag_score =  7
    elif flow_m >=   2.0: mag_score =  4
    elif flow_m >=  -2.0: mag_score =  0
    elif flow_m >=  -5.0: mag_score = -4
    elif flow_m >= -10.0: mag_score = -7
    else:                 mag_score = -10

    # If we couldn't get USD prices, fall back to count-based scoring only
    if price_usd <= 0 and cex_total >= 5:
        # Use count imbalance: outflows / total
        out_ratio = n_outflows / cex_total
        if   out_ratio >= 0.75: mag_score = 7
        elif out_ratio >= 0.60: mag_score = 4
        elif out_ratio >= 0.40: mag_score = 0
        elif out_ratio >= 0.25: mag_score = -4
        else:                   mag_score = -7

    # Map score to label and color
    if   mag_score >=  7: label, color = "STRONG OUTFLOW",  "#3fb950"
    elif mag_score >=  3: label, color = "OUTFLOW",         "#64ffda"
    elif mag_score >= -2: label, color = "BALANCED",        "#8892b0"
    elif mag_score >= -6: label, color = "INFLOW",          "#f0883e"
    else:                 label, color = "STRONG INFLOW",   "#f85149"

    # Build human-readable detail
    def _fmt_usd(v):
        if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
        return f"${v:.0f}"

    if price_usd > 0:
        detail = (
            f"Net {label.lower()}: {_fmt_usd(net_flow_usd)} over ~{window_hours:.1f}h "
            f"({n_outflows} withdrawals from CEX vs {n_inflows} deposits to CEX)"
        )
    else:
        detail = (
            f"{n_outflows} withdrawals vs {n_inflows} deposits over ~{window_hours:.1f}h "
            f"(price unavailable — score from count imbalance only)"
        )

    return {
        "ok": True, "supported": True,
        "score": mag_score, "label": label, "color": color, "detail": detail,
        "data": {
            "contract":       contract,
            "n_transfers":    len(transfers),
            "n_cex_inflows":  n_inflows,
            "n_cex_outflows": n_outflows,
            "inflow_usd":     inflow_usd,
            "outflow_usd":    outflow_usd,
            "net_flow_usd":   net_flow_usd,
            "window_hours":   window_hours,
            "top_inflow":     top_inflow,
            "top_outflow":    top_outflow,
            # NEW: Whale transactions expansion — top 5 biggest in each
            # direction by USD. Enables UI to show "who moved what recently"
            # rather than just aggregate net flow. Useful for spotting a
            # single-whale distribution event masked by otherwise balanced flow.
            "top_transactions": _build_top_whale_txs(whale_txs, price_usd),
            "needs_api_key":  False,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Solscan exchange flow module (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
# Parallel to Etherscan but for SPL tokens on Solana. Uses Solscan Pro API
# (v2.0) which requires a free API key. Same scoring logic as Etherscan:
#   - Transfers to known CEX Solana wallets = INFLOW = bearish
#   - Transfers from known CEX Solana wallets = OUTFLOW = bullish
#
# We only support SPL tokens with high CEX activity. SOL native transfers
# are tracked separately and not implemented here (would need a different
# endpoint per Solscan's API design).
# ─────────────────────────────────────────────────────────────────────────────
_SOLSCAN_BASE = "https://pro-api.solscan.io/v2.0"

# Symbol → SPL token mint address mapping (mainnet-beta)
# All addresses are case-sensitive on Solana; copy EXACTLY.
_SYMBOL_TO_SPL_MINT: dict = {
    "JUP":   "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "RAY":   "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "JTO":   "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
    "WIF":   "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "BONK":  "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "PYTH":  "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
    "RNDR":  "rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof",
    "HNT":   "hntyVP6YFm1Hg25TN9WGLqM12b8TQmcknKrdu1oxWux",
    "TNSR":  "TNSRxcUxoT9xBG3de7PiJyTDYu7kskLqcpddxnEJAS6",
    "POPCAT":"7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
    "MEW":   "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5",
    "ORCA":  "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "MSOL":  "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
}

# Known CEX Solana deposit/hot wallets. Case-sensitive.
# Sourced from Solscan's public wallet labels — rotates occasionally.
_KNOWN_CEX_SOLANA: dict = {
    # Binance
    "2ojv9BAiHUrvsm9gxDe7fJSzbNZSJcxZvf8dqmWGHG8S": "Binance 1",
    "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9": "Binance 2",
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM": "Binance 3",
    "3gd3dqgtJ4jWfBfLYTX67DALFetjc5iS72sCgRhCkW2u": "Binance 4",
    # Coinbase
    "H8sMJSCQxfKiFTCfDR3DUMLPwcRbM61LGFJ8N4dK3WjS": "Coinbase 1",
    "2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm": "Coinbase 2",
    "GJRs4FwHtemZ5ZE9x3FNvJ8TMwitKTh21yxdRPqn7npE": "Coinbase 3",
    # OKX
    "5VCwKtCXgCJ6kit5FybXjvriW3xELsFDhYrPSqtJNmcD": "OKX 1",
    "AobVSwdW9BbpMdJvTqeCN4hPAmh4rHm7vwLnQ5ATSyrS": "OKX 2",
    # Kraken
    "FWznbcNXWQuHTawe9RxvQ2LdCENssh12dsznf4RiouN5": "Kraken",
    # Bybit
    "AC5RDfQFmDS1deWZos921JfqscXdByf8BKHs5ACWjtW2": "Bybit",
    # KuCoin
    "BmFdpraQhkiDQE6SnfG5omcA1VwzqfXrwtNYBwWTymy6": "KuCoin",
    # Crypto.com
    "6RSutwAoRcQPAMwyxZdNeG76fdAxzhgxkCJXpqKCBPdJ": "Crypto.com",
    # Gate.io
    "u6PJ8DtQuPFnfmwHbGFULQ4u4EgjDiyYKjVEsynXq2w":  "Gate.io",
}


def _is_solana_cex(addr: str) -> bool:
    return addr in _KNOWN_CEX_SOLANA


def _solana_cex_label(addr: str) -> str:
    return _KNOWN_CEX_SOLANA.get(addr, "Unknown")


def _fetch_spl_transfers(mint: str, api_key: str,
                         n_recent: int = 100) -> Optional[list]:
    """
    Fetch recent SPL token transfers for a given mint from Solscan Pro v2.
    """
    cache_key = f"solflow:{mint}"
    cached = _cache_get(cache_key, "solana_flow")
    if cached is not None:
        return cached

    try:
        # v2.0 transfers endpoint — pagination param uses page_size / page
        params = {
            "token":     mint,
            "page":      1,
            "page_size": n_recent,
            "sort_by":   "block_time",
            "sort_order": "desc",
        }
        resp = requests.get(
            f"{_SOLSCAN_BASE}/token/transfer",
            params=params,
            headers={
                "token":      api_key,
                "User-Agent": "Mozilla/5.0 PulseIntel/1.0",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        data = resp.json() or {}
        # Solscan v2 returns { "success": true, "data": [...] }
        if not data.get("success", False):
            return None
        result = data.get("data", []) or []
        _cache_set(cache_key, result)
        return result
    except Exception:
        return None


def get_solana_flow_intel(symbol: str, api_key: str = "") -> dict:
    """
    Solana SPL-token CEX flow intel. Same return shape as get_exchange_flow_intel.
    """
    base = _normalize_symbol(symbol)

    if base not in _SYMBOL_TO_SPL_MINT:
        return {
            "ok": False, "supported": False,
            "score": 0, "label": "N/A", "color": "#8892b0",
            "detail": (
                f"{base} is not a Solana SPL token — it lives on another chain. "
                f"Solscan flow only applies to native SPL tokens (JUP/RAY/JTO/WIF/BONK/PYTH/etc)."
            ),
            "data": {"mint": "", "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"}, "needs_api_key": False},
        }

    if not api_key:
        return {
            "ok": False, "supported": True,
            "score": 0, "label": "NO KEY", "color": "#e3b341",
            "detail": (
                "Solscan Pro API key required. Get one at "
                "pro-api.solscan.io (free tier available) "
                "and paste it in the Pulse sidebar."
            ),
            "data": {"mint": _SYMBOL_TO_SPL_MINT[base], "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"}, "needs_api_key": True},
        }

    mint = _SYMBOL_TO_SPL_MINT[base]
    transfers = _fetch_spl_transfers(mint, api_key, n_recent=100)

    if transfers is None:
        return {
            "ok": False, "supported": True,
            "score": 0, "label": "API ERROR", "color": "#f85149",
            "detail": "Solscan API call failed (rate limit, network, or bad key)",
            "data": {"mint": mint, "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"}, "needs_api_key": False},
        }

    if not transfers:
        return {
            "ok": True, "supported": True,
            "score": 0, "label": "NO ACTIVITY", "color": "#8892b0",
            "detail": "No recent SPL transfers found",
            "data": {"mint": mint, "n_transfers": 0,
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": 0, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"}, "needs_api_key": False},
        }

    n_inflows = n_outflows = 0
    inflow_usd = outflow_usd = 0.0
    timestamps = []
    # NEW: mirror of Etherscan whale_txs — used to emit top_transactions
    whale_txs: list = []

    for tx in transfers:
        try:
            ts = int(tx.get("block_time", 0) or 0)
            timestamps.append(ts)
            from_addr = tx.get("from_address") or tx.get("src") or ""
            to_addr   = tx.get("to_address")   or tx.get("dst") or ""
            amount    = float(tx.get("amount", 0) or 0)
            decimals  = int(tx.get("token_decimals", 6) or 6)
            # Solscan v2 sometimes returns amount as raw integer, sometimes decimal.
            # Heuristic: if amount is absurdly large, treat as raw (apply decimals).
            if amount > 1e12:
                amount = amount / (10 ** decimals)
            # Solscan v2 doesn't always include USD value — optional field
            amt_usd = float(tx.get("value", 0) or 0)

            from_cex = _is_solana_cex(from_addr)
            to_cex   = _is_solana_cex(to_addr)

            if to_cex and not from_cex:
                n_inflows  += 1
                inflow_usd += amt_usd
                whale_txs.append({
                    "direction": "inflow",
                    "amt_usd":   amt_usd,
                    "amount":    amount,
                    "cex":       _solana_cex_label(to_addr),
                    "hash":      tx.get("trans_id") or tx.get("txHash") or tx.get("signature") or "",
                    "ts":        ts,
                })
            elif from_cex and not to_cex:
                n_outflows  += 1
                outflow_usd += amt_usd
                whale_txs.append({
                    "direction": "outflow",
                    "amt_usd":   amt_usd,
                    "amount":    amount,
                    "cex":       _solana_cex_label(from_addr),
                    "hash":      tx.get("trans_id") or tx.get("txHash") or tx.get("signature") or "",
                    "ts":        ts,
                })
        except Exception:
            continue

    if len(timestamps) >= 2:
        window_seconds = max(timestamps) - min(timestamps)
        window_hours   = round(window_seconds / 3600.0, 2)
    else:
        window_hours = 0.0

    net_flow_usd = outflow_usd - inflow_usd
    cex_total    = n_inflows + n_outflows

    # Score
    if cex_total == 0:
        return {
            "ok": True, "supported": True,
            "score": 0, "label": "NO CEX FLOW", "color": "#8892b0",
            "detail": (
                f"{len(transfers)} recent SPL transfers, none to known CEX wallets. "
                f"Either low CEX activity or our Solana CEX list is stale."
            ),
            "data": {"mint": mint, "n_transfers": len(transfers),
                     "n_cex_inflows": 0, "n_cex_outflows": 0,
                     "inflow_usd": 0, "outflow_usd": 0, "net_flow_usd": 0,
                     "window_hours": window_hours, "top_transactions": {"outflows": [], "inflows": [], "sort_by": "amt_usd"}, "needs_api_key": False},
        }

    # Scoring: prefer USD magnitude; fall back to count imbalance if
    # USD data is missing (Solscan v2 sometimes omits `value`).
    has_usd = (inflow_usd + outflow_usd) > 0
    if has_usd:
        flow_m = net_flow_usd / 1_000_000.0
        if   flow_m >=  5.0: mag_score = 10
        elif flow_m >=  2.0: mag_score =  7
        elif flow_m >=  0.5: mag_score =  4
        elif flow_m >= -0.5: mag_score =  0
        elif flow_m >= -2.0: mag_score = -4
        elif flow_m >= -5.0: mag_score = -7
        else:                mag_score = -10
    else:
        # Count-based scoring when USD is unavailable
        if cex_total < 5:
            mag_score = 0
        else:
            out_ratio = n_outflows / cex_total
            if   out_ratio >= 0.75: mag_score = 7
            elif out_ratio >= 0.60: mag_score = 4
            elif out_ratio >= 0.40: mag_score = 0
            elif out_ratio >= 0.25: mag_score = -4
            else:                   mag_score = -7

    if   mag_score >=  7: label, color = "STRONG OUTFLOW",  "#3fb950"
    elif mag_score >=  3: label, color = "OUTFLOW",         "#64ffda"
    elif mag_score >= -2: label, color = "BALANCED",        "#8892b0"
    elif mag_score >= -6: label, color = "INFLOW",          "#f0883e"
    else:                 label, color = "STRONG INFLOW",   "#f85149"

    if has_usd:
        _mag = abs(net_flow_usd)
        _mag_str = (f"${_mag/1e6:.2f}M" if _mag >= 1e6
                    else f"${_mag/1e3:.0f}K" if _mag >= 1e3
                    else f"${_mag:.0f}")
        detail = (
            f"SPL net {label.lower()}: {_mag_str} over ~{window_hours:.1f}h "
            f"({n_outflows} withdrawals / {n_inflows} deposits to CEX)"
        )
    else:
        detail = (
            f"SPL: {n_outflows} withdrawals / {n_inflows} deposits "
            f"over ~{window_hours:.1f}h (Solscan omitted USD — count-based score)"
        )

    return {
        "ok": True, "supported": True,
        "score": mag_score, "label": label, "color": color, "detail": detail,
        "data": {
            "mint":           mint,
            "n_transfers":    len(transfers),
            "n_cex_inflows":  n_inflows,
            "n_cex_outflows": n_outflows,
            "inflow_usd":     inflow_usd,
            "outflow_usd":    outflow_usd,
            "net_flow_usd":   net_flow_usd,
            "window_hours":   window_hours,
            # NEW: Whale transactions — top 5 per direction. When Solscan
            # omits USD values we still rank by on-chain token amount.
            # See _build_top_whale_txs for the sort_by flag.
            "top_transactions": _build_top_whale_txs(
                whale_txs,
                price_usd=(inflow_usd + outflow_usd) / max(1, (n_inflows + n_outflows)) if has_usd else 0.0,
            ),
            "needs_api_key":  False,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# LunarCrush social module (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
# Social/sentiment layer. Needs a free LunarCrush API key from
# lunarcrush.com/developers (free tier: limited queries/month).
#
# Galaxy Score (0-100): LC's blended score combining price and social.
# Sentiment (0-100): % of recent posts that are positive.
# AltRank: relative ranking vs. all tracked coins (lower = better).
#
# Scoring philosophy: social is confirmation only — never primary. Use as
# a ±2 modifier in the composite, not as a driver.
# ─────────────────────────────────────────────────────────────────────────────
_LUNARCRUSH_BASE = "https://lunarcrush.com/api4"


def _fetch_lunarcrush_coin(symbol_upper: str, api_key: str) -> Optional[dict]:
    """Fetch LunarCrush v4 coin data. Bearer auth.

    Returns one of:
      - dict with coin data on success
      - {"_lc_error": True, "status": int, "message": str} on HTTP failure
        (lets the caller surface WHY it failed, not just "API ERROR")
      - None only on network exception

    LunarCrush v4 topic/coin lookups require LOWERCASE (e.g. `aave` not `AAVE`).
    The /public/coins/:coin/v1 endpoint is the lightest-weight endpoint and is
    what the free Individual plan exposes; Builder+ plans unlock more endpoints.
    If you're on the "Discover" free plan it may return 402/403 — the error
    dict shape exposes that to the UI instead of silently masking it.
    """
    cache_key = f"lunar:{symbol_upper}"
    cached = _cache_get(cache_key, "social")
    if cached is not None:
        return cached
    # v4 endpoint requires lowercase coin slug
    coin_slug = symbol_upper.lower().strip()
    try:
        url = f"{_LUNARCRUSH_BASE}/public/coins/{coin_slug}/v1"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent":    "Mozilla/5.0 PulseIntel/1.0",
                "Accept":        "application/json",
            },
            timeout=12,
        )
        if resp.status_code != 200:
            # Build a diagnostic envelope. Cache it so we don't hammer a failing
            # endpoint on every rerun, but short-cache so key rotation recovers.
            _msg_map = {
                401: "Invalid API key (401 Unauthorized) — check key and Bearer auth.",
                402: "Payment required (402) — the free 'Discover' plan doesn't include API access. You need the Individual plan ($24/mo) or higher, OR generate a key while a trial/Builder plan is active.",
                403: "Forbidden (403) — key may be revoked or tier doesn't cover /public/coins endpoint.",
                404: f"Coin '{coin_slug}' not tracked by LunarCrush (404). Try major tokens like eth/btc/sol.",
                429: "Rate limit exceeded (429) — wait or upgrade tier for more RPM.",
            }
            # Try to extract the API's own error body for extra detail
            try:
                _body = resp.json()
                _body_msg = (_body.get("error") or _body.get("message")
                             or _body.get("config", {}).get("error_message") or "")
            except Exception:
                _body_msg = (resp.text or "")[:200]
            err = {
                "_lc_error": True,
                "status":    resp.status_code,
                "message":   _msg_map.get(resp.status_code,
                                          f"HTTP {resp.status_code}: {_body_msg[:160]}")
            }
            _cache_set(cache_key, err)
            return err
        data = resp.json() or {}
        payload = data.get("data")
        if not payload:
            err = {"_lc_error": True, "status": 200,
                   "message": f"Coin '{coin_slug}' returned empty payload (not actively tracked)."}
            _cache_set(cache_key, err)
            return err
        _cache_set(cache_key, payload)
        return payload
    except Exception as e:
        # Network-level failure — not cached so retry is cheap
        return {"_lc_error": True, "status": 0,
                "message": f"Network error: {str(e)[:120]}"}


def get_social_intel(symbol: str, api_key: str = "") -> dict:
    """
    Social pulse via LunarCrush v4.

    Returns standardized dict with a score in [-10, +10].
    Scoring: Galaxy Score is primary axis. Sentiment adjusts by ±2 when
    it strongly disagrees with Galaxy (divergence warning).
    """
    base = _normalize_symbol(symbol)
    if not base:
        return {
            "ok": False, "supported": False,
            "score": 0, "label": "N/A", "color": "#8892b0",
            "detail": "Symbol could not be normalized",
            "data": {"needs_api_key": False},
        }
    if not api_key:
        return {
            "ok": False, "supported": True,
            "score": 0, "label": "NO KEY", "color": "#e3b341",
            "detail": (
                "LunarCrush API key required. Free tier at "
                "lunarcrush.com/developers — paste it in the sidebar."
            ),
            "data": {"needs_api_key": True, "symbol": base},
        }

    data = _fetch_lunarcrush_coin(base, api_key)
    # Surface the actual failure reason instead of the generic "API ERROR"
    if isinstance(data, dict) and data.get("_lc_error"):
        return {
            "ok": False, "supported": True,
            "score": 0, "label": f"API {data.get('status',0)}", "color": "#f85149",
            "detail": data.get("message", "Unknown LunarCrush error"),
            "data": {"needs_api_key": False, "symbol": base,
                     "http_status": data.get("status", 0)},
        }
    if not data:
        return {
            "ok": False, "supported": True,
            "score": 0, "label": "API ERROR", "color": "#f85149",
            "detail": f"LunarCrush returned no data for {base} (connection issue).",
            "data": {"needs_api_key": False, "symbol": base},
        }

    # Extract fields with safe coercion. LunarCrush occasionally renames
    # fields between API versions — fall back to alternatives where known.
    def _f(key, *alts, default=0.0):
        for k in (key, *alts):
            v = data.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    continue
        return float(default)

    galaxy     = _f("galaxy_score")
    sentiment  = _f("sentiment",        "percent_change_sentiment", default=50.0)
    alt_rank   = int(_f("alt_rank", default=9999))
    social_vol = _f("interactions_24h", "social_volume_24h")
    social_dom = _f("social_dominance")

    # Base score from Galaxy Score
    if   galaxy >= 75: mag_score = 7
    elif galaxy >= 60: mag_score = 4
    elif galaxy >= 40: mag_score = 0
    elif galaxy >= 25: mag_score = -4
    else:              mag_score = -7

    # Sentiment divergence modifier — only when sentiment is a meaningful %
    if 0 < sentiment <= 100:
        if galaxy >= 55 and sentiment < 35:
            mag_score -= 2   # galaxy bullish but crowd is bearish → mixed
        elif galaxy <= 45 and sentiment > 70:
            mag_score += 2   # galaxy bearish but crowd is bullish → contrarian bounce risk

    mag_score = max(-10, min(10, int(mag_score)))

    if   mag_score >=  5: label, color = "HOT",     "#3fb950"
    elif mag_score >=  2: label, color = "WARM",    "#64ffda"
    elif mag_score >= -1: label, color = "NEUTRAL", "#8892b0"
    elif mag_score >= -4: label, color = "COLD",    "#f0883e"
    else:                 label, color = "FROZEN",  "#f85149"

    detail = (
        f"Galaxy {galaxy:.0f}/100 · Sentiment {sentiment:.0f}% bull · "
        f"AltRank #{alt_rank} · SocDom {social_dom:.2f}%"
    )

    return {
        "ok": True, "supported": True,
        "score": mag_score, "label": label, "color": color, "detail": detail,
        "data": {
            "symbol":           base,
            "galaxy_score":     galaxy,
            "sentiment":        sentiment,
            "alt_rank":         alt_rank,
            "interactions_24h": social_vol,
            "social_dominance": social_dom,
            "needs_api_key":    False,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Macro backdrop module (Phase 3) — NO API KEY NEEDED
# ─────────────────────────────────────────────────────────────────────────────
# Two free signals that modify per-token composites:
#   1. BTC dominance 7d delta (CoinGecko /global — free)
#      Rising BTC.D = capital rotates to BTC, alts suffer
#      Falling BTC.D = alt-favorable macro
#   2. Stablecoin supply 7d delta (DefiLlama /stablecoins — free)
#      Growing stables = dry powder entering crypto, bullish for everything
#      Shrinking stables = capital leaving crypto, bearish for everything
#
# Returns a macro_modifier in [-3, +3] that is ADDED to the per-token
# composite (not blended into the weighted average). This way a
# bullish-on-chain signal during falling-BTC.D + growing stables gets a
# +2 boost, while the same signal during rising-BTC.D + shrinking stables
# gets a -2 penalty — reflecting that macro matters.
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_btc_dominance() -> Optional[dict]:
    """
    Fetch current BTC dominance and approximate 7d delta from CoinGecko /global.
    CoinGecko doesn't serve historical dominance on the free tier, so for
    the "7d delta" we fall back to comparing current vs. 7d-ago BTC market cap
    as a proxy — this isn't perfect but captures direction of flow.
    """
    cache_key = "macro:btc_dom"
    cached = _cache_get(cache_key, "macro_btc_dom")
    if cached is not None:
        return cached
    try:
        # Current dominance
        r1 = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        if r1.status_code != 200:
            return None
        current = r1.json().get("data", {}) or {}
        btc_dom_now = float(current.get("market_cap_percentage", {}).get("btc", 0) or 0)

        # Historical BTC share: CoinGecko's /coins/bitcoin/market_chart gives
        # market_cap history; total crypto market cap via /global/market_cap_chart
        # is not on free. We use a simpler proxy: BTC price 7d change vs total
        # crypto 7d change — if BTC outperformed, dominance likely rose.
        r2 = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": 7, "interval": "daily"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        btc_7d_change = 0.0
        if r2.status_code == 200:
            caps = r2.json().get("market_caps", []) or []
            if len(caps) >= 2:
                first = float(caps[0][1] or 0)
                last  = float(caps[-1][1] or 0)
                btc_7d_change = ((last - first) / first * 100.0) if first > 0 else 0.0

        # Proxy 7d dominance delta: if BTC outperformed ETH by X%, dominance
        # shifted roughly in that direction. Not exact, but direction is right.
        r3 = requests.get(
            "https://api.coingecko.com/api/v3/coins/ethereum/market_chart",
            params={"vs_currency": "usd", "days": 7, "interval": "daily"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        eth_7d_change = 0.0
        if r3.status_code == 200:
            caps = r3.json().get("market_caps", []) or []
            if len(caps) >= 2:
                first = float(caps[0][1] or 0)
                last  = float(caps[-1][1] or 0)
                eth_7d_change = ((last - first) / first * 100.0) if first > 0 else 0.0

        # Rough dominance delta proxy (BTC outperformance vs ETH as a proxy
        # for overall alt underperformance). Scaled down — this isn't real dom delta.
        dom_delta_proxy = round((btc_7d_change - eth_7d_change) * 0.3, 2)

        out = {
            "btc_dominance_now":     btc_dom_now,
            "btc_7d_change_pct":     round(btc_7d_change, 2),
            "eth_7d_change_pct":     round(eth_7d_change, 2),
            "btc_dom_delta_proxy":   dom_delta_proxy,   # approx 7d dominance delta
        }
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


def _fetch_stablecoin_supply() -> Optional[dict]:
    """
    Fetch aggregate stablecoin circulating supply and 7d delta from DefiLlama.
    No API key needed. Returns total supply of USDT + USDC plus 7d % change.
    """
    cache_key = "macro:stables"
    cached = _cache_get(cache_key, "macro_stables")
    if cached is not None:
        return cached
    try:
        # Historical total stablecoin mcap
        resp = requests.get(
            "https://stablecoins.llama.fi/stablecoincharts/all",
            params={"stablecoin": ""},   # all stables
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        if resp.status_code != 200:
            return None
        series = resp.json() or []
        if not isinstance(series, list) or len(series) < 8:
            return None

        # Each entry: {"date": ts_str, "totalCirculatingUSD": {"peggedUSD": amount, ...}}
        def _get_total(row):
            circ = row.get("totalCirculatingUSD", {}) or {}
            if isinstance(circ, dict):
                return float(circ.get("peggedUSD", 0) or 0)
            # Fallback: older schema just had a number
            try:
                return float(circ)
            except Exception:
                return 0.0

        current = _get_total(series[-1])
        seven_ago = _get_total(series[-8]) if len(series) >= 8 else current

        delta_7d_pct = ((current - seven_ago) / seven_ago * 100.0) if seven_ago > 0 else 0.0

        out = {
            "stables_total_now":   current,
            "stables_total_7d":    seven_ago,
            "stables_7d_delta_pct": round(delta_7d_pct, 3),
        }
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


def get_macro_intel() -> dict:
    """
    Macro backdrop — BTC dominance delta + stablecoin supply delta.
    Returns a small modifier in [-3, +3] that gets ADDED to the per-token
    composite in get_pulse_intel (not blended).

    Returns:
      {
        "ok": bool,
        "modifier": int,                  # -3 to +3
        "label": str,                     # "RISK-ON", "NEUTRAL", "RISK-OFF", etc.
        "color": str,
        "detail": str,                    # one-line summary
        "data": {
          "btc_dominance_now": float,
          "btc_dom_delta_proxy": float,
          "stables_7d_delta_pct": float,
          ...
        }
      }
    """
    btc  = _fetch_btc_dominance()
    stab = _fetch_stablecoin_supply()

    if not btc and not stab:
        return {
            "ok": False, "modifier": 0,
            "label": "N/A", "color": "#8892b0",
            "detail": "Macro data unavailable (CoinGecko + DefiLlama both failed)",
            "data": {},
        }

    # Score each axis independently, then sum.
    # BTC dominance: positive delta (BTC outperforming) = bearish for alts
    # Stables: positive delta (supply growing) = bullish for crypto broadly
    btc_mod = 0
    if btc:
        d = btc["btc_dom_delta_proxy"]
        if   d >=  2.0: btc_mod = -2
        elif d >=  0.8: btc_mod = -1
        elif d <= -2.0: btc_mod =  2
        elif d <= -0.8: btc_mod =  1

    stab_mod = 0
    if stab:
        s = stab["stables_7d_delta_pct"]
        if   s >=  1.0: stab_mod =  2
        elif s >=  0.3: stab_mod =  1
        elif s <= -1.0: stab_mod = -2
        elif s <= -0.3: stab_mod = -1

    modifier = max(-3, min(3, btc_mod + stab_mod))

    if   modifier >=  2: label, color = "RISK-ON",          "#3fb950"
    elif modifier >=  1: label, color = "MILDLY RISK-ON",   "#64ffda"
    elif modifier >= -1: label, color = "NEUTRAL",          "#8892b0"
    elif modifier >= -2: label, color = "MILDLY RISK-OFF",  "#f0883e"
    else:                label, color = "RISK-OFF",         "#f85149"

    parts = []
    if btc:
        parts.append(
            f"BTC.D proxy {btc['btc_dom_delta_proxy']:+.2f}% 7d "
            f"(now {btc['btc_dominance_now']:.1f}%)"
        )
    if stab:
        parts.append(f"Stables supply {stab['stables_7d_delta_pct']:+.2f}% 7d")
    detail = " · ".join(parts) + f" → modifier {modifier:+d}"

    return {
        "ok": True, "modifier": int(modifier),
        "label": label, "color": color, "detail": detail,
        "data": {
            "btc": btc or {},
            "stables": stab or {},
            "btc_mod": btc_mod,
            "stab_mod": stab_mod,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Derivatives Pulse module (Phase 4 — added by continuation session)
# ─────────────────────────────────────────────────────────────────────────────
# Reads Binance USDⓈ-M futures public endpoints. NO API KEY NEEDED for any of
# these. For altcoin traders this is arguably the single most actionable
# on-chain-ish layer because derivatives positioning drives spot in the short
# term (funding pays, cascading liquidations, short squeezes).
#
# Three sub-signals blended into one score (-10 to +10):
#
# 1. FUNDING RATE (premiumIndex endpoint) — what longs pay shorts every 8h.
#    Extreme positive = longs crowded, stop-hunt/squeeze risk → bearish tilt.
#    Extreme negative = shorts crowded, short squeeze setup → bullish tilt.
#    Thresholds are in 8h-rate percent (e.g. 0.05% = 0.0005).
#
# 2. OI 24h DELTA (openInterestHist 1d endpoint) — positioning build/unwind.
#    Rising OI + rising price = new longs pushing → bullish follow-through.
#    Rising OI + falling price = new shorts → bearish follow-through.
#    Falling OI = unwinding → trend losing fuel (mild negative for trending).
#    We don't know direction at rate-call time, so we attach direction info
#    to the return dict and let the SCANNER pair it with signal direction.
#
# 3. LONG/SHORT ACCOUNT RATIO (globalLongShortAccountRatio 1d endpoint) —
#    retail positioning. >2.0 = retail crowded long (contrarian bearish).
#    <0.5 = retail crowded short (contrarian bullish). 0.8-1.5 = normal.
#
# Composite direction interpretation:
#    POSITIVE score → conditions favor further UPSIDE
#    NEGATIVE score → conditions favor further DOWNSIDE
#    The signal's own direction then filters: a long signal in POSITIVE
#    derivatives = confluence; a long signal in NEGATIVE derivatives = warn.
#
# Rate limit: Binance futures public = ~2400 req/min per IP — we stay well
# under via the 5-min cache. Each symbol does 3 calls per cache miss.
# ─────────────────────────────────────────────────────────────────────────────
_BINANCE_FUT_BASE = "https://fapi.binance.com"


def _fetch_binance_fut_funding(symbol_usdm: str) -> Optional[dict]:
    """Fetch current funding rate + mark price via /fapi/v1/premiumIndex."""
    cache_key = f"deriv:funding:{symbol_usdm}"
    cached = _cache_get(cache_key, "derivatives")
    if cached is not None:
        return cached
    try:
        resp = requests.get(
            f"{_BINANCE_FUT_BASE}/fapi/v1/premiumIndex",
            params={"symbol": symbol_usdm},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        j = resp.json() or {}
        out = {
            # lastFundingRate is the CURRENT funding rate for the ongoing period
            "funding_rate":      float(j.get("lastFundingRate", 0) or 0),
            "mark_price":        float(j.get("markPrice",       0) or 0),
            "next_funding_time": int(j.get("nextFundingTime",   0) or 0),
        }
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


def _fetch_binance_fut_oi_hist(symbol_usdm: str) -> Optional[dict]:
    """
    Fetch 24h-ago and current Open Interest via /futures/data/openInterestHist.
    Returns absolute OI values in coins + USD and the 24h delta %.

    Uses period=1h, limit=25 to guarantee a 24h span (25 hourly buckets spans
    24h). The earlier period=1d, limit=2 approach sometimes returned only
    a single row for newly-listed pairs, producing N/A. The 1h pattern is
    the same one proven in app.py `fetch_open_interest` (line 724).
    """
    cache_key = f"deriv:oi:{symbol_usdm}"
    cached = _cache_get(cache_key, "derivatives")
    if cached is not None:
        return cached
    try:
        resp = requests.get(
            f"{_BINANCE_FUT_BASE}/futures/data/openInterestHist",
            params={"symbol": symbol_usdm, "period": "1h", "limit": 25},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        rows = resp.json() or []
        if not isinstance(rows, list) or len(rows) < 2:
            _cache_set(cache_key, None)
            return None
        # Binance returns oldest → newest. With 1h period × 25 rows,
        # rows[0] = ~24h ago, rows[-1] = now. If fewer than 25 rows came
        # back (rare — some young contracts), we still use whatever oldest
        # bucket is available as the anchor rather than falling through to
        # N/A. Better a partial-window delta than no signal at all.
        prev = rows[0]
        curr = rows[-1]
        oi_curr_coin = float(curr.get("sumOpenInterest",      0) or 0)
        oi_prev_coin = float(prev.get("sumOpenInterest",      0) or 0)
        oi_curr_usd  = float(curr.get("sumOpenInterestValue", 0) or 0)
        oi_prev_usd  = float(prev.get("sumOpenInterestValue", 0) or 0)
        delta_pct = ((oi_curr_coin - oi_prev_coin) / oi_prev_coin * 100.0) if oi_prev_coin > 0 else 0.0
        out = {
            "oi_now_coin":   oi_curr_coin,
            "oi_now_usd":    oi_curr_usd,
            "oi_24h_coin":   oi_prev_coin,
            "oi_24h_usd":    oi_prev_usd,
            "oi_delta_pct":  round(delta_pct, 3),
            "window_hours":  len(rows) - 1,  # actual span we computed over
        }
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


def _fetch_binance_fut_ls_ratio(symbol_usdm: str) -> Optional[dict]:
    """
    Fetch global long/short ACCOUNT ratio (retail-weighted).
    Endpoint: /futures/data/globalLongShortAccountRatio
    """
    cache_key = f"deriv:lsr:{symbol_usdm}"
    cached = _cache_get(cache_key, "derivatives")
    if cached is not None:
        return cached
    try:
        resp = requests.get(
            f"{_BINANCE_FUT_BASE}/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol_usdm, "period": "1d", "limit": 1},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 PulseIntel/1.0"},
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        rows = resp.json() or []
        if not isinstance(rows, list) or not rows:
            return None
        r = rows[-1]
        out = {
            "ls_ratio":   float(r.get("longShortRatio", 1.0) or 1.0),
            "long_pct":   float(r.get("longAccount",    0.5) or 0.5),
            "short_pct":  float(r.get("shortAccount",   0.5) or 0.5),
        }
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


def get_derivatives_intel(symbol: str) -> dict:
    """
    Derivatives Pulse for a Binance-listed altcoin futures symbol.

    Combines funding rate, OI 24h delta, and retail L/S account ratio into
    a single score in [-10, +10]. Positive = conditions favor upside
    continuation; negative = downside.

    Returns:
      {
        "ok": bool,
        "supported": bool,         # Binance has a USDT-M future for this symbol
        "score": int,              # -10 to +10
        "label": str,              # same 5-bucket scheme as flow modules
        "color": str,
        "detail": str,
        "data": {
          "funding":  {funding_rate, mark_price, next_funding_time},
          "oi":       {oi_now_coin, oi_now_usd, oi_24h_coin, oi_24h_usd, oi_delta_pct},
          "ls":       {ls_ratio, long_pct, short_pct},
          "subscores": {"funding": int, "oi": int, "ls": int},
          "notes": [str, ...],
        }
      }
    """
    base = _normalize_symbol(symbol)
    if not base:
        return {
            "ok": False, "supported": False,
            "score": 0, "label": "N/A", "color": "#8892b0",
            "detail": "Empty/invalid symbol", "data": {},
        }

    # Binance USDⓈ-M perpetual symbol convention: {BASE}USDT
    symbol_usdm = f"{base}USDT"

    funding = _fetch_binance_fut_funding(symbol_usdm)
    oi      = _fetch_binance_fut_oi_hist(symbol_usdm)
    ls      = _fetch_binance_fut_ls_ratio(symbol_usdm)

    # If funding lookup itself failed AND nothing else came back either, the
    # coin probably has no USDT-M perpetual listed.
    if funding is None and oi is None and ls is None:
        return {
            "ok": False, "supported": False,
            "score": 0, "label": "NO FUTURES", "color": "#8892b0",
            "detail": f"No Binance USDT-M perpetual found for {base}",
            "data": {},
        }

    # ── Sub-score 1: Funding rate ────────────────────────────────────────────
    # Funding comes as a raw rate (0.0001 = 0.01% per 8h period). Thresholds
    # below in rate space. Signs flipped because extreme long funding is bearish.
    fund_score = 0
    fund_note = ""
    if funding is not None:
        fr = float(funding["funding_rate"])
        if   fr >=  0.0010: fund_score, fund_note = -4, f"Funding {fr*100:+.3f}% — longs heavily crowded (squeeze risk)"
        elif fr >=  0.0005: fund_score, fund_note = -2, f"Funding {fr*100:+.3f}% — longs mildly crowded"
        elif fr >= -0.0001: fund_score, fund_note =  0, f"Funding {fr*100:+.3f}% — neutral"
        elif fr >= -0.0005: fund_score, fund_note = +2, f"Funding {fr*100:+.3f}% — shorts starting to pay"
        else:               fund_score, fund_note = +4, f"Funding {fr*100:+.3f}% — shorts heavily crowded (squeeze setup)"

    # ── Sub-score 2: OI 24h delta ────────────────────────────────────────────
    # OI alone is direction-agnostic. We give a MILD positive tilt to rising OI
    # (trend fueled by new money), mild negative to falling OI (unwind). The
    # scanner layer can combine with price direction for stronger conviction.
    oi_score = 0
    oi_note = ""
    if oi is not None:
        d = float(oi["oi_delta_pct"])
        if   d >=  15.0: oi_score, oi_note = +3, f"OI +{d:.1f}% 24h — heavy new positioning (trend fuel)"
        elif d >=   5.0: oi_score, oi_note = +2, f"OI +{d:.1f}% 24h — meaningful positioning build"
        elif d >=  -5.0: oi_score, oi_note =  0, f"OI {d:+.1f}% 24h — stable"
        elif d >= -15.0: oi_score, oi_note = -2, f"OI {d:.1f}% 24h — unwinding (trend losing fuel)"
        else:            oi_score, oi_note = -3, f"OI {d:.1f}% 24h — heavy deleveraging"

    # ── Sub-score 3: Long/Short account ratio (contrarian) ────────────────────
    ls_score = 0
    ls_note = ""
    if ls is not None:
        r = float(ls["ls_ratio"])
        if   r >=  3.0: ls_score, ls_note = -3, f"L/S {r:.2f} — retail euphorically long (contrarian sell)"
        elif r >=  2.0: ls_score, ls_note = -2, f"L/S {r:.2f} — retail crowded long"
        elif r >=  0.8: ls_score, ls_note =  0, f"L/S {r:.2f} — balanced positioning"
        elif r >=  0.5: ls_score, ls_note = +2, f"L/S {r:.2f} — retail crowded short"
        else:           ls_score, ls_note = +3, f"L/S {r:.2f} — retail extremely short (contrarian buy)"

    # Composite: weighted average. Funding and L/S are interpretation-ready
    # (sign is right). OI is direction-agnostic so we weight it less.
    # Weights: funding 45%, LS 35%, OI 20%. Scaled to [-10, +10] envelope.
    weighted = (fund_score * 0.45 + ls_score * 0.35 + oi_score * 0.20)
    # Scale: sub-scores max ±4 for funding, ±3 for LS, ±3 for OI. Weighted max:
    #   4*0.45 + 3*0.35 + 3*0.20 = 1.8 + 1.05 + 0.6 = 3.45
    # Multiply by ~2.9 to map to ±10 envelope.
    score = int(round(weighted * 2.9))
    score = max(-10, min(10, score))

    if   score >=  7: label, color = "STRONG BULLISH DERIV", "#3fb950"
    elif score >=  3: label, color = "BULLISH DERIV",        "#64ffda"
    elif score >= -2: label, color = "NEUTRAL DERIV",        "#8892b0"
    elif score >= -6: label, color = "BEARISH DERIV",        "#f0883e"
    else:             label, color = "STRONG BEARISH DERIV", "#f85149"

    notes = [n for n in (fund_note, oi_note, ls_note) if n]
    detail = " · ".join(notes) if notes else "Partial derivatives data available"

    return {
        "ok":        True,
        "supported": True,
        "score":     score,
        "label":     label,
        "color":     color,
        "detail":    detail,
        "data": {
            "funding":   funding or {},
            "oi":        oi or {},
            "ls":        ls or {},
            "subscores": {
                "funding": fund_score,
                "oi":      oi_score,
                "ls":      ls_score,
            },
            "notes": notes,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Binance Futures Leaderboard — smart money proxy (Phase 5 — free, no key)
# ─────────────────────────────────────────────────────────────────────────────
# Why this exists: true wallet-labeled "smart money" (Nansen-style) isn't free.
# The strongest free approximation is Binance's public futures leaderboard —
# the top-ROI traders who opted to share their positions. These are demonstrably
# profitable (ranked by actual ROI), and seeing what % of them are LONG vs
# SHORT a given symbol is a real directional signal.
#
# Caveats (be honest with the user):
#   1. Leaderboard is SELF-SELECTED — traders opt in to share positions.
#      Biases toward traders who want followers; not a true random sample.
#   2. ROI rankings are gameable — someone up 1000% on a $500 account still
#      ranks above a fund up 20% on $10M. We filter by absolute PnL floor
#      (≥$50K) to weed out obvious small-account noise.
#   3. Binance's /bapi/ endpoints are INTERNAL (used by their website, not
#      their official REST API). They can change or block them without notice.
#      Module returns supported=False on any failure — composite still works.
#   4. Rate limit: we cap at 20 position fetches per call to stay fast. That's
#      enough for directional bias; precision improves marginally beyond it.
#
# Still strictly better than nothing — and strictly better than the old
# "FUTURE" placeholder that did nothing.
# ─────────────────────────────────────────────────────────────────────────────
_BINANCE_LEADERBOARD_BASE = "https://www.binance.com/bapi/futures/v1/public/future/leaderboard"
_LB_MAX_TRADERS           = 20    # cap per-trader position fetches to bound latency
_LB_MIN_PNL_USD           = 50_000  # absolute PnL floor — filters tiny-account top-ROI outliers


def _fetch_binance_leaderboard_rank(periodType: str = "EXACT_WEEKLY") -> Optional[list]:
    """
    Fetch the top ROI traders from Binance's public leaderboard.

    periodType options (per Binance's internal API):
      - "EXACT_WEEKLY"  — last 7 days
      - "EXACT_MONTHLY" — last 30 days
      - "EXACT_YEARLY"  — last year
      - "ALL"           — all-time

    We default to weekly: recent ROI is more relevant for "what are smart
    traders doing RIGHT NOW" than all-time stars who may be dormant.
    statisticsType="ROI" ranks by return, "PNL" by dollar amount.
    """
    cache_key = f"lb:rank:{periodType}"
    cached = _cache_get(cache_key, "derivatives")
    if cached is not None:
        return cached
    try:
        resp = requests.post(
            f"{_BINANCE_LEADERBOARD_BASE}/getLeaderboardRank",
            json={
                "isShared":       True,           # only traders sharing positions
                "isTrader":       False,
                "periodType":     periodType,
                "statisticsType": "ROI",
                "tradeType":      "PERPETUAL",
            },
            headers={
                "User-Agent":   "Mozilla/5.0 PulseIntel/1.0",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        payload = resp.json() or {}
        rows = payload.get("data") or []
        if not isinstance(rows, list):
            _cache_set(cache_key, None)
            return None
        _cache_set(cache_key, rows)
        return rows
    except Exception:
        return None


def _fetch_trader_positions(encrypted_uid: str) -> Optional[list]:
    """
    Fetch a single trader's open perpetual positions.
    Returns list of position dicts or None on failure.
    """
    cache_key = f"lb:pos:{encrypted_uid}"
    cached = _cache_get(cache_key, "derivatives")
    if cached is not None:
        return cached
    try:
        resp = requests.post(
            f"{_BINANCE_LEADERBOARD_BASE}/getOtherPosition",
            json={
                "encryptedUid": encrypted_uid,
                "tradeType":    "PERPETUAL",
            },
            headers={
                "User-Agent":   "Mozilla/5.0 PulseIntel/1.0",
                "Content-Type": "application/json",
            },
            timeout=6,
        )
        if resp.status_code != 200:
            _cache_set(cache_key, None)
            return None
        payload = resp.json() or {}
        data = payload.get("data") or {}
        # Per observed shape: data.otherPositionRetList = [positions...]
        # Some variants nest under "perpetual". Handle both.
        positions = (data.get("otherPositionRetList")
                     or (data.get("positions") or {}).get("perpetual")
                     or [])
        if not isinstance(positions, list):
            return None
        _cache_set(cache_key, positions)
        return positions
    except Exception:
        return None


def get_binance_leaderboard_intel(symbol: str) -> dict:
    """
    Smart-money proxy via Binance Futures Leaderboard.

    Aggregates positions from top-ROI traders (who share positions) on the
    target symbol. Returns:
      - score in [-10, +10]  (skewed toward positioning agreement strength)
      - label, color, detail + data.subscores for transparency

    Scoring logic:
      score = (long_pct - short_pct) * 10   clamped to [-10, +10]
      Example: 80% long / 20% short = +6 score
               50% long / 50% short = 0
               30% long / 70% short = -4

    We require ≥3 traders actively positioned on the symbol for a valid
    signal — smaller samples are too noisy. Below that threshold we return
    supported=True but score=0 with a "low coverage" note.
    """
    base = _normalize_symbol(symbol)
    sym_usdm = f"{base}USDT"   # Binance futures leaderboard positions use USDT-M symbol

    empty_data = {
        "n_traders_scanned":  0,
        "n_traders_on_sym":   0,
        "n_long":             0,
        "n_short":            0,
        "long_pct":           0.0,
        "short_pct":          0.0,
        "total_notional_usd": 0.0,
        "subscores":          {},
    }

    rank_rows = _fetch_binance_leaderboard_rank("EXACT_WEEKLY")
    if not rank_rows:
        return {
            "ok": False, "supported": False,
            "score": 0, "label": "N/A", "color": "#8892b0",
            "detail": "Binance leaderboard endpoint unavailable (may be rate-limited or blocked regionally)",
            "data": empty_data,
        }

    # Filter: only traders with real money (absolute PnL floor to avoid the
    # "up 5000% on $100" type entries that top ROI rankings attract).
    filtered = [r for r in rank_rows
                if abs(float(r.get("pnlValue", 0) or 0)) >= _LB_MIN_PNL_USD]
    if not filtered:
        filtered = rank_rows   # if nobody passes, fall back to raw list

    # Cap to _LB_MAX_TRADERS to keep latency reasonable (20 traders ≈ 6s).
    candidates = filtered[:_LB_MAX_TRADERS]
    n_scanned = len(candidates)

    n_long = 0
    n_short = 0
    total_notional = 0.0
    for trader in candidates:
        uid = trader.get("encryptedUid") or ""
        if not uid:
            continue
        positions = _fetch_trader_positions(uid) or []
        # Find position on our target symbol
        for pos in positions:
            psym = (pos.get("symbol") or "").upper()
            if psym != sym_usdm:
                continue
            is_long  = bool(pos.get("long",  False))
            is_short = bool(pos.get("short", False))
            # Some responses don't set long/short booleans — infer from amount sign.
            if not is_long and not is_short:
                amt = float(pos.get("amount", 0) or 0)
                if amt > 0:   is_long  = True
                elif amt < 0: is_short = True
            if is_long:
                n_long += 1
            elif is_short:
                n_short += 1
            # Accumulate notional (|amount| × markPrice) for context
            try:
                amt_abs = abs(float(pos.get("amount", 0) or 0))
                mark    = float(pos.get("markPrice", 0) or 0)
                total_notional += amt_abs * mark
            except Exception:
                pass
            break   # one position per trader on a given symbol

    n_on_sym = n_long + n_short
    # Need at least 3 traders positioned to make a directional claim
    if n_on_sym < 3:
        empty_data.update({
            "n_traders_scanned": n_scanned,
            "n_traders_on_sym":  n_on_sym,
            "n_long":            n_long,
            "n_short":           n_short,
            "total_notional_usd": round(total_notional, 0),
        })
        return {
            "ok": True, "supported": True,
            "score": 0, "label": "LOW COVERAGE", "color": "#8892b0",
            "detail": (
                f"Only {n_on_sym} of {n_scanned} top ROI traders have an open "
                f"{sym_usdm} position — too few for a directional read."
            ),
            "data": empty_data,
        }

    long_pct  = (n_long  / n_on_sym) * 100.0
    short_pct = (n_short / n_on_sym) * 100.0

    # Score: (long% − short%) / 10, clamped to [-10, +10]
    raw_score = (long_pct - short_pct) / 10.0
    score = int(round(max(-10, min(10, raw_score))))

    # Labeling — uses the long-minus-short bias as the severity axis
    if   score >=  6: label, color = "SMART LONG",      "#3fb950"
    elif score >=  3: label, color = "LEANING LONG",    "#64ffda"
    elif score >= -2: label, color = "SPLIT",           "#8892b0"
    elif score >= -5: label, color = "LEANING SHORT",   "#f0883e"
    else:             label, color = "SMART SHORT",     "#f85149"

    # Detail string — cite the actual numbers so user can judge sample quality
    detail = (
        f"{n_long}/{n_on_sym} top-ROI traders LONG "
        f"({long_pct:.0f}%) vs {n_short}/{n_on_sym} SHORT ({short_pct:.0f}%) "
        f"on {sym_usdm}. Total notional: ${total_notional/1e6:.1f}M. "
        f"Sampled top {n_scanned} weekly-ROI leaderboard traders (PnL ≥ ${_LB_MIN_PNL_USD/1000:.0f}K)."
    )

    return {
        "ok": True, "supported": True,
        "score": score, "label": label, "color": color,
        "detail": detail,
        "data": {
            "n_traders_scanned":  n_scanned,
            "n_traders_on_sym":   n_on_sym,
            "n_long":             n_long,
            "n_short":            n_short,
            "long_pct":           round(long_pct, 1),
            "short_pct":          round(short_pct, 1),
            "total_notional_usd": round(total_notional, 0),
            "subscores":          {"long_minus_short_pct": round(long_pct - short_pct, 1)},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite — Phase 5: blend TVL + Flow + Social + Derivatives + Leaderboard + Macro
# ─────────────────────────────────────────────────────────────────────────────
def get_pulse_intel(symbol: str,
                     etherscan_api_key: str = "",
                     lunarcrush_api_key: str = "",
                     solscan_api_key:    str = "") -> dict:
    """
    Main entry point. Returns the full Pulse intelligence dict for a symbol.

    Phase 5 (current): TVL + Etherscan Flow + Solscan Flow + LunarCrush Social
                       + Macro backdrop + Binance Derivatives Pulse
                       + Binance Futures Leaderboard (smart money proxy).

    Returns:
      {
        "symbol": str, "base_token": str,
        "tvl":          { ...get_tvl_intel(...) },
        "exchange_flow":{ ...get_exchange_flow_intel(...) },     # ETH
        "solana_flow":  { ...get_solana_flow_intel(...) },       # SOL
        "social":       { ...get_social_intel(...) },
        "macro":        { ...get_macro_intel() },
        "derivatives":  { ...get_derivatives_intel(...) },       # Phase 4
        "leaderboard":  { ...get_binance_leaderboard_intel(...) }, # Phase 5
        "composite_score":  int,   # -15 to +15 (includes macro modifier)
        "composite_label":  str,
        "composite_color":  str,
        "verdict_summary":  str,
        "phase":            str,
      }

    Composite weighting (per-token, before macro modifier):
        - Exchange Flow (ETH or SOL, whichever applies): 30%
        - TVL:                                           20%
        - Social:                                        15%
        - Derivatives (funding + OI + L/S):              20%
        - Leaderboard (smart money proxy):               15%
      If a module is unavailable, its weight is redistributed proportionally.
      The blended per-token score is then scaled to ±12, and the macro
      modifier (±3) is added to produce the final composite in [-15, +15].

    Why Leaderboard gets 15% and not more: it's a directional-bias signal
    (positioning), not a flow magnitude signal. Per-trader positions aren't
    weighted by size in the aggregation. The heavy lifters for magnitude
    remain Flow and TVL; Leaderboard adds a "what do provably profitable
    traders currently believe" layer that was missing.
    """
    base_token = _normalize_symbol(symbol)

    # ── Fetch all modules in parallel-ish order (sequential for simplicity) ──
    tvl         = get_tvl_intel(symbol)
    eth_flow    = get_exchange_flow_intel(symbol, api_key=etherscan_api_key)
    sol_flow    = get_solana_flow_intel(symbol,   api_key=solscan_api_key)
    social      = get_social_intel(symbol,        api_key=lunarcrush_api_key)
    macro       = get_macro_intel()
    derivatives = get_derivatives_intel(symbol)          # Phase 4 — no API key
    leaderboard = get_binance_leaderboard_intel(symbol)  # Phase 5 — no API key

    # For composite purposes, use whichever flow is actually providing data.
    # Never double-count: a token either lives on Ethereum OR Solana in our maps.
    if eth_flow.get("ok") and eth_flow.get("supported"):
        active_flow = eth_flow
        flow_chain  = "ETH"
    elif sol_flow.get("ok") and sol_flow.get("supported"):
        active_flow = sol_flow
        flow_chain  = "SOL"
    else:
        active_flow = None
        flow_chain  = "—"

    # ── Build per-token weighted composite ────────────────────────────────────
    # Phase 5 weights: Flow 30% / TVL 20% / Social 15% / Derivatives 20% / Leaderboard 15%
    # If a module is N/A (ok=False or supported=False), drop its weight and
    # re-normalize the remaining weights.
    raw_scores  = []
    raw_weights = []

    if tvl.get("ok") and tvl.get("supported"):
        raw_scores.append(float(tvl["score"]))
        raw_weights.append(0.20)

    if active_flow:
        raw_scores.append(float(active_flow["score"]))
        raw_weights.append(0.30)

    if social.get("ok") and social.get("supported"):
        raw_scores.append(float(social["score"]))
        raw_weights.append(0.15)

    if derivatives.get("ok") and derivatives.get("supported"):
        raw_scores.append(float(derivatives["score"]))
        raw_weights.append(0.20)

    # Leaderboard counts only when we had enough coverage (score != 0 OR label
    # is one of the real verdicts — "LOW COVERAGE" returns ok=True, score=0
    # but we don't want that zero to pull the composite toward neutral
    # artificially. Skip when coverage was low.)
    _lb_usable = (leaderboard.get("ok") and leaderboard.get("supported")
                  and leaderboard.get("label") != "LOW COVERAGE")
    if _lb_usable:
        raw_scores.append(float(leaderboard["score"]))
        raw_weights.append(0.15)

    if raw_weights:
        total_w = sum(raw_weights)
        weighted = sum(s * w for s, w in zip(raw_scores, raw_weights)) / total_w
        per_token = int(round(weighted * 1.2))
        per_token = max(-12, min(12, per_token))
    else:
        per_token = 0

    macro_mod = int(macro.get("modifier", 0) or 0) if macro.get("ok") else 0
    composite = max(-15, min(15, per_token + macro_mod))

    if   composite >= 10: c_label, c_color = "STRONGLY BULLISH", "#3fb950"
    elif composite >=  4: c_label, c_color = "BULLISH",          "#64ffda"
    elif composite >= -3: c_label, c_color = "NEUTRAL",          "#8892b0"
    elif composite >= -9: c_label, c_color = "BEARISH",          "#f0883e"
    else:                 c_label, c_color = "STRONGLY BEARISH", "#f85149"

    # ── Build verdict summary ────────────────────────────────────────────────
    parts = []
    if tvl.get("ok") and tvl.get("supported"):
        parts.append(f"TVL {tvl['label'].lower()}")
    if active_flow:
        if "OUTFLOW" in active_flow["label"]:
            parts.append(f"{flow_chain} CEX outflows")
        elif "INFLOW" in active_flow["label"]:
            parts.append(f"{flow_chain} CEX inflows")
        else:
            parts.append(f"{flow_chain} flow {active_flow['label'].lower()}")
    if social.get("ok") and social.get("supported"):
        parts.append(f"social {social['label'].lower()}")
    if derivatives.get("ok") and derivatives.get("supported"):
        _d_lbl = derivatives["label"].replace(" DERIV", "").lower()
        parts.append(f"derivatives {_d_lbl}")
    if _lb_usable:
        parts.append(f"smart-money {leaderboard['label'].lower()}")
    if macro.get("ok") and macro_mod != 0:
        parts.append(f"macro {macro['label'].lower()}")

    # Agreement/divergence detection (across the 5 per-token signals)
    _pt_dirs = []
    if tvl.get("ok") and tvl.get("supported"):
        _pt_dirs.append(1 if tvl["score"] > 0 else (-1 if tvl["score"] < 0 else 0))
    if active_flow:
        _pt_dirs.append(1 if active_flow["score"] > 0 else (-1 if active_flow["score"] < 0 else 0))
    if social.get("ok") and social.get("supported"):
        _pt_dirs.append(1 if social["score"] > 0 else (-1 if social["score"] < 0 else 0))
    if derivatives.get("ok") and derivatives.get("supported"):
        _pt_dirs.append(1 if derivatives["score"] > 0 else (-1 if derivatives["score"] < 0 else 0))
    if _lb_usable:
        _pt_dirs.append(1 if leaderboard["score"] > 0 else (-1 if leaderboard["score"] < 0 else 0))
    nonzero = [d for d in _pt_dirs if d != 0]
    if len(nonzero) >= 2 and all(d == nonzero[0] for d in nonzero):
        agreement = " — all signals AGREE"
    elif len(nonzero) >= 2 and len(set(nonzero)) > 1:
        agreement = " — signals DIVERGE (lower conviction)"
    else:
        agreement = ""

    if not parts:
        summary = (
            f"Pulse can't fully track {base_token} — not in our token maps. "
            f"Try a major DeFi token (AAVE/UNI/LDO) or L1 (ETH/SOL/AVAX)."
        )
    else:
        signals_str = " · ".join(parts)
        if composite >= 10:
            summary = (
                f"Strong on-chain bullish convergence: {signals_str}{agreement}. "
                f"Momentum signals have HIGH conviction backing."
            )
        elif composite >= 4:
            summary = (
                f"Moderately bullish on-chain: {signals_str}{agreement}. "
                f"Reasonable confluence for momentum trades."
            )
        elif composite >= -3:
            summary = (
                f"On-chain neutral: {signals_str}{agreement}. "
                f"Trade on technicals — no tailwind nor headwind."
            )
        elif composite >= -9:
            summary = (
                f"Bearish on-chain: {signals_str}{agreement}. "
                f"Momentum LONGS may lack follow-through; SHORTS get extra confluence."
            )
        else:
            summary = (
                f"STRONGLY bearish on-chain: {signals_str}{agreement}. "
                f"Avoid LONGS even on strong momentum — capital is leaving."
            )

    _deriv_tag = " + Deriv" if (derivatives.get("ok") and derivatives.get("supported")) else ""
    _lb_tag    = " + SmartMoney"  if _lb_usable else ""
    phase_label = (f"5 (TVL + {flow_chain} Flow + Social + Macro{_deriv_tag}{_lb_tag})"
                   if active_flow else
                   f"5 (limited modules available{_deriv_tag}{_lb_tag})")

    return {
        "symbol":           symbol,
        "base_token":       base_token,
        "tvl":              tvl,
        "exchange_flow":    eth_flow,
        "solana_flow":      sol_flow,
        "active_flow_chain": flow_chain,
        "social":           social,
        "macro":            macro,
        "derivatives":      derivatives,        # Phase 4 — Binance futures pulse
        "leaderboard":      leaderboard,        # Phase 5 — smart money proxy
        "smart_money":      leaderboard,        # alias kept for back-compat with any caller reading "smart_money"
        "composite_score":  composite,
        "composite_label":  c_label,
        "composite_color":  c_color,
        "verdict_summary":  summary,
        "phase":            phase_label,
        "per_token_score":  per_token,
        "macro_modifier":   macro_mod,
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
