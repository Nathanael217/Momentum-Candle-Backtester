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
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {},
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
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {},
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
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {},
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
                     "window_hours": 0, "top_inflow": {}, "top_outflow": {},
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
            elif from_is_cex and not to_is_cex:
                # Withdrawal from exchange = OUTFLOW = bullish (accumulation)
                n_outflows += 1
                outflow_usd += amt_usd
                if amt_usd > top_outflow["amt_usd"]:
                    top_outflow = {"amt_usd": amt_usd,
                                   "cex": _cex_label(from_addr),
                                   "hash": tx_hash}
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
            "needs_api_key":  False,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite — Phase 2: blend TVL (30%) + Exchange Flow (35%) into composite
# ─────────────────────────────────────────────────────────────────────────────
def get_pulse_intel(symbol: str, etherscan_api_key: str = "") -> dict:
    """
    Main entry point. Returns the full Pulse intelligence dict for a symbol.

    In Phase 1 this only contains TVL. As more sources are added (exchange
    flow, social, smart money), they'll be merged into the composite_score.

    Returns:
      {
        "symbol": str,                    # the original symbol passed in
        "base_token": str,                # normalized (ETH from ETHUSDT)
        "tvl": {...},                     # output of get_tvl_intel()
        "exchange_flow": {...},           # output of get_exchange_flow_intel()  ← NEW Phase 2
        "social": None,                   # Phase 3
        "smart_money": None,              # Phase 3
        "composite_score": int,           # -15 to +15, weighted aggregate
        "composite_label": str,           # BULLISH | BEARISH | NEUTRAL etc.
        "composite_color": str,           # hex for UI
        "verdict_summary": str,           # one-line takeaway
        "phase": str,                     # so the UI can label it
      }

    Composite weighting (Phase 2):
      - Exchange Flow: 50%  (highest signal quality when available)
      - TVL:           50%  (when no flow data, TVL gets effectively 100%)
      Final composite is then scaled up to ±15 range.
    """
    base_token = _normalize_symbol(symbol)
    tvl  = get_tvl_intel(symbol)
    flow = get_exchange_flow_intel(symbol, api_key=etherscan_api_key)

    # ─── Phase 2 composite blending ───────────────────────────────────────
    # Each module returns a score in [-10, +10]. We blend by availability:
    #   - If both modules ok: weighted average then scale to ±15
    #   - If only one ok: use that module's score directly (capped at ±10)
    #   - If neither: 0
    # The weight split (50/50 between TVL and Flow) is tunable. Flow is
    # often more responsive to short-term shifts; TVL captures structural
    # demand. Equal weighting respects both.
    tvl_ok  = tvl["ok"]
    flow_ok = flow["ok"]

    if tvl_ok and flow_ok:
        # Blended composite. Scale up by 1.5 so the full ±15 range is
        # accessible when both signals strongly agree.
        blended = 0.5 * tvl["score"] + 0.5 * flow["score"]
        composite = int(round(blended * 1.5))
        composite = max(-15, min(15, composite))
        phase_label = "2 (TVL + Exchange Flow)"
    elif tvl_ok:
        composite = tvl["score"]   # already in ±10
        phase_label = "2 (TVL only — no flow data)"
    elif flow_ok:
        composite = flow["score"]
        phase_label = "2 (Flow only — no TVL data)"
    else:
        composite = 0
        phase_label = "2 (no data available)"

    if   composite >= 10: c_label, c_color = "STRONGLY BULLISH",   "#3fb950"
    elif composite >=  4: c_label, c_color = "BULLISH",            "#64ffda"
    elif composite >= -3: c_label, c_color = "NEUTRAL",            "#8892b0"
    elif composite >= -9: c_label, c_color = "BEARISH",            "#f0883e"
    else:                 c_label, c_color = "STRONGLY BEARISH",   "#f85149"

    # ─── Build a one-line verdict summary that takes both signals into account
    parts = []
    if tvl_ok and tvl["supported"]:
        parts.append(f"TVL {tvl['label'].lower()}")
    if flow_ok and flow["supported"]:
        # Exchange flow gets a more nuanced descriptor
        if "OUTFLOW" in flow["label"]:
            parts.append(f"net withdrawals from CEX")
        elif "INFLOW" in flow["label"]:
            parts.append(f"net deposits to CEX")
        else:
            parts.append(f"flow {flow['label'].lower()}")

    # Detect agreement vs divergence — this is the most useful insight
    if tvl_ok and flow_ok and tvl["supported"] and flow["supported"]:
        tvl_dir = 1 if tvl["score"] > 0 else (-1 if tvl["score"] < 0 else 0)
        flow_dir = 1 if flow["score"] > 0 else (-1 if flow["score"] < 0 else 0)
        if tvl_dir != 0 and flow_dir != 0 and tvl_dir == flow_dir:
            agreement = " — both signals AGREE"
        elif tvl_dir != 0 and flow_dir != 0:
            agreement = " — signals DIVERGE (treat with caution)"
        else:
            agreement = ""
    else:
        agreement = ""

    if not parts:
        if not tvl["supported"] and not flow["supported"]:
            summary = (
                f"Pulse can't fully track {base_token} — not in our DefiLlama "
                f"or Etherscan ERC-20 maps. Phase 3 (social + smart money) "
                f"will cover the remaining gaps."
            )
        elif flow["data"].get("needs_api_key"):
            summary = (
                f"TVL: {tvl.get('label', 'N/A')}. Add an Etherscan API key "
                f"in the sidebar to also see exchange flow data — it's free "
                f"at etherscan.io/apis (5 calls/sec)."
            )
        else:
            summary = "On-chain data unavailable right now. Treat as neutral."
    else:
        signals_str = " · ".join(parts)
        if composite >= 10:
            summary = (
                f"Strong on-chain bullish convergence: {signals_str}{agreement}. "
                f"Momentum signals here have HIGH conviction backing."
            )
        elif composite >= 4:
            summary = (
                f"Moderately bullish on-chain: {signals_str}{agreement}. "
                f"Reasonable confluence for momentum trades."
            )
        elif composite >= -3:
            summary = (
                f"On-chain is neutral: {signals_str}{agreement}. "
                f"No tailwind nor headwind — trade purely on technicals."
            )
        elif composite >= -9:
            summary = (
                f"Bearish on-chain: {signals_str}{agreement}. "
                f"Momentum LONGS here may lack follow-through; SHORTS get extra confluence."
            )
        else:
            summary = (
                f"STRONGLY bearish on-chain: {signals_str}{agreement}. "
                f"Avoid LONGS even on strong momentum signals — capital is leaving."
            )

    return {
        "symbol":          symbol,
        "base_token":      base_token,
        "tvl":             tvl,
        "exchange_flow":   flow,
        "social":          None,    # Phase 3
        "smart_money":     None,    # Phase 3
        "composite_score":  composite,
        "composite_label":  c_label,
        "composite_color":  c_color,
        "verdict_summary":  summary,
        "phase":           phase_label,
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
