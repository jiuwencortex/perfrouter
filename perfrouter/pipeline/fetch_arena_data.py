#!/usr/bin/env python3
"""
fetch_arena_data.py
===================
Fetches Arena.ai ELO scores from the wulong.dev API for the 5 categories
that cover task types underserved by AA benchmarks, and merges them into
model_registry.json.

─────────────────────────────────────────────────────────────────────
WHY ARENA ELO SCORES
─────────────────────────────────────────────────────────────────────

AA benchmarks cover objective capabilities well (math, code, reasoning)
but have gaps for subjective/preference-based task types:

  social.conversation_analysis       → no clean automated metric
  social.negotiation_or_communication → no clean automated metric
  synthesis.creative_media           → no clean automated metric
  synthesis.creative_writing         → no clean automated metric
  retrieval.web_search               → partially covered by AA-Intel

Arena ELO scores (human preference votes) fill these gaps:
  text     → general text quality, conversation, creative writing
  search   → search and retrieval quality
  code     → coding (supplements AA coding benchmarks)
  vision   → visual understanding (supplements MMMU)
  document → document understanding and extraction

─────────────────────────────────────────────────────────────────────
ELO NORMALISATION
─────────────────────────────────────────────────────────────────────

Raw ELO scores are on an arbitrary scale (~1000-1600 in practice).
We normalise them to [0, 1] per category using min-max normalisation
across all models in that category's leaderboard:

  normalised = (elo - min_elo) / (max_elo - min_elo)

This gives a 0-1 score comparable to benchmark scores.

─────────────────────────────────────────────────────────────────────
MODEL MATCHING
─────────────────────────────────────────────────────────────────────

Arena model names (e.g. "claude-opus-4-6", "gpt-4o-mini-2024-07-18")
don't match our registry IDs (e.g. "anthropic/claude-sonnet-4.6").

Matching strategy (first hit wins):
  1. Exact match after normalisation (lowercase, separators → _)
  2. One name is a substring of the other
  3. Vendor match + partial name overlap

Unmatched models are logged and skipped.

─────────────────────────────────────────────────────────────────────
Usage:
  python3 fetch_arena_data.py --registry model_registry.json
  python3 fetch_arena_data.py --registry model_registry.json --dry-run
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)


# ── Arena categories and their task-type relevance ────────────────────────────

ARENA_CATEGORIES = {
    "text": {
        "field":       "arena_text_elo",
        "description": "General text quality, conversation, creative writing",
        "covers":      [
            "social.conversation_analysis",
            "social.negotiation_or_communication",
            "social.sentiment_analysis",
            "synthesis.creative_media",
            "synthesis.creative_writing",
            "synthesis.report_or_summary",
        ],
    },
    "search": {
        "field":       "arena_search_elo",
        "description": "Search and retrieval quality",
        "covers":      [
            "retrieval.web_search",
            "retrieval.knowledge_qa",
        ],
    },
    "code": {
        "field":       "arena_code_elo",
        "description": "Code generation and debugging quality",
        "covers":      [
            "code.generation",
            "code.debugging",
            "code.code_review",
            "code.scientific_coding",
        ],
    },
    "vision": {
        "field":       "arena_vision_elo",
        "description": "Visual understanding and multimodal reasoning",
        "covers":      [
            "multimodal.visual_reasoning",
            "multimodal.diagram_understanding",
        ],
    },
    "document": {
        "field":       "arena_document_elo",
        "description": "Document understanding and extraction",
        "covers":      [
            "retrieval.document_extraction",
            "synthesis.structured_data",
            "synthesis.data_analysis_report",
        ],
    },
}

ARENA_API_BASE = "https://api.wulong.dev/arena-ai-leaderboards/v1"
ARENA_TIMEOUT  = 15


# ── Vendor name normalisation ──────────────────────────────────────────────────

VENDOR_MAP = {
    "anthropic": ["anthropic"],
    "openai":    ["openai"],
    "google":    ["google", "deepmind"],
    "deepseek":  ["deepseek"],
    "meta":      ["meta", "facebook"],
    "alibaba":   ["alibaba", "qwen"],
    "nvidia":    ["nvidia"],
    "tencent":   ["tencent", "hunyuan"],
    "xiaomi":    ["xiaomi", "mimo"],
    "moonshot":  ["moonshot", "kimi"],
    "moonshotai": ["moonshot", "kimi"],
    "z-ai":      ["z.ai", "zhipu"],
    "minimax":   ["minimax"],
}

def vendor_of(model_id: str) -> str | None:
    """Extract vendor from our registry model ID (e.g. 'anthropic/...' → 'anthropic')."""
    parts = model_id.split("/")
    if len(parts) >= 2:
        return parts[0].lower()
    return None

def normalise_name(s: str) -> str:
    """Lowercase and collapse all separators to underscore."""
    s = s.lower()
    s = re.sub(r"[\s/\-:\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def vendors_match(registry_id: str, arena_vendor: str) -> bool:
    """Check if the registry vendor matches the arena vendor string."""
    our_vendor = vendor_of(registry_id) or ""
    arena_v    = arena_vendor.lower()
    aliases    = VENDOR_MAP.get(our_vendor, [our_vendor])
    return any(a in arena_v or arena_v in a for a in aliases)


# ── Model matching ────────────────────────────────────────────────────────────

# Effort/harness suffixes Arena appends to a model name; not part of identity.
_EFFORT = {"low", "medium", "high", "xhigh", "max", "harness"}
_STOP   = {"free", "preview", "latest", "thinking", "search", "grounding",
           "experimental", "non", "reasoning", "api", "chat", "instruct"}


def _identity_tokens(name: str) -> tuple[str, ...]:
    """
    Ordered identity tokens of a model name, version digits preserved in order.
    Drops parenthetical notes, effort levels, stopwords and 8-digit date stamps.
    "claude-opus-5.5-high" → (claude, opus, 5, 5)   "gpt-5-mini" → (gpt, 5, mini)
    """
    name = re.sub(r"\(.*?\)", " ", name.lower())
    toks = [t for t in normalise_name(name).split("_") if t]
    return tuple(t for t in toks
                 if t not in _EFFORT and t not in _STOP
                 and not re.fullmatch(r"\d{8}", t))


def find_arena_match(
    registry_model: dict,
    arena_models: list[dict],
) -> dict | None:
    """
    Find the Arena entry for a registry model.

    Strict: the ordered identity tokens (including version digits) must be
    identical, so "claude-opus-5" never matches "claude-opus-5.5-high" and
    "gpt-5" never matches "gpt-5.6-sol-xhigh". Arena lists several effort
    levels of one model; the highest-scoring entry is used.
    Prefers false negatives over false positives.
    """
    reg_id = registry_model["id"]
    ours   = _identity_tokens(reg_id.split("/")[-1].split(":")[0])
    if not ours:
        return None

    matches = [
        m for m in arena_models
        if vendors_match(reg_id, m.get("vendor", ""))
        and _identity_tokens(m["model"]) == ours
    ]
    return max(matches, key=lambda m: m.get("score") or 0) if matches else None


# ── ELO normalisation ─────────────────────────────────────────────────────────

def normalise_elo(scores: dict[str, float]) -> dict[str, float]:
    """
    Min-max normalise ELO scores to [0, 1] across all models in the category.
    Returns {model_id → normalised_score}.
    """
    if not scores:
        return {}
    min_elo = min(scores.values())
    max_elo = max(scores.values())
    spread  = max_elo - min_elo
    if spread == 0:
        return {k: 0.5 for k in scores}
    return {k: round((v - min_elo) / spread, 6) for k, v in scores.items()}


# ── API fetch ─────────────────────────────────────────────────────────────────

def fetch_category(category: str) -> list[dict] | None:
    """Fetch the leaderboard for one Arena category."""
    url = f"{ARENA_API_BASE}/leaderboard?name={category}"
    try:
        resp = requests.get(url, timeout=ARENA_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("models", [])
        else:
            print(f"  WARN: {category} returned {resp.status_code}")
            return None
    except requests.RequestException as e:
        print(f"  ERROR fetching {category}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Arena ELO scores and merge into model_registry.json"
    )
    parser.add_argument("--registry", default=str(_DATA_DIR / "model_registry.json"))
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show matches without updating registry")
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser()
    if not registry_path.exists():
        print(f"ERROR: {registry_path} not found. Run fetch_aa_data.py first.")
        sys.exit(1)

    # ── Load registry ─────────────────────────────────────────────────────────
    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    models        = registry_data.get("models", [])
    print(f"Loaded {len(models)} models from {registry_path}")

    # ── Fetch each Arena category ─────────────────────────────────────────────
    print(f"\nFetching Arena leaderboards...")
    category_data: dict[str, list[dict]] = {}

    for cat in ARENA_CATEGORIES:
        print(f"  Fetching '{cat}'...", end=" ", flush=True)
        arena_models = fetch_category(cat)
        if arena_models:
            print(f"{len(arena_models)} models")
            category_data[cat] = arena_models
        else:
            print("failed")
        time.sleep(0.3)   # be polite

    if not category_data:
        print("ERROR: No Arena data fetched", file=sys.stderr)
        sys.exit(1)

    # ── Match and collect raw ELO scores ─────────────────────────────────────
    # For each category, build {registry_model_id → raw_elo}
    raw_elo: dict[str, dict[str, float]] = {cat: {} for cat in category_data}
    match_log: dict[str, dict] = {m["id"]: {} for m in models}

    print(f"\nMatching models to Arena leaderboards...")
    for cat, arena_models in category_data.items():
        print(f"\n  Category: {cat}")
        for reg_model in models:
            match = find_arena_match(reg_model, arena_models)
            if match:
                raw_elo[cat][reg_model["id"]] = float(match["score"])
                match_log[reg_model["id"]][cat] = match["model"]
                print(f"    ✓ {reg_model['id']:<45} ← {match['model']} "
                      f"(rank {match['rank']}, ELO {match['score']})")
            else:
                print(f"    ✗ {reg_model['id']:<45} no match found")

    # ── Normalise ELO per category ────────────────────────────────────────────
    normalised_elo: dict[str, dict[str, float]] = {}
    for cat, scores in raw_elo.items():
        normalised_elo[cat] = normalise_elo(scores)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n── Arena ELO scores (normalised 0–1) ────────────────────────────────")
    fields = [ARENA_CATEGORIES[cat]["field"] for cat in ARENA_CATEGORIES]
    header = f"  {'model_id':<48}" + "".join(f"{f.replace('arena_','').replace('_elo',''):>10}" for f in fields)
    print(header)
    print("  " + "─" * (48 + 10 * len(fields)))

    for m in models:
        mid  = m["id"]
        line = f"  {mid:<48}"
        for cat in ARENA_CATEGORIES:
            score = normalised_elo.get(cat, {}).get(mid)
            line += f"{score:>10.3f}" if score is not None else f"{'—':>10}"
        print(line)

    if args.dry_run:
        print("\nDRY RUN — registry not updated.")
        return

    # ── Merge into registry ───────────────────────────────────────────────────
    for m in models:
        mid = m["id"]
        for cat, cat_info in ARENA_CATEGORIES.items():
            field = cat_info["field"]
            score = normalised_elo.get(cat, {}).get(mid)
            m[field] = score   # None if no match

        # Also store raw ELO for reference
        m["arena_raw_elo"] = {
            cat: raw_elo.get(cat, {}).get(mid)
            for cat in ARENA_CATEGORIES
        }

        # Store match info for debugging
        m["arena_matches"] = match_log.get(mid, {})

    # ── Write updated registry ────────────────────────────────────────────────
    registry_data["arena_fetch"] = True
    registry_data["arena_categories"] = list(ARENA_CATEGORIES.keys())
    registry_path.write_text(
        json.dumps(registry_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nUpdated {registry_path} with Arena ELO scores")
    print(f"\nDone. Re-run build_model_features.py to include Arena scores.")


if __name__ == "__main__":
    main()