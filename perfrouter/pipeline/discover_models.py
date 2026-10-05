#!/usr/bin/env python3
"""
discover_models.py
==================
Step 0 of the PerfRouter pipeline.

Fetches all models from OpenRouter, cross-references against Artificial Analysis
for benchmark coverage, scores and ranks candidates, then appends qualifying
models to the `models` list in models.yaml.

Only models with AA benchmark data (and aa_intelligence_index above the
configured threshold) are eligible. Manual entries are never overwritten.

Usage:
    python3 discover_models.py [--dry-run] [--verbose] [--force]
    python3 discover_models.py --models /path/to/models.yaml
    python3 discover_models.py --api-key AA_KEY --or-api-key OR_KEY
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

OR_MODELS_URL  = "https://openrouter.ai/api/v1/models"
AA_MODELS_URL  = "https://artificialanalysis.ai/api/v2/data/llms/models"
HTTP_TIMEOUT_S = 30
RETRY_COUNT    = 3
RETRY_DELAY    = 2.0

REASONING_RE = re.compile(r"(?i)(reason|think|r1|qwq|o[1-9])")
THINKING_RE  = re.compile(r"(?i)thinking")

# Known slug mismatches: OpenRouter id (after stripping :free) → aa_slug
# Check before the derivation algorithm.
SLUG_OVERRIDES: dict[str, str] = {
    "tencent/hy3-preview":          "hy3",
    "google/gemini-3.1-flash-lite": "gemini-3-1-flash-lite-preview",
    # add more as discovered
}

VERSION_SUFFIX_RE = re.compile(r"-(preview|latest|exp|it)$")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get_json(url: str, headers: dict, label: str) -> list[dict] | None:
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            print(f"  Fetching {label} (attempt {attempt}/{RETRY_COUNT})...")
            resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_S)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, list):
                    return data
                else:
                    print(f"  WARN: unexpected response structure from {label}")
                    return None
            elif resp.status_code == 401:
                print(f"  ERROR: {label} — 401 Unauthorized (bad or missing key)")
                return None
            elif resp.status_code == 429:
                delay = RETRY_DELAY * attempt
                print(f"  WARN: {label} rate-limited (429) — waiting {delay:.0f}s")
                time.sleep(delay)
            else:
                print(f"  WARN: {label} returned {resp.status_code}: {resp.text[:200]}")
                if attempt < RETRY_COUNT:
                    time.sleep(RETRY_DELAY)
        except requests.RequestException as e:
            print(f"  ERROR: {label} request failed: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
    return None


def fetch_or_models(or_api_key: str | None) -> list[dict] | None:
    headers: dict[str, str] = {
        "Accept":     "application/json",
        "User-Agent": "PerfRouter/1.0",
    }
    if or_api_key:
        headers["Authorization"] = f"Bearer {or_api_key}"
    return _get_json(OR_MODELS_URL, headers, "OpenRouter models")


def fetch_aa_models(api_key: str) -> list[dict] | None:
    headers: dict[str, str] = {
        "x-api-key":  api_key,
        "Accept":     "application/json",
        "User-Agent": "PerfRouter/1.0",
    }
    return _get_json(AA_MODELS_URL, headers, "Artificial Analysis models")


# ── AA index helpers ──────────────────────────────────────────────────────────

def build_aa_index(aa_models: list[dict]) -> dict[str, dict]:
    """Build a normalised-slug → model dict for O(1) lookup."""
    index: dict[str, dict] = {}
    for m in aa_models:
        for field in ("slug", "model_id", "id", "model_slug"):
            slug = m.get(field)
            if slug:
                index[slug.lower()] = m
                index[slug.lower().replace("-", "_")] = m
    return index


def lookup_aa(aa_index: dict[str, dict], slug: str) -> dict | None:
    key = slug.lower()
    return aa_index.get(key) or aa_index.get(key.replace("-", "_"))


def get_aa_intelligence_index(aa_model: dict) -> float | None:
    evals = aa_model.get("evaluations") or {}
    val = evals.get("artificial_analysis_intelligence_index")
    return float(val) if val is not None else None


def get_aa_arena_elo(aa_model: dict) -> float | None:
    # AA API v2 does not expose Arena ELO directly; check common field names.
    for field in ("chatbot_arena_elo", "arena_elo", "elo"):
        val = aa_model.get(field)
        if val is not None:
            return float(val)
    return None


# ── aa_slug derivation ────────────────────────────────────────────────────────

def derive_aa_slug(or_id: str, aa_index: dict[str, dict]) -> str | None:
    """
    Derive the aa_slug for an OpenRouter model id.

    Algorithm (first match wins):
      0. Check SLUG_OVERRIDES (keyed by or_id with :free stripped)
      1. Strip :free suffix, then strip provider prefix → raw_slug
      2. Exact AA match on raw_slug
      3. Normalised match: lowercase, replace _ with -, strip version suffixes
    Returns the matched slug string, or None if no AA match found.
    """
    base_id = or_id.split(":")[0]  # strip :free suffix

    # 0. Hard-coded overrides take priority
    if base_id in SLUG_OVERRIDES:
        slug = SLUG_OVERRIDES[base_id]
        if lookup_aa(aa_index, slug) is not None:
            return slug
        # Override exists but AA doesn't have it — skip
        return None

    raw_slug = base_id.split("/", 1)[-1]  # strip provider prefix

    # 1. Exact match
    if lookup_aa(aa_index, raw_slug) is not None:
        return raw_slug

    # 2. Normalised match
    norm = VERSION_SUFFIX_RE.sub("", raw_slug.lower().replace("_", "-"))
    if lookup_aa(aa_index, norm) is not None:
        return norm

    # 3. AA-style variants: dots → dashes, provider-prefixed, "-reasoning" suffix
    provider = base_id.split("/", 1)[0]
    stems = [raw_slug.lower().replace(".", "-").replace("_", "-")]
    stems.append(f"{provider}-{stems[0]}")
    for stem in stems:
        for cand in (stem, f"{stem}-reasoning"):
            if lookup_aa(aa_index, cand) is not None:
                return cand

    return None


# ── OpenRouter field extraction ───────────────────────────────────────────────

def _to_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val))
    except (ValueError, TypeError):
        return None


def _parse_params_b(raw: object) -> float | None:
    """Convert raw parameter count to billions. Handles both raw ints and B-scale floats."""
    v = _to_float(raw)
    if v is None:
        return None
    # If the value looks like a raw count (> 1 million), convert to billions.
    return round(v / 1e9, 3) if v > 1e6 else round(v, 3)


def extract_candidate(or_model: dict, aa_slug: str, aa_model: dict) -> dict:
    """Build the candidate entry dict from OpenRouter + AA data."""
    or_id = or_model["id"]
    name  = or_model.get("name", or_id)

    provider = or_id.split("/")[0]
    free     = or_id.endswith(":free")

    # Context window (default 128k if missing)
    ctx_raw = or_model.get("context_length") or (
        (or_model.get("top_provider") or {}).get("context_length")
    )
    context_window_k = round(int(ctx_raw) / 1000) if ctx_raw else 128

    # Vision support
    modality        = (or_model.get("architecture") or {}).get("modality", "")
    supports_vision = "image" in modality.lower()

    # Parameter counts
    arch            = or_model.get("architecture") or {}
    total_params_B  = _parse_params_b(arch.get("parameters") or arch.get("parameter_count"))
    active_params_B = _parse_params_b(
        arch.get("active_parameters") or arch.get("active_params")
    )

    # Pricing — OR values are per-token strings; multiply to per-1M
    pricing = or_model.get("pricing") or {}

    def _price(key: str) -> float:
        raw = pricing.get(key, "0") or "0"
        try:
            return round(float(raw) * 1_000_000, 6)
        except (ValueError, TypeError):
            return 0.0

    price_input  = _price("prompt")
    price_output = _price("completion")

    # License
    license_val = or_model.get("license") or "unknown"

    # AA-derived flags
    aa_idx = get_aa_intelligence_index(aa_model)
    is_reasoning = (
        (aa_idx is not None and aa_idx >= 60)
        or bool(REASONING_RE.search(name))
        or bool(REASONING_RE.search(or_id))
    )
    has_thinking_mode = bool(THINKING_RE.search(name + or_id))

    return {
        "id":                      or_id,
        "aa_slug":                 aa_slug,
        "provider":                provider,
        "local":                   False,
        "free":                    free,
        "total_params_B":          total_params_B,
        "active_params_B":         active_params_B,
        "context_window_k":        context_window_k,
        "effective_context_k":     context_window_k // 2,
        "supports_vision":         supports_vision,
        "supports_tools":          True,
        "is_reasoning":            is_reasoning,
        "has_thinking_mode":       has_thinking_mode,
        "price_input_per_1M":      price_input,
        "price_output_per_1M":     price_output,
        "price_cache_read_per_1M": 0.0,
        "license":                 license_val,
        "_discovered":             True,
    }


# ── models.yaml read/write ────────────────────────────────────────────────────

def read_yaml_with_header(path: Path) -> tuple[str, dict]:
    """
    Returns (header_comment, parsed_dict).
    Header = all leading '#' / blank lines before first non-comment content.
    """
    text  = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    header_lines: list[str] = []
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            header_lines.append(line)
        else:
            break
    header = "".join(header_lines).rstrip("\n")
    data   = yaml.safe_load(text) or {}
    return header, data


def write_yaml_with_header(path: Path, header: str, data: dict) -> None:
    body    = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    content = (header + "\n\n" + body) if header else body
    path.write_text(content, encoding="utf-8")


# ── Discovery logic ───────────────────────────────────────────────────────────

def discover(
    models_yaml: Path,
    aa_api_key:  str | None,
    or_api_key:  str | None,
    dry_run:     bool,
    force:       bool,
    verbose:     bool,
    reset:       bool = False,
) -> int:
    """Core discovery routine. Returns 0 on success, 1 on fatal error."""

    # ── Load models.yaml ──────────────────────────────────────────────────────
    if not models_yaml.exists():
        print(f"ERROR: {models_yaml} not found", file=sys.stderr)
        return 1

    header, cfg       = read_yaml_with_header(models_yaml)
    discovery_cfg     = cfg.get("discovery", {})
    existing_models   = cfg.get("models") or []
    if reset:
        n_drop = sum(1 for m in existing_models if m.get("_discovered"))
        existing_models = [m for m in existing_models if not m.get("_discovered")]
        print(f"\n  --reset: dropping {n_drop} auto-discovered entr{'y' if n_drop == 1 else 'ies'}, "
              f"keeping {len(existing_models)} manual")
    existing_ids: set[str] = {m["id"] for m in existing_models}

    pool_size:     int        = discovery_cfg.get("pool_size", 50)
    allowlist:     list[str]  = discovery_cfg.get("provider_allowlist") or []
    blocklist:     list[str]  = discovery_cfg.get("provider_blocklist") or []
    model_blocklist: list[str] = discovery_cfg.get("model_blocklist") or []
    selection_cfg: dict       = discovery_cfg.get("selection") or {}

    w_aa    = float(selection_cfg.get("aa_intelligence_index_weight", 0.7))
    w_elo   = float(selection_cfg.get("arena_elo_weight", 0.3))
    min_idx = float(selection_cfg.get("min_aa_intelligence_index", 10))

    slots_available = pool_size - len(existing_ids)
    print(f"\n  Pool: {len(existing_ids)} existing / {pool_size} max  "
          f"({slots_available} slot{'s' if slots_available != 1 else ''} available)")

    if slots_available <= 0 and not force:
        print("  Pool already at capacity. Use --force to re-discover.")
        return 0

    # ── Fetch OpenRouter models ───────────────────────────────────────────────
    or_models = fetch_or_models(or_api_key)
    if or_models is None:
        print("  WARN: OpenRouter API unavailable — skipping discovery")
        return 0
    print(f"  OpenRouter returned {len(or_models)} models")

    # ── Fetch AA models ───────────────────────────────────────────────────────
    if not aa_api_key:
        print("  WARN: No AA API key — cannot validate AA coverage. Skipping discovery.")
        return 0

    aa_models = fetch_aa_models(aa_api_key)
    if aa_models is None:
        print("  WARN: Artificial Analysis API unavailable — skipping discovery")
        return 0
    print(f"  Artificial Analysis returned {len(aa_models)} models")

    aa_index = build_aa_index(aa_models)

    # ── Filter and score candidates ───────────────────────────────────────────
    candidates: list[dict] = []
    n_existing  = 0
    n_no_aa     = 0
    n_low_score = 0

    for or_model in or_models:
        or_id = or_model.get("id", "")
        if not or_id or "/" not in or_id:
            continue

        provider = or_id.split("/")[0]

        if allowlist and provider not in allowlist:
            continue
        if provider in blocklist:
            continue
        if or_id in model_blocklist:
            continue
        if or_id.endswith(":batch"):
            continue   # batch endpoints duplicate the base model's AA slug
        if or_id in existing_ids:
            n_existing += 1
            continue

        # Derive aa_slug and look up AA data
        aa_slug = derive_aa_slug(or_id, aa_index)
        if aa_slug is None:
            n_no_aa += 1
            if verbose:
                print(f"    ↷ {or_id:<58} — no AA slug derived")
            continue

        aa_model = lookup_aa(aa_index, aa_slug)
        if aa_model is None:
            n_no_aa += 1
            if verbose:
                print(f"    ↷ {or_id:<58} — AA slug '{aa_slug}' not in AA index")
            continue

        aa_idx = get_aa_intelligence_index(aa_model)
        if aa_idx is None or aa_idx < min_idx:
            n_low_score += 1
            if verbose:
                print(f"    ✗ {or_id:<58} — AA idx {aa_idx} < min {min_idx}")
            continue

        elo   = get_aa_arena_elo(aa_model) or 0.0
        score = w_aa * aa_idx + w_elo * (elo / 1000)

        candidate = extract_candidate(or_model, aa_slug, aa_model)
        candidate["_score"]  = score
        candidate["_aa_idx"] = aa_idx
        candidate["_elo"]    = elo
        candidates.append(candidate)

    # ── Sort and cap ──────────────────────────────────────────────────────────
    candidates.sort(key=lambda c: c["_score"], reverse=True)
    to_add = candidates if force else candidates[:max(0, slots_available)]

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n  Filter summary:")
    print(f"    Already in pool   : {n_existing}")
    print(f"    No AA data        : {n_no_aa}")
    print(f"    Below min idx     : {n_low_score}")
    print(f"    Candidates scored : {len(candidates)}")
    print(f"    Will add          : {len(to_add)}")

    if to_add:
        print(f"\n  Candidates to add (top {len(to_add)}):")
        for c in to_add:
            if c["free"]:
                pricing_str = "free"
            else:
                pricing_str = f"${c['price_input_per_1M']:.3f}/${c['price_output_per_1M']:.3f}"
            print(f"    {c['id']:<58}  score={c['_score']:.2f}  "
                  f"aa_idx={c['_aa_idx']:.1f}  {pricing_str}")

    if dry_run:
        print("\n  DRY RUN — nothing written.")
        return 0

    if not to_add and not reset:
        print("\n  Nothing to add.")
        return 0

    if reset:
        backup = models_yaml.with_suffix(".yaml.bak")
        shutil.copy2(models_yaml, backup)
        print(f"\n  Backup → {backup}")

    # ── Append to models.yaml ─────────────────────────────────────────────────
    _SCRATCH = {"_score", "_aa_idx", "_elo"}
    for c in to_add:
        entry = {k: v for k, v in c.items() if k not in _SCRATCH}
        existing_models.append(entry)

    cfg["models"] = existing_models
    write_yaml_with_header(models_yaml, header, cfg)
    print(f"\n  ✓ Added {len(to_add)} model(s) to {models_yaml}")
    return 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 0: Discover and populate PerfRouter model pool"
    )
    parser.add_argument(
        "--models",
        default=str(_PROJECT_ROOT / "models.yaml"),
        help="Path to models.yaml (default: project root)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Artificial Analysis API key (default: AA_API_KEY env var)",
    )
    parser.add_argument(
        "--or-api-key",
        default=None,
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var, optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidates without writing models.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-discover even if pool_size already reached",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all auto-discovered entries (keep manual) and re-discover from scratch",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-model scoring details",
    )
    args = parser.parse_args()

    models_yaml = Path(args.models).expanduser().resolve()
    aa_api_key  = args.api_key    or os.environ.get("AA_API_KEY")
    or_api_key  = args.or_api_key or os.environ.get("OPENROUTER_API_KEY")

    rc = discover(
        models_yaml=models_yaml,
        aa_api_key=aa_api_key,
        or_api_key=or_api_key,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
        reset=args.reset,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
