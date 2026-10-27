# PerfRouter — Project Reference

## What This Is

PerfRouter is a benchmark-driven LLM router. Given an incoming query, it
classifies the query into a task type (via sentence-BERT cosine similarity,
no LLM call), then uses a trained XGBoost model to predict which model in
the pool will produce the highest utility (quality − α × cost) for that
task type.

It is a sibling project to **optmod** (`~/workspace/optmod`). PerfRouter
produces the artifacts that optmod consumes at inference time.

---

## Directory Layout

```
perfrouter/
├── models.yaml                        # Single source of truth for the model pool
├── pyproject.toml                     # Package + entry points
├── .env.example                       # API key template
│
├── perfrouter/
│   ├── pipeline/                      # Build pipeline — Steps 0–6
│   │   ├── discover_models.py         # Step 0: auto-populate models.yaml from OR + AA
│   │   ├── fetch_aa_data.py           # Step 1: fetch AA benchmarks → model_registry.json
│   │   ├── fetch_arena_data.py        # Step 2: fetch Arena ELO → model_registry.json
│   │   ├── patch_taxonomy_arena.py    # Step 3: update taxonomy with Arena weights
│   │   ├── build_benchmark_weights.py # Step 4: benchmark → task-type weight matrix
│   │   ├── build_model_features.py    # Step 5: per-model feature vectors → model_features.csv
│   │   ├── train_perf_router.py       # Step 6: train XGBoost → perf_router.pkl
│   │   ├── build_task_taxonomy.py     # One-shot: LLM-generate task_taxonomy.json (run once)
│   │   └── run_perf_router_pipeline.py # Orchestrator: runs Steps 0–6 in order
│   │
│   ├── inference/
│   │   └── perf_router_inference.py   # Inference wrapper used by optmod
│   │
│   └── tools/
│       └── simulate_perf_router_wcb.py # Dry-run simulation against WCB tasks
│
├── data/                              # Pipeline outputs (not source-of-truth)
│   ├── model_registry.json            # Merged AA + Arena data per model
│   ├── task_taxonomy.json             # ~32 task types with definitions + benchmark mappings
│   ├── benchmark_weights.json         # benchmark → task-type weight matrix
│   └── model_features.csv             # Feature vector per model (49 features)
│
└── models/
    └── perf_router.pkl                # Trained XGBoost model
```

---

## Data Flow

```
OpenRouter API ──┐
                 ├─► discover_models.py (Step 0) ──► models.yaml
AA API ──────────┘

models.yaml ──► fetch_aa_data.py (Step 1) ──► model_registry.json
                                                      │
Arena API ──────► fetch_arena_data.py (Step 2) ───────┘

task_taxonomy.json ──► patch_taxonomy_arena.py (Step 3) ──► task_taxonomy.json (updated)
                 └───► build_benchmark_weights.py (Step 4) ──► benchmark_weights.json

model_registry.json ──┐
benchmark_weights.json ─┤─► build_model_features.py (Step 5) ──► model_features.csv
WCB CSV (optional) ───┘

model_features.csv ──► train_perf_router.py (Step 6) ──► perf_router.pkl
WCB CSV (optional) ─────────────────────────────────────┘
```

`task_taxonomy.json` is generated once by `build_task_taxonomy.py` (one LLM
call to DeepSeek) and then treated as frozen. Steps 3 and 4 update it in
place; never regenerate it unless the taxonomy itself needs to change.

---

## `models.yaml` Schema

The single source of truth for the model pool. Contains three top-level blocks:

### `discovery:` block
Controls Step 0 auto-discovery:

```yaml
discovery:
  enabled: true           # set false to skip Step 0 entirely
  pool_size: 50           # ceiling — never adds beyond this
  provider_allowlist: [deepseek, google, openai, ...]
  provider_blocklist: [openrouter]
  model_blocklist: []     # specific OR model ids to skip
  selection:
    aa_intelligence_index_weight: 0.7
    arena_elo_weight: 0.3
    min_aa_intelligence_index: 10
```

### `settings:` block
Pipeline-wide config (`aa_api_key_env`, `cost_weight`, `encoder`,
`top_k_task_types`).

### `models:` list
One entry per model. Two kinds:

| Kind | Marker | Behaviour |
|------|--------|-----------|
| Manual | no `_discovered` field | Never overwritten by Step 0 |
| Auto-discovered | `_discovered: true` | Added by Step 0; remove by deleting or adding id to `model_blocklist` |

Key fields per entry:

| Field | Source | Notes |
|-------|--------|-------|
| `id` | manual / OR | Exact OpenRouter routing id (`:free` suffix preserved) |
| `aa_slug` | manual / derived | Matches slug on `artificialanalysis.ai/models/<slug>` |
| `provider` | manual / derived | First segment of `id` |
| `free` | manual / derived | `id.endswith(":free")` |
| `total_params_B` / `active_params_B` | manual / OR | In billions; null if unknown |
| `context_window_k` | manual / OR | context_length / 1000 |
| `effective_context_k` | manual | Conservative usable window; default = context_window_k / 2 |
| `supports_vision` | manual / OR | `"image"` in OR modality string |
| `supports_tools` | manual | Default true |
| `is_reasoning` | manual / heuristic | True if aa_intelligence_index ≥ 60 or name matches `(reason\|think\|r1\|qwq\|o[1-9])` |
| `has_thinking_mode` | manual / heuristic | True if name contains `"thinking"` |
| `price_input_per_1M` / `price_output_per_1M` | manual / OR | USD per 1M tokens |
| `price_cache_read_per_1M` | manual | 0.0 for auto-discovered (OR doesn't expose cache pricing) |
| `license` | manual / OR | `"unknown"` for auto-discovered if OR doesn't specify |

---

## Running the Pipeline

```bash
uv sync                            # install deps
cp .env.example .env && edit .env  # add AA_API_KEY, OPENROUTER_API_KEY
```

| Command | What runs |
|---------|-----------|
| `uv run perf-pipeline` | Steps 0–6, skipping any step whose output already exists |
| `uv run perf-pipeline --force` | Steps 0–6, re-run everything |
| `uv run perf-pipeline --force-step N` | Steps N–6, re-run from N onward |
| `uv run perf-pipeline --stop-after N` | Steps 0–N only |
| `uv run perf-pipeline --dry-run` | Print what would run without executing |

Common workflows:

```bash
# New models may be available on OR + AA
uv run perf-pipeline --force-step 0

# Prices changed in models.yaml
uv run perf-pipeline --force-step 1

# New WCB data available
uv run perf-pipeline --force-step 6

# Preview discovery without writing
uv run python perfrouter/pipeline/discover_models.py --dry-run --verbose
```

### Required environment variables

| Variable | Used by |
|----------|---------|
| `AA_API_KEY` | Steps 0, 1 — Artificial Analysis API |
| `OPENROUTER_API_KEY` | Step 0 — optional, public OR endpoint works without it |
| `DEEPSEEK_API_KEY` | `build_task_taxonomy.py` only (one-shot) |

---

## Inference

`perf_router_inference.py` is the entry point optmod calls. It loads four
files at startup (all from `data/` or `models/`):

- `perf_router.pkl` — XGBoost model
- `task_taxonomy.json` — task type definitions + sentence-BERT embeddings
- `model_registry.json` — per-model metadata
- `model_features.csv` — per-model feature vectors

At query time:
1. Encode query with sentence-BERT (all-MiniLM-L6-v2, local, no API call)
2. Cosine similarity against task type embeddings → top-k task types
3. For each model, predict utility = XGBoost score − α × normalised_cost
4. Route to highest-utility model
5. Log decision vs configured baseline

Runtime environment variables (set in optmod's `routing/config.yaml` or env):

| Variable | Default | Effect |
|----------|---------|--------|
| `PERF_ROUTER_COST_WEIGHT` | `0.3` | α — cost penalty (0 = quality-only, 1 = cost-only) |
| `PERF_ROUTER_DEGRADATION_THRESHOLD` | `0.0` | Accept up to this quality drop for a cheaper model |
| `PERF_ROUTER_MIN_SIMILARITY` | `0.20` | Below this, fall back to cheapest eligible model |
| `PERF_ROUTER_BASELINE` | `deepseek/deepseek-v4-pro` | Model to log decisions against |

---

## optmod Integration

Build artifacts that optmod needs are in `data/`, `models/` and `models.yaml`. The pipeline's
last step (step 7) copies them into optmod's `routing/perfrouter/`, the standalone fallback optmod
uses when this repo is not checked out next to it:

```bash
uv run python perfrouter/pipeline/run_perf_router_pipeline.py --export-only          # copy only
uv run python perfrouter/pipeline/run_perf_router_pipeline.py --force-step 6          # retrain, then export
uv run python perfrouter/pipeline/run_perf_router_pipeline.py --no-export             # build, don't copy
# target: --optmod-dir (default ~/workspace/optmod/routing/perfrouter)
```

Exported: `perf_router.pkl`, `model_features.csv`, `model_registry.json`, `task_taxonomy.json`
(with `agentic_examples`, merged in step 3b from `data/agentic_examples.json`) and `models.yaml`.
`perf_router_inference.py` is not copied: optmod keeps its own variant (`routing/perfrouter/inference.py`,
same logic, different default paths and a `models_yaml_path` argument), so port inference changes by hand.
Review and commit the result in optmod.

The 10-model inference pool in optmod is a manually curated subset of
whatever is in this training pool. XGBoost trains on all models here;
optmod routes using only the deployed subset.

---

## Key Design Decisions

- **No LLM at inference time.** Task classification uses sentence-BERT cosine
  similarity only. This makes routing microsecond-fast and free.
- **Two training signal sources.** WCB ground truth (weight 3×) + AA benchmark
  affinity scores (weight 1×). WCB rows dominate where available; benchmark
  scores fill in the rest of the pool.
- **`aa_slug` is always explicit.** Never derived implicitly at Step 1 —
  the slug is written into `models.yaml` at discovery time and auditable.
- **Manual entries win on every field conflict.** Discovery never
  overwrites a field that exists in a manual entry.
- **Idempotent pipeline.** Every step can be re-run without side effects.
  Steps skip automatically when their output file already exists.
