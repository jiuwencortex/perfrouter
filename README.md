# PerfRouter

Benchmark-driven LLM routing using XGBoost and sentence-BERT. Routes incoming queries to the model with the highest predicted utility (quality − α × cost) based on task type classification.

## How PerfRouter differs from TRouter

| | TRouter | PerfRouter |
|---|---|---|
| **Training signal** | Human preference labels from WildClawBench | Benchmark scores + WCB ground truth |
| **Routing input** | Query embedding → MLP | Task type (BERT cosine sim) → XGBoost |
| **Model pool** | Fixed at train time | Auto-discovered via `discover_models.py` |
| **Inference cost** | Neural forward pass | Microseconds (tree lookup) |
| **Adding a model** | Full retrain | Edit YAML or run Step 0, then step 1 |

## Quick start

```bash
cp .env.example .env
# edit .env with your API keys
uv sync
uv run perf-pipeline
```

## Build pipeline

Each step is idempotent — re-run only if its inputs changed.

| Step | Command | What it does |
|------|---------|--------------|
| 0 | `uv run perf-pipeline --force-step 0` | Auto-discover models from OpenRouter + AA → `models.yaml` |
| 1 | `uv run perf-pipeline --force-step 1` | Fetch AA benchmark data → `data/model_registry.json` |
| 2 | `uv run perf-pipeline --force-step 2` | Fetch Arena ELO scores → merged into registry |
| 3 | `uv run perf-pipeline --force-step 3` | Patch taxonomy with Arena weights |
| 4 | `uv run perf-pipeline --force-step 4` | Build benchmark weight matrix → `data/benchmark_weights.json` |
| 5 | `uv run perf-pipeline --force-step 5` | Compute model feature vectors → `data/model_features.csv` |
| 6 | `uv run perf-pipeline --force-step 6` | Train XGBoost → `models/perf_router.pkl` |

Full rebuild (includes discovery):

```bash
uv run perf-pipeline --force
```

## Model pool

The training pool (up to 50 models) is managed in `models.yaml` with two types of entries:

- **Manual entries** — hand-curated, never overwritten by discovery. Add these for models that need precise pricing, custom flags, or aren't yet on Artificial Analysis.
- **Auto-discovered entries** — added by Step 0 from OpenRouter + AA data, marked with `_discovered: true`. Remove by deleting the entry or adding the id to `discovery.model_blocklist` in `models.yaml`.

Discovery is controlled by the `discovery:` block at the top of `models.yaml`:

```yaml
discovery:
  enabled: true
  pool_size: 50
  provider_allowlist: [deepseek, google, openai, ...]
  provider_blocklist: [openrouter]
  model_blocklist: []
  selection:
    aa_intelligence_index_weight: 0.7
    arena_elo_weight: 0.3
    min_aa_intelligence_index: 10
```

To preview what would be added without writing:

```bash
uv run python perfrouter/pipeline/discover_models.py --dry-run --verbose
```

## Adding a new model

**Auto-discovery (recommended):** models from supported providers are picked up automatically on the next `--force` run once they appear on Artificial Analysis.

**Manual entry:** add the model directly to `models.yaml` with its `id`, `aa_slug`, pricing, and flags, then run:

```bash
uv run perf-pipeline --force-step 1
```

## Runtime environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PERF_ROUTER_COST_WEIGHT` | `0.3` | α — how much cost penalises quality (0 = quality only, 1 = cost only) |
| `PERF_ROUTER_DEGRADATION_THRESHOLD` | `0.0` | Accept up to this fraction of quality drop for a cheaper model |
| `PERF_ROUTER_MIN_SIMILARITY` | `0.20` | Below this task-type similarity, fall back to cheapest model |
| `PERF_ROUTER_BASELINE` | `deepseek/deepseek-v4-pro` | Baseline model for cost and quality comparison logging |

## WCB simulation

Validate routing decisions against WildClawBench task prompts without executing any tasks:

```bash
uv run perf-simulate --tasks-dir path/to/WildClawBench/tasks
```

Optional flags:
```bash
uv run perf-simulate \
    --tasks-dir path/to/WildClawBench/tasks \
    --cost-weight 0.5 \
    --degradation-threshold 0.10 \
    --out my_simulation.csv
```

## optmod integration

To use PerfRouter inside optmod:

1. Export the artifacts. The last pipeline step copies `perf_router.pkl`, `model_features.csv`,
   `model_registry.json`, `task_taxonomy.json` and `models.yaml` into optmod's `routing/perfrouter/`:
   ```bash
   uv run python perfrouter/pipeline/run_perf_router_pipeline.py --export-only   # --optmod-dir to change the target
   ```
   A full run (`--force-step N`) also exports unless you pass `--no-export`. `inference.py` is not copied;
   optmod keeps its own variant.

2. In `routing/config.yaml`, set:
   ```yaml
   router: perfrouter
   perf_router_cost_weight: 0.3
   perf_router_baseline: deepseek/deepseek-v4-pro
   ```
