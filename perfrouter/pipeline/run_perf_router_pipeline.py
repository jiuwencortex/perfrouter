#!/usr/bin/env python3
"""
run_perf_router_pipeline.py
============================
Runs the full PerfRouter build pipeline end-to-end from models.yaml.
Use this whenever you add/remove models, update pricing, or want to
incorporate new benchmark data.

─────────────────────────────────────────────────────────────────────
PIPELINE STEPS
─────────────────────────────────────────────────────────────────────

  Step 0 — discover_models.py
    Fetches models from OpenRouter + Artificial Analysis, scores them,
    and appends qualifying candidates to models.yaml. Skipped if
    discovery.enabled: false in models.yaml or pool is already full.

  Step 1 — fetch_aa_data.py
    Reads models.yaml, fetches Artificial Analysis API benchmark data,
    merges into model_registry.json.

  Step 2 — fetch_arena_data.py
    Fetches Arena.ai ELO scores for 5 categories (text, search, code,
    vision, document) and merges into model_registry.json.

  Step 3 — patch_taxonomy_arena.py
    Updates task_taxonomy.json to use Arena ELO weights for task types
    that AA benchmarks don't cover well (social, creative, retrieval).
    Only re-runs if taxonomy exists; skips if no taxonomy yet.

  Step 4 — build_benchmark_weights.py
    Rebuilds the benchmark weight matrix from task_taxonomy.json.
    Must run after step 3 whenever taxonomy changes.

  Step 5 — build_model_features.py
    Computes task-type affinity scores and structural features per model.
    Outputs model_features.csv.

  Step 6 — train_perf_router.py
    Trains XGBoost on model_features.csv + WCB seed data.
    Outputs perf_router.pkl.

─────────────────────────────────────────────────────────────────────
WHEN TO RUN EACH STEP
─────────────────────────────────────────────────────────────────────

  Full rebuild (new models added):
    python3 run_perf_router_pipeline.py --force

  Pricing update only (no new models, just prices changed in models.yaml):
    python3 run_perf_router_pipeline.py --force-step 1
    (re-fetches AA data with new pricing, rebuilds features + model)

  New WCB runs available:
    python3 run_perf_router_pipeline.py --force-step 6
    (just retrain XGBoost with new WCB seed data)

  Add a new model to models.yaml:
    1. Edit models.yaml (add the new model entry)
    2. python3 run_perf_router_pipeline.py --force-step 1
       (full rebuild from AA fetch through training)

─────────────────────────────────────────────────────────────────────
Usage:
  export AA_API_KEY=...
  export DEEPSEEK_API_KEY=...

  python3 run_perf_router_pipeline.py
  python3 run_perf_router_pipeline.py --force
  python3 run_perf_router_pipeline.py --force-step 5
  python3 run_perf_router_pipeline.py --dry-run
─────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"
_MODELS_DIR   = _PROJECT_ROOT / "models"


# ── Terminal colours ──────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    DIM    = "\033[2m"

def ok(msg):   print(f"{C.GREEN}  ✓{C.RESET}  {msg}")
def warn(msg): print(f"{C.YELLOW}  ⚠{C.RESET}  {msg}")
def err(msg):  print(f"{C.RED}  ✗{C.RESET}  {msg}")
def info(msg): print(f"{C.BLUE}  →{C.RESET}  {msg}")
def skip(msg): print(f"{C.DIM}  ↷  {msg}{C.RESET}")
def head(msg):
    w = 72
    print(f"\n{C.BOLD}{'─'*w}{C.RESET}")
    print(f"{C.BOLD}  {msg}{C.RESET}")
    print(f"{C.BOLD}{'─'*w}{C.RESET}")


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(
    step_num:    int,
    label:       str,
    script:      Path,
    args:        list[str],
    output_file: Path | None,
    force:       bool,
    dry_run:     bool,
) -> bool:
    head(f"Step {step_num} — {label}")

    if output_file and output_file.exists() and not force:
        skip(f"Output already exists: {output_file.name}")
        skip(f"Pass --force or --force-step {step_num} to re-run.")
        return True

    if not script.exists():
        err(f"Script not found: {script}")
        return False

    cmd = [sys.executable, str(script)] + args
    info(f"Running: {' '.join(str(c) for c in cmd)}")

    if dry_run:
        warn("DRY RUN — not executing.")
        return True

    start  = time.time()
    result = subprocess.run(cmd, cwd=script.parent)
    elapsed = time.time() - start

    if result.returncode == 0:
        ok(f"Completed in {elapsed:.1f}s")
        return True
    else:
        err(f"Failed (exit {result.returncode}) after {elapsed:.1f}s")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full PerfRouter build pipeline"
    )

    # ── Path config ───────────────────────────────────────────────────────────
    parser.add_argument("--models-yaml",   default=str(_PROJECT_ROOT / "models.yaml"))
    parser.add_argument("--registry",      default=str(_DATA_DIR / "model_registry.json"))
    parser.add_argument("--taxonomy",      default=str(_DATA_DIR / "task_taxonomy.json"))
    parser.add_argument("--weights",       default=str(_DATA_DIR / "benchmark_weights.json"))
    parser.add_argument("--features",      default=str(_DATA_DIR / "model_features.csv"))
    parser.add_argument("--router-out",    default=str(_MODELS_DIR / "perf_router.pkl"))
    parser.add_argument("--wcb-csv",       default=None,
                        help="Path to wcb_training_labeled.csv for XGBoost seeding")

    # ── Step control ──────────────────────────────────────────────────────────
    parser.add_argument("--force",         action="store_true",
                        help="Re-run all steps even if outputs exist")
    parser.add_argument("--force-step",    type=int, default=None, metavar="N",
                        help="Re-run from step N onwards (0-6)")
    parser.add_argument("--stop-after",    type=int, default=6, metavar="N",
                        help="Stop after step N (default: 6)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print what would run without executing")

    # ── Training config ───────────────────────────────────────────────────────
    parser.add_argument("--cost-weight",   type=float, default=0.3)
    parser.add_argument("--aa-api-key",    default=None,
                        help="Artificial Analysis API key (default: AA_API_KEY env)")
    parser.add_argument("--deepseek-key",  default=None,
                        help="DeepSeek API key for taxonomy generation")
    parser.add_argument("--label-model",   default="deepseek-v4-flash")
    parser.add_argument("--label-base-url",default="https://api.deepseek.com/v1")

    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    HERE = Path(__file__).parent.resolve()

    models_yaml  = (HERE / args.models_yaml).resolve()
    registry     = (HERE / args.registry).resolve()
    taxonomy     = (HERE / args.taxonomy).resolve()
    weights      = (HERE / args.weights).resolve()
    features     = (HERE / args.features).resolve()
    router_out   = (HERE / args.router_out).resolve()
    wcb_csv      = Path(args.wcb_csv).expanduser().resolve() if args.wcb_csv else None

    # ── Forced steps ──────────────────────────────────────────────────────────
    forced: set[int] = set()
    if args.force:
        forced = {0, 1, 2, 3, 4, 5, 6}
    elif args.force_step is not None:
        forced = set(range(args.force_step, 7))

    def is_forced(n): return n in forced

    # ── Print banner ──────────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{'═'*72}{C.RESET}")
    print(f"{C.BOLD}  PerfRouter Pipeline   {datetime.now().strftime('%Y-%m-%d %H:%M')}{C.RESET}")
    print(f"{C.BOLD}{'═'*72}{C.RESET}")
    print(f"\n  Models config : {models_yaml}")
    print(f"  Working dir   : {HERE}")
    print(f"  Cost weight   : {args.cost_weight}")
    if wcb_csv:
        print(f"  WCB seed      : {wcb_csv}")
    if args.dry_run:
        print(f"\n  {C.YELLOW}DRY RUN — nothing will be executed{C.RESET}")
    if forced:
        print(f"  Forced steps  : {sorted(forced)}")
    print()

    # ── Check API keys ────────────────────────────────────────────────────────
    aa_key = args.aa_api_key or os.environ.get("AA_API_KEY")
    if not aa_key and not args.dry_run:
        warn("AA_API_KEY not set — Step 1 will use YAML-only specs (no benchmark data)")

    pipeline_start = time.time()
    failed_step    = None

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 0 — Discover and populate model pool from OpenRouter + AA
    # ─────────────────────────────────────────────────────────────────────────
    if args.stop_after >= 0 and failed_step is None:
        import yaml as _yaml
        with open(models_yaml, encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f)
        discovery_enabled = (_cfg or {}).get("discovery", {}).get("enabled", False)

        if not discovery_enabled:
            skip("Step 0 — Model discovery disabled (discovery.enabled: false)")
        else:
            disc_args = ["--models", str(models_yaml)]
            if aa_key:
                disc_args += ["--api-key", aa_key]
            or_key = os.environ.get("OPENROUTER_API_KEY")
            if or_key:
                disc_args += ["--or-api-key", or_key]

            success = run_step(
                step_num    = 0,
                label       = "Discover model pool (OpenRouter + AA)",
                script      = HERE / "discover_models.py",
                args        = disc_args,
                output_file = None,
                force       = is_forced(0),
                dry_run     = args.dry_run,
            )
            if not success:
                failed_step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Fetch Artificial Analysis benchmark data
    # ─────────────────────────────────────────────────────────────────────────
    if args.stop_after >= 1 and failed_step is None:
        aa_args = [
            "--models", str(models_yaml),
            "--out",    str(registry),
            "--force",  # always re-fetch when this step runs
        ]
        if aa_key:
            aa_args += ["--api-key", aa_key]

        success = run_step(
            step_num    = 1,
            label       = "Fetch Artificial Analysis benchmark data",
            script      = HERE / "fetch_aa_data.py",
            args        = aa_args,
            output_file = registry,
            force       = is_forced(1),
            dry_run     = args.dry_run,
        )
        if not success:
            failed_step = 1

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Fetch Arena ELO scores
    # ─────────────────────────────────────────────────────────────────────────
    if args.stop_after >= 2 and failed_step is None:
        success = run_step(
            step_num    = 2,
            label       = "Fetch Arena.ai ELO scores",
            script      = HERE / "fetch_arena_data.py",
            args        = ["--registry", str(registry)],
            # Arena data is merged into registry — always re-run when step 2 is active
            output_file = None,
            force       = True,
            dry_run     = args.dry_run,
        )
        if not success:
            failed_step = 2

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Patch taxonomy with Arena ELO weights
    # Only runs if taxonomy already exists (first run skips — taxonomy must be
    # created manually via build_task_taxonomy.py once)
    # ─────────────────────────────────────────────────────────────────────────
    if args.stop_after >= 3 and failed_step is None:
        if taxonomy.exists():
            success = run_step(
                step_num    = 3,
                label       = "Patch taxonomy with Arena ELO weights",
                script      = HERE / "patch_taxonomy_arena.py",
                args        = [],
                output_file = None,
                force       = True,
                dry_run     = args.dry_run,
            )
            if not success:
                failed_step = 3
        else:
            warn(f"task_taxonomy.json not found — skipping step 3.")
            warn(f"Run build_task_taxonomy.py once to generate the taxonomy.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Rebuild benchmark weight matrix
    # ─────────────────────────────────────────────────────────────────────────
    if args.stop_after >= 4 and failed_step is None:
        if taxonomy.exists():
            success = run_step(
                step_num    = 4,
                label       = "Build benchmark weight matrix",
                script      = HERE / "build_benchmark_weights.py",
                args        = [
                    "--taxonomy", str(taxonomy),
                    "--out",      str(weights),
                ],
                output_file = weights,
                force       = is_forced(4),
                dry_run     = args.dry_run,
            )
            if not success:
                failed_step = 4
        else:
            warn("Skipping step 4 — no taxonomy file.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — Compute model feature vectors
    # ─────────────────────────────────────────────────────────────────────────
    if args.stop_after >= 5 and failed_step is None:
        feat_args = [
            "--registry", str(registry),
            "--weights",  str(weights),
            "--out",      str(features),
        ]
        if wcb_csv and wcb_csv.exists():
            feat_args += ["--wcb-csv", str(wcb_csv)]

        success = run_step(
            step_num    = 5,
            label       = "Compute model feature vectors",
            script      = HERE / "build_model_features.py",
            args        = feat_args,
            output_file = features,
            force       = is_forced(5),
            dry_run     = args.dry_run,
        )
        if not success:
            failed_step = 5

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — Train PerfRouter XGBoost
    # ─────────────────────────────────────────────────────────────────────────
    if args.stop_after >= 6 and failed_step is None:
        train_args = [
            "--features",    str(features),
            "--out",         str(router_out),
            "--cost-weight", str(args.cost_weight),
        ]
        if wcb_csv and wcb_csv.exists():
            train_args += ["--wcb-csv", str(wcb_csv)]

        success = run_step(
            step_num    = 6,
            label       = "Train PerfRouter XGBoost",
            script      = HERE / "train_perf_router.py",
            args        = train_args,
            output_file = router_out,
            force       = is_forced(6),
            dry_run     = args.dry_run,
        )
        if not success:
            failed_step = 6

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    print(f"\n{C.BOLD}{'═'*72}{C.RESET}")

    if failed_step:
        print(f"{C.BOLD}  Pipeline FAILED at step {failed_step}{C.RESET}")
        print(f"{C.BOLD}{'═'*72}{C.RESET}")
        err(f"Fix the issue and re-run with --force-step {failed_step}")
        sys.exit(1)
    else:
        print(f"{C.BOLD}  Pipeline complete  ({elapsed:.1f}s){C.RESET}")
        print(f"{C.BOLD}{'═'*72}{C.RESET}")

        if not args.dry_run:
            print(f"\n  Output files:")
            for path, label in [
                (registry,   "Model registry"),
                (taxonomy,   "Task taxonomy"),
                (weights,    "Benchmark weights"),
                (features,   "Model features"),
                (router_out, "PerfRouter weights"),
            ]:
                if path.exists():
                    size = path.stat().st_size
                    size_str = (
                        f"{size/1024/1024:.1f}MB" if size > 1024*1024 else
                        f"{size/1024:.0f}KB"      if size > 1024       else
                        f"{size}B"
                    )
                    ok(f"{label:<25} {path.name}  ({size_str})")
                else:
                    warn(f"{label:<25} {path.name}  (not produced)")

            print(f"\n  To deploy in optmod:")
            print(f"    cp {router_out} ~/workspace/optmod/routing/perfrouter/")
            print(f"    cp {features}   ~/workspace/optmod/routing/perfrouter/")
            print(f"    cp {registry}   ~/workspace/optmod/routing/perfrouter/")
            print(f"    cp {taxonomy}   ~/workspace/optmod/routing/perfrouter/")

        print()


if __name__ == "__main__":
    main()