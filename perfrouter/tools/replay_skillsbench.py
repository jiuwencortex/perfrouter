"""
Replay SkillsBench task prompts through PerfRouter's task classifier and report how it behaves.

    uv run python -m perfrouter.tools.replay_skillsbench --tasks ~/workspace/skillsbench/tasks \
        [--pooling none|max] [--min-sim 0.2] [--holdout-every 4] [--list]

The prompt is the task.md body after the YAML front matter (first --chars characters).
Reports: top-1 similarity distribution, share below the min-similarity threshold, and the
gap between the top-1 and top-2 types. Tasks whose sorted index % holdout-every == 0 form the
held-out set (the agentic examples were written without looking at any task text).
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from perfrouter.inference.perf_router_inference import PerfRouterInference

_ROOT = Path(__file__).resolve().parent.parent.parent


def task_body(path: Path, chars: int) -> str:
    text = path.read_text()
    if text.startswith("---"):
        parts = text.split("\n---", 2)
        text = parts[-1] if len(parts) > 1 else text
    return text.strip()[:chars]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--pooling", default="max", choices=["max", "none"])
    ap.add_argument("--min-sim", type=float, default=0.20)
    ap.add_argument("--chars", type=int, default=1500)
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print every task's top-3")
    ap.add_argument("--data", type=Path, default=_ROOT / "data")
    ap.add_argument("--models", type=Path, default=_ROOT / "models")
    a = ap.parse_args()

    pr = PerfRouterInference(
        a.models / "perf_router.pkl", a.data / "task_taxonomy.json",
        a.data / "model_registry.json", a.data / "model_features.csv",
        min_similarity_threshold=a.min_sim, example_pooling=a.pooling,
    )
    files = sorted(a.tasks.glob("*/task.md"))
    rows = []
    for i, f in enumerate(files):
        top = pr.classify_task(task_body(f, a.chars))
        rows.append((f.parent.name, i % a.holdout_every == 0, top))
        if a.list:
            print(f"{f.parent.name:45s} " + "  ".join(f"{t}:{s:.2f}" for t, s in top))

    def report(label: str, sel: list) -> None:
        sims = [r[2][0][1] for r in sel]
        gaps = [r[2][0][1] - r[2][1][1] for r in sel]
        types = {r[2][0][0] for r in sel}
        print(f"{label:9s} n={len(sel):3d}  median top-1 sim={statistics.median(sims):.3f}  "
              f"below {a.min_sim}: {sum(s < a.min_sim for s in sims):3d} "
              f"({100 * sum(s < a.min_sim for s in sims) / len(sel):.0f}%)  "
              f"median top1-top2 gap={statistics.median(gaps):.3f}  "
              f"gap<0.02: {100 * sum(g < 0.02 for g in gaps) / len(sel):.0f}%  distinct top-1 types={len(types)}")

    print(f"pooling={a.pooling} min_sim={a.min_sim}")
    report("all", rows)
    report("held-out", [r for r in rows if r[1]])
    report("train", [r for r in rows if not r[1]])


if __name__ == "__main__":
    main()
