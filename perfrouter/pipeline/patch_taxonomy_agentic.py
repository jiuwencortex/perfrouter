"""
Merge data/agentic_examples.json into data/task_taxonomy.json as `agentic_examples`.

Idempotent: replaces the key for every listed type, leaves everything else (definitions,
benchmark weights, `examples`) untouched, and never adds task types (that would need new
benchmark weights, affinity columns and an XGBoost retrain). Inference uses these prompts
when `example_pooling="max"`; training does not read them.

    uv run python -m perfrouter.pipeline.patch_taxonomy_agentic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent.parent / "data"


def patch(taxonomy_path: Path, examples_path: Path) -> int:
    taxonomy = json.loads(taxonomy_path.read_text())
    examples = json.loads(examples_path.read_text())["examples"]
    known = {t["id"]: t for t in taxonomy["task_types"]}
    unknown = sorted(set(examples) - set(known))
    if unknown:
        raise SystemExit(f"unknown task types in {examples_path.name}: {unknown}")
    for tid, items in examples.items():
        known[tid]["agentic_examples"] = list(items)
    taxonomy_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n")
    return sum(len(v) for v in examples.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxonomy", type=Path, default=_DATA / "task_taxonomy.json")
    ap.add_argument("--examples", type=Path, default=_DATA / "agentic_examples.json")
    a = ap.parse_args()
    n = patch(a.taxonomy, a.examples)
    print(f"merged {n} agentic examples into {a.taxonomy}")


if __name__ == "__main__":
    main()
