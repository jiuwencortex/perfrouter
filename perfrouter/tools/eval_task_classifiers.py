"""
Compare task-type classifiers on SkillsBench prompts against hand-made acceptable label sets.

    PYTHONPATH=~/workspace/perfrouter uv run --directory ~/workspace/optmod \
        python -m perfrouter.tools.eval_task_classifiers \
        --methods minilm_legacy,minilm_head,minilm_chunk,laya_flat,laya_2stage,jev_flat,llm_flat

Labels: data/skillsbench_task_labels.json (acceptable sets, first = primary; written by an LLM from
the task openings, so noisy). Metrics per method:
  primary   top-1 equals the first (primary) label
  accept    top-1 is any acceptable label
  family    top-1's family (id prefix) matches the family of any acceptable label
  conf      mean top-1 confidence (probability or cosine similarity; not comparable across methods)

Every method returns one of the taxonomy's task-type ids (the quality model has one affinity column
per type, so a classifier must map onto them). Results are cached per method in --cache so reruns
only pay for new methods. Optional dependencies are imported lazily per method.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
HEAD_CHARS = 2000                      # what hosted/other models receive (~500 tokens)
REQUEST_Q = "Which kind of work is `request` mainly asking an AI agent to do?"


def task_body(path: Path) -> str:
    text = path.read_text()
    if text.startswith("---"):
        parts = text.split("\n---", 2)
        text = parts[-1] if len(parts) > 1 else text
    return text.strip()


class Ctx:
    def __init__(self, data: Path, models: Path, tasks: Path):
        self.data, self.models, self.tasks = data, models, tasks
        tax = json.loads((data / "task_taxonomy.json").read_text())["task_types"]
        self.types = [t["id"] for t in tax]
        self.defs = {t["id"]: t["definition"] for t in tax}
        self.families = sorted({t.split(".")[0] for t in self.types})
        self._pr = {}

    def perf(self, pooling: str):
        if pooling not in self._pr:
            from perfrouter.inference.perf_router_inference import PerfRouterInference
            d = self.data
            self._pr[pooling] = PerfRouterInference(
                self.models / "perf_router.pkl", d / "task_taxonomy.json", d / "model_registry.json",
                d / "model_features.csv", example_pooling=pooling)
        return self._pr[pooling]


# ── methods: fn(ctx, body) -> (task_type_id, confidence) ─────────────────────

def minilm(pooling: str) -> Callable:
    def run(ctx: Ctx, body: str):
        top = ctx.perf(pooling).classify_task(body)
        return top[0][0], float(top[0][1])
    return run


def minilm_chunk(ctx: Ctx, body: str):
    pr = ctx.perf("max")
    tok = pr._encoder.tokenizer
    ids = tok(body, add_special_tokens=False)["input_ids"]
    chunks = [tok.decode(ids[i:i + 200]) for i in range(0, max(1, len(ids) - 50), 150)] or [body]
    sims = []
    for c in chunks:
        e = pr._encoder.encode(c, normalize_embeddings=True, show_progress_bar=False)
        s = np.full(pr._n_task_types, -1.0)
        np.maximum.at(s, pr._example_owner, pr._example_embeddings @ e)
        sims.append(s)
    m = np.mean(sims, axis=0)
    i = int(m.argmax())
    return pr._task_type_ids_ordered[i], float(m[i])


_LAYA = {}


def _laya(ctx: Ctx):
    if "a" not in _LAYA:
        import laya
        _LAYA["a"] = laya.load("convaiinnovations/laya")
    return _LAYA["a"]


def _choice(question: dict, predict) -> tuple[str, float]:
    ans = predict(question)
    probs = ans.get("probabilities") or {}
    c = ans["choice"]
    return c, float(probs.get(c, ans.get("answer_confidence") or 0.0))


def _flat_q(ctx: Ctx, ids: list[str]) -> dict:
    return {"task_type": {"type": "choice", "instructions": REQUEST_Q,
                          "criteria": {t: ctx.defs[t] for t in ids}}}


def _fam_q(ctx: Ctx) -> dict:
    fam_def = {f: "; ".join(ctx.defs[t].rstrip(".") for t in ctx.types if t.startswith(f + "."))[:400]
               for f in ctx.families}
    return {"family": {"type": "choice", "instructions": REQUEST_Q, "criteria": fam_def}}


def laya_flat(ctx: Ctx, body: str):
    a = _laya(ctx)
    return _choice(_flat_q(ctx, ctx.types),
                   lambda q: a.predict({"request": body[:HEAD_CHARS]}, q)["answers"]["task_type"])


def laya_2stage(ctx: Ctx, body: str):
    a = _laya(ctx)
    st = {"request": body[:HEAD_CHARS]}
    fam, p1 = _choice(_fam_q(ctx), lambda q: a.predict(st, q)["answers"]["family"])
    ids = [t for t in ctx.types if t.startswith(fam + ".")]
    if len(ids) == 1:
        return ids[0], p1
    t, p2 = _choice(_flat_q(ctx, ids), lambda q: a.predict(st, q)["answers"]["task_type"])
    return t, p1 * p2


def _jev_call(state: dict, questions: dict, retries: int = 3) -> dict:
    import requests
    key = os.environ["OPENROUTER_API_KEY"]
    for i in range(retries):
        r = requests.post("https://openrouter.ai/api/alpha/decisions",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": "typesafe/jev-1.13", "state": state, "questions": questions},
                          timeout=90)
        if r.status_code == 200:
            return r.json()["answers"]
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"jev {r.status_code}: {r.text[:200]}")


def jev_flat(ctx: Ctx, body: str):
    ans = _jev_call({"request": body[:HEAD_CHARS]}, _flat_q(ctx, ctx.types))["task_type"]
    return _choice(None, lambda _q: ans)


def jev_2stage(ctx: Ctx, body: str):
    st = {"request": body[:HEAD_CHARS]}
    a1 = _jev_call(st, _fam_q(ctx))["family"]
    fam, p1 = a1["choice"], float(a1["probabilities"][a1["choice"]])
    ids = [t for t in ctx.types if t.startswith(fam + ".")]
    if len(ids) == 1:
        return ids[0], p1
    a2 = _jev_call(st, _flat_q(ctx, ids))["task_type"]
    return a2["choice"], p1 * float(a2["probabilities"][a2["choice"]])


def llm_flat(ctx: Ctx, body: str):
    """Reference: a general chat LLM asked to pick one type (not a decision model)."""
    import requests
    key = os.environ["OPENROUTER_API_KEY"]
    menu = "\n".join(f"- {t}: {ctx.defs[t]}" for t in ctx.types)
    prompt = (f"Classify the task below into exactly one task type.\n\nTask types:\n{menu}\n\n"
              f"Task:\n{body[:HEAD_CHARS]}\n\nAnswer with only the task type id.")
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": "deepseek/deepseek-v4-flash", "max_tokens": 30, "temperature": 0,
                            "messages": [{"role": "user", "content": prompt}]}, timeout=90)
    out = r.json()["choices"][0]["message"]["content"].strip().strip("`").split()[0] if r.ok else ""
    return (out if out in ctx.types else "unparsed:" + out[:30]), 1.0


_ZS = {}


def zeroshot_nli(model: str):
    def run(ctx: Ctx, body: str):
        if model not in _ZS:
            import torch
            from transformers import pipeline
            _ZS[model] = pipeline("zero-shot-classification", model=model,
                                  device=0 if torch.cuda.is_available() else -1)
        labels = [ctx.defs[t].rstrip(".") for t in ctx.types]
        r = _ZS[model](body[:HEAD_CHARS], labels, hypothesis_template="The task is about: {}.",
                       multi_label=False)
        return ctx.types[labels.index(r["labels"][0])], float(r["scores"][0])
    return run


def gliclass(model: str):
    def run(ctx: Ctx, body: str):
        if model not in _ZS:
            import torch
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
            m = GLiClassModel.from_pretrained(model)
            tk = AutoTokenizer.from_pretrained(model)
            _ZS[model] = ZeroShotClassificationPipeline(
                m, tk, classification_type="single-label",
                device="cuda:0" if torch.cuda.is_available() else "cpu")
        labels = [ctx.defs[t].rstrip(".") for t in ctx.types]
        res = _ZS[model](body[:HEAD_CHARS], labels, threshold=0.0)[0]
        best = max(res, key=lambda x: x["score"])
        return ctx.types[labels.index(best["label"])], float(best["score"])
    return run


METHODS: dict[str, Callable] = {
    "minilm_legacy": minilm("none"),
    "minilm_head": minilm("max"),
    "minilm_chunk": minilm_chunk,
    "laya_flat": laya_flat,
    "laya_2stage": laya_2stage,
    "jev_flat": jev_flat,
    "jev_2stage": jev_2stage,
    "llm_flat": llm_flat,
    "nli_modernbert": zeroshot_nli("MoritzLaurer/ModernBERT-large-zeroshot-v2.0"),
    "nli_deberta": zeroshot_nli("MoritzLaurer/deberta-v3-large-zeroshot-v2.0"),
    "gliclass_modern": gliclass("knowledgator/gliclass-modern-base-v3.0"),
    "gliclass_large": gliclass("knowledgator/gliclass-large-v3.0"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, default=Path.home() / "workspace/skillsbench/tasks")
    ap.add_argument("--methods", default="minilm_legacy,minilm_head,minilm_chunk")
    ap.add_argument("--data", type=Path, default=_ROOT / "data")
    ap.add_argument("--models", type=Path, default=_ROOT / "models")
    ap.add_argument("--cache", type=Path, default=Path("/tmp/task_classifier_eval.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore cached results")
    a = ap.parse_args()

    labels = json.loads((a.data / "skillsbench_task_labels.json").read_text())["labels"]
    files = sorted(a.tasks.glob("*/task.md"))
    if a.limit:
        files = files[: a.limit]
    ctx = Ctx(a.data, a.models, a.tasks)
    cache = {} if a.fresh or not a.cache.exists() else json.loads(a.cache.read_text())

    summary = []
    wrong: dict[str, list] = defaultdict(list)
    for name in a.methods.split(","):
        fn = METHODS[name]
        res = cache.setdefault(name, {})
        t0 = time.perf_counter()
        n_new = 0
        for f in files:
            task = f.parent.name
            if task in res:
                continue
            try:
                res[task] = list(fn(ctx, task_body(f)))
            except Exception as exc:   # one bad call must not lose the run
                res[task] = [f"error:{type(exc).__name__}", 0.0]
            n_new += 1
        a.cache.write_text(json.dumps(cache))
        dt = (time.perf_counter() - t0) / max(n_new, 1) * 1000
        prim = acc = fam = 0
        confs = []
        for f in files:
            task = f.parent.name
            t, c = res[task]
            ok = labels[task]
            prim += t == ok[0]
            acc += t in ok
            fam += t.split(".")[0] in {x.split(".")[0] for x in ok}
            confs.append(c)
            if t not in ok:
                wrong[name].append((task, t, ok))
        n = len(files)
        summary.append((name, prim / n, acc / n, fam / n, float(np.mean(confs)), dt))

    print(f"\n{len(files)} tasks\n{'method':16s} {'primary':>8s} {'accept':>8s} {'family':>8s} {'conf':>6s} {'ms/task':>8s}")
    for name, p, ac, fm, cf, ms in summary:
        print(f"{name:16s} {100*p:7.0f}% {100*ac:7.0f}% {100*fm:7.0f}% {cf:6.2f} {ms:8.0f}")


if __name__ == "__main__":
    main()
