#!/usr/bin/env python3
"""
train_perf_router.py
====================
Trains PerfRouter — an XGBoost model that predicts which model in the pool
will produce the best utility (quality - α×cost) for a given task type.

─────────────────────────────────────────────────────────────────────
ARCHITECTURE
─────────────────────────────────────────────────────────────────────

Unlike TRouter (which routes based on query embeddings), PerfRouter
routes based on task type + model features. It answers the question:

  "Given that this query is of type X, which model has the best
   expected utility (quality adjusted for cost)?"

The XGBoost model is trained per-task-type:
  - Input:  model feature vector (49 features from model_features.csv)
  - Output: predicted utility score for that model on that task type

At inference time:
  1. Classify the incoming query into a task type (via sentence-BERT)
  2. For each model in the pool, predict its utility using XGBoost
  3. Route to the model with the highest predicted utility

─────────────────────────────────────────────────────────────────────
TRAINING DATA
─────────────────────────────────────────────────────────────────────

Training rows come from two sources:

  A) WCB ground truth (high confidence)
     For the models that ran WildClawBench, we have real
     overall_score per task type. These rows get sample_weight=3.0.

  B) Benchmark-derived affinity scores (lower confidence)
     For all models, the affinity score per task type is the
     weighted average of their benchmark scores. These rows get
     sample_weight=1.0.

The utility label combines quality and cost:
  utility = score - α × normalised_cost

Where:
  score          = WCB overall_score (source A) or affinity score (source B)
  normalised_cost = price_blended_per_1M / max(price_blended_per_1M across pool)
  α              = cost_weight (default 0.3)

─────────────────────────────────────────────────────────────────────
CROSS-VALIDATION
─────────────────────────────────────────────────────────────────────

10-fold stratified cross-validation is used to estimate generalisation
performance. Stratification is by task_type so every fold contains
rows from all task types.

CV is purely for evaluation — the final deployed model is trained on
all data (standard practice). CV metrics are saved in the checkpoint
alongside training metrics so they are visible in logs and dashboards.

─────────────────────────────────────────────────────────────────────
WHY XGBOOST
─────────────────────────────────────────────────────────────────────

With 50-80 models × 33 task types = 1,650–2,640 training rows,
a neural network would overfit badly. XGBoost:
  - Handles small datasets naturally (built-in regularisation)
  - Handles missing features (None/NaN) natively — no imputation needed
  - Produces feature importance scores
  - Fast to train (sub-second) and fast to infer (microseconds)
  - Robust to irrelevant features

─────────────────────────────────────────────────────────────────────
Usage:
  pip install xgboost scikit-learn
  python3 train_perf_router.py --features model_features.csv
  python3 train_perf_router.py --features model_features.csv \\
      --cost-weight 0.5 --out perf_router.pkl
  python3 train_perf_router.py --features model_features.csv \\
      --cv-folds 10 --no-cv   # skip CV, train-only
─────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"
_MODELS_DIR   = _PROJECT_ROOT / "models"


# ── Constants ─────────────────────────────────────────────────────────────────

EXCLUDE_FROM_FEATURES = {
    "model_id",
    "wcb_avg_score",
    "wcb_avg_score_nonzero",
    "wcb_zero_rate",
    # Pricing excluded from XGBoost features — cost is applied explicitly
    # at inference time as: adjusted_utility = predicted_quality - α × cost
    # Including it here would cause XGBoost to double-count cost signal.
    "price_blended_per_1M",
    "price_input_per_1M",
    "price_output_per_1M",
    "price_cache_read_per_1M",
}

WCB_SAMPLE_WEIGHT  = 3.0   # rows with actual WCB ground-truth labels
BASE_SAMPLE_WEIGHT = 1.0   # benchmark-derived rows

DEFAULT_CV_FOLDS   = 10


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            converted = {}
            for k, v in row.items():
                if v == "" or v == "None":
                    converted[k] = None
                else:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v
            rows.append(converted)
    return rows


def get_task_types(rows: list[dict]) -> list[str]:
    """Extract task type names from affinity column names."""
    sample = rows[0]
    return sorted([
        k.replace("affinity_", "").replace("_", ".", 1)
        for k in sample if k.startswith("affinity_")
    ])


def affinity_col(task_type: str) -> str:
    """task_type → column name in features CSV."""
    return "affinity_" + task_type.replace(".", "_", 1)


# ── Training data builder ─────────────────────────────────────────────────────

def build_training_data(
    feature_rows: list[dict],
    task_types:   list[str],
    cost_weight:  float,
    wcb_csv_path: Path | None,
) -> tuple[list, list, list, list, list]:
    """
    Build XGBoost training data: X (features), y (utility labels),
    weights (sample weights), task_labels (one per row), feature_cols.

    Returns (X, y, weights, task_labels, feature_cols).
    """
    wcb_scores = _load_wcb_scores(wcb_csv_path)

    costs    = [r.get("price_blended_per_1M") or 0.0 for r in feature_rows]
    max_cost = max(costs) if any(c > 0 for c in costs) else 1.0

    all_cols = [k for k in feature_rows[0].keys()
                if k not in EXCLUDE_FROM_FEATURES]

    X, y, weights, task_labels = [], [], [], []

    for task_type in task_types:
        acol = affinity_col(task_type)

        for row in feature_rows:
            model_id = row["model_id"]

            wcb_key = (_normalise_model_id(model_id), task_type)
            has_wcb = wcb_key in wcb_scores
            score   = wcb_scores[wcb_key] if has_wcb else row.get(acol)

            if score is None:
                continue

            # Label = quality score only (cost applied at inference time)
            utility = score

            acol_active = affinity_col(task_type)
            x_row = []
            for col in all_cols:
                val = row.get(col)
                if col.startswith("affinity_") and col != acol_active:
                    x_row.append(None)
                else:
                    x_row.append(val)

            weight = WCB_SAMPLE_WEIGHT if has_wcb else BASE_SAMPLE_WEIGHT

            X.append(x_row)
            y.append(utility)
            weights.append(weight)
            task_labels.append(task_type)

    # Normalise labels within each task type
    from collections import defaultdict as _dd
    task_scores = _dd(list)
    for i, tl in enumerate(task_labels):
        task_scores[tl].append((i, y[i]))

    y_norm = list(y)
    for tl, idx_scores in task_scores.items():
        vals = [s for _, s in idx_scores]
        mn, mx = min(vals), max(vals)
        spread = mx - mn
        for idx, score in idx_scores:
            if spread > 0.01:
                y_norm[idx] = (score - mn) / spread
            else:
                y_norm[idx] = 0.5

    return X, y_norm, weights, task_labels, all_cols


def _normalise_model_id(model_id: str) -> str:
    import re
    s = model_id.lower()
    if "/" in s:
        s = s.split("/")[-1]
    s = re.sub(r"[_:]free$", "", s)
    parts = s.split("-")
    if len(parts) > 1 and parts[0] == parts[1]:
        parts = parts[1:]
    s = "-".join(parts)
    s = re.sub(r"[-_./]+", "_", s)
    return s.strip("_")


def _load_wcb_scores(wcb_csv_path: Path | None) -> dict:
    if wcb_csv_path is None or not wcb_csv_path.exists():
        return {}

    raw: dict[tuple, list[float]] = defaultdict(list)
    with open(wcb_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            norm_model = _normalise_model_id(row["model"])
            key = (norm_model, row["task_type"])
            raw[key].append(float(row["overall_score"]))

    return {k: sum(v)/len(v) for k, v in raw.items()}


# ── XGBoost hyperparameters ───────────────────────────────────────────────────

def _xgb_params() -> dict:
    """Centralised XGBoost hyperparameters — shared by CV and final fit."""
    return dict(
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        tree_method       = "hist",
        enable_categorical = False,
    )


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(
    X_np,
    y_np,
    w_np,
    task_labels: list[str],
    n_folds:     int = DEFAULT_CV_FOLDS,
) -> dict:
    """
    10-fold stratified cross-validation, stratified by task_type.

    Stratification ensures every fold contains rows from all task types,
    which is critical because XGBoost learns task-type-aware routing —
    a fold missing a task type would produce biased OOF predictions for it.

    Returns a dict with per-fold and aggregate metrics.
    """
    try:
        import numpy as np
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as e:
        print(f"ERROR: {e}. Run: pip install xgboost scikit-learn numpy", file=sys.stderr)
        sys.exit(1)

    print(f"\n── {n_folds}-fold stratified cross-validation ────────────────────────────")
    print(f"  Stratified by task_type ({len(set(task_labels))} unique types)")
    print(f"  {len(X_np)} total rows → ~{len(X_np)//n_folds} rows per fold\n")

    skf         = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    task_arr    = np.array(task_labels)

    fold_mae, fold_rmse, fold_r2 = [], [], []

    # Routing accuracy: for each task type in each fold, did the model
    # with the highest OOF-predicted score match the actual best model?
    fold_routing_acc = []

    print(f"  {'Fold':>4}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  {'RoutingAcc':>11}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*11}")

    oof_preds  = np.zeros_like(y_np)   # out-of-fold predictions (full array)
    oof_mask   = np.zeros(len(y_np), dtype=bool)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_np, task_arr)):
        X_tr, X_val = X_np[train_idx], X_np[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]
        w_tr        = w_np[train_idx]

        model = xgb.XGBRegressor(**_xgb_params())
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        oof_mask[val_idx]  = True

        mae  = mean_absolute_error(y_val, preds)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        r2   = r2_score(y_val, preds)

        # Routing accuracy: per task type, does argmax(predicted) == argmax(actual)?
        task_types_in_fold = np.unique(task_arr[val_idx])
        correct, total = 0, 0
        for tt in task_types_in_fold:
            tt_mask  = task_arr[val_idx] == tt
            if tt_mask.sum() < 2:
                continue   # can't measure ranking with 1 model
            actual_best = int(np.argmax(y_val[tt_mask]))
            pred_best   = int(np.argmax(preds[tt_mask]))
            correct    += int(actual_best == pred_best)
            total      += 1

        routing_acc = correct / total if total > 0 else float("nan")
        fold_routing_acc.append(routing_acc)

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        routing_str = f"{routing_acc:.1%}" if not np.isnan(routing_acc) else "  n/a"
        print(f"  {fold_idx+1:>4}  {mae:>7.4f}  {rmse:>7.4f}  {r2:>7.4f}  {routing_str:>11}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mae_arr  = np.array(fold_mae)
    rmse_arr = np.array(fold_rmse)
    r2_arr   = np.array(fold_r2)
    ra_arr   = np.array([x for x in fold_routing_acc if not np.isnan(x)])

    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*11}")
    print(f"  {'mean':>4}  {mae_arr.mean():>7.4f}  {rmse_arr.mean():>7.4f}  "
          f"{r2_arr.mean():>7.4f}  {ra_arr.mean():>10.1%}")
    print(f"  {'±std':>4}  {mae_arr.std():>7.4f}  {rmse_arr.std():>7.4f}  "
          f"{r2_arr.std():>7.4f}  {ra_arr.std():>10.1%}")

    # OOF R² (computed over all held-out predictions simultaneously —
    # more robust than mean of per-fold R² values)
    oof_r2 = r2_score(y_np[oof_mask], oof_preds[oof_mask])
    print(f"\n  OOF R² (all folds combined): {oof_r2:.4f}")
    print(f"  Routing accuracy: how often argmax(predicted) == argmax(actual)")
    print(f"  per task type within each validation fold.")

    return {
        "n_folds":           n_folds,
        "fold_mae":          fold_mae,
        "fold_rmse":         fold_rmse,
        "fold_r2":           fold_r2,
        "fold_routing_acc":  fold_routing_acc,
        "mean_mae":          float(mae_arr.mean()),
        "std_mae":           float(mae_arr.std()),
        "mean_rmse":         float(rmse_arr.mean()),
        "std_rmse":          float(rmse_arr.std()),
        "mean_r2":           float(r2_arr.mean()),
        "std_r2":            float(r2_arr.std()),
        "mean_routing_acc":  float(ra_arr.mean()),
        "std_routing_acc":   float(ra_arr.std()),
        "oof_r2":            float(oof_r2),
    }


# ── Final model training (all data) ──────────────────────────────────────────

def train(X, y, weights, feature_cols: list[str], cost_weight: float):
    """
    Train final XGBoost model on all data.

    CV is used for evaluation only. The deployed model always trains
    on the full dataset for maximum coverage of model×task combinations.
    """
    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as e:
        print(f"ERROR: {e}. Run: pip install xgboost numpy", file=sys.stderr)
        sys.exit(1)

    X_np = np.array([
        [float("nan") if v is None else float(v) for v in row]
        for row in X
    ], dtype=np.float32)
    y_np = np.array(y,       dtype=np.float32)
    w_np = np.array(weights, dtype=np.float32)

    print(f"\n── Final model (trained on all data) ────────────────────────────────")
    print(f"  {len(X_np)} rows × {X_np.shape[1]} features")
    print(f"  Label range: [{y_np.min():.3f}, {y_np.max():.3f}]")
    print(f"  NaN rate per feature: "
          f"{(np.isnan(X_np).mean(axis=0) * 100).mean():.1f}% avg")

    model = xgb.XGBRegressor(**_xgb_params())
    model.fit(X_np, y_np, sample_weight=w_np)

    return model, X_np, y_np, w_np


# ── Evaluation (in-sample + routing simulation) ───────────────────────────────

def evaluate(model, X_np, y_np, feature_cols, task_labels, feature_rows):
    """
    Print in-sample training metrics and feature importance.

    Note: these are in-sample metrics — use CV metrics for unbiased
    generalisation estimates.
    """
    import numpy as np

    y_pred    = model.predict(X_np)
    residuals = y_pred - y_np
    mae  = float(np.abs(residuals).mean())
    rmse = float(np.sqrt((residuals ** 2).mean()))
    r2   = float(1 - np.var(residuals) / np.var(y_np))

    print(f"\n── In-sample training metrics (optimistic — see CV for unbiased) ────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:15]
    print(f"\n── Top 15 features by importance ────────────────────────────────────")
    for i in top_idx:
        bar = "█" * int(importances[i] * 200)
        print(f"  {feature_cols[i]:<50} {importances[i]:.4f}  {bar}")

    print(f"\n── Routing simulation per task type ─────────────────────────────────")
    print(f"  (Which model PerfRouter would choose for each task type)")

    model_ids = [r["model_id"] for r in feature_rows]
    n_models  = len(model_ids)

    task_type_groups = defaultdict(list)
    for i, tl in enumerate(task_labels):
        task_type_groups[tl].append(i)

    for task_type in sorted(task_type_groups.keys()):
        indices = task_type_groups[task_type]
        if len(indices) < n_models:
            continue

        preds  = y_pred[indices]
        costs  = np.array([
            feature_rows[i % n_models].get("price_blended_per_1M") or 0.0
            for i in indices
        ])
        max_c      = costs.max() if costs.max() > 0 else 1.0
        adjusted   = preds - 0.3 * (costs / max_c)
        best_idx   = int(np.argmax(adjusted))
        best_model = model_ids[best_idx % n_models]
        best_score = adjusted[best_idx]

        short = best_model.split("/")[-1][:35]
        print(f"  {task_type:<45} → {short:<35} ({best_score:.3f})")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PerfRouter XGBoost model on model feature vectors"
    )
    parser.add_argument("--features",    default=str(_DATA_DIR / "model_features.csv"))
    parser.add_argument("--wcb-csv",     default=None,
                        help="WildClawBench labeled CSV for ground-truth labels")
    parser.add_argument("--out",         default=str(_MODELS_DIR / "perf_router.pkl"))
    parser.add_argument("--cost-weight", type=float, default=0.3,
                        help="α: quality-cost trade-off (default 0.3)")
    parser.add_argument("--cv-folds",    type=int, default=DEFAULT_CV_FOLDS,
                        help=f"Number of CV folds (default: {DEFAULT_CV_FOLDS})")
    parser.add_argument("--no-cv",       action="store_true",
                        help="Skip cross-validation (faster, train-only)")
    args = parser.parse_args()

    features_path = Path(args.features).expanduser()
    wcb_path      = Path(args.wcb_csv).expanduser() if args.wcb_csv else None
    out_path      = Path(args.out).expanduser()

    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run build_model_features.py first.",
              file=sys.stderr)
        sys.exit(1)

    # ── Load features ─────────────────────────────────────────────────────────
    print(f"Loading features from {features_path}")
    feature_rows = load_features(features_path)
    task_types   = get_task_types(feature_rows)
    print(f"  {len(feature_rows)} models × {len(task_types)} task types")

    # ── Build training data ───────────────────────────────────────────────────
    print(f"\nBuilding training data (cost_weight α={args.cost_weight})...")
    X, y, weights, task_labels, feature_cols = build_training_data(
        feature_rows, task_types, args.cost_weight, wcb_path
    )
    print(f"  {len(X)} training rows")
    print(f"  {feature_cols[:5]}... ({len(feature_cols)} features total)")

    wcb_rows   = sum(1 for w in weights if w == WCB_SAMPLE_WEIGHT)
    bench_rows = len(weights) - wcb_rows
    if wcb_rows:
        print(f"  WCB ground-truth rows : {wcb_rows} (weight={WCB_SAMPLE_WEIGHT})")
        print(f"  Benchmark-derived rows: {bench_rows} (weight={BASE_SAMPLE_WEIGHT})")
    else:
        print(f"  Benchmark-derived rows: {bench_rows} (weight={BASE_SAMPLE_WEIGHT}, no WCB seed)")

    if len(X) < 10:
        print("ERROR: Too few training rows. Check feature CSV.", file=sys.stderr)
        sys.exit(1)

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_results = None
    if not args.no_cv:
        try:
            import numpy as np
            X_np_cv = np.array([
                [float("nan") if v is None else float(v) for v in row]
                for row in X
            ], dtype=np.float32)
            y_np_cv = np.array(y,       dtype=np.float32)
            w_np_cv = np.array(weights, dtype=np.float32)

            # Guard: StratifiedKFold requires at least n_folds samples per class.
            # With many task types and few models, some task types may have too
            # few rows to split into n_folds folds — warn and reduce if needed.
            from collections import Counter
            task_counts = Counter(task_labels)
            min_count   = min(task_counts.values())
            effective_folds = min(args.cv_folds, min_count)
            if effective_folds < args.cv_folds:
                print(f"\n  WARNING: Reducing CV folds from {args.cv_folds} to "
                      f"{effective_folds} — smallest task type has only "
                      f"{min_count} rows.")

            if effective_folds < 2:
                print("  WARNING: Cannot run CV — too few rows per task type. "
                      "Skipping CV.")
            else:
                cv_results = cross_validate(
                    X_np_cv, y_np_cv, w_np_cv,
                    task_labels,
                    n_folds=effective_folds,
                )
        except Exception as e:
            print(f"\n  WARNING: CV failed ({e}). Continuing with final training.")
    else:
        print("\n  Cross-validation skipped (--no-cv)")

    # ── Train final model on all data ─────────────────────────────────────────
    print(f"\nTraining final XGBoost on all data...")
    xgb_model, X_np, y_np, w_np = train(X, y, weights, feature_cols, args.cost_weight)

    # ── Evaluate (in-sample) ──────────────────────────────────────────────────
    train_metrics = evaluate(xgb_model, X_np, y_np, feature_cols, task_labels, feature_rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dict = {
        "model":        xgb_model,
        "feature_cols": feature_cols,
        "task_types":   task_types,
        "model_ids":    [r["model_id"] for r in feature_rows],
        "cost_weight":  args.cost_weight,
        "training": {
            "n_rows":               len(X),
            "wcb_rows":             wcb_rows,
            "benchmark_rows":       bench_rows,
            "wcb_sample_weight":    WCB_SAMPLE_WEIGHT,
            "in_sample_mae":        train_metrics["mae"],
            "in_sample_rmse":       train_metrics["rmse"],
            "in_sample_r2":         train_metrics["r2"],
        },
        # CV results stored separately for easy access in logs/dashboards.
        # None if --no-cv was passed or CV failed.
        "cv": cv_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(save_dict, f)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  PerfRouter trained and saved → {out_path}")
    print(f"{'═'*72}")
    if cv_results:
        print(f"  CV  MAE  : {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")
        print(f"  CV  RMSE : {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
        print(f"  CV  R²   : {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        print(f"  CV  OOF R²     : {cv_results['oof_r2']:.4f}")
        print(f"  CV  Routing Acc: {cv_results['mean_routing_acc']:.1%} "
              f"± {cv_results['std_routing_acc']:.1%}")
        print(f"  In-sample R²   : {train_metrics['r2']:.4f}  "
              f"(optimistic — CV is the reliable signal)")
    else:
        print(f"  In-sample MAE  : {train_metrics['mae']:.4f}")
        print(f"  In-sample RMSE : {train_metrics['rmse']:.4f}")
        print(f"  In-sample R²   : {train_metrics['r2']:.4f}")
    print(f"\n  Next step: python3 perf_router_inference.py")


if __name__ == "__main__":
    main()