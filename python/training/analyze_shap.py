"""SHAP and coefficient analysis for V6.2 trained models.

Supports all model types:
- Tree-based (DT, RF, GB): SHAP TreeExplainer
- Linear (LogisticRegression, LinearSVC): coefficient magnitude
- Naive Bayes: class-conditional mean (theta) spread

Aggregates importance by V6.1 feature groups for ablation guidance.

Usage:
    cd quant-trading-gym/python
    python -m training.analyze_shap
    python -m training.analyze_shap --config training/config.yaml --max-samples 2000
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .common import load_and_split, load_config
from .feature_schema import FEATURE_GROUPS, feature_group

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Per-model importance extraction
# ─────────────────────────────────────────────────────────────────────────────


def shap_importance(model, X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, str]:
    """Compute mean |SHAP| per feature via TreeExplainer.

    Falls back to sklearn feature_importances_ if SHAP doesn't support
    the model (e.g. multi-class GradientBoostingClassifier).

    Returns (importance_array, method_name).
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            # Multi-class (older shap): list of [n_samples, n_features] per class
            return sum(np.abs(sv).mean(axis=0) for sv in shap_values) / len(shap_values), "SHAP TreeExplainer"

        sv = np.abs(shap_values)
        if sv.ndim == 3:
            # Multi-class (newer shap): [n_samples, n_features, n_classes]
            return sv.mean(axis=(0, 2)), "SHAP TreeExplainer"
        return sv.mean(axis=0), "SHAP TreeExplainer"
    except Exception as e:
        # GradientBoostingClassifier multi-class not supported by TreeExplainer
        if hasattr(model, "feature_importances_"):
            print(f"  SHAP unsupported ({e}), using sklearn feature_importances_")
            return model.feature_importances_, "sklearn feature_importances"
        raise


def coefficient_importance(json_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """Extract coefficient magnitude from linear/SVM JSON.

    Returns (importance_per_feature, feature_names) or None.
    """
    with open(json_path) as f:
        data = json.load(f)

    model_type = data.get("model_type")
    feature_names = data.get("feature_names", [])

    if model_type == "linear_model":
        # coefficients: [n_classes, n_features]
        coefs = np.array(data["coefficients"])
    elif model_type == "svm_linear":
        # weights: [n_classes, n_features]
        coefs = np.array(data["weights"])
    else:
        return None

    # Mean absolute coefficient across classes
    importance = np.abs(coefs).mean(axis=0)
    return importance, feature_names


def naive_bayes_importance(json_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """Extract feature importance from NB theta (class-conditional means).

    Importance = std of per-class means across classes. Features where
    the class means differ most are most discriminative.
    """
    with open(json_path) as f:
        data = json.load(f)

    if data.get("model_type") != "gaussian_nb":
        return None

    feature_names = data.get("feature_names", [])
    theta = np.array(data["theta"])  # [n_classes, n_features]

    # Spread of class-conditional means = discriminative power
    importance = theta.std(axis=0)
    return importance, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Group aggregation
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_by_group(
    importance: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Sum importance per feature group."""
    groups: dict[str, float] = {}
    for name, imp in zip(feature_names, importance):
        g = feature_group(name)
        groups[g] = groups.get(g, 0.0) + float(imp)
    return dict(sorted(groups.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────


def print_feature_ranking(
    importance: np.ndarray, feature_names: list[str], model_name: str, method: str,
    top_n: int = 20,
):
    """Print top features and group breakdown."""
    ranked = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    top_val = ranked[0][1] if ranked else 1.0

    print(f"\n{'=' * 64}")
    print(f"{model_name}  ({method})")
    print(f"{'=' * 64}")

    show_n = min(top_n, len(ranked))
    print(f"\nTop {show_n} features:")
    print(f"{'Rank':>4}  {'Feature':30}  {'Group':16}  {'Score':>8}")
    print("-" * 64)
    for i, (name, val) in enumerate(ranked[:show_n], 1):
        bar = "#" * int(val / top_val * 15) if top_val > 0 else ""
        print(f"{i:4}  {name:30}  {feature_group(name):16}  {val:8.4f}  {bar}")

    groups = aggregate_by_group(importance, feature_names)
    total = sum(groups.values()) or 1.0

    print(f"\nGroup importance:")
    print(f"{'Group':20}  {'Score':>8}  {'Pct':>6}  {'#Feat':>5}")
    print("-" * 48)
    for g, val in groups.items():
        n_feat = sum(1 for f in feature_names if feature_group(f) == g)
        print(f"{g:20}  {val:8.4f}  {val / total * 100:5.1f}%  {n_feat:5}")

    top10_sum = sum(v for _, v in ranked[:10])
    print(f"\nTop 10 features explain {top10_sum / total * 100:.1f}% of total importance")


def print_cross_model_summary(all_results: dict):
    """Compare group importance across models."""
    if len(all_results) < 2:
        return

    print(f"\n{'=' * 80}")
    print("Cross-Model Group Comparison")
    print(f"{'=' * 80}")

    # Collect all groups seen
    all_groups = set()
    for result in all_results.values():
        all_groups.update(result["group_importance"].keys())
    all_groups = sorted(all_groups)

    # Normalize each model's groups to percentages
    model_names = list(all_results.keys())
    header = f"{'Group':20}" + "".join(f"  {n[:14]:>14}" for n in model_names)
    print(header)
    print("-" * len(header))

    for group in all_groups:
        row = f"{group:20}"
        for name in model_names:
            gi = all_results[name]["group_importance"]
            total = sum(gi.values()) or 1.0
            val = gi.get(group, 0.0)
            row += f"  {val / total * 100:13.1f}%"
        print(row)

    # Per-feature consensus
    print_cross_model_features(all_results)


def print_cross_model_features(all_results: dict):
    """Show all features ranked by average normalized importance across models."""
    if len(all_results) < 2:
        return

    n_models = len(all_results)

    # Collect normalized importance per feature per model
    feature_scores: dict[str, list[float]] = {}
    feature_top20: dict[str, int] = {}

    for result in all_results.values():
        fi = result["feature_importance"]
        total = sum(entry["score"] for entry in fi) or 1.0

        # Top-20 set for this model
        top20_names = {entry["name"] for entry in fi[:20]}

        for entry in fi:
            name = entry["name"]
            normalized = entry["score"] / total
            feature_scores.setdefault(name, []).append(normalized)
            if name not in feature_top20:
                feature_top20[name] = 0
            if name in top20_names:
                feature_top20[name] += 1

    # Average across models and rank
    ranked = []
    for name, scores in feature_scores.items():
        avg = sum(scores) / len(scores)
        ranked.append((name, avg, feature_top20.get(name, 0)))
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 88}")
    print(f"Cross-Model Feature Consensus ({n_models} models, normalized then averaged)")
    print(f"{'=' * 88}")
    print(f"{'Rank':>4}  {'Feature':30}  {'Group':16}  {'AvgNorm':>8}  {'Top20':>5}")
    print("-" * 88)

    for i, (name, avg, top20) in enumerate(ranked, 1):
        marker = " ***" if top20 == 0 else ""
        print(f"{i:4}  {name:30}  {feature_group(name):16}  {avg:8.4f}  {top20}/{n_models}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# Analyze a single model
# ─────────────────────────────────────────────────────────────────────────────


def analyze_model(
    model_path: Path, X: np.ndarray | None, feature_names: list[str],
    top_n: int = 20,
) -> dict | None:
    """Analyze one model. Returns result dict or None."""
    name = model_path.stem

    if model_path.suffix == ".pkl":
        if not SHAP_AVAILABLE:
            print(f"  Skipping {name}: shap not installed")
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        importance, method = shap_importance(model, X, feature_names)

    elif model_path.suffix == ".json":
        # Try coefficient extraction first (linear/SVM)
        result = coefficient_importance(model_path)
        if result is not None:
            importance, feature_names = result
            method = "coefficient magnitude"
        else:
            # Try NB theta analysis
            result = naive_bayes_importance(model_path)
            if result is not None:
                importance, feature_names = result
                method = "theta spread"
            else:
                print(f"  Skipping {name}: unsupported JSON model type")
                return None
    else:
        return None

    print_feature_ranking(importance, feature_names, name, method, top_n=top_n)

    return {
        "method": method,
        "feature_importance": [
            {"name": n, "score": float(v), "group": feature_group(n)}
            for n, v in sorted(
                zip(feature_names, importance), key=lambda x: x[1], reverse=True
            )
        ],
        "group_importance": aggregate_by_group(importance, feature_names),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SHAP + coefficient analysis for V6.2 models"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models-dir", type=Path, help="Override models directory")
    parser.add_argument(
        "--max-samples", type=int, help="Override max SHAP samples (default: from config)"
    )
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--input", help="Override data input path")
    parser.add_argument(
        "--top", type=int, default=20,
        help="Show top N features per model (default: 20, use 55 for all)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.input:
        config.setdefault("data", {})["input"] = args.input

    shap_config = config.get("shap", {})
    max_samples = args.max_samples or shap_config.get("max_samples", 5000)

    models_dir = args.models_dir or Path(
        config.get("data", {}).get("output_dir", "models")
    )

    # Load training data for SHAP (tree models need it)
    print("Loading training data...")
    X_train, X_test, _y_train, _y_test, feature_names = load_and_split(config)

    # Subsample for SHAP performance
    if len(X_test) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_test), size=max_samples, replace=False)
        X_shap = X_test[indices]
    else:
        X_shap = X_test
    print(f"SHAP sample size: {len(X_shap):,}")

    # Find models: .pkl for SHAP, .json for coefficient analysis
    pkl_files = sorted(models_dir.glob("*.pkl"))
    json_files = sorted(models_dir.glob("*.json"))

    # For JSON, only analyze linear/SVM/NB (trees already covered by .pkl)
    tree_types = {"decision_tree", "random_forest", "gradient_boosted"}
    linear_json = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        if data.get("model_type") not in tree_types:
            linear_json.append(jf)

    if not pkl_files and not linear_json:
        print(f"No models found in {models_dir}")
        print("Run: cd python && python -m training.train_models")
        return 1

    print(f"\nFound {len(pkl_files)} pkl + {len(linear_json)} json models")

    all_results = {}

    # Analyze tree models via SHAP
    for pkl_path in pkl_files:
        result = analyze_model(pkl_path, X_shap, feature_names, top_n=args.top)
        if result:
            all_results[pkl_path.stem] = result

    # Analyze linear/SVM/NB via coefficients
    for json_path in linear_json:
        result = analyze_model(json_path, None, feature_names, top_n=args.top)
        if result:
            all_results[json_path.stem] = result

    # Cross-model comparison
    print_cross_model_summary(all_results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
