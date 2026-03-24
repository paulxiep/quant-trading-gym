"""Train tree-based models and export JSON for Rust inference.

Models: Decision Tree, Random Forest, Gradient Boosted Trees.
Refactored from scripts/train_trees.py -- imports shared logic from .common.
"""
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from .common import (
    compute_balanced_sample_weights,
    load_and_split,
    load_config,
    print_report,
    rust_model_name,
    save_model_json,
)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Tree serialization
# ─────────────────────────────────────────────────────────────────────────────


def tree_to_dict(tree, feature_names: list[str]) -> dict:
    """Convert sklearn classification tree to JSON-serializable dict."""
    tree_ = tree.tree_
    n_nodes = tree_.node_count

    nodes = []
    for i in range(n_nodes):
        is_leaf = tree_.children_left[i] == -1

        if is_leaf:
            values = tree_.value[i][0]
            total = values.sum()
            probs = (values / total).tolist() if total > 0 else [0.33, 0.34, 0.33]
            nodes.append(
                {
                    "feature": -1,
                    "threshold": 0.0,
                    "left": -1,
                    "right": -1,
                    "value": probs,
                }
            )
        else:
            nodes.append(
                {
                    "feature": int(tree_.feature[i]),
                    "threshold": float(tree_.threshold[i]),
                    "left": int(tree_.children_left[i]),
                    "right": int(tree_.children_right[i]),
                    "value": None,
                }
            )

    return {"n_nodes": n_nodes, "nodes": nodes}


def tree_to_dict_regressor(tree_regressor, feature_names: list[str]) -> dict:
    """Convert sklearn regression tree (used in GradientBoosting) to JSON dict."""
    tree_ = tree_regressor.tree_
    n_nodes = tree_.node_count

    nodes = []
    for i in range(n_nodes):
        is_leaf = tree_.children_left[i] == -1

        if is_leaf:
            leaf_value = float(tree_.value[i][0, 0])
            nodes.append(
                {
                    "feature": -1,
                    "threshold": 0.0,
                    "left": -1,
                    "right": -1,
                    "value": leaf_value,
                }
            )
        else:
            nodes.append(
                {
                    "feature": int(tree_.feature[i]),
                    "threshold": float(tree_.threshold[i]),
                    "left": int(tree_.children_left[i]),
                    "right": int(tree_.children_right[i]),
                    "value": None,
                }
            )

    return {"n_nodes": n_nodes, "nodes": nodes}


# ─────────────────────────────────────────────────────────────────────────────
# Individual trainers
# ─────────────────────────────────────────────────────────────────────────────


def train_decision_tree(
    X_train, y_train, X_test, y_test, feature_names, config, name
):
    """Train a single decision tree. Returns (clf, result_dict)."""
    print(f"\n=== Training Decision Tree: {name} ===")

    clf = DecisionTreeClassifier(
        max_depth=config.get("max_depth", 10),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tree depth: {clf.get_depth()}, nodes: {clf.tree_.node_count}")

    result = {
        "model_type": "decision_tree",
        "model_name": name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": 3,
        "classes": [-1, 0, 1],
        "tree": tree_to_dict(clf, feature_names),
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "actual_depth": clf.get_depth(),
            "accuracy": float(accuracy),
        },
    }

    return clf, result


def train_random_forest(
    X_train, y_train, X_test, y_test, feature_names, config, name
):
    """Train a random forest classifier. Returns (clf, result_dict)."""
    print(f"\n=== Training Random Forest: {name} ===")

    n_estimators = config.get("n_estimators", 100)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=config.get("max_depth", 10),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        max_features=config.get("max_features", "sqrt"),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Trees: {n_estimators}, max_depth: {config.get('max_depth', 10)}")

    importances = clf.feature_importances_.tolist()
    top_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:10]
    print("Top 10 features:")
    for feat_name, imp in top_features:
        print(f"  {feat_name}: {imp:.4f}")

    trees = [tree_to_dict(est, feature_names) for est in clf.estimators_]

    result = {
        "model_type": "random_forest",
        "model_name": name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": 3,
        "classes": [-1, 0, 1],
        "n_estimators": n_estimators,
        "trees": trees,
        "feature_importances": importances,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "accuracy": float(accuracy),
        },
    }

    return clf, result


def train_gradient_boosted(
    X_train, y_train, X_test, y_test, feature_names, config, name
):
    """Train a gradient boosted classifier. Returns (clf, result_dict)."""
    print(f"\n=== Training Gradient Boosted: {name} ===")

    n_estimators = config.get("n_estimators", 100)
    learning_rate = config.get("learning_rate", 0.1)
    max_depth = config.get("max_depth", 5)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=config.get("subsample", 1.0),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        random_state=42,
    )
    sample_weights = compute_balanced_sample_weights(y_train)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Estimators: {n_estimators}, max_depth: {max_depth}, lr: {learning_rate}")

    importances = clf.feature_importances_.tolist()
    top_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:10]
    print("Top 10 features:")
    for feat_name, imp in top_features:
        print(f"  {feat_name}: {imp:.4f}")

    n_classes = len(clf.classes_)
    stages = []
    for stage_idx in range(n_estimators):
        stage_trees = []
        for class_idx in range(n_classes):
            tree = clf.estimators_[stage_idx, class_idx]
            stage_trees.append(tree_to_dict_regressor(tree, feature_names))
        stages.append(stage_trees)

    result = {
        "model_type": "gradient_boosted",
        "model_name": name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "classes": clf.classes_.tolist(),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "init_value": (
            clf.init_.class_prior_.tolist()
            if hasattr(clf.init_, "class_prior_")
            else None
        ),
        "stages": stages,
        "feature_importances": importances,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "accuracy": float(accuracy),
        },
    }

    return clf, result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator entry point
# ─────────────────────────────────────────────────────────────────────────────

MODEL_GROUPS = [
    ("decision_trees", "decision_tree", train_decision_tree),
    ("random_forests", "random_forest", train_random_forest),
    ("gradient_boosted", "gradient_boosted", train_gradient_boosted),
]


def train_all_trees(
    X_train, y_train, X_test, y_test, feature_names, config, output_dir
):
    """Train all tree models from config. Returns {rust_name: (clf, accuracy)}."""
    trained = {}

    for config_key, model_type, trainer in MODEL_GROUPS:
        for model_config in config.get(config_key, []):
            name = model_config.get("name", "unnamed")

            clf, result = trainer(
                X_train, y_train, X_test, y_test, feature_names, model_config, name
            )
            save_model_json(result, model_type, name, output_dir)

            # Save pickle for SHAP analysis
            pkl_path = output_dir / f"{name}_{model_type}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(clf, f)

            accuracy = result["metadata"]["accuracy"]
            rname = rust_model_name(model_type, name)
            trained[rname] = (clf, accuracy)
            print_report(y_test, clf.predict(X_test), f"{name} {model_type}")

    return trained


def main():
    """Standalone entry point for tree training only."""
    import argparse

    parser = argparse.ArgumentParser(description="Train tree-based models")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", help="Override data input path")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.input:
        config.setdefault("data", {})["input"] = args.input
    if args.output_dir:
        config.setdefault("data", {})["output_dir"] = args.output_dir

    output_dir = Path(config.get("data", {}).get("output_dir", "models"))
    output_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_and_split(config)
    train_all_trees(X_train, y_train, X_test, y_test, feature_names, config, output_dir)
    print("\n=== Tree Training Complete ===")


if __name__ == "__main__":
    main()
