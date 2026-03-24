"""Train LogisticRegression and export JSON for Rust LinearPredictor inference.

JSON uses "coefficients"/"intercepts" keys which map via serde alias to
the Rust LinearPredictor's weights/biases fields.

Features are standardized for training, then the scaler is baked into
the exported weights so Rust inference works on raw features.
"""
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .common import (
    load_and_split,
    load_config,
    print_report,
    rust_model_name,
    save_model_json,
)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _bake_scaler_into_linear(coef, intercept, scaler):
    """Fold StandardScaler into linear model weights.

    Given: logits = W @ x_scaled + b, where x_scaled = (x - mean) / std
    Returns: (W_eff, b_eff) such that logits = W_eff @ x_raw + b_eff
    """
    mean = scaler.mean_
    std = scaler.scale_
    # W_eff[c][i] = W[c][i] / std[i]
    coef_eff = coef / std[np.newaxis, :]
    # b_eff[c] = b[c] - sum_i(W[c][i] * mean[i] / std[i])
    intercept_eff = intercept - (coef_eff @ mean)
    return coef_eff, intercept_eff


def train_linear_model(X_train, y_train, X_test, y_test, feature_names, config, name):
    """Train one LogisticRegression model. Returns (clf, result_dict)."""
    print(f"\n=== Training LogisticRegression: {name} ===")

    # Standardize features (critical for linear models)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        solver=config.get("solver", "lbfgs"),
        max_iter=config.get("max_iter", 5000),
        C=config.get("C", 1.0),
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Ensure class ordering is [-1, 0, 1]
    class_order = clf.classes_.tolist()
    assert class_order == [-1, 0, 1], f"Unexpected class order: {class_order}"

    # Bake scaler into weights for Rust (no scaling needed at inference)
    coef_eff, intercept_eff = _bake_scaler_into_linear(
        clf.coef_, clf.intercept_, scaler
    )

    result = {
        "model_type": "linear_model",
        "model_name": name,
        "feature_names": list(feature_names),
        "n_features": X_train.shape[1],
        "n_classes": 3,
        "classes": [-1, 0, 1],
        "coefficients": coef_eff.tolist(),
        "intercepts": intercept_eff.tolist(),
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "accuracy": float(accuracy),
            "scaled": True,
        },
    }

    return clf, result


def train_all_linear(X_train, y_train, X_test, y_test, feature_names, config, output_dir):
    """Train all linear models from config. Returns {rust_name: (clf, accuracy)}."""
    trained = {}
    for model_config in config.get("linear_models", []):
        name = model_config.get("name", "unnamed")
        clf, result = train_linear_model(
            X_train, y_train, X_test, y_test, feature_names, model_config, name
        )
        save_model_json(result, "linear_model", name, output_dir)
        pkl_path = output_dir / f"{name}_linear_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(clf, f)
        rname = rust_model_name("linear_model", name)
        trained[rname] = (clf, result["metadata"]["accuracy"])
        print_report(y_test, clf.predict(
            StandardScaler().fit(X_train).transform(X_test)
        ), f"{name} linear_model")
    return trained


def main():
    """Standalone entry point for linear model training."""
    parser = argparse.ArgumentParser(description="Train linear models")
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
    train_all_linear(X_train, y_train, X_test, y_test, feature_names, config, output_dir)
    print("\n=== Linear Model Training Complete ===")


if __name__ == "__main__":
    main()
