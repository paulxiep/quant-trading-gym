"""Train GaussianNB and export JSON for Rust GaussianNBPredictor inference.

JSON includes class_log_prior (log of class priors), theta (per-class means),
and var (per-class variances). Rust applies variance smoothing at load time.

Features are standardized for training, then the scaler is baked into
the exported theta/var so Rust inference works on raw features.
"""
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB
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


def _bake_scaler_into_nb(theta, var, scaler):
    """Fold StandardScaler into GaussianNB parameters.

    Given: trained on x_scaled = (x - mean) / std
    theta_scaled[c][i] = E[x_scaled | class=c], var_scaled[c][i] = Var[x_scaled | class=c]

    For raw features:
      theta_raw[c][i] = theta_scaled[c][i] * std[i] + mean[i]
      var_raw[c][i]   = var_scaled[c][i] * std[i]^2
    """
    mean = scaler.mean_
    std = scaler.scale_
    theta_raw = theta * std[np.newaxis, :] + mean[np.newaxis, :]
    var_raw = var * (std[np.newaxis, :] ** 2)
    return theta_raw, var_raw


def train_naive_bayes_model(
    X_train, y_train, X_test, y_test, feature_names, config, name
):
    """Train one GaussianNB model. Returns (clf, result_dict)."""
    print(f"\n=== Training GaussianNB: {name} ===")

    # Standardize features (prevents large-scale features from dominating likelihood)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = GaussianNB()
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Ensure class ordering is [-1, 0, 1]
    class_order = clf.classes_.tolist()
    assert class_order == [-1, 0, 1], f"Unexpected class order: {class_order}"

    # Bake scaler into theta/var for Rust (no scaling needed at inference)
    theta_raw, var_raw = _bake_scaler_into_nb(clf.theta_, clf.var_, scaler)

    result = {
        "model_type": "gaussian_nb",
        "model_name": name,
        "feature_names": list(feature_names),
        "n_features": X_train.shape[1],
        "n_classes": 3,
        "classes": [-1, 0, 1],
        "class_log_prior": np.log(clf.class_prior_).tolist(),
        "theta": theta_raw.tolist(),
        "var": var_raw.tolist(),
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "accuracy": float(accuracy),
            "scaled": True,
        },
    }

    return clf, result


def train_all_naive_bayes(
    X_train, y_train, X_test, y_test, feature_names, config, output_dir
):
    """Train all NB models from config. Returns {rust_name: (clf, accuracy)}."""
    trained = {}
    for model_config in config.get("naive_bayes", []):
        name = model_config.get("name", "unnamed")
        clf, result = train_naive_bayes_model(
            X_train, y_train, X_test, y_test, feature_names, model_config, name
        )
        save_model_json(result, "gaussian_nb", name, output_dir)
        pkl_path = output_dir / f"{name}_gaussian_nb.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(clf, f)
        rname = rust_model_name("gaussian_nb", name)
        trained[rname] = (clf, result["metadata"]["accuracy"])
        print_report(y_test, clf.predict(
            StandardScaler().fit(X_train).transform(X_test)
        ), f"{name} gaussian_nb")
    return trained


def main():
    """Standalone entry point for Naive Bayes training."""
    parser = argparse.ArgumentParser(description="Train GaussianNB models")
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
    train_all_naive_bayes(
        X_train, y_train, X_test, y_test, feature_names, config, output_dir
    )
    print("\n=== Naive Bayes Training Complete ===")


if __name__ == "__main__":
    main()
