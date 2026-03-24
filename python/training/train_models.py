"""Unified training orchestrator for all V6.2 model types.

Trains trees (and optionally linear models, SVMs, Naive Bayes), then
auto-generates ensemble_config.yaml with validation accuracy as weights.

Usage:
    cd quant-trading-gym/python
    python -m training.train_models
    python -m training.train_models --config training/config.yaml
    python -m training.train_models --input ../data/training --output-dir ../models
"""
import argparse
from pathlib import Path

import yaml

from .common import load_and_split, load_config
from .train_trees import train_all_trees

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def generate_ensemble_config(
    trained_models: dict, config: dict, output_dir: Path
):
    """Auto-generate ensemble_config.yaml with accuracy-based weights.

    Args:
        trained_models: {rust_name: (clf, accuracy)} from all trainers
        config: Full training config dict
        output_dir: models/ directory
    """
    ensemble_config = config.get("ensemble", {})
    ensemble_name = ensemble_config.get("name", "ensemble_v6")
    mode = ensemble_config.get("mode", "auto")

    min_accuracy = ensemble_config.get("min_accuracy", 0.50)

    members = []
    excluded = []
    for rust_name, (_clf, accuracy) in trained_models.items():
        if accuracy < min_accuracy:
            excluded.append((rust_name, accuracy))
            continue
        if mode == "auto":
            weight = float(accuracy)
        else:
            weight = 1.0
        members.append({"model": rust_name, "weight": round(weight, 4)})

    # Sort by weight descending for readability
    members.sort(key=lambda m: m["weight"], reverse=True)

    yaml_data = {
        "ensemble": {
            "name": ensemble_name,
            "members": members,
        }
    }

    output_path = output_dir / "ensemble_config.yaml"
    with open(output_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nEnsemble config saved to {output_path}")
    print(f"  Name: Ensemble_{ensemble_name}, Members: {len(members)}, Mode: {mode}")
    for m in members:
        print(f"    {m['model']}: weight={m['weight']}")
    if excluded:
        print(f"  Excluded ({len(excluded)} below {min_accuracy:.0%} min_accuracy):")
        for name, acc in excluded:
            print(f"    {name}: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train all V6.2 models and generate ensemble config"
    )
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

    trained = {}
    trained.update(
        train_all_trees(
            X_train, y_train, X_test, y_test, feature_names, config, output_dir
        )
    )

    # Linear/SVM/NB: only if configured (trees outperform on this data)
    has_linear = (
        config.get("linear_models")
        or config.get("svm_models")
        or config.get("naive_bayes")
    )
    if has_linear:
        from .feature_engineering import engineer_features
        from .train_linear import train_all_linear
        from .train_naive_bayes import train_all_naive_bayes
        from .train_svm import train_all_svm

        eng_config = config.get("feature_engineering")
        X_train_eng, feat_names_eng = engineer_features(
            X_train, y_train, feature_names, eng_config
        )
        X_test_eng, _ = engineer_features(
            X_test, None, feature_names, eng_config
        )

        trained.update(
            train_all_linear(
                X_train_eng, y_train, X_test_eng, y_test, feat_names_eng, config, output_dir
            )
        )
        trained.update(
            train_all_svm(
                X_train_eng, y_train, X_test_eng, y_test, feat_names_eng, config, output_dir
            )
        )
        trained.update(
            train_all_naive_bayes(
                X_train_eng, y_train, X_test_eng, y_test, feat_names_eng, config, output_dir
            )
        )

    # Generate ensemble config if enough models trained
    if config.get("ensemble") and len(trained) >= 2:
        generate_ensemble_config(trained, config, output_dir)
    elif len(trained) < 2:
        print(f"\nSkipping ensemble config: {len(trained)} models trained (need >= 2)")

    print(f"\n=== Training Complete: {len(trained)} models ===")
    print(f"Models saved to {output_dir}/")


if __name__ == "__main__":
    main()
