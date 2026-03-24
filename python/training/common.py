"""Shared utilities for the V6.2 training pipeline.

Extracted from scripts/train_trees.py. Provides data loading, labeling,
feature preparation, and model export helpers used by all trainer modules.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .feature_schema import FEATURE_GROUPS, feature_group  # noqa: F401 (re-exported)
from .feature_selection import apply_feature_selection

# ─────────────────────────────────────────────────────────────────────────────
# Rust name contract
# ─────────────────────────────────────────────────────────────────────────────

# Must match the format!("{prefix}_{}", parsed.model_name) pattern in each
# Rust from_json() constructor. This is the single critical contract between
# Python training output and Rust model loading.
RUST_NAME_PREFIXES = {
    "decision_tree": "DecisionTree",
    "random_forest": "RandomForest",
    "gradient_boosted": "GradientBoosted",
    "linear_model": "LinearModel",
    "svm_linear": "SvmLinear",
    "gaussian_nb": "GaussianNB",
}


def rust_model_name(model_type: str, model_name: str) -> str:
    """Compute the Rust-side MlModel::name() for ensemble YAML references."""
    prefix = RUST_NAME_PREFIXES[model_type]
    return f"{prefix}_{model_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and labeling
# ─────────────────────────────────────────────────────────────────────────────


def load_data(base_path: str, label_config: dict) -> tuple:
    """Load market data for training.

    Loads all numbered parquet files matching {stem}_NNN_market.parquet pattern
    and concatenates them into a single dataset. Falls back to single file
    if no numbered files exist.

    Args:
        base_path: Base path for parquet files (e.g., "data/training" loads
                   all "data/training_NNN_market.parquet" files)
        label_config: Config dict with horizons, rolling_window, thresholds

    Returns:
        Tuple of (X, y, feature_names)
    """
    base = Path(base_path)
    parent = base.parent
    stem = base.stem

    # Find all numbered market parquet files: {stem}_NNN_market.parquet
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)_market\.parquet$")
    numbered_files = []

    if parent.exists():
        for f in parent.iterdir():
            match = pattern.match(f.name)
            if match:
                num = int(match.group(1))
                numbered_files.append((num, f))

    # Sort by number and load all
    numbered_files.sort(key=lambda x: x[0])
    print(f"Found {len(numbered_files)} recording files:")

    Xs = []
    ys = []
    feature_cols = None
    total_rows = 0
    for num, filepath in numbered_files:
        df = pl.read_parquet(filepath)
        df = df.with_columns(pl.lit(num).alias("run_id").cast(pl.UInt32))
        X, y, feature_cols = prepare_features(
            compute_lookahead_labels(df, label_config), label_config
        )
        Xs.append(X)
        ys.append(y)
        total_rows += len(df)
        print(f"  #{num:03d}: {len(df):,} rows from {filepath.name}")

    print(f"Combined: {total_rows:,} total rows, {len(feature_cols)} features")
    return np.concatenate(Xs), np.concatenate(ys), feature_cols


def compute_lookahead_labels(df: pl.DataFrame, label_config: dict) -> pl.DataFrame:
    """Compute lookahead price return rates for labeling.

    Args:
        df: DataFrame with tick, symbol, f_mid_price columns
        label_config: Config dict with horizons, rolling_window

    Returns:
        DataFrame with price_return_rate columns
    """
    price_horizons = label_config.get("horizons", [4, 8, 16, 32])
    rolling_window = label_config.get("rolling_window", 1)

    print(
        f"Computing lookahead labels: price_horizons={price_horizons}, "
        f"rolling_window={rolling_window}"
    )

    for n in price_horizons:
        if rolling_window <= 1:
            future_price = pl.col("f_mid_price").shift(-n).over("symbol")
        else:
            half = rolling_window // 2
            future_prices = [
                pl.col("f_mid_price").shift(-i).over("symbol")
                for i in range(n - half, n - half + rolling_window)
            ]
            future_price = sum(future_prices) / rolling_window

        df = df.with_columns(
            [
                (
                    (future_price - pl.col("f_mid_price")) / pl.col("f_mid_price")
                ).alias(f"price_return_rate_{n}")
            ]
        )

    # Discard rows without future data
    max_horizon = max(price_horizons)
    lookahead_needed = max_horizon + rolling_window - 1
    max_tick = df["tick"].max()
    rows_before = len(df)
    df = df.filter(pl.col("tick") <= max_tick - lookahead_needed)
    rows_after = len(df)
    print(
        f"Discarded {rows_before - rows_after:,} rows without lookahead data "
        f"(last {lookahead_needed} ticks)"
    )

    return df


def prepare_features(df: pl.DataFrame, label_config: dict | None = None) -> tuple:
    """Extract features and labels from dataframe.

    Label = avg price return rate across horizons.
    buy if > buy_threshold, sell if < sell_threshold, else hold.

    Returns:
        Tuple of (X, y, feature_cols)
    """
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"Using {len(feature_cols)} features")

    X = df.select(feature_cols).to_numpy()

    # Average price return rate across horizons
    price_horizons = (label_config or {}).get("horizons", [4, 8, 16, 32])
    buy_threshold = (label_config or {}).get("buy_threshold", 0.005)
    sell_threshold = (label_config or {}).get("sell_threshold", -0.005)

    return_rates = []
    for n in price_horizons:
        col = f"price_return_rate_{n}"
        if col in df.columns:
            return_rates.append(df[col].to_numpy())

    avg_return = np.nanmean(return_rates, axis=0)
    y = np.where(
        avg_return > buy_threshold, 1, np.where(avg_return < sell_threshold, -1, 0)
    )
    print(
        f"Label: avg price return rate, "
        f"buy>{buy_threshold:.4%}/tick, sell<{sell_threshold:.4%}/tick"
    )

    # Check for NaN values in features
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        nan_cols = np.isnan(X).any(axis=0)
        nan_features = [feature_cols[i] for i, has_nan in enumerate(nan_cols) if has_nan]
        print(
            f"Warning: {nan_count:,} NaN values in {len(nan_features)} features: "
            f"{nan_features}"
        )
        print("Imputing NaN with -1 (no history)...")
        X = np.nan_to_num(X, nan=-1)

    # Handle NaN in labels
    nan_labels = (
        np.isnan(y) if np.issubdtype(y.dtype, np.floating) else np.zeros(len(y), dtype=bool)
    )
    if nan_labels.any():
        print(f"Dropping {nan_labels.sum():,} rows with NaN labels")
        X = X[~nan_labels]
        y = y[~nan_labels]

    # Class distribution
    y = y.astype(int)
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(
        f"Class distribution: sell={dist.get(-1, 0)}, "
        f"hold={dist.get(0, 0)}, buy={dist.get(1, 0)}"
    )

    return X, y, feature_cols


def compute_balanced_sample_weights(y):
    """Compute balanced sample weights for classes (for GradientBoosting)."""
    from sklearn.utils.class_weight import compute_sample_weight

    return compute_sample_weight("balanced", y)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────


def load_and_split(config: dict) -> tuple:
    """Full data pipeline: load -> label -> features -> train/test split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    data_config = config.get("data", {})
    label_config = config.get("labels", {})
    input_base = data_config.get("input", "data/training")
    if input_base.endswith(".parquet"):
        input_base = input_base[:-8]

    X, y, feature_names = load_data(input_base, label_config)

    # Apply feature selection if configured
    fs_config = config.get("feature_selection")
    if fs_config:
        X, feature_names = apply_feature_selection(X, feature_names, fs_config)

    test_size = data_config.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Model export
# ─────────────────────────────────────────────────────────────────────────────


def save_model_json(result: dict, model_type: str, name: str, output_dir: Path) -> Path:
    """Save model JSON and return the output path.

    File naming convention: {name}_{model_type}.json
    (matches existing pattern: medium_decision_tree.json, small_random_forest.json)
    """
    filename = f"{name}_{model_type}.json"
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")
    return output_path


def print_report(y_test, y_pred, model_name: str):
    """Print detailed classification metrics."""
    print(f"\n=== {model_name} Classification Report ===")
    print(
        classification_report(
            y_test, y_pred, target_names=["sell (-1)", "hold (0)", "buy (+1)"]
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
