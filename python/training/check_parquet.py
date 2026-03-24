"""Data quality check for V6.1/V6.2 training parquet files.

Validates schema, checks nulls, value ranges, class balance, and
feature group coverage against the V6.1 FullFeatures spec (55 features).

Usage:
    cd quant-trading-gym/python
    python -m training.check_parquet
    python -m training.check_parquet --input ../data/training
    python -m training.check_parquet ../data/training_001_market.parquet
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

from .analyze_shap import FEATURE_GROUPS, feature_group

# Canonical V6.1 feature names (55 total)
V6_FEATURES = [
    "f_mid_price",
    *(f"f_price_change_{n}" for n in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),
    *(f"f_log_return_{n}" for n in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),
    "f_sma_8", "f_sma_16", "f_ema_8", "f_ema_16", "f_rsi_8",
    "f_macd_line", "f_macd_signal", "f_macd_histogram",
    "f_bb_upper", "f_bb_middle", "f_bb_lower", "f_bb_percent_b", "f_atr_8",
    "f_has_active_news", "f_news_sentiment", "f_news_magnitude", "f_news_ticks_remaining",
    "f_spread_bps", "f_book_imbalance", "f_net_order_flow",
    "f_realized_vol_8", "f_realized_vol_32", "f_vol_ratio",
    "f_fair_value_dev", "f_price_to_fair",
    "f_trend_strength", "f_rsi_divergence",
    "f_volume_surge", "f_trade_intensity", "f_sentiment_price_gap",
]

V5_COUNT = 42
V6_COUNT = 55


def find_parquet_files(base_path: str) -> list[Path]:
    """Find all numbered market parquet files."""
    base = Path(base_path)

    # If a specific file was given
    if base.suffix == ".parquet" and base.exists():
        return [base]

    parent = base.parent
    stem = base.stem
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)_market\.parquet$")

    files = []
    if parent.exists():
        for f in parent.iterdir():
            if pattern.match(f.name):
                files.append(f)
    files.sort()
    return files


def check_file(path: Path, verbose: bool = True) -> dict:
    """Run quality checks on a single parquet file. Returns summary dict."""
    df = pl.read_parquet(path)
    result = {"file": path.name, "rows": len(df), "columns": len(df.columns)}

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    meta_cols = [c for c in df.columns if not c.startswith("f_")]
    result["n_features"] = len(feature_cols)
    result["meta_columns"] = meta_cols

    print(f"\n{'=' * 64}")
    print(f"{path.name}")
    print(f"{'=' * 64}")
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}  |  Features: {len(feature_cols)}")
    print(f"Meta columns: {meta_cols}")

    # Detect V5 vs V6.1
    if len(feature_cols) >= V6_COUNT:
        version = "V6.1 (full)"
    elif len(feature_cols) >= V5_COUNT:
        version = "V5 (minimal)"
    else:
        version = f"Unknown ({len(feature_cols)} features)"
    print(f"Feature set: {version}")

    # Schema check: which expected features are present/missing?
    expected = set(V6_FEATURES[:V6_COUNT] if "V6" in version else V6_FEATURES[:V5_COUNT])
    present = set(feature_cols)
    missing = expected - present
    extra = present - set(V6_FEATURES)

    if missing:
        print(f"\nMissing features ({len(missing)}):")
        for f in sorted(missing):
            print(f"  - {f}  ({feature_group(f)})")
    if extra:
        print(f"\nUnexpected features ({len(extra)}):")
        for f in sorted(extra):
            print(f"  - {f}")

    result["missing_features"] = sorted(missing)
    result["extra_features"] = sorted(extra)

    # Null check
    null_counts = {}
    for col in feature_cols:
        n_null = df[col].null_count()
        if n_null > 0:
            null_counts[col] = n_null

    if null_counts:
        print(f"\nNull counts ({len(null_counts)} features with nulls):")
        for col, count in sorted(null_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(df) * 100
            print(f"  {col:30}  {count:>8,}  ({pct:.1f}%)")
    else:
        print("\nNo nulls in feature columns")
    result["null_counts"] = null_counts

    # Value ranges
    if verbose and feature_cols:
        print(f"\nValue ranges:")
        print(f"  {'Feature':30}  {'Min':>12}  {'Max':>12}  {'Mean':>12}  {'Std':>12}")
        print(f"  {'-' * 82}")

        stats = df.select(feature_cols).describe()
        for col in feature_cols:
            col_data = df[col].drop_nulls()
            if len(col_data) == 0:
                continue
            arr = col_data.to_numpy()
            print(
                f"  {col:30}  {arr.min():12.4f}  {arr.max():12.4f}"
                f"  {arr.mean():12.4f}  {arr.std():12.4f}"
            )

    # Feature group coverage
    print(f"\nFeature group coverage:")
    print(f"  {'Group':20}  {'Present':>7}  {'Expected':>8}  {'Status'}")
    print(f"  {'-' * 50}")
    for group_name, group_features in FEATURE_GROUPS.items():
        n_present = sum(1 for f in group_features if f in present)
        n_total = len(group_features)
        status = "OK" if n_present == n_total else ("partial" if n_present > 0 else "MISSING")
        print(f"  {group_name:20}  {n_present:>7}  {n_total:>8}  {status}")

    result["version"] = version
    return result


def check_multi_file_consistency(files: list[Path]):
    """Check that multiple recording files have consistent schemas."""
    if len(files) < 2:
        return

    print(f"\n{'=' * 64}")
    print(f"Multi-file consistency ({len(files)} files)")
    print(f"{'=' * 64}")

    schemas = []
    for f in files:
        df = pl.read_parquet(f, n_rows=1)
        feature_cols = sorted(c for c in df.columns if c.startswith("f_"))
        schemas.append((f.name, feature_cols, df.shape[1]))

    base_name, base_features, _ = schemas[0]
    all_match = True
    for name, features, n_cols in schemas[1:]:
        if features != base_features:
            all_match = False
            added = set(features) - set(base_features)
            removed = set(base_features) - set(features)
            print(f"  Schema mismatch: {name} vs {base_name}")
            if added:
                print(f"    Added: {sorted(added)}")
            if removed:
                print(f"    Removed: {sorted(removed)}")

    if all_match:
        print(f"  All {len(files)} files have identical feature schemas ({len(base_features)} features)")

    # Row count summary
    print(f"\n  {'File':40}  {'Rows':>10}")
    print(f"  {'-' * 55}")
    total = 0
    for f in files:
        df = pl.read_parquet(f)
        print(f"  {f.name:40}  {len(df):>10,}")
        total += len(df)
    print(f"  {'TOTAL':40}  {total:>10,}")


def main():
    parser = argparse.ArgumentParser(description="Check training parquet data quality")
    parser.add_argument(
        "file", nargs="?", help="Specific parquet file to check"
    )
    parser.add_argument(
        "--input", default="data/training",
        help="Base path for numbered files (default: data/training)"
    )
    parser.add_argument("--brief", action="store_true", help="Skip value ranges")
    args = parser.parse_args()

    if args.file:
        files = find_parquet_files(args.file)
    else:
        files = find_parquet_files(args.input)

    if not files:
        path = args.file or args.input
        print(f"No parquet files found at {path}")
        print("Generate data with:")
        print("  cargo run --release -- --headless-record --full-features --ticks 100000")
        return 1

    print(f"Found {len(files)} parquet file(s)")

    for f in files:
        check_file(f, verbose=not args.brief)

    if len(files) > 1:
        check_multi_file_consistency(files)

    return 0


if __name__ == "__main__":
    sys.exit(main())
