"""Feature selection for V6.2 training pipeline.

Filters features based on config-driven include/exclude rules.
Supports group-level and individual feature-level selection.

Config examples:

    # Exclude mode (drop named features/groups, keep the rest):
    feature_selection:
      exclude_groups: [News, Microstructure, VolumeCross]
      exclude_features:
        - f_price_change_2
        - f_log_return_2

    # Include mode (keep only named features/groups):
    feature_selection:
      include_groups: [Fundamental, Volatility, Technical]
      include_features:
        - f_mid_price
        - f_price_change_1
"""
import numpy as np

from .feature_schema import FEATURE_GROUPS


def apply_feature_selection(
    X: np.ndarray, feature_names: list[str], fs_config: dict,
) -> tuple[np.ndarray, list[str]]:
    """Filter features based on feature_selection config.

    Supports two modes:
    - Exclude mode (default): drop features matching exclude_groups / exclude_features
    - Include mode: keep only features matching include_groups / include_features

    Gracefully ignores group/feature names not present in the data.

    Args:
        X: (n_samples, n_features) array
        feature_names: list of feature name strings
        fs_config: dict with optional keys:
            exclude_groups, exclude_features (exclude mode)
            include_groups, include_features (include mode)

    Returns:
        (X_filtered, filtered_feature_names)
    """
    available = set(feature_names)
    has_include = "include_groups" in fs_config or "include_features" in fs_config
    has_exclude = "exclude_groups" in fs_config or "exclude_features" in fs_config

    if has_include and has_exclude:
        print("  Warning: both include and exclude specified; include takes precedence")

    if has_include:
        keep = _resolve_include(fs_config, available)
        keep_indices = [i for i, name in enumerate(feature_names) if name in keep]
    else:
        exclude = _resolve_exclude(fs_config, available)
        keep_indices = [i for i, name in enumerate(feature_names) if name not in exclude]

    kept_names = [feature_names[i] for i in keep_indices]
    dropped = len(feature_names) - len(kept_names)
    print(f"Feature selection: {len(feature_names)} → {len(kept_names)} features ({dropped} excluded)")

    return X[:, keep_indices], kept_names


def _resolve_include(fs_config: dict, available: set[str]) -> set[str]:
    """Build the set of features to keep (include mode)."""
    keep = set()
    for group_name in fs_config.get("include_groups", []):
        group_features = FEATURE_GROUPS.get(group_name)
        if group_features is None:
            print(f"  Warning: unknown feature group '{group_name}', skipping")
            continue
        keep.update(group_features & available)

    for name in fs_config.get("include_features", []):
        if name in available:
            keep.add(name)
        else:
            print(f"  Warning: include feature '{name}' not in data, skipping")

    return keep


def _resolve_exclude(fs_config: dict, available: set[str]) -> set[str]:
    """Build the set of features to exclude (exclude mode)."""
    exclude = set()
    for group_name in fs_config.get("exclude_groups", []):
        group_features = FEATURE_GROUPS.get(group_name)
        if group_features is None:
            print(f"  Warning: unknown feature group '{group_name}', skipping")
            continue
        exclude.update(group_features)

    for name in fs_config.get("exclude_features", []):
        if name not in available:
            print(f"  Warning: exclude feature '{name}' not in data, skipping")
        exclude.add(name)

    return exclude
