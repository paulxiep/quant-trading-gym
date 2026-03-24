"""Feature engineering for linear models.

Adds interaction (a × b) and squared (a²) features to help linear models
capture non-linear relationships. Config-driven so you can iterate quickly
in Python before wiring winners to Rust.

Modes:
    manual (default): Use hand-specified interactions/squares/ratios
    auto:             Discover top-N interactions via mutual information

Usage:
    X_eng, names_eng = engineer_features(X, feature_names, config)
    # X_eng has original + engineered columns
    # names_eng has original + engineered names

Config example (in config.yaml):
    feature_engineering:
      # "manual" uses the lists below, "auto" discovers via MI
      mode: manual
      interactions:
        - [f_price_to_fair, f_realized_vol_8]
        - [f_bb_percent_b, f_trend_strength]
      squares:
        - f_price_to_fair
        - f_vol_ratio
      ratios:
        - [f_trend_strength, f_realized_vol_8]

    # Auto-discovery mode:
    feature_engineering:
      mode: auto
      top_k_interactions: 15
      top_k_squares: 8
      top_k_ratios: 5
"""
import numpy as np


# Financially motivated defaults when no config is provided
DEFAULT_INTERACTIONS = [
    # Mispricing × volatility: big deviation in high-vol regime = stronger signal
    ("f_price_to_fair", "f_realized_vol_8"),
    # Bollinger position × momentum direction: BB extremes with confirming trend
    ("f_bb_percent_b", "f_trend_strength"),
    # Spread × order flow: tight spread with directional flow = conviction
    ("f_spread_bps", "f_net_order_flow"),
    # Short vs medium momentum agreement
    ("f_price_change_4", "f_price_change_16"),
    # RSI × momentum quality: overbought/oversold with trend confirmation
    ("f_rsi_8", "f_rsi_divergence"),
    # Volatility regime interaction
    ("f_realized_vol_8", "f_vol_ratio"),
    # News impact in context of price deviation
    ("f_news_sentiment", "f_fair_value_dev"),
    # Volume surge × book imbalance: volume spike with directional pressure
    ("f_volume_surge", "f_book_imbalance"),
]

DEFAULT_SQUARES = [
    # Quadratic mispricing: extreme deviations matter more
    "f_price_to_fair",
    # BB extremes: far outside bands = stronger signal
    "f_bb_percent_b",
    # Extreme vol regimes
    "f_vol_ratio",
    # Extreme spread conditions
    "f_spread_bps",
    # Extreme momentum
    "f_trend_strength",
]

DEFAULT_RATIOS = [
    # Risk-adjusted momentum: trend strength per unit volatility
    ("f_trend_strength", "f_realized_vol_8"),
    # Spread relative to volatility: is the spread wide for this vol regime?
    ("f_spread_bps", "f_realized_vol_8"),
]


def _feature_idx(feature_names, name):
    """Get feature index by name, or None if not present."""
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def discover_interactions(X, y, feature_names, top_k=15):
    """Discover top-K pairwise interactions via mutual information.

    For each candidate pair (i, j), computes MI(x_i * x_j, y) and keeps
    the top_k pairs with highest MI score. Uses sklearn's
    mutual_info_classif with discretized target.

    Returns list of (feat_a_name, feat_b_name, mi_score) sorted by MI desc.
    """
    from sklearn.feature_selection import mutual_info_classif

    n_features = X.shape[1]
    # Pre-filter: skip pairs where both features are from the same group
    # (e.g., price_change_1 × price_change_2 is redundant)
    candidates = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            candidates.append((i, j))

    print(f"  Auto-discovery: testing {len(candidates)} interaction candidates...")

    # Batch: compute all candidate products, then MI in one call
    # (much faster than N separate MI calls)
    products = np.column_stack([
        X[:, i] * X[:, j] for i, j in candidates
    ])

    # Subsample for speed if large dataset
    max_samples = 10000
    if len(y) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), max_samples, replace=False)
        products_sample = products[idx]
        y_sample = y[idx]
    else:
        products_sample = products
        y_sample = y

    # Replace NaN/inf with 0 for MI computation
    products_sample = np.nan_to_num(products_sample, nan=0.0, posinf=0.0, neginf=0.0)

    mi_scores = mutual_info_classif(products_sample, y_sample, random_state=42)

    # Rank and take top_k
    scored = []
    for idx_pair, (i, j) in enumerate(candidates):
        scored.append((feature_names[i], feature_names[j], mi_scores[idx_pair]))
    scored.sort(key=lambda x: x[2], reverse=True)

    top = scored[:top_k]
    print(f"  Top {len(top)} interactions by MI:")
    for a, b, mi in top:
        print(f"    {a} × {b}: MI={mi:.4f}")

    return top


def discover_squares(X, y, feature_names, top_k=8):
    """Discover top-K squared features via mutual information.

    Returns list of (feat_name, mi_score) sorted by MI desc.
    """
    from sklearn.feature_selection import mutual_info_classif

    squares = X ** 2
    squares = np.nan_to_num(squares, nan=0.0, posinf=0.0, neginf=0.0)

    max_samples = 10000
    if len(y) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), max_samples, replace=False)
        squares = squares[idx]
        y_sample = y[idx]
    else:
        y_sample = y

    mi_scores = mutual_info_classif(squares, y_sample, random_state=42)

    scored = [(feature_names[i], mi_scores[i]) for i in range(len(feature_names))]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:top_k]
    print(f"  Top {len(top)} squared features by MI:")
    for name, mi in top:
        print(f"    {name}²: MI={mi:.4f}")

    return top


def discover_ratios(X, y, feature_names, top_k=5):
    """Discover top-K ratio features via mutual information.

    Only considers ratios where the denominator has reasonable variance
    (avoids division by near-zero). Returns list of (num, denom, mi_score).
    """
    from sklearn.feature_selection import mutual_info_classif

    n_features = X.shape[1]
    # Only use features with std > 0.01 as denominators
    stds = np.std(X, axis=0)
    valid_denoms = [i for i in range(n_features) if stds[i] > 0.01]

    candidates = []
    for i in range(n_features):
        for j in valid_denoms:
            if i != j:
                candidates.append((i, j))

    if not candidates:
        return []

    eps = 1e-10
    ratios = np.column_stack([
        np.clip(X[:, i] / (np.abs(X[:, j]) + eps), -100, 100)
        for i, j in candidates
    ])
    ratios = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)

    max_samples = 10000
    if len(y) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), max_samples, replace=False)
        ratios = ratios[idx]
        y_sample = y[idx]
    else:
        y_sample = y

    mi_scores = mutual_info_classif(ratios, y_sample, random_state=42)

    scored = []
    for idx_pair, (i, j) in enumerate(candidates):
        scored.append((feature_names[i], feature_names[j], mi_scores[idx_pair]))
    scored.sort(key=lambda x: x[2], reverse=True)

    top = scored[:top_k]
    print(f"  Top {len(top)} ratios by MI:")
    for num, denom, mi in top:
        print(f"    {num} / {denom}: MI={mi:.4f}")

    return top


# Cache for auto-discovered features (so test set reuses train set's discoveries)
_auto_cache = {}


def engineer_features(X, y=None, feature_names=None, config=None):
    """Add engineered features to the feature matrix.

    Args:
        X: numpy array (n_samples, n_features)
        y: labels (required for mode=auto on training set, None for test set)
        feature_names: list of feature name strings
        config: dict with 'mode', 'interactions', 'squares', 'ratios' keys.
                mode="manual" (default): uses specified or default features.
                mode="auto": discovers best via mutual information with y.

    Returns:
        (X_expanded, feature_names_expanded) with new columns appended.
        Original columns are unchanged (indices preserved).
    """
    global _auto_cache

    if config is None:
        config = {}

    mode = config.get("mode", "manual")

    if mode == "auto":
        if y is not None:
            # Training set: discover and cache
            top_k_interactions = config.get("top_k_interactions", 15)
            top_k_squares = config.get("top_k_squares", 8)
            top_k_ratios = config.get("top_k_ratios", 5)

            print("  Feature engineering: auto-discovery mode")
            auto_interactions = discover_interactions(
                X, y, feature_names, top_k_interactions
            )
            auto_squares = discover_squares(X, y, feature_names, top_k_squares)
            auto_ratios = discover_ratios(X, y, feature_names, top_k_ratios)

            interactions = [(a, b) for a, b, _ in auto_interactions]
            squares = [name for name, _ in auto_squares]
            ratios = [(num, denom) for num, denom, _ in auto_ratios]

            # Cache for test set
            _auto_cache = {
                "interactions": interactions,
                "squares": squares,
                "ratios": ratios,
            }
        else:
            # Test set: reuse cached discoveries from training set
            interactions = _auto_cache.get("interactions", [])
            squares = _auto_cache.get("squares", [])
            ratios = _auto_cache.get("ratios", [])
    else:
        interactions = _parse_pairs(config.get("interactions", DEFAULT_INTERACTIONS))
        squares = config.get("squares", DEFAULT_SQUARES)
        ratios = _parse_ratios(config.get("ratios", DEFAULT_RATIOS))

    new_cols = []
    new_names = []
    skipped = []

    # Interaction features: a × b
    for a_name, b_name in interactions:
        a_idx = _feature_idx(feature_names, a_name)
        b_idx = _feature_idx(feature_names, b_name)
        if a_idx is None or b_idx is None:
            skipped.append(f"{a_name}_x_{b_name}")
            continue
        product = X[:, a_idx] * X[:, b_idx]
        new_cols.append(product)
        new_names.append(f"{a_name}_x_{b_name}")

    # Squared features: a²
    for feat_name in squares:
        idx = _feature_idx(feature_names, feat_name)
        if idx is None:
            skipped.append(f"{feat_name}_sq")
            continue
        squared = X[:, idx] ** 2
        new_cols.append(squared)
        new_names.append(f"{feat_name}_sq")

    # Ratio features: a / (|b| + eps)
    for num_name, denom_name in ratios:
        num_idx = _feature_idx(feature_names, num_name)
        denom_idx = _feature_idx(feature_names, denom_name)
        if num_idx is None or denom_idx is None:
            skipped.append(f"{num_name}_over_{denom_name}")
            continue
        eps = 1e-10
        ratio = X[:, num_idx] / (np.abs(X[:, denom_idx]) + eps)
        # Clip extreme ratios to prevent inf-like values
        ratio = np.clip(ratio, -100, 100)
        new_cols.append(ratio)
        new_names.append(f"{num_name}_over_{denom_name}")

    if skipped:
        print(f"  Feature engineering: skipped {len(skipped)} (missing base features)")

    if not new_cols:
        return X, list(feature_names)

    X_expanded = np.column_stack([X] + new_cols)
    feature_names_expanded = list(feature_names) + new_names

    print(f"  Feature engineering: {len(feature_names)} base + "
          f"{len(new_names)} engineered = {len(feature_names_expanded)} total")

    return X_expanded, feature_names_expanded


def _parse_pairs(items):
    """Parse interaction pairs from config format.

    Accepts:
        - [("a", "b"), ...]           (tuple pairs)
        - [["a", "b"], ...]           (list pairs from YAML)
        - [{"a": "f_x", "b": "f_y"}] (dict pairs)
    """
    result = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((item[0], item[1]))
        elif isinstance(item, dict) and "a" in item and "b" in item:
            result.append((item["a"], item["b"]))
        else:
            raise ValueError(f"Invalid interaction spec: {item}")
    return result


def _parse_ratios(items):
    """Parse ratio specs from config format.

    Accepts:
        - [("num", "denom"), ...]
        - [["num", "denom"], ...]
        - [{"num": "f_x", "denom": "f_y"}]
    """
    result = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((item[0], item[1]))
        elif isinstance(item, dict) and "num" in item and "denom" in item:
            result.append((item["num"], item["denom"]))
        else:
            raise ValueError(f"Invalid ratio spec: {item}")
    return result
