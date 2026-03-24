"""V6.1 feature group definitions.

Mirrors the Rust FeatureGroup enum in crates/types/src/features.rs.
Single source of truth for Python-side feature group membership.
"""

FEATURE_GROUPS = {
    "Price": {
        "f_mid_price",
        *(f"f_price_change_{n}" for n in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),
        *(f"f_log_return_{n}" for n in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),
    },
    "Technical": {
        "f_sma_8", "f_sma_16", "f_ema_8", "f_ema_16", "f_rsi_8",
        "f_macd_line", "f_macd_signal", "f_macd_histogram",
        "f_bb_upper", "f_bb_middle", "f_bb_lower", "f_bb_percent_b", "f_atr_8",
    },
    "News": {
        "f_has_active_news", "f_news_sentiment",
        "f_news_magnitude", "f_news_ticks_remaining",
    },
    "Microstructure": {"f_spread_bps", "f_book_imbalance", "f_net_order_flow"},
    "Volatility": {"f_realized_vol_8", "f_realized_vol_32", "f_vol_ratio"},
    "Fundamental": {"f_fair_value_dev", "f_price_to_fair"},
    "MomentumQuality": {"f_trend_strength", "f_rsi_divergence"},
    "VolumeCross": {"f_volume_surge", "f_trade_intensity", "f_sentiment_price_gap"},
}

# Reverse lookup: feature_name -> group_name
_FEATURE_TO_GROUP = {}
for _group, _features in FEATURE_GROUPS.items():
    for _f in _features:
        _FEATURE_TO_GROUP[_f] = _group


def feature_group(name: str) -> str:
    """Map feature name to its V6.1 group."""
    return _FEATURE_TO_GROUP.get(name, "Unknown")
