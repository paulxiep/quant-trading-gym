//! Market feature schema constants for Parquet recording.
//!
//! Pre-V6 refactor (section F): `MarketFeatures::extract()` has been removed.
//! Feature extraction is now handled by `FeatureExtractor` in the agents crate,
//! with the runner extracting features once and passing them to recording hooks
//! via `HookContext.features`. This eliminates the dual extraction path that
//! guaranteed training-serving skew.
//!
//! This module retains the `MarketFeatures` struct and schema constants for
//! Parquet column naming and backward compatibility.

use types::N_MARKET_FEATURES;

// ─────────────────────────────────────────────────────────────────────────────
// Market Features - Schema constants
// ─────────────────────────────────────────────────────────────────────────────

/// Market-level features for Parquet recording.
///
/// Feature count is determined by the extractor (42 for V5/MinimalFeatures,
/// 55+ for V6/FullFeatures).
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Feature values in order of extractor's `feature_names()`.
    pub features: Vec<f64>,
    /// Mid price (convenience - also features[0]).
    pub mid_price: f64,
}

impl MarketFeatures {
    /// Default number of market-level features (V5 MinimalFeatures).
    pub const DEFAULT_COUNT: usize = N_MARKET_FEATURES;

    /// Default feature names (V5 MinimalFeatures).
    pub fn default_feature_names() -> &'static [&'static str] {
        types::MARKET_FEATURE_NAMES
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_feature_count() {
        assert_eq!(
            types::MARKET_FEATURE_NAMES.len(),
            MarketFeatures::DEFAULT_COUNT
        );
        assert_eq!(MarketFeatures::DEFAULT_COUNT, 42);
    }

    #[test]
    fn test_feature_name_prefix() {
        for name in types::MARKET_FEATURE_NAMES {
            assert!(
                name.starts_with("f_"),
                "Market feature name '{}' should start with 'f_'",
                name
            );
        }
    }
}
