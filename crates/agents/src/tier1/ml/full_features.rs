//! V6.1 full feature extractor (55 features).
//!
//! `FullFeatures` implements `FeatureExtractor` using modular group extraction
//! functions. Supports group-level ablation for post-V6.2 feature importance testing.
//!
//! # Usage
//!
//! ```ignore
//! use agents::tier1::ml::FullFeatures;
//!
//! let extractor = FullFeatures::new();
//! let features = extractor.extract_market(&symbol, &ctx);
//! assert_eq!(features.len(), 55);
//!
//! // Disable a group for ablation testing
//! let mut extractor = FullFeatures::new();
//! extractor.disable_group(FeatureGroup::Microstructure);
//! // Microstructure indices will be NaN → imputed to neutral values
//! ```

use std::collections::HashSet;

use types::{FeatureGroup, N_FULL_FEATURES};

use super::group_extractors;

/// V6.1 full feature extractor producing 55 market features.
///
/// Uses group extraction functions for modular computation.
/// Supports ablation testing by disabling specific feature groups.
pub struct FullFeatures {
    disabled_groups: HashSet<FeatureGroup>,
}

impl FullFeatures {
    /// Create a new extractor with all groups enabled.
    pub fn new() -> Self {
        Self {
            disabled_groups: HashSet::new(),
        }
    }

    /// Disable a feature group (for ablation testing).
    ///
    /// Disabled groups leave NaN in the buffer; imputation fills neutral values.
    pub fn disable_group(&mut self, group: FeatureGroup) {
        self.disabled_groups.insert(group);
    }

    /// Re-enable a previously disabled feature group.
    pub fn enable_group(&mut self, group: FeatureGroup) {
        self.disabled_groups.remove(&group);
    }

    /// Check if a group is enabled.
    fn group_enabled(&self, group: FeatureGroup) -> bool {
        !self.disabled_groups.contains(&group)
    }
}

impl Default for FullFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl super::FeatureExtractor for FullFeatures {
    fn n_features(&self) -> usize {
        N_FULL_FEATURES
    }

    fn extract_market(
        &self,
        symbol: &types::Symbol,
        ctx: &crate::StrategyContext<'_>,
    ) -> crate::ml_cache::FeatureVec {
        let mut buf = [f64::NAN; N_FULL_FEATURES];

        // V5 base groups
        if self.group_enabled(FeatureGroup::Price) {
            group_extractors::extract_price(symbol, ctx, &mut buf);
        }
        if self.group_enabled(FeatureGroup::TechnicalIndicator) {
            group_extractors::extract_technical(symbol, ctx, &mut buf);
        }
        if self.group_enabled(FeatureGroup::News) {
            group_extractors::extract_news(symbol, ctx, &mut buf);
        }

        // V6.1 new groups
        if self.group_enabled(FeatureGroup::Microstructure) {
            group_extractors::extract_microstructure(symbol, ctx, &mut buf);
        }
        if self.group_enabled(FeatureGroup::Volatility) {
            group_extractors::extract_volatility(symbol, ctx, &mut buf);
        }
        // Fundamental must run before VolumeCross (dependency on fair_value_dev)
        if self.group_enabled(FeatureGroup::Fundamental) {
            group_extractors::extract_fundamental(symbol, ctx, &mut buf);
        }
        if self.group_enabled(FeatureGroup::MomentumQuality) {
            group_extractors::extract_momentum_quality(symbol, ctx, &mut buf);
        }
        if self.group_enabled(FeatureGroup::VolumeCross) {
            group_extractors::extract_volume_cross(symbol, ctx, &mut buf);
        }

        smallvec::SmallVec::from_slice(&buf)
    }

    fn feature_names(&self) -> &[&str] {
        types::FULL_FEATURE_NAMES
    }

    fn neutral_values(&self) -> &[f64] {
        &types::FULL_FEATURE_NEUTRALS
    }

    fn registry(&self) -> &'static types::FeatureRegistry {
        &types::FULL_REGISTRY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier1::ml::FeatureExtractor;

    fn make_test_ctx() -> (
        sim_core::OrderBook,
        std::collections::HashMap<String, Vec<types::Candle>>,
        quant::IndicatorSnapshot,
        std::collections::HashMap<String, Vec<types::Trade>>,
        Vec<news::NewsEvent>,
        news::SymbolFundamentals,
    ) {
        let book = sim_core::OrderBook::new("ACME");
        let candles = std::collections::HashMap::new();
        let indicators = quant::IndicatorSnapshot::new(100);
        let recent_trades = std::collections::HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();
        (
            book,
            candles,
            indicators,
            recent_trades,
            events,
            fundamentals,
        )
    }

    #[test]
    fn test_full_features_produces_55() {
        let (book, candles, indicators, recent_trades, events, fundamentals) = make_test_ctx();
        let market = sim_core::SingleSymbolMarket::new(&book);
        let ctx = crate::StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let extractor = FullFeatures::new();
        let features = extractor.extract_market(&"ACME".to_string(), &ctx);
        assert_eq!(features.len(), 55);
    }

    #[test]
    fn test_full_features_registry() {
        let extractor = FullFeatures::new();
        let registry = extractor.registry();
        assert_eq!(registry.len(), 55);
        assert_eq!(registry.names().len(), 55);
        assert_eq!(registry.neutrals().len(), 55);
    }

    #[test]
    fn test_full_features_ablation() {
        let (book, candles, indicators, recent_trades, events, fundamentals) = make_test_ctx();
        let market = sim_core::SingleSymbolMarket::new(&book);
        let ctx = crate::StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let mut extractor = FullFeatures::new();
        extractor.disable_group(FeatureGroup::News);

        let features = extractor.extract_market(&"ACME".to_string(), &ctx);

        // News features (38-41) should all be NaN when group is disabled
        assert!(features[38].is_nan());
        assert!(features[39].is_nan());
        assert!(features[40].is_nan());
        assert!(features[41].is_nan());
    }

    #[test]
    fn test_full_features_v5_prefix_matches_minimal() {
        let (book, candles, indicators, recent_trades, events, fundamentals) = make_test_ctx();
        let market = sim_core::SingleSymbolMarket::new(&book);
        let ctx = crate::StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let symbol = "ACME".to_string();

        // Full extractor
        let full = FullFeatures::new();
        let full_features = full.extract_market(&symbol, &ctx);

        // Minimal extractor
        let minimal = super::super::MinimalFeatures;
        let minimal_features = minimal.extract_market(&symbol, &ctx);

        // First 42 features should match exactly
        for i in 0..42 {
            let f = full_features[i];
            let m = minimal_features[i];
            if f.is_nan() {
                assert!(m.is_nan(), "feature {i}: full=NaN, minimal={m}");
            } else {
                assert!((f - m).abs() < 1e-12, "feature {i}: full={f}, minimal={m}");
            }
        }
    }
}
