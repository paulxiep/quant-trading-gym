//! V6.3 canonical feature extractor (28 SHAP-validated features).
//!
//! `CanonicalFeatures` implements `FeatureExtractor` using self-contained
//! extraction functions for 5 groups: Price (8), Technical (13),
//! Volatility (3), Fundamental (2), MomentumQuality (2).
//!
//! Dropped groups (News, Microstructure, VolumeCross) contributed <1% each
//! across all model types in V6.2 SHAP analysis.
//!
//! # Usage
//!
//! ```ignore
//! use agents::tier1::ml::CanonicalFeatures;
//!
//! let extractor = CanonicalFeatures;
//! let features = extractor.extract_market(&symbol, &ctx);
//! assert_eq!(features.len(), 28);
//! ```

use crate::StrategyContext;
use types::{
    IndicatorType, N_CANONICAL_FEATURES, Symbol, bollinger_percent_b, canonical_idx as cidx,
    log_return_from_candles, price_change_from_candles, realized_volatility,
};

/// V6.3 canonical feature extractor producing 28 SHAP-validated features.
///
/// Self-contained module — no dependency on V5/V6.1 group extractors.
/// All extraction functions write to canonical indices (0-27).
pub struct CanonicalFeatures;

impl Default for CanonicalFeatures {
    fn default() -> Self {
        Self
    }
}

impl super::FeatureExtractor for CanonicalFeatures {
    fn n_features(&self) -> usize {
        N_CANONICAL_FEATURES
    }

    fn extract_market(
        &self,
        symbol: &Symbol,
        ctx: &StrategyContext<'_>,
    ) -> crate::ml_cache::FeatureVec {
        let mut buf = [f64::NAN; N_CANONICAL_FEATURES];

        extract_price(symbol, ctx, &mut buf);
        extract_technical(symbol, ctx, &mut buf); // reads buf[0] for BB %B
        extract_volatility(symbol, ctx, &mut buf);
        extract_fundamental(symbol, ctx, &mut buf);
        extract_momentum_quality(symbol, ctx, &mut buf);

        smallvec::SmallVec::from_slice(&buf)
    }

    fn feature_names(&self) -> &[&str] {
        types::CANONICAL_FEATURE_NAMES
    }

    fn neutral_values(&self) -> &[f64] {
        &types::CANONICAL_FEATURE_NEUTRALS
    }

    fn registry(&self) -> &'static types::FeatureRegistry {
        &types::CANONICAL_REGISTRY
    }
}

// =============================================================================
// Extraction Functions (self-contained, canonical indices)
// =============================================================================

/// Extract price features: mid_price, price_change_{1,32,48,64}, log_return_{32,48,64}.
fn extract_price(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    buf[cidx::MID_PRICE] = ctx
        .mid_price(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);

    let candles = ctx.candles(symbol);

    buf[cidx::PRICE_CHANGE_1] = price_change_from_candles(candles, 1);
    buf[cidx::PRICE_CHANGE_32] = price_change_from_candles(candles, 32);
    buf[cidx::PRICE_CHANGE_48] = price_change_from_candles(candles, 48);
    buf[cidx::PRICE_CHANGE_64] = price_change_from_candles(candles, 64);

    buf[cidx::LOG_RETURN_32] = log_return_from_candles(candles, 32);
    buf[cidx::LOG_RETURN_48] = log_return_from_candles(candles, 48);
    buf[cidx::LOG_RETURN_64] = log_return_from_candles(candles, 64);
}

/// Extract technical indicators: SMA, EMA, RSI, MACD, Bollinger, ATR.
fn extract_technical(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    buf[cidx::SMA_8] = ctx
        .get_indicator(symbol, IndicatorType::Sma(8))
        .unwrap_or(f64::NAN);
    buf[cidx::SMA_16] = ctx
        .get_indicator(symbol, IndicatorType::Sma(16))
        .unwrap_or(f64::NAN);
    buf[cidx::EMA_8] = ctx
        .get_indicator(symbol, IndicatorType::Ema(8))
        .unwrap_or(f64::NAN);
    buf[cidx::EMA_16] = ctx
        .get_indicator(symbol, IndicatorType::Ema(16))
        .unwrap_or(f64::NAN);
    buf[cidx::RSI_8] = ctx
        .get_indicator(symbol, IndicatorType::Rsi(8))
        .unwrap_or(f64::NAN);

    buf[cidx::MACD_LINE] = ctx
        .get_indicator(symbol, IndicatorType::MACD_LINE_STANDARD)
        .unwrap_or(f64::NAN);
    buf[cidx::MACD_SIGNAL] = ctx
        .get_indicator(symbol, IndicatorType::MACD_SIGNAL_STANDARD)
        .unwrap_or(f64::NAN);
    buf[cidx::MACD_HISTOGRAM] = ctx
        .get_indicator(symbol, IndicatorType::MACD_HISTOGRAM_STANDARD)
        .unwrap_or(f64::NAN);

    let bb_upper = ctx
        .get_indicator(symbol, IndicatorType::BOLLINGER_UPPER_STANDARD)
        .unwrap_or(f64::NAN);
    let bb_middle = ctx
        .get_indicator(symbol, IndicatorType::BOLLINGER_MIDDLE_STANDARD)
        .unwrap_or(f64::NAN);
    let bb_lower = ctx
        .get_indicator(symbol, IndicatorType::BOLLINGER_LOWER_STANDARD)
        .unwrap_or(f64::NAN);
    buf[cidx::BB_UPPER] = bb_upper;
    buf[cidx::BB_MIDDLE] = bb_middle;
    buf[cidx::BB_LOWER] = bb_lower;

    // %B needs mid_price from buf (already written by extract_price)
    let mid_price = buf[cidx::MID_PRICE];
    buf[cidx::BB_PERCENT_B] = bollinger_percent_b(mid_price, bb_upper, bb_lower);

    buf[cidx::ATR_8] = ctx
        .get_indicator(symbol, IndicatorType::Atr(8))
        .unwrap_or(f64::NAN);
}

/// Extract volatility features: realized_vol_8, realized_vol_32, vol_ratio.
fn extract_volatility(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let candles = ctx.candles(symbol);

    let vol_8 = realized_volatility(candles, 8);
    let vol_32 = realized_volatility(candles, 32);

    buf[cidx::REALIZED_VOL_8] = vol_8;
    buf[cidx::REALIZED_VOL_32] = vol_32;

    buf[cidx::VOL_RATIO] = if vol_8.is_finite() && vol_32.is_finite() && vol_32 > 0.0 {
        vol_8 / vol_32
    } else {
        f64::NAN
    };
}

/// Extract fundamental features: fair_value_dev, price_to_fair.
fn extract_fundamental(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let mid = ctx
        .mid_price(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);
    let fair = ctx
        .fair_value(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);

    buf[cidx::FAIR_VALUE_DEV] = if mid.is_finite() && fair.is_finite() && fair > 0.0 {
        (mid - fair) / fair
    } else {
        f64::NAN
    };

    buf[cidx::PRICE_TO_FAIR] = if mid.is_finite() && fair.is_finite() && fair > 0.0 {
        mid / fair
    } else {
        f64::NAN
    };
}

/// Extract momentum quality features: trend_strength, rsi_divergence.
fn extract_momentum_quality(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let ema_8 = ctx
        .get_indicator(symbol, IndicatorType::Ema(8))
        .unwrap_or(f64::NAN);
    let ema_16 = ctx
        .get_indicator(symbol, IndicatorType::Ema(16))
        .unwrap_or(f64::NAN);
    let atr_8 = ctx
        .get_indicator(symbol, IndicatorType::Atr(8))
        .unwrap_or(f64::NAN);

    buf[cidx::TREND_STRENGTH] =
        if ema_8.is_finite() && ema_16.is_finite() && atr_8.is_finite() && atr_8 > 0.0 {
            (ema_8 - ema_16).abs() / atr_8
        } else {
            f64::NAN
        };

    let rsi = ctx
        .get_indicator(symbol, IndicatorType::Rsi(8))
        .unwrap_or(f64::NAN);
    buf[cidx::RSI_DIVERGENCE] = if rsi.is_finite() {
        rsi - 50.0
    } else {
        f64::NAN
    };
}

// =============================================================================
// Tests
// =============================================================================

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
    fn test_canonical_produces_28() {
        let (book, candles, indicators, recent_trades, events, fundamentals) = make_test_ctx();
        let market = sim_core::SingleSymbolMarket::new(&book);
        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let extractor = CanonicalFeatures;
        let features = extractor.extract_market(&"ACME".to_string(), &ctx);
        assert_eq!(features.len(), 28);
    }

    #[test]
    fn test_canonical_registry() {
        let extractor = CanonicalFeatures;
        let registry = extractor.registry();
        assert_eq!(registry.len(), 28);
        assert_eq!(registry.names().len(), 28);
        assert_eq!(registry.neutrals().len(), 28);
    }

    #[test]
    fn test_canonical_feature_names_match() {
        let extractor = CanonicalFeatures;
        assert_eq!(extractor.feature_names(), types::CANONICAL_FEATURE_NAMES);
        assert_eq!(extractor.feature_names().len(), extractor.n_features());
    }

    #[test]
    fn test_canonical_shared_features_match_full() {
        // Verify that shared features (same name) produce identical values
        // between CanonicalFeatures and FullFeatures
        let (book, candles, indicators, recent_trades, events, fundamentals) = make_test_ctx();
        let market = sim_core::SingleSymbolMarket::new(&book);
        let ctx = StrategyContext::new(
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

        let canonical = CanonicalFeatures;
        let canonical_features = canonical.extract_market(&symbol, &ctx);

        let full = super::super::FullFeatures::new();
        let full_features = full.extract_market(&symbol, &ctx);

        // For each canonical feature, find its position in full and compare
        for (c_idx, c_name) in types::CANONICAL_FEATURE_NAMES.iter().enumerate() {
            let f_idx = types::FULL_FEATURE_NAMES
                .iter()
                .position(|&n| n == *c_name)
                .unwrap_or_else(|| panic!("Feature '{}' not found in FULL", c_name));

            let c_val = canonical_features[c_idx];
            let f_val = full_features[f_idx];

            if c_val.is_nan() {
                assert!(
                    f_val.is_nan(),
                    "feature {} (canonical[{}], full[{}]): canonical=NaN, full={}",
                    c_name,
                    c_idx,
                    f_idx,
                    f_val
                );
            } else {
                assert!(
                    (c_val - f_val).abs() < 1e-12,
                    "feature {} (canonical[{}], full[{}]): canonical={}, full={}",
                    c_name,
                    c_idx,
                    f_idx,
                    c_val,
                    f_val
                );
            }
        }
    }
}
