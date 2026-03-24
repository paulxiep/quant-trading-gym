//! V5 market feature extraction (42 features).
//!
//! `MinimalFeatures` implements `FeatureExtractor` using V5 group extractors
//! from `group_extractors` (Price, Technical, News).
//!
//! # Features Extracted (42 total)
//!
//! - **Price history** (25): mid price, 12 price changes, 12 log returns at lookbacks
//! - **Technical indicators** (13): SMA(8,16), EMA(8,16), RSI(8), MACD(3), Bollinger(4), ATR(8)
//! - **News features** (4): has_active_news, sentiment, magnitude, ticks_remaining
//!
//! # NaN Handling
//!
//! `extract_features_raw()` preserves NaN — imputation is a separate pipeline step.
//! `extract_features()` applies V5-compatible NaN→-1.0 for backward compatibility.
//!
//! Note: `extract_features_raw` and `extract_features` are deprecated legacy functions.
//! `MinimalFeatures::extract_market()` now uses modular group extractors.

use crate::StrategyContext;
use types::{
    IndicatorType, LOOKBACKS, N_MARKET_FEATURES, Symbol, bollinger_percent_b, feature_idx as idx,
    log_return_from_candles, price_change_from_candles,
};

/// Extract raw market features for a symbol (NaN values preserved).
///
/// Pure extraction — no imputation. NaN indicates missing data.
/// Use [`extract_features`] for V5-compatible behavior with NaN→-1.0 imputation.
///
/// # Returns
///
/// An array of 42 features in canonical order (see [`types::features`]).
#[deprecated(note = "Use MinimalFeatures::extract_market() via FeatureExtractor trait instead")]
pub fn extract_features_raw(
    symbol: &Symbol,
    ctx: &StrategyContext<'_>,
) -> [f64; N_MARKET_FEATURES] {
    let mut features = [f64::NAN; N_MARKET_FEATURES];

    // Mid price
    let mid_price = ctx
        .mid_price(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);
    features[idx::MID_PRICE] = mid_price;

    // Get candles for price history calculations
    let candles = ctx.candles(symbol);

    // Price changes at lookback horizons (using unified computation)
    LOOKBACKS.iter().enumerate().for_each(|(i, &lookback)| {
        features[idx::PRICE_CHANGE_START + i] = price_change_from_candles(candles, lookback);
    });

    // Log returns at lookback horizons
    LOOKBACKS.iter().enumerate().for_each(|(i, &lookback)| {
        features[idx::LOG_RETURN_START + i] = log_return_from_candles(candles, lookback);
    });

    // Technical indicators from pre-computed IndicatorSnapshot
    features[idx::SMA_8] = ctx
        .get_indicator(symbol, IndicatorType::Sma(8))
        .unwrap_or(f64::NAN);
    features[idx::SMA_16] = ctx
        .get_indicator(symbol, IndicatorType::Sma(16))
        .unwrap_or(f64::NAN);
    features[idx::EMA_8] = ctx
        .get_indicator(symbol, IndicatorType::Ema(8))
        .unwrap_or(f64::NAN);
    features[idx::EMA_16] = ctx
        .get_indicator(symbol, IndicatorType::Ema(16))
        .unwrap_or(f64::NAN);
    features[idx::RSI_8] = ctx
        .get_indicator(symbol, IndicatorType::Rsi(8))
        .unwrap_or(f64::NAN);

    // MACD components (8/16/4 standard parameters)
    features[idx::MACD_LINE] = ctx
        .get_indicator(symbol, IndicatorType::MACD_LINE_STANDARD)
        .unwrap_or(f64::NAN);
    features[idx::MACD_SIGNAL] = ctx
        .get_indicator(symbol, IndicatorType::MACD_SIGNAL_STANDARD)
        .unwrap_or(f64::NAN);
    features[idx::MACD_HISTOGRAM] = ctx
        .get_indicator(symbol, IndicatorType::MACD_HISTOGRAM_STANDARD)
        .unwrap_or(f64::NAN);

    // Bollinger Bands components (12 period, 2.0 std dev)
    let bb_upper = ctx
        .get_indicator(symbol, IndicatorType::BOLLINGER_UPPER_STANDARD)
        .unwrap_or(f64::NAN);
    let bb_middle = ctx
        .get_indicator(symbol, IndicatorType::BOLLINGER_MIDDLE_STANDARD)
        .unwrap_or(f64::NAN);
    let bb_lower = ctx
        .get_indicator(symbol, IndicatorType::BOLLINGER_LOWER_STANDARD)
        .unwrap_or(f64::NAN);
    features[idx::BB_UPPER] = bb_upper;
    features[idx::BB_MIDDLE] = bb_middle;
    features[idx::BB_LOWER] = bb_lower;

    // Bollinger %B using unified computation function
    features[idx::BB_PERCENT_B] = bollinger_percent_b(mid_price, bb_upper, bb_lower);

    features[idx::ATR_8] = ctx
        .get_indicator(symbol, IndicatorType::Atr(8))
        .unwrap_or(f64::NAN);

    // News features
    let events = ctx.events_for_symbol(symbol);

    features[idx::HAS_ACTIVE_NEWS] = if events.is_empty() { 0.0 } else { 1.0 };

    // News sentiment, magnitude, ticks remaining
    let (sentiment, magnitude, ticks_remaining) = events
        .first()
        .map(|event| {
            let end_tick = event.start_tick + event.duration_ticks;
            let remaining = if ctx.tick < end_tick {
                (end_tick - ctx.tick) as f64
            } else {
                0.0
            };
            (
                event.effective_sentiment(ctx.tick),
                event.magnitude,
                remaining,
            )
        })
        .unwrap_or((0.0, 0.0, 0.0));

    features[idx::NEWS_SENTIMENT] = sentiment;
    features[idx::NEWS_MAGNITUDE] = magnitude;
    features[idx::NEWS_TICKS_REMAINING] = ticks_remaining;

    features
}

/// Extract market features with V5-compatible NaN→-1.0 imputation.
///
/// Wraps [`extract_features_raw`] and applies uniform -1.0 imputation
/// matching the V5 training convention (`nan_to_num(X, nan=-1.0)`).
#[deprecated(
    note = "Use MinimalFeatures via FeatureExtractor trait with impute_features() instead"
)]
pub fn extract_features(symbol: &Symbol, ctx: &StrategyContext<'_>) -> [f64; N_MARKET_FEATURES] {
    #[allow(deprecated)]
    let mut features = extract_features_raw(symbol, ctx);
    features.iter_mut().for_each(|f| {
        if f.is_nan() {
            *f = -1.0;
        }
    });
    features
}

/// V5-compatible feature extractor producing 42 raw market features.
///
/// Uses modular group extractors (Price, Technical, News) from `group_extractors`.
/// Imputation uses `neutral_values()` which returns -1.0 for all features
/// (V5 training convention).
pub struct MinimalFeatures;

impl super::FeatureExtractor for MinimalFeatures {
    fn n_features(&self) -> usize {
        N_MARKET_FEATURES
    }

    fn extract_market(
        &self,
        symbol: &Symbol,
        ctx: &crate::StrategyContext<'_>,
    ) -> crate::ml_cache::FeatureVec {
        let mut buf = [f64::NAN; N_MARKET_FEATURES];
        super::group_extractors::extract_price(symbol, ctx, &mut buf);
        super::group_extractors::extract_technical(symbol, ctx, &mut buf);
        super::group_extractors::extract_news(symbol, ctx, &mut buf);
        smallvec::SmallVec::from_slice(&buf)
    }

    fn feature_names(&self) -> &[&str] {
        types::MARKET_FEATURE_NAMES
    }

    fn neutral_values(&self) -> &[f64] {
        &types::MINIMAL_FEATURE_NEUTRALS
    }

    fn registry(&self) -> &'static types::FeatureRegistry {
        &types::MINIMAL_REGISTRY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier1::ml::FeatureExtractor;

    #[test]
    fn test_feature_count() {
        assert_eq!(N_MARKET_FEATURES, 42);
    }

    #[test]
    fn test_lookback_count() {
        assert_eq!(LOOKBACKS.len(), 12);
    }

    #[test]
    fn test_minimal_produces_42() {
        let book = sim_core::OrderBook::new("ACME");
        let market = sim_core::SingleSymbolMarket::new(&book);
        let candles = std::collections::HashMap::new();
        let indicators = quant::IndicatorSnapshot::new(100);
        let recent_trades = std::collections::HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

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

        let extractor = MinimalFeatures;
        let features = extractor.extract_market(&"ACME".to_string(), &ctx);
        assert_eq!(features.len(), 42);
    }

    #[test]
    fn test_minimal_modular_matches_monolithic() {
        // Verify modularized MinimalFeatures produces same output as deprecated extract_features_raw
        let book = sim_core::OrderBook::new("ACME");
        let market = sim_core::SingleSymbolMarket::new(&book);
        let candles = std::collections::HashMap::new();
        let indicators = quant::IndicatorSnapshot::new(100);
        let recent_trades = std::collections::HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

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

        // Modular (new)
        let extractor = MinimalFeatures;
        let modular = extractor.extract_market(&symbol, &ctx);

        // Monolithic (deprecated)
        #[allow(deprecated)]
        let monolithic = extract_features_raw(&symbol, &ctx);

        for i in 0..N_MARKET_FEATURES {
            let m = modular[i];
            let o = monolithic[i];
            if m.is_nan() {
                assert!(o.is_nan(), "feature {i}: modular=NaN, monolithic={o}");
            } else {
                assert!(
                    (m - o).abs() < 1e-12,
                    "feature {i}: modular={m}, monolithic={o}"
                );
            }
        }
    }
}
