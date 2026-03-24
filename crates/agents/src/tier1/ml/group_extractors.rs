//! Group-level feature extraction functions (V6.1).
//!
//! Each feature group has its own extraction function that writes into a `&mut [f64]`
//! buffer at the correct indices. Functions are free (not trait objects) — simpler,
//! inlineable, and equally modular.
//!
//! # Groups
//!
//! - V5 base (refactored from `extract_features_raw`):
//!   - [`extract_price`] — mid price, 12 price changes, 12 log returns (indices 0-24)
//!   - [`extract_technical`] — SMA, EMA, RSI, MACD, Bollinger, ATR (indices 25-37)
//!   - [`extract_news`] — has_active_news, sentiment, magnitude, ticks_remaining (indices 38-41)
//!
//! - V6.1 new:
//!   - [`extract_microstructure`] — spread, book imbalance, net order flow (indices 42-44)
//!   - [`extract_volatility`] — realized vol 8/32, vol ratio (indices 45-47)
//!   - [`extract_fundamental`] — fair value deviation, price-to-fair (indices 48-49)
//!   - [`extract_momentum_quality`] — trend strength, RSI divergence (indices 50-51)
//!   - [`extract_volume_cross`] — volume surge, trade intensity, sentiment-price gap (indices 52-54)
//!
//! # Call Order
//!
//! `extract_volume_cross` reads `buf[extended_idx::FAIR_VALUE_DEV]` written by
//! `extract_fundamental`. Call `extract_fundamental` before `extract_volume_cross`.

use crate::StrategyContext;
use types::{
    IndicatorType, LOOKBACKS, Symbol, TRADE_INTENSITY_BASELINE, bollinger_percent_b, extended_idx,
    features::idx, log_return_from_candles, price_change_from_candles, realized_volatility,
    spread_bps,
};

// =============================================================================
// V5 Base Groups (refactored from extract_features_raw)
// =============================================================================

/// Extract price features: mid price, 12 price changes, 12 log returns (indices 0-24).
pub fn extract_price(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let mid_price = ctx
        .mid_price(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);
    buf[idx::MID_PRICE] = mid_price;

    let candles = ctx.candles(symbol);

    LOOKBACKS.iter().enumerate().for_each(|(i, &lookback)| {
        buf[idx::PRICE_CHANGE_START + i] = price_change_from_candles(candles, lookback);
    });

    LOOKBACKS.iter().enumerate().for_each(|(i, &lookback)| {
        buf[idx::LOG_RETURN_START + i] = log_return_from_candles(candles, lookback);
    });
}

/// Extract technical indicator features: SMA, EMA, RSI, MACD, Bollinger, ATR (indices 25-37).
pub fn extract_technical(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    buf[idx::SMA_8] = ctx
        .get_indicator(symbol, IndicatorType::Sma(8))
        .unwrap_or(f64::NAN);
    buf[idx::SMA_16] = ctx
        .get_indicator(symbol, IndicatorType::Sma(16))
        .unwrap_or(f64::NAN);
    buf[idx::EMA_8] = ctx
        .get_indicator(symbol, IndicatorType::Ema(8))
        .unwrap_or(f64::NAN);
    buf[idx::EMA_16] = ctx
        .get_indicator(symbol, IndicatorType::Ema(16))
        .unwrap_or(f64::NAN);
    buf[idx::RSI_8] = ctx
        .get_indicator(symbol, IndicatorType::Rsi(8))
        .unwrap_or(f64::NAN);

    buf[idx::MACD_LINE] = ctx
        .get_indicator(symbol, IndicatorType::MACD_LINE_STANDARD)
        .unwrap_or(f64::NAN);
    buf[idx::MACD_SIGNAL] = ctx
        .get_indicator(symbol, IndicatorType::MACD_SIGNAL_STANDARD)
        .unwrap_or(f64::NAN);
    buf[idx::MACD_HISTOGRAM] = ctx
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
    buf[idx::BB_UPPER] = bb_upper;
    buf[idx::BB_MIDDLE] = bb_middle;
    buf[idx::BB_LOWER] = bb_lower;

    // %B needs mid_price from buf — read it back (already written by extract_price)
    let mid_price = buf[idx::MID_PRICE];
    buf[idx::BB_PERCENT_B] = bollinger_percent_b(mid_price, bb_upper, bb_lower);

    buf[idx::ATR_8] = ctx
        .get_indicator(symbol, IndicatorType::Atr(8))
        .unwrap_or(f64::NAN);
}

/// Extract news features: has_active_news, sentiment, magnitude, ticks_remaining (indices 38-41).
pub fn extract_news(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let events = ctx.events_for_symbol(symbol);

    buf[idx::HAS_ACTIVE_NEWS] = if events.is_empty() { 0.0 } else { 1.0 };

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

    buf[idx::NEWS_SENTIMENT] = sentiment;
    buf[idx::NEWS_MAGNITUDE] = magnitude;
    buf[idx::NEWS_TICKS_REMAINING] = ticks_remaining;
}

// =============================================================================
// V6.1 New Groups
// =============================================================================

/// Extract microstructure features: spread_bps, book_imbalance, net_order_flow (indices 42-44).
pub fn extract_microstructure(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    // Spread in basis points
    let bid = ctx
        .best_bid(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);
    let ask = ctx
        .best_ask(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);
    let mid = ctx
        .mid_price(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);
    buf[extended_idx::SPREAD_BPS] = spread_bps(bid, ask, mid);

    // Book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    let bid_vol = ctx.total_bid_volume(symbol).0 as f64;
    let ask_vol = ctx.total_ask_volume(symbol).0 as f64;
    let total_vol = bid_vol + ask_vol;
    buf[extended_idx::BOOK_IMBALANCE] = if total_vol > 0.0 {
        (bid_vol - ask_vol) / total_vol
    } else {
        f64::NAN
    };

    // Net order flow via tick rule: classify each trade as buyer/seller-initiated
    // by comparing trade price to previous trade price.
    let trades = ctx.recent_trades(symbol);
    if trades.len() < 2 {
        buf[extended_idx::NET_ORDER_FLOW] = f64::NAN;
    } else {
        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;
        // recent_trades are most-recent-first; iterate in pairs
        for pair in trades.windows(2) {
            let current = &pair[0];
            let previous = &pair[1];
            let qty = current.quantity.0 as f64;
            if current.price > previous.price {
                buy_vol += qty;
            } else if current.price < previous.price {
                sell_vol += qty;
            }
            // equal price: omitted (no classification)
        }
        let total = buy_vol + sell_vol;
        buf[extended_idx::NET_ORDER_FLOW] = if total > 0.0 {
            (buy_vol - sell_vol) / total
        } else {
            0.0
        };
    }
}

/// Extract volatility features: realized_vol_8, realized_vol_32, vol_ratio (indices 45-47).
pub fn extract_volatility(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let candles = ctx.candles(symbol);

    let vol_8 = realized_volatility(candles, 8);
    let vol_32 = realized_volatility(candles, 32);

    buf[extended_idx::REALIZED_VOL_8] = vol_8;
    buf[extended_idx::REALIZED_VOL_32] = vol_32;

    // Vol ratio: vol_8 / vol_32. >1 = expanding, <1 = contracting.
    buf[extended_idx::VOL_RATIO] = if vol_8.is_finite() && vol_32.is_finite() && vol_32 > 0.0 {
        vol_8 / vol_32
    } else {
        f64::NAN
    };
}

/// Extract fundamental features: fair_value_dev, price_to_fair (indices 48-49).
pub fn extract_fundamental(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let mid = ctx
        .mid_price(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);
    let fair = ctx
        .fair_value(symbol)
        .map(|p| p.to_float())
        .unwrap_or(f64::NAN);

    buf[extended_idx::FAIR_VALUE_DEV] = if mid.is_finite() && fair.is_finite() && fair > 0.0 {
        (mid - fair) / fair
    } else {
        f64::NAN
    };

    buf[extended_idx::PRICE_TO_FAIR] = if mid.is_finite() && fair.is_finite() && fair > 0.0 {
        mid / fair
    } else {
        f64::NAN
    };
}

/// Extract momentum quality features: trend_strength, rsi_divergence (indices 50-51).
pub fn extract_momentum_quality(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    // Trend strength: abs(ema_8 - ema_16) / atr_8
    let ema_8 = ctx
        .get_indicator(symbol, IndicatorType::Ema(8))
        .unwrap_or(f64::NAN);
    let ema_16 = ctx
        .get_indicator(symbol, IndicatorType::Ema(16))
        .unwrap_or(f64::NAN);
    let atr_8 = ctx
        .get_indicator(symbol, IndicatorType::Atr(8))
        .unwrap_or(f64::NAN);

    buf[extended_idx::TREND_STRENGTH] =
        if ema_8.is_finite() && ema_16.is_finite() && atr_8.is_finite() && atr_8 > 0.0 {
            (ema_8 - ema_16).abs() / atr_8
        } else {
            f64::NAN
        };

    // RSI divergence: rsi_8 - 50.0
    let rsi = ctx
        .get_indicator(symbol, IndicatorType::Rsi(8))
        .unwrap_or(f64::NAN);
    buf[extended_idx::RSI_DIVERGENCE] = if rsi.is_finite() {
        rsi - 50.0
    } else {
        f64::NAN
    };
}

/// Extract volume/cross features: volume_surge, trade_intensity, sentiment_price_gap (indices 52-54).
///
/// **Dependency**: Reads `buf[extended_idx::FAIR_VALUE_DEV]`. Call `extract_fundamental` first.
pub fn extract_volume_cross(symbol: &Symbol, ctx: &StrategyContext<'_>, buf: &mut [f64]) {
    let candles = ctx.candles(symbol);

    // Volume surge: latest_volume / avg_volume_8
    if candles.is_empty() {
        buf[extended_idx::VOLUME_SURGE] = f64::NAN;
    } else {
        let latest_vol = candles.last().unwrap().volume.0 as f64;
        let n = candles.len().min(8);
        let avg_vol: f64 = candles[candles.len().saturating_sub(8)..]
            .iter()
            .map(|c| c.volume.0 as f64)
            .sum::<f64>()
            / n as f64;
        buf[extended_idx::VOLUME_SURGE] = if avg_vol > 0.0 {
            latest_vol / avg_vol
        } else {
            f64::NAN
        };
    }

    // Trade intensity: n_recent_trades / baseline
    let trades = ctx.recent_trades(symbol);
    buf[extended_idx::TRADE_INTENSITY] = trades.len() as f64 / TRADE_INTENSITY_BASELINE;

    // Sentiment-price gap: symbol_sentiment * fair_value_dev
    let sentiment = ctx.symbol_sentiment(symbol);
    let fair_value_dev = buf[extended_idx::FAIR_VALUE_DEV];
    buf[extended_idx::SENTIMENT_PRICE_GAP] = if fair_value_dev.is_finite() && sentiment.is_finite()
    {
        sentiment * fair_value_dev
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
    use types::{N_FULL_FEATURES, N_MARKET_FEATURES};

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
    fn test_extract_price_writes_correct_indices() {
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

        let mut buf = [f64::NAN; N_FULL_FEATURES];
        extract_price(&"ACME".to_string(), &ctx, &mut buf);

        // Mid price should be NaN (no orders in empty book)
        assert!(buf[idx::MID_PRICE].is_nan());
        // Price changes should be NaN (no candles)
        assert!(buf[idx::PRICE_CHANGE_START].is_nan());
        // Log returns should be NaN (no candles)
        assert!(buf[idx::LOG_RETURN_START].is_nan());
        // Indices beyond price group should still be NaN (untouched)
        assert!(buf[idx::SMA_8].is_nan());
    }

    #[test]
    fn test_extract_news_no_events() {
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

        let mut buf = [f64::NAN; N_FULL_FEATURES];
        extract_news(&"ACME".to_string(), &ctx, &mut buf);

        assert_eq!(buf[idx::HAS_ACTIVE_NEWS], 0.0);
        assert_eq!(buf[idx::NEWS_SENTIMENT], 0.0);
        assert_eq!(buf[idx::NEWS_MAGNITUDE], 0.0);
        assert_eq!(buf[idx::NEWS_TICKS_REMAINING], 0.0);
    }

    #[test]
    fn test_v5_groups_match_extract_features_raw() {
        // Verify that the 3 V5 group functions produce identical output
        // to the monolithic extract_features_raw()
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

        // MinimalFeatures extraction (uses group extractors internally)
        let minimal = super::super::MinimalFeatures;
        let expected = minimal.extract_market(&symbol, &ctx);

        // Group extraction into full-size buffer
        let mut actual = [f64::NAN; N_FULL_FEATURES];
        extract_price(&symbol, &ctx, &mut actual);
        extract_technical(&symbol, &ctx, &mut actual);
        extract_news(&symbol, &ctx, &mut actual);

        // Compare first 42 features
        for i in 0..N_MARKET_FEATURES {
            let e = expected[i];
            let a = actual[i];
            if e.is_nan() {
                assert!(a.is_nan(), "feature {i}: expected NaN, got {a}");
            } else {
                assert!((e - a).abs() < 1e-12, "feature {i}: expected {e}, got {a}");
            }
        }
    }

    #[test]
    fn test_extract_microstructure_empty_book() {
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

        let mut buf = [f64::NAN; N_FULL_FEATURES];
        extract_microstructure(&"ACME".to_string(), &ctx, &mut buf);

        // Empty book: all microstructure features should be NaN
        assert!(buf[extended_idx::SPREAD_BPS].is_nan());
        assert!(buf[extended_idx::BOOK_IMBALANCE].is_nan());
        assert!(buf[extended_idx::NET_ORDER_FLOW].is_nan());
    }

    #[test]
    fn test_extract_volatility_no_candles() {
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

        let mut buf = [f64::NAN; N_FULL_FEATURES];
        extract_volatility(&"ACME".to_string(), &ctx, &mut buf);

        assert!(buf[extended_idx::REALIZED_VOL_8].is_nan());
        assert!(buf[extended_idx::REALIZED_VOL_32].is_nan());
        assert!(buf[extended_idx::VOL_RATIO].is_nan());
    }

    #[test]
    fn test_extract_fundamental_no_data() {
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

        let mut buf = [f64::NAN; N_FULL_FEATURES];
        extract_fundamental(&"ACME".to_string(), &ctx, &mut buf);

        assert!(buf[extended_idx::FAIR_VALUE_DEV].is_nan());
        assert!(buf[extended_idx::PRICE_TO_FAIR].is_nan());
    }

    #[test]
    fn test_extract_volume_cross_empty() {
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

        let mut buf = [f64::NAN; N_FULL_FEATURES];
        // Must call fundamental first (dependency)
        extract_fundamental(&"ACME".to_string(), &ctx, &mut buf);
        extract_volume_cross(&"ACME".to_string(), &ctx, &mut buf);

        assert!(buf[extended_idx::VOLUME_SURGE].is_nan()); // no candles
        assert_eq!(buf[extended_idx::TRADE_INTENSITY], 0.0); // 0 trades / baseline
        assert!(buf[extended_idx::SENTIMENT_PRICE_GAP].is_nan()); // fair_value_dev is NaN
    }
}
