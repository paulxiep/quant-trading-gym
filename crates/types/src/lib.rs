//! Core types for the quant-trading-gym simulation.
//!
//! This crate provides all the fundamental types used across the trading simulation:
//! - **Identifiers**: Order, Agent, Trade, and Fill IDs
//! - **Monetary types**: Price, Cash, and Quantity with fixed-point arithmetic
//! - **Order types**: Order, OrderSide, OrderType, OrderStatus
//! - **Trade types**: Trade, Fill, and slippage metrics
//! - **Market data**: Candle, BookLevel, BookSnapshot
//! - **Indicators**: Technical analysis types
//! - **Configuration**: Symbol and short-selling settings
//!
//! # Module Organization
//!
//! The types are organized into focused modules:
//! - [`ids`] - Identifier types and constants
//! - [`money`] - Fixed-point monetary types
//! - [`order`] - Order-related types
//! - [`trade`] - Trade, fill, and slippage types
//! - [`market_data`] - OHLCV candles and order book snapshots
//! - [`indicators`] - Technical indicator types
//! - [`config`] - Simulation configuration types

// =============================================================================
// Module Declarations
// =============================================================================

pub mod config;
pub mod features;
pub mod ids;
pub mod indicators;
pub mod market_data;
pub mod money;
pub mod order;
pub mod trade;

// =============================================================================
// Re-exports for Convenience
// =============================================================================

// IDs and constants
pub use ids::{AgentId, FillId, OrderId, PRICE_SCALE, Symbol, Tick, Timestamp, TradeId};

// Monetary types
pub use money::{Cash, Price, Quantity};

// Order types
pub use order::{Order, OrderSide, OrderStatus, OrderType};

// Trade and execution types
pub use trade::{Fill, SlippageConfig, SlippageMetrics, Trade};

// Market data types
pub use market_data::{BookLevel, BookSnapshot, Candle};

// Indicator types
pub use indicators::{BollingerOutput, IndicatorType, IndicatorValue, MacdOutput};

// Configuration types
pub use config::{RiskViolation, Sector, ShortSellingConfig, SymbolConfig};

// Feature extraction types (V5.5.2 - unified training/inference)
pub use features::{
    LOOKBACKS, MARKET_FEATURE_NAMES, MINIMAL_FEATURE_NEUTRALS, N_LOOKBACKS, N_MARKET_FEATURES,
    bollinger_percent_b, idx as feature_idx, log_return, log_return_from_candles,
    price_change_from_candles, price_change_pct, required_indicators,
};

// V6.1 feature registry types
pub use features::{
    FULL_DESCRIPTORS, FULL_FEATURE_NAMES, FULL_FEATURE_NEUTRALS, FULL_REGISTRY, FeatureDescriptor,
    FeatureGroup, FeatureRegistry, MINIMAL_DESCRIPTORS, MINIMAL_REGISTRY, N_FULL_FEATURES,
    TRADE_INTENSITY_BASELINE, extended_idx, realized_volatility, spread_bps,
};

// V6.3 canonical feature schema (28 SHAP-validated features)
pub use features::{
    CANONICAL_DESCRIPTORS, CANONICAL_FEATURE_NAMES, CANONICAL_FEATURE_NEUTRALS, CANONICAL_REGISTRY,
    N_CANONICAL_FEATURES, canonical_idx,
};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_from_float() {
        assert_eq!(Price::from_float(1.0), Price(10_000));
        assert_eq!(Price::from_float(1.50), Price(15_000));
        assert_eq!(Price::from_float(0.01), Price(100));
        assert_eq!(Price::from_float(100.0), Price(1_000_000));
    }

    #[test]
    fn test_price_to_float() {
        assert!((Price(10_000).to_float() - 1.0).abs() < 1e-10);
        assert!((Price(15_000).to_float() - 1.50).abs() < 1e-10);
        assert!((Price(100).to_float() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_price_arithmetic() {
        let p1 = Price::from_float(10.0);
        let p2 = Price::from_float(3.5);

        assert_eq!((p1 + p2).to_float(), 13.5);
        assert_eq!((p1 - p2).to_float(), 6.5);
    }

    #[test]
    fn test_price_quantity_multiplication() {
        let price = Price::from_float(50.0);
        let quantity = Quantity(100);

        let total = price * quantity;
        assert_eq!(total.to_float(), 5000.0);
    }

    #[test]
    fn test_cash_operations() {
        let c1 = Cash::from_float(1000.0);
        let c2 = Cash::from_float(250.0);

        assert_eq!((c1 - c2).to_float(), 750.0);
        assert!(c1.is_positive());
        assert!(!c1.is_negative());
    }

    #[test]
    fn test_order_side_opposite() {
        assert_eq!(OrderSide::Buy.opposite(), OrderSide::Sell);
        assert_eq!(OrderSide::Sell.opposite(), OrderSide::Buy);
    }

    #[test]
    fn test_limit_order_creation() {
        let order = Order::limit(
            AgentId(1),
            "AAPL",
            OrderSide::Buy,
            Price::from_float(150.0),
            Quantity(100),
        );

        assert_eq!(order.agent_id, AgentId(1));
        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.limit_price(), Some(Price::from_float(150.0)));
        assert_eq!(order.quantity, 100);
        assert!(!order.is_filled());
    }

    #[test]
    fn test_market_order_creation() {
        let order = Order::market(AgentId(2), "GOOGL", OrderSide::Sell, Quantity(50));

        assert_eq!(order.agent_id, AgentId(2));
        assert_eq!(order.symbol, "GOOGL");
        assert_eq!(order.side, OrderSide::Sell);
        assert_eq!(order.limit_price(), None);
        assert!(order.is_sell());
    }

    #[test]
    fn test_trade_value() {
        let trade = Trade {
            id: TradeId(1),
            symbol: "AAPL".to_string(),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: OrderId(1),
            seller_order_id: OrderId(2),
            price: Price::from_float(150.0),
            quantity: Quantity(100),
            timestamp: 0,
            tick: 0,
        };

        assert_eq!(trade.value().to_float(), 15000.0);
    }

    #[test]
    fn test_book_snapshot() {
        let snapshot = BookSnapshot {
            symbol: "AAPL".to_string(),
            bids: vec![
                BookLevel {
                    price: Price::from_float(99.0),
                    quantity: Quantity(100),
                    order_count: 2,
                },
                BookLevel {
                    price: Price::from_float(98.0),
                    quantity: Quantity(200),
                    order_count: 3,
                },
            ],
            asks: vec![
                BookLevel {
                    price: Price::from_float(101.0),
                    quantity: Quantity(150),
                    order_count: 1,
                },
                BookLevel {
                    price: Price::from_float(102.0),
                    quantity: Quantity(250),
                    order_count: 2,
                },
            ],
            timestamp: 0,
            tick: 0,
        };

        assert_eq!(snapshot.best_bid(), Some(Price::from_float(99.0)));
        assert_eq!(snapshot.best_ask(), Some(Price::from_float(101.0)));
        assert_eq!(snapshot.spread(), Some(Price::from_float(2.0)));
        assert_eq!(snapshot.mid_price(), Some(Price::from_float(100.0)));
    }

    #[test]
    fn test_symbol_config_defaults() {
        let config = SymbolConfig::default();
        assert_eq!(config.symbol, "SIM");
        assert_eq!(config.shares_outstanding, Quantity(1_000_000));
        assert_eq!(config.borrow_pool_bps, 1500);
    }

    #[test]
    fn test_symbol_config_borrow_pool_size() {
        let config = SymbolConfig::new("TEST", Quantity(10_000_000), Price::from_float(50.0))
            .with_borrow_pool_bps(1500); // 15%

        // 10,000,000 * 15% = 1,500,000
        assert_eq!(config.borrow_pool_size(), Quantity(1_500_000));
    }

    #[test]
    fn test_short_selling_config() {
        let disabled = ShortSellingConfig::disabled();
        assert!(!disabled.enabled);

        let enabled = ShortSellingConfig::enabled_default()
            .with_borrow_rate_bps(100)
            .with_max_short(Quantity(5_000));
        assert!(enabled.enabled);
        assert_eq!(enabled.borrow_rate_bps, 100);
        assert_eq!(enabled.max_short_per_agent, Quantity(5_000));
    }

    // V2.2 Fill Tests
    #[test]
    fn test_fill_value() {
        let fill = Fill {
            id: FillId(1),
            symbol: "TEST".to_string(),
            order_id: OrderId(1),
            aggressor_id: AgentId(1),
            resting_id: AgentId(2),
            resting_order_id: OrderId(2),
            aggressor_side: OrderSide::Buy,
            price: Price::from_float(100.0),
            quantity: Quantity(50),
            reference_price: Some(Price::from_float(99.5)),
            timestamp: 0,
            tick: 0,
        };

        assert_eq!(fill.value().to_float(), 5000.0);
    }

    #[test]
    fn test_fill_slippage_buy() {
        let fill = Fill {
            id: FillId(1),
            symbol: "TEST".to_string(),
            order_id: OrderId(1),
            aggressor_id: AgentId(1),
            resting_id: AgentId(2),
            resting_order_id: OrderId(2),
            aggressor_side: OrderSide::Buy,
            price: Price::from_float(101.0),
            quantity: Quantity(50),
            reference_price: Some(Price::from_float(100.0)),
            timestamp: 0,
            tick: 0,
        };

        // Paid $101, reference was $100 = $1 slippage = 100 bps
        let slippage = fill.slippage().unwrap();
        assert_eq!(slippage.to_float(), 1.0);

        let slippage_bps = fill.slippage_bps().unwrap();
        assert_eq!(slippage_bps, 100);
    }

    #[test]
    fn test_fill_slippage_sell() {
        let fill = Fill {
            id: FillId(1),
            symbol: "TEST".to_string(),
            order_id: OrderId(1),
            aggressor_id: AgentId(1),
            resting_id: AgentId(2),
            resting_order_id: OrderId(2),
            aggressor_side: OrderSide::Sell,
            price: Price::from_float(99.0),
            quantity: Quantity(50),
            reference_price: Some(Price::from_float(100.0)),
            timestamp: 0,
            tick: 0,
        };

        // Received $99, reference was $100 = $1 slippage = 100 bps
        let slippage = fill.slippage().unwrap();
        assert_eq!(slippage.to_float(), 1.0);

        let slippage_bps = fill.slippage_bps().unwrap();
        assert_eq!(slippage_bps, 100);
    }

    #[test]
    fn test_fill_no_reference_price() {
        let fill = Fill {
            id: FillId(1),
            symbol: "TEST".to_string(),
            order_id: OrderId(1),
            aggressor_id: AgentId(1),
            resting_id: AgentId(2),
            resting_order_id: OrderId(2),
            aggressor_side: OrderSide::Buy,
            price: Price::from_float(100.0),
            quantity: Quantity(50),
            reference_price: None,
            timestamp: 0,
            tick: 0,
        };

        assert!(fill.slippage().is_none());
        assert!(fill.slippage_bps().is_none());
    }

    // V2.2 SlippageMetrics Tests
    #[test]
    fn test_slippage_metrics_empty() {
        let metrics = SlippageMetrics::new(Some(Price::from_float(100.0)));
        assert!(metrics.vwap().is_none());
        assert!(metrics.slippage_buy().is_none());
        assert_eq!(metrics.levels_crossed, 0);
    }

    #[test]
    fn test_slippage_metrics_single_fill() {
        let mut metrics = SlippageMetrics::new(Some(Price::from_float(100.0)));
        metrics.record_fill(Price::from_float(101.0), Quantity(100));

        assert_eq!(metrics.vwap().unwrap().to_float(), 101.0);
        assert_eq!(metrics.filled_quantity, Quantity(100));
        assert_eq!(metrics.levels_crossed, 1);

        // Buy slippage = 101 - 100 = $1 = 100 bps
        let slippage = metrics.slippage_buy().unwrap();
        assert_eq!(slippage.to_float(), 1.0);
        assert_eq!(metrics.slippage_bps(OrderSide::Buy), Some(100));
    }

    #[test]
    fn test_slippage_metrics_multiple_fills() {
        let mut metrics = SlippageMetrics::new(Some(Price::from_float(100.0)));

        // Fill 1: 60 shares at $101
        metrics.record_fill(Price::from_float(101.0), Quantity(60));
        // Fill 2: 40 shares at $102
        metrics.record_fill(Price::from_float(102.0), Quantity(40));

        // VWAP = (60*101 + 40*102) / 100 = (6060 + 4080) / 100 = 101.40
        let vwap = metrics.vwap().unwrap();
        assert!((vwap.to_float() - 101.4).abs() < 0.01);

        assert_eq!(metrics.filled_quantity, Quantity(100));
        assert_eq!(metrics.levels_crossed, 2);
        assert_eq!(metrics.best_fill_price, Some(Price::from_float(101.0)));
        assert_eq!(metrics.worst_fill_price, Some(Price::from_float(102.0)));

        // Buy slippage = 101.4 - 100 = $1.40 = 140 bps
        let slippage_bps = metrics.slippage_bps(OrderSide::Buy).unwrap();
        assert_eq!(slippage_bps, 140);
    }

    #[test]
    fn test_slippage_metrics_fill_range() {
        let mut metrics = SlippageMetrics::new(None);
        metrics.record_fill(Price::from_float(99.0), Quantity(10));
        metrics.record_fill(Price::from_float(101.0), Quantity(20));
        metrics.record_fill(Price::from_float(100.0), Quantity(30));

        let range = metrics.fill_range().unwrap();
        assert_eq!(range.to_float(), 2.0); // 101 - 99 = 2
    }

    #[test]
    fn test_slippage_config_builders() {
        let config = SlippageConfig::enabled()
            .with_sqrt_model()
            .with_linear_impact_bps(20)
            .with_impact_threshold_bps(50);

        assert!(config.enabled);
        assert!(config.use_sqrt_model);
        assert_eq!(config.linear_impact_bps, 20);
        assert_eq!(config.impact_threshold_bps, 50);

        let disabled = SlippageConfig::disabled();
        assert!(!disabled.enabled);
    }
}
