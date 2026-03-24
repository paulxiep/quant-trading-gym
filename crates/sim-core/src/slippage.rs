//! Slippage and market impact calculation (V2.2).
//!
//! This module provides tools for estimating and measuring the price impact
//! of orders based on their size relative to available liquidity.
//!
//! # Impact Models
//!
//! Two impact models are supported:
//! - **Linear**: `impact = coefficient * (order_size / liquidity)`
//! - **Square-root**: `impact = coefficient * sqrt(order_size / liquidity)`
//!
//! The square-root model is more realistic for larger orders, as price impact
//! tends to grow sub-linearly with order size.

use types::{OrderSide, Price, Quantity, SlippageConfig};

use crate::order_book::OrderBook;

/// Calculator for estimating market impact and slippage.
///
/// Given an order book and slippage configuration, this calculator can:
/// - Estimate the expected price impact of an order before execution
/// - Calculate the available liquidity at various price levels
/// - Determine if an order is "large" (above impact threshold)
#[derive(Debug, Clone)]
pub struct SlippageCalculator {
    config: SlippageConfig,
}

impl Default for SlippageCalculator {
    fn default() -> Self {
        Self::new(SlippageConfig::default())
    }
}

impl SlippageCalculator {
    /// Create a new slippage calculator with the given configuration.
    pub fn new(config: SlippageConfig) -> Self {
        Self { config }
    }

    /// Check if slippage tracking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SlippageConfig {
        &self.config
    }

    /// Calculate available liquidity on the opposite side of the book.
    ///
    /// For buy orders, returns total ask volume.
    /// For sell orders, returns total bid volume.
    pub fn available_liquidity(&self, book: &OrderBook, side: OrderSide) -> Quantity {
        match side {
            OrderSide::Buy => book.total_ask_volume(),
            OrderSide::Sell => book.total_bid_volume(),
        }
    }

    /// Calculate available liquidity within a price range.
    ///
    /// For buy orders, returns ask volume up to max_price.
    /// For sell orders, returns bid volume down to min_price.
    pub fn available_liquidity_in_range(
        &self,
        book: &OrderBook,
        side: OrderSide,
        price_limit: Option<Price>,
    ) -> Quantity {
        match (side, price_limit) {
            (OrderSide::Buy, Some(max_price)) => book.ask_depth_to_price(max_price),
            (OrderSide::Buy, None) => book.total_ask_volume(),
            (OrderSide::Sell, Some(min_price)) => book.bid_depth_to_price(min_price),
            (OrderSide::Sell, None) => book.total_bid_volume(),
        }
    }

    /// Calculate the liquidity ratio (order_size / available_liquidity).
    ///
    /// Returns the ratio in basis points (e.g., 100 = 1% of liquidity).
    pub fn liquidity_ratio_bps(&self, order_size: Quantity, liquidity: Quantity) -> u64 {
        if liquidity.is_zero() {
            return 10_000; // 100% if no liquidity
        }
        (order_size.raw() * 10_000) / liquidity.raw()
    }

    /// Check if an order is considered "large" (above impact threshold).
    pub fn is_large_order(&self, order_size: Quantity, liquidity: Quantity) -> bool {
        let ratio_bps = self.liquidity_ratio_bps(order_size, liquidity);
        ratio_bps >= self.config.impact_threshold_bps as u64
    }

    /// Estimate the expected price impact in basis points.
    ///
    /// Uses the configured impact model (linear or sqrt) to estimate
    /// how much the execution price will deviate from the reference price.
    ///
    /// Returns `None` if slippage is disabled or liquidity is zero.
    pub fn estimate_impact_bps(&self, order_size: Quantity, liquidity: Quantity) -> Option<i64> {
        if !self.config.enabled || liquidity.is_zero() {
            return None;
        }

        let ratio = order_size.raw() as f64 / liquidity.raw() as f64;

        // Check if below threshold
        let ratio_bps = (ratio * 10_000.0) as u64;
        if ratio_bps < self.config.impact_threshold_bps as u64 {
            return Some(0);
        }

        let impact = if self.config.use_sqrt_model {
            // Square-root impact: coefficient * sqrt(ratio)
            self.config.linear_impact_bps as f64 * ratio.sqrt() * 100.0
        } else {
            // Linear impact: coefficient * ratio
            // linear_impact_bps is per 1% of liquidity, so multiply by ratio * 100
            self.config.linear_impact_bps as f64 * ratio * 100.0
        };

        Some(impact.round() as i64)
    }

    /// Estimate the expected execution price for an order.
    ///
    /// Takes the current mid price and estimates where the order would
    /// execute based on size and available liquidity.
    pub fn estimate_execution_price(
        &self,
        book: &OrderBook,
        side: OrderSide,
        quantity: Quantity,
    ) -> Option<Price> {
        let mid = book.mid_price()?;
        let liquidity = self.available_liquidity(book, side);
        let impact_bps = self.estimate_impact_bps(quantity, liquidity)?;

        // Convert bps to price movement
        // impact_bps positive means worse price
        let price_impact = Price((mid.raw() * impact_bps) / 10_000);

        let estimated = match side {
            OrderSide::Buy => mid + price_impact,  // Pay more
            OrderSide::Sell => mid - price_impact, // Receive less
        };

        Some(estimated)
    }

    /// Calculate the actual slippage from a completed order.
    ///
    /// Compares the volume-weighted average price (VWAP) of fills
    /// against the reference price to determine realized slippage.
    pub fn calculate_realized_slippage_bps(
        &self,
        vwap: Price,
        reference_price: Price,
        side: OrderSide,
    ) -> i64 {
        if reference_price.raw() == 0 {
            return 0;
        }

        let diff = match side {
            OrderSide::Buy => vwap - reference_price, // Positive = paid more
            OrderSide::Sell => reference_price - vwap, // Positive = received less
        };

        (diff.raw() * 10_000) / reference_price.raw()
    }
}

/// Pre-trade analysis for estimating execution cost.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpactEstimate {
    /// Expected impact in basis points.
    pub impact_bps: i64,
    /// Estimated execution price.
    pub estimated_price: Option<Price>,
    /// Ratio of order size to available liquidity (bps).
    pub liquidity_ratio_bps: u64,
    /// Whether order is considered "large".
    pub is_large_order: bool,
    /// Available liquidity on the opposite side.
    pub available_liquidity: Quantity,
}

impl SlippageCalculator {
    /// Perform a comprehensive pre-trade impact analysis.
    pub fn analyze_impact(
        &self,
        book: &OrderBook,
        side: OrderSide,
        quantity: Quantity,
    ) -> ImpactEstimate {
        let available_liquidity = self.available_liquidity(book, side);
        let liquidity_ratio_bps = self.liquidity_ratio_bps(quantity, available_liquidity);
        let is_large_order = self.is_large_order(quantity, available_liquidity);

        let impact_bps = self
            .estimate_impact_bps(quantity, available_liquidity)
            .unwrap_or(0);

        let estimated_price = self.estimate_execution_price(book, side, quantity);

        ImpactEstimate {
            impact_bps,
            estimated_price,
            liquidity_ratio_bps,
            is_large_order,
            available_liquidity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{AgentId, Order, OrderId};

    fn setup_book() -> OrderBook {
        let mut book = OrderBook::new("TEST");

        // Add asks (sell orders)
        for i in 0..5 {
            let price = Price::from_float(100.0 + i as f64);
            let mut order = Order::limit(
                AgentId(i as u64 + 1),
                "TEST",
                OrderSide::Sell,
                price,
                Quantity(100),
            );
            order.id = OrderId(i as u64 + 1);
            book.add_order(order).unwrap();
        }

        // Add bids (buy orders)
        for i in 0..5 {
            let price = Price::from_float(99.0 - i as f64);
            let mut order = Order::limit(
                AgentId(i as u64 + 10),
                "TEST",
                OrderSide::Buy,
                price,
                Quantity(100),
            );
            order.id = OrderId(i as u64 + 10);
            book.add_order(order).unwrap();
        }

        book
    }

    #[test]
    fn test_available_liquidity() {
        let book = setup_book();
        let calc = SlippageCalculator::default();

        assert_eq!(
            calc.available_liquidity(&book, OrderSide::Buy),
            Quantity(500)
        );
        assert_eq!(
            calc.available_liquidity(&book, OrderSide::Sell),
            Quantity(500)
        );
    }

    #[test]
    fn test_liquidity_ratio() {
        let calc = SlippageCalculator::default();

        // 10% of liquidity
        let ratio = calc.liquidity_ratio_bps(Quantity(50), Quantity(500));
        assert_eq!(ratio, 1000); // 10% = 1000 bps

        // 50% of liquidity
        let ratio = calc.liquidity_ratio_bps(Quantity(250), Quantity(500));
        assert_eq!(ratio, 5000); // 50% = 5000 bps
    }

    #[test]
    fn test_is_large_order() {
        let calc =
            SlippageCalculator::new(SlippageConfig::enabled().with_impact_threshold_bps(100));

        // Small order: 0.5% of liquidity
        assert!(!calc.is_large_order(Quantity(5), Quantity(1000)));

        // Large order: 5% of liquidity
        assert!(calc.is_large_order(Quantity(50), Quantity(1000)));
    }

    #[test]
    fn test_estimate_impact_linear() {
        let calc = SlippageCalculator::new(
            SlippageConfig::enabled()
                .with_linear_impact_bps(10)
                .with_impact_threshold_bps(0), // No threshold for testing
        );

        // 10% of liquidity with 10bps coefficient
        // Impact = 10 * 0.10 * 100 = 100 bps
        let impact = calc.estimate_impact_bps(Quantity(100), Quantity(1000));
        assert_eq!(impact, Some(100));
    }

    #[test]
    fn test_estimate_impact_sqrt() {
        let calc = SlippageCalculator::new(
            SlippageConfig::enabled()
                .with_sqrt_model()
                .with_linear_impact_bps(10)
                .with_impact_threshold_bps(0),
        );

        // 25% of liquidity with sqrt model
        // Impact = 10 * sqrt(0.25) * 100 = 10 * 0.5 * 100 = 500 bps
        let impact = calc.estimate_impact_bps(Quantity(250), Quantity(1000));
        assert_eq!(impact, Some(500));
    }

    #[test]
    fn test_impact_below_threshold() {
        let calc =
            SlippageCalculator::new(SlippageConfig::enabled().with_impact_threshold_bps(500)); // 5%

        // Order consuming 2% of liquidity - below threshold
        let impact = calc.estimate_impact_bps(Quantity(20), Quantity(1000));
        assert_eq!(impact, Some(0));
    }

    #[test]
    fn test_analyze_impact() {
        let book = setup_book();
        let calc = SlippageCalculator::new(
            SlippageConfig::enabled()
                .with_linear_impact_bps(10)
                .with_impact_threshold_bps(100),
        );

        let estimate = calc.analyze_impact(&book, OrderSide::Buy, Quantity(100));

        assert_eq!(estimate.available_liquidity, Quantity(500));
        assert_eq!(estimate.liquidity_ratio_bps, 2000); // 20%
        assert!(estimate.is_large_order);
        assert!(estimate.impact_bps > 0);
    }

    #[test]
    fn test_realized_slippage() {
        let calc = SlippageCalculator::default();

        // Buy at $101 when reference was $100 = 100 bps slippage
        let slippage = calc.calculate_realized_slippage_bps(
            Price::from_float(101.0),
            Price::from_float(100.0),
            OrderSide::Buy,
        );
        assert_eq!(slippage, 100);

        // Sell at $99 when reference was $100 = 100 bps slippage
        let slippage = calc.calculate_realized_slippage_bps(
            Price::from_float(99.0),
            Price::from_float(100.0),
            OrderSide::Sell,
        );
        assert_eq!(slippage, 100);
    }

    #[test]
    fn test_disabled_slippage() {
        let calc = SlippageCalculator::new(SlippageConfig::disabled());

        assert!(!calc.is_enabled());
        assert_eq!(
            calc.estimate_impact_bps(Quantity(100), Quantity(1000)),
            None
        );
    }
}
