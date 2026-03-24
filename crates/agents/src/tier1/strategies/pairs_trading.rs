//! Pairs Trading Strategy - Cointegration-based spread trading (V3.3).
//!
//! This is a Tier 1 multi-symbol strategy that exploits mean-reversion in the
//! spread between two cointegrated assets.
//!
//! # Strategy Logic
//!
//! 1. Track price relationship between two correlated symbols (e.g., XOM/CVX)
//! 2. Compute spread using OLS hedge ratio: `spread = A - hedge_ratio * B`
//! 3. Enter when spread diverges: `|z_score| > entry_threshold`
//! 4. Exit when spread converges: `|z_score| < exit_threshold`
//!
//! # Position Management
//!
//! - **Long spread** (z < -entry): Buy A, sell B (expect spread to widen)
//! - **Short spread** (z > +entry): Sell A, buy B (expect spread to narrow)
//! - **Exit**: Close both legs when spread mean-reverts
//!
//! # Design (Declarative, Modular, SoC)
//!
//! - **Declarative**: Config defines symbols, thresholds; strategy handles logic
//! - **Modular**: Uses `CointegrationTracker` from `quant` crate
//! - **SoC**: Computes signals only; simulation handles order execution
//!
//! # Borrow-Checker Safety
//!
//! - Owns `CointegrationTracker` and `AgentState` (no shared references)
//! - `ctx.mid_price()` returns owned `Option<Price>` (no overlapping borrows)
//! - Returns `AgentAction::multiple()` for simultaneous leg orders

use std::collections::HashMap;

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use quant::CointegrationTracker;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Symbol, Trade};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for a pairs trading strategy.
///
/// # Example
///
/// ```ignore
/// let config = PairsTradingConfig {
///     symbol_a: "XOM".into(),
///     symbol_b: "CVX".into(),
///     entry_z_threshold: 2.0,
///     exit_z_threshold: 0.5,
///     max_position_per_leg: 100,
///     lookback_window: 100,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct PairsTradingConfig {
    /// First symbol in the pair (the "dependent" variable in OLS).
    pub symbol_a: Symbol,
    /// Second symbol in the pair (the "independent" variable in OLS).
    pub symbol_b: Symbol,
    /// Lookback window for cointegration calculations.
    pub lookback_window: usize,
    /// Z-score threshold to enter a position (e.g., 2.0 = 2 std devs).
    pub entry_z_threshold: f64,
    /// Z-score threshold to exit a position (e.g., 0.5 = reversion).
    pub exit_z_threshold: f64,
    /// Maximum position size per leg (shares).
    pub max_position_per_leg: i64,
    /// Whether to dynamically rebalance hedge ratio.
    pub rebalance_on_drift: bool,
    /// Hedge ratio drift threshold for rebalancing (percentage change).
    pub hedge_drift_threshold: f64,
    /// Starting cash balance.
    pub initial_cash: Cash,
}

impl Default for PairsTradingConfig {
    fn default() -> Self {
        Self {
            symbol_a: "SYMBOL_A".to_string(),
            symbol_b: "SYMBOL_B".to_string(),
            lookback_window: 100,
            entry_z_threshold: 2.0,
            exit_z_threshold: 0.5,
            max_position_per_leg: 100,
            rebalance_on_drift: false,
            hedge_drift_threshold: 0.1, // 10% change triggers rebalance
            initial_cash: Cash::from_float(100_000.0),
        }
    }
}

impl PairsTradingConfig {
    /// Create a new config for a specific pair.
    pub fn new(symbol_a: impl Into<Symbol>, symbol_b: impl Into<Symbol>) -> Self {
        Self {
            symbol_a: symbol_a.into(),
            symbol_b: symbol_b.into(),
            ..Default::default()
        }
    }

    /// Builder: set entry z-score threshold.
    pub fn with_entry_threshold(mut self, threshold: f64) -> Self {
        self.entry_z_threshold = threshold.abs();
        self
    }

    /// Builder: set exit z-score threshold.
    pub fn with_exit_threshold(mut self, threshold: f64) -> Self {
        self.exit_z_threshold = threshold.abs();
        self
    }

    /// Builder: set maximum position per leg.
    pub fn with_max_position(mut self, max_pos: i64) -> Self {
        self.max_position_per_leg = max_pos;
        self
    }

    /// Builder: set lookback window.
    pub fn with_lookback(mut self, lookback: usize) -> Self {
        self.lookback_window = lookback;
        self
    }

    /// Builder: set initial cash.
    pub fn with_initial_cash(mut self, cash: Cash) -> Self {
        self.initial_cash = cash;
        self
    }

    /// Builder: enable hedge ratio rebalancing.
    pub fn with_rebalancing(mut self, enabled: bool) -> Self {
        self.rebalance_on_drift = enabled;
        self
    }
}

// =============================================================================
// Position State
// =============================================================================

/// Current spread position state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadPosition {
    /// No position in either leg.
    Flat,
    /// Long spread: long A, short B (expect spread to widen/rise).
    LongSpread,
    /// Short spread: short A, long B (expect spread to narrow/fall).
    ShortSpread,
}

// =============================================================================
// PairsTrading Strategy
// =============================================================================

/// Pairs trading strategy using cointegration-based spread trading.
///
/// This is a Tier 1 agent that runs every tick, monitoring the spread
/// between two cointegrated symbols and trading mean-reversion.
///
/// # Multi-Symbol Design (V3.3)
///
/// - Tracks positions in both symbols via `AgentState`
/// - Returns `AgentAction::multiple()` for simultaneous leg orders
/// - `watched_symbols()` returns both symbols for proper equity calculation
pub struct PairsTrading {
    /// Unique agent identifier.
    id: AgentId,
    /// Strategy configuration.
    config: PairsTradingConfig,
    /// Multi-symbol position and cash tracking.
    state: AgentState,
    /// Cointegration analysis.
    cointegration: CointegrationTracker,
    /// Current spread position.
    position: SpreadPosition,
    /// Last computed hedge ratio (for drift detection).
    last_hedge_ratio: Option<f64>,
    /// Symbols as vec for `watched_symbols()` return.
    watched: Vec<Symbol>,
    /// Random number generator for order price variation.
    rng: StdRng,
}

impl PairsTrading {
    /// Create a new pairs trading strategy.
    pub fn new(id: AgentId, config: PairsTradingConfig) -> Self {
        let watched = vec![config.symbol_a.clone(), config.symbol_b.clone()];
        let state = AgentState::with_symbols(config.initial_cash, watched.clone());
        let cointegration = CointegrationTracker::new(config.lookback_window);

        Self {
            id,
            config,
            state,
            cointegration,
            position: SpreadPosition::Flat,
            last_hedge_ratio: None,
            watched,
            rng: StdRng::from_entropy(),
        }
    }

    /// Get the current spread position state.
    pub fn spread_position(&self) -> SpreadPosition {
        self.position
    }

    /// Check if we can enter a new position.
    fn can_enter(&self) -> bool {
        self.position == SpreadPosition::Flat
    }

    /// Check if we're currently in a position that can be exited.
    fn can_exit(&self) -> bool {
        self.position != SpreadPosition::Flat
    }

    /// Generate entry orders for long spread (buy A, sell B).
    fn enter_long_spread(&mut self, ctx: &StrategyContext<'_>, hedge_ratio: f64) -> Vec<Order> {
        let price_a = match ctx.mid_price(&self.config.symbol_a) {
            Some(p) => p,
            None => return vec![],
        };
        let price_b = match ctx.mid_price(&self.config.symbol_b) {
            Some(p) => p,
            None => return vec![],
        };

        // Size leg B based on hedge ratio to maintain dollar-neutral exposure
        let qty_a = self.config.max_position_per_leg as u64;
        let qty_b = ((qty_a as f64) * hedge_ratio).round() as u64;

        // Random multipliers for price variation
        let mult_a = self.rng.r#gen_range(0.99..1.01);
        let mult_b = self.rng.r#gen_range(0.99..1.01);

        vec![
            // Buy A
            Order::limit(
                self.id,
                &self.config.symbol_a,
                OrderSide::Buy,
                Price::from_float(floor_price(price_a.to_float() * mult_a)),
                Quantity(qty_a),
            ),
            // Sell B
            Order::limit(
                self.id,
                &self.config.symbol_b,
                OrderSide::Sell,
                Price::from_float(floor_price(price_b.to_float() * mult_b)),
                Quantity(qty_b),
            ),
        ]
    }

    /// Generate entry orders for short spread (sell A, buy B).
    fn enter_short_spread(&mut self, ctx: &StrategyContext<'_>, hedge_ratio: f64) -> Vec<Order> {
        let price_a = match ctx.mid_price(&self.config.symbol_a) {
            Some(p) => p,
            None => return vec![],
        };
        let price_b = match ctx.mid_price(&self.config.symbol_b) {
            Some(p) => p,
            None => return vec![],
        };

        let qty_a = self.config.max_position_per_leg as u64;
        let qty_b = ((qty_a as f64) * hedge_ratio).round() as u64;

        // Random multipliers for price variation
        let mult_a = self.rng.r#gen_range(0.99..1.01);
        let mult_b = self.rng.r#gen_range(0.99..1.01);

        vec![
            // Sell A
            Order::limit(
                self.id,
                &self.config.symbol_a,
                OrderSide::Sell,
                Price::from_float(floor_price(price_a.to_float() * mult_a)),
                Quantity(qty_a),
            ),
            // Buy B
            Order::limit(
                self.id,
                &self.config.symbol_b,
                OrderSide::Buy,
                Price::from_float(floor_price(price_b.to_float() * mult_b)),
                Quantity(qty_b),
            ),
        ]
    }

    /// Generate exit orders to close both legs.
    fn exit_spread(&mut self, ctx: &StrategyContext<'_>) -> Vec<Order> {
        // Define legs to close: (symbol, position)
        let legs = [
            (
                &self.config.symbol_a,
                self.state.position_for(&self.config.symbol_a),
            ),
            (
                &self.config.symbol_b,
                self.state.position_for(&self.config.symbol_b),
            ),
        ];

        // Generate exit orders for non-zero positions with available prices
        let mut orders = Vec::new();
        for (symbol, pos) in legs.into_iter() {
            if pos == 0 {
                continue;
            }
            if let Some(price) = ctx.mid_price(symbol) {
                let side = if pos > 0 {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };
                // Random multiplier for price variation
                let mult = self.rng.r#gen_range(0.99..1.01);
                orders.push(Order::limit(
                    self.id,
                    symbol,
                    side,
                    Price::from_float(floor_price(price.to_float() * mult)),
                    Quantity(pos.unsigned_abs()),
                ));
            }
        }
        orders
    }
}

impl Agent for PairsTrading {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Get prices for both symbols
        let price_a = match ctx.mid_price(&self.config.symbol_a) {
            Some(p) => p.to_float(),
            None => return AgentAction::none(),
        };
        let price_b = match ctx.mid_price(&self.config.symbol_b) {
            Some(p) => p.to_float(),
            None => return AgentAction::none(),
        };

        // Update cointegration tracker
        let result = match self.cointegration.update(price_a, price_b) {
            Some(r) => r,
            None => return AgentAction::none(), // Not enough data yet
        };

        let z_score = result.z_score;
        let hedge_ratio = result.hedge_ratio;

        // Track hedge ratio for potential rebalancing
        self.last_hedge_ratio = Some(hedge_ratio);

        // Decision logic based on z-score
        match self.position {
            SpreadPosition::Flat => {
                // Check for entry signals
                if z_score < -self.config.entry_z_threshold && self.can_enter() {
                    // Spread is too low → long spread (buy A, sell B)
                    let orders = self.enter_long_spread(ctx, hedge_ratio);
                    if !orders.is_empty() {
                        self.position = SpreadPosition::LongSpread;
                        (0..orders.len()).for_each(|_| self.state.record_order());
                        return AgentAction::multiple(orders);
                    }
                } else if z_score > self.config.entry_z_threshold && self.can_enter() {
                    // Spread is too high → short spread (sell A, buy B)
                    let orders = self.enter_short_spread(ctx, hedge_ratio);
                    if !orders.is_empty() {
                        self.position = SpreadPosition::ShortSpread;
                        (0..orders.len()).for_each(|_| self.state.record_order());
                        return AgentAction::multiple(orders);
                    }
                }
            }
            SpreadPosition::LongSpread | SpreadPosition::ShortSpread => {
                // Exit when z-score reverts toward mean
                if z_score.abs() < self.config.exit_z_threshold && self.can_exit() {
                    let orders = self.exit_spread(ctx);
                    if !orders.is_empty() {
                        self.position = SpreadPosition::Flat;
                        (0..orders.len()).for_each(|_| self.state.record_order());
                        return AgentAction::multiple(orders);
                    }
                }
            }
        }

        AgentAction::none()
    }

    fn on_fill(&mut self, trade: &Trade) {
        // Use separate if blocks (not else if) to handle self-trades correctly.
        if trade.buyer_id == self.id {
            self.state
                .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
        }
        if trade.seller_id == self.id {
            self.state
                .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
        }
    }

    fn name(&self) -> &str {
        "PairsTrading"
    }

    fn state(&self) -> &AgentState {
        &self.state
    }

    fn position(&self) -> i64 {
        self.state.position()
    }

    fn position_for(&self, symbol: &str) -> i64 {
        self.state.position_for(symbol)
    }

    fn positions(&self) -> &HashMap<Symbol, crate::state::PositionEntry> {
        self.state.positions()
    }

    fn watched_symbols(&self) -> Vec<Symbol> {
        self.watched.clone()
    }

    fn equity(&self, prices: &HashMap<Symbol, Price>) -> Cash {
        self.state.equity(prices)
    }

    fn equity_for(&self, symbol: &str, price: Price) -> Cash {
        self.state.equity_for(symbol, price)
    }

    fn is_market_maker(&self) -> bool {
        false
    }

    fn is_reactive(&self) -> bool {
        false // Tier 1: runs every tick
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quant::IndicatorSnapshot;
    use sim_core::Market;
    use std::collections::HashMap;
    use types::{Candle, OrderId};

    fn setup_test_market() -> Market {
        let mut market = Market::new();

        // Add SYMBOL_A
        market.add_symbol("SYMBOL_A");
        {
            let book_a = market.get_book_mut(&"SYMBOL_A".to_string()).unwrap();
            let mut bid_a = Order::limit(
                AgentId(99),
                "SYMBOL_A",
                OrderSide::Buy,
                Price::from_float(99.0),
                Quantity(1000),
            );
            bid_a.id = OrderId(1);
            let mut ask_a = Order::limit(
                AgentId(99),
                "SYMBOL_A",
                OrderSide::Sell,
                Price::from_float(101.0),
                Quantity(1000),
            );
            ask_a.id = OrderId(2);
            book_a.add_order(bid_a).unwrap();
            book_a.add_order(ask_a).unwrap();
        }

        // Add SYMBOL_B
        market.add_symbol("SYMBOL_B");
        {
            let book_b = market.get_book_mut(&"SYMBOL_B".to_string()).unwrap();
            let mut bid_b = Order::limit(
                AgentId(99),
                "SYMBOL_B",
                OrderSide::Buy,
                Price::from_float(49.0),
                Quantity(1000),
            );
            bid_b.id = OrderId(3);
            let mut ask_b = Order::limit(
                AgentId(99),
                "SYMBOL_B",
                OrderSide::Sell,
                Price::from_float(51.0),
                Quantity(1000),
            );
            ask_b.id = OrderId(4);
            book_b.add_order(bid_b).unwrap();
            book_b.add_order(ask_b).unwrap();
        }

        market
    }

    #[test]
    fn test_pairs_trading_config_builder() {
        let config = PairsTradingConfig::new("XOM", "CVX")
            .with_entry_threshold(2.5)
            .with_exit_threshold(0.3)
            .with_max_position(200)
            .with_lookback(50);

        assert_eq!(config.symbol_a, "XOM");
        assert_eq!(config.symbol_b, "CVX");
        assert!((config.entry_z_threshold - 2.5).abs() < 0.001);
        assert!((config.exit_z_threshold - 0.3).abs() < 0.001);
        assert_eq!(config.max_position_per_leg, 200);
        assert_eq!(config.lookback_window, 50);
    }

    #[test]
    fn test_pairs_trading_new() {
        let config = PairsTradingConfig::new("SYMBOL_A", "SYMBOL_B");
        let agent = PairsTrading::new(AgentId(1), config);

        assert_eq!(agent.id(), AgentId(1));
        assert_eq!(agent.spread_position(), SpreadPosition::Flat);
        assert_eq!(agent.watched_symbols(), vec!["SYMBOL_A", "SYMBOL_B"]);
        assert!(!agent.is_market_maker());
        assert!(!agent.is_reactive());
    }

    #[test]
    fn test_pairs_trading_no_action_insufficient_data() {
        let config = PairsTradingConfig::new("SYMBOL_A", "SYMBOL_B").with_lookback(20);
        let mut agent = PairsTrading::new(AgentId(1), config);

        let market = setup_test_market();
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let indicators = IndicatorSnapshot::new(100);
        let recent_trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // First tick - not enough data for cointegration
        let ctx = StrategyContext::new(
            1,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let action = agent.on_tick(&ctx);
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_pairs_trading_watched_symbols() {
        let config = PairsTradingConfig::new("AAPL", "MSFT");
        let agent = PairsTrading::new(AgentId(1), config);

        let symbols = agent.watched_symbols();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&"AAPL".to_string()));
        assert!(symbols.contains(&"MSFT".to_string()));
    }
}
