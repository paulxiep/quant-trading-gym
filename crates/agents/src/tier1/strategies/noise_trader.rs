//! Noise Trader - generates random market activity.
//!
//! A simple agent that places random orders near the current price.
//! This provides liquidity and price discovery by generating trades.
//!
//! # Zombie Risk Prevention
//! NoiseTrader orders use a reference price hierarchy:
//! 1. Fair value from fundamentals (if available)
//! 2. Mid price from order book
//! 3. Last trade price
//! 4. Configured initial price
//!
//! # V2.4 Fair Value Integration
//! When fundamentals are configured, noise trades anchor around the
//! Gordon Growth Model fair value. This creates price discovery pressure
//! toward intrinsic value while still generating market noise.

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Symbol, Trade};

/// Configuration for a NoiseTrader agent.
#[derive(Debug, Clone)]
pub struct NoiseTraderConfig {
    /// Symbol to trade.
    pub symbol: Symbol,
    /// Probability of placing an order each tick (0.0 to 1.0).
    pub order_probability: f64,
    /// Maximum price deviation from mid as a fraction (e.g., 0.02 = 2%).
    pub price_deviation: f64,
    /// Minimum order size.
    pub min_quantity: u64,
    /// Maximum order size.
    pub max_quantity: u64,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Starting position in shares (allows some traders to start as sellers).
    pub initial_position: i64,
    /// Maximum long position (stops buying when reached).
    pub max_long_position: i64,
    /// Maximum short position as positive number (stops selling when reached).
    pub max_short_position: i64,
}

impl Default for NoiseTraderConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            order_probability: 0.3,
            price_deviation: 0.02,
            min_quantity: 10,
            max_quantity: 100,
            initial_price: Price::from_float(100.0),
            initial_cash: Cash::from_float(100_000.0),
            initial_position: 50, // Start with some shares to allow selling
            max_long_position: 100,
            max_short_position: 0, // No short selling by default
        }
    }
}

/// A random trader that generates market activity.
///
/// NoiseTraders provide essential liquidity and price movement in the
/// simulation. They place limit orders randomly near the current mid price.
pub struct NoiseTrader {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: NoiseTraderConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Random number generator (Send-compatible).
    rng: StdRng,
    /// Per-agent bias on fair value (e.g., +0.10 means this agent thinks fair value is 10% higher).
    /// Creates heterogeneous beliefs across agents.
    fair_value_bias: f64,
}

impl NoiseTrader {
    /// Create a new NoiseTrader with the given configuration.
    pub fn new(id: AgentId, config: NoiseTraderConfig) -> Self {
        let initial_cash = config.initial_cash;
        let initial_position = config.initial_position;
        let mut state = AgentState::new(initial_cash, &[&config.symbol]);
        state.set_position(&config.symbol, initial_position);
        // Each agent gets a random bias on fair value: ±30%
        // This creates heterogeneous beliefs - some bulls, some bears
        let mut rng = StdRng::from_entropy();
        let fair_value_bias = rng.r#gen_range(-0.30..0.30);
        Self {
            id,
            config,
            state,
            rng,
            fair_value_bias,
        }
    }

    /// Create a new NoiseTrader with a specific seed (for reproducible testing).
    pub fn with_seed(id: AgentId, config: NoiseTraderConfig, seed: u64) -> Self {
        let initial_cash = config.initial_cash;
        let initial_position = config.initial_position;
        let mut state = AgentState::new(initial_cash, &[&config.symbol]);
        state.set_position(&config.symbol, initial_position);
        let mut rng = StdRng::seed_from_u64(seed);
        let fair_value_bias = rng.r#gen_range(-0.30..0.30);
        Self {
            id,
            config,
            state,
            rng,
            fair_value_bias,
        }
    }

    /// Create a NoiseTrader with default configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, NoiseTraderConfig::default())
    }

    /// Get current position for this trader's symbol.
    pub fn position(&self) -> i64 {
        self.state.position_for(&self.config.symbol)
    }

    /// Get current cash balance.
    pub fn cash(&self) -> Cash {
        self.state.cash()
    }

    /// Determine the reference price for order generation.
    ///
    /// Uses fair value with per-agent bias to create heterogeneous beliefs.
    /// Each agent has a random bias (±30%), so some think fair value is higher,
    /// others think it's lower. This creates natural two-sided order flow.
    ///
    /// Falls back to mid price or initial price if fair value unavailable.
    fn get_reference_price(&self, ctx: &StrategyContext<'_>) -> Price {
        if let Some(fair) = ctx.fair_value(&self.config.symbol) {
            // Apply this agent's personal bias to fair value
            let biased_fair = fair.to_float() * (1.0 + self.fair_value_bias);
            Price::from_float(biased_fair)
        } else {
            ctx.mid_price(&self.config.symbol)
                .or_else(|| ctx.last_price(&self.config.symbol))
                .unwrap_or(self.config.initial_price)
        }
    }

    /// Generate a random order around the reference price.
    /// Respects position limits - won't buy beyond max_long or sell beyond max_short.
    fn generate_order(&mut self, reference_price: Price) -> Option<Order> {
        // V3.1: Use per-symbol position, not aggregate
        let position = self.state.position_for(&self.config.symbol);

        // Check what actions are allowed based on position limits
        let can_buy = position < self.config.max_long_position - self.config.max_quantity as i64;
        let can_sell = position > -self.config.max_short_position + self.config.max_quantity as i64;

        // Determine side based on what's allowed
        let side = match (can_buy, can_sell) {
            (true, true) => {
                // Both allowed - flip skewed coin (54% sell bias)
                if self.rng.r#gen_bool(0.54) {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                }
            }
            (true, false) => OrderSide::Buy,
            (false, true) => OrderSide::Sell,
            (false, false) => return None, // At both limits, can't trade
        };

        // Random price within deviation range
        let deviation_range = self.config.price_deviation;
        let deviation = self
            .rng
            .r#gen_range((-deviation_range / 5.0)..deviation_range);
        let mut price_float = reference_price.to_float();
        if side == OrderSide::Buy {
            price_float *= 1.0 - deviation;
        } else {
            price_float *= 1.0 + deviation;
        }

        let price = Price::from_float(price_float.max(0.01)); // Ensure positive

        // Random quantity, capped by position limits
        let max_qty = if side == OrderSide::Sell {
            // For sells, cap at current position
            (position as u64).min(self.config.max_quantity)
        } else {
            // For buys, cap at remaining room before max_long
            let room = (self.config.max_long_position - position).max(0) as u64;
            room.min(self.config.max_quantity)
        };

        if max_qty < self.config.min_quantity {
            return None; // Not enough to trade
        }

        let quantity = Quantity(self.rng.r#gen_range(self.config.min_quantity..=max_qty));

        Some(Order::limit(
            self.id,
            &self.config.symbol,
            side,
            price,
            quantity,
        ))
    }
}

impl Agent for NoiseTrader {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Randomly decide whether to place an order
        if !self.rng.r#gen_bool(self.config.order_probability) {
            return AgentAction::none();
        }

        let reference_price = self.get_reference_price(ctx);

        if let Some(order) = self.generate_order(reference_price) {
            self.state.record_order();
            AgentAction::single(order)
        } else {
            AgentAction::none()
        }
    }

    fn on_fill(&mut self, trade: &Trade) {
        let trade_value = trade.value();

        // Use separate if blocks (not else if) to handle self-trades correctly.
        // When buyer_id == seller_id, both buy and sell must be applied (net zero).
        if trade.buyer_id == self.id {
            self.state
                .on_buy(&trade.symbol, trade.quantity.raw(), trade_value);
        }
        if trade.seller_id == self.id {
            self.state
                .on_sell(&trade.symbol, trade.quantity.raw(), trade_value);
        }
    }

    fn name(&self) -> &str {
        "NoiseTrader"
    }

    fn state(&self) -> &AgentState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quant::IndicatorSnapshot;
    use sim_core::SingleSymbolMarket;
    use std::collections::HashMap;
    use types::OrderId;

    fn mock_context_with_price(
        mid_price: Option<Price>,
    ) -> (
        sim_core::OrderBook,
        HashMap<types::Symbol, Vec<types::Candle>>,
        IndicatorSnapshot,
        HashMap<types::Symbol, Vec<Trade>>,
    ) {
        let mut book = sim_core::OrderBook::new("ACME");
        if let Some(mid) = mid_price {
            let mut bid = Order::limit(
                AgentId(99),
                "ACME",
                OrderSide::Buy,
                Price::from_float(mid.to_float() - 0.5),
                Quantity(100),
            );
            bid.id = OrderId(1);
            let mut ask = Order::limit(
                AgentId(99),
                "ACME",
                OrderSide::Sell,
                Price::from_float(mid.to_float() + 0.5),
                Quantity(100),
            );
            ask.id = OrderId(2);
            book.add_order(bid).unwrap();
            book.add_order(ask).unwrap();
        }
        let candles = HashMap::new();
        let indicators = IndicatorSnapshot::new(1);
        let recent_trades = HashMap::new();
        (book, candles, indicators, recent_trades)
    }

    #[test]
    fn test_noise_trader_creation() {
        let trader = NoiseTrader::with_defaults(AgentId(1));
        assert_eq!(trader.id(), AgentId(1));
        assert_eq!(trader.position(), 50); // Default initial_position
        assert_eq!(trader.cash(), Cash::from_float(100_000.0));
    }

    #[test]
    fn test_reference_price_priority() {
        let trader = NoiseTrader::with_defaults(AgentId(1));
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Empty market: use initial price
        let (book, candles, indicators, recent_trades) = mock_context_with_price(None);
        let market = SingleSymbolMarket::new(&book);
        let ctx = StrategyContext::new(
            1,
            0,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );
        let ref_price = trader.get_reference_price(&ctx);
        assert_eq!(ref_price, Price::from_float(100.0));

        // With mid price: use mid price
        let (book, candles, indicators, recent_trades) =
            mock_context_with_price(Some(Price::from_float(150.0)));
        let market = SingleSymbolMarket::new(&book);
        let ctx = StrategyContext::new(
            1,
            0,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );
        let ref_price = trader.get_reference_price(&ctx);
        assert_eq!(ref_price, Price::from_float(150.0));
    }

    #[test]
    fn test_on_fill_updates_state() {
        let mut trader = NoiseTrader::with_defaults(AgentId(1));

        // Simulate a buy fill
        let trade = Trade {
            id: types::TradeId(1),
            symbol: "ACME".to_string(),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: types::OrderId(1),
            seller_order_id: types::OrderId(2),
            price: Price::from_float(100.0),
            quantity: Quantity(10),
            timestamp: 0,
            tick: 1,
        };

        trader.on_fill(&trade);

        assert_eq!(trader.position(), 60); // 50 initial + 10 bought
        // Cash decreased by 10 * 100 = 1000
        assert_eq!(trader.cash(), Cash::from_float(99_000.0));
    }

    #[test]
    fn test_on_fill_sell_updates_state() {
        let mut trader = NoiseTrader::with_defaults(AgentId(1));

        // Simulate a sell fill
        let trade = Trade {
            id: types::TradeId(1),
            symbol: "ACME".to_string(),
            buyer_id: AgentId(2),
            seller_id: AgentId(1),
            buyer_order_id: types::OrderId(2),
            seller_order_id: types::OrderId(1),
            price: Price::from_float(100.0),
            quantity: Quantity(10),
            timestamp: 0,
            tick: 1,
        };

        trader.on_fill(&trade);

        assert_eq!(trader.position(), 40); // 50 initial - 10 sold
        // Cash increased by 10 * 100 = 1000
        assert_eq!(trader.cash(), Cash::from_float(101_000.0));
    }
}
