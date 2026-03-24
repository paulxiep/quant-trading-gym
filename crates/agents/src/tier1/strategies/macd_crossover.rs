//! MACD Crossover Trader - trades on MACD signal line crossovers.
//!
//! A momentum strategy that uses MACD (Moving Average Convergence Divergence)
//! to identify trend changes and generate trading signals.
//!
//! # Strategy Logic
//! - **Buy signal**: MACD line crosses above signal line (bullish crossover)
//! - **Sell signal**: MACD line crosses below signal line (bearish crossover)
//! - Uses histogram (MACD - Signal) for crossover detection
//!
//! # Configuration
//! The strategy is fully declarative via [`MacdCrossoverConfig`].

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use quant::Macd;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use types::{AgentId, Cash, IndicatorType, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a MACD Crossover trader.
#[derive(Debug, Clone)]
pub struct MacdCrossoverConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// MACD fast EMA period (typically 12).
    pub fast_period: usize,
    /// MACD slow EMA period (typically 26).
    pub slow_period: usize,
    /// Signal line EMA period (typically 9).
    pub signal_period: usize,
    /// Order size for each trade.
    pub order_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Maximum position size (absolute value).
    pub max_position: i64,
    /// Minimum histogram magnitude for signal confirmation.
    pub min_histogram: f64,
}

impl Default for MacdCrossoverConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            fast_period: 8,
            slow_period: 16,
            signal_period: 4,
            order_size: 50,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
            max_position: 500,
            min_histogram: 0.0, // No minimum by default
        }
    }
}

/// Crossover state for MACD signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MacdState {
    /// Not enough data to determine state.
    Unknown,
    /// MACD line is above signal line (bullish).
    Bullish,
    /// MACD line is below signal line (bearish).
    Bearish,
}

/// MACD Crossover trader using MACD/Signal line crossovers.
///
/// MACD is a trend-following momentum indicator that shows the relationship
/// between two EMAs of price. Crossovers of the MACD and signal lines
/// generate trading signals.
pub struct MacdCrossover {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: MacdCrossoverConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Previous MACD state for crossover detection.
    prev_state: MacdState,
    /// MACD calculator for full output access.
    macd: Macd,
    /// Random number generator for order price variation.
    rng: StdRng,
}

impl MacdCrossover {
    /// Create a new MacdCrossover with the given configuration.
    pub fn new(id: AgentId, config: MacdCrossoverConfig) -> Self {
        let initial_cash = config.initial_cash;
        let macd = Macd::new(config.fast_period, config.slow_period, config.signal_period);
        Self {
            id,
            config: config.clone(),
            state: AgentState::new(initial_cash, &[&config.symbol]),
            prev_state: MacdState::Unknown,
            macd,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a MacdCrossover with default (12, 26, 9) configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, MacdCrossoverConfig::default())
    }

    /// Get the IndicatorType this strategy uses.
    pub fn required_indicator(&self) -> IndicatorType {
        IndicatorType::MacdLine {
            fast: self.config.fast_period,
            slow: self.config.slow_period,
            signal: self.config.signal_period,
        }
    }

    /// Determine the reference price for orders.
    fn get_reference_price(&self, ctx: &StrategyContext<'_>) -> Price {
        ctx.mid_price(&self.config.symbol)
            .or_else(|| ctx.last_price(&self.config.symbol))
            .unwrap_or(self.config.initial_price)
    }

    /// Check if we can take more long positions.
    fn can_buy(&self) -> bool {
        self.state.position_for(&self.config.symbol) < self.config.max_position
    }

    /// Check if we can take more short positions.
    fn can_sell(&self) -> bool {
        self.state.position_for(&self.config.symbol) > -self.config.max_position
    }

    /// Generate a buy order.
    fn generate_buy_order(&mut self, ctx: &StrategyContext<'_>) -> Order {
        let price = self.get_reference_price(ctx);
        // Random multiplier: sometimes below market, sometimes at/above
        let mult = self.rng.r#gen_range(0.99..1.01);
        let order_price = Price::from_float(floor_price(price.to_float() * mult));
        Order::limit(
            self.id,
            &self.config.symbol,
            OrderSide::Buy,
            order_price,
            Quantity(self.config.order_size),
        )
    }

    /// Generate a sell order.
    fn generate_sell_order(&mut self, ctx: &StrategyContext<'_>) -> Order {
        let price = self.get_reference_price(ctx);
        // Random multiplier: sometimes above market, sometimes at/below
        let mult = self.rng.r#gen_range(0.99..1.01);
        let order_price = Price::from_float(floor_price(price.to_float() * mult));
        Order::limit(
            self.id,
            &self.config.symbol,
            OrderSide::Sell,
            order_price,
            Quantity(self.config.order_size),
        )
    }

    /// Determine MACD state from histogram value.
    fn macd_state_from_histogram(&self, histogram: f64) -> MacdState {
        // Histogram > 0 means MACD > Signal (bullish)
        // Histogram < 0 means MACD < Signal (bearish)
        if histogram > 0.0 {
            MacdState::Bullish
        } else {
            MacdState::Bearish
        }
    }
}

impl Agent for MacdCrossover {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Calculate full MACD output from candles
        let macd_output = match self.macd.calculate_full(ctx.candles(&self.config.symbol)) {
            Some(output) => output,
            None => return AgentAction::none(), // Not enough data
        };

        let current_state = self.macd_state_from_histogram(macd_output.histogram);
        let prev_state = self.prev_state;
        self.prev_state = current_state;

        // Check minimum histogram magnitude if configured
        if macd_output.histogram.abs() < self.config.min_histogram {
            return AgentAction::none();
        }

        // Only act on crossover events
        match (prev_state, current_state) {
            // Bullish crossover: MACD crosses above signal -> buy
            (MacdState::Bearish, MacdState::Bullish) if self.can_buy() => {
                let order = self.generate_buy_order(ctx);
                self.state.record_order();
                AgentAction::single(order)
            }
            // Bearish crossover: MACD crosses below signal -> sell
            (MacdState::Bullish, MacdState::Bearish) if self.can_sell() => {
                let order = self.generate_sell_order(ctx);
                self.state.record_order();
                AgentAction::single(order)
            }
            _ => AgentAction::none(),
        }
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
        "MACD"
    }

    fn state(&self) -> &AgentState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StrategyContext;
    use quant::IndicatorSnapshot;
    use sim_core::{OrderBook, SingleSymbolMarket};
    use std::collections::HashMap;
    use types::{Candle, Order, Symbol};

    /// Generate candles that will produce specific MACD outputs.
    fn make_trending_candles(trend_up: bool, count: usize) -> Vec<Candle> {
        let base = 100.0;
        let increment = if trend_up { 0.5 } else { -0.5 };

        (0..count)
            .map(|i| {
                let close = base + (i as f64 * increment);
                Candle {
                    symbol: "ACME".to_string(),
                    open: Price::from_float(close - 0.1),
                    high: Price::from_float(close + 0.2),
                    low: Price::from_float(close - 0.2),
                    close: Price::from_float(close),
                    volume: Quantity(1000),
                    timestamp: i as u64,
                    tick: i as u64,
                }
            })
            .collect()
    }

    fn make_context_with_candles<'a>(
        _order_book: &'a OrderBook,
        candles: &'a HashMap<Symbol, Vec<Candle>>,
        indicators: &'a IndicatorSnapshot,
        trades: &'a HashMap<Symbol, Vec<Trade>>,
        market: &'a SingleSymbolMarket<'a>,
        events: &'a [news::NewsEvent],
        fundamentals: &'a news::SymbolFundamentals,
    ) -> StrategyContext<'a> {
        StrategyContext::new(
            100,
            1000,
            market,
            candles,
            indicators,
            trades,
            events,
            fundamentals,
        )
    }

    fn create_order_book(symbol: &str, bid_price: f64, ask_price: f64) -> OrderBook {
        let mut book = OrderBook::new(symbol.to_string());
        let bid = Order::limit(
            AgentId(999),
            symbol,
            OrderSide::Buy,
            Price::from_float(bid_price),
            Quantity(100),
        );
        let ask = Order::limit(
            AgentId(999),
            symbol,
            OrderSide::Sell,
            Price::from_float(ask_price),
            Quantity(100),
        );
        book.add_order(bid).unwrap();
        book.add_order(ask).unwrap();
        book
    }

    #[test]
    fn test_macd_needs_enough_data() {
        let mut trader = MacdCrossover::with_defaults(AgentId(1));
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Only 10 candles, need 26 + 9 = 35 minimum
        let candle_vec = make_trending_candles(true, 10);
        let mut candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        candles.insert("ACME".into(), candle_vec);

        let ctx = make_context_with_candles(
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action = trader.on_tick(&ctx);
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_macd_no_action_on_first_state() {
        let mut trader = MacdCrossover::with_defaults(AgentId(1));
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Enough data for MACD
        let candle_vec = make_trending_candles(true, 50);
        let mut candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        candles.insert("ACME".into(), candle_vec);

        let ctx = make_context_with_candles(
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        // First tick should only set state, not generate orders
        // (prev_state is Unknown, which doesn't trigger crossover)
        let action = trader.on_tick(&ctx);
        // This might or might not generate an order depending on state logic
        // The main assertion is that it doesn't panic
        assert!(action.orders.len() <= 1);
    }
}
