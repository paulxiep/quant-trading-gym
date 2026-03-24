//! Trend Following Trader - follows SMA crossover signals.
//!
//! A classic trend-following strategy that uses two Simple Moving Averages
//! (fast and slow) to identify trend direction and generate trading signals.
//!
//! # Strategy Logic
//! - **Buy signal**: Fast SMA crosses above Slow SMA (golden cross)
//! - **Sell signal**: Fast SMA crosses below Slow SMA (death cross)
//! - Tracks crossover state to avoid repeated signals
//!
//! # Configuration
//! The strategy is fully declarative via [`TrendFollowerConfig`].

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use types::{AgentId, Cash, IndicatorType, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a Trend Following trader.
#[derive(Debug, Clone)]
pub struct TrendFollowerConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Fast SMA period (shorter period, more responsive).
    pub fast_period: usize,
    /// Slow SMA period (longer period, smoother).
    pub slow_period: usize,
    /// Order size for each trade.
    pub order_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Maximum position size (absolute value).
    pub max_position: i64,
}

impl Default for TrendFollowerConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            fast_period: 8,
            slow_period: 16,
            order_size: 50,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
            max_position: 500,
        }
    }
}

/// Crossover state to track signal changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossoverState {
    /// Not enough data to determine state.
    Unknown,
    /// Fast SMA is above slow SMA (bullish).
    Above,
    /// Fast SMA is below slow SMA (bearish).
    Below,
}

/// Trend Following trader using SMA crossover strategy.
///
/// This is a classic trend-following approach that aims to capture
/// sustained price movements by identifying trend direction through
/// moving average relationships.
pub struct TrendFollower {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: TrendFollowerConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Previous crossover state for signal detection.
    prev_state: CrossoverState,
    /// Random number generator for order price variation.
    rng: StdRng,
}

impl TrendFollower {
    /// Create a new TrendFollower with the given configuration.
    pub fn new(id: AgentId, config: TrendFollowerConfig) -> Self {
        assert!(
            config.fast_period < config.slow_period,
            "Fast period must be less than slow period"
        );
        let initial_cash = config.initial_cash;
        Self {
            id,
            config: config.clone(),
            state: AgentState::new(initial_cash, &[&config.symbol]),
            prev_state: CrossoverState::Unknown,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a TrendFollower with default configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, TrendFollowerConfig::default())
    }

    /// Get the IndicatorTypes this strategy requires.
    pub fn required_indicators(&self) -> Vec<IndicatorType> {
        vec![
            IndicatorType::Sma(self.config.fast_period),
            IndicatorType::Sma(self.config.slow_period),
        ]
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

    /// Determine the current crossover state.
    fn current_crossover_state(&self, fast_sma: f64, slow_sma: f64) -> CrossoverState {
        if fast_sma > slow_sma {
            CrossoverState::Above
        } else {
            CrossoverState::Below
        }
    }
}

impl Agent for TrendFollower {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Get both SMAs from pre-computed indicators
        let fast_sma = match ctx.get_indicator(
            &self.config.symbol,
            IndicatorType::Sma(self.config.fast_period),
        ) {
            Some(v) => v,
            None => return AgentAction::none(),
        };

        let slow_sma = match ctx.get_indicator(
            &self.config.symbol,
            IndicatorType::Sma(self.config.slow_period),
        ) {
            Some(v) => v,
            None => return AgentAction::none(),
        };

        let current_state = self.current_crossover_state(fast_sma, slow_sma);
        let prev_state = self.prev_state;
        self.prev_state = current_state;

        // Only act on crossover events (state changes)
        match (prev_state, current_state) {
            // Golden cross: fast crosses above slow -> buy
            (CrossoverState::Below, CrossoverState::Above) if self.can_buy() => {
                let order = self.generate_buy_order(ctx);
                self.state.record_order();
                AgentAction::single(order)
            }
            // Death cross: fast crosses below slow -> sell
            (CrossoverState::Above, CrossoverState::Below) if self.can_sell() => {
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
        "TrendFollow"
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

    fn make_context_with_indicators<'a>(
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

    fn make_indicators(
        config: &TrendFollowerConfig,
        fast_sma: Option<f64>,
        slow_sma: Option<f64>,
    ) -> IndicatorSnapshot {
        match (fast_sma, slow_sma) {
            (Some(fast), Some(slow)) => {
                let mut snap = IndicatorSnapshot::new(100);
                let mut indicators = HashMap::new();
                indicators.insert(IndicatorType::Sma(config.fast_period), fast);
                indicators.insert(IndicatorType::Sma(config.slow_period), slow);
                snap.insert(config.symbol.clone(), indicators);
                snap
            }
            _ => IndicatorSnapshot::default(),
        }
    }

    #[test]
    fn test_trend_golden_cross_buys() {
        let config = TrendFollowerConfig::default();
        let mut trader = TrendFollower::new(AgentId(1), config.clone());
        let order_book = create_order_book(&config.symbol, 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // First tick: fast below slow (set prev_state)
        let indicators1 = make_indicators(&config, Some(49.0), Some(50.0));
        let ctx1 = make_context_with_indicators(
            &order_book,
            &candles,
            &indicators1,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let _ = trader.on_tick(&ctx1);

        // Second tick: fast above slow (golden cross!)
        let indicators2 = make_indicators(&config, Some(51.0), Some(50.0));
        let ctx2 = make_context_with_indicators(
            &order_book,
            &candles,
            &indicators2,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action = trader.on_tick(&ctx2);

        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Buy);
    }

    #[test]
    fn test_trend_death_cross_sells() {
        let config = TrendFollowerConfig::default();
        let mut trader = TrendFollower::new(AgentId(1), config.clone());
        let order_book = create_order_book(&config.symbol, 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // First tick: fast above slow
        let indicators1 = make_indicators(&config, Some(51.0), Some(50.0));
        let ctx1 = make_context_with_indicators(
            &order_book,
            &candles,
            &indicators1,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let _ = trader.on_tick(&ctx1);

        // Second tick: fast below slow (death cross!)
        let indicators2 = make_indicators(&config, Some(49.0), Some(50.0));
        let ctx2 = make_context_with_indicators(
            &order_book,
            &candles,
            &indicators2,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action = trader.on_tick(&ctx2);

        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Sell);
    }

    #[test]
    fn test_trend_no_action_without_crossover() {
        let config = TrendFollowerConfig::default();
        let mut trader = TrendFollower::new(AgentId(1), config.clone());
        let order_book = create_order_book(&config.symbol, 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Both ticks: fast above slow (no crossover)
        let indicators1 = make_indicators(&config, Some(51.0), Some(50.0));
        let ctx1 = make_context_with_indicators(
            &order_book,
            &candles,
            &indicators1,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let _ = trader.on_tick(&ctx1);

        let indicators2 = make_indicators(&config, Some(52.0), Some(50.0));
        let ctx2 = make_context_with_indicators(
            &order_book,
            &candles,
            &indicators2,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action = trader.on_tick(&ctx2);

        assert!(action.orders.is_empty());
    }
}
