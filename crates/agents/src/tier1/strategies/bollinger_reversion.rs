//! Bollinger Bands Mean Reversion Trader.
//!
//! A mean reversion strategy that uses Bollinger Bands to identify when
//! price has moved too far from its mean and is likely to revert.
//!
//! # Strategy Logic
//! - **Buy signal**: Price touches or crosses below the lower band (oversold)
//! - **Sell signal**: Price touches or crosses above the upper band (overbought)
//! - Uses %B indicator: (Price - Lower) / (Upper - Lower)
//!   - %B < 0: Below lower band -> buy
//!   - %B > 1: Above upper band -> sell
//!
//! # Configuration
//! The strategy is fully declarative via [`BollingerReversionConfig`].

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use quant::BollingerBands;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use types::{AgentId, Cash, IndicatorType, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a Bollinger Bands Reversion trader.
#[derive(Debug, Clone)]
pub struct BollingerReversionConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Bollinger Bands period (typically 20).
    pub period: usize,
    /// Standard deviation multiplier (typically 2.0).
    pub std_dev_multiplier: f64,
    /// Order size for each trade.
    pub order_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Maximum position size (absolute value).
    pub max_position: i64,
    /// Lower %B threshold for buy signal (typically 0.0 or slightly above).
    pub lower_threshold: f64,
    /// Upper %B threshold for sell signal (typically 1.0 or slightly below).
    pub upper_threshold: f64,
}

impl Default for BollingerReversionConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            period: 12,
            std_dev_multiplier: 2.0,
            order_size: 50,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
            max_position: 500,
            lower_threshold: 0.0, // Buy when at or below lower band
            upper_threshold: 1.0, // Sell when at or above upper band
        }
    }
}

/// Bollinger Bands Mean Reversion trader.
///
/// This strategy is based on the statistical concept that prices tend to
/// return to their mean. When price moves beyond the bands (2 std devs
/// by default), it's considered overextended and likely to revert.
pub struct BollingerReversion {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: BollingerReversionConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Bollinger Bands calculator.
    bollinger: BollingerBands,
    /// Random number generator for order price variation.
    rng: StdRng,
}

impl BollingerReversion {
    /// Create a new BollingerReversion with the given configuration.
    pub fn new(id: AgentId, config: BollingerReversionConfig) -> Self {
        let initial_cash = config.initial_cash;
        let bollinger = BollingerBands::new(config.period, config.std_dev_multiplier);
        Self {
            id,
            config: config.clone(),
            state: AgentState::new(initial_cash, &[&config.symbol]),
            bollinger,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a BollingerReversion with default (20, 2.0) configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, BollingerReversionConfig::default())
    }

    /// Get the IndicatorType this strategy uses.
    pub fn required_indicator(&self) -> IndicatorType {
        IndicatorType::BollingerMiddle {
            period: self.config.period,
            std_dev_bp: (self.config.std_dev_multiplier * 100.0) as u32,
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
}

impl Agent for BollingerReversion {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Calculate full Bollinger Bands output from candles
        let bb_output = match self
            .bollinger
            .calculate_full(ctx.candles(&self.config.symbol))
        {
            Some(output) => output,
            None => return AgentAction::none(), // Not enough data
        };

        // %B indicates where current price is relative to the bands
        // %B < lower_threshold: price at/below lower band -> buy
        // %B > upper_threshold: price at/above upper band -> sell
        let percent_b = bb_output.percent_b;

        if percent_b <= self.config.lower_threshold && self.can_buy() {
            let order = self.generate_buy_order(ctx);
            self.state.record_order();
            return AgentAction::single(order);
        }

        if percent_b >= self.config.upper_threshold && self.can_sell() {
            let order = self.generate_sell_order(ctx);
            self.state.record_order();
            return AgentAction::single(order);
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
        "Bollinger"
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

    /// Generate candles with high volatility that will push price beyond bands.
    fn make_volatile_candles(spike_up: bool, count: usize) -> Vec<Candle> {
        let base = 100.0;
        let mut candles = Vec::with_capacity(count);

        for i in 0..count {
            // Create mostly stable prices, then a spike at the end
            let close = if i == count - 1 {
                if spike_up {
                    base + 10.0 // 10% spike up (beyond 2 std devs)
                } else {
                    base - 10.0 // 10% spike down
                }
            } else {
                // Add small random-ish variation to make bands meaningful
                base + ((i % 3) as f64 - 1.0) * 0.5
            };

            candles.push(Candle {
                symbol: "ACME".to_string(),
                open: Price::from_float(close - 0.1),
                high: Price::from_float(close + 0.2),
                low: Price::from_float(close - 0.2),
                close: Price::from_float(close),
                volume: Quantity(1000),
                timestamp: i as u64,
                tick: i as u64,
            });
        }

        candles
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
    fn test_bollinger_needs_enough_data() {
        let mut trader = BollingerReversion::with_defaults(AgentId(1));
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Only 10 candles, need 20 minimum
        let candle_vec = make_volatile_candles(true, 10);
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
    fn test_bollinger_buys_on_lower_band() {
        let mut trader = BollingerReversion::with_defaults(AgentId(1));
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Price spikes down beyond lower band
        let candle_vec = make_volatile_candles(false, 25);
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
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Buy);
    }

    #[test]
    fn test_bollinger_sells_on_upper_band() {
        let mut trader = BollingerReversion::with_defaults(AgentId(1));
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Price spikes up beyond upper band
        let candle_vec = make_volatile_candles(true, 25);
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
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Sell);
    }
}
