//! Market Maker - provides liquidity with bid/ask spread.
//!
//! A market maker continuously quotes two-sided markets by placing
//! both bid and ask orders. It seeds liquidity and helps prevent
//! the "zombie simulation" problem where no trades occur.
//!
//! # Strategy
//! - Anchors quotes to fair value when fundamentals are available
//! - Falls back to mid price or initial price otherwise  
//! - Adjusts quotes based on inventory (skew away from large positions)
//! - Cancels stale orders when prices move significantly
//!
//! # V2.4 Fair Value Integration
//! When fundamentals are configured, quotes anchor to the Gordon Growth Model
//! fair value rather than market mid price. This creates price discovery
//! pressure toward intrinsic value.

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use types::{AgentId, Cash, Order, OrderId, OrderSide, Price, Quantity, Trade};

/// Configuration for a MarketMaker agent.
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Half-spread as a fraction of mid price (e.g., 0.005 = 0.5%).
    pub half_spread: f64,
    /// Order size to quote on each side.
    pub quote_size: u64,
    /// Initial price to seed the market (used when book is empty).
    pub initial_price: Price,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial share position (from the float).
    /// MarketMakers start with inventory to provide liquidity.
    pub initial_position: i64,
    /// Maximum inventory before skewing quotes (in shares).
    pub max_inventory: i64,
    /// Inventory skew factor (how much to adjust price per unit of inventory).
    pub inventory_skew: f64,
    /// Ticks between quote refreshes.
    pub refresh_interval: u64,
    /// Weight given to fair value vs mid price (0.0 = pure mid, 1.0 = pure fair value).
    /// A blend allows MMs to provide liquidity at market prices while having
    /// some pull toward fundamentals. Default 0.3 = 30% fair value, 70% mid.
    pub fair_value_weight: f64,
    /// Maximum long position (stops buying when reached).
    pub max_long_position: i64,
    /// Maximum short position as positive number (stops selling when reached).
    pub max_short_position: i64,
}

impl Default for MarketMakerConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            half_spread: 0.005, // 0.5% half spread = 1% total spread
            quote_size: 100,
            initial_price: Price::from_float(100.0),
            initial_cash: Cash::from_float(1_000_000.0),
            initial_position: 500, // Start with inventory from the float
            max_inventory: 1000,
            inventory_skew: 0.0001, // Adjust price 0.01% per share of inventory
            refresh_interval: 1,    // Quote every tick (required for IOC mode)
            fair_value_weight: 0.3, // 30% fair value, 70% mid price
            max_long_position: 1000,
            max_short_position: 0, // MMs shouldn't go short by default
        }
    }
}

/// A market maker that provides liquidity.
///
/// MarketMakers are essential for a functioning market. They seed the
/// order book with initial quotes and continuously provide two-sided
/// liquidity, enabling price discovery and trade execution.
pub struct MarketMaker {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: MarketMakerConfig,
    /// Shared agent state (position, cash, metrics).
    state: AgentState,
    /// Last tick when quotes were placed.
    last_quote_tick: u64,
    /// Total trading volume.
    total_volume: u64,
    /// Outstanding bid order ID (to cancel on refresh).
    outstanding_bid: Option<OrderId>,
    /// Outstanding ask order ID (to cancel on refresh).
    outstanding_ask: Option<OrderId>,
    /// Per-agent bias on fair value (e.g., +0.15 means this MM thinks fair value is 15% higher).
    /// Creates heterogeneous beliefs across market makers.
    fair_value_bias: f64,
}

impl MarketMaker {
    /// Create a new MarketMaker with the given configuration.
    pub fn new(id: AgentId, config: MarketMakerConfig) -> Self {
        let initial_cash = config.initial_cash;
        let initial_position = config.initial_position;
        let mut state = AgentState::new(initial_cash, &[&config.symbol]);
        // MarketMakers start with inventory from the float
        state.set_position(&config.symbol, initial_position);
        // Each MM gets a random bias on fair value: ±20%
        // This creates heterogeneous beliefs across market makers
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::from_entropy();
        let fair_value_bias = rng.r#gen_range(-0.20..0.20);
        Self {
            id,
            config,
            state,
            last_quote_tick: 0,
            total_volume: 0,
            outstanding_bid: None,
            outstanding_ask: None,
            fair_value_bias,
        }
    }

    /// Create a MarketMaker with default configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, MarketMakerConfig::default())
    }

    /// Get current position for this market maker's symbol.
    pub fn position(&self) -> i64 {
        self.state.position_for(&self.config.symbol)
    }

    /// Get current cash balance.
    pub fn cash(&self) -> Cash {
        self.state.cash()
    }

    /// Determine the reference price for quoting.
    ///
    /// Uses a blend of market price and biased fair value:
    /// - Market price (mid or last) ensures quotes are near tradeable levels
    /// - Fair value (with per-agent bias) creates heterogeneous fundamental views
    ///
    /// The blend ratio is controlled by `fair_value_weight` config.
    fn get_reference_price(&self, ctx: &StrategyContext<'_>) -> Price {
        let market_price = ctx
            .mid_price(&self.config.symbol)
            .or_else(|| ctx.last_price(&self.config.symbol))
            .unwrap_or(self.config.initial_price);

        // If fair value available and weight > 0, blend with biased fair value
        if let Some(fair) = ctx.fair_value(&self.config.symbol) {
            let w = self.config.fair_value_weight.clamp(0.0, 1.0);
            // Apply this MM's personal bias to fair value
            let biased_fair = fair.to_float() * (1.0 + self.fair_value_bias);
            let blended = market_price.to_float() * (1.0 - w) + biased_fair * w;
            Price::from_float(blended)
        } else {
            market_price
        }
    }

    /// Calculate inventory-adjusted skew.
    ///
    /// When we have positive inventory (long), we want to:
    /// - Lower our ask to sell more easily
    /// - Lower our bid to reduce buying
    ///
    /// When we have negative inventory (short), we do the opposite.
    fn calculate_skew(&self) -> f64 {
        // Clamp inventory to avoid extreme skews
        let clamped_inventory = self
            .state
            .position()
            .clamp(-self.config.max_inventory, self.config.max_inventory);

        // Negative skew means lower prices (to sell inventory)
        // Positive skew means higher prices (to buy back)
        -self.config.inventory_skew * clamped_inventory as f64
    }

    /// Generate bid and ask orders around reference price.
    /// Respects position limits - skips bid if at max long, skips ask if at max short.
    fn generate_quotes(&self, reference_price: Price) -> Vec<Order> {
        let ref_float = reference_price.to_float();
        let half_spread = self.config.half_spread;
        let skew = self.calculate_skew();

        // Calculate bid and ask prices with inventory skew
        // Apply floor_price to prevent negative price spirals
        let bid_price = Price::from_float(floor_price(ref_float * (1.0 - half_spread + skew)));
        let ask_price = Price::from_float(floor_price(ref_float * (1.0 + half_spread + skew)));

        let quote_size = Quantity(self.config.quote_size);
        let position = self.state.position();

        let mut orders = Vec::with_capacity(2);

        // Only quote bid (buy) if not at max long position
        if position < self.config.max_long_position - self.config.quote_size as i64 {
            orders.push(Order::limit(
                self.id,
                &self.config.symbol,
                OrderSide::Buy,
                bid_price,
                quote_size,
            ));
        }

        // Only quote ask (sell) if not at max short position
        if position > -self.config.max_short_position + self.config.quote_size as i64 {
            orders.push(Order::limit(
                self.id,
                &self.config.symbol,
                OrderSide::Sell,
                ask_price,
                quote_size,
            ));
        }

        orders
    }

    /// Check if we should refresh quotes this tick.
    fn should_refresh(&self, current_tick: u64) -> bool {
        current_tick == 0 || current_tick >= self.last_quote_tick + self.config.refresh_interval
    }
}

impl Agent for MarketMaker {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Only refresh quotes periodically
        if !self.should_refresh(ctx.tick) {
            return AgentAction::none();
        }

        let reference_price = self.get_reference_price(ctx);
        let orders = self.generate_quotes(reference_price);

        self.state.record_orders(orders.len() as u64);
        self.last_quote_tick = ctx.tick;

        // Collect any outstanding orders to cancel
        let mut cancellations = Vec::new();
        if let Some(bid_id) = self.outstanding_bid.take() {
            cancellations.push(bid_id);
        }
        if let Some(ask_id) = self.outstanding_ask.take() {
            cancellations.push(ask_id);
        }

        if cancellations.is_empty() {
            AgentAction::multiple(orders)
        } else {
            AgentAction::cancel_and_replace(cancellations, orders)
        }
    }

    fn on_fill(&mut self, trade: &Trade) {
        self.total_volume += trade.quantity.raw();

        // Use separate if blocks (not else if) to handle self-trades correctly.
        // When buyer_id == seller_id, both buy and sell must be applied (net zero).
        if trade.buyer_id == self.id {
            self.state
                .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
        }
        if trade.seller_id == self.id {
            self.state
                .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
        }
        // Note: We don't clear outstanding_bid/ask here because partial fills
        // leave the remainder on the book with the same OrderId. The cancellation
        // logic handles cleanup - if fully filled, cancel is a harmless no-op.
    }

    fn on_order_resting(&mut self, order_id: OrderId, order: &types::Order) {
        // Track resting orders so we can cancel them later
        match order.side {
            OrderSide::Buy => self.outstanding_bid = Some(order_id),
            OrderSide::Sell => self.outstanding_ask = Some(order_id),
        }
    }

    fn name(&self) -> &str {
        "MarketMaker"
    }

    fn is_market_maker(&self) -> bool {
        true
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

    fn mock_context(
        mid_price: Option<Price>,
        tick: u64,
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
        let indicators = IndicatorSnapshot::new(tick);
        let recent_trades = HashMap::new();
        (book, candles, indicators, recent_trades)
    }

    #[test]
    fn test_market_maker_creation() {
        let mm = MarketMaker::with_defaults(AgentId(1));
        assert_eq!(mm.id(), AgentId(1));
        assert_eq!(mm.position(), 500); // Default initial_position
        assert_eq!(mm.cash(), Cash::from_float(1_000_000.0));
    }

    #[test]
    fn test_market_maker_generates_two_sided_quotes() {
        let config = MarketMakerConfig {
            initial_position: 0,     // Start neutral for this test
            max_short_position: 200, // Allow selling (going short) so ask is quoted
            ..Default::default()
        };
        let mut mm = MarketMaker::new(AgentId(1), config);

        let (book, candles, indicators, recent_trades) =
            mock_context(Some(Price::from_float(100.0)), 0);
        let market = SingleSymbolMarket::new(&book);
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();
        let ctx = StrategyContext::new(
            0,
            0,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );
        let action = mm.on_tick(&ctx);

        assert_eq!(action.orders.len(), 2);

        // Find bid and ask
        let bid = action.orders.iter().find(|o| o.side == OrderSide::Buy);
        let ask = action.orders.iter().find(|o| o.side == OrderSide::Sell);

        assert!(bid.is_some());
        assert!(ask.is_some());

        // Bid should be below reference, ask should be above
        let bid_price = bid.unwrap().limit_price().unwrap();
        let ask_price = ask.unwrap().limit_price().unwrap();

        assert!(bid_price < Price::from_float(100.0));
        assert!(ask_price > Price::from_float(100.0));
    }

    #[test]
    fn test_market_maker_respects_refresh_interval() {
        let config = MarketMakerConfig {
            refresh_interval: 10,
            ..Default::default()
        };
        let mut mm = MarketMaker::new(AgentId(1), config);
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Tick 0: should place orders
        let (book, candles, indicators, recent_trades) =
            mock_context(Some(Price::from_float(100.0)), 0);
        let market = SingleSymbolMarket::new(&book);
        let ctx = StrategyContext::new(
            0,
            0,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );
        let action = mm.on_tick(&ctx);
        assert_eq!(action.orders.len(), 2);

        // Tick 5: should NOT place orders
        let (book, candles, indicators, recent_trades) =
            mock_context(Some(Price::from_float(100.0)), 5);
        let market = SingleSymbolMarket::new(&book);
        let ctx = StrategyContext::new(
            5,
            0,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );
        let action = mm.on_tick(&ctx);
        assert!(action.orders.is_empty());

        // Tick 10: should place orders again
        let (book, candles, indicators, recent_trades) =
            mock_context(Some(Price::from_float(100.0)), 10);
        let market = SingleSymbolMarket::new(&book);
        let ctx = StrategyContext::new(
            10,
            0,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );
        let action = mm.on_tick(&ctx);
        assert_eq!(action.orders.len(), 2);
    }

    #[test]
    fn test_inventory_skew() {
        let mut mm = MarketMaker::with_defaults(AgentId(1));

        // Simulate large long position
        mm.state.set_position(&"ACME".to_string(), 500);

        // Skew should be negative (lower prices to sell)
        let skew = mm.calculate_skew();
        assert!(skew < 0.0);

        // Simulate large short position
        mm.state.set_position(&"ACME".to_string(), -500);

        // Skew should be positive (higher prices to buy)
        let skew = mm.calculate_skew();
        assert!(skew > 0.0);
    }

    #[test]
    fn test_on_fill_updates_state() {
        let config = MarketMakerConfig {
            initial_position: 0, // Start at 0 for this test
            ..Default::default()
        };
        let mut mm = MarketMaker::new(AgentId(1), config);

        // Simulate a buy fill
        let trade = Trade {
            id: types::TradeId(1),
            symbol: "ACME".to_string(),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: types::OrderId(1),
            seller_order_id: types::OrderId(2),
            price: Price::from_float(100.0),
            quantity: Quantity(50),
            timestamp: 0,
            tick: 1,
        };

        mm.on_fill(&trade);

        assert_eq!(mm.position(), 50);
        assert_eq!(mm.total_volume, 50);
        // Cash decreased by 50 * 100 = 5000
        assert_eq!(mm.cash(), Cash::from_float(995_000.0));
    }
}
