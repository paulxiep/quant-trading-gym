//! VWAP Executor - executes orders targeting Volume Weighted Average Price.
//!
//! A trade execution algorithm that aims to match or beat the VWAP benchmark
//! by spreading order execution across time periods.
//!
//! # Strategy Logic
//! - Calculates current VWAP from trade history
//! - For buy orders: executes when price < VWAP (favorable)
//! - For sell orders: executes when price > VWAP (favorable)
//! - Slices large orders into smaller chunks to minimize market impact
//!
//! # Use Case
//! VWAP execution is commonly used by institutional traders who need to
//! execute large orders without significant market impact. The algorithm
//! ensures average execution price is close to VWAP.
//!
//! # Configuration
//! The strategy is fully declarative via [`VwapExecutorConfig`].

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a VWAP Executor.
#[derive(Debug, Clone)]
pub struct VwapExecutorConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Total quantity to execute.
    pub target_quantity: u64,
    /// Order side (Buy or Sell).
    pub side: OrderSide,
    /// Maximum order size per slice.
    pub max_slice_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Price tolerance: how much above/below VWAP to accept (as fraction).
    pub price_tolerance: f64,
    /// Minimum ticks between order slices.
    pub slice_interval: u64,
}

impl Default for VwapExecutorConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            target_quantity: 1000,
            side: OrderSide::Buy,
            max_slice_size: 100,
            initial_cash: Cash::from_float(1_000_000.0),
            initial_price: Price::from_float(100.0),
            price_tolerance: 0.002, // 0.2% tolerance
            slice_interval: 10,     // Wait 10 ticks between slices
        }
    }
}

/// VWAP Executor that executes orders at favorable prices relative to VWAP.
///
/// VWAP (Volume Weighted Average Price) is a benchmark that represents
/// the average price weighted by volume. Executing at or better than
/// VWAP is a common institutional trading goal.
pub struct VwapExecutor {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: VwapExecutorConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Remaining quantity to execute.
    remaining_quantity: u64,
    /// Last tick when we placed an order.
    last_order_tick: u64,
    /// Running VWAP calculation: total value traded.
    total_value: f64,
    /// Running VWAP calculation: total volume traded.
    total_volume: u64,
}

impl VwapExecutor {
    /// Create a new VwapExecutor with the given configuration.
    pub fn new(id: AgentId, config: VwapExecutorConfig) -> Self {
        let initial_cash = config.initial_cash;
        let remaining = config.target_quantity;
        Self {
            id,
            config: config.clone(),
            state: AgentState::new(initial_cash, &[&config.symbol]),
            remaining_quantity: remaining,
            last_order_tick: 0,
            total_value: 0.0,
            total_volume: 0,
        }
    }

    /// Create a VwapExecutor with default buy configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, VwapExecutorConfig::default())
    }

    /// Create a VWAP Executor for selling.
    pub fn new_seller(id: AgentId, symbol: &str, target_quantity: u64) -> Self {
        let config = VwapExecutorConfig {
            symbol: symbol.to_string(),
            target_quantity,
            side: OrderSide::Sell,
            ..Default::default()
        };
        Self::new(id, config)
    }

    /// Check if execution is complete.
    pub fn is_complete(&self) -> bool {
        self.remaining_quantity == 0
    }

    /// Get remaining quantity to execute.
    pub fn remaining(&self) -> u64 {
        self.remaining_quantity
    }

    /// Calculate current VWAP from recent trades.
    fn calculate_vwap(&self, recent_trades: &[Trade]) -> Option<f64> {
        if recent_trades.is_empty() && self.total_volume == 0 {
            return None;
        }

        // Combine historical total with recent trades
        let (total_value, total_volume) = recent_trades.iter().fold(
            (self.total_value, self.total_volume),
            |(value, volume), trade| {
                let trade_value = trade.price.to_float() * trade.quantity.raw() as f64;
                (value + trade_value, volume + trade.quantity.raw())
            },
        );

        if total_volume == 0 {
            None
        } else {
            Some(total_value / total_volume as f64)
        }
    }

    /// Update running VWAP with new trades.
    fn update_vwap(&mut self, recent_trades: &[Trade]) {
        let (delta_value, delta_volume) =
            recent_trades
                .iter()
                .fold((0.0, 0u64), |(value, volume), trade| {
                    let trade_value = trade.price.to_float() * trade.quantity.raw() as f64;
                    (value + trade_value, volume + trade.quantity.raw())
                });
        self.total_value += delta_value;
        self.total_volume += delta_volume;
    }

    /// Determine the reference price for orders.
    fn get_reference_price(&self, ctx: &StrategyContext<'_>) -> Price {
        ctx.mid_price(&self.config.symbol)
            .or_else(|| ctx.last_price(&self.config.symbol))
            .unwrap_or(self.config.initial_price)
    }

    /// Check if current price is favorable relative to VWAP.
    fn is_price_favorable(&self, current_price: f64, vwap: f64) -> bool {
        match self.config.side {
            // For buying: favorable when price is at or below VWAP (+ tolerance)
            OrderSide::Buy => current_price <= vwap * (1.0 - self.config.price_tolerance),
            // For selling: favorable when price is at or above VWAP (- tolerance)
            OrderSide::Sell => current_price >= vwap * (1.0 + self.config.price_tolerance),
        }
    }

    /// Calculate order size for this slice.
    fn calculate_slice_size(&self) -> u64 {
        self.remaining_quantity.min(self.config.max_slice_size)
    }

    /// Check if enough time has passed since last order.
    fn can_place_order(&self, current_tick: u64) -> bool {
        current_tick == 0 || current_tick >= self.last_order_tick + self.config.slice_interval
    }

    /// Generate an order at the reference price.
    fn generate_order(&self, ctx: &StrategyContext<'_>) -> Order {
        let price = self.get_reference_price(ctx);
        let quantity = Quantity(self.calculate_slice_size());

        // Price adjustment based on side to improve fill probability
        // Apply floor_price to prevent negative price spirals
        let order_price = match self.config.side {
            OrderSide::Buy => Price::from_float(floor_price(price.to_float() * 1.001)),
            OrderSide::Sell => Price::from_float(floor_price(price.to_float() * 0.999)),
        };

        Order::limit(
            self.id,
            &self.config.symbol,
            self.config.side,
            order_price,
            quantity,
        )
    }
}

impl Agent for VwapExecutor {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Update running VWAP calculation from recent trades
        let recent_trades = ctx.recent_trades(&self.config.symbol);
        self.update_vwap(recent_trades);

        // Check if we're done
        if self.is_complete() {
            return AgentAction::none();
        }

        // Check timing constraint
        if !self.can_place_order(ctx.tick) {
            return AgentAction::none();
        }

        // Get current price - try last_price first, then mid_price
        let current_price = match ctx
            .last_price(&self.config.symbol)
            .or_else(|| ctx.mid_price(&self.config.symbol))
        {
            Some(p) => p.to_float(),
            None => return AgentAction::none(),
        };

        // Calculate VWAP and check if price is favorable
        if let Some(vwap) = self.calculate_vwap(&[]) {
            // Don't include recent_trades again, already in total
            if !self.is_price_favorable(current_price, vwap) {
                return AgentAction::none();
            }
        }
        // If no VWAP yet (no trades), execute to get started

        let order = self.generate_order(ctx);
        self.state.record_order();
        self.last_order_tick = ctx.tick;

        AgentAction::single(order)
    }

    fn on_fill(&mut self, trade: &Trade) {
        let filled_qty = trade.quantity.raw();

        // Update remaining quantity
        self.remaining_quantity = self.remaining_quantity.saturating_sub(filled_qty);

        // Use separate if blocks (not else if) to handle self-trades correctly.
        if trade.buyer_id == self.id {
            self.state.on_buy(&trade.symbol, filled_qty, trade.value());
        }
        if trade.seller_id == self.id {
            self.state.on_sell(&trade.symbol, filled_qty, trade.value());
        }
    }

    fn name(&self) -> &str {
        "VWAP"
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

    fn make_context_with_orderbook<'a>(
        tick: u64,
        _order_book: &'a OrderBook,
        candles: &'a HashMap<Symbol, Vec<Candle>>,
        indicators: &'a IndicatorSnapshot,
        trades: &'a HashMap<Symbol, Vec<Trade>>,
        market: &'a SingleSymbolMarket<'a>,
        events: &'a [news::NewsEvent],
        fundamentals: &'a news::SymbolFundamentals,
    ) -> StrategyContext<'a> {
        StrategyContext::new(
            tick,
            tick * 100,
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
    fn test_vwap_initial_order() {
        let mut executor = VwapExecutor::with_defaults(AgentId(1));
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        let ctx = make_context_with_orderbook(
            0,
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action = executor.on_tick(&ctx);

        // Should place initial order even without VWAP
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Buy);
    }

    #[test]
    fn test_vwap_respects_interval() {
        let mut executor = VwapExecutor::with_defaults(AgentId(1));
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // First order at tick 0
        let ctx0 = make_context_with_orderbook(
            0,
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let _ = executor.on_tick(&ctx0);

        // Should not order at tick 5 (interval is 10)
        let ctx5 = make_context_with_orderbook(
            5,
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action5 = executor.on_tick(&ctx5);
        assert!(action5.orders.is_empty());

        // Should order at tick 10
        let ctx10 = make_context_with_orderbook(
            10,
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action10 = executor.on_tick(&ctx10);
        assert_eq!(action10.orders.len(), 1);
    }

    #[test]
    fn test_vwap_completes_execution() {
        use types::{OrderId, TradeId};

        let config = VwapExecutorConfig {
            target_quantity: 100,
            max_slice_size: 100, // Complete in one fill
            slice_interval: 1,
            ..Default::default()
        };
        let mut executor = VwapExecutor::new(AgentId(1), config);
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Place order
        let ctx = make_context_with_orderbook(
            0,
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let _ = executor.on_tick(&ctx);

        // Simulate fill
        let trade = Trade {
            id: TradeId(1),
            symbol: "ACME".to_string(),
            price: Price::from_float(100.0),
            quantity: Quantity(100),
            buyer_id: AgentId(1),
            seller_id: AgentId(999),
            buyer_order_id: OrderId(1),
            seller_order_id: OrderId(2),
            timestamp: 100,
            tick: 1,
        };
        executor.on_fill(&trade);

        assert!(executor.is_complete());
        assert_eq!(executor.remaining(), 0);
    }

    #[test]
    fn test_vwap_stops_when_complete() {
        let config = VwapExecutorConfig {
            target_quantity: 100,
            max_slice_size: 100,
            ..Default::default()
        };
        let mut executor = VwapExecutor::new(AgentId(1), config);
        let order_book = create_order_book("ACME", 99.0, 101.0);
        let market = SingleSymbolMarket::new(&order_book);
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let indicators = IndicatorSnapshot::default();
        let trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();

        // Manually mark as complete
        executor.remaining_quantity = 0;

        let ctx = make_context_with_orderbook(
            0,
            &order_book,
            &candles,
            &indicators,
            &trades,
            &market,
            &events,
            &fundamentals,
        );
        let action = executor.on_tick(&ctx);

        assert!(action.orders.is_empty());
    }
}
