//! Order book implementation using BTreeMap for price-time priority.
//!
//! The order book maintains buy (bid) and sell (ask) orders organized by price.
//! Within each price level, orders are queued in FIFO order (time priority).

use std::collections::{BTreeMap, HashMap, VecDeque};

use types::{
    AgentId, BookLevel, BookSnapshot, Order, OrderId, OrderSide, OrderType, Price, Quantity, Tick,
    Timestamp,
};

use crate::error::{Result, SimCoreError};

/// A price level containing orders at a single price point.
#[derive(Debug, Clone, Default)]
pub struct PriceLevel {
    /// Total quantity available at this price.
    pub total_quantity: Quantity,
    /// Orders at this price, in time priority order (FIFO).
    pub orders: VecDeque<Order>,
}

impl PriceLevel {
    /// Create a new empty price level.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an order to the back of the queue.
    pub fn push(&mut self, order: Order) {
        self.total_quantity += order.remaining_quantity;
        self.orders.push_back(order);
    }

    /// Peek at the first order without removing it.
    pub fn front(&self) -> Option<&Order> {
        self.orders.front()
    }

    /// Get mutable reference to the first order.
    pub fn front_mut(&mut self) -> Option<&mut Order> {
        self.orders.front_mut()
    }

    /// Remove the first order from the queue.
    pub fn pop(&mut self) -> Option<Order> {
        if let Some(order) = self.orders.pop_front() {
            self.total_quantity = self.total_quantity.saturating_sub(order.remaining_quantity);
            Some(order)
        } else {
            None
        }
    }

    /// Check if this price level is empty.
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Number of orders at this price level.
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Update total quantity after a partial fill.
    pub fn reduce_quantity(&mut self, qty: Quantity) {
        self.total_quantity = self.total_quantity.saturating_sub(qty);
    }
}

/// Order book for a single symbol.
///
/// Uses `BTreeMap` to maintain price levels in sorted order:
/// - Bids: Highest price first (descending)
/// - Asks: Lowest price first (ascending)
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// The symbol this order book is for.
    symbol: String,
    /// Buy orders indexed by price (highest first when iterating in reverse).
    bids: BTreeMap<Price, PriceLevel>,
    /// Sell orders indexed by price (lowest first when iterating forward).
    asks: BTreeMap<Price, PriceLevel>,
    /// Quick lookup of orders by ID.
    order_index: HashMap<OrderId, (OrderSide, Price)>,
    /// Last trade price (for market order reference).
    last_price: Option<Price>,
}

impl OrderBook {
    /// Create a new empty order book for a symbol.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            order_index: HashMap::new(),
            last_price: None,
        }
    }

    /// Get the symbol this book is for.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Add an order to the book.
    ///
    /// This method only handles limit orders. Market orders should be
    /// processed through the matching engine first.
    pub fn add_order(&mut self, order: Order) -> Result<()> {
        // Validate order
        if order.remaining_quantity.is_zero() {
            return Err(SimCoreError::ZeroQuantity);
        }

        let price = match order.order_type {
            OrderType::Limit { price } => {
                if !price.is_positive() {
                    return Err(SimCoreError::InvalidPrice);
                }
                price
            }
            OrderType::Market => {
                // Market orders shouldn't be added to the book - they should be matched immediately
                return Err(SimCoreError::InvalidPrice);
            }
        };

        // Index the order for quick lookup
        self.order_index.insert(order.id, (order.side, price));

        // Add to appropriate side
        let book_side = match order.side {
            OrderSide::Buy => &mut self.bids,
            OrderSide::Sell => &mut self.asks,
        };

        book_side.entry(price).or_default().push(order);

        Ok(())
    }

    /// Remove an order from the book by ID.
    pub fn cancel_order(&mut self, order_id: OrderId) -> Result<Order> {
        let (side, price) = self
            .order_index
            .remove(&order_id)
            .ok_or(SimCoreError::OrderNotFound(order_id))?;

        let book_side = match side {
            OrderSide::Buy => &mut self.bids,
            OrderSide::Sell => &mut self.asks,
        };

        if let Some(level) = book_side.get_mut(&price) {
            // Find and remove the order
            if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                let order = level.orders.remove(pos).unwrap();
                level.total_quantity = level
                    .total_quantity
                    .saturating_sub(order.remaining_quantity);

                // Clean up empty price levels
                if level.is_empty() {
                    book_side.remove(&price);
                }

                return Ok(order);
            }
        }

        Err(SimCoreError::OrderNotFound(order_id))
    }

    /// Get the best bid (highest buy price).
    pub fn best_bid(&self) -> Option<(Price, &PriceLevel)> {
        self.bids.iter().next_back().map(|(p, l)| (*p, l))
    }

    /// Get the best ask (lowest sell price).
    pub fn best_ask(&self) -> Option<(Price, &PriceLevel)> {
        self.asks.iter().next().map(|(p, l)| (*p, l))
    }

    /// Get the best bid price.
    pub fn best_bid_price(&self) -> Option<Price> {
        self.best_bid().map(|(p, _)| p)
    }

    /// Get the best ask price.
    pub fn best_ask_price(&self) -> Option<Price> {
        self.best_ask().map(|(p, _)| p)
    }

    /// Calculate the spread between best bid and ask.
    pub fn spread(&self) -> Option<Price> {
        match (self.best_bid_price(), self.best_ask_price()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate the mid price.
    pub fn mid_price(&self) -> Option<Price> {
        match (self.best_bid_price(), self.best_ask_price()) {
            (Some(bid), Some(ask)) => Some(Price((bid.raw() + ask.raw()) / 2)),
            // Fall back to last price if one side is empty
            (Some(bid), None) => Some(bid),
            (None, Some(ask)) => Some(ask),
            (None, None) => self.last_price,
        }
    }

    /// Get mutable reference to the best bid level.
    pub fn best_bid_mut(&mut self) -> Option<(Price, &mut PriceLevel)> {
        self.bids.iter_mut().next_back().map(|(p, l)| (*p, l))
    }

    /// Get mutable reference to the best ask level.
    pub fn best_ask_mut(&mut self) -> Option<(Price, &mut PriceLevel)> {
        self.asks.iter_mut().next().map(|(p, l)| (*p, l))
    }

    /// Remove empty price levels and update order index after matching.
    pub fn cleanup_level(&mut self, side: OrderSide, price: Price) {
        let book_side = match side {
            OrderSide::Buy => &mut self.bids,
            OrderSide::Sell => &mut self.asks,
        };

        if let Some(level) = book_side.get(&price)
            && level.is_empty()
        {
            book_side.remove(&price);
        }
    }

    /// Remove an order from the index.
    pub fn remove_from_index(&mut self, order_id: OrderId) {
        self.order_index.remove(&order_id);
    }

    /// Peek at the best bid order's info without modifying.
    /// Returns (agent_id, order_id, remaining_quantity).
    pub fn peek_best_bid_order(&self) -> Option<(AgentId, OrderId, Quantity)> {
        self.bids
            .iter()
            .next_back()
            .and_then(|(_, level)| level.front())
            .map(|order| (order.agent_id, order.id, order.remaining_quantity))
    }

    /// Peek at the best ask order's info without modifying.
    /// Returns (agent_id, order_id, remaining_quantity).
    pub fn peek_best_ask_order(&self) -> Option<(AgentId, OrderId, Quantity)> {
        self.asks
            .iter()
            .next()
            .and_then(|(_, level)| level.front())
            .map(|order| (order.agent_id, order.id, order.remaining_quantity))
    }

    /// Fill (reduce) the best bid by the given quantity.
    /// Removes the order if fully filled and cleans up empty levels.
    pub fn fill_best_bid(&mut self, quantity: Quantity) {
        let mut price_to_cleanup = None;
        let mut order_to_remove = None;
        let mut should_pop = false;

        if let Some((price, level)) = self.bids.iter_mut().next_back() {
            let price = *price;
            if let Some(order) = level.front_mut() {
                order.remaining_quantity = order.remaining_quantity.saturating_sub(quantity);

                if order.remaining_quantity.is_zero() {
                    order_to_remove = Some(order.id);
                    should_pop = true;
                }
            }
            // Update level quantity
            level.total_quantity = level.total_quantity.saturating_sub(quantity);

            if should_pop {
                level.orders.pop_front();
            }

            if level.is_empty() {
                price_to_cleanup = Some(price);
            }
        }

        if let Some(order_id) = order_to_remove {
            self.order_index.remove(&order_id);
        }

        if let Some(price) = price_to_cleanup {
            self.bids.remove(&price);
        }
    }

    /// Fill (reduce) the best ask by the given quantity.
    /// Removes the order if fully filled and cleans up empty levels.
    pub fn fill_best_ask(&mut self, quantity: Quantity) {
        let mut price_to_cleanup = None;
        let mut order_to_remove = None;
        let mut should_pop = false;

        if let Some((price, level)) = self.asks.iter_mut().next() {
            let price = *price;
            if let Some(order) = level.front_mut() {
                order.remaining_quantity = order.remaining_quantity.saturating_sub(quantity);

                if order.remaining_quantity.is_zero() {
                    order_to_remove = Some(order.id);
                    should_pop = true;
                }
            }
            // Update level quantity
            level.total_quantity = level.total_quantity.saturating_sub(quantity);

            if should_pop {
                level.orders.pop_front();
            }

            if level.is_empty() {
                price_to_cleanup = Some(price);
            }
        }

        if let Some(order_id) = order_to_remove {
            self.order_index.remove(&order_id);
        }

        if let Some(price) = price_to_cleanup {
            self.asks.remove(&price);
        }
    }

    /// Update the last traded price.
    pub fn set_last_price(&mut self, price: Price) {
        self.last_price = Some(price);
    }

    /// Get the last traded price.
    pub fn last_price(&self) -> Option<Price> {
        self.last_price
    }

    /// Get total bid depth (volume) up to N levels.
    pub fn bid_depth(&self, levels: usize) -> Quantity {
        self.bids
            .iter()
            .rev()
            .take(levels)
            .map(|(_, l)| l.total_quantity)
            .sum()
    }

    /// Get total ask depth (volume) up to N levels.
    pub fn ask_depth(&self, levels: usize) -> Quantity {
        self.asks
            .iter()
            .take(levels)
            .map(|(_, l)| l.total_quantity)
            .sum()
    }

    /// Get total volume of all bid orders (V2.2).
    pub fn total_bid_volume(&self) -> Quantity {
        self.bids.values().map(|l| l.total_quantity).sum()
    }

    /// Get total volume of all ask orders (V2.2).
    pub fn total_ask_volume(&self) -> Quantity {
        self.asks.values().map(|l| l.total_quantity).sum()
    }

    /// Get total bid depth up to a given price (V2.2).
    ///
    /// Returns total volume of bids at or above the given price.
    pub fn bid_depth_to_price(&self, min_price: Price) -> Quantity {
        self.bids
            .iter()
            .rev()
            .take_while(|(price, _)| **price >= min_price)
            .map(|(_, l)| l.total_quantity)
            .sum()
    }

    /// Get total ask depth up to a given price (V2.2).
    ///
    /// Returns total volume of asks at or below the given price.
    pub fn ask_depth_to_price(&self, max_price: Price) -> Quantity {
        self.asks
            .iter()
            .take_while(|(price, _)| **price <= max_price)
            .map(|(_, l)| l.total_quantity)
            .sum()
    }

    /// Check if the book has any orders.
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Get a snapshot of the current book state.
    pub fn snapshot(&self, timestamp: Timestamp, tick: Tick, depth: usize) -> BookSnapshot {
        let bids: Vec<BookLevel> = self
            .bids
            .iter()
            .rev()
            .take(depth)
            .map(|(price, level)| BookLevel {
                price: *price,
                quantity: level.total_quantity,
                order_count: level.order_count(),
            })
            .collect();

        let asks: Vec<BookLevel> = self
            .asks
            .iter()
            .take(depth)
            .map(|(price, level)| BookLevel {
                price: *price,
                quantity: level.total_quantity,
                order_count: level.order_count(),
            })
            .collect();

        BookSnapshot {
            symbol: self.symbol.clone(),
            bids,
            asks,
            timestamp,
            tick,
        }
    }

    /// Number of price levels on the bid side.
    pub fn bid_levels(&self) -> usize {
        self.bids.len()
    }

    /// Number of price levels on the ask side.
    pub fn ask_levels(&self) -> usize {
        self.asks.len()
    }

    /// Total number of orders in the book.
    pub fn order_count(&self) -> usize {
        self.order_index.len()
    }

    /// Clear all orders from the book (for end-of-tick expiration).
    ///
    /// Removes all bids, asks, and order index entries.
    /// Preserves `last_price` for reference.
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.order_index.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::AgentId;

    fn make_limit_order(
        id: u64,
        agent_id: u64,
        side: OrderSide,
        price: f64,
        quantity: u64,
    ) -> Order {
        let mut order = Order::limit(
            AgentId(agent_id),
            "TEST",
            side,
            Price::from_float(price),
            Quantity(quantity),
        );
        order.id = OrderId(id);
        order
    }

    #[test]
    fn test_new_order_book() {
        let book = OrderBook::new("AAPL");
        assert_eq!(book.symbol(), "AAPL");
        assert!(book.is_empty());
        assert_eq!(book.best_bid_price(), None);
        assert_eq!(book.best_ask_price(), None);
    }

    #[test]
    fn test_add_buy_order() {
        let mut book = OrderBook::new("AAPL");
        let order = make_limit_order(1, 1, OrderSide::Buy, 100.0, 50);

        book.add_order(order).unwrap();

        assert!(!book.is_empty());
        assert_eq!(book.best_bid_price(), Some(Price::from_float(100.0)));
        assert_eq!(book.best_ask_price(), None);
        assert_eq!(book.bid_depth(10), 50);
    }

    #[test]
    fn test_add_sell_order() {
        let mut book = OrderBook::new("AAPL");
        let order = make_limit_order(1, 1, OrderSide::Sell, 101.0, 75);

        book.add_order(order).unwrap();

        assert_eq!(book.best_bid_price(), None);
        assert_eq!(book.best_ask_price(), Some(Price::from_float(101.0)));
        assert_eq!(book.ask_depth(10), 75);
    }

    #[test]
    fn test_multiple_price_levels() {
        let mut book = OrderBook::new("AAPL");

        // Add bids at different prices
        book.add_order(make_limit_order(1, 1, OrderSide::Buy, 99.0, 100))
            .unwrap();
        book.add_order(make_limit_order(2, 1, OrderSide::Buy, 100.0, 50))
            .unwrap();
        book.add_order(make_limit_order(3, 1, OrderSide::Buy, 98.0, 200))
            .unwrap();

        // Best bid should be highest price
        assert_eq!(book.best_bid_price(), Some(Price::from_float(100.0)));
        assert_eq!(book.bid_levels(), 3);

        // Add asks at different prices
        book.add_order(make_limit_order(4, 2, OrderSide::Sell, 102.0, 150))
            .unwrap();
        book.add_order(make_limit_order(5, 2, OrderSide::Sell, 101.0, 75))
            .unwrap();

        // Best ask should be lowest price
        assert_eq!(book.best_ask_price(), Some(Price::from_float(101.0)));
        assert_eq!(book.ask_levels(), 2);
    }

    #[test]
    fn test_time_priority_same_price() {
        let mut book = OrderBook::new("AAPL");

        // Add multiple orders at same price
        let order1 = make_limit_order(1, 1, OrderSide::Buy, 100.0, 50);
        let order2 = make_limit_order(2, 2, OrderSide::Buy, 100.0, 75);
        let order3 = make_limit_order(3, 3, OrderSide::Buy, 100.0, 25);

        book.add_order(order1).unwrap();
        book.add_order(order2).unwrap();
        book.add_order(order3).unwrap();

        // First order should be at front (time priority)
        let (_, level) = book.best_bid().unwrap();
        assert_eq!(level.front().unwrap().id, OrderId(1));
        assert_eq!(level.total_quantity, 150); // 50 + 75 + 25
        assert_eq!(level.order_count(), 3);
    }

    #[test]
    fn test_cancel_order() {
        let mut book = OrderBook::new("AAPL");

        book.add_order(make_limit_order(1, 1, OrderSide::Buy, 100.0, 50))
            .unwrap();
        book.add_order(make_limit_order(2, 1, OrderSide::Buy, 100.0, 75))
            .unwrap();

        // Cancel first order
        let cancelled = book.cancel_order(OrderId(1)).unwrap();
        assert_eq!(cancelled.id, OrderId(1));

        // Second order should now be at front
        let (_, level) = book.best_bid().unwrap();
        assert_eq!(level.front().unwrap().id, OrderId(2));
        assert_eq!(level.total_quantity, 75);
    }

    #[test]
    fn test_cancel_nonexistent_order() {
        let mut book = OrderBook::new("AAPL");
        let result = book.cancel_order(OrderId(999));
        assert!(matches!(result, Err(SimCoreError::OrderNotFound(_))));
    }

    #[test]
    fn test_spread_calculation() {
        let mut book = OrderBook::new("AAPL");

        book.add_order(make_limit_order(1, 1, OrderSide::Buy, 99.0, 100))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 101.0, 100))
            .unwrap();

        assert_eq!(book.spread(), Some(Price::from_float(2.0)));
        assert_eq!(book.mid_price(), Some(Price::from_float(100.0)));
    }

    #[test]
    fn test_snapshot() {
        let mut book = OrderBook::new("AAPL");

        book.add_order(make_limit_order(1, 1, OrderSide::Buy, 99.0, 100))
            .unwrap();
        book.add_order(make_limit_order(2, 1, OrderSide::Buy, 98.0, 200))
            .unwrap();
        book.add_order(make_limit_order(3, 2, OrderSide::Sell, 101.0, 150))
            .unwrap();

        let snapshot = book.snapshot(1000, 5, 10);

        assert_eq!(snapshot.symbol, "AAPL");
        assert_eq!(snapshot.bids.len(), 2);
        assert_eq!(snapshot.asks.len(), 1);
        assert_eq!(snapshot.best_bid(), Some(Price::from_float(99.0)));
        assert_eq!(snapshot.best_ask(), Some(Price::from_float(101.0)));
    }

    #[test]
    fn test_zero_quantity_rejected() {
        let mut book = OrderBook::new("AAPL");
        let mut order = make_limit_order(1, 1, OrderSide::Buy, 100.0, 0);
        order.remaining_quantity = Quantity::ZERO;

        let result = book.add_order(order);
        assert!(matches!(result, Err(SimCoreError::ZeroQuantity)));
    }

    #[test]
    fn test_invalid_price_rejected() {
        let mut book = OrderBook::new("AAPL");
        let order = make_limit_order(1, 1, OrderSide::Buy, 0.0, 100);

        let result = book.add_order(order);
        assert!(matches!(result, Err(SimCoreError::InvalidPrice)));
    }

    #[test]
    fn test_cancel_removes_empty_level() {
        let mut book = OrderBook::new("AAPL");

        book.add_order(make_limit_order(1, 1, OrderSide::Buy, 100.0, 50))
            .unwrap();

        assert_eq!(book.bid_levels(), 1);

        book.cancel_order(OrderId(1)).unwrap();

        assert_eq!(book.bid_levels(), 0);
        assert!(book.is_empty());
    }
}
