//! Market abstractions for multi-symbol support (V2.3).
//!
//! This module provides:
//! - [`MarketView`] - Read-only trait for accessing market state
//! - [`SingleSymbolMarket`] - Adapter for legacy single-symbol simulations
//!
//! # Architecture
//!
//! The `MarketView` trait abstracts market data access, enabling:
//! - Single-symbol simulations (via `SingleSymbolMarket` adapter)
//! - Multi-symbol simulations (via future `Market` struct)
//!
//! Agents receive a `&dyn MarketView` through `StrategyContext`, allowing
//! them to query any symbol without knowing the underlying implementation.
//!
//! # Borrow Checker Design
//!
//! `SingleSymbolMarket` holds a reference to an `OrderBook`. To satisfy
//! lifetime requirements, it should be stored in the simulation struct
//! or rebuilt each tick before building `StrategyContext`.

use std::collections::HashMap;

use types::{BookSnapshot, Price, Quantity, Symbol, Tick, Timestamp};

use crate::OrderBook;

// =============================================================================
// MarketView Trait
// =============================================================================

/// Read-only interface for market state access.
///
/// This trait provides a unified interface for strategies to access market
/// data regardless of whether the simulation is single-symbol or multi-symbol.
///
/// All methods take `&Symbol` to support multi-symbol access patterns,
/// even when there's only one symbol in the market.
///
/// # Design Rationale
///
/// - **References over owned data**: All methods return borrowed data or
///   compute values on the fly to avoid cloning.
/// - **Option returns**: Methods return `Option` for symbols that may not
///   exist in the market.
/// - **No mutation**: This is a read-only view; order submission goes
///   through separate channels.
pub trait MarketView: Send + Sync {
    /// Get the list of symbols available in this market.
    fn symbols(&self) -> Vec<Symbol>;

    /// Check if a symbol exists in this market.
    fn has_symbol(&self, symbol: &Symbol) -> bool;

    /// Get the mid price for a symbol.
    ///
    /// Returns the average of best bid and best ask, or falls back to
    /// last price if one side is empty.
    fn mid_price(&self, symbol: &Symbol) -> Option<Price>;

    /// Get the best bid price for a symbol.
    fn best_bid(&self, symbol: &Symbol) -> Option<Price>;

    /// Get the best ask price for a symbol.
    fn best_ask(&self, symbol: &Symbol) -> Option<Price>;

    /// Get the last traded price for a symbol.
    fn last_price(&self, symbol: &Symbol) -> Option<Price>;

    /// Get a snapshot of the order book for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - The symbol to snapshot
    /// * `depth` - Maximum number of price levels per side
    /// * `timestamp` - Current timestamp for the snapshot
    /// * `tick` - Current tick for the snapshot
    fn snapshot(
        &self,
        symbol: &Symbol,
        depth: usize,
        timestamp: Timestamp,
        tick: Tick,
    ) -> Option<BookSnapshot>;

    /// Get the total bid volume for a symbol (sum of all bid quantities).
    fn total_bid_volume(&self, symbol: &Symbol) -> Quantity;

    /// Get the total ask volume for a symbol (sum of all ask quantities).
    fn total_ask_volume(&self, symbol: &Symbol) -> Quantity;

    /// Get the spread for a symbol (ask - bid).
    fn spread(&self, symbol: &Symbol) -> Option<Price> {
        match (self.best_bid(symbol), self.best_ask(symbol)) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }
}

// =============================================================================
// SingleSymbolMarket Adapter
// =============================================================================

/// Adapter wrapping a single `OrderBook` to implement `MarketView`.
///
/// This allows single-symbol simulations to use the same `StrategyContext`
/// interface as multi-symbol simulations.
///
/// # Lifetime
///
/// The adapter holds a reference to the `OrderBook`, so it must live at least
/// as long as any `StrategyContext` that references it. In practice, this means:
/// - Create `SingleSymbolMarket` at the start of the tick
/// - Build `StrategyContext` with a reference to it
/// - Drop `StrategyContext` before modifying the `OrderBook`
///
/// # Example
///
/// ```ignore
/// let market = SingleSymbolMarket::new(&order_book);
/// let ctx = StrategyContext::new(&market, ...);
/// // Use ctx with agents
/// drop(ctx);  // Or let it go out of scope
/// // Now safe to modify order_book
/// ```
#[derive(Debug)]
pub struct SingleSymbolMarket<'a> {
    /// The underlying order book.
    book: &'a OrderBook,
}

impl<'a> SingleSymbolMarket<'a> {
    /// Create a new single-symbol market adapter.
    pub fn new(book: &'a OrderBook) -> Self {
        Self { book }
    }

    /// Get a reference to the underlying order book.
    pub fn book(&self) -> &OrderBook {
        self.book
    }
}

impl MarketView for SingleSymbolMarket<'_> {
    fn symbols(&self) -> Vec<Symbol> {
        vec![self.book.symbol().to_string()]
    }

    fn has_symbol(&self, symbol: &Symbol) -> bool {
        self.book.symbol() == symbol
    }

    fn mid_price(&self, symbol: &Symbol) -> Option<Price> {
        if self.has_symbol(symbol) {
            self.book.mid_price()
        } else {
            None
        }
    }

    fn best_bid(&self, symbol: &Symbol) -> Option<Price> {
        if self.has_symbol(symbol) {
            self.book.best_bid_price()
        } else {
            None
        }
    }

    fn best_ask(&self, symbol: &Symbol) -> Option<Price> {
        if self.has_symbol(symbol) {
            self.book.best_ask_price()
        } else {
            None
        }
    }

    fn last_price(&self, symbol: &Symbol) -> Option<Price> {
        if self.has_symbol(symbol) {
            self.book.last_price()
        } else {
            None
        }
    }

    fn snapshot(
        &self,
        symbol: &Symbol,
        depth: usize,
        timestamp: Timestamp,
        tick: Tick,
    ) -> Option<BookSnapshot> {
        if self.has_symbol(symbol) {
            Some(self.book.snapshot(timestamp, tick, depth))
        } else {
            None
        }
    }

    fn total_bid_volume(&self, symbol: &Symbol) -> Quantity {
        if self.has_symbol(symbol) {
            self.book.total_bid_volume()
        } else {
            Quantity::ZERO
        }
    }

    fn total_ask_volume(&self, symbol: &Symbol) -> Quantity {
        if self.has_symbol(symbol) {
            self.book.total_ask_volume()
        } else {
            Quantity::ZERO
        }
    }
}

// =============================================================================
// Multi-Symbol Market (Future V2.3 Full Implementation)
// =============================================================================

/// Multi-symbol market container.
///
/// Holds multiple order books, one per symbol. This will be the primary
/// market implementation for multi-symbol simulations.
///
/// # Note
///
/// This is a forward-looking struct. For V2.3, we primarily use
/// `SingleSymbolMarket` but this provides the foundation for future
/// multi-symbol support.
#[derive(Debug, Default)]
pub struct Market {
    /// Order books by symbol.
    books: HashMap<Symbol, OrderBook>,
}

impl Market {
    /// Create a new empty multi-symbol market.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symbol to the market.
    pub fn add_symbol(&mut self, symbol: impl Into<Symbol>) {
        let symbol = symbol.into();
        self.books.insert(symbol.clone(), OrderBook::new(symbol));
    }

    /// Add a symbol with an initial reference price.
    ///
    /// This sets the order book's last_price so charts and agents have
    /// a reference price even before any trades occur.
    pub fn add_symbol_with_price(&mut self, symbol: impl Into<Symbol>, initial_price: Price) {
        let symbol = symbol.into();
        let mut book = OrderBook::new(symbol.clone());
        book.set_last_price(initial_price);
        self.books.insert(symbol, book);
    }

    /// Get a reference to an order book by symbol.
    pub fn get_book(&self, symbol: &Symbol) -> Option<&OrderBook> {
        self.books.get(symbol)
    }

    /// Get a mutable reference to an order book by symbol.
    pub fn get_book_mut(&mut self, symbol: &Symbol) -> Option<&mut OrderBook> {
        self.books.get_mut(symbol)
    }

    /// Get mutable references to all order books.
    pub fn books_mut(&mut self) -> impl Iterator<Item = &mut OrderBook> {
        self.books.values_mut()
    }

    /// Get all symbols in the market.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.books.keys()
    }

    /// Get the number of symbols.
    pub fn symbol_count(&self) -> usize {
        self.books.len()
    }
}

impl MarketView for Market {
    fn symbols(&self) -> Vec<Symbol> {
        self.books.keys().cloned().collect()
    }

    fn has_symbol(&self, symbol: &Symbol) -> bool {
        self.books.contains_key(symbol)
    }

    fn mid_price(&self, symbol: &Symbol) -> Option<Price> {
        self.books.get(symbol).and_then(|b| b.mid_price())
    }

    fn best_bid(&self, symbol: &Symbol) -> Option<Price> {
        self.books.get(symbol).and_then(|b| b.best_bid_price())
    }

    fn best_ask(&self, symbol: &Symbol) -> Option<Price> {
        self.books.get(symbol).and_then(|b| b.best_ask_price())
    }

    fn last_price(&self, symbol: &Symbol) -> Option<Price> {
        self.books.get(symbol).and_then(|b| b.last_price())
    }

    fn snapshot(
        &self,
        symbol: &Symbol,
        depth: usize,
        timestamp: Timestamp,
        tick: Tick,
    ) -> Option<BookSnapshot> {
        self.books
            .get(symbol)
            .map(|b| b.snapshot(timestamp, tick, depth))
    }

    fn total_bid_volume(&self, symbol: &Symbol) -> Quantity {
        self.books
            .get(symbol)
            .map(|b| b.total_bid_volume())
            .unwrap_or(Quantity::ZERO)
    }

    fn total_ask_volume(&self, symbol: &Symbol) -> Quantity {
        self.books
            .get(symbol)
            .map(|b| b.total_ask_volume())
            .unwrap_or(Quantity::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{Order, OrderId, OrderSide, Quantity};

    fn setup_book_with_orders() -> OrderBook {
        let mut book = OrderBook::new("TEST");
        let mut bid = Order::limit(
            types::AgentId(1),
            "TEST",
            OrderSide::Buy,
            Price::from_float(99.0),
            Quantity(100),
        );
        bid.id = OrderId(1);
        let mut ask = Order::limit(
            types::AgentId(2),
            "TEST",
            OrderSide::Sell,
            Price::from_float(101.0),
            Quantity(100),
        );
        ask.id = OrderId(2);
        book.add_order(bid).unwrap();
        book.add_order(ask).unwrap();
        book
    }

    #[test]
    fn test_single_symbol_market_symbols() {
        let book = OrderBook::new("AAPL");
        let market = SingleSymbolMarket::new(&book);

        assert_eq!(market.symbols(), vec!["AAPL".to_string()]);
        assert!(market.has_symbol(&"AAPL".to_string()));
        assert!(!market.has_symbol(&"GOOG".to_string()));
    }

    #[test]
    fn test_single_symbol_market_prices() {
        let book = setup_book_with_orders();
        let market = SingleSymbolMarket::new(&book);
        let symbol = "TEST".to_string();

        assert_eq!(market.best_bid(&symbol), Some(Price::from_float(99.0)));
        assert_eq!(market.best_ask(&symbol), Some(Price::from_float(101.0)));
        assert_eq!(market.mid_price(&symbol), Some(Price::from_float(100.0)));
        assert_eq!(market.spread(&symbol), Some(Price::from_float(2.0)));
    }

    #[test]
    fn test_single_symbol_market_wrong_symbol() {
        let book = setup_book_with_orders();
        let market = SingleSymbolMarket::new(&book);
        let wrong_symbol = "WRONG".to_string();

        assert_eq!(market.best_bid(&wrong_symbol), None);
        assert_eq!(market.best_ask(&wrong_symbol), None);
        assert_eq!(market.mid_price(&wrong_symbol), None);
    }

    #[test]
    fn test_multi_symbol_market() {
        let mut market = Market::new();
        market.add_symbol("AAPL");
        market.add_symbol("GOOG");

        assert_eq!(market.symbol_count(), 2);
        assert!(market.has_symbol(&"AAPL".to_string()));
        assert!(market.has_symbol(&"GOOG".to_string()));
        assert!(!market.has_symbol(&"MSFT".to_string()));
    }

    #[test]
    fn test_snapshot() {
        let book = setup_book_with_orders();
        let market = SingleSymbolMarket::new(&book);
        let symbol = "TEST".to_string();

        let snapshot = market.snapshot(&symbol, 10, 1000, 100);
        assert!(snapshot.is_some());

        let snapshot = snapshot.unwrap();
        assert_eq!(snapshot.symbol, symbol);
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }
}
