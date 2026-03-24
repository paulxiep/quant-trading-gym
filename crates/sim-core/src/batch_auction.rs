//! Reference-price auction matching engine for parallel order processing.
//!
//! This auction uses the reference price (WAP equilibrium) as the crossing threshold:
//!
//! 1. Filter qualifying orders (bids >= ref_price, asks <= ref_price)
//! 2. Compute traded volume as min(bid_vol, ask_vol)
//! 3. Trade price = WAP of qualifying asks (sellers get their aggregate price)
//! 4. Pro-rata fill the oversubscribed side
//!
//! This is O(n) with no sorting required, fully parallelizable per symbol.
//!
//! # Rationale
//!
//! - Reference price reflects supply/demand equilibrium from order imbalance
//! - Trading at seller WAP gives buyers price improvement, sellers their aggregate limit
//! - Pro-rata allocation is fair when no arrival-time priority exists

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use types::{
    AgentId, Order, OrderId, OrderSide, OrderType, Price, Quantity, Tick, Timestamp, Trade, TradeId,
};

/// Result of a batch auction for a single symbol.
#[derive(Debug, Clone, Default)]
pub struct BatchAuctionResult {
    /// The trade price (WAP of qualifying asks). None if no trades.
    pub clearing_price: Option<Price>,
    /// All trades executed.
    pub trades: Vec<Trade>,
    /// Orders that were fully filled.
    pub filled_orders: Vec<OrderId>,
    /// Orders that were partially filled (order_id, filled_qty).
    pub partial_fills: Vec<(OrderId, Quantity)>,
    /// Orders that didn't qualify (price didn't cross reference).
    pub unfilled_orders: Vec<OrderId>,
    /// Total qualifying bid volume.
    pub total_bid_volume: Quantity,
    /// Total qualifying ask volume.
    pub total_ask_volume: Quantity,
}

impl BatchAuctionResult {
    /// Check if any trades occurred.
    pub fn has_trades(&self) -> bool {
        !self.trades.is_empty()
    }

    /// Total quantity traded.
    pub fn traded_volume(&self) -> Quantity {
        self.trades.iter().map(|t| t.quantity).sum()
    }
}

/// Reference-price auction engine for a single symbol.
pub struct BatchAuction {
    /// Counter for generating unique trade IDs.
    next_trade_id: u64,
}

impl Default for BatchAuction {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchAuction {
    /// Create a new batch auction engine.
    pub fn new() -> Self {
        Self { next_trade_id: 1 }
    }

    /// Create with a starting trade ID (for coordinating across symbols).
    pub fn with_starting_id(start_id: u64) -> Self {
        Self {
            next_trade_id: start_id,
        }
    }

    /// Generate the next trade ID.
    fn next_trade_id(&mut self) -> TradeId {
        let id = TradeId(self.next_trade_id);
        self.next_trade_id += 1;
        id
    }

    /// Run a reference-price auction on the provided orders.
    ///
    /// # Algorithm
    ///
    /// 1. Filter qualifying bids (limit >= reference or market)
    /// 2. Filter qualifying asks (limit <= reference or market)
    /// 3. Compute traded_vol = min(bid_vol, ask_vol)
    /// 4. Trade price = WAP of qualifying asks
    /// 5. Pro-rata fill the oversubscribed side
    ///
    /// # Arguments
    /// * `symbol` - The symbol being auctioned
    /// * `orders` - All orders for this symbol
    /// * `timestamp` - Current timestamp for trades
    /// * `tick` - Current tick for trades
    /// * `reference_price` - The crossing threshold (WAP equilibrium from runner)
    pub fn run(
        &mut self,
        symbol: &str,
        orders: Vec<Order>,
        timestamp: Timestamp,
        tick: Tick,
        reference_price: Option<Price>,
    ) -> BatchAuctionResult {
        if orders.is_empty() {
            return BatchAuctionResult::default();
        }

        let Some(ref_price) = reference_price else {
            // No reference price - can't determine crossing
            let unfilled: Vec<_> = orders.iter().map(|o| o.id).collect();
            return BatchAuctionResult {
                unfilled_orders: unfilled,
                ..Default::default()
            };
        };

        // Partition into bids and asks
        let (all_bids, all_asks): (Vec<_>, Vec<_>) =
            orders.into_iter().partition(|o| o.side == OrderSide::Buy);

        // Filter qualifying orders
        let qualifying_bids: Vec<_> = all_bids
            .iter()
            .filter(|b| Self::bid_qualifies(b, ref_price))
            .cloned()
            .collect();

        let qualifying_asks: Vec<_> = all_asks
            .iter()
            .filter(|a| Self::ask_qualifies(a, ref_price))
            .cloned()
            .collect();

        // Compute volumes
        let bid_vol: u64 = qualifying_bids
            .iter()
            .map(|b| b.remaining_quantity.raw())
            .sum();
        let ask_vol: u64 = qualifying_asks
            .iter()
            .map(|a| a.remaining_quantity.raw())
            .sum();

        if bid_vol == 0 || ask_vol == 0 {
            // No crossing possible
            let unfilled: Vec<_> = all_bids
                .iter()
                .chain(all_asks.iter())
                .map(|o| o.id)
                .collect();
            return BatchAuctionResult {
                unfilled_orders: unfilled,
                total_bid_volume: Quantity(bid_vol),
                total_ask_volume: Quantity(ask_vol),
                ..Default::default()
            };
        }

        let traded_vol = bid_vol.min(ask_vol);

        // Compute trade price = WAP of qualifying asks
        let trade_price = Self::compute_ask_wap(&qualifying_asks, ref_price);

        // Determine fill ratios
        let (bid_fill_ratio, ask_fill_ratio) = if bid_vol <= ask_vol {
            // Buyers fully fill, sellers pro-rata
            (1.0, traded_vol as f64 / ask_vol as f64)
        } else {
            // Sellers fully fill, buyers pro-rata
            (traded_vol as f64 / bid_vol as f64, 1.0)
        };

        // Generate fills and trades
        let mut trades = Vec::new();
        let mut filled_orders = Vec::new();
        let mut partial_fills = Vec::new();

        // Compute bid allocations (fill recording deferred until after trade creation)
        let mut bid_allocations: Vec<(AgentId, OrderId, u64, u64)> = Vec::new(); // (agent, order_id, allocated_qty, order_qty)
        for bid in &qualifying_bids {
            let fill_qty = (bid.remaining_quantity.raw() as f64 * bid_fill_ratio).round() as u64;
            let fill_qty = fill_qty.min(bid.remaining_quantity.raw());
            if fill_qty > 0 {
                bid_allocations.push((
                    bid.agent_id,
                    bid.id,
                    fill_qty,
                    bid.remaining_quantity.raw(),
                ));
            }
        }

        // Track actual bid fills during trade creation
        let mut actual_bid_fills: Vec<u64> = vec![0; bid_allocations.len()];

        // Process asks and create trades by pairing with bids
        let mut bid_idx = 0;
        let mut bid_remaining = if !bid_allocations.is_empty() {
            bid_allocations[0].2
        } else {
            0
        };

        for ask in &qualifying_asks {
            let mut ask_fill_qty =
                (ask.remaining_quantity.raw() as f64 * ask_fill_ratio).round() as u64;
            ask_fill_qty = ask_fill_qty.min(ask.remaining_quantity.raw());

            if ask_fill_qty == 0 {
                continue;
            }

            // Create trades by consuming from bid allocations
            let mut remaining_ask = ask_fill_qty;
            let mut actual_ask_fill = 0u64;
            while remaining_ask > 0 && bid_idx < bid_allocations.len() {
                // Self-trade prevention: skip bids from the same agent
                if bid_allocations[bid_idx].0 == ask.agent_id {
                    bid_idx += 1;
                    if bid_idx < bid_allocations.len() {
                        bid_remaining = bid_allocations[bid_idx].2 - actual_bid_fills[bid_idx];
                    }
                    continue;
                }

                let trade_qty = remaining_ask.min(bid_remaining);
                if trade_qty > 0 {
                    trades.push(Trade {
                        id: self.next_trade_id(),
                        symbol: symbol.to_string(),
                        buyer_id: bid_allocations[bid_idx].0,
                        seller_id: ask.agent_id,
                        buyer_order_id: bid_allocations[bid_idx].1,
                        seller_order_id: ask.id,
                        price: trade_price,
                        quantity: Quantity(trade_qty),
                        timestamp,
                        tick,
                    });
                    remaining_ask -= trade_qty;
                    bid_remaining -= trade_qty;
                    actual_bid_fills[bid_idx] += trade_qty;
                    actual_ask_fill += trade_qty;
                }

                if bid_remaining == 0 {
                    bid_idx += 1;
                    if bid_idx < bid_allocations.len() {
                        bid_remaining = bid_allocations[bid_idx].2 - actual_bid_fills[bid_idx];
                    }
                }
            }

            // Record ask fill based on actual traded quantity
            if actual_ask_fill > 0 {
                if actual_ask_fill == ask.remaining_quantity.raw() {
                    filled_orders.push(ask.id);
                } else {
                    partial_fills.push((ask.id, Quantity(actual_ask_fill)));
                }
            }
        }

        // Record bid fills based on actual traded quantities
        for (idx, &(_, order_id, _, order_qty)) in bid_allocations.iter().enumerate() {
            let filled = actual_bid_fills[idx];
            if filled > 0 {
                if filled == order_qty {
                    filled_orders.push(order_id);
                } else {
                    partial_fills.push((order_id, Quantity(filled)));
                }
            }
        }

        // Collect unfilled orders
        let unfilled_orders: Vec<_> = all_bids
            .iter()
            .chain(all_asks.iter())
            .filter(|o| {
                !filled_orders.contains(&o.id) && !partial_fills.iter().any(|(id, _)| *id == o.id)
            })
            .map(|o| o.id)
            .collect();

        BatchAuctionResult {
            clearing_price: Some(trade_price),
            trades,
            filled_orders,
            partial_fills,
            unfilled_orders,
            total_bid_volume: Quantity(bid_vol),
            total_ask_volume: Quantity(ask_vol),
        }
    }

    /// Check if a bid qualifies (willing to buy at reference price or higher).
    #[inline]
    fn bid_qualifies(bid: &Order, ref_price: Price) -> bool {
        match bid.order_type {
            OrderType::Market => true,
            OrderType::Limit { price } => price >= ref_price,
        }
    }

    /// Check if an ask qualifies (willing to sell at reference price or lower).
    #[inline]
    fn ask_qualifies(ask: &Order, ref_price: Price) -> bool {
        match ask.order_type {
            OrderType::Market => true,
            OrderType::Limit { price } => price <= ref_price,
        }
    }

    /// Compute WAP of qualifying asks (trade price).
    /// Market orders use reference price.
    fn compute_ask_wap(asks: &[Order], ref_price: Price) -> Price {
        let (price_x_qty, total_qty) = asks.iter().fold((0.0, 0u64), |(pq, q), ask| {
            let price = ask.limit_price().unwrap_or(ref_price);
            let qty = ask.remaining_quantity.raw();
            (pq + price.to_float() * qty as f64, q + qty)
        });

        if total_qty == 0 {
            ref_price
        } else {
            Price::from_float(price_x_qty / total_qty as f64)
        }
    }
}

/// Run reference-price auctions for multiple symbols in parallel.
///
/// Each symbol is processed independently with its own orders.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled (V3.7)
#[cfg(feature = "parallel")]
pub fn run_parallel_auctions(
    orders_by_symbol: HashMap<String, Vec<Order>>,
    reference_prices: &HashMap<String, Price>,
    timestamp: Timestamp,
    tick: Tick,
    starting_trade_id: u64,
    force_sequential: bool,
) -> HashMap<String, BatchAuctionResult> {
    let symbols: Vec<_> = orders_by_symbol.keys().cloned().collect();
    let max_orders_per_symbol = orders_by_symbol
        .values()
        .map(|v| v.len() as u64)
        .max()
        .unwrap_or(0);

    if force_sequential {
        symbols
            .into_iter()
            .enumerate()
            .map(|(idx, symbol)| {
                let orders = orders_by_symbol.get(&symbol).cloned().unwrap_or_default();
                let ref_price = reference_prices.get(&symbol).copied();
                let symbol_start_id = starting_trade_id + (idx as u64 * max_orders_per_symbol);
                let mut auction = BatchAuction::with_starting_id(symbol_start_id);
                let result = auction.run(&symbol, orders, timestamp, tick, ref_price);
                (symbol, result)
            })
            .collect()
    } else {
        symbols
            .into_par_iter()
            .enumerate()
            .map(|(idx, symbol)| {
                let orders = orders_by_symbol.get(&symbol).cloned().unwrap_or_default();
                let ref_price = reference_prices.get(&symbol).copied();
                let symbol_start_id = starting_trade_id + (idx as u64 * max_orders_per_symbol);
                let mut auction = BatchAuction::with_starting_id(symbol_start_id);
                let result = auction.run(&symbol, orders, timestamp, tick, ref_price);
                (symbol, result)
            })
            .collect()
    }
}

/// Sequential version of multi-symbol auctions.
#[cfg(not(feature = "parallel"))]
pub fn run_parallel_auctions(
    orders_by_symbol: HashMap<String, Vec<Order>>,
    reference_prices: &HashMap<String, Price>,
    timestamp: Timestamp,
    tick: Tick,
    starting_trade_id: u64,
    force_sequential: bool,
) -> HashMap<String, BatchAuctionResult> {
    let _ = force_sequential; // Suppress unused warning
    let mut results = HashMap::new();
    let mut auction = BatchAuction::with_starting_id(starting_trade_id);

    for (symbol, orders) in orders_by_symbol {
        let ref_price = reference_prices.get(&symbol).copied();
        let result = auction.run(&symbol, orders, timestamp, tick, ref_price);
        results.insert(symbol, result);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bid(id: u64, agent: u64, price: f64, qty: u64) -> Order {
        let mut order = Order::limit(
            AgentId(agent),
            "TEST",
            OrderSide::Buy,
            Price::from_float(price),
            Quantity(qty),
        );
        order.id = OrderId(id);
        order
    }

    fn make_ask(id: u64, agent: u64, price: f64, qty: u64) -> Order {
        let mut order = Order::limit(
            AgentId(agent),
            "TEST",
            OrderSide::Sell,
            Price::from_float(price),
            Quantity(qty),
        );
        order.id = OrderId(id);
        order
    }

    #[test]
    fn test_no_reference_price() {
        let mut auction = BatchAuction::new();
        let orders = vec![make_bid(1, 1, 100.0, 50), make_ask(2, 2, 100.0, 50)];

        let result = auction.run("TEST", orders, 0, 0, None);

        assert!(result.clearing_price.is_none());
        assert!(result.trades.is_empty());
        assert_eq!(result.unfilled_orders.len(), 2);
    }

    #[test]
    fn test_no_crossing() {
        let mut auction = BatchAuction::new();
        // Bid at 99, ask at 101, reference at 100
        // Bid doesn't qualify (99 < 100), ask doesn't qualify (101 > 100)
        let orders = vec![make_bid(1, 1, 99.0, 100), make_ask(2, 2, 101.0, 100)];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        assert!(result.clearing_price.is_none());
        assert!(result.trades.is_empty());
        assert_eq!(result.unfilled_orders.len(), 2);
    }

    #[test]
    fn test_simple_crossing() {
        let mut auction = BatchAuction::new();
        // Bid at 101 >= ref 100, ask at 99 <= ref 100
        let orders = vec![make_bid(1, 1, 101.0, 50), make_ask(2, 2, 99.0, 50)];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        assert!(result.clearing_price.is_some());
        // Trade price = WAP of asks = 99
        assert_eq!(result.clearing_price, Some(Price::from_float(99.0)));
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].quantity, Quantity(50));
        assert_eq!(result.filled_orders.len(), 2);
    }

    #[test]
    fn test_buyers_oversubscribed() {
        let mut auction = BatchAuction::new();
        // 200 bid vol, 100 ask vol -> buyers fill pro-rata (50%)
        let orders = vec![
            make_bid(1, 1, 102.0, 100),
            make_bid(2, 2, 101.0, 100),
            make_ask(3, 3, 98.0, 50),
            make_ask(4, 4, 99.0, 50),
        ];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        assert!(result.clearing_price.is_some());
        // Trade price = WAP of asks = (98*50 + 99*50) / 100 = 98.5
        let expected_price = Price::from_float(98.5);
        assert_eq!(result.clearing_price, Some(expected_price));

        // Total traded = 100 (min of 200 bid, 100 ask)
        let total_traded: u64 = result.trades.iter().map(|t| t.quantity.raw()).sum();
        assert_eq!(total_traded, 100);
    }

    #[test]
    fn test_sellers_oversubscribed() {
        let mut auction = BatchAuction::new();
        // 100 bid vol, 200 ask vol -> sellers fill pro-rata (50%)
        let orders = vec![
            make_bid(1, 1, 102.0, 50),
            make_bid(2, 2, 101.0, 50),
            make_ask(3, 3, 98.0, 100),
            make_ask(4, 4, 99.0, 100),
        ];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        assert!(result.clearing_price.is_some());

        // Total traded = 100 (min of 100 bid, 200 ask)
        let total_traded: u64 = result.trades.iter().map(|t| t.quantity.raw()).sum();
        assert_eq!(total_traded, 100);
    }

    #[test]
    fn test_trade_price_is_seller_wap() {
        let mut auction = BatchAuction::new();
        // Two asks at different prices
        let orders = vec![
            make_bid(1, 1, 105.0, 100),
            make_ask(2, 2, 90.0, 60), // 60 @ 90
            make_ask(3, 3, 95.0, 40), // 40 @ 95
        ];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        // WAP = (90*60 + 95*40) / 100 = (5400 + 3800) / 100 = 92
        assert_eq!(result.clearing_price, Some(Price::from_float(92.0)));
    }

    #[test]
    fn test_trades_have_correct_buyer_seller() {
        let mut auction = BatchAuction::new();
        let orders = vec![make_bid(1, 10, 101.0, 50), make_ask(2, 20, 99.0, 50)];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].buyer_id, AgentId(10));
        assert_eq!(result.trades[0].seller_id, AgentId(20));
    }

    #[test]
    fn test_self_trade_prevention() {
        let mut auction = BatchAuction::new();
        // Same agent (1) places both bid and ask - should NOT match with self
        let orders = vec![
            make_bid(1, 1, 101.0, 50), // Agent 1 bids
            make_ask(2, 1, 99.0, 50),  // Agent 1 asks (same agent!)
        ];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        // No trades should occur - self-trade prevented
        assert!(result.trades.is_empty());
    }

    #[test]
    fn test_self_trade_prevention_with_other_counterparty() {
        let mut auction = BatchAuction::new();
        // Agent 1 has both bid and ask, but agent 2 also has a bid
        // Agent 1's ask should match with agent 2's bid, not agent 1's bid
        let orders = vec![
            make_bid(1, 1, 101.0, 50), // Agent 1 bids 50
            make_bid(2, 2, 101.0, 50), // Agent 2 bids 50
            make_ask(3, 1, 99.0, 50),  // Agent 1 asks 50 (should match with agent 2, not self)
        ];

        let result = auction.run("TEST", orders, 0, 0, Some(Price::from_float(100.0)));

        // Should have 1 trade: agent 1 sells to agent 2 (not to self)
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].seller_id, AgentId(1)); // Agent 1 sells
        assert_eq!(result.trades[0].buyer_id, AgentId(2)); // Agent 2 buys (not agent 1!)
    }
}
