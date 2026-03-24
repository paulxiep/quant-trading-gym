//! Auction engine subsystem.
//!
//! Handles order collection, validation, and batch auction execution.
//! Order collection is pure: validate → group → assign IDs.

use std::collections::HashMap;

use crate::traits::PositionTracker;
use crate::traits::agents::AgentActionWithState;
use sim_core::{BatchAuctionResult, Market, run_parallel_auctions};
use types::{Order, OrderId, OrderSide, Price, Symbol, SymbolConfig, Tick, Timestamp};

/// Handles order collection and batch auction execution.
///
/// Owns order ID generation. Uses external RiskManager for validation.
pub struct AuctionEngine {
    /// Counter for generating unique order IDs.
    next_order_id: u64,
}

/// Immutable context for order validation.
pub struct OrderValidationCtx<'a> {
    pub position_tracker: &'a dyn PositionTracker,
    pub enforce_limits: bool,
    pub timestamp: Timestamp,
    pub force_sequential: bool,
}

/// Result of order collection (before ID assignment).
pub struct OrderCollectionResult {
    pub orders_by_symbol: HashMap<Symbol, Vec<Order>>,
    pub total_orders: u64,
    pub rejected_orders: u64,
}

impl AuctionEngine {
    /// Create a new auction engine.
    pub fn new() -> Self {
        Self { next_order_id: 1 }
    }

    /// Generate the next order ID.
    pub fn next_order_id(&mut self) -> OrderId {
        let id = OrderId(self.next_order_id);
        self.next_order_id += 1;
        id
    }

    /// Get the current order ID counter (for external auction calls).
    pub fn order_id_counter(&self) -> u64 {
        self.next_order_id
    }

    /// Collect, validate, and assign IDs to orders from agent actions.
    ///
    /// Entire pipeline runs in parallel: expand agents → validate orders → collect.
    /// ID assignment uses atomic base + index for uniqueness without synchronization.
    pub fn collect_orders(
        &mut self,
        actions: Vec<AgentActionWithState>,
        ctx: &OrderValidationCtx<'_>,
    ) -> OrderCollectionResult {
        let base_id = self.next_order_id;
        let timestamp = ctx.timestamp;

        // Count total orders for ID space reservation
        let total_submitted: u64 = actions
            .iter()
            .map(|(_, a, _, _, _)| a.orders.len() as u64)
            .sum();

        // Parallel pipeline: expand → validate → collect
        // Uses enumerated index for ID assignment (no mutation during parallel phase)
        let valid_orders: Vec<Order> = parallel::flat_filter_map_vec(
            actions,
            // Expand: agent → orders with context and global index
            |(_, action, agent_positions, agent_cash, is_market_maker)| {
                action.orders.into_iter().map(move |order| {
                    let symbol_position = agent_positions.get(&order.symbol).copied().unwrap_or(0);
                    (order, symbol_position, agent_cash, is_market_maker)
                })
            },
            // Validate + assign ID
            |(order, symbol_position, agent_cash, is_market_maker)| {
                ctx.position_tracker
                    .validate_order(
                        &order,
                        symbol_position,
                        agent_cash,
                        is_market_maker,
                        ctx.enforce_limits,
                    )
                    .ok()
                    .map(|_| order)
            },
            ctx.force_sequential,
        );

        // Assign IDs sequentially (quick, and maintains deterministic ordering)
        let valid_orders: Vec<Order> = valid_orders
            .into_iter()
            .enumerate()
            .map(|(idx, mut order)| {
                order.id = OrderId(base_id + idx as u64);
                order.timestamp = timestamp;
                order
            })
            .collect();

        // Reserve ID space for next batch
        self.next_order_id += total_submitted;

        let rejected_count = total_submitted - valid_orders.len() as u64;

        // Group by symbol
        let orders_by_symbol: HashMap<Symbol, Vec<Order>> =
            valid_orders
                .into_iter()
                .fold(HashMap::new(), |mut acc, order| {
                    acc.entry(order.symbol.clone()).or_default().push(order);
                    acc
                });

        let total_orders: u64 = orders_by_symbol.values().map(|v| v.len() as u64).sum();

        OrderCollectionResult {
            orders_by_symbol,
            total_orders,
            rejected_orders: rejected_count,
        }
    }

    /// Build reference prices for batch auction clearing.
    pub fn build_reference_prices(
        &self,
        orders_by_symbol: &HashMap<Symbol, Vec<Order>>,
        market: &Market,
        symbol_configs: &[SymbolConfig],
        force_sequential: bool,
    ) -> HashMap<Symbol, Price> {
        let symbols: Vec<_> = orders_by_symbol.keys().collect();

        parallel::filter_map_to_hashmap(
            &symbols,
            |symbol| {
                let price = Self::compute_order_derived_anchor(orders_by_symbol.get(*symbol))
                    .or_else(|| {
                        // Blend mid_price (70%) with last_price (30%) for more permissive matching
                        market.get_book(symbol).and_then(|book| {
                            match (book.mid_price(), book.last_price()) {
                                (Some(mid), Some(last)) => {
                                    let blended = mid.to_float() * 0.7 + last.to_float() * 0.3;
                                    Some(Price::from_float(blended))
                                }
                                (Some(mid), None) => Some(mid),
                                (None, Some(last)) => Some(last),
                                (None, None) => None,
                            }
                        })
                    })
                    .or_else(|| {
                        symbol_configs
                            .iter()
                            .find(|sc| &sc.symbol == *symbol)
                            .map(|sc| sc.initial_price)
                    });
                price.map(|p| ((*symbol).clone(), p))
            },
            force_sequential,
        )
    }

    /// Compute order-derived anchor price.
    fn compute_order_derived_anchor(orders: Option<&Vec<Order>>) -> Option<Price> {
        let orders = orders?;
        if orders.is_empty() {
            return None;
        }

        let (bid_wgm, bid_vol) = Self::compute_side_wgm(orders, OrderSide::Buy);
        let (ask_wgm, ask_vol) = Self::compute_side_wgm(orders, OrderSide::Sell);

        match (bid_wgm, ask_wgm) {
            (Some(b_wgm), Some(a_wgm)) => {
                if b_wgm > a_wgm {
                    // Crossed market: use midpoint to ensure both sides qualify
                    Some(Price::from_float(
                        (b_wgm.to_float() + a_wgm.to_float()) / 2.0,
                    ))
                } else {
                    // Normal market: use volume-weighted blend
                    let b_val = b_wgm.to_float() * ask_vol as f64;
                    let a_val = a_wgm.to_float() * bid_vol as f64;
                    let total_vol = bid_vol + ask_vol;
                    Some(Price::from_float((b_val + a_val) / total_vol as f64))
                }
            }
            (Some(b_wgm), None) => Some(b_wgm),
            (None, Some(a_wgm)) => Some(a_wgm),
            (None, None) => None,
        }
    }

    fn compute_side_wgm(orders: &[Order], side: OrderSide) -> (Option<Price>, u64) {
        let (log_price_x_qty_sum, qty_sum) = orders
            .iter()
            .filter(|o| o.side == side)
            .filter_map(|o| o.limit_price().map(|p| (p, o.quantity.raw())))
            .fold((0.0, 0u64), |(lpq_sum, q_sum), (price, qty)| {
                // GM Logic: Sum(Weight * ln(Value))
                let log_price = price.to_float().ln();
                (lpq_sum + log_price * qty as f64, q_sum + qty)
            });

        if qty_sum == 0 {
            (None, 0)
        } else {
            // Finalize: exp(Sum / TotalWeight)
            let geometric_mean = (log_price_x_qty_sum / qty_sum as f64).exp();
            (Some(Price::from_float(geometric_mean)), qty_sum)
        }
    }

    /// Compute weighted average price and total volume for one side.
    #[allow(dead_code)]
    fn compute_side_wap(orders: &[Order], side: OrderSide) -> (Option<Price>, u64) {
        let (price_x_qty_sum, qty_sum) = orders
            .iter()
            .filter(|o| o.side == side)
            .filter_map(|o| o.limit_price().map(|p| (p, o.quantity.raw())))
            .fold((0.0, 0u64), |(pq_sum, q_sum), (price, qty)| {
                (pq_sum + price.to_float() * qty as f64, q_sum + qty)
            });

        if qty_sum == 0 {
            (None, 0)
        } else {
            (
                Some(Price::from_float(price_x_qty_sum / qty_sum as f64)),
                qty_sum,
            )
        }
    }

    /// Run batch auctions for all symbols.
    pub fn run_auctions(
        &self,
        orders: HashMap<Symbol, Vec<Order>>,
        reference_prices: &HashMap<Symbol, Price>,
        timestamp: Timestamp,
        tick: Tick,
        force_sequential: bool,
    ) -> HashMap<Symbol, BatchAuctionResult> {
        run_parallel_auctions(
            orders,
            reference_prices,
            timestamp,
            tick,
            self.next_order_id,
            force_sequential,
        )
    }
}

impl Default for AuctionEngine {
    fn default() -> Self {
        Self::new()
    }
}
