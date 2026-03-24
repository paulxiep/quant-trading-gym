//! Auctioneer trait for order collection and batch auction execution.

use std::collections::HashMap;

use sim_core::{BatchAuctionResult, Market};
use types::{Order, Price, Symbol, SymbolConfig, Tick, Timestamp};

use super::agents::AgentActionWithState;
use super::risk::PositionTracker;
use crate::SimulationStats;

/// Handles order collection and batch auction execution.
///
/// This trait abstracts the auction mechanism, enabling:
/// - Swapping auction algorithms (continuous, call, pro-rata)
/// - Testing order validation with mock position trackers
/// - A/B testing of auction mechanisms
pub trait Auctioneer {
    /// Collect and validate orders from agent actions.
    ///
    /// Processes cancellations and validates new orders against position limits.
    fn collect_orders(
        &mut self,
        actions: Vec<AgentActionWithState>,
        market: &mut Market,
        position_tracker: &dyn PositionTracker,
        enforce_limits: bool,
        verbose: bool,
        stats: &mut SimulationStats,
    ) -> HashMap<Symbol, Vec<Order>>;

    /// Build reference prices for batch auction clearing.
    fn build_reference_prices(
        &self,
        orders: &HashMap<Symbol, Vec<Order>>,
        market: &Market,
        symbol_configs: &[SymbolConfig],
        force_sequential: bool,
    ) -> HashMap<Symbol, Price>;

    /// Run batch auctions for all symbols.
    fn run_auctions(
        &mut self,
        orders: HashMap<Symbol, Vec<Order>>,
        reference_prices: &HashMap<Symbol, Price>,
        timestamp: Timestamp,
        tick: Tick,
        force_sequential: bool,
    ) -> HashMap<Symbol, BatchAuctionResult>;
}
