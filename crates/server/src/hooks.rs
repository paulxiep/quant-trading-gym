//! SimulationHook implementations for broadcasting updates (V4.2, V4.4).
//!
//! Provides the bridge from sync simulation to async server via hooks.
//!
//! # Architecture
//!
//! ```text
//! Simulation (sync)          BroadcastHook           Server (async)
//!       │                         │                       │
//!       │── on_tick_end() ───────▶│                       │
//!       │                         │── tick_tx.send() ────▶│
//!       │                         │                       │── ws broadcast
//!
//! Simulation (sync)          DataServiceHook         REST Handlers
//!       │                         │                       │
//!       │── on_tick_end() ───────▶│                       │
//!       │                         │── update SimData ────▶│
//!       │                         │    (RwLock write)     │── read SimData
//! ```
//!
//! # Design Principles
//!
//! - **Declarative**: Hook declares what events it handles
//! - **Modular**: Hook is self-contained, no dependencies on server internals
//! - **SoC**: Hook observes simulation, server distributes updates
//!
//! # V4.4 DataServiceHook
//!
//! The DataServiceHook updates SimData cache with simulation state on each tick.
//! Uses the enriched HookContext from V4.4 which includes candles, indicators,
//! agent summaries, risk metrics, and fair values.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use simulation::{AgentSummary, HookContext, SimulationHook, SimulationStats};
use tokio::sync::{RwLock, broadcast};
use types::{AgentId, Order, OrderSide, OrderType, Symbol};

use crate::bridge::{SymbolData, TickData};
use crate::state::{AgentData, AgentPosition, NewsEventSnapshot, OrderDistribution, SimData};

/// Hook that broadcasts simulation updates to the server.
///
/// Implements SimulationHook to observe tick events and broadcast
/// via tokio broadcast channel to async WebSocket handlers.
pub struct BroadcastHook {
    /// Broadcast sender for tick data.
    tick_tx: broadcast::Sender<TickData>,
    /// Total agent count (set once at start).
    total_agents: AtomicU64,
    /// Whether simulation is running.
    running: AtomicBool,
    /// Whether simulation has finished.
    finished: AtomicBool,
}

impl BroadcastHook {
    /// Create a new broadcast hook.
    pub fn new(tick_tx: broadcast::Sender<TickData>) -> Self {
        Self {
            tick_tx,
            total_agents: AtomicU64::new(0),
            running: AtomicBool::new(false),
            finished: AtomicBool::new(false),
        }
    }

    /// Set total agent count.
    pub fn set_agents(&self, count: u64) {
        self.total_agents.store(count, Ordering::Relaxed);
    }

    /// Set running state.
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// Set finished state.
    pub fn set_finished(&self, finished: bool) {
        self.finished.store(finished, Ordering::Relaxed);
    }

    /// Get sender for server state.
    pub fn sender(&self) -> broadcast::Sender<TickData> {
        self.tick_tx.clone()
    }

    /// Build TickData from hook context and stats.
    fn build_tick_data(&self, stats: &SimulationStats, ctx: &HookContext) -> TickData {
        let mut symbols = HashMap::new();

        // Build per-symbol data from market snapshot
        for (symbol, book) in &ctx.market.books {
            let data = SymbolData::new(symbol.clone())
                .with_prices(book.last_price, book.best_bid, book.best_ask)
                .with_depth(book.bid_depth.0, book.ask_depth.0);
            symbols.insert(symbol.clone(), data);
        }

        TickData {
            tick: ctx.tick,
            timestamp: ctx.timestamp,
            symbols,
            trades_this_tick: 0, // TODO: track from on_trades
            total_trades: stats.total_trades,
            total_orders: stats.total_orders,
            agents_called: stats.agents_called_this_tick,
        }
    }
}

impl SimulationHook for BroadcastHook {
    fn name(&self) -> &str {
        "BroadcastHook"
    }

    fn on_tick_end(&self, stats: &SimulationStats, ctx: &HookContext) {
        let tick_data = self.build_tick_data(stats, ctx);

        // Fire-and-forget: if no receivers, drop the message
        let _ = self.tick_tx.send(tick_data);
    }

    fn on_simulation_end(&self, _final_stats: &SimulationStats) {
        self.set_finished(true);
        self.set_running(false);
    }
}

// Required for Arc<dyn SimulationHook>
unsafe impl Send for BroadcastHook {}
unsafe impl Sync for BroadcastHook {}

// =============================================================================
// DataServiceHook (V4.4)
// =============================================================================

/// Hook that updates SimData cache for REST API endpoints (V4.4).
///
/// On each tick, this hook:
/// 1. Reads prices from HookContext (market snapshot)
/// 2. Reads enriched data from HookContext (candles, indicators, agents, etc.)
/// 3. Writes to SimData (Arc<RwLock>) for REST handlers to read
///
/// # Thread Safety
///
/// - Hook is called from simulation thread (sync)
/// - SimData is read by Axum handlers (async)
/// - RwLock ensures safe concurrent access
pub struct DataServiceHook {
    /// Shared SimData cache for REST endpoints.
    sim_data: Arc<RwLock<SimData>>,
    /// Update interval (only update SimData every N ticks for performance).
    update_interval: u64,
}

impl DataServiceHook {
    /// Create a new DataServiceHook.
    ///
    /// # Arguments
    /// - `sim_data`: Shared SimData from ServerState
    pub fn new(sim_data: Arc<RwLock<SimData>>) -> Self {
        Self {
            sim_data,
            update_interval: 1,
        }
    }

    /// Create with custom update interval.
    pub fn with_interval(sim_data: Arc<RwLock<SimData>>, interval: u64) -> Self {
        Self {
            sim_data,
            update_interval: interval.max(1),
        }
    }

    /// Update order distribution from pre-auction orders (V4.4).
    ///
    /// Called from `on_orders_collected` to capture order flow before batch auction.
    /// Aggregates orders by symbol and price level for "order book depth" visualization.
    fn update_order_distribution(&self, orders: Vec<Order>) {
        use std::collections::BTreeMap;

        // Group orders by symbol
        let mut by_symbol: HashMap<Symbol, Vec<&Order>> = HashMap::new();
        for order in &orders {
            by_symbol
                .entry(order.symbol.clone())
                .or_default()
                .push(order);
        }

        // Build distribution per symbol
        let mut distributions: HashMap<Symbol, OrderDistribution> = HashMap::new();

        for (symbol, symbol_orders) in by_symbol {
            let mut bids: BTreeMap<types::Price, u64> = BTreeMap::new();
            let mut asks: BTreeMap<types::Price, u64> = BTreeMap::new();

            for order in symbol_orders {
                // Only process limit orders (have price)
                let price = match &order.order_type {
                    OrderType::Limit { price } => *price,
                    OrderType::Market => continue, // Skip market orders (no price level)
                };

                let map = match order.side {
                    OrderSide::Buy => &mut bids,
                    OrderSide::Sell => &mut asks,
                };
                *map.entry(price).or_default() += order.quantity.raw();
            }

            // Convert to sorted vecs (bids descending, asks ascending)
            let bids_vec: Vec<(types::Price, u64)> = bids.into_iter().rev().collect();
            let asks_vec: Vec<(types::Price, u64)> = asks.into_iter().collect();

            distributions.insert(
                symbol,
                OrderDistribution {
                    bids: bids_vec,
                    asks: asks_vec,
                },
            );
        }

        // Write to SimData
        if let Ok(mut data) = self.sim_data.try_write() {
            data.order_distribution = distributions;
        }
    }

    /// Update SimData from enriched hook context (V4.4).
    fn update_sim_data(&self, ctx: &HookContext, enriched: &simulation::EnrichedData) {
        // Build prices from hook context (declarative)
        let prices: HashMap<Symbol, types::Price> = ctx
            .market
            .mid_prices
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        // Convert agent summaries to AgentData (parallel for large agent counts)
        // Enumerate inline to avoid intermediate allocation
        let agents: Vec<AgentData> = parallel::map_slice(
            &enriched.agent_summaries,
            |summary| Self::summary_to_agent_data(summary, &prices),
            false, // use parallel when available
        );

        // Build equity curves from risk metrics
        let equity_curves: HashMap<AgentId, Vec<f64>> = enriched
            .risk_metrics
            .keys()
            .map(|id| (*id, Vec::new())) // TODO: Track equity history
            .collect();

        // Convert news events to our NewsEventSnapshot
        let active_events: Vec<NewsEventSnapshot> = enriched
            .news_events
            .iter()
            .map(|e| NewsEventSnapshot {
                id: e.id,
                event: e.event.clone(),
                sentiment: e.sentiment,
                magnitude: e.magnitude,
                start_tick: e.start_tick,
                duration_ticks: e.duration_ticks,
            })
            .collect();

        // Write to SimData (blocking write from sync context)
        // Using try_write to avoid blocking if readers are active
        if let Ok(mut data) = self.sim_data.try_write() {
            data.tick = ctx.tick;
            data.candles = enriched.candles.clone();
            // V5.5: Convert IndicatorType keys to strings at JSON boundary
            data.indicators = enriched.indicators_as_string_keys();
            data.prices = prices;
            data.fair_values = enriched
                .fair_values
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            data.agents = agents;
            data.risk_metrics = enriched.risk_metrics.clone();
            data.equity_curves = equity_curves;
            data.active_events = active_events;
        }
        // If try_write fails, skip this update (next tick will catch up)
    }

    /// Convert an AgentSummary to AgentData (pure function, parallelizable).
    ///
    /// Factored out for parallel processing of large agent counts.
    /// Follows SoC: transformation logic is separate from collection iteration.
    fn summary_to_agent_data(
        summary: &AgentSummary,
        prices: &HashMap<Symbol, types::Price>,
    ) -> AgentData {
        // Determine agent tier from name (heuristic)
        let tier = if summary.name.contains("Reactive") || summary.name.contains("SectorRotator") {
            2
        } else if summary.name.contains("T3") || summary.name.contains("Pool") {
            3
        } else {
            1
        };

        // Calculate equity from positions
        let position_value: f64 = summary
            .positions
            .iter()
            .map(|(sym, qty)| {
                let price = prices.get(sym).map(|p| p.to_float()).unwrap_or(100.0);
                *qty as f64 * price
            })
            .sum();
        let equity = summary.cash.to_float() + position_value;

        // Convert positions to AgentPosition format
        let agent_positions: HashMap<Symbol, AgentPosition> = summary
            .positions
            .iter()
            .map(|(sym, qty)| {
                (
                    sym.clone(),
                    AgentPosition {
                        symbol: sym.clone(),
                        quantity: *qty,
                        avg_cost: 0.0, // Not available from summary
                    },
                )
            })
            .collect();

        AgentData {
            id: summary.id.0,
            name: summary.name.clone(),
            cash: summary.cash.to_float(),
            equity,
            total_pnl: summary.total_pnl.to_float(),
            realized_pnl: 0.0, // Not available from summary
            positions: agent_positions,
            is_market_maker: summary.is_market_maker,
            is_ml_agent: summary.is_ml_agent,
            tier,
        }
    }
}

impl SimulationHook for DataServiceHook {
    fn name(&self) -> &str {
        "DataServiceHook"
    }

    fn on_orders_collected(&self, orders: Vec<Order>, _ctx: &HookContext) {
        // Capture pre-auction order distribution for "order book depth" visualization.
        // In batch auction mode, this shows order flow intent before clearing.
        self.update_order_distribution(orders);
    }

    fn on_tick_end(&self, _stats: &SimulationStats, ctx: &HookContext) {
        // Rate limit updates
        if !ctx.tick.is_multiple_of(self.update_interval) {
            return;
        }

        // Get enriched data from context (V4.4)
        let Some(enriched) = ctx.enriched.as_ref() else {
            return; // No enriched data, skip update
        };

        // Update SimData from enriched context
        self.update_sim_data(ctx, enriched);
    }
}

// Required for Arc<dyn SimulationHook>
unsafe impl Send for DataServiceHook {}
unsafe impl Sync for DataServiceHook {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_hook_creation() {
        let (tx, _rx) = broadcast::channel(16);
        let hook = BroadcastHook::new(tx);

        assert_eq!(hook.name(), "BroadcastHook");
    }

    #[test]
    fn test_broadcast_hook_state() {
        let (tx, _rx) = broadcast::channel(16);
        let hook = BroadcastHook::new(tx);

        hook.set_agents(25000);
        hook.set_running(true);

        assert_eq!(hook.total_agents.load(Ordering::Relaxed), 25000);
        assert!(hook.running.load(Ordering::Relaxed));
    }

    #[test]
    fn test_tick_data_broadcast() {
        let (tx, mut rx) = broadcast::channel(16);
        let hook = BroadcastHook::new(tx);

        // Create minimal context
        let ctx = HookContext::new(100, 1000);
        let stats = SimulationStats {
            tick: 100,
            total_trades: 500,
            total_orders: 1000,
            ..Default::default()
        };

        hook.on_tick_end(&stats, &ctx);

        // Verify message was sent
        let received = rx.try_recv().unwrap();
        assert_eq!(received.tick, 100);
        assert_eq!(received.total_trades, 500);
    }
}
