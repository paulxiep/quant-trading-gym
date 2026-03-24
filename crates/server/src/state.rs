//! Shared server state (V4.2, extended V4.3).
//!
//! Contains channels, metrics, and simulation data shared across handlers.
//!
//! # Design Principles
//!
//! - **Declarative**: State is data, handlers extract what they need
//! - **Modular**: State independent of route logic
//! - **SoC**: State holds references, doesn't own simulation
//!
//! # V4.3 Additions
//!
//! - `SimData`: Cached simulation data for analytics/portfolio/risk endpoints
//! - Updated by `DataServiceHook` on each tick

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::{RwLock, broadcast};

use crate::bridge::{SimCommand, TickData};
use quant::AgentRiskSnapshot;
use types::{AgentId, Candle, Price, Symbol};

// =============================================================================
// SimData - Cached simulation data for V4.3 Data Service
// =============================================================================

/// Agent position data for API responses.
#[derive(Debug, Clone, Default)]
pub struct AgentPosition {
    pub symbol: Symbol,
    pub quantity: i64,
    pub avg_cost: f64,
}

/// Agent data snapshot for API responses.
#[derive(Debug, Clone)]
pub struct AgentData {
    pub id: u64,
    pub name: String,
    pub cash: f64,
    pub equity: f64,
    pub total_pnl: f64,
    pub realized_pnl: f64,
    pub positions: HashMap<Symbol, AgentPosition>,
    pub is_market_maker: bool,
    pub is_ml_agent: bool,
    pub tier: u8,
}

/// News event data for API responses (mirrors news::NewsEvent).
#[derive(Debug, Clone)]
pub struct NewsEventSnapshot {
    pub id: u64,
    pub event: news::FundamentalEvent,
    pub sentiment: f64,
    pub magnitude: f64,
    pub start_tick: u64,
    pub duration_ticks: u64,
}

impl NewsEventSnapshot {
    /// Check if the event is active at the given tick.
    pub fn is_active(&self, tick: u64) -> bool {
        tick >= self.start_tick && tick < self.start_tick + self.duration_ticks
    }

    /// Get the decay factor at the given tick.
    pub fn decay_factor(&self, tick: u64) -> f64 {
        if !self.is_active(tick) {
            return 0.0;
        }
        let elapsed = (tick - self.start_tick) as f64;
        let total = self.duration_ticks as f64;
        1.0 - (elapsed / total)
    }

    /// Get the effective sentiment at the given tick.
    pub fn effective_sentiment(&self, tick: u64) -> f64 {
        self.sentiment * self.decay_factor(tick)
    }

    /// Get the primary symbol affected, if any.
    pub fn symbol(&self) -> Option<&Symbol> {
        self.event.symbol()
    }

    /// Get the sector affected, if any.
    pub fn sector(&self) -> Option<types::Sector> {
        self.event.sector()
    }
}

/// Cached simulation data for V4.3 Data Service endpoints.
///
/// Updated by `DataServiceHook` on each tick. Read by API handlers.
///
/// # V4.4 Note on Order Book Data
///
/// In batch auction mode, order book is cleared after each tick. The `order_distribution`
/// field captures pre-auction order demand/supply by price level, aggregated from
/// `on_orders_collected` hook (before auction runs). This shows order flow intent
/// rather than resting liquidity.
#[derive(Debug, Default)]
pub struct SimData {
    /// Current tick number.
    pub tick: u64,
    /// Candles per symbol.
    pub candles: HashMap<Symbol, Vec<Candle>>,
    /// Technical indicators per symbol (indicator_name -> value).
    pub indicators: HashMap<Symbol, HashMap<String, f64>>,
    /// Current prices per symbol.
    pub prices: HashMap<Symbol, Price>,
    /// Fair values per symbol.
    pub fair_values: HashMap<Symbol, Price>,
    /// Agent data.
    pub agents: Vec<AgentData>,
    /// Risk metrics per agent.
    pub risk_metrics: HashMap<AgentId, AgentRiskSnapshot>,
    /// Equity curves per agent (for portfolio detail).
    pub equity_curves: HashMap<AgentId, Vec<f64>>,
    /// Active news events.
    pub active_events: Vec<NewsEventSnapshot>,
    /// Pre-auction order distribution by symbol (V4.4).
    /// Captured in `on_orders_collected` before batch auction.
    /// Format: symbol -> (bid_levels, ask_levels) where level = (price, total_quantity)
    pub order_distribution: HashMap<Symbol, OrderDistribution>,
}

/// Pre-auction order distribution for a symbol (V4.4).
///
/// Represents aggregated buy/sell orders by price level before batch auction.
/// Used for "order book depth" visualization in UI.
#[derive(Debug, Clone, Default)]
pub struct OrderDistribution {
    /// Bid (buy) orders aggregated by price level: (price, total_quantity)
    pub bids: Vec<(Price, u64)>,
    /// Ask (sell) orders aggregated by price level: (price, total_quantity)
    pub asks: Vec<(Price, u64)>,
}

impl SimData {
    /// Create empty SimData.
    pub fn new() -> Self {
        Self::default()
    }
}

// =============================================================================
// ServerState
// =============================================================================

/// Shared state for all route handlers.
///
/// Cloned into each handler via Axum's State extractor.
#[derive(Clone)]
pub struct ServerState {
    /// Broadcast channel for tick updates (simulation → clients).
    pub tick_tx: broadcast::Sender<TickData>,

    /// Command sender (server → simulation).
    pub cmd_tx: crossbeam_channel::Sender<SimCommand>,

    /// Server start time.
    pub start_time: Instant,

    /// Shared metrics.
    pub metrics: Arc<ServerMetrics>,

    /// V4.3: Cached simulation data for data service endpoints.
    pub sim_data: Arc<RwLock<SimData>>,
}

impl ServerState {
    /// Create new server state with channels.
    pub fn new(
        tick_tx: broadcast::Sender<TickData>,
        cmd_tx: crossbeam_channel::Sender<SimCommand>,
    ) -> Self {
        Self {
            tick_tx,
            cmd_tx,
            start_time: Instant::now(),
            metrics: Arc::new(ServerMetrics::new()),
            sim_data: Arc::new(RwLock::new(SimData::new())),
        }
    }

    /// Create new server state with pre-initialized SimData.
    pub fn with_sim_data(
        tick_tx: broadcast::Sender<TickData>,
        cmd_tx: crossbeam_channel::Sender<SimCommand>,
        sim_data: Arc<RwLock<SimData>>,
    ) -> Self {
        Self {
            tick_tx,
            cmd_tx,
            start_time: Instant::now(),
            metrics: Arc::new(ServerMetrics::new()),
            sim_data,
        }
    }

    /// Get uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Subscribe to tick updates.
    pub fn subscribe_ticks(&self) -> broadcast::Receiver<TickData> {
        self.tick_tx.subscribe()
    }

    /// Send command to simulation.
    pub fn send_command(
        &self,
        cmd: SimCommand,
    ) -> Result<(), crossbeam_channel::SendError<SimCommand>> {
        self.cmd_tx.send(cmd)
    }
}

/// Server-side metrics.
pub struct ServerMetrics {
    /// Current tick from simulation.
    pub current_tick: AtomicU64,
    /// Total agents in simulation.
    pub total_agents: AtomicU64,
    /// Whether simulation is running.
    pub sim_running: AtomicBool,
    /// Whether simulation has finished.
    pub sim_finished: AtomicBool,
    /// Active WebSocket connections.
    pub ws_connections: AtomicU64,
}

impl ServerMetrics {
    /// Create new metrics.
    pub fn new() -> Self {
        Self {
            current_tick: AtomicU64::new(0),
            total_agents: AtomicU64::new(0),
            sim_running: AtomicBool::new(false),
            sim_finished: AtomicBool::new(false),
            ws_connections: AtomicU64::new(0),
        }
    }

    /// Update from simulation update.
    pub fn update_from_tick(&self, tick: u64, agents: u64, running: bool, finished: bool) {
        self.current_tick.store(tick, Ordering::Relaxed);
        self.total_agents.store(agents, Ordering::Relaxed);
        self.sim_running.store(running, Ordering::Relaxed);
        self.sim_finished.store(finished, Ordering::Relaxed);
    }

    /// Increment WebSocket connection count.
    pub fn ws_connect(&self) {
        self.ws_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement WebSocket connection count.
    pub fn ws_disconnect(&self) {
        self.ws_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current tick.
    pub fn tick(&self) -> u64 {
        self.current_tick.load(Ordering::Relaxed)
    }

    /// Get agent count.
    pub fn agents(&self) -> u64 {
        self.total_agents.load(Ordering::Relaxed)
    }

    /// Check if simulation is running.
    pub fn is_running(&self) -> bool {
        self.sim_running.load(Ordering::Relaxed)
    }

    /// Check if simulation has finished.
    pub fn is_finished(&self) -> bool {
        self.sim_finished.load(Ordering::Relaxed)
    }

    /// Get WebSocket connection count.
    pub fn ws_count(&self) -> u64 {
        self.ws_connections.load(Ordering::Relaxed)
    }
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_update() {
        let metrics = ServerMetrics::new();
        metrics.update_from_tick(100, 25000, true, false);

        assert_eq!(metrics.tick(), 100);
        assert_eq!(metrics.agents(), 25000);
        assert!(metrics.is_running());
        assert!(!metrics.is_finished());
    }

    #[test]
    fn test_ws_connections() {
        let metrics = ServerMetrics::new();
        assert_eq!(metrics.ws_count(), 0);

        metrics.ws_connect();
        metrics.ws_connect();
        assert_eq!(metrics.ws_count(), 2);

        metrics.ws_disconnect();
        assert_eq!(metrics.ws_count(), 1);
    }
}
