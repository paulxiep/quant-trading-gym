//! Simulation hooks for observing simulation lifecycle events (V3.6).
//!
//! Hooks are **observers** that receive owned snapshots of simulation state
//! at key lifecycle points. They cannot modify simulation state.
//!
//! # Design Principles
//!
//! - **Declarative**: Hooks declare what events they care about via trait methods
//! - **Modular**: Each hook is independent; add/remove without affecting simulation
//! - **SoC**: Simulation owns state; hooks observe and report
//!
//! # Borrow-Checker Safety
//!
//! Hooks receive **owned data** (not references) to avoid lifetime complexity:
//! - Enables async processing (network hooks, persistence)
//! - Enables serialization without borrowing issues
//! - Hooks can store/process data independently
//!
//! # Example
//!
//! ```ignore
//! use simulation::hooks::{SimulationHook, HookContext};
//! use std::sync::atomic::{AtomicU64, Ordering};
//!
//! struct TradeCounter {
//!     count: AtomicU64,
//! }
//!
//! impl SimulationHook for TradeCounter {
//!     fn name(&self) -> &str { "TradeCounter" }
//!
//!     fn on_trades(&self, trades: Vec<types::Trade>, _ctx: &HookContext) {
//!         self.count.fetch_add(trades.len() as u64, Ordering::Relaxed);
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use quant::AgentRiskSnapshot;
use types::{
    AgentId, Candle, IndicatorType, Order, Price, Quantity, Symbol, Tick, Timestamp, Trade,
};

use crate::SimulationStats;

// ─────────────────────────────────────────────────────────────────────────────
// Hook Context
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of a single symbol's order book state.
#[derive(Debug, Clone)]
pub struct BookSnapshot {
    /// Best bid price (highest buy order).
    pub best_bid: Option<Price>,
    /// Best ask price (lowest sell order).
    pub best_ask: Option<Price>,
    /// Total bid volume at top 5 levels.
    pub bid_depth: Quantity,
    /// Total ask volume at top 5 levels.
    pub ask_depth: Quantity,
    /// Last trade price.
    pub last_price: Option<Price>,
}

impl Default for BookSnapshot {
    fn default() -> Self {
        Self {
            best_bid: None,
            best_ask: None,
            bid_depth: Quantity::ZERO,
            ask_depth: Quantity::ZERO,
            last_price: None,
        }
    }
}

/// Read-only snapshot of market state for hooks.
///
/// All data is **owned**, not borrowed, enabling:
/// - Serialization for network hooks
/// - Async processing
/// - Storage without lifetime constraints
#[derive(Debug, Clone, Default)]
pub struct MarketSnapshot {
    /// Per-symbol book snapshots.
    pub books: HashMap<Symbol, BookSnapshot>,
    /// Mid prices per symbol.
    pub mid_prices: HashMap<Symbol, Price>,
}

impl MarketSnapshot {
    /// Create a new empty market snapshot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a book snapshot for a symbol.
    ///
    /// Computes mid price from bid/ask if available, otherwise falls back to
    /// last_price. This handles batch auction markets where books are cleared
    /// between ticks.
    pub fn add_book(&mut self, symbol: Symbol, snapshot: BookSnapshot) {
        // Try to compute mid from bid/ask first
        let mid = match (snapshot.best_bid, snapshot.best_ask) {
            (Some(bid), Some(ask)) => Some(Price((bid.0 + ask.0) / 2)),
            _ => snapshot.last_price, // Fallback to last trade price
        };
        if let Some(price) = mid {
            self.mid_prices.insert(symbol.clone(), price);
        }
        self.books.insert(symbol, snapshot);
    }

    /// Get the mid price for a symbol.
    pub fn mid_price(&self, symbol: &Symbol) -> Option<Price> {
        self.mid_prices.get(symbol).copied()
    }

    /// Get book snapshot for a symbol.
    pub fn book(&self, symbol: &Symbol) -> Option<&BookSnapshot> {
        self.books.get(symbol)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Enriched Data (V4.4)
// ─────────────────────────────────────────────────────────────────────────────

use crate::traits::AgentSummary;

/// Enriched simulation data for hooks (V4.4).
///
/// Contains optional snapshots of candles, indicators, agents, etc.
/// Set by the simulation runner when calling `on_tick_end`.
///
/// V5.5: Indicators use `IndicatorType` enum keys for type safety.
/// Convert to string keys using `IndicatorType::to_key()` at JSON serialization boundary.
#[derive(Debug, Clone, Default)]
pub struct EnrichedData {
    /// Historical candles per symbol.
    pub candles: HashMap<Symbol, Vec<Candle>>,
    /// Indicator values per symbol with enum keys (V5.5).
    pub indicators: HashMap<Symbol, HashMap<IndicatorType, f64>>,
    /// Agent summaries.
    pub agent_summaries: Vec<AgentSummary>,
    /// Risk metrics per agent.
    pub risk_metrics: HashMap<AgentId, AgentRiskSnapshot>,
    /// Fair values per symbol from factor engine.
    pub fair_values: HashMap<Symbol, Price>,
    /// Active news events.
    pub news_events: Vec<NewsEventSnapshot>,
    /// Recent trades per symbol (pre-V6 refactor section F).
    pub recent_trades: HashMap<Symbol, Vec<Trade>>,
}

impl EnrichedData {
    /// Convert indicators to string keys for JSON serialization.
    ///
    /// Use this at the boundary when serializing to external systems.
    pub fn indicators_as_string_keys(&self) -> HashMap<Symbol, HashMap<String, f64>> {
        self.indicators
            .iter()
            .map(|(symbol, values)| {
                let string_values: HashMap<String, f64> = values
                    .iter()
                    .map(|(itype, v)| (itype.to_key(), *v))
                    .collect();
                (symbol.clone(), string_values)
            })
            .collect()
    }
}

/// Snapshot of a news event for hooks.
#[derive(Debug, Clone)]
pub struct NewsEventSnapshot {
    /// Event ID.
    pub id: u64,
    /// The underlying fundamental event.
    pub event: news::FundamentalEvent,
    /// Sentiment (-1.0 to 1.0).
    pub sentiment: f64,
    /// Magnitude (0.0 to 1.0).
    pub magnitude: f64,
    /// Tick when event started.
    pub start_tick: Tick,
    /// Duration in ticks.
    pub duration_ticks: u64,
}

/// Context passed to hooks at each lifecycle point.
///
/// Contains owned snapshots of simulation state at the time of the hook call.
/// Hooks can freely store, serialize, or process this data.
#[derive(Debug, Clone)]
pub struct HookContext {
    /// Current simulation tick.
    pub tick: Tick,
    /// Current timestamp.
    pub timestamp: Timestamp,
    /// Market state snapshot.
    pub market: MarketSnapshot,
    /// Number of T1 agents.
    pub tier1_count: usize,
    /// Number of T2 agents.
    pub tier2_count: usize,
    /// Number of T3 background pool agents.
    pub tier3_count: usize,
    /// Enriched data for V4.4 hooks (candles, indicators, agents, etc.)
    pub enriched: Option<EnrichedData>,
    /// Pre-extracted feature vectors per symbol (pre-V6 refactor section F).
    ///
    /// Separate from `enriched` by design: features are a computed artifact for
    /// ML and recording, while enriched data is observational state for display hooks.
    /// These may overlap but won't be the same — keeping them separate avoids
    /// recalculation and maintains SoC.
    ///
    /// `None` during warmup or when no `FeatureExtractor` is configured.
    pub features: Option<HashMap<Symbol, agents::FeatureVec>>,
}

impl HookContext {
    /// Create a new hook context.
    pub fn new(tick: Tick, timestamp: Timestamp) -> Self {
        Self {
            tick,
            timestamp,
            market: MarketSnapshot::new(),
            tier1_count: 0,
            tier2_count: 0,
            tier3_count: 0,
            enriched: None,
            features: None,
        }
    }

    /// Set the market snapshot.
    pub fn with_market(mut self, market: MarketSnapshot) -> Self {
        self.market = market;
        self
    }

    /// Set agent tier counts.
    pub fn with_tiers(mut self, t1: usize, t2: usize, t3: usize) -> Self {
        self.tier1_count = t1;
        self.tier2_count = t2;
        self.tier3_count = t3;
        self
    }

    /// Set pre-extracted feature vectors (pre-V6 refactor section F).
    pub fn with_features(mut self, features: HashMap<Symbol, agents::FeatureVec>) -> Self {
        self.features = Some(features);
        self
    }

    /// Set enriched data (V4.4).
    pub fn with_enriched(mut self, enriched: EnrichedData) -> Self {
        self.enriched = Some(enriched);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulationHook Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for simulation observers.
///
/// Hooks receive **owned snapshots** and cannot modify simulation state.
/// Use interior mutability (`Mutex`, `AtomicU64`, channels) for hook-owned state.
///
/// # Thread Safety
///
/// Hooks must be `Send + Sync` to support:
/// - Registration from any thread
/// - Invocation during parallel phases
/// - Future async hook implementations
///
/// # Lifecycle
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │  Simulation.step()                                              │
/// │                                                                 │
/// │  ┌─────────────────┐                                            │
/// │  │ on_tick_start() │ ← Before any agent execution               │
/// │  └────────┬────────┘                                            │
/// │           ▼                                                     │
/// │  ┌─────────────────────────┐                                    │
/// │  │ on_orders_collected()   │ ← After agents submit, before match│
/// │  └────────┬────────────────┘                                    │
/// │           ▼                                                     │
/// │  ┌─────────────────┐                                            │
/// │  │ on_trades()     │ ← After matching, with trade results       │
/// │  └────────┬────────┘                                            │
/// │           ▼                                                     │
/// │  ┌─────────────────┐                                            │
/// │  │ on_tick_end()   │ ← After all processing, with full stats    │
/// │  └─────────────────┘                                            │
/// │                                                                 │
/// └─────────────────────────────────────────────────────────────────┘
/// │
/// ▼ (after all ticks)
/// ┌───────────────────────┐
/// │ on_simulation_end()   │ ← Final stats, cleanup
/// └───────────────────────┘
/// ```
pub trait SimulationHook: Send + Sync {
    /// Human-readable name for logging and debugging.
    fn name(&self) -> &str;

    /// Called at the start of each tick, before agent execution.
    ///
    /// Use for: Tick timing, pre-tick state capture.
    #[allow(unused_variables)]
    fn on_tick_start(&self, ctx: &HookContext) {}

    /// Called after orders are collected, before matching.
    ///
    /// Receives owned vec of orders (cloned from submission).
    /// Use for: Order flow analysis, pre-trade logging.
    #[allow(unused_variables)]
    fn on_orders_collected(&self, orders: Vec<Order>, ctx: &HookContext) {}

    /// Called after matching completes with trades produced this tick.
    ///
    /// Receives owned vec of trades.
    /// Use for: Trade logging, P&L calculation, persistence.
    #[allow(unused_variables)]
    fn on_trades(&self, trades: Vec<Trade>, ctx: &HookContext) {}

    /// Called at the end of each tick with full statistics.
    ///
    /// Use for: Metrics aggregation, TUI updates, progress reporting.
    #[allow(unused_variables)]
    fn on_tick_end(&self, stats: &SimulationStats, ctx: &HookContext) {}

    /// Called once when simulation completes.
    ///
    /// Use for: Final reports, cleanup, summary statistics.
    #[allow(unused_variables)]
    fn on_simulation_end(&self, final_stats: &SimulationStats) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// HookRunner
// ─────────────────────────────────────────────────────────────────────────────

/// Manages hook registration and sequential invocation.
///
/// Hooks are called in registration order. Each hook call is synchronous;
/// for async behavior, hooks should use interior channels/queues.
#[derive(Default)]
pub struct HookRunner {
    hooks: Vec<Arc<dyn SimulationHook>>,
}

impl HookRunner {
    /// Create a new empty hook runner.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a hook. Hooks are called in registration order.
    pub fn add(&mut self, hook: Arc<dyn SimulationHook>) {
        self.hooks.push(hook);
    }

    /// Get the number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Check if no hooks are registered.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Get hook names for debugging.
    pub fn hook_names(&self) -> Vec<&str> {
        self.hooks.iter().map(|h| h.name()).collect()
    }

    /// Invoke `on_tick_start` on all hooks.
    pub fn on_tick_start(&self, ctx: &HookContext) {
        for hook in &self.hooks {
            hook.on_tick_start(ctx);
        }
    }

    /// Invoke `on_orders_collected` on all hooks.
    ///
    /// Clones orders for each hook to maintain owned-data contract.
    pub fn on_orders_collected(&self, orders: &[Order], ctx: &HookContext) {
        if self.hooks.is_empty() {
            return;
        }
        for hook in &self.hooks {
            hook.on_orders_collected(orders.to_vec(), ctx);
        }
    }

    /// Invoke `on_trades` on all hooks.
    ///
    /// Clones trades for each hook to maintain owned-data contract.
    pub fn on_trades(&self, trades: &[Trade], ctx: &HookContext) {
        if self.hooks.is_empty() {
            return;
        }
        for hook in &self.hooks {
            hook.on_trades(trades.to_vec(), ctx);
        }
    }

    /// Invoke `on_tick_end` on all hooks.
    pub fn on_tick_end(&self, stats: &SimulationStats, ctx: &HookContext) {
        for hook in &self.hooks {
            hook.on_tick_end(stats, ctx);
        }
    }

    /// Invoke `on_simulation_end` on all hooks.
    pub fn on_simulation_end(&self, final_stats: &SimulationStats) {
        for hook in &self.hooks {
            hook.on_simulation_end(final_stats);
        }
    }
}

impl std::fmt::Debug for HookRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRunner")
            .field("hooks", &self.hook_names())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Hooks
// ─────────────────────────────────────────────────────────────────────────────

/// A no-op hook useful for testing.
#[derive(Debug, Default)]
pub struct NoOpHook;

impl SimulationHook for NoOpHook {
    fn name(&self) -> &str {
        "NoOp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct CountingHook {
        tick_starts: AtomicU64,
        tick_ends: AtomicU64,
        trade_count: AtomicU64,
    }

    impl CountingHook {
        fn new() -> Self {
            Self {
                tick_starts: AtomicU64::new(0),
                tick_ends: AtomicU64::new(0),
                trade_count: AtomicU64::new(0),
            }
        }
    }

    impl SimulationHook for CountingHook {
        fn name(&self) -> &str {
            "CountingHook"
        }

        fn on_tick_start(&self, _ctx: &HookContext) {
            self.tick_starts.fetch_add(1, Ordering::Relaxed);
        }

        fn on_trades(&self, trades: Vec<Trade>, _ctx: &HookContext) {
            self.trade_count
                .fetch_add(trades.len() as u64, Ordering::Relaxed);
        }

        fn on_tick_end(&self, _stats: &SimulationStats, _ctx: &HookContext) {
            self.tick_ends.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn test_hook_runner_invocation() {
        let hook = Arc::new(CountingHook::new());
        let mut runner = HookRunner::new();
        runner.add(hook.clone());

        let ctx = HookContext::new(1, 1000);
        let stats = SimulationStats::default();

        runner.on_tick_start(&ctx);
        runner.on_tick_start(&ctx);
        runner.on_tick_end(&stats, &ctx);

        assert_eq!(hook.tick_starts.load(Ordering::Relaxed), 2);
        assert_eq!(hook.tick_ends.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_multiple_hooks() {
        let hook1 = Arc::new(CountingHook::new());
        let hook2 = Arc::new(CountingHook::new());

        let mut runner = HookRunner::new();
        runner.add(hook1.clone());
        runner.add(hook2.clone());

        let ctx = HookContext::new(1, 1000);
        runner.on_tick_start(&ctx);

        assert_eq!(hook1.tick_starts.load(Ordering::Relaxed), 1);
        assert_eq!(hook2.tick_starts.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_hook_names() {
        let mut runner = HookRunner::new();
        runner.add(Arc::new(NoOpHook));
        runner.add(Arc::new(CountingHook::new()));

        let names = runner.hook_names();
        assert_eq!(names, vec!["NoOp", "CountingHook"]);
    }
}
