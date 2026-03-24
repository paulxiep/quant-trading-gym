//! MetricsHook - Built-in hook for aggregating simulation statistics.
//!
//! Collects per-tick metrics and computes aggregate statistics.
//! Useful for performance analysis and post-simulation reports.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::SimulationStats;
use crate::hooks::{HookContext, SimulationHook};

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    /// Total ticks processed.
    pub total_ticks: u64,
    /// Total trades executed.
    pub total_trades: u64,
    /// Total orders submitted.
    pub total_orders: u64,
    /// Average trades per tick.
    pub avg_trades_per_tick: f64,
    /// Average orders per tick.
    pub avg_orders_per_tick: f64,
    /// Peak trades in a single tick.
    pub peak_trades_per_tick: u64,
    /// Peak orders in a single tick.
    pub peak_orders_per_tick: u64,
    /// Fill rate (filled orders / total orders).
    pub fill_rate: f64,
}

/// Built-in hook for collecting simulation metrics.
///
/// Thread-safe via atomics and mutex for interior mutability.
/// Designed for efficient per-tick updates.
///
/// # Example
///
/// ```ignore
/// use simulation::{MetricsHook, HookRunner};
/// use std::sync::Arc;
///
/// let metrics = Arc::new(MetricsHook::new());
/// let mut runner = HookRunner::new();
/// runner.add(metrics.clone());
///
/// // After simulation runs...
/// let snapshot = metrics.snapshot();
/// println!("Avg trades/tick: {:.2}", snapshot.avg_trades_per_tick);
/// ```
pub struct MetricsHook {
    /// Total ticks seen.
    tick_count: AtomicU64,
    /// Total trades seen.
    trade_count: AtomicU64,
    /// Total orders seen.
    order_count: AtomicU64,
    /// Peak trades in a single tick.
    peak_trades: AtomicU64,
    /// Peak orders in a single tick.
    peak_orders: AtomicU64,
    /// Running total of filled orders.
    filled_orders: AtomicU64,
    /// Per-tick trade counts (for variance calculation).
    /// Limited to max_history entries to bound memory.
    trade_history: Mutex<Vec<u64>>,
    /// Maximum history entries to keep.
    max_history: usize,
}

impl MetricsHook {
    /// Create a new metrics hook with default settings.
    pub fn new() -> Self {
        Self::with_max_history(10_000)
    }

    /// Create a metrics hook with custom history limit.
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            tick_count: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
            order_count: AtomicU64::new(0),
            peak_trades: AtomicU64::new(0),
            peak_orders: AtomicU64::new(0),
            filled_orders: AtomicU64::new(0),
            trade_history: Mutex::new(Vec::with_capacity(max_history.min(10_000))),
            max_history,
        }
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let total_ticks = self.tick_count.load(Ordering::Relaxed);
        let total_trades = self.trade_count.load(Ordering::Relaxed);
        let total_orders = self.order_count.load(Ordering::Relaxed);
        let filled = self.filled_orders.load(Ordering::Relaxed);

        let avg_trades = if total_ticks > 0 {
            total_trades as f64 / total_ticks as f64
        } else {
            0.0
        };

        let avg_orders = if total_ticks > 0 {
            total_orders as f64 / total_ticks as f64
        } else {
            0.0
        };

        let fill_rate = if total_orders > 0 {
            filled as f64 / total_orders as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            total_ticks,
            total_trades,
            total_orders,
            avg_trades_per_tick: avg_trades,
            avg_orders_per_tick: avg_orders,
            peak_trades_per_tick: self.peak_trades.load(Ordering::Relaxed),
            peak_orders_per_tick: self.peak_orders.load(Ordering::Relaxed),
            fill_rate,
        }
    }

    /// Get the trade count history (for variance/distribution analysis).
    pub fn trade_history(&self) -> Vec<u64> {
        self.trade_history.lock().clone()
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.tick_count.store(0, Ordering::Relaxed);
        self.trade_count.store(0, Ordering::Relaxed);
        self.order_count.store(0, Ordering::Relaxed);
        self.peak_trades.store(0, Ordering::Relaxed);
        self.peak_orders.store(0, Ordering::Relaxed);
        self.filled_orders.store(0, Ordering::Relaxed);
        self.trade_history.lock().clear();
    }

    /// Update peak value atomically (CAS loop).
    fn update_peak(peak: &AtomicU64, value: u64) {
        let mut current = peak.load(Ordering::Relaxed);
        while value > current {
            match peak.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

impl Default for MetricsHook {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulationHook for MetricsHook {
    fn name(&self) -> &str {
        "Metrics"
    }

    fn on_orders_collected(&self, orders: Vec<types::Order>, _ctx: &HookContext) {
        let count = orders.len() as u64;
        self.order_count.fetch_add(count, Ordering::Relaxed);
        Self::update_peak(&self.peak_orders, count);
    }

    fn on_trades(&self, trades: Vec<types::Trade>, _ctx: &HookContext) {
        let count = trades.len() as u64;
        self.trade_count.fetch_add(count, Ordering::Relaxed);
        self.filled_orders.fetch_add(count * 2, Ordering::Relaxed); // 2 fills per trade
        Self::update_peak(&self.peak_trades, count);

        // Record in history (bounded)
        let mut history = self.trade_history.lock();
        if history.len() < self.max_history {
            history.push(count);
        }
    }

    fn on_tick_end(&self, _stats: &SimulationStats, _ctx: &HookContext) {
        self.tick_count.fetch_add(1, Ordering::Relaxed);
    }

    fn on_simulation_end(&self, _final_stats: &SimulationStats) {
        // Could log final summary here if desired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::HookRunner;
    use std::sync::Arc;
    use types::{AgentId, Order, OrderSide, Price, Quantity, Trade, TradeId};

    fn make_order() -> Order {
        Order::limit(
            AgentId(1),
            "TEST",
            OrderSide::Buy,
            Price(1000),
            Quantity(10),
        )
    }

    fn make_trade() -> Trade {
        Trade {
            id: TradeId(1),
            symbol: "TEST".to_string(),
            price: Price(1000),
            quantity: Quantity(10),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: types::OrderId(1),
            seller_order_id: types::OrderId(2),
            timestamp: 0,
            tick: 0,
        }
    }

    #[test]
    fn test_metrics_accumulation() {
        let metrics = Arc::new(MetricsHook::new());
        let mut runner = HookRunner::new();
        runner.add(metrics.clone());

        let ctx = HookContext::new(1, 1000);
        let stats = SimulationStats::default();

        // Simulate 3 ticks
        for _ in 0..3 {
            runner.on_orders_collected(&[make_order(), make_order()], &ctx);
            runner.on_trades(&[make_trade()], &ctx);
            runner.on_tick_end(&stats, &ctx);
        }

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_ticks, 3);
        assert_eq!(snapshot.total_trades, 3);
        assert_eq!(snapshot.total_orders, 6);
        assert!((snapshot.avg_trades_per_tick - 1.0).abs() < 0.001);
        assert!((snapshot.avg_orders_per_tick - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_peak_tracking() {
        let metrics = Arc::new(MetricsHook::new());
        let ctx = HookContext::new(1, 1000);

        // First tick: 2 trades
        metrics.on_trades(vec![make_trade(), make_trade()], &ctx);

        // Second tick: 5 trades (new peak)
        metrics.on_trades(vec![make_trade(); 5], &ctx);

        // Third tick: 1 trade (no peak update)
        metrics.on_trades(vec![make_trade()], &ctx);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.peak_trades_per_tick, 5);
        assert_eq!(snapshot.total_trades, 8);
    }

    #[test]
    fn test_reset() {
        let metrics = MetricsHook::new();
        let ctx = HookContext::new(1, 1000);
        let stats = SimulationStats::default();

        metrics.on_trades(vec![make_trade()], &ctx);
        metrics.on_tick_end(&stats, &ctx);

        assert_eq!(metrics.snapshot().total_ticks, 1);

        metrics.reset();

        assert_eq!(metrics.snapshot().total_ticks, 0);
        assert_eq!(metrics.snapshot().total_trades, 0);
    }
}
