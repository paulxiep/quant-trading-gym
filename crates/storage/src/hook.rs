//! Storage hook implementation
//!
//! **Philosophy:** Declarative, Modular, SoC
//! - Implements SimulationHook trait (modular)
//! - Config-driven behavior (declarative)
//! - Only persistence logic, no simulation concerns (SoC)

use parking_lot::Mutex;
use rusqlite::Connection;
use serde_json::json;
use simulation::{HookContext, SimulationHook, SimulationStats};
use std::collections::HashMap;
use types::{Symbol, Trade};

use crate::candles::CandleAggregator;
use crate::schema::{StorageConfig, init_schema};

/// Storage hook for persisting simulation data
///
/// Uses interior mutability (Mutex) because SimulationHook requires &self.
/// Buffers trades and flushes every `trade_write_interval` ticks for performance.
pub struct StorageHook {
    config: StorageConfig,
    conn: Mutex<Connection>,
    aggregators: Mutex<HashMap<u64, CandleAggregator>>,
    /// Buffered trades waiting to be flushed (tick, trade)
    trade_buffer: Mutex<Vec<(u64, Trade)>>,
}

impl StorageHook {
    /// Create new storage hook with config
    pub fn new(config: StorageConfig) -> rusqlite::Result<Self> {
        let conn = if config.path == ":memory:" {
            Connection::open_in_memory()?
        } else {
            Connection::open(&config.path)?
        };

        init_schema(&conn)?;

        // Create aggregators for each timeframe
        let aggregators: HashMap<u64, CandleAggregator> = config
            .candle_timeframes
            .iter()
            .map(|&tf| (tf, CandleAggregator::new(tf)))
            .collect();

        Ok(Self {
            config,
            conn: Mutex::new(conn),
            aggregators: Mutex::new(aggregators),
            trade_buffer: Mutex::new(Vec::with_capacity(10_000)), // Pre-allocate for performance
        })
    }

    /// Create storage hook from path (uses default config)
    pub fn from_path(path: impl AsRef<std::path::Path>) -> rusqlite::Result<Self> {
        Self::new(StorageConfig::from_path(path))
    }

    /// Persist trades in a single batched transaction (V4.5 fix)
    fn flush_trade_buffer(&self) -> rusqlite::Result<()> {
        let mut buffer = self.trade_buffer.lock();
        if buffer.is_empty() {
            return Ok(());
        }

        let conn = self.conn.lock();
        // Use a transaction for batch insert (much faster than individual inserts)
        conn.execute("BEGIN TRANSACTION", [])?;

        for (tick, trade) in buffer.iter() {
            conn.execute(
                "INSERT INTO trades (tick, symbol, price, quantity, buyer_id, seller_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    *tick as i64,
                    trade.symbol.as_str(),
                    trade.price.0,
                    i64::try_from(trade.quantity.0).unwrap_or(0),
                    i64::try_from(trade.buyer_id.0).unwrap_or(0),
                    i64::try_from(trade.seller_id.0).unwrap_or(0),
                ],
            )?;
        }

        conn.execute("COMMIT", [])?;
        buffer.clear(); // Clear buffer after successful flush
        Ok(())
    }

    /// Flush completed candles to database
    fn flush_candles(&self) -> rusqlite::Result<()> {
        let mut aggregators = self.aggregators.lock();
        let conn = self.conn.lock();

        for agg in aggregators.values_mut() {
            let completed = agg.flush();
            for (symbol, tick_start, candle) in completed {
                conn.execute(
                    "INSERT OR REPLACE INTO candles
                     (symbol, timeframe, tick_start, open, high, low, close, volume)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        symbol.as_str(),
                        agg.timeframe() as i64,
                        tick_start as i64,
                        candle.open.0,
                        candle.high.0,
                        candle.low.0,
                        candle.close.0,
                        i64::try_from(candle.volume.0).unwrap(),
                    ],
                )?;
            }
        }

        Ok(())
    }

    /// Persist portfolio snapshots for all agents
    #[allow(dead_code)] // Will be used in main.rs integration
    fn persist_snapshots(
        &self,
        tick: u64,
        summaries: &[(String, HashMap<Symbol, i64>, types::Cash, types::Cash)],
    ) -> rusqlite::Result<()> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "INSERT OR REPLACE INTO portfolio_snapshots
             (tick, agent_id, cash, positions_json, realized_pnl, equity)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;

        for (idx, (_, positions, cash, total_pnl)) in summaries.iter().enumerate() {
            // Convert positions HashMap to JSON
            let positions_json = json!(positions).to_string();

            // For V3.9: realized_pnl == total_pnl (no separate tracking yet)
            // equity = cash + unrealized P&L (simplified: use total_pnl)
            stmt.execute(rusqlite::params![
                tick as i64,
                idx as i64, // Use index as agent_id (simplified)
                cash.0,
                positions_json,
                total_pnl.0,
                cash.0 + total_pnl.0, // Approximation for equity
            ])?;
        }

        Ok(())
    }
}

impl SimulationHook for StorageHook {
    fn name(&self) -> &str {
        "StorageHook"
    }

    fn on_trades(&self, trades: Vec<Trade>, ctx: &HookContext) {
        // Buffer trades for batched writing (V4.5 perf fix)
        {
            let mut buffer = self.trade_buffer.lock();
            for trade in trades.iter().cloned() {
                buffer.push((ctx.tick, trade));
            }
        }

        // Update candle aggregators (always, for real-time candle generation)
        let mut aggregators = self.aggregators.lock();
        for trade in &trades {
            for agg in aggregators.values_mut() {
                agg.process_trade(ctx.tick, trade.symbol.clone(), trade.price, trade.quantity);
            }
        }
    }

    fn on_tick_end(&self, stats: &SimulationStats, _ctx: &HookContext) {
        // Flush trade buffer at configured interval (every N ticks)
        if stats.tick > 0
            && stats.tick.is_multiple_of(self.config.trade_write_interval)
            && let Err(e) = self.flush_trade_buffer()
        {
            eprintln!("[StorageHook] Failed to flush trades: {}", e);
        }

        // Flush completed candles
        if let Err(e) = self.flush_candles() {
            eprintln!("[StorageHook] Failed to flush candles: {}", e);
        }

        // Persist portfolio snapshots at configured interval
        if stats.tick.is_multiple_of(self.config.snapshot_interval) {
            // NOTE: We need access to Simulation.agent_summaries() here
            // This will be passed via context in main.rs integration
            // For now, this is a placeholder - actual integration in next step
        }
    }

    fn on_simulation_end(&self, _final_stats: &SimulationStats) {
        // Final flush of any remaining buffered trades
        if let Err(e) = self.flush_trade_buffer() {
            eprintln!("[StorageHook] Failed final trade flush: {}", e);
        }
        // Final candle flush
        if let Err(e) = self.flush_candles() {
            eprintln!("[StorageHook] Failed final candle flush: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{AgentId, Price};

    #[test]
    fn test_storage_hook_creation() {
        let config = StorageConfig::default();
        let hook = StorageHook::new(config).unwrap();
        assert_eq!(hook.name(), "StorageHook");
    }

    #[test]
    fn test_trade_persistence_via_buffer() {
        use types::{OrderId, TradeId};

        let hook = StorageHook::new(StorageConfig::default()).unwrap();

        let trade = Trade {
            id: TradeId(1),
            symbol: Symbol::from("AAPL"),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: OrderId(100),
            seller_order_id: OrderId(200),
            price: Price::from(100_0000),
            quantity: types::Quantity(10),
            timestamp: 0,
            tick: 0,
        };

        // Use on_trades to buffer the trade (current V4.5 API)
        let ctx = HookContext::new(0, 0);
        hook.on_trades(vec![trade], &ctx);

        // Flush the buffer to persist
        hook.flush_trade_buffer().unwrap();

        // Verify trade was written
        let conn = hook.conn.lock();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM trades", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
}
