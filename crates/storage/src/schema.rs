//! Database schema and configuration
//!
//! **SoC:** This module ONLY defines schema, no business logic

use rusqlite::Connection;
use std::path::Path;

/// Storage configuration (declarative)
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Path to SQLite database (`:memory:` for in-memory)
    pub path: String,
    /// Candle timeframes in ticks (e.g., [100, 300, 3600])
    pub candle_timeframes: Vec<u64>,
    /// Snapshot interval in ticks (default: 1000)
    pub snapshot_interval: u64,
    /// Trade write interval in ticks (default: 100) - buffer trades and flush every N ticks
    pub trade_write_interval: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: ":memory:".to_string(),
            candle_timeframes: vec![100, 300, 3600], // Aligned with trade_write_interval
            snapshot_interval: 1000,
            trade_write_interval: 100, // Flush trades every 100 ticks
        }
    }
}

impl StorageConfig {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            ..Default::default()
        }
    }
}

/// Initialize database with schema
pub fn init_schema(conn: &Connection) -> rusqlite::Result<()> {
    // Trade history (append-only event log)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            buyer_id INTEGER NOT NULL,
            seller_id INTEGER NOT NULL,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )",
        [],
    )?;

    // Index for queries by tick and symbol
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_tick ON trades(tick)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
        [],
    )?;

    // Candle aggregation (time-series OLAP)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe INTEGER NOT NULL,
            tick_start INTEGER NOT NULL,
            open INTEGER NOT NULL,
            high INTEGER NOT NULL,
            low INTEGER NOT NULL,
            close INTEGER NOT NULL,
            volume INTEGER NOT NULL,
            UNIQUE(symbol, timeframe, tick_start)
        )",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(symbol, timeframe, tick_start)",
        [],
    )?;

    // Portfolio snapshots (analysis checkpoints)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            agent_id INTEGER NOT NULL,
            cash INTEGER NOT NULL,
            positions_json TEXT NOT NULL,
            realized_pnl INTEGER NOT NULL,
            equity INTEGER NOT NULL,
            UNIQUE(tick, agent_id)
        )",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_snapshots_tick ON portfolio_snapshots(tick)",
        [],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Verify tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();

        assert!(tables.contains(&"trades".to_string()));
        assert!(tables.contains(&"candles".to_string()));
        assert!(tables.contains(&"portfolio_snapshots".to_string()));
    }
}
