//! Storage layer for quant-trading-gym
//!
//! **Philosophy:** Declarative, Modular, SoC
//! - Declarative: Schema defined upfront, behavior driven by config
//! - Modular: Storage is swappable via SimulationHook trait
//! - SoC: This crate ONLY handles persistence, no simulation logic
//!
//! **V3.9 Scope:** Minimal storage for V4 bifurcation
//! - Trade history (append-only event log)
//! - Candle aggregation (time-series OLAP)
//! - Portfolio snapshots (analysis checkpoints)
//!
//! **V5.3 Scope:** ML training data recording
//! - Feature extraction (price, technical, fundamental, sentiment)
//! - Parquet output for Python ML consumption
//! - RecordingHook for capturing training data
//!
//! **V5.5.2 Scope:** Unified feature extraction
//! - Market features (42): `{name}_market.parquet` - 1 row per tick per symbol
//! - Agent features removed (tree agents only use market features)
//! - Uses types::features for training-serving parity

mod candles;
mod hook;
mod schema;

// V5.3/V5.5.2: ML training data recording
pub mod comprehensive_features;
pub mod parquet_writer;
pub mod recording_hook;

// Deprecated: PriceHistory was only used by MarketFeatures::extract() which has been removed.
// Kept as a module for now but not re-exported. Delete in next cleanup pass.
#[doc(hidden)]
pub mod price_history;

pub use hook::StorageHook;
pub use schema::StorageConfig;

// Re-export recording types
pub use comprehensive_features::MarketFeatures;
pub use parquet_writer::{MarketParquetWriter, MarketRecord, ParquetWriterError};
pub use recording_hook::{RecordingConfig, RecordingHook};

#[cfg(test)]
mod tests;
