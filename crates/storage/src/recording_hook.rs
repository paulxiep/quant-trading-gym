//! Recording hook for ML training data capture.
//!
//! Pre-V6 refactor (section F): Reads pre-extracted features from `HookContext.features`
//! instead of extracting features internally. This guarantees training-serving parity
//! by construction — the same `FeatureExtractor` produces features for both ML inference
//! and recording. RecordingHook is now a simple Parquet writer.
//!
//! ## Output Files
//!
//! - `{name}_market.parquet`: Market features (1 row per tick per symbol)

use parking_lot::Mutex;
use simulation::{HookContext, SimulationHook, SimulationStats};
use types::Tick;

use crate::parquet_writer::{MarketParquetWriter, MarketRecord, ParquetWriterError};

/// Configuration for the recording hook.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Output Parquet file path (base name, creates `{name}_market.parquet`).
    pub output_path: String,
    /// Skip first N ticks before recording (warmup period).
    pub warmup: u64,
    /// Record every N ticks (1 = every tick).
    pub interval: u64,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            output_path: "data/training.parquet".to_string(),
            warmup: 100,
            interval: 1,
        }
    }
}

impl RecordingConfig {
    /// Create a new recording config with the given output path.
    pub fn new(output_path: impl Into<String>) -> Self {
        Self {
            output_path: output_path.into(),
            ..Default::default()
        }
    }

    /// Set the warmup period.
    pub fn with_warmup(mut self, warmup: u64) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set the recording interval.
    pub fn with_interval(mut self, interval: u64) -> Self {
        self.interval = interval;
        self
    }
}

/// Internal state protected by Mutex.
struct RecordingState {
    /// Parquet writer.
    writer: Option<MarketParquetWriter>,
    /// Configuration.
    config: RecordingConfig,
    /// Error encountered (if any).
    #[allow(dead_code)]
    error: Option<String>,
}

/// Recording hook for ML training data.
///
/// Reads pre-extracted features from `HookContext.features` and writes to
/// Parquet files. Does NOT extract features itself — the runner handles
/// extraction, imputation, and passes features via the hook context.
///
/// # Lifecycle
///
/// 1. `on_tick_end()`: Read pre-extracted features, write market records
/// 2. `on_simulation_end()`: Final flush and close files
pub struct RecordingHook {
    state: Mutex<RecordingState>,
}

impl RecordingHook {
    /// Create a new recording hook with custom feature names for Parquet schema.
    ///
    /// Pass `extractor.feature_names()` to use an extractor's schema, or
    /// `MarketFeatures::default_feature_names()` for V5 compatibility.
    pub fn new(
        config: RecordingConfig,
        feature_names: &[&str],
    ) -> Result<Self, ParquetWriterError> {
        let writer = MarketParquetWriter::new(&config.output_path, feature_names)?;

        Ok(Self {
            state: Mutex::new(RecordingState {
                writer: Some(writer),
                config,
                error: None,
            }),
        })
    }

    /// Check if a tick should be recorded based on warmup and interval.
    fn should_record(tick: Tick, config: &RecordingConfig) -> bool {
        if tick < config.warmup {
            return false;
        }
        let adjusted_tick = tick - config.warmup;
        adjusted_tick.is_multiple_of(config.interval)
    }
}

impl SimulationHook for RecordingHook {
    fn name(&self) -> &str {
        "RecordingHook"
    }

    fn on_tick_end(&self, _stats: &SimulationStats, ctx: &HookContext) {
        let mut state = self.state.lock();

        if !Self::should_record(ctx.tick, &state.config) {
            return;
        }

        // Read pre-extracted features (guaranteed parity with ML serving path)
        let features_map = match ctx.features.as_ref() {
            Some(f) => f,
            None => return, // No features extracted (warmup or no extractor)
        };

        let tick = ctx.tick;
        let market_records: Vec<MarketRecord> = features_map
            .iter()
            .map(|(symbol, features)| MarketRecord {
                tick,
                symbol: symbol.to_string(),
                features: features.to_vec(),
            })
            .collect();

        if let Some(ref mut writer) = state.writer {
            for record in market_records {
                if let Err(e) = writer.write(record) {
                    state.error = Some(e.to_string());
                    return;
                }
            }
        }
    }

    fn on_simulation_end(&self, _final_stats: &SimulationStats) {
        let mut state = self.state.lock();

        // Finish writing
        if let Some(writer) = state.writer.take() {
            match writer.finish() {
                Ok(count) => {
                    eprintln!("[RecordingHook] Finished: {} market rows", count);
                }
                Err(e) => {
                    eprintln!("[RecordingHook] Error closing Parquet file: {}", e);
                    state.error = Some(e.to_string());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_record() {
        let config = RecordingConfig {
            warmup: 100,
            interval: 5,
            ..Default::default()
        };

        // Before warmup
        assert!(!RecordingHook::should_record(0, &config));
        assert!(!RecordingHook::should_record(50, &config));
        assert!(!RecordingHook::should_record(99, &config));

        // At and after warmup
        assert!(RecordingHook::should_record(100, &config)); // 100 - 100 = 0, 0 % 5 == 0
        assert!(!RecordingHook::should_record(101, &config)); // 1 % 5 != 0
        assert!(!RecordingHook::should_record(104, &config)); // 4 % 5 != 0
        assert!(RecordingHook::should_record(105, &config)); // 5 % 5 == 0
        assert!(RecordingHook::should_record(110, &config)); // 10 % 5 == 0
    }

    #[test]
    fn test_config_builder() {
        let config = RecordingConfig::new("output.parquet")
            .with_warmup(200)
            .with_interval(10);

        assert_eq!(config.output_path, "output.parquet");
        assert_eq!(config.warmup, 200);
        assert_eq!(config.interval, 10);
    }
}
