//! Simulation configuration options.

use types::{Price, Quantity, ShortSellingConfig, Symbol, SymbolConfig};

/// Configuration for parallelization behavior (V3.7).
///
/// Allows fine-grained control over which phases of the simulation
/// run in parallel vs sequential. Useful for profiling and performance tuning.
#[derive(Debug, Clone)]
pub struct ParallelizationConfig {
    /// Phase 4: Collect agent actions in parallel
    pub parallel_agent_collection: bool,
    /// Phase 3: Build indicator snapshot in parallel per-symbol
    pub parallel_indicators: bool,
    /// Phase 5: Validate orders in parallel
    pub parallel_order_validation: bool,
    /// Phase 6: Run batch auctions in parallel across symbols
    pub parallel_auctions: bool,
    /// Phase 9: Update candles in parallel per-symbol
    pub parallel_candle_updates: bool,
    /// Phase 9: Update recent trades in parallel per-symbol
    pub parallel_trade_updates: bool,
    /// Phase 10: Process fill notifications in parallel
    pub parallel_fill_notifications: bool,
    /// Phase 10: Restore T2 wake conditions in parallel
    pub parallel_wake_conditions: bool,
    /// Phase 11: Update risk tracking in parallel
    pub parallel_risk_tracking: bool,
    /// Phase 3b: Extract ML features in parallel per-symbol (pre-V6)
    pub parallel_features: bool,
}

impl Default for ParallelizationConfig {
    fn default() -> Self {
        Self {
            // Enable all parallel phases by default
            parallel_agent_collection: true,
            parallel_indicators: true,
            parallel_order_validation: true,
            parallel_auctions: true,
            parallel_candle_updates: true,
            parallel_trade_updates: true,
            parallel_fill_notifications: true,
            parallel_wake_conditions: true,
            parallel_risk_tracking: true,
            parallel_features: true,
        }
    }
}

impl ParallelizationConfig {
    /// All phases run sequentially (for benchmarking baseline)
    pub fn all_sequential() -> Self {
        Self {
            parallel_agent_collection: false,
            parallel_indicators: false,
            parallel_order_validation: false,
            parallel_auctions: false,
            parallel_candle_updates: false,
            parallel_trade_updates: false,
            parallel_fill_notifications: false,
            parallel_wake_conditions: false,
            parallel_risk_tracking: false,
            parallel_features: false,
        }
    }

    /// All phases run in parallel (default)
    pub fn all_parallel() -> Self {
        Self::default()
    }
}

/// Configuration for the simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Symbol configurations (supports multiple symbols).
    pub symbol_configs: Vec<SymbolConfig>,

    /// Short-selling configuration (enabled, limits, rates).
    pub short_selling: ShortSellingConfig,

    /// Number of book levels to include in snapshots.
    pub snapshot_depth: usize,

    /// Maximum number of recent trades to keep in market data.
    pub max_recent_trades: usize,

    /// Number of ticks per candle (for OHLCV aggregation).
    pub candle_interval: u64,

    /// Maximum number of candles to keep in history.
    pub max_candles: usize,

    /// Whether to validate orders against position limits.
    /// When disabled, orders are processed without constraint checks.
    pub enforce_position_limits: bool,

    /// Enable verbose logging.
    pub verbose: bool,

    /// News event generation configuration (V2.4).
    pub news: news::NewsGeneratorConfig,

    /// Fair value drift configuration (V2.5).
    pub fair_value_drift: news::FairValueDriftConfig,

    /// Random seed for deterministic simulation.
    pub seed: u64,

    /// Parallelization configuration (V3.7).
    pub parallelization: ParallelizationConfig,

    /// Number of ticks to wait before ML models make predictions (V5.6).
    /// During warmup, indicators have insufficient data and produce NaN values.
    pub ml_warmup_ticks: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            symbol_configs: vec![SymbolConfig::default()],
            short_selling: ShortSellingConfig::disabled(),
            snapshot_depth: 10,
            max_recent_trades: 100,
            candle_interval: 4, // Aggregate 4 ticks per candle (meaningful OHLC)
            max_candles: 200,   // 200 candles = 800 ticks of history
            enforce_position_limits: true,
            verbose: false,
            news: news::NewsGeneratorConfig::default(),
            fair_value_drift: news::FairValueDriftConfig::default(),
            seed: rand::random(),
            parallelization: ParallelizationConfig::default(),
            ml_warmup_ticks: 1000,
        }
    }
}

impl SimulationConfig {
    /// Create a new configuration with the given symbol.
    pub fn new(symbol: impl Into<String>) -> Self {
        let symbol_config = SymbolConfig {
            symbol: symbol.into(),
            ..SymbolConfig::default()
        };
        Self {
            symbol_configs: vec![symbol_config],
            ..Default::default()
        }
    }

    /// Create a configuration with multiple symbols.
    pub fn with_symbols(symbols: Vec<SymbolConfig>) -> Self {
        Self {
            symbol_configs: symbols,
            ..Default::default()
        }
    }

    /// Get all symbol names.
    pub fn symbols(&self) -> Vec<Symbol> {
        self.symbol_configs
            .iter()
            .map(|c| c.symbol.clone())
            .collect()
    }

    /// Get the primary (first) symbol name.
    pub fn symbol(&self) -> &str {
        &self.symbol_configs[0].symbol
    }

    /// Get the primary symbol's initial/reference price.
    pub fn initial_price(&self) -> Price {
        self.symbol_configs[0].initial_price
    }

    /// Get symbol config by symbol name.
    pub fn get_symbol_config(&self, symbol: &str) -> Option<&SymbolConfig> {
        self.symbol_configs.iter().find(|c| c.symbol == symbol)
    }

    /// Get all symbol configs.
    pub fn get_symbol_configs(&self) -> &[SymbolConfig] {
        &self.symbol_configs
    }

    /// Set a single symbol configuration (replaces all).
    pub fn with_symbol_config(mut self, config: SymbolConfig) -> Self {
        self.symbol_configs = vec![config];
        self
    }

    /// Add a symbol configuration.
    pub fn add_symbol_config(mut self, config: SymbolConfig) -> Self {
        self.symbol_configs.push(config);
        self
    }

    /// Set the initial price for the primary symbol.
    pub fn with_initial_price(mut self, price: Price) -> Self {
        if !self.symbol_configs.is_empty() {
            self.symbol_configs[0].initial_price = price;
        }
        self
    }

    /// Set shares outstanding for the primary symbol.
    pub fn with_shares_outstanding(mut self, shares: Quantity) -> Self {
        if !self.symbol_configs.is_empty() {
            self.symbol_configs[0].shares_outstanding = shares;
        }
        self
    }

    /// Set short-selling configuration.
    pub fn with_short_selling(mut self, config: ShortSellingConfig) -> Self {
        self.short_selling = config;
        self
    }

    /// Enable short selling with default settings.
    pub fn with_short_selling_enabled(mut self) -> Self {
        self.short_selling = ShortSellingConfig::enabled_default();
        self
    }

    /// Set the snapshot depth.
    pub fn with_snapshot_depth(mut self, depth: usize) -> Self {
        self.snapshot_depth = depth;
        self
    }

    /// Set the maximum recent trades.
    pub fn with_max_recent_trades(mut self, max: usize) -> Self {
        self.max_recent_trades = max;
        self
    }

    /// Set the candle interval (ticks per candle).
    pub fn with_candle_interval(mut self, interval: u64) -> Self {
        self.candle_interval = interval;
        self
    }

    /// Set the maximum number of candles to keep.
    pub fn with_max_candles(mut self, max: usize) -> Self {
        self.max_candles = max;
        self
    }

    /// Enable or disable position limit enforcement.
    pub fn with_position_limits(mut self, enforce: bool) -> Self {
        self.enforce_position_limits = enforce;
        self
    }

    /// Enable verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set news event generation configuration (V2.4).
    pub fn with_news_config(mut self, news: news::NewsGeneratorConfig) -> Self {
        self.news = news;
        self
    }

    /// Disable news event generation.
    pub fn with_news_disabled(mut self) -> Self {
        self.news = news::NewsGeneratorConfig::disabled();
        self
    }

    /// Set fair value drift configuration (V2.5).
    pub fn with_fair_value_drift(mut self, config: news::FairValueDriftConfig) -> Self {
        self.fair_value_drift = config;
        self
    }

    /// Disable fair value drift (for deterministic tests).
    pub fn with_fair_value_drift_disabled(mut self) -> Self {
        self.fair_value_drift = news::FairValueDriftConfig::disabled();
        self
    }

    /// Set random seed for deterministic simulation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set parallelization configuration (V3.7).
    pub fn with_parallelization(mut self, config: ParallelizationConfig) -> Self {
        self.parallelization = config;
        self
    }
}
