//! Agent-level configuration for the Quant Trading Gym simulation.
//!
//! Contains `SimConfig` (master agent configuration), `SymbolSpec` (symbol definitions),
//! and `Tier1AgentType` (spawnable agent types). These types are needed by both the
//! binary (CLI) and the gym crate (episode configuration).
//!
//! # Configuration Strategy
//! 1. Specify minimum count for each specific agent type
//! 2. Specify minimum total for each tier
//! 3. If tier minimum not met by specific agents, random tier agents are spawned

use std::collections::HashMap;

use agents::tier3::MarketRegime;
use rand::Rng;
use rand::prelude::SliceRandom;
use types::{Cash, Price, Sector};

/// Configuration for a single symbol.
#[derive(Debug, Clone)]
pub struct SymbolSpec {
    /// Symbol ticker (e.g., "AAPL", "GOOG").
    pub symbol: String,
    /// Initial price of the asset.
    pub initial_price: Price,
    /// Industry sector for news events and grouping (V2.4).
    pub sector: Sector,
}

impl SymbolSpec {
    /// Create a new symbol specification.
    pub fn new(symbol: impl Into<String>, initial_price: f64) -> Self {
        Self {
            symbol: symbol.into(),
            initial_price: Price::from_float(initial_price),
            sector: Sector::Tech, // Default sector
        }
    }

    /// Create a symbol specification with explicit sector.
    pub fn with_sector(symbol: impl Into<String>, initial_price: f64, sector: Sector) -> Self {
        Self {
            symbol: symbol.into(),
            initial_price: Price::from_float(initial_price),
            sector,
        }
    }
}

/// Types of Tier 1 agents that can be randomly spawned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier1AgentType {
    NoiseTrader,
    MomentumTrader,
    TrendFollower,
    MacdTrader,
    BollingerTrader,
    VwapExecutor,
    PairsTrading, // V3.3: multi-symbol pairs trading
}

impl Tier1AgentType {
    /// All spawnable Tier 1 agent types (excludes MarketMaker as it's infrastructure).
    /// Note: PairsTrading excluded from random spawn (requires 2-symbol pairs config).
    pub const SPAWNABLE: &'static [Tier1AgentType] = &[
        Tier1AgentType::NoiseTrader,
        Tier1AgentType::MomentumTrader,
        Tier1AgentType::TrendFollower,
        Tier1AgentType::MacdTrader,
        Tier1AgentType::BollingerTrader,
        Tier1AgentType::VwapExecutor,
    ];

    /// Pick a random spawnable agent type.
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        *Self::SPAWNABLE.choose(rng).unwrap()
    }
}

/// Master configuration for the entire simulation.
#[derive(Debug, Clone)]
pub struct SimConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Simulation Control
    // ─────────────────────────────────────────────────────────────────────────
    /// Symbols to trade (first symbol is primary for single-symbol simulation).
    pub symbols: Vec<SymbolSpec>,
    /// Total ticks to run (0 = infinite).
    pub total_ticks: u64,
    /// Delay between ticks in milliseconds (0 = fastest).
    pub tick_delay_ms: u64,
    /// Enable verbose logging.
    pub verbose: bool,
    /// Maximum CPU usage percentage (1-100). Controls rayon thread pool size.
    pub max_cpu_percent: u8,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 1 Agent Counts (minimum for each type)
    // ─────────────────────────────────────────────────────────────────────────
    /// Minimum number of market makers.
    pub num_market_makers: usize,
    /// Minimum number of noise traders.
    pub num_noise_traders: usize,
    /// Minimum number of momentum (RSI) traders.
    pub num_momentum_traders: usize,
    /// Minimum number of trend followers (SMA crossover).
    pub num_trend_followers: usize,
    /// Minimum number of MACD crossover traders.
    pub num_macd_traders: usize,
    /// Minimum number of Bollinger reversion traders.
    pub num_bollinger_traders: usize,
    /// Minimum number of VWAP executors.
    pub num_vwap_executors: usize,
    /// Minimum number of pairs trading agents (V3.3 multi-symbol).
    pub num_pairs_traders: usize,
    /// Number of sector rotator agents (V3.3 multi-symbol, subset of Tier 2 budget).
    /// Deducted from `num_tier2_agents` — the remainder become reactive agents.
    pub num_sector_rotators: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 1 ML Agent Counts (V5.5, V6.2 HashMap refactor)
    // ─────────────────────────────────────────────────────────────────────────
    /// ML agent counts by model type (e.g., "decision_tree" -> 400).
    /// Agents for each type are distributed round-robin across loaded models of that type.
    /// Adding new model types requires only adding an entry here.
    pub ml_agent_counts: HashMap<String, usize>,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier Minimums
    // ─────────────────────────────────────────────────────────────────────────
    /// Minimum total Tier 1 agents. If specific agent types don't reach this,
    /// random Tier 1 agents are spawned to fill the gap.
    pub min_tier1_agents: usize,

    /// Total number of Tier 2 agents (reactive + sector rotators).
    /// Sector rotators consume part of this budget; the remainder are reactive agents.
    pub num_tier2_agents: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 2 Reactive Agent Parameters (V3.2)
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for each Tier 2 reactive agent.
    pub t2_initial_cash: Cash,
    /// Maximum position size for Tier 2 agents.
    pub t2_max_position: u64,
    /// ThresholdBuyer: minimum buy price (dollars).
    pub t2_buy_threshold_min: f64,
    /// ThresholdBuyer: maximum buy price (dollars).
    pub t2_buy_threshold_max: f64,
    /// StopLoss: minimum stop percentage (e.g., 0.02 = 2%).
    pub t2_stop_loss_min: f64,
    /// StopLoss: maximum stop percentage (e.g., 0.08 = 8%).
    pub t2_stop_loss_max: f64,
    /// TakeProfit: minimum target percentage (e.g., 0.10 = 10%).
    pub t2_take_profit_min: f64,
    /// TakeProfit: maximum target percentage (e.g., 0.30 = 30%).
    pub t2_take_profit_max: f64,
    /// ThresholdSeller: minimum sell price (dollars).
    pub t2_sell_threshold_min: f64,
    /// ThresholdSeller: maximum sell price (dollars).
    pub t2_sell_threshold_max: f64,
    /// Probability of having TakeProfit vs ThresholdSeller (0.0 - 1.0).
    pub t2_take_profit_prob: f64,
    /// Probability of having NewsReactor strategy (0.0 - 1.0).
    pub t2_news_reactor_prob: f64,
    /// ThresholdBuyer: minimum order size fraction (0.0 - 1.0).
    pub t2_order_size_min: f64,
    /// ThresholdBuyer: maximum order size fraction (0.0 - 1.0).
    pub t2_order_size_max: f64,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 3 Background Pool Parameters (V3.4)
    // ─────────────────────────────────────────────────────────────────────────
    /// Enable background pool (statistical order generation).
    pub enable_background_pool: bool,
    /// Number of simulated background agents (scales order rate).
    pub background_pool_size: usize,
    /// Market regime for background pool behavior.
    pub background_regime: MarketRegime,
    /// Mean order size for background pool.
    pub t3_mean_order_size: f64,
    /// Maximum order size for background pool.
    pub t3_max_order_size: u64,
    /// Order size standard deviation for background pool.
    pub t3_order_size_stddev: f64,
    /// Base activity rate override (None = use regime default).
    pub t3_base_activity: Option<f64>,
    /// Price spread lambda (higher = tighter around mid price).
    pub t3_price_spread_lambda: f64,
    /// Maximum price deviation from mid (as fraction).
    pub t3_max_price_deviation: f64,

    // ─────────────────────────────────────────────────────────────────────────
    // Market Maker Parameters
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for each market maker.
    pub mm_initial_cash: Cash,
    /// Starting position in shares (the "float" each MM provides).
    pub mm_initial_position: i64,
    /// Half-spread as a fraction (e.g., 0.0025 = 0.25%).
    pub mm_half_spread: f64,
    /// Number of shares to quote on each side.
    pub mm_quote_size: u64,
    /// Ticks between quote refreshes.
    pub mm_refresh_interval: u64,
    /// Maximum inventory before reducing quotes.
    pub mm_max_inventory: i64,
    /// Price adjustment per unit of inventory.
    pub mm_inventory_skew: f64,
    /// Maximum long position for market makers.
    pub mm_max_long_position: i64,
    /// Maximum short position for market makers (as positive number).
    pub mm_max_short_position: i64,

    // ─────────────────────────────────────────────────────────────────────────
    // Noise Trader Parameters
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for each noise trader.
    pub nt_initial_cash: Cash,
    /// Starting position in shares (allows balanced buy/sell from start).
    pub nt_initial_position: i64,
    /// Probability of placing an order each tick (0.0 - 1.0).
    pub nt_order_probability: f64,
    /// Maximum price deviation from mid price as a fraction.
    pub nt_price_deviation: f64,
    /// Minimum order quantity.
    pub nt_min_quantity: u64,
    /// Maximum order quantity.
    pub nt_max_quantity: u64,
    /// Maximum long position for noise traders.
    pub nt_max_long_position: i64,
    /// Maximum short position for noise traders (as positive number).
    pub nt_max_short_position: i64,

    // ─────────────────────────────────────────────────────────────────────────
    // Quant Strategy Parameters (shared defaults)
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for quant strategies.
    pub quant_initial_cash: Cash,
    /// Order size for quant strategies.
    pub quant_order_size: u64,
    /// Maximum long position for quant strategies.
    pub quant_max_long_position: i64,
    /// Maximum short position for quant strategies (as positive number).
    pub quant_max_short_position: i64,

    // ─────────────────────────────────────────────────────────────────────────
    // Tree Agent Parameters (V5.5)
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for tree-based ML agents.
    pub tree_agent_initial_cash: Cash,
    /// Order size for tree agents.
    pub tree_agent_order_size: u64,
    /// Maximum long position per symbol for tree agents.
    pub tree_agent_max_long_position: i64,
    /// Maximum short position per symbol for tree agents (as positive number).
    pub tree_agent_max_short_position: i64,
    /// Buy threshold probability (0.5-1.0).
    pub tree_agent_buy_threshold: f64,
    /// Sell threshold probability (0.5-1.0).
    pub tree_agent_sell_threshold: f64,

    // ─────────────────────────────────────────────────────────────────────────
    // TUI Parameters
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum price history points to display.
    pub max_price_history: usize,
    /// TUI display frame rate (frames per second).
    pub tui_frame_rate: u64,
    /// Simulation data update rate (updates per second).
    pub data_update_rate: u64,

    // ─────────────────────────────────────────────────────────────────────────
    // Event/News Generation Parameters (V2.4)
    // ─────────────────────────────────────────────────────────────────────────
    /// Enable event/news generation.
    pub events_enabled: bool,
    /// Earnings event probability per tick.
    pub event_earnings_prob: f64,
    /// Minimum ticks between earnings events.
    pub event_earnings_interval: u64,
    /// Guidance event probability per tick.
    pub event_guidance_prob: f64,
    /// Minimum ticks between guidance events.
    pub event_guidance_interval: u64,
    /// Rate decision probability per tick.
    pub event_rate_decision_prob: f64,
    /// Minimum ticks between rate decisions.
    pub event_rate_decision_interval: u64,
    /// Sector news probability per tick.
    pub event_sector_news_prob: f64,
    /// Minimum ticks between sector news.
    pub event_sector_news_interval: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            symbols: vec![
                SymbolSpec::with_sector("Duck Delish", 100.0, Sector::Consumer),
                SymbolSpec::with_sector("Picky Piglet", 100.0, Sector::Consumer),
                SymbolSpec::with_sector("Vraiment Villa", 100.0, Sector::RealEstate),
                SymbolSpec::with_sector("Meilleur Mansion", 100.0, Sector::RealEstate),
                SymbolSpec::with_sector("Quant Quotation", 100.0, Sector::Finance),
                SymbolSpec::with_sector("Lumen Ledger", 100.0, Sector::Finance),
                SymbolSpec::with_sector("Hello Handy", 100.0, Sector::Communications),
                SymbolSpec::with_sector("Thalas Telecoms", 100.0, Sector::Communications),
                SymbolSpec::with_sector("Zephyr Zap", 100.0, Sector::Utilities),
                SymbolSpec::with_sector("Aeon Anthracite", 100.0, Sector::Utilities),
            ],
            total_ticks: 10000,
            tick_delay_ms: 0,
            verbose: false,
            max_cpu_percent: 75,
            num_market_makers: 600,
            num_noise_traders: 2400,
            num_momentum_traders: 800,
            num_trend_followers: 800,
            num_macd_traders: 800,
            num_bollinger_traders: 800,
            num_vwap_executors: 200,
            num_pairs_traders: 400,
            num_sector_rotators: 500,
            ml_agent_counts: HashMap::from([
                ("decision_tree".into(), 300),
                ("random_forest".into(), 100),
                ("gradient_boosted".into(), 200),
                ("linear_model".into(), 0),
                ("svm_linear".into(), 0),
                ("gaussian_nb".into(), 0),
                ("ensemble".into(), 100),
            ]),
            min_tier1_agents: 7500,
            num_tier2_agents: 5000,
            t2_initial_cash: Cash::from_float(100_000.0),
            t2_max_position: 400,
            t2_buy_threshold_min: 20.0,
            t2_buy_threshold_max: 60.0,
            t2_stop_loss_min: 0.25,
            t2_stop_loss_max: 0.5,
            t2_take_profit_min: 0.25,
            t2_take_profit_max: 0.5,
            t2_sell_threshold_min: 75.0,
            t2_sell_threshold_max: 100.0,
            t2_take_profit_prob: 0.5,
            t2_news_reactor_prob: 0.3,
            t2_order_size_min: 0.1,
            t2_order_size_max: 0.2,
            enable_background_pool: true,
            background_pool_size: 87500,
            background_regime: MarketRegime::Normal,
            t3_mean_order_size: 40.0,
            t3_max_order_size: 120,
            t3_order_size_stddev: 10.0,
            t3_base_activity: Some(0.01),
            t3_price_spread_lambda: 10.0,
            t3_max_price_deviation: 0.05,
            mm_initial_cash: Cash::from_float(100_000.0),
            mm_initial_position: 800,
            mm_half_spread: 0.0005,
            mm_quote_size: 60,
            mm_refresh_interval: 1,
            mm_max_inventory: 1500,
            mm_inventory_skew: 0.0001,
            mm_max_long_position: 1500,
            mm_max_short_position: 0,
            nt_initial_cash: Cash::from_float(100_000.0),
            nt_initial_position: 0,
            nt_order_probability: 0.3,
            nt_price_deviation: 0.03,
            nt_min_quantity: 15,
            nt_max_quantity: 15,
            nt_max_long_position: 600,
            nt_max_short_position: 0,
            quant_initial_cash: Cash::from_float(100_000.0),
            quant_order_size: 15,
            quant_max_long_position: 1200,
            quant_max_short_position: 300,
            tree_agent_initial_cash: Cash::from_float(100_000.0),
            tree_agent_order_size: 30,
            tree_agent_max_long_position: 1200,
            tree_agent_max_short_position: 300,
            tree_agent_buy_threshold: 0.65,
            tree_agent_sell_threshold: 0.75,
            max_price_history: 500,
            tui_frame_rate: 30,
            data_update_rate: 30,
            events_enabled: true,
            event_earnings_prob: 0.006,
            event_earnings_interval: 25,
            event_guidance_prob: 0.002,
            event_guidance_interval: 50,
            event_rate_decision_prob: 0.001,
            event_rate_decision_interval: 125,
            event_sector_news_prob: 0.006,
            event_sector_news_interval: 12,
        }
    }
}

impl SimConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Builder-style setters
    // ─────────────────────────────────────────────────────────────────────────

    pub fn symbols(mut self, symbols: Vec<SymbolSpec>) -> Self {
        self.symbols = symbols;
        self
    }

    pub fn add_symbol(mut self, symbol: impl Into<String>, initial_price: f64) -> Self {
        self.symbols.push(SymbolSpec::new(symbol, initial_price));
        self
    }

    pub fn symbol(mut self, symbol: impl Into<String>) -> Self {
        if self.symbols.is_empty() {
            self.symbols.push(SymbolSpec::new(symbol, 100.0));
        } else {
            self.symbols[0].symbol = symbol.into();
        }
        self
    }

    pub fn initial_price(mut self, price: f64) -> Self {
        if self.symbols.is_empty() {
            self.symbols.push(SymbolSpec::new("ACME", price));
        } else {
            self.symbols[0].initial_price = Price::from_float(price);
        }
        self
    }

    pub fn total_ticks(mut self, ticks: u64) -> Self {
        self.total_ticks = ticks;
        self
    }

    pub fn tick_delay_ms(mut self, ms: u64) -> Self {
        self.tick_delay_ms = ms;
        self
    }

    pub fn market_makers(mut self, count: usize) -> Self {
        self.num_market_makers = count;
        self
    }

    pub fn noise_traders(mut self, count: usize) -> Self {
        self.num_noise_traders = count;
        self
    }

    pub fn momentum_traders(mut self, count: usize) -> Self {
        self.num_momentum_traders = count;
        self
    }

    pub fn trend_followers(mut self, count: usize) -> Self {
        self.num_trend_followers = count;
        self
    }

    pub fn macd_traders(mut self, count: usize) -> Self {
        self.num_macd_traders = count;
        self
    }

    pub fn bollinger_traders(mut self, count: usize) -> Self {
        self.num_bollinger_traders = count;
        self
    }

    pub fn vwap_executors(mut self, count: usize) -> Self {
        self.num_vwap_executors = count;
        self
    }

    pub fn pairs_traders(mut self, count: usize) -> Self {
        self.num_pairs_traders = count;
        self
    }

    pub fn sector_rotators(mut self, count: usize) -> Self {
        self.num_sector_rotators = count;
        self
    }

    /// Set the agent count for a specific ML model type.
    ///
    /// # Example
    /// ```ignore
    /// let config = SimConfig::new()
    ///     .ml_agents("decision_tree", 400)
    ///     .ml_agents("random_forest", 100);
    /// ```
    pub fn ml_agents(mut self, model_type: impl Into<String>, count: usize) -> Self {
        self.ml_agent_counts.insert(model_type.into(), count);
        self
    }

    pub fn min_tier1(mut self, count: usize) -> Self {
        self.min_tier1_agents = count;
        self
    }

    pub fn tier2_agents(mut self, count: usize) -> Self {
        self.num_tier2_agents = count;
        self
    }

    pub fn t2_cash(mut self, cash: f64) -> Self {
        self.t2_initial_cash = Cash::from_float(cash);
        self
    }

    pub fn t2_max_position(mut self, max_pos: u64) -> Self {
        self.t2_max_position = max_pos;
        self
    }

    pub fn mm_cash(mut self, cash: f64) -> Self {
        self.mm_initial_cash = Cash::from_float(cash);
        self
    }

    pub fn nt_cash(mut self, cash: f64) -> Self {
        self.nt_initial_cash = Cash::from_float(cash);
        self
    }

    pub fn quant_cash(mut self, cash: f64) -> Self {
        self.quant_initial_cash = Cash::from_float(cash);
        self
    }

    pub fn mm_spread(mut self, half_spread: f64) -> Self {
        self.mm_half_spread = half_spread;
        self
    }

    pub fn nt_probability(mut self, prob: f64) -> Self {
        self.nt_order_probability = prob;
        self
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    pub fn get_symbols(&self) -> &[SymbolSpec] {
        &self.symbols
    }

    pub fn primary_symbol(&self) -> &str {
        &self.symbols[0].symbol
    }

    pub fn primary_initial_price(&self) -> Price {
        self.symbols[0].initial_price
    }

    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Computed Properties
    // ─────────────────────────────────────────────────────────────────────────

    /// Total number of specified (non-random) Tier 1 agents.
    pub fn specified_tier1_agents(&self) -> usize {
        self.num_market_makers
            + self.num_noise_traders
            + self.num_momentum_traders
            + self.num_trend_followers
            + self.num_macd_traders
            + self.num_bollinger_traders
            + self.num_vwap_executors
            + self.num_pairs_traders
            + self.total_ml_agents()
    }

    /// Total number of ML agents across all model types.
    pub fn total_ml_agents(&self) -> usize {
        self.ml_agent_counts.values().sum()
    }

    /// Get the agent count for a specific ML model type.
    pub fn ml_agent_count(&self, model_type: &str) -> usize {
        self.ml_agent_counts.get(model_type).copied().unwrap_or(0)
    }

    /// Number of random Tier 1 agents to spawn to meet minimum.
    pub fn random_tier1_count(&self) -> usize {
        let specified = self.specified_tier1_agents();
        self.min_tier1_agents.saturating_sub(specified)
    }

    /// Total number of Tier 1 agents (specified + random fill).
    pub fn total_tier1_agents(&self) -> usize {
        self.specified_tier1_agents() + self.random_tier1_count()
    }

    /// Number of reactive Tier 2 agents (total T2 minus sector rotators).
    pub fn reactive_tier2_count(&self) -> usize {
        self.num_tier2_agents
            .saturating_sub(self.num_sector_rotators)
    }

    /// Effective sector rotators (clamped to tier2 budget).
    pub fn effective_sector_rotators(&self) -> usize {
        self.num_sector_rotators.min(self.num_tier2_agents)
    }

    /// Total number of agents (Tier 1 + Tier 2).
    /// Tier 2 already includes sector rotators.
    pub fn total_agents(&self) -> usize {
        self.total_tier1_agents() + self.num_tier2_agents
    }

    /// Total starting cash in the system (estimate).
    pub fn total_starting_cash(&self) -> Cash {
        let mm_total =
            Cash::from_float(self.mm_initial_cash.to_float() * self.num_market_makers as f64);
        let nt_total =
            Cash::from_float(self.nt_initial_cash.to_float() * self.num_noise_traders as f64);
        let quant_count = self.num_momentum_traders
            + self.num_trend_followers
            + self.num_macd_traders
            + self.num_bollinger_traders;
        let quant_total = Cash::from_float(self.quant_initial_cash.to_float() * quant_count as f64);
        let tree_total = Cash::from_float(
            self.tree_agent_initial_cash.to_float() * self.total_ml_agents() as f64,
        );
        mm_total + nt_total + quant_total + tree_total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset Configurations
// ─────────────────────────────────────────────────────────────────────────────

impl SimConfig {
    /// Quick demo: 10% of default agents, fewer ticks, faster visualization.
    pub fn demo() -> Self {
        Self::default()
            .total_ticks(1000)
            .tick_delay_ms(5)
            .market_makers(10)
            .noise_traders(40)
            .momentum_traders(5)
            .trend_followers(5)
            .macd_traders(5)
            .bollinger_traders(5)
            .vwap_executors(5)
            .min_tier1(100)
    }

    /// Stress test: 2x default agents, many ticks, no delay.
    pub fn stress_test() -> Self {
        Self::default()
            .total_ticks(100_000)
            .tick_delay_ms(0)
            .min_tier1(2000)
    }

    /// Low activity: 20% of default agents, conservative parameters.
    pub fn low_activity() -> Self {
        Self::default()
            .market_makers(20)
            .noise_traders(80)
            .momentum_traders(10)
            .trend_followers(10)
            .macd_traders(10)
            .bollinger_traders(10)
            .vwap_executors(10)
            .nt_probability(0.02)
            .min_tier1(200)
    }

    /// High volatility: aggressive noise traders, wider spreads.
    pub fn high_volatility() -> Self {
        Self::default()
            .noise_traders(600)
            .nt_probability(0.5)
            .nt_cash(50_000.0)
            .mm_spread(0.005)
    }

    /// Quant-heavy: More algorithmic traders, fewer noise traders.
    pub fn quant_heavy() -> Self {
        Self::default()
            .noise_traders(100)
            .momentum_traders(150)
            .trend_followers(150)
            .macd_traders(150)
            .bollinger_traders(150)
            .vwap_executors(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_consistency() {
        let config = SimConfig::default();
        let expected_specified = config.num_market_makers
            + config.num_noise_traders
            + config.num_momentum_traders
            + config.num_trend_followers
            + config.num_macd_traders
            + config.num_bollinger_traders
            + config.num_vwap_executors
            + config.num_pairs_traders
            + config.total_ml_agents();
        assert_eq!(config.specified_tier1_agents(), expected_specified);
        assert!(config.num_market_makers >= 1);
        assert!(config.total_ticks > 0);
        assert!(config.primary_initial_price() > Price::ZERO);
        // ML agent counts sum should match total
        let ml_sum: usize = config.ml_agent_counts.values().sum();
        assert_eq!(config.total_ml_agents(), ml_sum);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SimConfig::new()
            .market_makers(7)
            .noise_traders(10)
            .momentum_traders(3)
            .min_tier1(25);
        assert_eq!(config.num_market_makers, 7);
        assert_eq!(config.num_noise_traders, 10);
        assert_eq!(config.num_momentum_traders, 3);
        assert_eq!(config.min_tier1_agents, 25);
    }

    #[test]
    fn test_random_tier1_fill() {
        let mut config = SimConfig::new()
            .market_makers(2)
            .noise_traders(3)
            .momentum_traders(0)
            .trend_followers(0)
            .macd_traders(0)
            .bollinger_traders(0)
            .vwap_executors(0)
            .pairs_traders(0)
            .min_tier1(10)
            .tier2_agents(0);
        config.ml_agent_counts.clear(); // No ML agents
        assert_eq!(config.specified_tier1_agents(), 5);
        assert_eq!(config.random_tier1_count(), 5);
        assert_eq!(config.total_tier1_agents(), 10);
    }

    #[test]
    fn test_no_random_fill_when_specified_meets_minimum() {
        let config = SimConfig::new()
            .market_makers(5)
            .noise_traders(10)
            .min_tier1(10);
        assert!(config.specified_tier1_agents() >= config.min_tier1_agents);
        assert_eq!(config.random_tier1_count(), 0);
    }

    #[test]
    fn test_total_starting_cash() {
        let mut config = SimConfig::new()
            .market_makers(2)
            .noise_traders(10)
            .momentum_traders(1)
            .trend_followers(1)
            .macd_traders(0)
            .bollinger_traders(0)
            .mm_cash(1_000_000.0)
            .nt_cash(10_000.0)
            .quant_cash(100_000.0);
        config.ml_agent_counts.clear(); // No ML agents
        assert_eq!(config.total_starting_cash(), Cash::from_float(2_300_000.0));
    }

    #[test]
    fn test_preset_configs_differ_from_default() {
        let default = SimConfig::default();
        let demo = SimConfig::demo();
        let stress = SimConfig::stress_test();
        let low = SimConfig::low_activity();
        let high = SimConfig::high_volatility();
        let quant = SimConfig::quant_heavy();
        assert_ne!(demo.total_ticks, default.total_ticks);
        assert_ne!(stress.total_ticks, default.total_ticks);
        assert_ne!(low.nt_order_probability, default.nt_order_probability);
        assert_ne!(high.nt_order_probability, default.nt_order_probability);
        assert_ne!(quant.num_noise_traders, default.num_noise_traders);
    }

    #[test]
    fn test_tier1_agent_type_random() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let agent_type = Tier1AgentType::random(&mut rng);
            assert!(Tier1AgentType::SPAWNABLE.contains(&agent_type));
        }
    }
}
