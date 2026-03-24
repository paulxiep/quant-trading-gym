//! Background Agent Pool for Tier 3 statistical order generation (V3.4).
//!
//! The pool generates orders based on statistical distributions without
//! maintaining individual agent state. This enables 90k+ simulated agents
//! with ~10KB memory overhead.
//!
//! # Architecture
//!
//! One pool instance trades ALL symbols:
//! - Randomly selects which symbol to trade each order
//! - Tracks sentiment per-symbol (sector news affects right symbols)
//! - Uses single accounting ledger for aggregate P&L
//!
//! # Design Principles
//!
//! - **Declarative**: Behavior controlled by `BackgroundPoolConfig`
//! - **Modular**: Uses trait-based distributions (swappable)
//! - **SoC**: Pool generates orders; Simulation applies; Accounting tracks
//!
//! # Borrow-Checker Safety
//!
//! - `generate()` takes immutable `&PoolContext`, returns owned `Vec<Order>`
//! - No shared references to Market during generation
//! - Accounting mutations happen AFTER simulation processes orders

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use crate::MIN_ORDER_PRICE;
use news::NewsEvent;
use types::{AgentId, Order, OrderSide, Price, Sector, Symbol, Tick};

use super::accounting::BackgroundPoolAccounting;
use super::config::{BackgroundPoolConfig, MarketRegime, RegimePreset};
use super::distributions::{
    ExponentialPriceSpread, LogNormalSize, PriceDistribution, SizeDistribution,
};

// =============================================================================
// Constants
// =============================================================================

/// Sentinel AgentId for all background pool orders.
///
/// The simulation uses this to identify pool trades for accounting.
/// Individual agents are NOT tracked — this is the whole point of Tier 3.
pub const BACKGROUND_POOL_ID: AgentId = AgentId(0);

// =============================================================================
// PoolContext
// =============================================================================

/// Context provided to pool each tick.
///
/// Contains all information needed to generate orders without
/// borrowing from Simulation's mutable state.
pub struct PoolContext<'a> {
    /// Current tick number
    pub tick: Tick,

    /// Current mid prices per symbol (owned by caller)
    pub mid_prices: &'a HashMap<Symbol, Price>,

    /// Active news events (for sentiment updates)
    pub active_events: &'a [NewsEvent],

    /// Symbol-to-sector mapping (for sector news)
    pub symbol_sectors: &'a HashMap<Symbol, Sector>,
}

// =============================================================================
// SymbolSentiment
// =============================================================================

/// Per-symbol sentiment state.
#[derive(Debug, Clone)]
struct SymbolSentiment {
    /// Current sentiment (-1.0 = very bearish, +1.0 = very bullish)
    value: f64,

    /// Sector for this symbol (for sector-wide news)
    sector: Sector,
}

impl Default for SymbolSentiment {
    fn default() -> Self {
        Self {
            value: 0.0,
            sector: Sector::Tech, // Default sector
        }
    }
}

// =============================================================================
// BackgroundAgentPool
// =============================================================================

/// The Tier 3 Background Agent Pool.
///
/// Generates orders statistically based on:
/// - Configurable order size/price distributions
/// - Dynamic sentiment from news events
/// - Contrarian fraction for mean reversion
///
/// # Memory Budget
///
/// - Config: ~200 bytes
/// - Sentiments: ~100 bytes per symbol (typically 4-10 symbols = 400-1000 bytes)
/// - Distributions: ~50 bytes
/// - Accounting: ~500 bytes
/// - RNG: ~200 bytes
/// - **Total: ~1-2 KB** (not 10KB as originally estimated — even better!)
pub struct BackgroundAgentPool {
    /// Configuration
    config: BackgroundPoolConfig,

    /// Random number generator (seeded for reproducibility)
    rng: StdRng,

    /// Per-symbol sentiment tracking
    sentiments: HashMap<Symbol, SymbolSentiment>,

    /// Current regime preset values
    preset: RegimePreset,

    /// Price distribution (exponential decay from mid)
    price_dist: ExponentialPriceSpread,

    /// Size distribution (log-normal: many small, few large)
    size_dist: LogNormalSize,

    /// Accounting for P&L sanity checks
    accounting: BackgroundPoolAccounting,

    /// Orders generated this tick (for TUI display)
    orders_this_tick: usize,
}

impl BackgroundAgentPool {
    /// Create a new background pool from configuration.
    pub fn new(config: BackgroundPoolConfig, seed: u64) -> Self {
        let preset = config.regime.preset();

        let price_dist =
            ExponentialPriceSpread::new(config.price_spread_lambda, config.max_price_deviation);

        let size_dist = LogNormalSize::new(
            config.mean_order_size,
            config.order_size_stddev,
            config.min_order_size,
            config.max_order_size,
        );

        // Initialize sentiment for each configured symbol
        let sentiments = config
            .symbols
            .iter()
            .map(|s| (s.clone(), SymbolSentiment::default()))
            .collect();

        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
            sentiments,
            preset,
            price_dist,
            size_dist,
            accounting: BackgroundPoolAccounting::new(),
            orders_this_tick: 0,
        }
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set market regime (updates preset values dynamically).
    pub fn set_regime(&mut self, regime: MarketRegime) {
        self.preset = regime.preset();
    }

    /// Initialize symbol sectors from external mapping.
    ///
    /// Call this after construction if sectors aren't in config.
    pub fn init_sectors(&mut self, symbol_sectors: &HashMap<Symbol, Sector>) {
        symbol_sectors.iter().for_each(|(symbol, sector)| {
            self.sentiments
                .entry(symbol.clone())
                .and_modify(|sent| sent.sector = *sector)
                .or_insert_with(|| SymbolSentiment {
                    value: 0.0,
                    sector: *sector,
                });
        });
    }

    /// Update symbols the pool trades.
    pub fn set_symbols(&mut self, symbols: Vec<Symbol>) {
        // Preserve existing sentiments, add new ones
        symbols.iter().for_each(|symbol| {
            self.sentiments.entry(symbol.clone()).or_default();
        });
        self.config.symbols = symbols;
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the pool configuration.
    pub fn config(&self) -> &BackgroundPoolConfig {
        &self.config
    }

    /// Get mutable accounting reference (for simulation to record fills).
    pub fn accounting_mut(&mut self) -> &mut BackgroundPoolAccounting {
        &mut self.accounting
    }

    /// Get accounting reference (read-only).
    pub fn accounting(&self) -> &BackgroundPoolAccounting {
        &self.accounting
    }

    /// Get orders generated this tick.
    pub fn orders_this_tick(&self) -> usize {
        self.orders_this_tick
    }

    /// Get current sentiment for a symbol.
    pub fn sentiment(&self, symbol: &Symbol) -> f64 {
        self.sentiments.get(symbol).map(|s| s.value).unwrap_or(0.0)
    }

    /// Run sanity check and return result.
    pub fn sanity_check(&self) -> super::accounting::SanityCheckResult {
        self.accounting
            .sanity_check(self.config.max_pnl_loss_fraction)
    }

    // =========================================================================
    // Order Generation (Main Entry Point)
    // =========================================================================

    /// Generate orders for this tick.
    ///
    /// This is the main entry point called by the simulation each tick.
    /// Returns owned `Vec<Order>` — no borrows retained.
    pub fn generate(&mut self, ctx: &PoolContext<'_>) -> Vec<Order> {
        // Step 1: Update sentiment from active news events
        self.update_sentiment(ctx);

        // Step 2: Decay all sentiments toward neutral
        self.decay_sentiment();

        // Step 3: Calculate how many orders to generate this tick
        let num_orders = self.calculate_order_count();

        // Step 4: Generate orders (filter out None for missing mid prices)
        let orders: Vec<Order> = (0..num_orders)
            .filter_map(|_| self.generate_single_order(ctx))
            .collect();

        // Step 5: Track for TUI display
        self.orders_this_tick = orders.len();
        self.accounting.record_orders_generated(orders.len());

        orders
    }

    // =========================================================================
    // Sentiment Management
    // =========================================================================

    /// Update sentiment based on active news events.
    fn update_sentiment(&mut self, ctx: &PoolContext<'_>) {
        let max_sentiment = self.config.max_sentiment;
        let news_scale = self.config.news_sentiment_scale;

        ctx.active_events.iter().for_each(|event| {
            let effective = event.effective_sentiment(ctx.tick) * news_scale;

            // Symbol-specific events affect that symbol directly
            if let Some(symbol) = event.symbol()
                && let Some(sent) = self.sentiments.get_mut(symbol)
            {
                sent.value = (sent.value + effective).clamp(-max_sentiment, max_sentiment);
            }

            // Sector-wide events affect all symbols in that sector (at 50% strength)
            if let Some(sector) = event.sector() {
                self.sentiments
                    .values_mut()
                    .filter(|sent| sent.sector == sector)
                    .for_each(|sent| {
                        sent.value =
                            (sent.value + effective * 0.5).clamp(-max_sentiment, max_sentiment);
                    });
            }
        });
    }

    /// Decay all sentiments toward neutral (0.0).
    fn decay_sentiment(&mut self) {
        let decay = self.config.sentiment_decay;
        self.sentiments
            .values_mut()
            .for_each(|sent| sent.value *= decay);
    }

    // =========================================================================
    // Order Count Calculation
    // =========================================================================

    /// Calculate how many orders to generate this tick.
    fn calculate_order_count(&mut self) -> usize {
        // Use override if set, otherwise regime preset
        let base_rate = self
            .config
            .base_activity_override
            .unwrap_or(self.preset.base_activity);

        // Scale by pool size
        let expected = (self.config.pool_size as f64 * base_rate).round() as usize;

        // Add ±20% randomness for realistic variance
        let variance = (expected as f64 * 0.2).max(1.0) as i64;
        let adjustment = self.rng.r#gen_range(-variance..=variance);

        (expected as i64 + adjustment).max(0) as usize
    }

    // =========================================================================
    // Single Order Generation
    // =========================================================================

    /// Generate a single order.
    ///
    /// Returns `None` if mid price not available for selected symbol.
    fn generate_single_order(&mut self, ctx: &PoolContext<'_>) -> Option<Order> {
        // Pick random symbol from configured list
        if self.config.symbols.is_empty() {
            return None;
        }

        let idx = self.rng.r#gen_range(0..self.config.symbols.len());
        let symbol = self.config.symbols[idx].clone();

        // Get mid price for this symbol
        let mid_price = ctx.mid_prices.get(&symbol)?;

        // Determine order side based on sentiment + contrarian fraction
        let sentiment = self.sentiments.get(&symbol).map(|s| s.value).unwrap_or(0.0);
        let side = self.determine_side(sentiment);

        // Generate price offset from mid
        let price_offset = self.price_dist.sample_offset(&mut self.rng, *mid_price);

        // Apply offset in direction that makes sense for the side:
        // - BUY orders should be at or below mid (negative or zero offset)
        // - SELL orders should be at or above mid (positive or zero offset)
        let adjusted_offset = match side {
            OrderSide::Buy => -price_offset.abs(), // Always below or at mid
            OrderSide::Sell => price_offset.abs(), // Always above or at mid
        };

        // Apply minimum price floor to prevent negative price spirals
        // MIN_ORDER_PRICE is $0.01, represented as 100 (i64 in cents * 100)
        let min_price_i64 = (MIN_ORDER_PRICE * 10_000.0) as i64;
        let price = Price((mid_price.0 + adjusted_offset).max(min_price_i64));

        // Generate order size
        let quantity = self.size_dist.sample(&mut self.rng);

        Some(Order::limit(
            BACKGROUND_POOL_ID,
            &symbol,
            side,
            price,
            quantity,
        ))
    }

    /// Determine order side based on sentiment and contrarian fraction.
    fn determine_side(&mut self, sentiment: f64) -> OrderSide {
        // Map sentiment to base probability of buy:
        // sentiment -1.0 → buy_prob 0.0 (all sell)
        // sentiment  0.0 → buy_prob 0.5 (balanced)
        // sentiment +1.0 → buy_prob 1.0 (all buy)
        let base_buy_prob = 0.5 + sentiment * 0.5;

        // Contrarian traders go against sentiment (provides mean reversion)
        let is_contrarian = self.rng.r#gen_bool(self.preset.contrarian_fraction);
        let buy_prob = if is_contrarian {
            1.0 - base_buy_prob
        } else {
            base_buy_prob
        };

        if self.rng.r#gen_bool(buy_prob.clamp(0.0, 1.0)) {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BackgroundPoolConfig {
        BackgroundPoolConfig {
            pool_size: 1000,
            symbols: vec!["TEST".to_string()],
            ..Default::default()
        }
    }

    fn test_context<'a>(
        tick: Tick,
        mid_prices: &'a HashMap<Symbol, Price>,
        active_events: &'a [NewsEvent],
        symbol_sectors: &'a HashMap<Symbol, Sector>,
    ) -> PoolContext<'a> {
        PoolContext {
            tick,
            mid_prices,
            active_events,
            symbol_sectors,
        }
    }

    #[test]
    fn test_pool_generates_orders() {
        let mut pool = BackgroundAgentPool::new(test_config(), 42);

        let mid_prices: HashMap<_, _> = vec![("TEST".to_string(), Price::from_float(100.0))]
            .into_iter()
            .collect();
        let sectors = HashMap::new();

        let ctx = test_context(1, &mid_prices, &[], &sectors);
        let orders = pool.generate(&ctx);

        assert!(!orders.is_empty(), "Pool should generate orders");

        for order in &orders {
            assert_eq!(order.agent_id, BACKGROUND_POOL_ID);
            assert_eq!(order.symbol, "TEST");
        }
    }

    #[test]
    fn test_all_orders_use_pool_id() {
        let mut pool = BackgroundAgentPool::new(test_config(), 42);

        let mid_prices: HashMap<_, _> = vec![("TEST".to_string(), Price::from_float(100.0))]
            .into_iter()
            .collect();
        let sectors = HashMap::new();

        for tick in 0..100 {
            let ctx = test_context(tick, &mid_prices, &[], &sectors);
            let orders = pool.generate(&ctx);

            for order in orders {
                assert_eq!(
                    order.agent_id, BACKGROUND_POOL_ID,
                    "All pool orders must use BACKGROUND_POOL_ID"
                );
            }
        }
    }

    #[test]
    fn test_multi_symbol_trading() {
        let config = BackgroundPoolConfig {
            pool_size: 5000,
            symbols: vec!["AAPL".to_string(), "GOOG".to_string(), "MSFT".to_string()],
            ..Default::default()
        };
        let mut pool = BackgroundAgentPool::new(config, 42);

        let mid_prices: HashMap<_, _> = vec![
            ("AAPL".to_string(), Price::from_float(150.0)),
            ("GOOG".to_string(), Price::from_float(100.0)),
            ("MSFT".to_string(), Price::from_float(300.0)),
        ]
        .into_iter()
        .collect();
        let sectors = HashMap::new();

        let ctx = test_context(1, &mid_prices, &[], &sectors);
        let orders = pool.generate(&ctx);

        // Should have orders for multiple symbols
        let symbols: std::collections::HashSet<_> =
            orders.iter().map(|o| o.symbol.clone()).collect();

        assert!(
            symbols.len() > 1,
            "Should trade multiple symbols, got: {:?}",
            symbols
        );
    }

    #[test]
    fn test_sentiment_affects_bias() {
        let mut pool = BackgroundAgentPool::new(test_config(), 42);

        // Manually set strong positive sentiment
        pool.sentiments.get_mut("TEST").unwrap().value = 0.8;

        let mid_prices: HashMap<_, _> = vec![("TEST".to_string(), Price::from_float(100.0))]
            .into_iter()
            .collect();
        let sectors = HashMap::new();

        // Generate many orders and count buy/sell ratio
        let mut total_buys = 0;
        let mut total_sells = 0;

        for tick in 0..100 {
            let ctx = test_context(tick, &mid_prices, &[], &sectors);
            let orders = pool.generate(&ctx);

            total_buys += orders.iter().filter(|o| o.is_buy()).count();
            total_sells += orders.iter().filter(|o| o.is_sell()).count();

            // Prevent sentiment from decaying completely
            pool.sentiments.get_mut("TEST").unwrap().value = 0.8;
        }

        // With positive sentiment, should have more buys (accounting for contrarians)
        assert!(
            total_buys > total_sells,
            "Expected buy bias with positive sentiment: {} buys, {} sells",
            total_buys,
            total_sells
        );
    }

    #[test]
    fn test_sentiment_decay() {
        let mut pool = BackgroundAgentPool::new(test_config(), 42);
        pool.sentiments.get_mut("TEST").unwrap().value = 0.5;

        pool.decay_sentiment();

        let after = pool.sentiments.get("TEST").unwrap().value;
        assert!(after < 0.5, "Sentiment should decay toward zero");
        assert!(after > 0.49, "Decay should be gradual"); // 0.5 * 0.995 = 0.4975
    }

    #[test]
    fn test_order_prices_sensible() {
        let mut pool = BackgroundAgentPool::new(test_config(), 42);

        let mid_prices: HashMap<_, _> = vec![("TEST".to_string(), Price::from_float(100.0))]
            .into_iter()
            .collect();
        let sectors = HashMap::new();

        let ctx = test_context(1, &mid_prices, &[], &sectors);
        let orders = pool.generate(&ctx);

        let mid_raw = Price::from_float(100.0).0;

        for order in &orders {
            let price = order.limit_price().unwrap();

            // Buy orders should be at or below mid
            if order.is_buy() {
                assert!(
                    price.0 <= mid_raw,
                    "Buy order at {} should be <= mid {}",
                    price.to_float(),
                    mid_raw as f64 / 10_000.0
                );
            }

            // Sell orders should be at or above mid
            if order.is_sell() {
                assert!(
                    price.0 >= mid_raw,
                    "Sell order at {} should be >= mid {}",
                    price.to_float(),
                    mid_raw as f64 / 10_000.0
                );
            }

            // All prices should be within max deviation
            let deviation = (price.0 - mid_raw).abs() as f64 / mid_raw as f64;
            assert!(
                deviation <= pool.config.max_price_deviation + 0.001,
                "Price {} deviates {}% from mid, max is {}%",
                price.to_float(),
                deviation * 100.0,
                pool.config.max_price_deviation * 100.0
            );
        }
    }

    #[test]
    fn test_regime_changes_activity() {
        let calm_config = BackgroundPoolConfig {
            pool_size: 10_000,
            regime: super::super::config::MarketRegime::Calm,
            symbols: vec!["TEST".to_string()],
            ..Default::default()
        };

        let crisis_config = BackgroundPoolConfig {
            pool_size: 10_000,
            regime: super::super::config::MarketRegime::Crisis,
            symbols: vec!["TEST".to_string()],
            ..Default::default()
        };

        let mut calm_pool = BackgroundAgentPool::new(calm_config, 42);
        let mut crisis_pool = BackgroundAgentPool::new(crisis_config, 42);

        let mid_prices: HashMap<_, _> = vec![("TEST".to_string(), Price::from_float(100.0))]
            .into_iter()
            .collect();
        let sectors = HashMap::new();

        let ctx = test_context(1, &mid_prices, &[], &sectors);

        let calm_orders = calm_pool.generate(&ctx);
        let crisis_orders = crisis_pool.generate(&ctx);

        assert!(
            crisis_orders.len() > calm_orders.len() * 2,
            "Crisis should generate more orders: {} vs {}",
            crisis_orders.len(),
            calm_orders.len()
        );
    }
}
