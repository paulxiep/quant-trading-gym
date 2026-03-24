//! Simulation runner implementing the tick-based event loop.
//!
//! The simulation holds the order book, agents, and coordinates the tick loop.
//!
//! # Position Limits (V2.1)
//!
//! When `enforce_position_limits` is enabled in config, orders are validated
//! against:
//! - Cash sufficiency for buys
//! - Shares outstanding limits for long positions
//! - Short-selling constraints and borrow availability
//!
//! Rejected orders are logged but do not cause errors.
//!
//! # Multi-Symbol Support (V2.3)
//!
//! The simulation supports multiple symbols via `Market` (HashMap<Symbol, OrderBook>).
//! Each symbol has independent candles, trades, and indicators.
//!
//! # Tiered Agent Architecture (V3.2)
//!
//! Agents are split into two tiers:
//! - **Tier 1**: Called every tick via `on_tick()` (market makers, technical traders)
//! - **Tier 2**: Reactive agents woken only when conditions trigger (via WakeConditionIndex)
//!
//! This reduces per-tick overhead from O(n) to O(k) where k << n triggered agents.
//!
//! # Parallel Execution & Batch Auction (V3.5)
//!
//! With the `parallel` feature enabled:
//! - T1 and triggered T2 agents execute `on_tick()` in parallel via rayon
//! - Orders are grouped by symbol and processed via **batch auction**
//! - Each symbol's auction runs independently (fully parallel across symbols)
//!
//! Batch auction semantics:
//! 1. **Collection phase**: All agents run `on_tick()` in parallel, collecting orders
//! 2. **Auction phase**: Per-symbol clearing price computed, all crossing orders matched
//!
//! This differs from continuous matching: all agents see the same market state and
//! compete in a single auction per tick, rather than sequential price-time priority.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agents::{
    Agent, AgentAction, BackgroundAgentPool, MlPredictionCache, ModelRegistry, PoolContext,
    StrategyContext,
};
use quant::{AgentRiskSnapshot, IndicatorCache, IndicatorEngine, IndicatorSnapshot};
use sim_core::{Market, MarketView, OrderBook, run_parallel_auctions};
use types::{AgentId, Candle, Order, Price, Quantity, Symbol, Tick, Timestamp, Trade};

use crate::config::SimulationConfig;
use crate::hooks::{BookSnapshot, HookContext, HookRunner, MarketSnapshot, SimulationHook};
use crate::subsystems::{
    AgentOrchestrator, AuctionEngine, MarketDataManager, NewsEngine, OrderValidationCtx,
    RiskManager,
};
use crate::traits::{
    AgentExecutionCoordinator, FundamentalsProvider, MarketDataProvider, PositionTracker,
    RiskTracker,
};

/// Agent action with captured state for order validation.
/// (agent_id, action, per-symbol positions, cash, is_market_maker)
type AgentActionWithState = (
    AgentId,
    AgentAction,
    HashMap<Symbol, i64>,
    types::Cash,
    bool,
);

/// Statistics about the simulation state.
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    /// Current tick number.
    pub tick: Tick,

    /// Total trades executed.
    pub total_trades: u64,

    /// Total orders submitted.
    pub total_orders: u64,

    /// Total orders that resulted in fills.
    pub filled_orders: u64,

    /// Total orders that were added to book (resting).
    pub resting_orders: u64,

    /// Total orders rejected due to position limit violations (V2.1).
    pub rejected_orders: u64,

    /// Agents called this tick (V3.2 debug).
    pub agents_called_this_tick: usize,

    /// T2 agents triggered this tick (V3.2 debug).
    pub t2_triggered_this_tick: usize,

    /// T3 background pool orders generated this tick (V3.4).
    pub t3_orders_this_tick: usize,
}

/// The main simulation runner.
///
/// Coordinates the tick-based event loop:
/// 1. Build market data snapshot
/// 2. Call each agent's `on_tick` to get their orders
/// 3. Validate orders against position limits (V2.1)
/// 4. Run batch auction per symbol (V3.5 - parallel across symbols)
/// 5. Update borrow ledger on short trades (V2.1)
/// 6. Notify agents of fills
/// 7. Advance tick counter
///
/// # V3.5 Parallel Execution & Batch Auction
///
/// Agents are wrapped in `Mutex` to enable parallel `on_tick()` execution.
/// Orders are processed via batch auction (single clearing price per symbol),
/// enabling full parallelism across symbols.
pub struct Simulation {
    /// Configuration for this simulation.
    config: SimulationConfig,

    /// Multi-symbol market container (V2.3).
    market: Market,

    /// Current tick.
    tick: Tick,

    /// Current timestamp.
    timestamp: Timestamp,

    /// Simulation statistics.
    stats: SimulationStats,

    /// Indicator cache for current tick (reserved for future per-tick caching).
    #[allow(dead_code)]
    indicator_cache: IndicatorCache,

    /// Hook runner for simulation observers (V3.6).
    hooks: HookRunner,

    // =========================================================================
    // V5.2: Subsystems
    // =========================================================================
    /// Market data subsystem (candles, trades, indicators).
    market_data: MarketDataManager,

    /// Agent orchestration subsystem (T1/T2/T3 management).
    agent_orchestrator: AgentOrchestrator,

    /// Auction engine subsystem (order collection, batch auctions).
    auction_engine: AuctionEngine,

    /// Risk manager subsystem (position tracking, borrow ledger).
    risk_manager: RiskManager,

    /// News engine subsystem (fundamentals, events, sectors).
    news_engine: NewsEngine,

    /// ML model registry for centralized prediction caching (V5.6).
    /// When populated, enables O(M × S) predictions instead of O(N).
    model_registry: Option<ModelRegistry>,

    /// Feature extractor for ML features and recording (pre-V6 refactor section F).
    ///
    /// Set via `set_feature_extractor()`. When present, the runner extracts
    /// features in Phase 3, imputes NaN using `neutral_values()`, and passes
    /// them to both the ML cache (for prediction) and hooks (for recording).
    feature_extractor: Option<Box<dyn agents::FeatureExtractor>>,
}

impl Simulation {
    /// Create a new simulation with the given configuration.
    pub fn new(config: SimulationConfig) -> Self {
        // Initialize multi-symbol market
        let mut market = Market::new();

        // Initialize fundamentals for each symbol (V2.4)
        let mut fundamentals = news::SymbolFundamentals::new(news::MacroEnvironment::default());
        let mut sector_model = news::SectorModel::new();
        let symbols: Vec<_> = config
            .get_symbol_configs()
            .iter()
            .map(|c| c.symbol.clone())
            .collect();

        // Initialize each symbol with initial price for chart display
        for symbol_config in config.get_symbol_configs() {
            market.add_symbol_with_price(&symbol_config.symbol, symbol_config.initial_price);

            // Initialize default fundamentals based on initial price (V2.4)
            // EPS derived as price / 20 (P/E of 20), 5% growth, 40% payout
            let eps = Price::from_float(symbol_config.initial_price.to_float() / 20.0);
            fundamentals.insert(
                &symbol_config.symbol,
                news::Fundamentals::new(eps, 0.05, 0.40),
            );

            // Add symbol to sector model (V2.4)
            sector_model.add(&symbol_config.symbol, symbol_config.sector);
        }

        // Initialize news generator (V2.4) - clone for legacy field (will be removed in V5.2)
        // Get primary symbol config for RiskManager
        let primary_config = config
            .get_symbol_configs()
            .first()
            .cloned()
            .unwrap_or_default();

        // V3.8: Cache symbol->sector mapping (never changes, compute once)
        let symbol_sectors: HashMap<Symbol, types::Sector> = config
            .get_symbol_configs()
            .iter()
            .map(|sc| (sc.symbol.clone(), sc.sector))
            .collect();

        // =====================================================================
        // V5.2: Initialize subsystems
        // =====================================================================

        // Market data subsystem
        let market_data = MarketDataManager::new(
            &symbols,
            config.candle_interval,
            config.max_candles,
            config.max_recent_trades,
        );

        // Agent orchestrator subsystem
        let agent_orchestrator = AgentOrchestrator::new();

        // Auction engine subsystem
        let auction_engine = AuctionEngine::new();

        // Risk manager subsystem
        let borrow_pool_sizes: HashMap<Symbol, Quantity> = config
            .get_symbol_configs()
            .iter()
            .map(|sc| (sc.symbol.clone(), sc.borrow_pool_size()))
            .collect();
        let risk_manager = RiskManager::new(
            &symbols,
            &borrow_pool_sizes,
            primary_config,
            config.short_selling.clone(),
        );

        // News engine subsystem
        let news_engine = NewsEngine::new(
            config.news.clone(),
            symbols.clone(),
            sector_model.clone(),
            fundamentals.clone(),
            config.fair_value_drift.clone(),
            symbol_sectors,
            config.seed,
            config.verbose,
        );

        Self {
            market,
            tick: 0,
            timestamp: 0,
            stats: SimulationStats::default(),
            indicator_cache: IndicatorCache::new(),
            hooks: HookRunner::new(),
            config,
            // V5.2: Subsystems
            market_data,
            agent_orchestrator,
            auction_engine,
            risk_manager,
            news_engine,
            // V5.6: ML prediction caching
            model_registry: None,
            // Pre-V6 refactor: Feature extraction
            feature_extractor: None,
        }
    }

    /// Create a simulation with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SimulationConfig::default())
    }

    // =========================================================================
    // Hooks (V3.6)
    // =========================================================================

    /// Register a simulation hook.
    ///
    /// Hooks are called in registration order at each lifecycle point.
    /// Use `Arc` to share hooks between multiple simulations or retain access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use simulation::hooks::MetricsHook;
    /// use std::sync::Arc;
    ///
    /// let metrics = Arc::new(MetricsHook::new());
    /// sim.add_hook(metrics.clone());
    /// sim.run(1000);
    /// println!("Avg trades/tick: {:.2}", metrics.snapshot().avg_trades_per_tick);
    /// ```
    pub fn add_hook(&mut self, hook: Arc<dyn SimulationHook>) {
        self.hooks.add(hook);
    }

    /// Get the number of registered hooks.
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    // =========================================================================
    // ML Model Registry (V5.6)
    // =========================================================================

    /// Register an ML model for centralized prediction caching.
    ///
    /// When models are registered, the simulation will compute predictions
    /// once per (model, symbol) pair in Phase 3, enabling O(M × S) predictions
    /// instead of O(N) per-agent computations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = DecisionTree::from_json("model.json")?;
    /// sim.register_ml_model(model);
    /// ```
    pub fn register_ml_model<M: agents::MlModel + 'static>(&mut self, model: M) {
        // Auto-set CanonicalFeatures if no extractor configured (V6.3 default)
        if self.feature_extractor.is_none() {
            self.feature_extractor = Some(Box::new(agents::CanonicalFeatures));
        }
        self.model_registry
            .get_or_insert_with(ModelRegistry::new)
            .register(model);
    }

    /// Register an Arc-wrapped ML model for centralized prediction caching.
    ///
    /// Same as [`register_ml_model`] but accepts a pre-wrapped `Arc<dyn MlModel>`,
    /// enabling shared ownership (e.g., ensemble sub-models).
    pub fn register_ml_model_arc(&mut self, model: std::sync::Arc<dyn agents::MlModel>) {
        if self.feature_extractor.is_none() {
            self.feature_extractor = Some(Box::new(agents::CanonicalFeatures));
        }
        self.model_registry
            .get_or_insert_with(ModelRegistry::new)
            .register_arc(model);
    }

    /// Set the feature extractor for ML features and recording.
    ///
    /// When set, the runner extracts features in Phase 3 and passes them to
    /// both the ML cache (for prediction) and recording hooks (for Parquet).
    /// Use `CanonicalFeatures` (V6.3, 28 features), `FullFeatures` (V6.1, 55),
    /// or `MinimalFeatures` (V5, 42).
    pub fn set_feature_extractor(&mut self, extractor: Box<dyn agents::FeatureExtractor>) {
        self.feature_extractor = Some(extractor);
    }

    /// Check if the model registry has any registered models.
    pub fn has_ml_models(&self) -> bool {
        self.model_registry.as_ref().is_some_and(|r| !r.is_empty())
    }

    /// Get the number of registered ML models.
    pub fn ml_model_count(&self) -> usize {
        self.model_registry.as_ref().map(|r| r.len()).unwrap_or(0)
    }

    /// Build hook context with current market state.
    ///
    /// Creates owned snapshots to avoid borrow conflicts.
    fn build_hook_context(&self) -> HookContext {
        let symbols: Vec<_> = self.market.symbols().cloned().collect();
        let snapshots = parallel::filter_map_slice(
            &symbols,
            |symbol| {
                self.market.get_book(symbol).map(|book| {
                    (
                        symbol.clone(),
                        BookSnapshot {
                            best_bid: book.best_bid().map(|(price, _)| price),
                            best_ask: book.best_ask().map(|(price, _)| price),
                            bid_depth: book.bid_depth(5),
                            ask_depth: book.ask_depth(5),
                            last_price: book.last_price(),
                        },
                    )
                })
            },
            true, // Sequential for small collections
        );

        let mut market_snapshot = MarketSnapshot::new();
        for (symbol, snapshot) in snapshots {
            market_snapshot.add_book(symbol, snapshot);
        }

        let (t1_count, t2_count, t3_count) = self.agent_orchestrator.tier_counts();

        HookContext::new(self.tick, self.timestamp)
            .with_market(market_snapshot)
            .with_tiers(t1_count, t2_count, t3_count)
    }

    /// Build hook context with enriched data for `on_tick_end` (V4.4).
    ///
    /// Includes candles, indicators, agent summaries, risk metrics, recent trades, etc.
    /// Pre-extracted features are passed separately on HookContext (SoC — see hooks.rs).
    fn build_enriched_hook_context(
        &self,
        indicators: &quant::IndicatorSnapshot,
        features: Option<HashMap<Symbol, agents::FeatureVec>>,
    ) -> HookContext {
        use crate::hooks::EnrichedData;

        // Start with base context
        let base = self.build_hook_context();

        // V5.5: Clone indicator values from IndicatorSnapshot (single source of truth)
        let indicator_values: HashMap<Symbol, HashMap<types::IndicatorType, f64>> = indicators
            .symbols()
            .filter_map(|symbol| {
                indicators
                    .get_symbol(symbol)
                    .map(|values| (symbol.clone(), values.clone()))
            })
            .collect();

        let enriched = EnrichedData {
            candles: self.build_candles_map(),
            indicators: indicator_values,
            agent_summaries: self.agent_summaries(),
            risk_metrics: self.risk_manager.compute_all_metrics(),
            fair_values: HashMap::new(),
            news_events: self.get_active_news_snapshots(),
            recent_trades: self.market_data.all_recent_trades().clone(),
        };

        let ctx = base.with_enriched(enriched);
        match features {
            Some(f) => ctx.with_features(f),
            None => ctx,
        }
    }

    /// Get active news events as snapshots for hooks.
    fn get_active_news_snapshots(&self) -> Vec<crate::hooks::NewsEventSnapshot> {
        self.news_engine.get_news_snapshots(self.tick)
    }

    /// Set the Tier 3 background pool for statistical order generation (V3.4).
    ///
    /// The pool generates orders each tick based on statistical distributions,
    /// simulating 90k+ background agents without individual instances.
    pub fn set_background_pool(&mut self, pool: BackgroundAgentPool) {
        self.agent_orchestrator.set_background_pool(pool);
    }

    /// Get a reference to the background pool (if configured).
    ///
    /// For mutation, use `set_background_pool()` to replace the pool entirely.
    pub fn background_pool(&self) -> Option<&BackgroundAgentPool> {
        self.agent_orchestrator.background_pool()
    }

    /// Add an agent to the simulation.
    ///
    /// For reactive (T2) agents, registers their initial wake conditions.
    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        self.agent_orchestrator.add_agent(agent, self.tick);
    }

    /// Get the current tick.
    pub fn tick(&self) -> Tick {
        self.tick
    }

    /// Get the current timestamp.
    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    /// Get simulation statistics.
    pub fn stats(&self) -> &SimulationStats {
        &self.stats
    }

    /// Get a reference to a specific symbol's order book.
    pub fn get_book(&self, symbol: &Symbol) -> Option<&OrderBook> {
        self.market.get_book(symbol)
    }

    /// Get a reference to the multi-symbol market (V2.3).
    pub fn market(&self) -> &Market {
        &self.market
    }

    /// Get the number of agents.
    pub fn agent_count(&self) -> usize {
        self.agent_orchestrator.agent_count()
    }

    /// Get a clone of an agent's state by ID (for gym observation extraction).
    pub fn agent_state(&self, id: types::AgentId) -> Option<agents::AgentState> {
        self.agent_orchestrator.agent_state(id)
    }

    /// Get a reference to the simulation configuration.
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    /// Get total shares held for a specific symbol.
    pub fn total_shares_held_for(&self, symbol: &Symbol) -> Quantity {
        self.risk_manager.total_shares_held_for(symbol)
    }

    /// Get the number of shares an agent has borrowed for a symbol.
    pub fn borrowed_shares(&self, agent_id: AgentId, symbol: &Symbol) -> Quantity {
        self.risk_manager.borrowed_shares(agent_id, symbol)
    }

    /// Get agent summaries for display (V3.1: per-symbol positions).
    pub fn agent_summaries(&self) -> Vec<crate::traits::AgentSummary> {
        // Get current prices for all symbols
        let prices: HashMap<Symbol, types::Price> = self
            .market
            .symbols()
            .filter_map(|sym| {
                self.market
                    .get_book(sym)
                    .and_then(|b| b.last_price())
                    .map(|p| (sym.clone(), p))
            })
            .collect();

        self.agent_orchestrator.agent_summaries(
            &prices,
            !self.config.parallelization.parallel_agent_collection,
        )
    }

    /// Get risk metrics for all agents.
    pub fn agent_risk_metrics(&self) -> HashMap<AgentId, AgentRiskSnapshot> {
        self.risk_manager.compute_all_metrics()
    }

    /// Get risk metrics for a specific agent.
    pub fn agent_risk(&self, agent_id: AgentId) -> AgentRiskSnapshot {
        self.risk_manager.compute_metrics(agent_id)
    }

    /// Get recent trades for a specific symbol.
    pub fn recent_trades_for(&self, symbol: &Symbol) -> &[Trade] {
        self.market_data.recent_trades_for(symbol)
    }

    /// Get all recent trades across all symbols.
    pub fn all_recent_trades(&self) -> &HashMap<Symbol, Vec<Trade>> {
        self.market_data.all_recent_trades()
    }

    /// Get historical candles for a specific symbol.
    pub fn candles_for(&mut self, symbol: &Symbol) -> &[Candle] {
        self.market_data.candles_for_mut(symbol)
    }

    /// Get all candles across all symbols.
    pub fn all_candles(&self) -> &HashMap<Symbol, VecDeque<Candle>> {
        self.market_data.all_candles()
    }

    /// Get a reference to the indicator engine.
    pub fn indicator_engine(&self) -> &IndicatorEngine {
        self.market_data.indicator_engine()
    }

    /// Get a mutable reference to the indicator engine for registration.
    pub fn indicator_engine_mut(&mut self) -> &mut IndicatorEngine {
        self.market_data.indicator_engine_mut()
    }

    /// Build indicator snapshot for current tick (all symbols).
    fn build_indicator_snapshot(&mut self) -> IndicatorSnapshot {
        self.market_data.build_indicator_snapshot()
    }

    /// Build candles map for StrategyContext (converts VecDeque to Vec).
    fn build_candles_map(&self) -> HashMap<Symbol, Vec<Candle>> {
        self.market_data.build_candles_map()
    }

    /// Build trades map for StrategyContext (just return reference).
    fn build_trades_map(&self) -> HashMap<Symbol, Vec<Trade>> {
        self.market_data.build_trades_map()
    }

    /// Update candle with trade data for the trade's symbol.
    fn update_candles(&mut self, trades: &[Trade]) {
        self.market_data
            .update_candles(trades, self.tick, self.timestamp, &self.market);
    }

    // =========================================================================
    // V3.5: Parallel Agent Collection (delegated to AgentOrchestrator)
    // =========================================================================

    /// Collect agent actions from the given indices.
    fn collect_agent_actions(
        &self,
        indices: &[usize],
        ctx: &StrategyContext<'_>,
    ) -> Vec<AgentActionWithState> {
        self.agent_orchestrator.collect_actions(
            indices,
            ctx,
            !self.config.parallelization.parallel_agent_collection,
        )
    }

    /// Build position cache for all agents (for counterparty lookup during trade processing).
    fn build_position_cache(&self) -> HashMap<AgentId, HashMap<Symbol, i64>> {
        self.agent_orchestrator
            .build_position_cache(!self.config.parallelization.parallel_agent_collection)
    }

    /// Collect current prices for all symbols (for wake condition checking).
    fn collect_current_prices(&self) -> Vec<(Symbol, Price)> {
        parallel::map_slice(
            self.config.get_symbol_configs(),
            |sc| {
                let price = self
                    .market
                    .last_price(&sc.symbol)
                    .unwrap_or(sc.initial_price);
                (sc.symbol.clone(), price)
            },
            true, // Sequential for small collections
        )
    }

    /// Determine which agent indices should be called this tick.
    ///
    /// T1 agents are always called; T2 agents only when their conditions trigger.
    /// Returns (indices_to_call, triggered_t2_map).
    fn compute_agents_to_call(
        &mut self,
        current_prices: &[(Symbol, Price)],
    ) -> (
        Vec<usize>,
        HashMap<AgentId, smallvec::SmallVec<[agents::WakeCondition; 2]>>,
    ) {
        // Get news symbols for condition checking
        let news_symbols: Vec<Symbol> = self
            .news_engine
            .active_events()
            .iter()
            .filter_map(|e| e.symbol().cloned())
            .collect();

        // Delegate to agent orchestrator
        let (indices_to_call, triggered_t2) = self.agent_orchestrator.compute_agents_to_call(
            self.tick,
            current_prices,
            &news_symbols,
        );

        // Track triggered count for stats
        self.stats.t2_triggered_this_tick = triggered_t2.len();

        (indices_to_call, triggered_t2)
    }

    /// Validate and group orders by symbol for batch auction.
    ///
    /// Delegates to `AuctionEngine::collect_orders`. Updates stats from result.
    fn collect_orders_for_auction(
        &mut self,
        actions_with_state: Vec<AgentActionWithState>,
    ) -> HashMap<Symbol, Vec<Order>> {
        let ctx = OrderValidationCtx {
            position_tracker: &self.risk_manager,
            enforce_limits: self.config.enforce_position_limits,
            timestamp: self.timestamp,
            force_sequential: !self.config.parallelization.parallel_order_validation,
        };
        let result = self.auction_engine.collect_orders(actions_with_state, &ctx);

        // Apply stats from pure collection result
        self.stats.total_orders += result.total_orders;
        self.stats.rejected_orders += result.rejected_orders;

        result.orders_by_symbol
    }

    /// Build reference prices for batch auction clearing.
    ///
    /// Delegates to `AuctionEngine::build_reference_prices`.
    fn build_reference_prices(
        &self,
        orders_by_symbol: &HashMap<Symbol, Vec<Order>>,
    ) -> HashMap<String, Price> {
        self.auction_engine.build_reference_prices(
            orders_by_symbol,
            &self.market,
            self.config.get_symbol_configs(),
            !self.config.parallelization.parallel_order_validation,
        )
    }

    /// Process batch auction results into trades.
    ///
    /// Updates stats, borrow ledger, share tracking, and collects fill notifications.
    fn process_auction_results(
        &mut self,
        auction_results: HashMap<Symbol, sim_core::BatchAuctionResult>,
        position_cache: &HashMap<AgentId, HashMap<Symbol, i64>>,
    ) -> (Vec<Trade>, Vec<(AgentId, Trade, i64)>) {
        let mut tick_trades = Vec::new();
        let mut fill_notifications = Vec::new();

        for (symbol, result) in auction_results {
            self.stats.filled_orders += result.filled_orders.len() as u64;
            self.stats.total_trades += result.trades.len() as u64;

            // Update last price in order book
            if let Some(clearing_price) = result.clearing_price
                && let Some(book) = self.market.get_book_mut(&symbol)
            {
                book.set_last_price(clearing_price);
            }

            for trade in result.trades {
                let buyer_pos_before = position_cache
                    .get(&trade.buyer_id)
                    .and_then(|positions| positions.get(&trade.symbol).copied())
                    .unwrap_or(0);

                let seller_pos_before = position_cache
                    .get(&trade.seller_id)
                    .and_then(|positions| positions.get(&trade.symbol).copied())
                    .unwrap_or(0);

                // Update borrow ledger and total shares held via risk manager
                self.risk_manager.process_trade(
                    &trade,
                    seller_pos_before,
                    buyer_pos_before,
                    self.tick,
                );

                fill_notifications.push((trade.buyer_id, trade.clone(), buyer_pos_before));
                fill_notifications.push((trade.seller_id, trade.clone(), seller_pos_before));

                tick_trades.push(trade);
            }
        }

        (tick_trades, fill_notifications)
    }

    /// Generate Tier 3 background pool orders for batch auction (V3.8).
    ///
    /// Returns orders to be included in the main batch auction alongside T1/T2 orders.
    /// This is more efficient than continuous matching and ensures consistent semantics.
    fn generate_background_pool_orders(&mut self) -> Vec<Order> {
        // Build mid prices map
        let mid_prices: HashMap<Symbol, Price> = parallel::filter_map_to_hashmap(
            self.config.get_symbol_configs(),
            |sc| {
                self.market
                    .mid_price(&sc.symbol)
                    .map(|p| (sc.symbol.clone(), p))
            },
            true, // Sequential for small collections
        );

        // Build pool context (V3.8: use cached symbol_sectors from news_engine)
        let pool_ctx = PoolContext {
            tick: self.tick,
            mid_prices: &mid_prices,
            active_events: self.news_engine.active_events(),
            symbol_sectors: self.news_engine.symbol_sectors(),
        };

        // Generate orders via agent orchestrator
        let t3_orders = self
            .agent_orchestrator
            .generate_background_pool_orders(&pool_ctx);
        self.stats.t3_orders_this_tick = t3_orders.len();
        t3_orders
    }

    /// Update T3 pool accounting from batch auction results (V3.8).
    ///
    /// Called after auction completes to record T3 trades and notify counterparty agents.
    fn update_background_pool_accounting(&mut self, trades: &[Trade]) {
        self.agent_orchestrator
            .update_background_pool_accounting(trades);
    }

    /// Update recent trades storage with new trades.
    fn update_recent_trades(&mut self, tick_trades: &[Trade]) {
        self.market_data.update_recent_trades(tick_trades);
    }

    /// Notify agents of fills and update wake conditions.
    fn process_fill_notifications(&mut self, fill_notifications: Vec<(AgentId, Trade, i64)>) {
        self.agent_orchestrator.notify_fills(
            &fill_notifications,
            !self.config.parallelization.parallel_fill_notifications,
        );
    }

    /// Restore wake conditions for triggered T2 agents.
    ///
    /// After triggering, agents may need new conditions (e.g., exit conditions after entry).
    fn restore_t2_wake_conditions(
        &mut self,
        triggered_t2: &HashMap<AgentId, smallvec::SmallVec<[agents::WakeCondition; 2]>>,
    ) {
        self.agent_orchestrator.restore_wake_conditions(
            triggered_t2,
            !self.config.parallelization.parallel_wake_conditions,
        );
    }

    /// Update risk tracking with current equity values.
    fn update_risk_tracking(&mut self) {
        let prices: HashMap<Symbol, Price> = parallel::map_to_hashmap(
            self.config.get_symbol_configs(),
            |sc| {
                let price = self
                    .market
                    .last_price(&sc.symbol)
                    .unwrap_or(sc.initial_price);
                (sc.symbol.clone(), price)
            },
            !self.config.parallelization.parallel_risk_tracking, // Sequential for small collections
        );

        let equities = self
            .agent_orchestrator
            .collect_equities(&prices, !self.config.parallelization.parallel_risk_tracking);

        equities.into_iter().for_each(|(agent_id, equity)| {
            self.risk_manager.record_equity(agent_id, equity);
        });
    }

    /// Clear order books and advance tick counter.
    fn finalize_tick(&mut self) {
        // Clear all order books (orders expire after tick)
        self.market.books_mut().for_each(|book| book.clear());

        // Cleanup expired time-based wake conditions (V3.9 fix for progressive slowdown)
        self.agent_orchestrator
            .cleanup_expired_conditions(self.tick);

        // Advance time
        self.tick += 1;
        self.timestamp += 1;
        self.stats.tick = self.tick;
    }

    /// Advance the simulation by one tick.
    ///
    /// Returns trades that occurred during this tick.
    ///
    /// # Phases
    ///
    /// 0. Process news events (updates fundamentals and active events)
    /// 1. Hook: on_tick_start (V3.6)
    /// 2. Determine which agents to call (T1 always, T2 when triggered)
    /// 3. Build strategy context for agents
    /// 4. Collect agent actions (orders and cancellations)
    /// 5. Generate Tier 3 background pool orders (V3.8)
    /// 6. Hook: on_orders_collected (V3.6)
    /// 7. Run batch auction for ALL orders (T1/T2/T3) (V3.8)
    /// 8. Update T3 pool accounting (V3.8)
    /// 9. Hook: on_trades (V3.6)
    /// 10. Update market data (recent trades, candles)
    /// 11. Notify agents of fills and update wake conditions
    /// 12. Update risk tracking
    /// 13. Hook: on_tick_end (V3.6)
    /// 14. Finalize tick (clear books, advance time)
    pub fn step(&mut self) -> Vec<Trade> {
        // Phase 0: Process news events
        self.process_news_events();

        // Phase 0b (V2.5): Apply fair value drift
        self.apply_fair_value_drift();

        // V5.2: Build hook context once for pre-auction hooks (if needed)
        let has_hooks = !self.hooks.is_empty();
        let pre_auction_ctx = if has_hooks {
            Some(self.build_hook_context())
        } else {
            None
        };

        // Phase 1 (V3.6): Hook - tick start
        if let Some(ref ctx) = pre_auction_ctx {
            self.hooks.on_tick_start(ctx);
        }

        // Phase 2: Determine which agents to call (before building ctx to avoid borrow conflict)
        // This mutates wake_index, so must happen before taking immutable refs to self
        let current_prices = self.collect_current_prices();
        let (indices_to_call, triggered_t2) = self.compute_agents_to_call(&current_prices);

        // Phase 3: Build strategy context for agents
        // V5.5: Single source of truth - indicators computed once and shared
        let candles_map = self.build_candles_map();
        let trades_map = self.build_trades_map();
        let indicators = self.build_indicator_snapshot();

        // Phase 3b: Extract features and build ML prediction cache (pre-V6 SoC)
        //
        // Pipeline: extract_raw → impute(neutral_values) → cache → predict
        // Features are extracted ONCE per symbol per tick. Both ML prediction and
        // recording hooks consume the same features (training-serving parity).
        //
        // Skip during warmup — indicators have NaN values until sufficient price history.
        let past_warmup = self.tick >= self.config.ml_warmup_ticks;
        let has_extractor = self.feature_extractor.is_some();

        // Extract and impute features (if extractor configured and past warmup)
        let extracted_features: Option<HashMap<Symbol, agents::FeatureVec>> =
            if past_warmup && has_extractor {
                let symbols = self.config.symbols();
                let temp_ctx = StrategyContext::new(
                    self.tick,
                    self.timestamp,
                    &self.market,
                    &candles_map,
                    &indicators,
                    &trades_map,
                    self.news_engine.active_events(),
                    self.news_engine.fundamentals(),
                );
                let extractor = self.feature_extractor.as_ref().unwrap();
                let neutrals = extractor.neutral_values();
                // Parallel extraction + imputation
                let features_vec: Vec<_> = parallel::map_slice(
                    &symbols,
                    |symbol| {
                        let mut features = extractor.extract_market(symbol, &temp_ctx);
                        agents::impute_features(&mut features, neutrals);
                        (symbol.clone(), features)
                    },
                    !self.config.parallelization.parallel_features,
                );
                Some(features_vec.into_iter().collect())
            } else {
                None
            };

        // Build ML cache: insert features, then predict
        let ml_cache = self.model_registry.as_ref().and_then(|registry| {
            let features = extracted_features.as_ref()?;
            let mut cache = MlPredictionCache::new(self.tick);
            for (symbol, fv) in features {
                cache.insert_features(symbol.clone(), fv.clone());
            }
            registry.predict_from_cache(&mut cache);
            Some(cache)
        });

        // Build strategy context with or without ML cache
        let ctx = match &ml_cache {
            Some(cache) => StrategyContext::with_ml_cache(
                self.tick,
                self.timestamp,
                &self.market,
                &candles_map,
                &indicators,
                &trades_map,
                self.news_engine.active_events(),
                self.news_engine.fundamentals(),
                cache,
            ),
            None => StrategyContext::new(
                self.tick,
                self.timestamp,
                &self.market,
                &candles_map,
                &indicators,
                &trades_map,
                self.news_engine.active_events(),
                self.news_engine.fundamentals(),
            ),
        };

        // Phase 4: Collect agent actions
        let actions_with_state = self.collect_agent_actions(&indices_to_call, &ctx);
        self.stats.agents_called_this_tick = actions_with_state.len();

        // Phase 5: Collect orders from T1/T2 agents
        let position_cache = self.build_position_cache();
        let mut orders_by_symbol = self.collect_orders_for_auction(actions_with_state);

        // Phase 5b: Generate Tier 3 background pool orders (V3.8)
        let t3_orders = self.generate_background_pool_orders();

        // Add T3 orders to batch auction (V3.8 optimization)
        for mut order in t3_orders {
            order.id = self.auction_engine.next_order_id();
            order.timestamp = self.timestamp;
            self.stats.total_orders += 1;

            orders_by_symbol
                .entry(order.symbol.clone())
                .or_default()
                .push(order);
        }

        // Phase 6 (V3.6): Hook - orders collected (reuse pre-auction context)
        if let Some(ref ctx) = pre_auction_ctx {
            let all_orders: Vec<Order> = orders_by_symbol.values().flatten().cloned().collect();
            self.hooks.on_orders_collected(&all_orders, ctx);
        }

        // Phase 7: Run batch auction for ALL orders (T1/T2/T3) (V3.8)
        let reference_prices = self.build_reference_prices(&orders_by_symbol);
        let auction_results = run_parallel_auctions(
            orders_by_symbol,
            &reference_prices,
            self.timestamp,
            self.tick,
            self.auction_engine.order_id_counter(),
            !self.config.parallelization.parallel_auctions,
        );
        let (tick_trades, fill_notifications) =
            self.process_auction_results(auction_results, &position_cache);

        // Phase 8: Update T3 pool accounting (V3.8)
        self.update_background_pool_accounting(&tick_trades);

        // Phase 9 (V3.6): Hook - trades produced (rebuild context post-auction)
        if has_hooks {
            let hook_ctx = self.build_hook_context();
            self.hooks.on_trades(&tick_trades, &hook_ctx);
        }

        // Phase 10: Update market data
        self.update_recent_trades(&tick_trades);
        self.update_candles(&tick_trades);

        // Phase 11: Notify agents and update wake conditions
        self.process_fill_notifications(fill_notifications);
        self.restore_t2_wake_conditions(&triggered_t2);

        // Phase 12: Update risk tracking
        self.update_risk_tracking();

        // Phase 13 (V3.6): Hook - tick end with enriched data (V4.4)
        // V5.5: Reuse indicators computed in Phase 3 (single source of truth)
        // Pre-V6 SoC: Pass pre-extracted features for recording hooks
        if has_hooks {
            // Reuse features from ML cache if available; otherwise use extracted_features
            let hook_features = ml_cache
                .as_ref()
                .map(|c| c.all_features())
                .or(extracted_features);
            let hook_ctx = self.build_enriched_hook_context(&indicators, hook_features);
            self.hooks.on_tick_end(&self.stats, &hook_ctx);
        }

        // Phase 14: Finalize tick
        self.finalize_tick();

        tick_trades
    }

    /// Run the simulation for a given number of ticks.
    ///
    /// Returns total trades across all ticks.
    pub fn run(&mut self, ticks: u64) -> Vec<Trade> {
        let result = (0..ticks).fold(Vec::new(), |mut all_trades, _| {
            all_trades.extend(self.step());
            all_trades
        });

        // V3.6: Notify hooks that simulation ended
        self.finish();

        result
    }

    /// Notify hooks that simulation has ended (V5.3).
    ///
    /// Call this after manual tick loops to trigger `on_simulation_end` hooks.
    /// This is automatically called by `run()`.
    pub fn finish(&self) {
        self.hooks.on_simulation_end(&self.stats);
    }

    // =========================================================================
    // News & Fundamentals (V2.4) - delegated to NewsEngine
    // =========================================================================

    /// Process news events for the current tick.
    ///
    /// Delegates to news engine which:
    /// 1. Generates new events from the news generator
    /// 2. Applies permanent fundamental changes (earnings, guidance, rate decisions)
    /// 3. Prunes expired events from the active list
    /// 4. Applies fair value drift
    fn process_news_events(&mut self) {
        self.news_engine
            .process_tick(self.tick, self.config.verbose);
    }

    /// Apply fair value drift to all symbols (V2.5).
    ///
    /// Now handled by process_tick - this is a no-op for compatibility.
    fn apply_fair_value_drift(&mut self) {
        // Drift is now applied inside process_tick via news_engine
    }

    /// Get the currently active news events.
    pub fn active_events(&self) -> &[news::NewsEvent] {
        self.news_engine.active_events()
    }

    /// Get the symbol fundamentals.
    pub fn fundamentals(&self) -> &news::SymbolFundamentals {
        self.news_engine.fundamentals()
    }

    /// Get fair value for a symbol.
    pub fn fair_value(&self, symbol: &types::Symbol) -> Option<Price> {
        self.news_engine.fair_value(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agents::{AgentState, StrategyContext};
    use types::{Cash, Order, OrderSide, Price, Quantity};

    /// A simple test agent that does nothing.
    struct PassiveAgent {
        id: AgentId,
        state: AgentState,
    }

    impl Agent for PassiveAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _ctx: &StrategyContext<'_>) -> AgentAction {
            AgentAction::none()
        }

        fn name(&self) -> &str {
            "PassiveAgent"
        }

        fn state(&self) -> &AgentState {
            &self.state
        }
    }

    /// An agent that places a single order on construction.
    struct OneShotAgent {
        id: AgentId,
        state: AgentState,
        order: Option<Order>,
    }

    impl OneShotAgent {
        fn new(id: AgentId, order: Order) -> Self {
            let symbol = order.symbol.clone();
            Self {
                id,
                state: AgentState::new(Cash::from_float(100_000.0), &[&symbol]),
                order: Some(order),
            }
        }
    }

    impl Agent for OneShotAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _ctx: &StrategyContext<'_>) -> AgentAction {
            if let Some(order) = self.order.take() {
                AgentAction::single(order)
            } else {
                AgentAction::none()
            }
        }

        fn name(&self) -> &str {
            "OneShotAgent"
        }

        fn state(&self) -> &AgentState {
            &self.state
        }
    }

    #[test]
    fn test_empty_simulation_runs() {
        let mut sim = Simulation::with_defaults();

        // Run 1000 ticks with no agents
        let trades = sim.run(1000);

        assert!(trades.is_empty());
        assert_eq!(sim.tick(), 1000);
        assert_eq!(sim.stats().total_trades, 0);
    }

    #[test]
    fn test_passive_agents_no_trades() {
        let mut sim = Simulation::with_defaults();

        // Add some passive agents
        for i in 1..=10 {
            sim.add_agent(Box::new(PassiveAgent {
                id: AgentId(i),
                state: AgentState::default(),
            }));
        }

        let trades = sim.run(100);

        assert!(trades.is_empty());
        assert_eq!(sim.tick(), 100);
        assert_eq!(sim.agent_count(), 10);
    }

    #[test]
    fn test_orders_match() {
        let config = SimulationConfig::new("TEST").with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        // Agent 1 places a sell order
        let sell_order = Order::limit(
            AgentId(1),
            "TEST",
            OrderSide::Sell,
            Price::from_float(100.0),
            Quantity(50),
        );
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(1), sell_order)));

        // Agent 2 places a buy order at same price
        let buy_order = Order::limit(
            AgentId(2),
            "TEST",
            OrderSide::Buy,
            Price::from_float(100.0),
            Quantity(50),
        );
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(2), buy_order)));

        // First tick: both orders submitted, one rests briefly, other matches against it
        let trades = sim.step();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, Quantity(50));
        assert_eq!(trades[0].price, Price::from_float(100.0));

        assert_eq!(sim.stats().total_trades, 1);

        // After tick, book should be cleared
        assert!(sim.get_book(&"TEST".to_string()).unwrap().is_empty());
    }

    #[test]
    fn test_book_state_after_order() {
        let config = SimulationConfig::new("TEST").with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        // Add an agent that places a limit order
        let sell_order = Order::limit(
            AgentId(1),
            "TEST",
            OrderSide::Sell,
            Price::from_float(105.0),
            Quantity(100),
        );
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(1), sell_order)));

        // Tick 0: order placed, then book cleared at end of tick
        sim.step();

        // Book should be empty after tick (orders expire)
        let book = sim.get_book(&"TEST".to_string()).unwrap();
        assert_eq!(book.best_ask_price(), None);
        assert_eq!(book.best_bid_price(), None);
        assert!(book.is_empty());
    }

    #[test]
    fn test_recent_trades_tracked() {
        let config = SimulationConfig::new("TEST")
            .with_max_recent_trades(5)
            .with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        // Add seller agents (will generate orders that batch auction can match)
        for i in 1..=10 {
            let sell = Order::limit(
                AgentId(100 + i),
                "TEST",
                OrderSide::Sell,
                Price::from_float(100.0),
                Quantity(10),
            );
            sim.add_agent(Box::new(OneShotAgent::new(AgentId(100 + i), sell)));
        }

        // Add buyer agents
        for i in 1..=3 {
            let buy = Order::limit(
                AgentId(i),
                "TEST",
                OrderSide::Buy,
                Price::from_float(100.0),
                Quantity(10),
            );
            sim.add_agent(Box::new(OneShotAgent::new(AgentId(i), buy)));
        }

        // Run single tick to generate trades (all agents submit orders in same tick)
        sim.step();

        // Should have at most 5 recent trades (configured max)
        assert!(sim.recent_trades_for(&"TEST".to_string()).len() <= 5);
    }

    #[test]
    fn test_order_ids_unique() {
        let config = SimulationConfig::new("SIM").with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        let order1 = Order::market(AgentId(1), "SIM", OrderSide::Buy, Quantity(10));
        let order2 = Order::market(AgentId(1), "SIM", OrderSide::Buy, Quantity(10));

        sim.add_agent(Box::new(OneShotAgent::new(AgentId(1), order1)));
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(2), order2)));

        sim.step();

        assert_eq!(sim.stats().total_orders, 2);
        // Orders should have gotten unique IDs (1 and 2)
    }
}
