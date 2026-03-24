//! Sector Rotator Strategy - Sentiment-driven portfolio rotation (V3.3).
//!
//! This is a Tier 2 multi-symbol strategy that shifts portfolio allocation
//! toward sectors with positive sentiment and away from negative sentiment.
//!
//! # Strategy Logic
//!
//! 1. Wake on news events for any watched sector
//! 2. Aggregate sentiment per sector using `SectorSentimentAggregator`
//! 3. Compute target allocations: `base + (sentiment * scale)`, clamped to bounds
//! 4. Rebalance if drift exceeds threshold
//!
//! # Tier 2 Design
//!
//! Unlike reactive single-symbol agents, SectorRotator:
//! - Watches multiple symbols across sectors
//! - Wakes on news events (not price thresholds)
//! - Manages a portfolio of positions
//!
//! # Design (Declarative, Modular, SoC)
//!
//! - **Declarative**: Config defines sectors, bounds; strategy handles allocation
//! - **Modular**: Uses `SectorSentimentAggregator` from `quant` crate
//! - **SoC**: Computes allocations only; simulation handles execution
//!
//! # Borrow-Checker Safety
//!
//! - Owns `AgentState` and allocations (no shared references)
//! - `ctx.active_events()` returns `&[NewsEvent]` (immutable, read-only)
//! - Returns `AgentAction::multiple()` for multi-symbol rebalance orders
//! - `target_allocations` stored internally; `&mut self` allows modification

use std::collections::HashMap;

use crate::state::AgentState;
use crate::tiers::WakeCondition;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use quant::stats::{NewsEventLike, SectorSentimentAggregator};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Sector, Symbol, Tick, Trade};

// =============================================================================
// NewsEvent adapter for quant::NewsEventLike
// =============================================================================

/// Adapter to make news::NewsEvent implement quant::NewsEventLike.
///
/// This allows `SectorSentimentAggregator` to work with actual news events
/// without coupling `quant` crate to `news` crate.
struct NewsEventAdapter<'a>(&'a news::NewsEvent);

impl NewsEventLike for NewsEventAdapter<'_> {
    fn sector(&self) -> Option<Sector> {
        self.0.sector()
    }

    fn sentiment(&self) -> f64 {
        self.0.sentiment
    }

    fn magnitude(&self) -> f64 {
        self.0.magnitude
    }

    fn decay_factor(&self, current_tick: u64) -> f64 {
        self.0.decay_factor(current_tick)
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for a sector rotation strategy.
///
/// # Example
///
/// ```ignore
/// let config = SectorRotatorConfig::new()
///     .with_sector(Sector::Tech, vec!["AAPL", "MSFT", "GOOGL"])
///     .with_sector(Sector::Utilities, vec!["XOM", "CVX"])
///     .with_sentiment_scale(0.2)
///     .with_rebalance_threshold(0.05);
/// ```
#[derive(Debug, Clone)]
pub struct SectorRotatorConfig {
    /// Sectors to watch and their constituent symbols.
    pub symbols_per_sector: HashMap<Sector, Vec<Symbol>>,
    /// Base allocation per sector (default: equal weight).
    pub base_allocation: f64,
    /// How much sentiment shifts allocation (0.0 to 1.0).
    pub sentiment_scale: f64,
    /// Minimum allocation per sector (floor).
    pub min_allocation: f64,
    /// Maximum allocation per sector (ceiling).
    pub max_allocation: f64,
    /// Only trade if allocation drift exceeds this threshold.
    pub rebalance_threshold: f64,
    /// Target total portfolio value.
    pub total_capital: Cash,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Minimum ticks between rebalances (to avoid churn).
    pub min_rebalance_interval: u64,
    /// Minimum event magnitude to trigger consideration.
    pub min_event_magnitude: f64,
}

impl Default for SectorRotatorConfig {
    fn default() -> Self {
        Self {
            symbols_per_sector: HashMap::new(),
            base_allocation: 0.0,      // Will be computed from sector count
            sentiment_scale: 0.2,      // ±20% shift from base
            min_allocation: 0.05,      // Never below 5%
            max_allocation: 0.40,      // Never above 40%
            rebalance_threshold: 0.03, // 3% drift triggers rebalance
            total_capital: Cash::from_float(100_000.0),
            initial_cash: Cash::from_float(100_000.0),
            min_rebalance_interval: 10, // At least 10 ticks between rebalances
            min_event_magnitude: 0.1,   // Ignore tiny events
        }
    }
}

impl SectorRotatorConfig {
    /// Create a new empty config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: add a sector with its symbols.
    pub fn with_sector(
        mut self,
        sector: Sector,
        symbols: impl IntoIterator<Item = impl Into<Symbol>>,
    ) -> Self {
        self.symbols_per_sector
            .insert(sector, symbols.into_iter().map(|s| s.into()).collect());
        self
    }

    /// Builder: set sentiment scale.
    pub fn with_sentiment_scale(mut self, scale: f64) -> Self {
        self.sentiment_scale = scale.clamp(0.0, 1.0);
        self
    }

    /// Builder: set rebalance threshold.
    pub fn with_rebalance_threshold(mut self, threshold: f64) -> Self {
        self.rebalance_threshold = threshold.clamp(0.01, 0.5);
        self
    }

    /// Builder: set allocation bounds.
    pub fn with_allocation_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_allocation = min.clamp(0.0, 0.5);
        self.max_allocation = max.clamp(0.1, 1.0);
        self
    }

    /// Builder: set initial cash.
    pub fn with_initial_cash(mut self, cash: Cash) -> Self {
        self.initial_cash = cash;
        self.total_capital = cash;
        self
    }

    /// Builder: set minimum rebalance interval.
    pub fn with_min_rebalance_interval(mut self, ticks: u64) -> Self {
        self.min_rebalance_interval = ticks;
        self
    }

    /// Get all watched sectors.
    pub fn watched_sectors(&self) -> Vec<Sector> {
        self.symbols_per_sector.keys().copied().collect()
    }

    /// Get all watched symbols across all sectors.
    pub fn all_symbols(&self) -> Vec<Symbol> {
        self.symbols_per_sector
            .values()
            .flatten()
            .cloned()
            .collect()
    }

    /// Compute equal-weight base allocation (if not explicitly set).
    pub fn computed_base_allocation(&self) -> f64 {
        if self.base_allocation > 0.0 {
            self.base_allocation
        } else {
            let n = self.symbols_per_sector.len();
            if n > 0 { 1.0 / n as f64 } else { 0.0 }
        }
    }
}

// =============================================================================
// SectorRotator Strategy
// =============================================================================

/// Sector rotation strategy using sentiment-driven allocation.
///
/// This is a Tier 2 agent that wakes on news events and rebalances
/// portfolio allocation based on aggregated sector sentiment.
///
/// # Multi-Symbol Design (V3.3)
///
/// - Tracks positions across all symbols in watched sectors
/// - Wakes on news events for any watched symbol
/// - Returns `AgentAction::multiple()` for rebalance orders
pub struct SectorRotator {
    /// Unique agent identifier.
    id: AgentId,
    /// Strategy configuration.
    config: SectorRotatorConfig,
    /// Multi-symbol position and cash tracking.
    state: AgentState,
    /// Sentiment aggregator.
    aggregator: SectorSentimentAggregator,
    /// Current target allocations per sector.
    target_allocations: HashMap<Sector, f64>,
    /// Tick of last rebalance.
    last_rebalance_tick: Tick,
    /// All watched symbols (cached for efficiency).
    all_symbols: Vec<Symbol>,
    /// Whether we need to recompute allocations on next tick.
    needs_recompute: bool,
}

impl SectorRotator {
    /// Create a new sector rotator strategy.
    ///
    /// # Panics
    /// Panics if no sectors are configured.
    pub fn new(id: AgentId, config: SectorRotatorConfig) -> Self {
        assert!(
            !config.symbols_per_sector.is_empty(),
            "SectorRotator requires at least one sector"
        );

        let all_symbols = config.all_symbols();
        let state = AgentState::with_symbols(config.initial_cash, all_symbols.clone());

        // Initialize with equal-weight allocations
        let base = config.computed_base_allocation();
        let target_allocations: HashMap<Sector, f64> = config
            .watched_sectors()
            .into_iter()
            .map(|s| (s, base))
            .collect();

        let aggregator = SectorSentimentAggregator::with_min_magnitude(config.min_event_magnitude);

        Self {
            id,
            config,
            state,
            aggregator,
            target_allocations,
            last_rebalance_tick: 0,
            all_symbols,
            needs_recompute: false,
        }
    }

    /// Get initial wake conditions for registration with WakeConditionIndex.
    ///
    /// Registers news subscriptions for all watched symbols.
    pub fn initial_wake_conditions(&self) -> Vec<WakeCondition> {
        // Subscribe to news for all symbols in all watched sectors
        vec![WakeCondition::NewsEvent {
            symbols: self.all_symbols.iter().cloned().collect(),
        }]
    }

    /// Recompute target allocations based on current sentiment.
    fn recompute_allocations(&mut self, events: &[news::NewsEvent], current_tick: Tick) {
        // Wrap events for the aggregator
        let adapted: Vec<NewsEventAdapter> = events.iter().map(NewsEventAdapter).collect();

        // Get sentiment per sector
        let sentiments = self.aggregator.aggregate_all(&adapted, current_tick);

        // Compute new allocations
        let base = self.config.computed_base_allocation();
        let scale = self.config.sentiment_scale;
        let min = self.config.min_allocation;
        let max = self.config.max_allocation;

        // Update allocations declaratively
        self.target_allocations = self
            .config
            .watched_sectors()
            .into_iter()
            .map(|sector| {
                let sentiment = sentiments.get(&sector).map(|s| s.sentiment).unwrap_or(0.0);
                let raw_allocation = base + (sentiment * scale);
                (sector, raw_allocation.clamp(min, max))
            })
            .collect();

        // Normalize to sum to 1.0
        let total: f64 = self.target_allocations.values().sum();
        if total > 0.0 {
            self.target_allocations
                .values_mut()
                .for_each(|a| *a /= total);
        }
    }

    /// Generate rebalance orders to move toward target allocations.
    fn generate_rebalance_orders(&self, ctx: &StrategyContext<'_>) -> Vec<Order> {
        // Get current prices for all symbols
        let prices: HashMap<Symbol, Price> = self
            .all_symbols
            .iter()
            .filter_map(|s| ctx.mid_price(s).map(|p| (s.clone(), p)))
            .collect();

        if prices.is_empty() {
            return Vec::new();
        }

        // Compute current portfolio value
        let current_equity = self.state.equity(&prices);
        let target_portfolio_value = current_equity.to_float();

        // Generate orders for each sector's symbols
        self.target_allocations
            .iter()
            .flat_map(|(sector, target_pct)| {
                let sector_symbols = self
                    .config
                    .symbols_per_sector
                    .get(sector)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                if sector_symbols.is_empty() {
                    return Vec::new();
                }

                // Target value for this sector, equal weight within sector
                let sector_target_value = target_portfolio_value * target_pct;
                let per_symbol_target = sector_target_value / sector_symbols.len() as f64;

                sector_symbols
                    .iter()
                    .filter_map(|symbol| {
                        let price = prices.get(symbol)?;
                        let price_f64 = price.to_float();
                        if price_f64 <= 0.0 {
                            return None;
                        }

                        // Current value in this symbol
                        let current_position = self.state.position_for(symbol);
                        let current_value = current_position as f64 * price_f64;

                        // Delta needed
                        let delta_value = per_symbol_target - current_value;
                        let delta_pct = if current_value.abs() > f64::EPSILON {
                            delta_value.abs() / current_value.abs()
                        } else if per_symbol_target.abs() > f64::EPSILON {
                            1.0 // New position from zero
                        } else {
                            0.0
                        };

                        // Only rebalance if drift exceeds threshold
                        if delta_pct < self.config.rebalance_threshold && current_position != 0 {
                            return None;
                        }

                        // Calculate shares to trade
                        let target_shares = (per_symbol_target / price_f64).round() as i64;
                        let delta_shares = target_shares - current_position;

                        if delta_shares == 0 {
                            return None;
                        }

                        let (side, price_mult, qty) = if delta_shares > 0 {
                            (OrderSide::Buy, 0.999, delta_shares as u64) // Bid above mid to qualify
                        } else {
                            (OrderSide::Sell, 1.001, (-delta_shares) as u64) // Ask below mid to qualify
                        };

                        // Apply floor_price to prevent negative price spirals
                        Some(Order::limit(
                            self.id,
                            symbol,
                            side,
                            Price::from_float(floor_price(price_f64 * price_mult)),
                            Quantity(qty),
                        ))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl Agent for SectorRotator {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Check minimum interval between rebalances
        if ctx.tick < self.last_rebalance_tick + self.config.min_rebalance_interval {
            return AgentAction::none();
        }

        // Get active events
        let events = ctx.active_events();

        // Check if any events affect our watched sectors
        let has_relevant_events = events.iter().any(|e| {
            if let Some(sector) = e.sector() {
                self.config.symbols_per_sector.contains_key(&sector)
            } else {
                // Check if event's symbol is in our watched list
                e.symbol()
                    .map(|s| self.all_symbols.contains(s))
                    .unwrap_or(false)
            }
        });

        if !has_relevant_events && !self.needs_recompute {
            return AgentAction::none();
        }

        // Recompute allocations based on current sentiment
        self.recompute_allocations(events, ctx.tick);
        self.needs_recompute = false;

        // Generate rebalance orders
        let orders = self.generate_rebalance_orders(ctx);

        if orders.is_empty() {
            return AgentAction::none();
        }

        self.last_rebalance_tick = ctx.tick;
        (0..orders.len()).for_each(|_| self.state.record_order());

        AgentAction::multiple(orders)
    }

    fn on_fill(&mut self, trade: &Trade) {
        // Use separate if blocks (not else if) to handle self-trades correctly.
        if trade.buyer_id == self.id {
            self.state
                .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
        }
        if trade.seller_id == self.id {
            self.state
                .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
        }
    }

    fn name(&self) -> &str {
        "SectorRotator"
    }

    fn state(&self) -> &AgentState {
        &self.state
    }

    fn position(&self) -> i64 {
        self.state.position()
    }

    fn position_for(&self, symbol: &str) -> i64 {
        self.state.position_for(symbol)
    }

    fn positions(&self) -> &HashMap<Symbol, crate::state::PositionEntry> {
        self.state.positions()
    }

    fn watched_symbols(&self) -> Vec<Symbol> {
        self.all_symbols.clone()
    }

    fn equity(&self, prices: &HashMap<Symbol, Price>) -> Cash {
        self.state.equity(prices)
    }

    fn equity_for(&self, symbol: &str, price: Price) -> Cash {
        self.state.equity_for(symbol, price)
    }

    fn is_market_maker(&self) -> bool {
        false
    }

    fn is_reactive(&self) -> bool {
        true // Tier 2: wakes on news events
    }

    fn initial_wake_conditions(&self, _current_tick: Tick) -> Vec<WakeCondition> {
        // Subscribe to news for all symbols in all watched sectors
        vec![WakeCondition::NewsEvent {
            symbols: self.all_symbols.iter().cloned().collect(),
        }]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quant::IndicatorSnapshot;
    use sim_core::Market;
    use std::collections::HashMap;
    use types::{Candle, OrderId};

    fn setup_test_market() -> Market {
        let mut market = Market::new();

        // Add order books for tech symbols
        for (symbol, price) in [
            ("AAPL", 150.0),
            ("MSFT", 350.0),
            ("XOM", 100.0),
            ("CVX", 120.0),
        ] {
            market.add_symbol(symbol);
            let book = market.get_book_mut(&symbol.to_string()).unwrap();
            let mut bid = Order::limit(
                AgentId(99),
                symbol,
                OrderSide::Buy,
                Price::from_float(price * 0.99),
                Quantity(1000),
            );
            bid.id = OrderId(rand::random::<u64>());
            let mut ask = Order::limit(
                AgentId(99),
                symbol,
                OrderSide::Sell,
                Price::from_float(price * 1.01),
                Quantity(1000),
            );
            ask.id = OrderId(rand::random::<u64>());
            book.add_order(bid).unwrap();
            book.add_order(ask).unwrap();
        }

        market
    }

    #[test]
    fn test_sector_rotator_config_builder() {
        let config = SectorRotatorConfig::new()
            .with_sector(Sector::Tech, vec!["AAPL", "MSFT"])
            .with_sector(Sector::Utilities, vec!["XOM", "CVX"])
            .with_sentiment_scale(0.3)
            .with_rebalance_threshold(0.05);

        assert_eq!(config.symbols_per_sector.len(), 2);
        assert_eq!(config.symbols_per_sector[&Sector::Tech].len(), 2);
        assert!((config.sentiment_scale - 0.3).abs() < 0.001);
        assert!((config.rebalance_threshold - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_sector_rotator_new() {
        let config = SectorRotatorConfig::new()
            .with_sector(Sector::Tech, vec!["AAPL", "MSFT"])
            .with_sector(Sector::Utilities, vec!["XOM"]);

        let agent = SectorRotator::new(AgentId(1), config);

        assert_eq!(agent.id(), AgentId(1));
        assert_eq!(agent.watched_symbols().len(), 3);
        assert!(agent.is_reactive());
        assert!(!agent.is_market_maker());
    }

    #[test]
    fn test_sector_rotator_initial_wake_conditions() {
        let config = SectorRotatorConfig::new()
            .with_sector(Sector::Tech, vec!["AAPL", "MSFT"])
            .with_sector(Sector::Utilities, vec!["XOM"]);

        let agent = SectorRotator::new(AgentId(1), config);
        let conditions = agent.initial_wake_conditions();

        assert_eq!(conditions.len(), 1);
        match &conditions[0] {
            WakeCondition::NewsEvent { symbols } => {
                assert_eq!(symbols.len(), 3);
            }
            _ => panic!("Expected NewsEvent condition"),
        }
    }

    #[test]
    #[should_panic(expected = "requires at least one sector")]
    fn test_sector_rotator_empty_config() {
        let config = SectorRotatorConfig::new();
        SectorRotator::new(AgentId(1), config);
    }

    #[test]
    fn test_sector_rotator_equal_allocation() {
        let config = SectorRotatorConfig::new()
            .with_sector(Sector::Tech, vec!["AAPL"])
            .with_sector(Sector::Utilities, vec!["XOM"])
            .with_sector(Sector::Healthcare, vec!["JNJ"]);

        let agent = SectorRotator::new(AgentId(1), config);

        // Should have equal 1/3 allocation per sector
        let expected = 1.0 / 3.0;
        for allocation in agent.target_allocations.values() {
            assert!(
                (*allocation - expected).abs() < 0.001,
                "Expected {}, got {}",
                expected,
                allocation
            );
        }
    }

    #[test]
    fn test_sector_rotator_no_action_without_events() {
        let config = SectorRotatorConfig::new()
            .with_sector(Sector::Tech, vec!["AAPL", "MSFT"])
            .with_min_rebalance_interval(0);

        let mut agent = SectorRotator::new(AgentId(1), config);

        let market = setup_test_market();
        let candles: HashMap<Symbol, Vec<Candle>> = HashMap::new();
        let indicators = IndicatorSnapshot::new(100);
        let recent_trades: HashMap<Symbol, Vec<Trade>> = HashMap::new();
        let events = vec![]; // No events
        let fundamentals = news::SymbolFundamentals::default();

        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let action = agent.on_tick(&ctx);
        assert!(action.orders.is_empty());
    }
}
