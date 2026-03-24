//! Reactive agent implementation for Tier 2.
//!
//! The `ReactiveAgent` is the core struct for Tier 2 lightweight agents.
//! It holds strategies, tracks position state, and generates orders when
//! wake conditions are triggered.
//!
//! # Design Principles
//!
//! - **Declarative**: Agent declares strategies; wake conditions derived automatically
//! - **Modular**: Strategies are composable enum variants
//! - **Position Guards**: First-match logic with position checks prevents invalid trades
//!
//! # Integration with Simulation (V3.2)
//!
//! ReactiveAgents implement the `Agent` trait for integration with the Tier 1
//! simulation loop. In `on_tick`, they check wake conditions and act when
//! conditions are met, rather than polling every tick.

use crate::Agent;
use crate::StrategyContext;
use crate::state::AgentState;
use crate::tier2::context::LightweightContext;
use crate::tier2::portfolio::ReactivePortfolio;
use crate::tier2::strategies::{ReactiveStrategyType, StrategyValidation, validate_strategies};
use crate::tiers::{ConditionUpdate, CrossDirection, PriceReference, WakeCondition};
use crate::traits::AgentAction;
use smallvec::SmallVec;
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Tick, Trade};

/// A lightweight reactive agent for Tier 2 simulations.
///
/// Reactive agents wake only when specific conditions are met, making them
/// efficient for large-scale simulations (1K-10K agents).
///
/// # Integration with Simulation (V3.2)
///
/// ReactiveAgents implement the `Agent` trait, delegating position/cash tracking
/// to `AgentState` (SoC). Strategy-specific tracking (cost_basis, high_water_mark)
/// remains here for reactive decision-making.
///
/// # Memory Budget
///
/// Target ~300 bytes per agent (increased from 200 due to AgentState):
/// - `id`: 8 bytes
/// - `state`: ~80 bytes (AgentState with single symbol)
/// - `portfolio`: ~32 bytes (String inline)
/// - `strategies`: ~128 bytes (SmallVec inline for 4 strategies)
/// - `tracking fields`: ~48 bytes
#[derive(Debug, Clone)]
pub struct ReactiveAgent {
    /// Unique agent identifier.
    id: AgentId,

    /// Agent state for position, cash, and P&L tracking (delegated).
    state: AgentState,

    /// Portfolio scope (which symbols this agent trades).
    portfolio: ReactivePortfolio,

    /// Strategies this agent uses.
    ///
    /// SmallVec stores up to 4 strategies inline without heap allocation.
    /// Most agents have 2-3 (one entry + one or more exits).
    strategies: SmallVec<[ReactiveStrategyType; 4]>,

    /// Maximum position size (shares). Enforced on entry strategies.
    max_position: Quantity,

    /// Cost basis (weighted average entry price) for strategy decisions.
    /// Note: AgentState tracks its own avg_cost for P&L; this is for reactive logic.
    cost_basis: Price,

    /// High water mark since position opened (for trailing stops).
    high_water_mark: Price,

    /// Low water mark since position opened.
    low_water_mark: Price,

    /// Tick when current position was opened (for timed exits).
    position_open_tick: Option<Tick>,

    /// Session open price (captured once).
    session_open_price: Option<Price>,

    /// Whether agent is enabled (can be disabled to pause trading).
    enabled: bool,

    /// Last tick this agent evaluated conditions (for interval strategies).
    last_evaluated_tick: Tick,
}

impl ReactiveAgent {
    /// Create a new reactive agent with initial cash.
    ///
    /// # Arguments
    /// * `id` - Unique agent identifier
    /// * `portfolio` - Portfolio scope (which symbols to trade)
    /// * `strategies` - Entry and exit strategies
    /// * `max_position` - Maximum position size
    /// * `initial_cash` - Starting cash balance
    ///
    /// # Panics
    ///
    /// Panics if strategies don't include both entry and exit capability.
    pub fn new(
        id: AgentId,
        portfolio: ReactivePortfolio,
        strategies: Vec<ReactiveStrategyType>,
        max_position: Quantity,
        initial_cash: Cash,
    ) -> Self {
        let strategies: SmallVec<[ReactiveStrategyType; 4]> = strategies.into_iter().collect();

        // Validate strategy set
        match validate_strategies(&strategies) {
            StrategyValidation::Valid => {}
            StrategyValidation::MissingEntry => {
                panic!("ReactiveAgent requires at least one entry strategy")
            }
            StrategyValidation::MissingExit => {
                panic!("ReactiveAgent requires at least one exit strategy")
            }
        }

        // Initialize AgentState with the symbol from portfolio
        let symbol = portfolio.primary_symbol().clone();
        let state = AgentState::new(initial_cash, &[&symbol]);

        Self {
            id,
            state,
            portfolio,
            strategies,
            max_position,
            cost_basis: Price(0),
            high_water_mark: Price(0),
            low_water_mark: Price(i64::MAX),
            position_open_tick: None,
            session_open_price: None,
            enabled: true,
            last_evaluated_tick: 0,
        }
    }

    /// Create a builder for constructing agents.
    pub fn builder(id: AgentId, portfolio: ReactivePortfolio) -> ReactiveAgentBuilder {
        ReactiveAgentBuilder::new(id, portfolio)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the agent ID.
    pub fn agent_id(&self) -> AgentId {
        self.id
    }

    /// Get a reference to the agent state.
    pub fn agent_state(&self) -> &AgentState {
        &self.state
    }

    /// Get the portfolio scope.
    pub fn portfolio(&self) -> &ReactivePortfolio {
        &self.portfolio
    }

    /// Get the current position (delegates to AgentState).
    pub fn current_position(&self) -> i64 {
        self.state.position_for(self.portfolio.primary_symbol())
    }

    /// Check if agent has a position.
    pub fn has_position(&self) -> bool {
        self.current_position() != 0
    }

    /// Check if agent is long.
    pub fn is_long(&self) -> bool {
        self.current_position() > 0
    }

    /// Check if agent is short.
    pub fn is_short(&self) -> bool {
        self.current_position() < 0
    }

    /// Check if agent is flat.
    pub fn is_flat(&self) -> bool {
        self.current_position() == 0
    }

    /// Get the cost basis (for strategy decisions).
    pub fn cost_basis(&self) -> Price {
        self.cost_basis
    }

    /// Get remaining capacity for new positions.
    pub fn remaining_capacity(&self) -> Quantity {
        let position = self.current_position();
        Quantity(self.max_position.0.saturating_sub(position.unsigned_abs()))
    }

    /// Check if agent is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable the agent.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    // =========================================================================
    // Wake Condition Generation
    // =========================================================================

    /// Generate initial wake conditions based on strategies.
    ///
    /// Called when agent is first registered with the simulation.
    /// Only registers ENTRY conditions (ThresholdBuyer, NewsReactor).
    /// Exit conditions (ThresholdSeller, StopLoss, TakeProfit) are registered
    /// later via fill_wake_conditions() when a position is opened.
    pub fn initial_wake_conditions(&self, _current_tick: Tick) -> Vec<WakeCondition> {
        let symbol = self.portfolio.primary_symbol().clone();

        self.strategies
            .iter()
            .filter_map(|strategy| match strategy {
                // Entry condition: register at spawn
                ReactiveStrategyType::ThresholdBuyer { buy_price, .. } => {
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold: *buy_price,
                        direction: CrossDirection::Below,
                    })
                }
                // NewsReactor is both entry and exit, register at spawn
                ReactiveStrategyType::NewsReactor { .. } => Some(WakeCondition::NewsEvent {
                    symbols: smallvec::smallvec![symbol.clone()],
                }),
                // Exit conditions: registered on fill via fill_wake_conditions()
                ReactiveStrategyType::ThresholdSeller { .. }
                | ReactiveStrategyType::StopLoss { .. }
                | ReactiveStrategyType::TakeProfit { .. } => None,
            })
            .collect()
    }

    /// Generate wake conditions after a fill (for exit strategies).
    ///
    /// Called when agent opens a position. Computes absolute thresholds
    /// from cost_basis for percentage-based exit strategies.
    pub fn fill_wake_conditions(&self) -> Vec<WakeCondition> {
        if !self.has_position() {
            return Vec::new();
        }

        let symbol = self.portfolio.primary_symbol().clone();

        self.strategies
            .iter()
            .filter_map(|strategy| match strategy {
                ReactiveStrategyType::StopLoss { stop_pct } => {
                    let threshold = Price((self.cost_basis.0 as f64 * (1.0 - stop_pct)) as i64);
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold,
                        direction: CrossDirection::Below,
                    })
                }
                ReactiveStrategyType::TakeProfit { target_pct } => {
                    let threshold = Price((self.cost_basis.0 as f64 * (1.0 + target_pct)) as i64);
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold,
                        direction: CrossDirection::Above,
                    })
                }
                ReactiveStrategyType::ThresholdSeller { sell_price, .. } => {
                    // Fixed-price exit condition
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold: *sell_price,
                        direction: CrossDirection::Above,
                    })
                }
                // Entry strategies don't need fill-time conditions
                _ => None,
            })
            .collect()
    }

    /// Get ALL wake conditions that should currently be active based on state.
    ///
    /// This is called after a trigger fires to restore the correct set of conditions.
    /// Unlike `compute_condition_update`, this doesn't track transitions - it just
    /// returns what SHOULD be active right now.
    ///
    /// Rules:
    /// - Entry conditions (ThresholdBuyer): active if flat (no position)
    /// - Exit conditions (StopLoss, TakeProfit, ThresholdSeller): active if holding
    /// - NewsReactor: always active
    pub fn current_wake_conditions(&self) -> Vec<WakeCondition> {
        let symbol = self.portfolio.primary_symbol().clone();
        let is_flat = self.is_flat();
        let has_position = self.has_position();

        self.strategies
            .iter()
            .filter_map(|strategy| match strategy {
                // Entry: active when flat
                ReactiveStrategyType::ThresholdBuyer { buy_price, .. } if is_flat => {
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold: *buy_price,
                        direction: CrossDirection::Below,
                    })
                }

                // Exits: active when holding position
                ReactiveStrategyType::StopLoss { stop_pct } if has_position => {
                    let threshold = Price((self.cost_basis.0 as f64 * (1.0 - stop_pct)) as i64);
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold,
                        direction: CrossDirection::Below,
                    })
                }
                ReactiveStrategyType::TakeProfit { target_pct } if has_position => {
                    let threshold = Price((self.cost_basis.0 as f64 * (1.0 + target_pct)) as i64);
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold,
                        direction: CrossDirection::Above,
                    })
                }
                ReactiveStrategyType::ThresholdSeller { sell_price, .. } if has_position => {
                    Some(WakeCondition::PriceCross {
                        symbol: symbol.clone(),
                        threshold: *sell_price,
                        direction: CrossDirection::Above,
                    })
                }

                // NewsReactor: always active (but registered at spawn, not here)
                // Other cases: not active
                _ => None,
            })
            .collect()
    }

    /// Compute condition updates after a fill based on state transitions.
    ///
    /// Detects transitions and returns appropriate add/remove operations:
    /// - Entry conditions: removed when at capacity, added back when has capacity
    /// - Exit conditions: added when opening position, removed when closing
    ///
    /// # Arguments
    /// * `position_before` - Agent's position before the fill
    pub fn compute_condition_update(&self, position_before: i64) -> Option<ConditionUpdate> {
        let position_after = self.current_position();
        let symbol = self.portfolio.primary_symbol().clone();

        let had_position = position_before > 0;
        let has_position = position_after > 0;
        let bought = position_after > position_before;
        let sold_to_flat = had_position && !has_position;

        let mut update = ConditionUpdate::new(self.id);

        // Process each strategy for condition changes
        for strategy in &self.strategies {
            match strategy {
                // Entry condition: ThresholdBuyer
                // Remove only when at max capacity (can't buy more)
                // Re-add when has capacity again (after any sell)
                ReactiveStrategyType::ThresholdBuyer { buy_price, .. } => {
                    let at_capacity = self.remaining_capacity().0 == 0;
                    let was_at_capacity = position_before >= self.max_position.0 as i64;

                    if bought && at_capacity {
                        // Just hit capacity - remove entry condition
                        update.remove.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold: *buy_price,
                            direction: CrossDirection::Below,
                        });
                    }
                    if sold_to_flat || (was_at_capacity && !at_capacity) {
                        // Sold some/all - re-add entry condition if we have capacity
                        update.add.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold: *buy_price,
                            direction: CrossDirection::Below,
                        });
                    }
                }

                // Exit condition: StopLoss (percentage-based, uses cost_basis)
                ReactiveStrategyType::StopLoss { stop_pct } => {
                    if !had_position && has_position {
                        // Just opened position - add stop loss
                        let threshold = Price((self.cost_basis.0 as f64 * (1.0 - stop_pct)) as i64);
                        update.add.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold,
                            direction: CrossDirection::Below,
                        });
                    }
                    if sold_to_flat {
                        // Closed position - remove stop loss
                        let threshold = Price((self.cost_basis.0 as f64 * (1.0 - stop_pct)) as i64);
                        update.remove.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold,
                            direction: CrossDirection::Below,
                        });
                    }
                }

                // Exit condition: TakeProfit (percentage-based, uses cost_basis)
                ReactiveStrategyType::TakeProfit { target_pct } => {
                    if !had_position && has_position {
                        // Just opened position - add take profit
                        let threshold =
                            Price((self.cost_basis.0 as f64 * (1.0 + target_pct)) as i64);
                        update.add.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold,
                            direction: CrossDirection::Above,
                        });
                    }
                    if sold_to_flat {
                        // Closed position - remove take profit
                        let threshold =
                            Price((self.cost_basis.0 as f64 * (1.0 + target_pct)) as i64);
                        update.remove.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold,
                            direction: CrossDirection::Above,
                        });
                    }
                }

                // Exit condition: ThresholdSeller (fixed-price exit)
                ReactiveStrategyType::ThresholdSeller { sell_price, .. } => {
                    if !had_position && has_position {
                        // Just opened position - add threshold sell
                        update.add.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold: *sell_price,
                            direction: CrossDirection::Above,
                        });
                    }
                    if sold_to_flat {
                        // Closed position - remove threshold sell
                        update.remove.push(WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold: *sell_price,
                            direction: CrossDirection::Above,
                        });
                    }
                }

                // NewsReactor: Always active, no updates needed
                _ => {}
            }
        }

        if update.is_empty() {
            None
        } else {
            Some(update)
        }
    }

    /// Resolve a price reference to an actual price.
    /// Reserved for future use with deferred strategies (DipBuyer, BreakoutEntry).
    #[allow(dead_code)]
    fn resolve_reference(&self, reference: &PriceReference) -> Option<Price> {
        match reference {
            PriceReference::CostBasis => {
                if self.cost_basis.0 > 0 {
                    Some(self.cost_basis)
                } else {
                    None
                }
            }
            PriceReference::OpenPrice => self.session_open_price,
            PriceReference::HighWaterMark => {
                if self.high_water_mark.0 > 0 {
                    Some(self.high_water_mark)
                } else {
                    None
                }
            }
            PriceReference::LowWaterMark => {
                if self.low_water_mark.0 < i64::MAX {
                    Some(self.low_water_mark)
                } else {
                    None
                }
            }
            PriceReference::FundamentalValue => None, // Must come from context
            PriceReference::Snapshot(price) => Some(*price),
        }
    }

    // =========================================================================
    // Wake Processing
    // =========================================================================

    /// Process a wake event and return any actions.
    ///
    /// Returns the action to take and any condition updates needed.
    pub fn on_wake(
        &mut self,
        ctx: &LightweightContext<'_>,
    ) -> (Option<AgentAction>, Option<ConditionUpdate>) {
        if !self.enabled {
            return (None, None);
        }

        // Update session open price if not set
        if self.session_open_price.is_none()
            && let Some(snapshot) = ctx.price_snapshot(self.portfolio.primary_symbol())
        {
            self.session_open_price = Some(snapshot.open);
        }

        // Update high/low water marks if we have a position
        if self.has_position()
            && let Some(price) = ctx.price(self.portfolio.primary_symbol())
        {
            if price.0 > self.high_water_mark.0 {
                self.high_water_mark = price;
            }
            if price.0 < self.low_water_mark.0 {
                self.low_water_mark = price;
            }
        }

        // Try strategies in order until one triggers
        self.strategies
            .iter()
            .find_map(|strategy| self.try_strategy(strategy, ctx))
            .map(|action| {
                let update = self.generate_condition_update(&action, ctx.tick);
                (Some(action), update)
            })
            .unwrap_or((None, None))
    }

    /// Try to execute a strategy, returning an action if conditions are met.
    fn try_strategy(
        &self,
        strategy: &ReactiveStrategyType,
        ctx: &LightweightContext<'_>,
    ) -> Option<AgentAction> {
        let symbol = self.portfolio.primary_symbol();

        match strategy {
            // Entry strategy - ThresholdBuyer
            ReactiveStrategyType::ThresholdBuyer { size_fraction, .. }
                if self.remaining_capacity().0 > 0 =>
            {
                if ctx.woke_on_price_cross() {
                    let size = self.calculate_order_size(*size_fraction);
                    if size.0 > 0 {
                        return Some(self.market_order(symbol, OrderSide::Buy, size));
                    }
                }
                None
            }

            // Exit strategies
            ReactiveStrategyType::StopLoss { .. } if self.is_long() => {
                if ctx.woke_on_price_cross() {
                    return Some(self.market_order(
                        symbol,
                        OrderSide::Sell,
                        Quantity(self.current_position() as u64),
                    ));
                }
                None
            }
            ReactiveStrategyType::TakeProfit { .. } if self.is_long() => {
                if ctx.woke_on_price_cross() {
                    return Some(self.market_order(
                        symbol,
                        OrderSide::Sell,
                        Quantity(self.current_position() as u64),
                    ));
                }
                None
            }
            ReactiveStrategyType::ThresholdSeller { size_fraction, .. } if self.is_long() => {
                if ctx.woke_on_price_cross() {
                    let position = self.current_position() as u64;
                    let size = (position as f64 * size_fraction) as u64;
                    if size > 0 {
                        return Some(self.market_order(symbol, OrderSide::Sell, Quantity(size)));
                    }
                }
                None
            }

            // Optional modifier - NewsReactor (can enter or exit)
            ReactiveStrategyType::NewsReactor {
                min_magnitude,
                sentiment_multiplier,
            } => {
                if let Some(news) = ctx.news_event
                    && news.magnitude >= *min_magnitude
                {
                    let base_size = (self.max_position.0 as f64 * 0.1) as u64;
                    let size =
                        (base_size as f64 * news.sentiment.abs() * sentiment_multiplier) as u64;
                    if size > 0 {
                        let side = if news.sentiment > 0.0 {
                            OrderSide::Buy
                        } else {
                            OrderSide::Sell
                        };
                        // Check position guards
                        let can_trade = match side {
                            OrderSide::Buy => self.remaining_capacity().0 > 0,
                            OrderSide::Sell => self.has_position(),
                        };
                        if can_trade {
                            let order_size = match side {
                                OrderSide::Buy => size.min(self.remaining_capacity().0),
                                OrderSide::Sell => size.min(self.current_position() as u64),
                            };
                            return Some(self.market_order(symbol, side, Quantity(order_size)));
                        }
                    }
                }
                None
            }

            // Strategy didn't match conditions or position guard failed
            _ => None,
        }
    }

    /// Create a market order action.
    fn market_order(&self, symbol: &str, side: OrderSide, quantity: Quantity) -> AgentAction {
        AgentAction::single(Order::market(self.id, symbol, side, quantity))
    }

    /// Calculate order size based on fraction of max position.
    fn calculate_order_size(&self, fraction: f64) -> Quantity {
        let target = (self.max_position.0 as f64 * fraction) as u64;
        let remaining = self.remaining_capacity().0;
        Quantity(target.min(remaining))
    }

    /// Generate condition updates after an action.
    fn generate_condition_update(
        &self,
        _action: &AgentAction,
        _tick: Tick,
    ) -> Option<ConditionUpdate> {
        // TODO: Implement condition updates based on position changes
        // For now, return None - conditions will be regenerated
        None
    }

    // =========================================================================
    // Fill Processing (internal, called by Agent trait impl)
    // =========================================================================

    /// Process a fill notification and update strategy-specific tracking.
    ///
    /// Updates cost_basis, water marks, and position timing for strategy decisions.
    /// AgentState position/cash is updated via the Agent trait impl's on_fill.
    fn process_fill(&mut self, side: OrderSide, quantity: Quantity, price: Price, tick: Tick) {
        let qty = quantity.0 as i64;
        let old_position = self.current_position();
        let symbol = self.portfolio.primary_symbol();

        // Update AgentState (position and cash tracking)
        let trade_value = Cash(price.0 * quantity.0 as i64); // value in fixed-point
        match side {
            OrderSide::Buy => {
                self.state.on_buy(symbol, quantity.raw(), trade_value);
            }
            OrderSide::Sell => {
                self.state.on_sell(symbol, quantity.raw(), trade_value);
            }
        }

        let new_position = self.current_position();

        // Update strategy-specific cost basis tracking
        match side {
            OrderSide::Buy => {
                if old_position >= 0 {
                    let total_cost = self.cost_basis.0 * old_position + price.0 * qty;
                    if new_position > 0 {
                        self.cost_basis = Price(total_cost / new_position);
                    }
                } else if new_position > 0 {
                    // Covering short and going long
                    self.cost_basis = price;
                }
            }
            OrderSide::Sell => {
                if new_position < 0 && old_position >= 0 {
                    // Opening short
                    self.cost_basis = price;
                }
            }
        }

        // Track position open tick
        if old_position == 0 && new_position != 0 {
            self.position_open_tick = Some(tick);
            self.high_water_mark = price;
            self.low_water_mark = price;
        } else if new_position == 0 {
            self.position_open_tick = None;
            self.high_water_mark = Price(0);
            self.low_water_mark = Price(i64::MAX);
            self.cost_basis = Price(0);
        }
    }
}

// =============================================================================
// Agent Trait Implementation (V3.2)
// =============================================================================

impl Agent for ReactiveAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "ReactiveAgent"
    }

    fn state(&self) -> &AgentState {
        &self.state
    }

    /// Tier 2 agents are reactive - they wake only when conditions trigger.
    fn is_reactive(&self) -> bool {
        true
    }

    /// Tier 2 agents check conditions on each tick and act when triggered.
    ///
    /// In a full Tier 2 implementation, the WakeConditionIndex would wake agents
    /// only when conditions are met. For V3.2 integration with Tier 1 loop,
    /// we poll conditions each tick (still efficient due to simple checks).
    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        if !self.enabled {
            return AgentAction::none();
        }

        let symbol = self.portfolio.primary_symbol();
        let tick = ctx.tick;

        // Get current price
        let Some(current_price) = ctx.last_price(symbol) else {
            return AgentAction::none();
        };

        // Update session open price if not set
        if self.session_open_price.is_none() {
            self.session_open_price = Some(current_price);
        }

        // Update high/low water marks if we have a position
        if self.has_position() {
            if current_price.0 > self.high_water_mark.0 {
                self.high_water_mark = current_price;
            }
            if current_price.0 < self.low_water_mark.0 {
                self.low_water_mark = current_price;
            }
        }

        // Check each strategy for trigger conditions
        self.strategies
            .clone()
            .iter()
            .find_map(|strategy| {
                self.check_strategy_trigger(strategy, current_price, tick, ctx)
                    .inspect(|_| {
                        self.last_evaluated_tick = tick;
                    })
            })
            .unwrap_or_else(AgentAction::none)
    }

    fn on_fill(&mut self, trade: &Trade) {
        // Handle self-trades correctly by processing both sides.
        // Use tick 0 as placeholder - real tick comes from simulation.
        if trade.buyer_id == self.id {
            self.process_fill(OrderSide::Buy, trade.quantity, trade.price, 0);
        }
        if trade.seller_id == self.id {
            self.process_fill(OrderSide::Sell, trade.quantity, trade.price, 0);
        }
    }

    fn initial_wake_conditions(&self, current_tick: types::Tick) -> Vec<crate::WakeCondition> {
        // Delegate to the inherent method
        ReactiveAgent::initial_wake_conditions(self, current_tick)
    }

    fn fill_wake_conditions(&self) -> Vec<crate::WakeCondition> {
        // Delegate to the inherent method
        ReactiveAgent::fill_wake_conditions(self)
    }

    fn current_wake_conditions(&self) -> Vec<crate::WakeCondition> {
        // Delegate to the inherent method
        ReactiveAgent::current_wake_conditions(self)
    }

    fn post_fill_condition_update(
        &self,
        position_before: i64,
    ) -> Option<crate::tiers::ConditionUpdate> {
        // Delegate to the inherent method
        ReactiveAgent::compute_condition_update(self, position_before)
    }
}

impl ReactiveAgent {
    /// Check if a strategy's conditions are triggered.
    ///
    /// This is the polling path used in on_tick(). With WakeConditionIndex
    /// integration, most agents will use on_wake() instead and this becomes
    /// a fallback for edge cases.
    fn check_strategy_trigger(
        &self,
        strategy: &ReactiveStrategyType,
        current_price: Price,
        _tick: Tick,
        ctx: &StrategyContext<'_>,
    ) -> Option<AgentAction> {
        let symbol = self.portfolio.primary_symbol();

        match strategy {
            // Entry strategy - ThresholdBuyer
            ReactiveStrategyType::ThresholdBuyer {
                buy_price,
                size_fraction,
            } if self.remaining_capacity().0 > 0 => {
                if current_price.0 <= buy_price.0 {
                    let size = self.calculate_order_size(*size_fraction);
                    if size.0 > 0 {
                        return Some(self.market_order(symbol, OrderSide::Buy, size));
                    }
                }
                None
            }

            // Exit strategies
            ReactiveStrategyType::StopLoss { stop_pct } if self.is_long() => {
                let threshold = Price((self.cost_basis.0 as f64 * (1.0 - stop_pct)) as i64);
                if current_price.0 <= threshold.0 {
                    let position = self.current_position();
                    return Some(self.market_order(
                        symbol,
                        OrderSide::Sell,
                        Quantity(position as u64),
                    ));
                }
                None
            }

            ReactiveStrategyType::TakeProfit { target_pct } if self.is_long() => {
                let threshold = Price((self.cost_basis.0 as f64 * (1.0 + target_pct)) as i64);
                if current_price.0 >= threshold.0 {
                    let position = self.current_position();
                    return Some(self.market_order(
                        symbol,
                        OrderSide::Sell,
                        Quantity(position as u64),
                    ));
                }
                None
            }

            ReactiveStrategyType::ThresholdSeller {
                sell_price,
                size_fraction,
            } if self.is_long() => {
                if current_price.0 >= sell_price.0 {
                    let position = self.current_position() as u64;
                    let size = (position as f64 * size_fraction) as u64;
                    if size > 0 {
                        return Some(self.market_order(symbol, OrderSide::Sell, Quantity(size)));
                    }
                }
                None
            }

            // Optional modifier - NewsReactor
            ReactiveStrategyType::NewsReactor {
                min_magnitude,
                sentiment_multiplier,
            } => {
                // Check for active news events - find first actionable event
                ctx.active_events()
                    .iter()
                    .filter(|event| event.magnitude >= *min_magnitude)
                    .filter_map(|event| {
                        let base_size = (self.max_position.0 as f64 * 0.1) as u64;
                        let size = (base_size as f64 * event.sentiment.abs() * sentiment_multiplier)
                            as u64;
                        if size == 0 {
                            return None;
                        }
                        let side = if event.sentiment > 0.0 {
                            OrderSide::Buy
                        } else {
                            OrderSide::Sell
                        };
                        // Check position guards
                        let can_trade = match side {
                            OrderSide::Buy => self.remaining_capacity().0 > 0,
                            OrderSide::Sell => self.has_position(),
                        };
                        if !can_trade {
                            return None;
                        }
                        let order_size = match side {
                            OrderSide::Buy => size.min(self.remaining_capacity().0),
                            OrderSide::Sell => size.min(self.current_position() as u64),
                        };
                        Some(self.market_order(symbol, side, Quantity(order_size)))
                    })
                    .next()
            }

            // Strategy didn't match conditions or position guard failed
            _ => None,
        }
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for constructing ReactiveAgent instances.
#[derive(Debug)]
pub struct ReactiveAgentBuilder {
    id: AgentId,
    portfolio: ReactivePortfolio,
    strategies: Vec<ReactiveStrategyType>,
    max_position: Quantity,
    initial_cash: Cash,
    enabled: bool,
}

impl ReactiveAgentBuilder {
    /// Create a new builder.
    pub fn new(id: AgentId, portfolio: ReactivePortfolio) -> Self {
        Self {
            id,
            portfolio,
            strategies: Vec::new(),
            max_position: Quantity(100),
            initial_cash: Cash::from_float(100_000.0),
            enabled: true,
        }
    }

    /// Add a strategy.
    pub fn strategy(mut self, strategy: ReactiveStrategyType) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Set maximum position size.
    pub fn max_position(mut self, max: Quantity) -> Self {
        self.max_position = max;
        self
    }

    /// Set initial cash.
    pub fn initial_cash(mut self, cash: Cash) -> Self {
        self.initial_cash = cash;
        self
    }

    /// Set enabled state.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Build the agent.
    ///
    /// # Panics
    ///
    /// Panics if strategies don't include both entry and exit capability.
    pub fn build(self) -> ReactiveAgent {
        let mut agent = ReactiveAgent::new(
            self.id,
            self.portfolio,
            self.strategies,
            self.max_position,
            self.initial_cash,
        );
        agent.enabled = self.enabled;
        agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agent() -> ReactiveAgent {
        ReactiveAgent::new(
            AgentId(1),
            "ACME".into(),
            vec![
                ReactiveStrategyType::ThresholdBuyer {
                    buy_price: Price(950000), // Buy at $95
                    size_fraction: 0.5,
                },
                ReactiveStrategyType::StopLoss { stop_pct: 0.03 },
            ],
            Quantity(100),
            Cash::from_float(100_000.0),
        )
    }

    #[test]
    fn test_agent_creation() {
        let agent = test_agent();
        assert_eq!(agent.id(), AgentId(1));
        assert!(agent.is_flat());
        assert_eq!(agent.remaining_capacity(), Quantity(100));
        assert_eq!(agent.state.cash(), Cash::from_float(100_000.0));
    }

    #[test]
    fn test_position_tracking() {
        let mut agent = test_agent();

        // Buy 50 shares at price 10000 (i.e., $1.00 in fixed-point)
        agent.process_fill(OrderSide::Buy, Quantity(50), Price(10000), 0);
        assert!(agent.is_long());
        assert_eq!(agent.current_position(), 50);
        assert_eq!(agent.cost_basis(), Price(10000));
        assert_eq!(agent.remaining_capacity(), Quantity(50));

        // Sell 50 shares
        agent.process_fill(OrderSide::Sell, Quantity(50), Price(10500), 10);
        assert!(agent.is_flat());
        assert_eq!(agent.current_position(), 0);
    }

    #[test]
    fn test_builder() {
        let agent = ReactiveAgent::builder(AgentId(2), "BETA".into())
            .strategy(ReactiveStrategyType::ThresholdBuyer {
                buy_price: Price(950000),
                size_fraction: 0.25,
            })
            .strategy(ReactiveStrategyType::TakeProfit { target_pct: 0.10 })
            .max_position(Quantity(200))
            .initial_cash(Cash::from_float(50_000.0))
            .build();

        assert_eq!(agent.id(), AgentId(2));
        assert_eq!(agent.max_position, Quantity(200));
        assert_eq!(agent.state.cash(), Cash::from_float(50_000.0));
    }

    #[test]
    #[should_panic(expected = "entry strategy")]
    fn test_missing_entry_panics() {
        ReactiveAgent::new(
            AgentId(1),
            "ACME".into(),
            vec![ReactiveStrategyType::StopLoss { stop_pct: 0.03 }],
            Quantity(100),
            Cash::from_float(100_000.0),
        );
    }

    #[test]
    #[should_panic(expected = "exit strategy")]
    fn test_missing_exit_panics() {
        ReactiveAgent::new(
            AgentId(1),
            "ACME".into(),
            vec![ReactiveStrategyType::ThresholdBuyer {
                buy_price: Price(950000),
                size_fraction: 0.5,
            }],
            Quantity(100),
            Cash::from_float(100_000.0),
        );
    }
}
