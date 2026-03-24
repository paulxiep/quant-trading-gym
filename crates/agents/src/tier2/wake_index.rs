//! Wake condition index for efficient Tier 2 agent triggering.
//!
//! The `WakeConditionIndex` provides O(log n) lookup for price-based conditions
//! using BTreeMap, and O(1) lookup for time-based conditions.
//!
//! # Design Rationale
//!
//! With 10K agents, linear scanning for triggered conditions would be expensive.
//! Instead, we maintain sorted indices:
//! - Price thresholds in a BTreeMap keyed by (symbol, price)
//! - Time triggers in a BTreeMap keyed by tick
//! - News subscriptions in a HashMap keyed by symbol
//!
//! # Borrow-Checker Safety
//!
//! The index cannot be mutated while iterating over triggered agents.
//! Use the two-phase pattern:
//! 1. Collect triggered agent IDs
//! 2. Process agents and collect `ConditionUpdate`s
//! 3. Apply updates after iteration complete

use crate::tiers::{ConditionUpdate, CrossDirection, OrderedPrice, WakeCondition};
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap};
use types::{AgentId, Price, Symbol, Tick};

/// Index for efficient wake condition lookups.
///
/// Maintains sorted structures for O(log n) price threshold checks
/// and O(1) time-based trigger checks.
#[derive(Debug, Default)]
pub struct WakeConditionIndex {
    /// Price thresholds indexed by (symbol, direction, price).
    /// BTreeMap provides ordered access for range queries.
    ///
    /// Key: (symbol, direction, price)
    /// Value: Set of agent IDs watching this threshold
    price_above: BTreeMap<(Symbol, OrderedPrice), SmallVec<[AgentId; 4]>>,
    price_below: BTreeMap<(Symbol, OrderedPrice), SmallVec<[AgentId; 4]>>,

    /// Time-exact triggers indexed by wake tick.
    time_exact: BTreeMap<Tick, SmallVec<[AgentId; 8]>>,

    /// Time-interval triggers (checked every tick).
    /// Value: (agent_id, next_wake, interval)
    time_intervals: Vec<(AgentId, Tick, u64)>,

    /// News subscriptions by symbol.
    news_subscriptions: HashMap<Symbol, SmallVec<[AgentId; 8]>>,

    /// Reverse index: agent -> conditions (for removal).
    agent_conditions: HashMap<AgentId, SmallVec<[WakeCondition; 4]>>,
}

impl WakeConditionIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a wake condition for an agent.
    pub fn register(&mut self, agent_id: AgentId, condition: WakeCondition) {
        // Add to reverse index
        self.agent_conditions
            .entry(agent_id)
            .or_default()
            .push(condition.clone());

        // Add to appropriate forward index
        match &condition {
            WakeCondition::PriceCross {
                symbol,
                threshold,
                direction,
            } => {
                let key = (symbol.clone(), OrderedPrice::from_price(*threshold));
                match direction {
                    CrossDirection::Above => {
                        self.price_above.entry(key).or_default().push(agent_id);
                    }
                    CrossDirection::Below => {
                        self.price_below.entry(key).or_default().push(agent_id);
                    }
                }
            }
            WakeCondition::TimeExact { wake_tick } => {
                self.time_exact
                    .entry(*wake_tick)
                    .or_default()
                    .push(agent_id);
            }
            WakeCondition::TimeInterval {
                next_wake,
                interval,
            } => {
                self.time_intervals.push((agent_id, *next_wake, *interval));
            }
            WakeCondition::NewsEvent { symbols } => {
                for symbol in symbols {
                    self.news_subscriptions
                        .entry(symbol.clone())
                        .or_default()
                        .push(agent_id);
                }
            }
        }
    }

    /// Unregister a specific condition for an agent.
    pub fn unregister(&mut self, agent_id: AgentId, condition: &WakeCondition) {
        // Remove from reverse index
        if let Some(conditions) = self.agent_conditions.get_mut(&agent_id) {
            conditions.retain(|c| c != condition);
        }

        // Remove from forward index
        match condition {
            WakeCondition::PriceCross {
                symbol,
                threshold,
                direction,
            } => {
                let key = (symbol.clone(), OrderedPrice::from_price(*threshold));
                match direction {
                    CrossDirection::Above => {
                        if let Some(agents) = self.price_above.get_mut(&key) {
                            agents.retain(|id| *id != agent_id);
                            if agents.is_empty() {
                                self.price_above.remove(&key);
                            }
                        }
                    }
                    CrossDirection::Below => {
                        if let Some(agents) = self.price_below.get_mut(&key) {
                            agents.retain(|id| *id != agent_id);
                            if agents.is_empty() {
                                self.price_below.remove(&key);
                            }
                        }
                    }
                }
            }
            WakeCondition::TimeExact { wake_tick } => {
                if let Some(agents) = self.time_exact.get_mut(wake_tick) {
                    agents.retain(|id| *id != agent_id);
                    if agents.is_empty() {
                        self.time_exact.remove(wake_tick);
                    }
                }
            }
            WakeCondition::TimeInterval { .. } => {
                self.time_intervals.retain(|(id, _, _)| *id != agent_id);
            }
            WakeCondition::NewsEvent { symbols } => {
                for symbol in symbols {
                    if let Some(agents) = self.news_subscriptions.get_mut(symbol) {
                        agents.retain(|id| *id != agent_id);
                        if agents.is_empty() {
                            self.news_subscriptions.remove(symbol);
                        }
                    }
                }
            }
        }
    }

    /// Unregister all conditions for an agent.
    pub fn unregister_all(&mut self, agent_id: AgentId) {
        if let Some(conditions) = self.agent_conditions.remove(&agent_id) {
            for condition in conditions {
                self.unregister(agent_id, &condition);
            }
        }
    }

    /// Get agents triggered by current prices.
    ///
    /// Triggers when current price satisfies the condition:
    /// - PriceCross::Below at threshold → triggers if price ≤ threshold
    /// - PriceCross::Above at threshold → triggers if price ≥ threshold
    pub fn triggered_by_price(
        &self,
        current_prices: &[(Symbol, Price)],
    ) -> Vec<(AgentId, WakeCondition)> {
        current_prices
            .iter()
            .flat_map(|(symbol, current_price)| {
                let current = OrderedPrice::from_price(*current_price);
                let symbol = symbol.clone();

                // "Below" conditions: trigger if price ≤ threshold (all thresholds >= current)
                let symbol_below = symbol.clone();
                let below_triggers = self
                    .price_below
                    .range((symbol.clone(), current)..)
                    .take_while(move |((s, _), _)| s == &symbol_below)
                    .flat_map({
                        let symbol = symbol.clone();
                        move |((_, threshold), agents)| {
                            let condition = WakeCondition::PriceCross {
                                symbol: symbol.clone(),
                                threshold: threshold.to_price(),
                                direction: CrossDirection::Below,
                            };
                            agents.iter().map(move |&id| (id, condition.clone()))
                        }
                    });

                // "Above" conditions: trigger if price ≥ threshold (all thresholds <= current)
                let symbol_above = symbol.clone();
                let above_triggers = self
                    .price_above
                    .range(..(symbol.clone(), OrderedPrice(current.0 + 1)))
                    .rev()
                    .take_while(move |((s, _), _)| s == &symbol_above)
                    .filter(move |((_, threshold), _)| threshold.0 <= current.0)
                    .flat_map(move |((_, threshold), agents)| {
                        let condition = WakeCondition::PriceCross {
                            symbol: symbol.clone(),
                            threshold: threshold.to_price(),
                            direction: CrossDirection::Above,
                        };
                        agents.iter().map(move |&id| (id, condition.clone()))
                    });

                below_triggers.chain(above_triggers)
            })
            .collect()
    }

    /// Get agents triggered by exact time.
    pub fn triggered_by_time_exact(&self, tick: Tick) -> Vec<(AgentId, WakeCondition)> {
        self.time_exact
            .get(&tick)
            .map(|agents| {
                agents
                    .iter()
                    .map(|&id| (id, WakeCondition::TimeExact { wake_tick: tick }))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get agents triggered by time intervals and return updated conditions.
    pub fn triggered_by_time_interval(&self, tick: Tick) -> Vec<(AgentId, WakeCondition, Tick)> {
        self.time_intervals
            .iter()
            .filter(|(_, next_wake, _)| *next_wake == tick)
            .map(|(agent_id, next_wake, interval)| {
                let new_next = next_wake + interval;
                (
                    *agent_id,
                    WakeCondition::TimeInterval {
                        next_wake: *next_wake,
                        interval: *interval,
                    },
                    new_next,
                )
            })
            .collect()
    }

    /// Get agents subscribed to news for a symbol.
    pub fn triggered_by_news(&self, symbol: &Symbol) -> Vec<AgentId> {
        self.news_subscriptions
            .get(symbol)
            .map(|agents| agents.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Collect all triggered agents for this tick.
    ///
    /// Returns a deduplicated set of agent IDs along with their trigger reasons.
    pub fn collect_triggered(
        &self,
        tick: Tick,
        current_prices: &[(Symbol, Price)],
        news_symbols: &[Symbol],
    ) -> HashMap<AgentId, SmallVec<[WakeCondition; 2]>> {
        let mut result: HashMap<AgentId, SmallVec<[WakeCondition; 2]>> = HashMap::new();

        // Price triggers
        for (agent_id, condition) in self.triggered_by_price(current_prices) {
            result.entry(agent_id).or_default().push(condition);
        }

        // Time exact triggers
        for (agent_id, condition) in self.triggered_by_time_exact(tick) {
            result.entry(agent_id).or_default().push(condition);
        }

        // Time interval triggers
        for (agent_id, condition, _) in self.triggered_by_time_interval(tick) {
            result.entry(agent_id).or_default().push(condition);
        }

        // News triggers
        for symbol in news_symbols {
            for agent_id in self.triggered_by_news(symbol) {
                result
                    .entry(agent_id)
                    .or_default()
                    .push(WakeCondition::NewsEvent {
                        symbols: smallvec::smallvec![symbol.clone()],
                    });
            }
        }

        result
    }

    /// Apply deferred condition updates.
    ///
    /// Call this after processing all triggered agents to avoid
    /// borrow-checker issues during iteration.
    pub fn apply_updates(&mut self, updates: Vec<ConditionUpdate>) {
        for update in updates {
            for condition in update.remove {
                self.unregister(update.agent_id, &condition);
            }
            for condition in update.add {
                self.register(update.agent_id, condition);
            }
        }
    }

    /// Update interval triggers to their next wake time.
    pub fn advance_intervals(&mut self, tick: Tick) {
        for (_, next_wake, interval) in &mut self.time_intervals {
            if *next_wake == tick {
                *next_wake += *interval;
            }
        }
    }

    /// Get count of registered conditions by type.
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            price_above_count: self.price_above.values().map(|v| v.len()).sum(),
            price_below_count: self.price_below.values().map(|v| v.len()).sum(),
            price_above_keys: self.price_above.len(),
            price_below_keys: self.price_below.len(),
            time_exact_count: self.time_exact.values().map(|v| v.len()).sum(),
            time_interval_count: self.time_intervals.len(),
            news_subscription_count: self.news_subscriptions.values().map(|v| v.len()).sum(),
            agent_count: self.agent_conditions.len(),
        }
    }

    /// Clean up expired time-exact conditions.
    pub fn cleanup_expired(&mut self, current_tick: Tick) {
        // Retain only entries for future ticks
        self.time_exact.retain(|&tick, _| tick >= current_tick);
    }
}

/// Statistics about the wake condition index.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub price_above_count: usize,
    pub price_below_count: usize,
    pub price_above_keys: usize,
    pub price_below_keys: usize,
    pub time_exact_count: usize,
    pub time_interval_count: usize,
    pub news_subscription_count: usize,
    pub agent_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_unregister() {
        let mut index = WakeConditionIndex::new();
        let agent = AgentId(1);
        let condition = WakeCondition::price_above("ACME".to_string(), Price(10000));

        index.register(agent, condition.clone());

        let stats = index.stats();
        assert_eq!(stats.price_above_count, 1);
        assert_eq!(stats.agent_count, 1);

        index.unregister(agent, &condition);

        let stats = index.stats();
        assert_eq!(stats.price_above_count, 0);
    }

    #[test]
    fn test_time_exact_trigger() {
        let mut index = WakeConditionIndex::new();
        let agent = AgentId(1);

        index.register(agent, WakeCondition::at_tick(100));

        let triggered = index.triggered_by_time_exact(99);
        assert!(triggered.is_empty());

        let triggered = index.triggered_by_time_exact(100);
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0].0, agent);
    }

    #[test]
    fn test_time_interval_trigger() {
        let mut index = WakeConditionIndex::new();
        let agent = AgentId(1);

        index.register(agent, WakeCondition::every_n_ticks(10, 0));

        // Should trigger at tick 0
        let triggered = index.triggered_by_time_interval(0);
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0].2, 10); // Next wake at tick 10

        // Advance intervals
        index.advance_intervals(0);

        // Should trigger at tick 10
        let triggered = index.triggered_by_time_interval(10);
        assert_eq!(triggered.len(), 1);
    }

    #[test]
    fn test_news_trigger() {
        let mut index = WakeConditionIndex::new();
        let agent1 = AgentId(1);
        let agent2 = AgentId(2);

        index.register(agent1, WakeCondition::on_news(vec!["ACME".to_string()]));
        index.register(
            agent2,
            WakeCondition::on_news(vec!["ACME".to_string(), "BETA".to_string()]),
        );

        let triggered = index.triggered_by_news(&"ACME".to_string());
        assert_eq!(triggered.len(), 2);

        let triggered = index.triggered_by_news(&"BETA".to_string());
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0], agent2);
    }

    #[test]
    fn test_apply_updates() {
        let mut index = WakeConditionIndex::new();
        let agent = AgentId(1);

        index.register(agent, WakeCondition::at_tick(100));

        let update = ConditionUpdate::new(agent)
            .remove_condition(WakeCondition::at_tick(100))
            .add_condition(WakeCondition::at_tick(200));

        index.apply_updates(vec![update]);

        let triggered_100 = index.triggered_by_time_exact(100);
        assert!(triggered_100.is_empty());

        let triggered_200 = index.triggered_by_time_exact(200);
        assert_eq!(triggered_200.len(), 1);
    }

    #[test]
    fn test_price_below_trigger() {
        use types::Price;

        let mut index = WakeConditionIndex::new();
        let agent1 = AgentId(1); // threshold $60
        let agent2 = AgentId(2); // threshold $80
        let agent3 = AgentId(3); // threshold $65
        let symbol: Symbol = "ACME".into();

        // Register "Below" conditions (ThresholdBuyer style)
        // These should trigger when price DROPS TO OR BELOW the threshold
        index.register(
            agent1,
            WakeCondition::PriceCross {
                symbol: symbol.clone(),
                threshold: Price::from_float(60.0), // Buy at $60
                direction: CrossDirection::Below,
            },
        );
        index.register(
            agent2,
            WakeCondition::PriceCross {
                symbol: symbol.clone(),
                threshold: Price::from_float(80.0), // Buy at $80
                direction: CrossDirection::Below,
            },
        );
        index.register(
            agent3,
            WakeCondition::PriceCross {
                symbol: symbol.clone(),
                threshold: Price::from_float(65.0), // Buy at $65
                direction: CrossDirection::Below,
            },
        );

        // Price at $75 - should NOT trigger agent1 ($60) or agent3 ($65)
        // Should trigger agent2 ($80) because price ($75) is below threshold ($80)
        let current_prices = vec![(symbol.clone(), Price::from_float(75.0))];
        let triggered = index.triggered_by_price(&current_prices);

        println!(
            "Price $75, triggered agents: {:?}",
            triggered.iter().map(|(id, _)| id.0).collect::<Vec<_>>()
        );

        assert_eq!(
            triggered.len(),
            1,
            "Only agent2 ($80 threshold) should trigger at price $75"
        );
        assert_eq!(triggered[0].0, agent2);

        // Price at $55 - should trigger ALL agents (price below all thresholds)
        let current_prices = vec![(symbol.clone(), Price::from_float(55.0))];
        let triggered = index.triggered_by_price(&current_prices);

        println!(
            "Price $55, triggered agents: {:?}",
            triggered.iter().map(|(id, _)| id.0).collect::<Vec<_>>()
        );

        assert_eq!(triggered.len(), 3, "All agents should trigger at price $55");

        // Price at $90 - should trigger NO agents (price above all thresholds)
        let current_prices = vec![(symbol.clone(), Price::from_float(90.0))];
        let triggered = index.triggered_by_price(&current_prices);

        println!(
            "Price $90, triggered agents: {:?}",
            triggered.iter().map(|(id, _)| id.0).collect::<Vec<_>>()
        );

        assert!(
            triggered.is_empty(),
            "No agents should trigger at price $90"
        );
    }

    #[test]
    fn test_price_above_trigger() {
        use types::Price;

        let mut index = WakeConditionIndex::new();
        let agent1 = AgentId(1); // threshold $110
        let agent2 = AgentId(2); // threshold $90
        let agent3 = AgentId(3); // threshold $120
        let symbol: Symbol = "ACME".into();

        // Register "Above" conditions (ThresholdSeller/TakeProfit style)
        // These should trigger when price RISES TO OR ABOVE the threshold
        index.register(
            agent1,
            WakeCondition::PriceCross {
                symbol: symbol.clone(),
                threshold: Price::from_float(110.0), // Sell at $110
                direction: CrossDirection::Above,
            },
        );
        index.register(
            agent2,
            WakeCondition::PriceCross {
                symbol: symbol.clone(),
                threshold: Price::from_float(90.0), // Sell at $90
                direction: CrossDirection::Above,
            },
        );
        index.register(
            agent3,
            WakeCondition::PriceCross {
                symbol: symbol.clone(),
                threshold: Price::from_float(120.0), // Sell at $120
                direction: CrossDirection::Above,
            },
        );

        // Price at $100 - should trigger agent2 ($90) only
        let current_prices = vec![(symbol.clone(), Price::from_float(100.0))];
        let triggered = index.triggered_by_price(&current_prices);

        println!(
            "Price $100, triggered agents: {:?}",
            triggered.iter().map(|(id, _)| id.0).collect::<Vec<_>>()
        );

        assert_eq!(
            triggered.len(),
            1,
            "Only agent2 ($90 threshold) should trigger at price $100"
        );
        assert_eq!(triggered[0].0, agent2);

        // Price at $75 - should trigger NO agents
        let current_prices = vec![(symbol.clone(), Price::from_float(75.0))];
        let triggered = index.triggered_by_price(&current_prices);

        println!(
            "Price $75, triggered agents: {:?}",
            triggered.iter().map(|(id, _)| id.0).collect::<Vec<_>>()
        );

        assert!(
            triggered.is_empty(),
            "No agents should trigger at price $75"
        );

        // Price at $125 - should trigger ALL agents
        let current_prices = vec![(symbol.clone(), Price::from_float(125.0))];
        let triggered = index.triggered_by_price(&current_prices);

        println!(
            "Price $125, triggered agents: {:?}",
            triggered.iter().map(|(id, _)| id.0).collect::<Vec<_>>()
        );

        assert_eq!(
            triggered.len(),
            3,
            "All agents should trigger at price $125"
        );
    }
}
