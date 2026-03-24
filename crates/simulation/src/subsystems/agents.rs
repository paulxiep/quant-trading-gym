//! Agent orchestrator subsystem.
//!
//! Manages agent execution across tiers (T1, T2, T3).

use std::collections::HashMap;

use crate::traits::{AgentExecutionCoordinator, AgentSummary};
use agents::{
    Agent, BACKGROUND_POOL_ID, BackgroundAgentPool, IndexStats, PoolContext, StrategyContext,
    WakeCondition, WakeConditionIndex,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use types::{AgentId, Cash, Order, Price, Symbol, Tick, Trade};

/// Orchestrates agent execution across tiers.
///
/// Owns agents, tier indices, wake condition index, and background pool.
pub struct AgentOrchestrator {
    /// Trading agents wrapped in Mutex for parallel access.
    agents: Vec<Mutex<Box<dyn Agent>>>,

    /// Indices of Tier 1 agents (called every tick).
    t1_indices: Vec<usize>,

    /// Indices of Tier 2 agents (called only when triggered).
    t2_indices: Vec<usize>,

    /// Map from AgentId to index in agents vec.
    agent_id_to_index: HashMap<AgentId, usize>,

    /// Wake condition index for Tier 2 reactive agents.
    wake_index: WakeConditionIndex,

    /// Optional Tier 3 background pool.
    background_pool: Option<BackgroundAgentPool>,
}

impl AgentOrchestrator {
    /// Create a new agent orchestrator.
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            t1_indices: Vec::new(),
            t2_indices: Vec::new(),
            agent_id_to_index: HashMap::new(),
            wake_index: WakeConditionIndex::new(),
            background_pool: None,
        }
    }

    /// Set the Tier 3 background pool.
    pub fn set_background_pool(&mut self, pool: BackgroundAgentPool) {
        self.background_pool = Some(pool);
    }

    /// Get a reference to the background pool.
    pub fn background_pool(&self) -> Option<&BackgroundAgentPool> {
        self.background_pool.as_ref()
    }

    /// Get wake condition index statistics.
    pub fn wake_index_stats(&self) -> IndexStats {
        self.wake_index.stats()
    }

    /// Get tier counts: (t1_count, t2_count, t3_count).
    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let t3 = self
            .background_pool
            .as_ref()
            .map(|p| p.config().pool_size)
            .unwrap_or(0);
        (self.t1_indices.len(), self.t2_indices.len(), t3)
    }

    /// Cleanup expired time-based wake conditions.
    pub fn cleanup_expired_conditions(&mut self, tick: Tick) {
        self.wake_index.cleanup_expired(tick);
    }

    /// Generate Tier 3 background pool orders.
    pub fn generate_background_pool_orders(&mut self, ctx: &PoolContext<'_>) -> Vec<Order> {
        let Some(pool) = self.background_pool.as_mut() else {
            return Vec::new();
        };
        pool.generate(ctx)
    }

    /// Update T3 pool accounting from trades.
    pub fn update_background_pool_accounting(&mut self, trades: &[Trade]) {
        let Some(pool) = self.background_pool.as_mut() else {
            return;
        };

        for trade in trades {
            if trade.buyer_id == BACKGROUND_POOL_ID {
                pool.accounting_mut().record_trade_as_buyer(
                    &trade.symbol,
                    trade.price,
                    trade.quantity,
                );
            } else if trade.seller_id == BACKGROUND_POOL_ID {
                pool.accounting_mut().record_trade_as_seller(
                    &trade.symbol,
                    trade.price,
                    trade.quantity,
                );
            }
        }
    }

    /// Collect current equities from all agents.
    pub fn collect_equities(
        &self,
        prices: &HashMap<Symbol, Price>,
        force_sequential: bool,
    ) -> Vec<(AgentId, f64)> {
        parallel::map_mutex_slice_ref(
            &self.agents,
            |agent| (agent.id(), agent.equity(prices).to_float()),
            force_sequential,
        )
    }

    /// Get a clone of an agent's state by ID (for gym observation extraction).
    ///
    /// Returns `None` if the agent ID is not found.
    /// Acquires the agent's mutex lock briefly to clone the state.
    pub fn agent_state(&self, id: AgentId) -> Option<agents::AgentState> {
        let &idx = self.agent_id_to_index.get(&id)?;
        let agent = self.agents[idx].lock();
        Some(agent.state().clone())
    }

    /// Apply wake condition updates from fill notifications.
    pub fn apply_wake_updates(&mut self, updates: Vec<agents::ConditionUpdate>) {
        self.wake_index.apply_updates(updates);
    }
}

impl Default for AgentOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentExecutionCoordinator for AgentOrchestrator {
    fn add_agent(&mut self, agent: Box<dyn Agent>, initial_tick: Tick) {
        let agent_id = agent.id();
        let is_reactive = agent.is_reactive();
        let index = self.agents.len();

        if is_reactive {
            self.t2_indices.push(index);
            for condition in agent.initial_wake_conditions(initial_tick) {
                self.wake_index.register(agent_id, condition);
            }
        } else {
            self.t1_indices.push(index);
        }

        self.agent_id_to_index.insert(agent_id, index);
        self.agents.push(Mutex::new(agent));
    }

    fn agent_count(&self) -> usize {
        self.agents.len()
    }

    fn compute_agents_to_call(
        &mut self,
        tick: Tick,
        prices: &[(Symbol, Price)],
        news_symbols: &[Symbol],
    ) -> (Vec<usize>, HashMap<AgentId, SmallVec<[WakeCondition; 2]>>) {
        use rand::seq::SliceRandom;

        // Collect triggered T2 agents from wake index
        let triggered_t2 = self
            .wake_index
            .collect_triggered(tick, prices, news_symbols);

        // Remove triggered PriceCross conditions immediately
        for (agent_id, conditions) in &triggered_t2 {
            for condition in conditions {
                if matches!(condition, WakeCondition::PriceCross { .. }) {
                    self.wake_index.unregister(*agent_id, condition);
                }
            }
        }

        // Build list: T1 always, T2 only if triggered
        let mut indices_to_call: Vec<usize> = self.t1_indices.clone();
        indices_to_call.extend(
            triggered_t2
                .keys()
                .filter_map(|agent_id| self.agent_id_to_index.get(agent_id).copied()),
        );

        // Randomize order to avoid systematic bias
        indices_to_call.shuffle(&mut rand::thread_rng());

        (indices_to_call, triggered_t2)
    }

    fn collect_actions(
        &self,
        indices: &[usize],
        ctx: &StrategyContext<'_>,
        force_sequential: bool,
    ) -> Vec<(
        AgentId,
        agents::AgentAction,
        HashMap<Symbol, i64>,
        Cash,
        bool,
    )> {
        parallel::map_indices(
            indices,
            |i| {
                let mut agent = self.agents[i].lock();
                let agent_id = agent.id();
                let action = agent.on_tick(ctx);
                let positions: HashMap<Symbol, i64> = agent
                    .positions()
                    .iter()
                    .map(|(sym, entry)| (sym.clone(), entry.quantity))
                    .collect();
                let cash = agent.cash();
                let is_mm = agent.is_market_maker();
                (agent_id, action, positions, cash, is_mm)
            },
            force_sequential,
        )
    }

    fn build_position_cache(
        &self,
        force_sequential: bool,
    ) -> HashMap<AgentId, HashMap<Symbol, i64>> {
        parallel::map_mutex_slice_ref_to_hashmap(
            &self.agents,
            |agent| {
                let positions: HashMap<Symbol, i64> = agent
                    .positions()
                    .iter()
                    .map(|(sym, entry)| (sym.clone(), entry.quantity))
                    .collect();
                (agent.id(), positions)
            },
            force_sequential,
        )
    }

    fn notify_fills(&mut self, fills: &[(AgentId, Trade, i64)], force_sequential: bool) {
        let condition_updates = parallel::filter_map_slice(
            fills,
            |(agent_id, trade, pos_before)| {
                self.agent_id_to_index.get(agent_id).and_then(|&idx| {
                    let mut agent = self.agents[idx].lock();
                    agent.on_fill(trade);
                    agent
                        .is_reactive()
                        .then(|| agent.post_fill_condition_update(*pos_before))
                        .flatten()
                })
            },
            force_sequential,
        );

        self.wake_index.apply_updates(condition_updates);
    }

    fn restore_wake_conditions(
        &mut self,
        triggered: &HashMap<AgentId, SmallVec<[WakeCondition; 2]>>,
        force_sequential: bool,
    ) {
        let triggered_keys: Vec<_> = triggered.keys().copied().collect();
        let t2_conditions = parallel::filter_map_slice(
            &triggered_keys,
            |agent_id| {
                self.agent_id_to_index.get(agent_id).and_then(|&idx| {
                    let agent = self.agents[idx].lock();
                    agent
                        .is_reactive()
                        .then(|| (*agent_id, agent.current_wake_conditions().to_vec()))
                })
            },
            force_sequential,
        );

        for (agent_id, conditions) in t2_conditions {
            // Unregister ALL existing conditions to prevent accumulation
            self.wake_index.unregister_all(agent_id);
            // Register new conditions
            for condition in conditions {
                self.wake_index.register(agent_id, condition);
            }
        }
    }

    fn agent_summaries(
        &self,
        prices: &HashMap<Symbol, Price>,
        force_sequential: bool,
    ) -> Vec<AgentSummary> {
        parallel::map_mutex_slice_ref(
            &self.agents,
            |agent| {
                let positions: HashMap<Symbol, i64> = agent
                    .positions()
                    .iter()
                    .map(|(sym, entry)| (sym.clone(), entry.quantity))
                    .collect();
                let total_pnl = agent.state().total_pnl(prices);
                AgentSummary {
                    id: agent.id(),
                    name: agent.name().to_owned(),
                    positions,
                    cash: agent.cash(),
                    total_pnl,
                    is_market_maker: agent.is_market_maker(),
                    is_ml_agent: agent.is_ml_agent(),
                }
            },
            force_sequential,
        )
    }
}
