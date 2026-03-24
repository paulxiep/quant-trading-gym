//! Agent Factory: spawns all simulation agents from configuration.
//!
//! Extracted from main.rs so the gym crate can reuse agent spawning logic.
//! Split into sub-modules by agent tier for maintainability.

mod background_pool;
mod ml_agents;
mod tier1;
mod tier2;

pub use ml_agents::MlModels;

use crate::Simulation;
use crate::sim_config::{SimConfig, SymbolSpec};

/// Result of spawning all agents.
pub struct SpawnResult {
    /// Number of Tier 1 agents spawned.
    pub tier1_count: usize,
    /// Number of Tier 2 agents spawned (reactive + sector rotators).
    pub tier2_count: usize,
}

/// Spawn all agents into the simulation according to configuration.
///
/// This is the single entry point for agent creation. It:
/// 1. Spawns Tier 1 agents (market makers, noise traders, quant, pairs, ML)
/// 2. Spawns Tier 2 agents (reactive, sector rotators)
/// 3. Sets up Tier 3 background pool
///
/// ML models must be pre-loaded — pass `MlModels::default()` if none available.
pub fn spawn_all(
    sim: &mut Simulation,
    config: &SimConfig,
    models: &MlModels,
    rng: &mut rand::prelude::ThreadRng,
) -> SpawnResult {
    let symbols: Vec<_> = config.get_symbols().to_vec();

    // Phase 1: Spawn Tier 1 agents
    let mut next_id = spawn_all_tier1(sim, config, &symbols, models, rng);
    let tier1_count = (next_id - 1) as usize; // IDs start at 1

    // Phase 2: Spawn Tier 2 agents (reactive + sector rotators)
    let t2_start = next_id;
    tier2::spawn_tier2_agents(sim, &mut next_id, config, &symbols, rng);
    tier2::spawn_sector_rotators(sim, &mut next_id, config, &symbols);
    let tier2_count = (next_id - t2_start) as usize;

    // Phase 3: Setup background pool
    background_pool::setup_background_pool(sim, config, &symbols);

    SpawnResult {
        tier1_count,
        tier2_count,
    }
}

/// Spawn all Tier 1 agents. Returns next available agent ID.
fn spawn_all_tier1(
    sim: &mut Simulation,
    config: &SimConfig,
    symbols: &[SymbolSpec],
    models: &MlModels,
    rng: &mut rand::prelude::ThreadRng,
) -> u64 {
    let mut next_id = 1u64;

    let (mm_agents, id) = tier1::spawn_market_makers(config, symbols, next_id, rng);
    next_id = id;

    let (nt_agents, id) = tier1::spawn_noise_traders(config, symbols, next_id, rng);
    next_id = id;

    let (quant_agents, id) = tier1::spawn_quant_agents(config, symbols, next_id, rng);
    next_id = id;

    let (pairs_agents, id) = tier1::spawn_pairs_traders(config, symbols, next_id);
    next_id = id;

    let (tree_agents, id) = ml_agents::spawn_tree_agents(sim, config, symbols, next_id, models);
    next_id = id;

    let (random_agents, id) = tier1::spawn_random_tier1_agents(config, symbols, next_id, rng);
    next_id = id;

    for agent in mm_agents
        .into_iter()
        .chain(nt_agents)
        .chain(quant_agents)
        .chain(pairs_agents)
        .chain(tree_agents)
        .chain(random_agents)
    {
        sim.add_agent(agent);
    }

    next_id
}
