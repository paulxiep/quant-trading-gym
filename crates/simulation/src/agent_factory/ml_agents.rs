//! ML agent spawning: creates MlAgent instances for all registered models.

use std::sync::Arc;

use agents::{Agent, MlAgent, MlAgentConfig, MlModel};
use types::{AgentId, Price};

use crate::Simulation;
use crate::sim_config::{SimConfig, SymbolSpec};

/// Pre-loaded ML models, ready for registration and agent creation.
///
/// Models are stored as `(model_type, Arc<dyn MlModel>)` pairs.
/// The binary (CLI) populates this from JSON files on disk.
/// The gym crate can populate it from episode config or bundled models.
#[derive(Default)]
pub struct MlModels {
    models: Vec<(String, Arc<dyn MlModel>)>,
}

impl MlModels {
    /// Create an empty model collection.
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Add a model with its type tag (e.g., "decision_tree", "random_forest").
    pub fn push(&mut self, model_type: &str, model: impl MlModel + 'static) {
        self.models.push((model_type.to_string(), Arc::new(model)));
    }

    /// True if any models are loaded.
    pub fn has_models(&self) -> bool {
        !self.models.is_empty()
    }

    /// Get all models.
    pub fn all(&self) -> &[(String, Arc<dyn MlModel>)] {
        &self.models
    }

    /// Get models matching a specific type.
    pub fn of_type(&self, model_type: &str) -> Vec<&Arc<dyn MlModel>> {
        self.models
            .iter()
            .filter(|(t, _)| t == model_type)
            .map(|(_, m)| m)
            .collect()
    }

    /// Get the number of loaded models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Look up a model by its `MlModel::name()` value.
    pub fn by_name(&self, name: &str) -> Option<&Arc<dyn MlModel>> {
        self.models
            .iter()
            .find(|(_, m)| m.name() == name)
            .map(|(_, m)| m)
    }

    /// All model names (for error messages).
    pub fn model_names(&self) -> Vec<&str> {
        self.models.iter().map(|(_, m)| m.name()).collect()
    }
}

/// Spawn MlAgent instances for a set of model names, distributed round-robin.
fn spawn_ml_agents(
    model_names: &[String],
    count: usize,
    start_id: u64,
    agent_config: MlAgentConfig,
) -> Vec<Box<dyn Agent>> {
    if model_names.is_empty() || count == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|i| {
            let name = model_names[i % model_names.len()].clone();
            Box::new(MlAgent::new(
                AgentId(start_id + i as u64),
                name,
                agent_config.clone(),
            )) as Box<dyn Agent>
        })
        .collect()
}

/// Spawn ML agents and register models with the simulation.
///
/// Models must be pre-loaded (see `MlModels`). This function:
/// 1. Registers all models with the simulation for centralized prediction caching
/// 2. Creates MlAgent instances per model type, distributed round-robin
pub(crate) fn spawn_tree_agents(
    sim: &mut Simulation,
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    models: &MlModels,
) -> (Vec<Box<dyn Agent>>, u64) {
    // Register all models for centralized prediction caching
    for (_, model) in models.all() {
        sim.register_ml_model_arc(model.clone());
    }
    if sim.has_ml_models() {
        eprintln!(
            "  Registered {} unique ML models for centralized caching",
            sim.ml_model_count()
        );
    }

    let agent_config = MlAgentConfig {
        symbols: symbols.iter().map(|s| s.symbol.clone()).collect(),
        buy_threshold: config.tree_agent_buy_threshold,
        sell_threshold: config.tree_agent_sell_threshold,
        order_size: config.tree_agent_order_size,
        max_long_position: config.tree_agent_max_long_position,
        max_short_position: config.tree_agent_max_short_position,
        initial_cash: config.tree_agent_initial_cash,
        initial_price: symbols
            .first()
            .map(|s| s.initial_price)
            .unwrap_or(Price::from_float(100.0)),
    };

    let mut all_agents: Vec<Box<dyn Agent>> = Vec::new();
    let mut next_id = start_id;

    // Spawn agents per model type based on config counts
    for (model_type, count) in &config.ml_agent_counts {
        let count = *count;
        if count == 0 {
            continue;
        }

        let model_names: Vec<String> = models
            .of_type(model_type)
            .iter()
            .map(|m| m.name().to_string())
            .collect();

        if model_names.is_empty() {
            if count > 0 {
                eprintln!("  Warning: No '{}' models found in models/", model_type);
            }
            continue;
        }

        let agents = spawn_ml_agents(&model_names, count, next_id, agent_config.clone());
        next_id += agents.len() as u64;
        all_agents.extend(agents);
    }

    (all_agents, next_id)
}
