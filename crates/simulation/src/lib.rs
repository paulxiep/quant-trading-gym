//! Simulation crate: The event loop for the Quant Trading Gym.
//!
//! This crate provides the simulation runner that coordinates:
//! - Tick-based event loop
//! - Agent execution
//! - Order processing through the matching engine
//! - Market data distribution
//! - Hook-based observation (V3.6)
//!
//! # Architecture
//!
//! The simulation runs in discrete ticks:
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │              Simulation.step()          │
//! │                                         │
//! │  1. Build MarketData snapshot           │
//! │  2. Hook: on_tick_start                 │
//! │  3. Call agent.on_tick() for each agent │
//! │  4. Collect AgentActions (orders)       │
//! │  5. Hook: on_orders_collected           │
//! │  6. Batch auction per symbol (parallel) │
//! │  7. Hook: on_trades                     │
//! │  8. Notify agents of fills via on_fill()│
//! │  9. Hook: on_tick_end                   │
//! │  10. Advance tick counter               │
//! │                                         │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Hooks (V3.6)
//!
//! The simulation supports pluggable hooks for observation:
//!
//! ```ignore
//! use simulation::{Simulation, SimulationConfig};
//! use simulation::hooks::MetricsHook;
//! use std::sync::Arc;
//!
//! let mut sim = Simulation::new(SimulationConfig::default());
//! let metrics = Arc::new(MetricsHook::new());
//! sim.add_hook(metrics.clone());
//!
//! sim.run(1000);
//! println!("Avg trades/tick: {:.2}", metrics.snapshot().avg_trades_per_tick);
//! ```
//!
//! # Parallel Execution (V3.5)
//!
//! With the `parallel` feature, the simulation parallelizes:
//! - Agent `on_tick()` collection via rayon
//! - Batch auction matching per symbol (independent symbols run in parallel)
//! - Fill notification processing
//!
//! The `parallel` module provides declarative helpers that abstract over
//! `par_iter` vs `iter` based on the feature flag.
//!
//! # Example
//!
//! ```ignore
//! use simulation::{Simulation, SimulationConfig};
//! use agents::{Agent, AgentAction, MarketData};
//! use types::AgentId;
//!
//! // Create a simple agent
//! struct MyAgent { id: AgentId }
//! impl Agent for MyAgent {
//!     fn id(&self) -> AgentId { self.id }
//!     fn on_tick(&mut self, _: &MarketData) -> AgentAction { AgentAction::none() }
//! }
//!
//! // Set up and run simulation
//! let mut sim = Simulation::new(SimulationConfig::new("AAPL"));
//! sim.add_agent(Box::new(MyAgent { id: AgentId(1) }));
//! let trades = sim.run(1000);
//! println!("Executed {} trades over 1000 ticks", trades.len());
//! ```

pub mod agent_factory;
pub mod config;
mod hooks;
mod metrics;
mod runner;
pub mod sim_config;
pub mod subsystems;
pub mod traits;

pub use config::{ParallelizationConfig, SimulationConfig};
pub use runner::{Simulation, SimulationStats};

// Re-export hook types
pub use hooks::{
    BookSnapshot, EnrichedData, HookContext, HookRunner, MarketSnapshot, NewsEventSnapshot,
    NoOpHook, SimulationHook,
};
pub use metrics::{MetricsHook, MetricsSnapshot};

// Re-export from traits
pub use traits::AgentSummary;

// Re-export agent config types (used by binary and gym)
pub use agent_factory::{MlModels, SpawnResult};
pub use sim_config::{SimConfig, SymbolSpec, Tier1AgentType};
