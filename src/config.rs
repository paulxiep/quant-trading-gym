//! Re-exports agent configuration types from the simulation crate.
//!
//! The canonical definitions live in `simulation::sim_config`.
//! This module provides backwards-compatible access from the binary.

pub use simulation::{SimConfig, SymbolSpec, Tier1AgentType};
