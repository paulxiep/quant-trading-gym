//! Trait definitions for simulation subsystems.
//!
//! These traits define the interfaces for modular subsystems, enabling:
//! - Clear separation of concerns
//! - Testability through mocking (if needed)
//! - Documentation of subsystem responsibilities
//!
//! # Design Note
//!
//! While traits are defined here for interface clarity, `Simulation` uses
//! concrete types for subsystems (not trait objects) for maximum performance.

pub mod agents;
mod auction;
mod fundamentals;
mod market_data;
mod risk;

pub use agents::{AgentActionWithState, AgentExecutionCoordinator, AgentSummary};
pub use auction::Auctioneer;
pub use fundamentals::FundamentalsProvider;
pub use market_data::MarketDataProvider;
pub use risk::{PositionTracker, RiskTracker};
