//! Tier 2: Reactive Agents - Lightweight event-driven agents.
//!
//! Tier 2 agents are optimized for scale (1K-10K agents) by using a
//! wake-on-condition model instead of polling every tick.
//!
//! # Design Philosophy (Declarative, Modular, SoC)
//!
//! - **Declarative**: Agents declare wake conditions, not how to check them
//! - **Modular**: Strategies are composable enum variants
//! - **Separation of Concerns**: Wake index tracks conditions; agents define behavior
//!
//! # Key Components
//!
//! - [`ReactiveAgent`] - Main agent struct with strategies and position tracking
//! - [`ReactiveStrategyType`] - Enum of 9 strategy variants (3 entry, 5 exit, 1 bidirectional)
//! - [`LightweightContext`] - Minimal context passed on wake (vs full StrategyContext)
//! - [`WakeConditionIndex`] - O(log n) lookup for price/time triggers
//! - [`ReactivePortfolio`] - Portfolio scope (SingleSymbol for V3.2)
//!
//! # V3.3 Multi-Symbol Strategies
//!
//! - [`SectorRotator`] - Sentiment-driven multi-symbol portfolio rotation
//!
//! # Memory Budget
//!
//! Target: ~200 bytes per agent (vs ~3KB for Tier 1)
//! - SmallVec<[ReactiveStrategyType; 4]> for inline strategy storage
//! - No indicator state (uses pre-computed or agent-tracked values)
//! - Compact position tracking
//!
//! # Borrow-Checker Safety
//!
//! The two-phase tick pattern prevents borrow conflicts:
//! 1. **Collection phase**: Iterate triggered agents, collect actions
//! 2. **Mutation phase**: Apply condition updates after iteration complete
//!
//! See [`crate::tiers::ConditionUpdate`] for the deferred update pattern.
//!
//! # Example
//!
//! ```ignore
//! use agents::tier2::{ReactiveAgent, ReactiveStrategyType, ReactivePortfolio};
//! use agents::tiers::WakeCondition;
//! use types::{AgentId, Price};
//!
//! // Create an agent that buys dips and uses trailing stops
//! let agent = ReactiveAgent::new(
//!     AgentId(1001),
//!     ReactivePortfolio::SingleSymbol("ACME".into()),
//!     vec![
//!         ReactiveStrategyType::DipBuyer { threshold_pct: 0.05 },
//!         ReactiveStrategyType::TrailingStop { trail_pct: 0.03 },
//!     ],
//! );
//! ```

mod agent;
mod context;
mod portfolio;
pub mod sector_rotator;
mod strategies;
mod wake_index;

pub use agent::ReactiveAgent;
pub use context::LightweightContext;
pub use portfolio::ReactivePortfolio;
pub use sector_rotator::{SectorRotator, SectorRotatorConfig};
pub use strategies::ReactiveStrategyType;
pub use wake_index::{IndexStats, WakeConditionIndex};
