//! News and fundamentals system for the trading simulation (V2.4).
//!
//! This crate provides:
//! - **Fundamentals**: Company financials and fair value calculation (Gordon Growth Model)
//! - **Events**: Market-moving events (earnings, rate decisions, sector news)
//! - **Generator**: Configurable event generation with deterministic seeding
//! - **Sectors**: Industry classification and sector-symbol mapping
//!
//! # Architecture
//!
//! The news system generates events at the start of each tick, which are then
//! passed to agents as immutable references. This avoids borrow-check issues:
//!
//! ```text
//! Tick N:
//!   1. NewsGenerator::tick() → Vec<NewsEvent>  (mutable)
//!   2. Prune expired events                    (mutable)
//!   3. Apply permanent fundamental changes     (mutable)
//!   4. Build StrategyContext with &events      (immutable borrow starts)
//!   5. Run all agents                          (immutable borrow held)
//!   6. Drop context, process orders            (immutable borrow ends)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use news::{NewsGenerator, NewsGeneratorConfig, SymbolFundamentals};
//!
//! // Setup
//! let config = NewsGeneratorConfig::default();
//! let mut generator = NewsGenerator::new(config, 42);
//! let mut fundamentals = SymbolFundamentals::new(MacroEnvironment::default());
//!
//! // Each tick
//! let new_events = generator.tick(current_tick);
//! for event in &new_events {
//!     fundamentals.apply_event(event);
//! }
//! ```

// =============================================================================
// Module Declarations
// =============================================================================

pub mod config;
pub mod events;
pub mod fundamentals;
pub mod generator;
pub mod sectors;

// =============================================================================
// Re-exports
// =============================================================================

pub use config::{EventFrequency, FairValueDriftConfig, NewsGeneratorConfig};
pub use events::{FundamentalEvent, NewsEvent};
pub use fundamentals::{Fundamentals, MacroEnvironment, SymbolFundamentals};
pub use generator::NewsGenerator;
pub use sectors::SectorModel;
