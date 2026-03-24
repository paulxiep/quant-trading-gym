//! Subsystem implementations for simulation decomposition.
//!
//! Each subsystem encapsulates one area of responsibility:
//! - `MarketDataManager`: Candles, trades, indicators
//! - `AgentOrchestrator`: Agent execution across tiers
//! - `AuctionEngine`: Order collection and batch auctions
//! - `RiskManager`: Position tracking and risk metrics
//! - `NewsEngine`: News events and fundamentals

mod agents;
mod auction;
mod market_data;
mod news;
mod risk;

pub use agents::AgentOrchestrator;
pub use auction::{AuctionEngine, OrderCollectionResult, OrderValidationCtx};
pub use market_data::MarketDataManager;
pub use news::NewsEngine;
pub use risk::RiskManager;
