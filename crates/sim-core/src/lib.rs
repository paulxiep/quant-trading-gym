//! Sim-core: Market mechanics for the Quant Trading Gym.
//!
//! This crate provides the core market simulation components:
//! - Order book management with price-time priority
//! - Batch auction for parallel order processing (V3.6)
//! - Slippage and market impact calculation (V2.2)
//! - Market abstractions for multi-symbol support (V2.3)
//! - Error handling for market operations

mod batch_auction;
mod error;
mod market;
mod order_book;
mod slippage;

pub use batch_auction::{BatchAuction, BatchAuctionResult, run_parallel_auctions};
pub use error::{Result, SimCoreError};
pub use market::{Market, MarketView, SingleSymbolMarket};
pub use order_book::{OrderBook, PriceLevel};
pub use slippage::{ImpactEstimate, SlippageCalculator};
