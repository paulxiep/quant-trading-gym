//! Tier 3: Background Agent Pool (V3.4).
//!
//! Statistical order generation without individual agent instances.
//! Enables simulation of 90k+ background agents with ~2KB memory.
//!
//! # Architecture
//!
//! One pool instance trades ALL symbols:
//! - Randomly selects which symbol to trade each order
//! - Tracks sentiment per-symbol (sector news affects right symbols)
//! - Uses single accounting ledger for aggregate P&L
//!
//! # Usage
//!
//! ```ignore
//! use agents::tier3::{BackgroundAgentPool, BackgroundPoolConfig, PoolContext};
//!
//! // Create pool with all tradeable symbols
//! let config = BackgroundPoolConfig::new(vec!["AAPL", "GOOG", "MSFT"]);
//! let mut pool = BackgroundAgentPool::new(config, 42);
//!
//! // Each tick in simulation:
//! let mid_prices = market.mid_prices(); // HashMap<Symbol, Price>
//! let ctx = PoolContext {
//!     tick,
//!     mid_prices: &mid_prices,
//!     active_events: &events,
//!     symbol_sectors: &sectors,
//! };
//! let orders = pool.generate(&ctx);
//!
//! // Process orders through matching engine...
//! // Then record fills in accounting
//! for trade in trades {
//!     if trade.buyer_id == BACKGROUND_POOL_ID {
//!         pool.accounting_mut().record_trade_as_buyer(&trade.symbol, trade.price, trade.quantity);
//!     }
//! }
//! ```
//!
//! # Design Principles
//!
//! - **Declarative**: Behavior controlled by `BackgroundPoolConfig`
//! - **Modular**: Trait-based distributions allow swapping
//! - **SoC**: Pool generates; Simulation applies; Accounting tracks

mod accounting;
mod config;
mod distributions;
mod pool;

pub use accounting::{BackgroundPoolAccounting, SanityCheckResult, SymbolStats};
pub use config::{BackgroundPoolConfig, MarketRegime, RegimePreset};
pub use distributions::{
    ExponentialPriceSpread, LogNormalSize, PriceDistribution, SizeDistribution,
};
pub use pool::{BACKGROUND_POOL_ID, BackgroundAgentPool, PoolContext};
