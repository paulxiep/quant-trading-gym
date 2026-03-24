//! Agents crate: Trading agents for the Quant Trading Gym.
//!
//! This crate provides:
//! - The `Agent` trait that all trading agents must implement
//! - `StrategyContext` - unified context passed to agents each tick (V2.3)
//! - `AgentAction` for returning agent decisions
//! - `AgentState` for common state tracking (position, cash, metrics)
//! - `BorrowLedger` for tracking short-selling borrows (V2.1)
//! - `PositionValidator` for order validation against position limits (V2.1)
//! - Concrete strategy implementations (`strategies` module)
//! - `MIN_ORDER_PRICE` constant and `floor_price()` helper for price floor enforcement
//!
//! # Architecture (V2.3)
//!
//! Agents receive a `StrategyContext` each tick, providing:
//! - Multi-symbol market access via `MarketView` trait
//! - Per-symbol candles and indicators
//! - Per-symbol recent trades
//!
//! The simulation handles order routing, matching, and notifying agents of fills.
//!
//! # Position Limits (V2.1)
//!
//! The `PositionValidator` enforces realistic constraints:
//! - **Long positions**: Limited by cash available and shares_outstanding
//! - **Short positions**: Require borrows from `BorrowLedger`, limited by max_short_per_agent
//!
//! # Tiered Agent Architecture (V3.2+)
//!
//! - **Tier 1**: Smart agents running full strategy every tick
//! - **Tier 2**: Reactive agents waking on conditions (price cross, news, interval)
//! - **Tier 3**: Background pool for statistical order generation (V3.4)
//!
//! # Available Strategies
//!
//! ## Market Infrastructure (Phase 5)
//! - [`strategies::NoiseTrader`] - Random orders to generate market activity
//! - [`strategies::MarketMaker`] - Two-sided liquidity provider
//!
//! ## Technical Strategies (Phase 7)
//! - [`strategies::MomentumTrader`] - RSI-based momentum (buy oversold, sell overbought)
//! - [`strategies::TrendFollower`] - SMA crossover trend following
//! - [`strategies::MacdCrossover`] - MACD signal line crossover
//! - [`strategies::BollingerReversion`] - Bollinger Bands mean reversion
//!
//! ## Execution Algorithms (Phase 8)
//! - [`strategies::VwapExecutor`] - VWAP-targeting order execution
//!
//! # Example
//! ```ignore
//! use agents::{Agent, AgentAction, StrategyContext};
//! use types::AgentId;
//!
//! struct MyAgent {
//!     id: AgentId,
//!     symbol: String,
//! }
//!
//! impl Agent for MyAgent {
//!     fn id(&self) -> AgentId { self.id }
//!
//!     fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
//!         let mid = ctx.mid_price(&self.symbol);
//!         AgentAction::none()
//!     }
//! }
//! ```

mod borrow_ledger;
mod context;
pub mod ml_cache;
mod position_limits;
mod state;
pub mod tiers;
mod traits;

// Tiered agent architecture (V3.2+)
pub mod tier1;
pub mod tier2;
pub mod tier3;

pub use borrow_ledger::{BorrowLedger, BorrowPosition};
pub use context::StrategyContext;
pub use ml_cache::MlPredictionCache;
pub use position_limits::PositionValidator;
pub use state::{AgentState, PositionEntry};
// Re-export strategies from tier1 for backward compatibility
pub use tier1::{
    BollingerReversion, BollingerReversionConfig, MacdCrossover, MacdCrossoverConfig, MarketMaker,
    MarketMakerConfig, MomentumConfig, MomentumTrader, NoiseTrader, NoiseTraderConfig,
    PairsTrading, PairsTradingConfig, TrendFollower, TrendFollowerConfig, VwapExecutor,
    VwapExecutorConfig,
};
// Re-export ML model types (V5.5, V6.2, V6.3), MlAgent, and feature extraction
pub use ml_cache::FeatureVec;
#[allow(deprecated)]
pub use tier1::{
    CanonicalFeatures, ClassProbabilities, DecisionTree, EnsembleModel, FeatureExtractor,
    FullFeatures, GaussianNBPredictor, GradientBoosted, LinearPredictor, MinimalFeatures, MlAgent,
    MlAgentConfig, MlModel, ModelRegistry, RandomForest, extract_features, extract_features_raw,
    impute_features,
};
// Re-export Tier 2 types for V3.2
pub use tier2::{
    IndexStats, ReactiveAgent, ReactivePortfolio, ReactiveStrategyType, SectorRotator,
    SectorRotatorConfig, WakeConditionIndex,
};
// Re-export Tier 3 types for V3.4
pub use tier3::{
    BACKGROUND_POOL_ID, BackgroundAgentPool, BackgroundPoolAccounting, BackgroundPoolConfig,
    MarketRegime, PoolContext, SanityCheckResult,
};
pub use tiers::{
    ConditionUpdate, CrossDirection, OrderedPrice, PriceReference, TickFrequency, WakeCondition,
};
pub use traits::{Agent, AgentAction};

/// Minimum price floor for all orders ($0.01)
pub const MIN_ORDER_PRICE: f64 = 0.01;

/// Ensure price is at least the minimum floor.
/// This prevents negative price spirals where agents submit increasingly lower prices.
#[inline]
pub fn floor_price(price: f64) -> f64 {
    price.max(MIN_ORDER_PRICE)
}
