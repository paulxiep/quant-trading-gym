//! Tier 1 Trading Strategies - Full-context agents that run every tick.
//!
//! These strategies implement the `Agent` trait and receive complete
//! `StrategyContext` each tick, enabling complex computations including
//! indicator calculations, factor scoring, and risk analysis.
//!
//! # Available Strategies
//!
//! ## Market Infrastructure
//! - [`NoiseTrader`] - Random orders near mid price to generate activity
//! - [`MarketMaker`] - Provides two-sided liquidity with bid/ask spread
//!
//! ## Technical Strategies
//! - [`MomentumTrader`] - RSI-based momentum (buy oversold, sell overbought)
//! - [`TrendFollower`] - SMA crossover trend following (golden/death cross)
//! - [`MacdCrossover`] - MACD signal line crossover strategy
//! - [`BollingerReversion`] - Mean reversion using Bollinger Bands
//!
//! ## Execution Algorithms
//! - [`VwapExecutor`] - VWAP-targeting order execution algorithm
//!
//! ## Multi-Symbol Strategies (V3.3)
//! - [`PairsTrading`] - Cointegration-based spread trading between two symbols

mod bollinger_reversion;
mod macd_crossover;
mod market_maker;
mod momentum;
mod noise_trader;
mod pairs_trading;
mod trend_follower;
mod vwap_executor;

pub use bollinger_reversion::{BollingerReversion, BollingerReversionConfig};
pub use macd_crossover::{MacdCrossover, MacdCrossoverConfig};
pub use market_maker::{MarketMaker, MarketMakerConfig};
pub use momentum::{MomentumConfig, MomentumTrader};
pub use noise_trader::{NoiseTrader, NoiseTraderConfig};
pub use pairs_trading::{PairsTrading, PairsTradingConfig};
pub use trend_follower::{TrendFollower, TrendFollowerConfig};
pub use vwap_executor::{VwapExecutor, VwapExecutorConfig};
