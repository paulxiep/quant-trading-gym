//! Tier 3 background pool setup.

use agents::{BackgroundAgentPool, BackgroundPoolConfig};

use crate::Simulation;
use crate::sim_config::{SimConfig, SymbolSpec};

/// Setup the Tier 3 background pool if enabled.
pub(crate) fn setup_background_pool(
    sim: &mut Simulation,
    config: &SimConfig,
    symbols: &[SymbolSpec],
) {
    if !config.enable_background_pool {
        return;
    }

    let pool_symbols: Vec<String> = symbols.iter().map(|s| s.symbol.clone()).collect();
    let symbol_sectors: std::collections::HashMap<String, types::Sector> = symbols
        .iter()
        .map(|s| (s.symbol.clone(), s.sector))
        .collect();

    let pool_config = BackgroundPoolConfig {
        pool_size: config.background_pool_size,
        regime: config.background_regime,
        symbols: pool_symbols,
        mean_order_size: config.t3_mean_order_size,
        order_size_stddev: config.t3_order_size_stddev,
        max_order_size: config.t3_max_order_size,
        min_order_size: 1,
        price_spread_lambda: config.t3_price_spread_lambda,
        max_price_deviation: config.t3_max_price_deviation,
        sentiment_decay: 0.995,
        max_sentiment: 0.8,
        news_sentiment_scale: 0.5,
        enable_sanity_check: true,
        max_pnl_loss_fraction: 0.05,
        base_activity_override: config.t3_base_activity,
    };

    let mut pool = BackgroundAgentPool::new(pool_config, 42);
    pool.init_sectors(&symbol_sectors);
    sim.set_background_pool(pool);
}
