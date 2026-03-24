//! Tier 1 agent spawning: market makers, noise traders, quant strategies, pairs traders.

use agents::{
    Agent, BollingerReversion, BollingerReversionConfig, MacdCrossover, MacdCrossoverConfig,
    MarketMaker, MarketMakerConfig, MomentumConfig, MomentumTrader, NoiseTrader, NoiseTraderConfig,
    PairsTrading, PairsTradingConfig, TrendFollower, TrendFollowerConfig, VwapExecutor,
    VwapExecutorConfig,
};
use rand::prelude::SliceRandom;
use types::{AgentId, Cash, Price};

use crate::sim_config::{SimConfig, SymbolSpec, Tier1AgentType};

/// Create a MarketMakerConfig for a given symbol spec.
pub(crate) fn make_mm_config(spec: &SymbolSpec, config: &SimConfig) -> MarketMakerConfig {
    MarketMakerConfig {
        symbol: spec.symbol.clone(),
        initial_price: spec.initial_price,
        half_spread: config.mm_half_spread,
        quote_size: config.mm_quote_size,
        refresh_interval: config.mm_refresh_interval,
        max_inventory: config.mm_max_inventory,
        inventory_skew: config.mm_inventory_skew,
        initial_cash: config.mm_initial_cash,
        initial_position: config.mm_initial_position,
        fair_value_weight: 0.3,
        max_long_position: config.mm_max_long_position,
        max_short_position: config.mm_max_short_position,
    }
}

/// Spawn a single agent of the given type with the specified ID.
pub(crate) fn create_agent(
    agent_type: Tier1AgentType,
    id: u64,
    config: &SimConfig,
    symbol: &str,
    initial_price: Price,
) -> Box<dyn Agent> {
    let id = AgentId(id);

    match agent_type {
        Tier1AgentType::NoiseTrader => {
            let target_equity = config.quant_initial_cash.to_float();
            let position_value = config.nt_initial_position as f64 * initial_price.to_float();
            let adjusted_cash = target_equity - position_value;

            let nt_config = NoiseTraderConfig {
                symbol: symbol.to_string(),
                order_probability: config.nt_order_probability,
                initial_price,
                price_deviation: config.nt_price_deviation,
                min_quantity: config.nt_min_quantity,
                max_quantity: config.nt_max_quantity,
                initial_cash: Cash::from_float(adjusted_cash),
                initial_position: config.nt_initial_position,
                max_long_position: config.nt_max_long_position,
                max_short_position: config.nt_max_short_position,
            };
            Box::new(NoiseTrader::new(id, nt_config))
        }
        Tier1AgentType::MomentumTrader => {
            let momentum_config = MomentumConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_long_position,
                ..Default::default()
            };
            Box::new(MomentumTrader::new(id, momentum_config))
        }
        Tier1AgentType::TrendFollower => {
            let trend_config = TrendFollowerConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_long_position,
                ..Default::default()
            };
            Box::new(TrendFollower::new(id, trend_config))
        }
        Tier1AgentType::MacdTrader => {
            let macd_config = MacdCrossoverConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_long_position,
                ..Default::default()
            };
            Box::new(MacdCrossover::new(id, macd_config))
        }
        Tier1AgentType::BollingerTrader => {
            let bollinger_config = BollingerReversionConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_long_position,
                ..Default::default()
            };
            Box::new(BollingerReversion::new(id, bollinger_config))
        }
        Tier1AgentType::VwapExecutor => {
            let vwap_config = VwapExecutorConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                ..Default::default()
            };
            Box::new(VwapExecutor::new(id, vwap_config))
        }
        Tier1AgentType::PairsTrading => {
            let pairs_config = PairsTradingConfig::new(symbol, symbol)
                .with_initial_cash(config.quant_initial_cash)
                .with_max_position(config.quant_max_long_position);
            Box::new(PairsTrading::new(id, pairs_config))
        }
    }
}

/// Spawn market makers distributed across symbols.
pub(crate) fn spawn_market_makers(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let num_symbols = symbols.len();
    let per_symbol = config.num_market_makers / num_symbols;
    let remainder = config.num_market_makers % num_symbols;

    let mut next_id = start_id;

    let distributed: Vec<_> = symbols
        .iter()
        .flat_map(|spec| std::iter::repeat_n(spec, per_symbol))
        .zip(next_id..)
        .map(|(spec, id)| {
            Box::new(MarketMaker::new(AgentId(id), make_mm_config(spec, config))) as Box<dyn Agent>
        })
        .collect();
    next_id += distributed.len() as u64;

    let remainder_agents: Vec<_> = (0..remainder)
        .map(|_| symbols.choose(rng).unwrap())
        .zip(next_id..)
        .map(|(spec, id)| {
            Box::new(MarketMaker::new(AgentId(id), make_mm_config(spec, config))) as Box<dyn Agent>
        })
        .collect();
    next_id += remainder_agents.len() as u64;

    let agents = distributed.into_iter().chain(remainder_agents).collect();
    (agents, next_id)
}

/// Spawn noise traders distributed across symbols.
pub(crate) fn spawn_noise_traders(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let num_symbols = symbols.len();
    let per_symbol = config.num_noise_traders / num_symbols;
    let remainder = config.num_noise_traders % num_symbols;

    let mut next_id = start_id;

    let distributed: Vec<_> = symbols
        .iter()
        .flat_map(|spec| std::iter::repeat_n(spec, per_symbol))
        .zip(next_id..)
        .map(|(spec, id)| {
            create_agent(
                Tier1AgentType::NoiseTrader,
                id,
                config,
                &spec.symbol,
                spec.initial_price,
            )
        })
        .collect();
    next_id += distributed.len() as u64;

    let remainder_agents: Vec<_> = (0..remainder)
        .map(|_| symbols.choose(rng).unwrap())
        .zip(next_id..)
        .map(|(spec, id)| {
            create_agent(
                Tier1AgentType::NoiseTrader,
                id,
                config,
                &spec.symbol,
                spec.initial_price,
            )
        })
        .collect();
    next_id += remainder_agents.len() as u64;

    let agents = distributed.into_iter().chain(remainder_agents).collect();
    (agents, next_id)
}

/// Spawn quant strategy agents (momentum, trend, MACD, etc.) randomly across symbols.
pub(crate) fn spawn_quant_agents(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let agent_counts = [
        (Tier1AgentType::MomentumTrader, config.num_momentum_traders),
        (Tier1AgentType::TrendFollower, config.num_trend_followers),
        (Tier1AgentType::MacdTrader, config.num_macd_traders),
        (
            Tier1AgentType::BollingerTrader,
            config.num_bollinger_traders,
        ),
        (Tier1AgentType::VwapExecutor, config.num_vwap_executors),
    ];

    let specs: Vec<_> = agent_counts
        .iter()
        .flat_map(|(agent_type, count)| std::iter::repeat_n(*agent_type, *count))
        .map(|agent_type| (agent_type, symbols.choose(rng).unwrap()))
        .collect();

    let agents: Vec<_> = specs
        .iter()
        .zip(start_id..)
        .map(|((agent_type, spec), id)| {
            create_agent(*agent_type, id, config, &spec.symbol, spec.initial_price)
        })
        .collect();

    let next_id = start_id + agents.len() as u64;
    (agents, next_id)
}

/// Spawn pairs trading agents (requires at least 2 symbols).
pub(crate) fn spawn_pairs_traders(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
) -> (Vec<Box<dyn Agent>>, u64) {
    if symbols.len() < 2 {
        return (Vec::new(), start_id);
    }

    let num_symbols = symbols.len();
    let agents: Vec<_> = (0..config.num_pairs_traders)
        .zip(start_id..)
        .map(|(i, id)| {
            let idx_a = i % num_symbols;
            let idx_b = (i + 1) % num_symbols;
            let spec_a = &symbols[idx_a];
            let spec_b = &symbols[idx_b];

            let pairs_config = PairsTradingConfig::new(&spec_a.symbol, &spec_b.symbol)
                .with_initial_cash(config.quant_initial_cash)
                .with_max_position(config.quant_max_long_position);

            Box::new(PairsTrading::new(AgentId(id), pairs_config)) as Box<dyn Agent>
        })
        .collect();

    let next_id = start_id + agents.len() as u64;
    (agents, next_id)
}

/// Spawn random Tier 1 agents to fill to minimum count.
pub(crate) fn spawn_random_tier1_agents(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let count = config.random_tier1_count();
    let agents: Vec<_> = (0..count)
        .map(|_| (Tier1AgentType::random(rng), symbols.choose(rng).unwrap()))
        .zip(start_id..)
        .map(|((agent_type, spec), id)| {
            create_agent(agent_type, id, config, &spec.symbol, spec.initial_price)
        })
        .collect();

    let next_id = start_id + agents.len() as u64;
    (agents, next_id)
}
