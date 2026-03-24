//! Tier 2 agent spawning: reactive agents and sector rotators.

use agents::{Agent, ReactiveAgent, ReactiveStrategyType, SectorRotator, SectorRotatorConfig};
use rand::Rng;
use std::collections::HashMap;
use types::{AgentId, Price, Quantity, Sector};

use crate::Simulation;
use crate::sim_config::{SimConfig, SymbolSpec};

/// Spawn Tier 2 reactive agents distributed across symbols.
///
/// Uses `config.reactive_tier2_count()` to respect the sector rotator budget.
pub(crate) fn spawn_tier2_agents(
    sim: &mut Simulation,
    next_id: &mut u64,
    config: &SimConfig,
    symbols: &[SymbolSpec],
    rng: &mut rand::prelude::ThreadRng,
) {
    let num_agents = config.reactive_tier2_count();
    if num_agents == 0 {
        return;
    }

    let num_symbols = symbols.len();
    let agents_per_symbol = num_agents / num_symbols;
    let remainder = num_agents % num_symbols;

    const PRICE_SCALE: i64 = 10_000;

    let make_strategies = |rng: &mut rand::prelude::ThreadRng| -> Vec<ReactiveStrategyType> {
        let buy_dollars = config.t2_buy_threshold_min
            + rng.r#gen::<f64>() * (config.t2_buy_threshold_max - config.t2_buy_threshold_min);
        let buy_price = Price((buy_dollars * PRICE_SCALE as f64) as i64);
        let entry_size = config.t2_order_size_min
            + rng.r#gen::<f64>() * (config.t2_order_size_max - config.t2_order_size_min);

        let stop_pct = config.t2_stop_loss_min
            + rng.r#gen::<f64>() * (config.t2_stop_loss_max - config.t2_stop_loss_min);

        let mut strategies = vec![
            ReactiveStrategyType::ThresholdBuyer {
                buy_price,
                size_fraction: entry_size,
            },
            ReactiveStrategyType::StopLoss { stop_pct },
        ];

        if rng.r#gen::<f64>() < config.t2_take_profit_prob {
            let target_pct = config.t2_take_profit_min
                + rng.r#gen::<f64>() * (config.t2_take_profit_max - config.t2_take_profit_min);
            strategies.push(ReactiveStrategyType::TakeProfit { target_pct });
        } else {
            let sell_dollars = config.t2_sell_threshold_min
                + rng.r#gen::<f64>()
                    * (config.t2_sell_threshold_max - config.t2_sell_threshold_min);
            strategies.push(ReactiveStrategyType::ThresholdSeller {
                sell_price: Price((sell_dollars * PRICE_SCALE as f64) as i64),
                size_fraction: 1.0,
            });
        }

        if rng.r#gen::<f64>() < config.t2_news_reactor_prob {
            strategies.push(ReactiveStrategyType::NewsReactor {
                min_magnitude: 0.3 + rng.r#gen::<f64>() * 0.4,
                sentiment_multiplier: 1.0 + rng.r#gen::<f64>() * 2.0,
            });
        }

        strategies
    };

    let agent_specs: Vec<_> = symbols
        .iter()
        .enumerate()
        .flat_map(|(sym_idx, spec)| {
            let count = agents_per_symbol + if sym_idx < remainder { 1 } else { 0 };
            std::iter::repeat_n(spec, count)
        })
        .collect();

    let start_id = *next_id;
    let agents: Vec<_> = agent_specs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            Box::new(ReactiveAgent::new(
                AgentId(start_id + i as u64),
                spec.symbol.clone().into(),
                make_strategies(rng),
                Quantity(config.t2_max_position),
                config.t2_initial_cash,
            )) as Box<dyn Agent>
        })
        .collect();

    *next_id += agents.len() as u64;
    for agent in agents {
        sim.add_agent(agent);
    }
}

/// Spawn Sector Rotator agents (V3.3 - part of Tier 2 budget).
///
/// Uses `config.effective_sector_rotators()` to clamp to tier2 budget.
pub(crate) fn spawn_sector_rotators(
    sim: &mut Simulation,
    next_id: &mut u64,
    config: &SimConfig,
    symbols: &[SymbolSpec],
) {
    let num_agents = config.effective_sector_rotators();
    if num_agents == 0 || symbols.is_empty() {
        return;
    }

    // Group symbols by sector
    let mut sector_symbols: HashMap<Sector, Vec<String>> = HashMap::new();
    for spec in symbols {
        sector_symbols
            .entry(spec.sector)
            .or_default()
            .push(spec.symbol.clone());
    }

    let start_id = *next_id;
    let agents: Vec<_> = (0..num_agents)
        .map(|i| {
            let mut rotator_config = SectorRotatorConfig::new()
                .with_initial_cash(config.quant_initial_cash)
                .with_sentiment_scale(0.3)
                .with_rebalance_threshold(0.05);

            for (sector, syms) in &sector_symbols {
                rotator_config = rotator_config.with_sector(*sector, syms.clone());
            }

            Box::new(SectorRotator::new(
                AgentId(start_id + i as u64),
                rotator_config,
            )) as Box<dyn Agent>
        })
        .collect();

    *next_id += agents.len() as u64;
    for agent in agents {
        sim.add_agent(agent);
    }
}
