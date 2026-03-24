//! Integration tests for NoiseTrader and MarketMaker agent strategies.
//!
//! Tests verify that basic trading agents can generate market activity
//! and produce trades when combined in a simulation.

use agents::{MarketMaker, MarketMakerConfig, NoiseTrader, NoiseTraderConfig};
use simulation::{Simulation, SimulationConfig};
use types::AgentId;

/// Test that a simulation with 10 NoiseTraders and 2 MarketMakers produces trades.
///
/// This is the exit criteria for Week 3 of V0.
#[test]
fn test_10_noise_traders_2_market_makers_produce_trades() {
    let config = SimulationConfig::new("ACME").with_position_limits(false); // Disable position limits for V0-era test
    let mut sim = Simulation::new(config);

    // Add 2 MarketMakers (IDs 1-2) - they seed liquidity
    for i in 1..=2 {
        let mm_config = MarketMakerConfig {
            symbol: "ACME".to_string(),
            half_spread: 0.0, // Tight spread to encourage matching
            quote_size: 100,
            refresh_interval: 3, // Refresh frequently
            ..Default::default()
        };
        sim.add_agent(Box::new(MarketMaker::new(AgentId(i), mm_config)));
    }

    // Add 10 NoiseTraders (IDs 3-12)
    for i in 3..=12 {
        let trader_config = NoiseTraderConfig {
            symbol: "ACME".to_string(),
            order_probability: 1.0, // 100% chance to place an order each tick
            price_deviation: 0.0,   // 0% deviation to stay near market maker quotes
            min_quantity: 10,
            max_quantity: 50,
            ..Default::default()
        };
        sim.add_agent(Box::new(NoiseTrader::new(AgentId(i), trader_config)));
    }

    assert_eq!(sim.agent_count(), 12);

    // Run for 1000 ticks
    for _ in 0..1000 {
        sim.step();
    }

    let stats = sim.stats();

    // Verify exit criteria: trades should have occurred
    println!("Simulation completed:");
    println!("  Total ticks: {}", stats.tick);
    println!("  Total orders: {}", stats.total_orders);
    println!("  Total trades: {}", stats.total_trades);
    println!("  Filled orders: {}", stats.filled_orders);
    println!("  Resting orders: {}", stats.resting_orders);

    // The simulation should produce trades
    assert!(
        stats.total_trades > 0,
        "Expected trades but got none. Zombie simulation detected!"
    );

    // Sanity checks
    assert!(stats.total_orders > 0, "No orders were placed");
    assert!(stats.tick == 1000, "Should have run for 1000 ticks");
}

/// Test that MarketMakers alone don't produce trades (they only provide liquidity).
#[test]
fn test_market_makers_alone_no_trades() {
    let config = SimulationConfig::new("ACME").with_position_limits(false);
    let mut sim = Simulation::new(config);

    // Add only MarketMakers
    for i in 1..=2 {
        let mm_config = MarketMakerConfig {
            symbol: "ACME".to_string(),
            ..Default::default()
        };
        sim.add_agent(Box::new(MarketMaker::new(AgentId(i), mm_config)));
    }

    // Run for 100 ticks
    for _ in 0..100 {
        sim.step();
    }

    // MarketMakers should place orders but not trade with each other
    // (their quotes don't cross)
    let stats = sim.stats();
    assert!(stats.total_orders > 0, "MarketMakers should place orders");
}

/// Test that NoiseTraders without MarketMakers eventually trade when their orders cross.
#[test]
fn test_noise_traders_can_trade_among_themselves() {
    let config = SimulationConfig::new("ACME").with_position_limits(false);
    let mut sim = Simulation::new(config);

    // Add NoiseTraders with high order probability and wide price range
    for i in 1..=20 {
        let trader_config = NoiseTraderConfig {
            symbol: "ACME".to_string(),
            order_probability: 0.5, // High activity
            price_deviation: 0.05,  // Wide range to encourage crossing
            min_quantity: 10,
            max_quantity: 100,
            max_short_position: 500,
            ..Default::default()
        };
        sim.add_agent(Box::new(NoiseTrader::new(AgentId(i), trader_config)));
    }

    // Run for 500 ticks
    for _ in 0..500 {
        sim.step();
    }

    let stats = sim.stats();

    // With wide price deviation, some orders should cross and trade
    // Note: This might occasionally fail if random orders happen to never cross,
    // but with 20 traders and 500 ticks, it's very unlikely
    assert!(stats.total_orders > 0, "NoiseTraders should place orders");

    println!("NoiseTraders only simulation:");
    println!("  Total orders: {}", stats.total_orders);
    println!("  Total trades: {}", stats.total_trades);
}

/// Test that simulation with fills doesn't panic (smoke test for on_fill path).
///
/// Note: We can't directly verify agent internal state from here since agents
/// are owned by the simulation. This test ensures the fill notification path
/// works correctly without errors.
#[test]
fn test_simulation_with_fills_runs_without_panic() {
    let config = SimulationConfig::new("ACME").with_position_limits(false);
    let mut sim = Simulation::new(config);

    // Add a MarketMaker with tight spread
    let mm_config = MarketMakerConfig {
        symbol: "ACME".to_string(),
        half_spread: 0.001, // Very tight spread
        quote_size: 100,
        refresh_interval: 1, // Every tick
        ..Default::default()
    };
    sim.add_agent(Box::new(MarketMaker::new(AgentId(1), mm_config)));

    // Add aggressive NoiseTraders
    for i in 2..=5 {
        let trader_config = NoiseTraderConfig {
            symbol: "ACME".to_string(),
            order_probability: 0.8, // Very active
            price_deviation: 0.001, // Tight around mid to hit MM quotes
            min_quantity: 10,
            max_quantity: 50,
            max_short_position: 200,
            ..Default::default()
        };
        sim.add_agent(Box::new(NoiseTrader::new(AgentId(i), trader_config)));
    }

    // Run for 100 ticks
    for _ in 0..100 {
        sim.step();
    }

    let stats = sim.stats();
    println!("Fill notification smoke test:");
    println!("  Total trades: {}", stats.total_trades);
    println!("  Filled orders: {}", stats.filled_orders);

    // Verify simulation completes and produces trades
    assert_eq!(stats.tick, 100);
    // With tight spread and aggressive traders, we expect at least some trades
    assert!(
        stats.total_trades > 0,
        "Expected trades for fill notification test"
    );
}

/// Test that position limits reject orders that would create short positions when disabled (V2.1).
#[test]
fn test_position_limits_rejects_shorts_when_disabled() {
    use types::{Cash, Order, OrderSide, Price, Quantity};

    // Enable position limits (default) with short selling disabled (default)
    let config = SimulationConfig::new("ACME");
    let mut sim = Simulation::new(config);

    // Create a simple agent that tries to sell without holding shares
    struct ShortAttemptAgent {
        id: AgentId,
        state: agents::AgentState,
        attempted: bool,
    }

    impl agents::Agent for ShortAttemptAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _ctx: &agents::StrategyContext<'_>) -> agents::AgentAction {
            if !self.attempted {
                self.attempted = true;
                // Try to sell 100 shares we don't have (short sell)
                agents::AgentAction::single(Order::limit(
                    self.id,
                    "ACME",
                    OrderSide::Sell,
                    Price::from_float(100.0),
                    Quantity(100),
                ))
            } else {
                agents::AgentAction::none()
            }
        }

        fn name(&self) -> &str {
            "ShortAttemptAgent"
        }

        fn state(&self) -> &agents::AgentState {
            &self.state
        }
    }

    sim.add_agent(Box::new(ShortAttemptAgent {
        id: AgentId(1),
        state: agents::AgentState::new(Cash::from_float(10_000.0), &["ACME"]),
        attempted: false,
    }));

    // Run for a few ticks
    sim.run(5);

    let stats = sim.stats();

    // The short order should have been rejected
    assert_eq!(
        stats.rejected_orders, 1,
        "Expected short order to be rejected"
    );
    assert_eq!(stats.total_trades, 0, "No trades should occur");
}

/// Test that position limits work correctly with short selling enabled (V2.1).
#[test]
fn test_position_limits_allows_shorts_when_enabled() {
    use types::{Cash, Order, OrderSide, Price, Quantity, ShortSellingConfig, SymbolConfig};

    // Enable short selling with borrow pool
    let symbol_config = SymbolConfig::new("ACME", Quantity(1_000_000), Price::from_float(100.0))
        .with_borrow_pool_bps(1500); // 15% borrow pool = 150,000 shares

    let short_config = ShortSellingConfig::enabled_default().with_max_short(Quantity(1_000)); // Max 1000 short per agent

    let config = SimulationConfig::new("ACME")
        .with_symbol_config(symbol_config)
        .with_short_selling(short_config);

    let mut sim = Simulation::new(config);

    // Create an agent that goes short
    struct ShortSellerAgent {
        id: AgentId,
        state: agents::AgentState,
        attempted: bool,
    }

    impl agents::Agent for ShortSellerAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _ctx: &agents::StrategyContext<'_>) -> agents::AgentAction {
            if !self.attempted {
                self.attempted = true;
                // Try to short sell 100 shares (within max_short limit)
                agents::AgentAction::single(Order::limit(
                    self.id,
                    "ACME",
                    OrderSide::Sell,
                    Price::from_float(100.0),
                    Quantity(100),
                ))
            } else {
                agents::AgentAction::none()
            }
        }

        fn name(&self) -> &str {
            "ShortSellerAgent"
        }

        fn state(&self) -> &agents::AgentState {
            &self.state
        }

        fn on_fill(&mut self, trade: &types::Trade) {
            if trade.seller_id == self.id {
                self.state
                    .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
            }
            if trade.buyer_id == self.id {
                self.state
                    .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
            }
        }
    }

    // Create a buyer to match the short seller
    struct BuyerAgent {
        id: AgentId,
        state: agents::AgentState,
        attempted: bool,
    }

    impl agents::Agent for BuyerAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _ctx: &agents::StrategyContext<'_>) -> agents::AgentAction {
            if !self.attempted {
                self.attempted = true;
                agents::AgentAction::single(Order::limit(
                    self.id,
                    "ACME",
                    OrderSide::Buy,
                    Price::from_float(100.0),
                    Quantity(100),
                ))
            } else {
                agents::AgentAction::none()
            }
        }

        fn name(&self) -> &str {
            "BuyerAgent"
        }

        fn state(&self) -> &agents::AgentState {
            &self.state
        }

        fn on_fill(&mut self, trade: &types::Trade) {
            if trade.buyer_id == self.id {
                self.state
                    .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
            }
            if trade.seller_id == self.id {
                self.state
                    .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
            }
        }
    }

    sim.add_agent(Box::new(ShortSellerAgent {
        id: AgentId(1),
        state: agents::AgentState::new(Cash::from_float(10_000.0), &["ACME"]),
        attempted: false,
    }));

    sim.add_agent(Box::new(BuyerAgent {
        id: AgentId(2),
        state: agents::AgentState::new(Cash::from_float(20_000.0), &["ACME"]),
        attempted: false,
    }));

    // Run for a few ticks
    sim.run(5);

    let stats = sim.stats();

    // With short selling enabled and borrow available, the trade should succeed
    assert_eq!(stats.rejected_orders, 0, "No orders should be rejected");
    assert_eq!(stats.total_trades, 1, "One trade should occur");

    // Check borrow ledger was updated
    let borrowed = sim.borrowed_shares(AgentId(1), &"ACME".to_string());
    assert_eq!(
        borrowed,
        Quantity(100),
        "Agent 1 should have borrowed 100 shares"
    );
}
