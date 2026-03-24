# Quant Trading Gym: Project Plan

Ideated with LLM assistance, summarised by Opus 4.5 Thinking.

---

# Part 1: Project Overview

## Summary

A quantitative trading simulation built in Rust with RL training capabilities, modular quant strategies, risk management, microservices architecture, and chatbot interface. Designed to support 100,000+ agents through a tiered architecture.

**Repository:** `quant-trading-gym`

## Language Split

| Language | % | Components |
|----------|---|------------|
| Rust | 85% | types, sim-core, quant, news, agents, simulation, gym, storage, services |
| Python | 10% | Training scripts, experiments (via PyO3) |
| TypeScript | 5% | Frontend (Phase 22 G) |

## Guiding Mantra

> **"Declarative, Modular, SoC"**

Every implementation decision should be evaluated against these three principles. Before writing code, ask:
1. Am I describing behavior or implementing mechanics? (Declarative)
2. Can this be swapped out without ripple effects? (Modular)
3. Does this component have exactly one responsibility? (SoC)

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Declarative** | Config-driven behavior, strategies declare needs, data-driven logic |
| **Modular** | Crates compile independently, strategies are plugins, components swappable |
| Separation of Concerns | Each crate has single responsibility |
| Modularity | Strategies, observations, rewards are plugins |
| Trait Boundaries | Crates communicate through traits, not concrete types |
| Context Isolation | Each phase/task fits in one LLM session |
| Financial Precision | `i64` fixed-point for all monetary values (4 decimal places) |
| Sync/Async Separation | Simulation is sync, services are async, bridge via channels |
| Training Parity | Observation contracts ensure Rust == Python |
| Realistic Latency | Order latency prevents look-ahead bias |
| Agent Tiering | Smart/Reactive/Background tiers for 100k+ scale |
| Statistical Modeling | Background agents modeled statistically, not individually |
| Human-Playable Design | Time controls + quant dashboard for meaningful human gameplay |
| Position Realism | Short-selling requires borrows; position limits enforced per-agent |

## Scaling Targets

| Metric | Target |
|--------|--------|
| Total agents | 100,000+ |
| Memory budget | ≤2 GB |
| Tick latency | <10ms at 100k agents |
| Smart agents (Tier 1) | 10-100 |
| Reactive agents (Tier 2) | 1,000-10,000 |
| Background agents (Tier 3) | 90,000+ |

---

# Part 2: Agent Tiering Architecture

## Overview

| Tier | Name | Count | Behavior | Portfolio | P&L Tracking |
|------|------|-------|----------|-----------|--------------|
| 1 | Smart | 10-100 | Full strategy, every tick | Full | Individual |
| 2 | Reactive | 1k-10k | Event-triggered, simple rules | Single or Full | Individual |
| 3 | Background | 90k+ | Statistical generation | None | Aggregate only |

## Tier Details

**Tier 1: Smart Agents**
- Run full `Strategy::decide()` every tick
- Receive complete `StrategyContext` with indicators, factors, risk
- Individual portfolio tracking with full position history
- Used for: RL agents, complex quant strategies, market makers

**Tier 2: Reactive Agents**
- Only wake on specific conditions (price threshold, news event, interval)
- Lightweight decision logic via enum dispatch (not trait objects)
- Portfolio options: single-symbol (lightweight) or full
- Used for: threshold traders, news reactors, momentum followers

**Tier 3: Background Pool**
- No individual agent instances—statistical model only
- Order generation based on configurable parameters
- Reactive to market conditions and news events at aggregate level
- Trades recorded with sentinel `BACKGROUND_POOL_ID`
- Aggregate P&L tracked for sanity checking

## Memory Budget

| Component | Count | Per-Unit | Total |
|-----------|-------|----------|-------|
| Tier 1 agents | 100 | ~3 KB | 300 KB |
| Tier 2 agents (worst case: all multi-symbol) | 10,000 | ~1 KB | 10 MB |
| Tier 2 agents (best case: all single-symbol) | 10,000 | ~150 bytes | 1.5 MB |
| Tier 3 pool state | 1 | ~10 KB | 10 KB |
| Order books, caches, buffers | — | — | ~25 MB |
| **Total core (worst case)** | | | **~35 MB** |
| Headroom | | | ~1.9 GB |

*Note: Tier 2 rows show alternatives based on portfolio type. Actual usage depends on configuration.*

## Background Pool Behavior

The pool reacts dynamically to market conditions:

| Input | Effect |
|-------|--------|
| Macro news event | Shifts overall buy/sell bias |
| Sector news event | Shifts bias for stocks in that sector |
| Price volatility spike | Increases activity rate |
| Time passing | Sentiment decays toward neutral |

Stability mechanisms:
- Sentiment clamped to prevent runaway trends
- Contrarian fraction provides natural mean reversion
- Decay factor ensures return to baseline

### News Event Duration & Sentiment Decay

```rust
impl NewsGenerator {
    /// Returns events that are currently active (not expired).
    pub fn active_events(&self, current_tick: Tick) -> Vec<&NewsEvent> {
        self.events.iter()
            .filter(|e| current_tick < e.start_tick + e.duration_ticks)
            .collect()
    }
    
    /// Sentiment decay factor per tick (e.g., 0.99 = 1% decay/tick)
    pub const SENTIMENT_DECAY: f64 = 0.99;
}

impl BackgroundAgentPool {
    /// Called each tick to decay sentiment toward neutral.
    pub fn decay_sentiment(&mut self) {
        self.sentiment *= NewsGenerator::SENTIMENT_DECAY;
    }
}
```

### MarketRegime Preset Defaults

| Regime | Base Activity | Sentiment Volatility | Contrarian Fraction |
|--------|--------------|---------------------|-------------------|
| Calm | 0.1 | 0.05 | 0.3 |
| Normal | 0.3 | 0.15 | 0.25 |
| Volatile | 0.6 | 0.3 | 0.2 |
| Crisis | 0.9 | 0.5 | 0.15 |

**P&L Tracking (Sanity Check):**
- All fills against `BACKGROUND_POOL_ID` are recorded in `BackgroundPoolAccounting`
- Tracks: total buy volume, total sell volume, VWAP buy price, VWAP sell price
- Computed P&L = (sell_volume × vwap_sell) - (buy_volume × vwap_buy)
- Sanity check fails if |computed P&L| exceeds configurable threshold (e.g., 5% of market cap)
- Failure indicates misconfigured order generation parameters (e.g., excessive directional bias)

Order generation characteristics:
- Price: Exponential decay from mid (most orders near spread, few far away)
- Size: Power-law or log-normal distribution (many small, few large)
- Latency: Random 1-5 ticks

---

# Part 3: Crate Structure

```
quant-trading-gym/
├── crates/
│   ├── types/                  # Shared data types
│   │   ├── lib.rs              # All types including OrderId, AgentId
│   │   └── constants.rs        # BACKGROUND_POOL_ID sentinel
│   │
│   ├── sim-core/               # Market mechanics (SYNC)
│   │   ├── lib.rs
│   │   ├── error.rs            # SimCoreError
│   │   ├── order_book.rs
│   │   ├── matching.rs
│   │   ├── pending_orders.rs   # Latency queue
│   │   └── market.rs
│   │
│   ├── quant/                  # Pure math calculations (SYNC)
│   │   ├── lib.rs
│   │   ├── error.rs            # QuantError
│   │   ├── indicators.rs
│   │   ├── risk.rs
│   │   ├── factors.rs
│   │   ├── execution.rs
│   │   └── stats.rs
│   │
│   ├── news/                   # Event generation (SYNC)
│   │   ├── lib.rs
│   │   ├── generator.rs
│   │   └── sectors.rs
│   │
│   ├── agents/                 # Trading agents (SYNC)
│   │   ├── lib.rs
│   │   ├── error.rs            # AgentError
│   │   ├── traits.rs           # Strategy trait with wake conditions
│   │   ├── tiers.rs            # TickFrequency, WakeCondition
│   │   ├── context.rs          # StrategyContext, LightweightContext
│   │   ├── orchestrator.rs     # TieredOrchestrator
│   │   ├── registry.rs
│   │   │
│   │   ├── tier1/              # Smart agents
│   │   │   ├── mod.rs
│   │   │   ├── agent.rs        # Full Agent struct
│   │   │   └── strategies/
│   │   │       ├── mod.rs
│   │   │       ├── noise_trader.rs   # Random buy/sell (market microstructure term)
│   │   │       ├── market_maker.rs
│   │   │       ├── rsi_momentum.rs
│   │   │       ├── macd_crossover.rs
│   │   │       ├── bollinger_reversion.rs
│   │   │       ├── trend_following.rs  # SMA/EMA crossover
│   │   │       ├── pairs_trading.rs
│   │   │       ├── factor_long_short.rs
│   │   │       ├── vwap_executor.rs
│   │   │       ├── news_reactive.rs    # Event-driven trading
│   │   │       └── rl_agent.rs
│   │   │
│   │   ├── tier2/              # Reactive agents
│   │   │   ├── mod.rs
│   │   │   ├── agent.rs        # ReactiveAgent (lightweight)
│   │   │   ├── portfolio.rs    # ReactivePortfolio enum
│   │   │   ├── strategies.rs   # ReactiveStrategyType enum
│   │   │   └── wake_index.rs   # WakeConditionIndex
│   │   │
│   │   └── tier3/              # Background pool
│   │       ├── mod.rs
│   │       ├── pool.rs         # BackgroundAgentPool
│   │       ├── config.rs       # BackgroundConfig, MarketRegime
│   │       ├── distributions.rs # SizeDistribution, price modeling
│   │       └── accounting.rs   # BackgroundPoolAccounting
│   │
│   ├── simulation/             # Orchestration (SYNC)
│   │   ├── lib.rs
│   │   ├── error.rs            # SimulationError
│   │   ├── runner.rs
│   │   ├── config.rs           # SimulationConfig, SimulationPreset
│   │   ├── builder.rs          # SimulationBuilder
│   │   ├── hooks.rs
│   │   ├── metrics.rs          # TickMetrics, MetricsHook
│   │   ├── warmup.rs           # Warm-up tick logic
│   │   └── snapshot.rs         # to_snapshot(), from_snapshot()
│   │
│   ├── gym/                    # RL environment (SYNC)
│   │   ├── lib.rs
│   │   ├── error.rs            # GymError
│   │   ├── env.rs
│   │   ├── builder.rs          # TradingEnvBuilder
│   │   ├── observation/
│   │   │   ├── mod.rs
│   │   │   ├── traits.rs
│   │   │   ├── contract.rs     # Observation parity contract
│   │   │   ├── price.rs
│   │   │   ├── book.rs
│   │   │   ├── indicators.rs
│   │   │   ├── portfolio.rs
│   │   │   ├── microstructure.rs  # Order flow, depth ratio
│   │   │   └── composite.rs
│   │   ├── reward/
│   │   │   ├── mod.rs
│   │   │   ├── traits.rs
│   │   │   ├── pnl.rs
│   │   │   ├── sharpe.rs
│   │   │   ├── drawdown.rs
│   │   │   └── composite.rs
│   │   └── pyo3.rs
│   │
│   ├── storage/                # Persistence (SYNC internals)
│   │   ├── lib.rs
│   │   ├── error.rs            # StorageError
│   │   ├── schema.rs
│   │   ├── connection.rs
│   │   ├── buffer.rs           # In-memory trade buffer
│   │   ├── decimal_serde.rs    # Decimal to/from string helpers
│   │   ├── persistence_hook.rs # Implements SimulationHook
│   │   └── stores/
│   │       ├── trades.rs
│   │       ├── candles.rs
│   │       ├── portfolios.rs
│   │       ├── risk.rs
│   │       └── snapshots.rs
│   │
│   └── services/               # HTTP APIs (ASYNC)
│       ├── common/
│       │   ├── errors.rs
│       │   ├── middleware.rs
│       │   └── bridge.rs       # Sync/async bridge
│       ├── data/               # Consolidated: analytics, portfolio, risk, news
│       ├── game/               # WebSocket, sessions, time control, orders, BFF
│       ├── storage/            # Snapshots, trade log, queries
│       └── chatbot/            # NLP → API routing
│
├── python/
│   ├── quant_trading_gym/      # PyO3 module
│   │   ├── __init__.py
│   │   └── observation.py      # Python observation contract
│   ├── train_dqn.py
│   ├── train_ppo.py
│   ├── export_onnx.py          # ONNX export script
│   └── notebooks/
│
├── tests/
│   ├── parity/
│   │   └── test_observation_parity.py
│   └── scale/                  # Scale tests (feature-gated)
│       ├── test_100k_agents.rs
│       └── test_throughput.rs
│
└── frontend/                   # React app (Phase 22 G)

---

# Part 4: Dependency Graph

```
                              ┌─────────┐
                              │  types  │
                              └────┬────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ sim-core │  │  quant   │  │   news   │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
                   └──────────────┼──────────────┘
                                  ▼
                            ┌──────────┐
                            │  agents  │
                            └────┬─────┘
                                 │
                                 ▼
                           ┌───────────┐
                           │simulation │
                           └─────┬─────┘
                                 │
                                 ▼
                           ┌───────────┐
                           │  storage  │
                           └─────┬─────┘
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │ 11: Integration     │
                       │ 12: Scale Testing   │
                       └─────────┬───────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
   ┌──────────────┐                             ┌──────────────┐
   │  RL Track    │                             │  Game Track  │
   │  (13-18 RL)  │                             │  (13-22 G)   │
   └──────┬───────┘                             └──────┬───────┘
          │                                            │
          ▼                                            ▼
   ┌──────────┐                               ┌────────────┐
   │   gym    │                               │  services  │
   │   PyO3   │                               │  common    │
   │   ONNX   │                               └──────┬─────┘
   └────┬─────┘                                      │
        │                                            ▼
        ▼                                     ┌────────────┐
   ┌──────────┐                               │ individual │
   │RL Agent  │                               │  services  │
   │(Strategy)│                               └──────┬─────┘
   └──────────┘                                      │
                                                     ▼
                                              ┌────────────┐
                                              │ game infra │
                                              │  frontend  │
                                              └────────────┘
```

**Phase Structure:**
- **Core (1-10):** types → sim-core/quant/news → agents → simulation → storage
- **Integration (11):** Integration tests across all core crates
- **Scale Testing (12):** 100k agent performance verification
- **Parallel Tracks (13+):** RL and Game tracks proceed independently with parallel numbering

**Track Details:**
- **RL Track (13-18 RL):** gym → PyO3 → training → ONNX → RL Agent Strategy
- **Game Track (13-22 G):** services foundation → individual services → game infra → frontend

**Key Architecture Points:**
- **RL Track** and **Game Track** are PARALLEL—they do NOT depend on each other
- Integration and Scale testing happen BEFORE the tracks split (validates core is solid)
- **RL Agent** is NOT a service—it's a Strategy implementation in `agents` crate
- **storage** is shared infrastructure used by services for persistence

---

# Part 5: Types Crate

## File: `crates/types/Cargo.toml`

```toml
[package]
name = "types"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
```

## Core Types

| Type | Definition | Notes |
|------|------------|-------|
| `Symbol` | `String` (ticker) | Stock symbol |
| `Price` | `struct Price(i64)` | Fixed-point, 4 decimals (10000 = $1.00) |
| `Cash` | `struct Cash(i64)` | Fixed-point, 4 decimals |
| `Quantity` | `u64` | Number of shares |
| `OrderId`, `AgentId` | `u64` | Sequential, not UUID |
| `Timestamp` | `u64` (epoch ms) | Wall clock time |
| `Tick` | `u64` | Simulation tick number |
| `BACKGROUND_POOL_ID` | `AgentId = 0` | Sentinel for Tier 3 trades |

### Price and Cash Newtypes

```rust
/// Fixed-point price with 4 decimal places.
/// 10000 = $1.00, 15000 = $1.50, 100 = $0.01
pub const PRICE_SCALE: i64 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Price(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Cash(pub i64);

impl Price {
    pub fn from_float(v: f64) -> Self { Self((v * PRICE_SCALE as f64) as i64) }
    pub fn to_float(self) -> f64 { self.0 as f64 / PRICE_SCALE as f64 }
}

impl Cash {
    pub fn from_float(v: f64) -> Self { Self((v * PRICE_SCALE as f64) as i64) }
    pub fn to_float(self) -> f64 { self.0 as f64 / PRICE_SCALE as f64 }
}
```

## Order Types

| Type | Variants/Fields |
|------|-----------------|
| `OrderSide` | `Buy`, `Sell` |
| `OrderType` | `Market`, `Limit { price }` |
| `ExecutionAlgoType` | `VWAP { target_qty, duration }`, `TWAP { target_qty, duration }` — *Execution algorithms generate child orders; not handled by matching engine directly* |
| `Order` | `id`, `agent_id`, `symbol`, `side`, `order_type`, `quantity`, `timestamp`, `latency_ticks` |
| `Trade` | `id`, `symbol`, `buyer_id`, `seller_id`, `price`, `quantity`, `timestamp`, `tick` |
| `OrderStatus` | `Pending`, `Queued { execute_at }`, `PartialFill { filled }`, `Filled`, `Cancelled` |

### Order with Latency

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub agent_id: AgentId,
    pub symbol: Symbol,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Quantity,
    pub timestamp: Timestamp,
    pub latency_ticks: u64,  // Delay before matching (0 = instant)
}
```

### OrderId Assignment

**Critical:** Agents create orders with `id: 0` as a placeholder. The `Market::submit_order()` method assigns the real OrderId atomically:

```rust
// In Market struct
order_id_counter: AtomicU64,

impl Market {
    pub fn next_order_id(&self) -> OrderId {
        self.order_id_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    pub fn submit_order(&mut self, mut order: Order) -> OrderId {
        order.id = self.next_order_id();
        // ... queue or process order
        order.id
    }
}
```

This ensures thread-safe ID generation when using parallel agent execution (Rayon).

## Market Data Types

| Type | Fields |
|------|--------|
| `Candle` | `symbol`, `open: Price`, `high: Price`, `low: Price`, `close: Price`, `volume`, `timestamp` |
| `BookLevel` | `price: Price`, `quantity` |
| `BookSnapshot` | `bids: Vec<BookLevel>`, `asks: Vec<BookLevel>`, `timestamp` |
| `TickData` | `symbol`, `price: Price`, `volume`, `timestamp` |

## Events

| Type | Variants/Fields |
|------|-----------------|
| `Sector` | `Tech`, `Energy`, `Finance`, `Healthcare`, `Consumer`, `Industrials`, `Materials`, `Utilities`, `RealEstate`, `Communications` |
| `EventType` | `Macro`, `Sector { sector }`, `Company { symbol }` |
| `Sentiment` | `f64` (-1.0 to +1.0) — Note: f64 OK here, not money |
| `NewsEvent` | `id`, `event_type`, `sentiment`, `duration_ticks`, `timestamp` |

## Quant Types

| Type | Fields |
|------|--------|
| `IndicatorType` | `SMA`, `EMA`, `RSI`, `MACD`, `BollingerBands`, `ATR` |
| `IndicatorValue` | `indicator_type`, `symbol`, `value: f64`, `timestamp` |
| `FactorType` | `Momentum`, `Value`, `Volatility`, `MeanReversion` |
| `FactorScore` | `factor_type`, `symbol`, `score: f64` |
| `Signal` | `StrongBuy`, `Buy`, `Neutral`, `Sell`, `StrongSell` |

Note: Indicator/factor values are `f64` — they're statistical, not monetary.

## Risk Types

| Type | Fields |
|------|--------|
| `RiskMetrics` | `var_95: f64`, `var_99: f64`, `sharpe: f64`, `sortino: f64`, `max_drawdown: f64`, `volatility: f64` |
| `PositionLimits` | `max_short: Quantity`, `max_sector_exposure: f64`, `max_drawdown: f64` |
| `RiskViolation` | `DrawdownLimit`, `SectorExposure`, `InsufficientCash`, `ShortLimitExceeded`, `NoBorrowAvailable`, `InsufficientShares` |
| `ShortSellingConfig` | `enabled: bool`, `borrow_rate_bps: u32`, `locate_required: bool` |
| `BorrowPosition` | `symbol`, `quantity`, `borrow_rate`, `borrowed_at: Tick` |

**Note on Long Positions:** There is no `max_long` limit. Long positions are naturally constrained by:
1. **Cash available** — can't buy what you can't afford
2. **Shares outstanding** — can't buy more shares than exist (see `SymbolConfig.shares_outstanding`)

## Portfolio Types

| Type | Fields |
|------|--------|
| `Position` | `symbol`, `quantity`, `avg_cost: Price` |
| `Portfolio` | `agent_id`, `cash: Cash`, `positions: Vec<Position>` |
| `PnL` | `realized: Cash`, `unrealized: Cash`, `total: Cash` |

## Configuration Types

| Type | Fields |
|------|--------|
| `SymbolConfig` | `symbol`, `sector`, `initial_price: Price`, `tick_size: Price`, `lot_size: Quantity`, `shares_outstanding: Quantity` |
| `AgentConfig` | `id`, `initial_cash: Cash`, `strategy_name`, `strategy_params: serde_json::Value`, `tier: AgentTier` |
| `AgentTier` | `Smart`, `Reactive`, `Background` |

### SymbolConfig

```rust
/// Configuration for a tradeable symbol, including initial state and trading constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolConfig {
    pub symbol: Symbol,
    pub sector: Sector,
    pub initial_price: Price,
    /// Minimum price increment (e.g., 0.01 for most equities)
    pub tick_size: Price,
    /// Minimum order quantity (e.g., 1 for most equities)
    pub lot_size: Quantity,
    /// Total shares that exist for this symbol (limits max purchasable)
    pub shares_outstanding: Quantity,
}
```

### AgentConfig

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AgentTier { Smart, Reactive, Background }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: AgentId,
    pub initial_cash: Cash,
    pub strategy_name: String,
    pub strategy_params: serde_json::Value,
    pub tier: AgentTier,
}
```

## Game Save/Resume Types

| Type | Fields |
|------|--------|
| `GameId` | `Uuid` |
| `GameStatus` | `InProgress`, `Paused`, `Completed` |
| `GameSnapshot` | `game_id`, `tick`, `timestamp`, `market_state`, `agents`, `price_history`, `config` |
| `AgentSnapshot` | `agent_id`, `agent_type`, `portfolio`, `strategy_state` |

### GameSnapshot

```rust
/// Complete simulation state for save/resume.
/// Saved on: auto-save (every N ticks), manual save, save & exit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameSnapshot {
    pub game_id: GameId,
    pub tick: Tick,
    pub saved_at: Timestamp,
    
    // Market state
    pub prices: HashMap<Symbol, Price>,
    pub books: HashMap<Symbol, BookSnapshot>,
    
    // Agent states
    pub agents: Vec<AgentSnapshot>,
    
    // Rolling window history (needed for indicators on resume)
    pub price_history: HashMap<Symbol, Vec<Price>>,  // Last N prices
    
    // Game configuration
    pub config: GameConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub agent_id: AgentId,
    pub agent_type: String,  // Strategy name for reconstruction
    pub portfolio: Portfolio,
    pub strategy_params: serde_json::Value,  // Strategy-specific state
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub symbols: Vec<SymbolConfig>,
    pub auto_save_interval: Tick,  // Default: 10_000
    pub duration_ticks: Option<Tick>,
    pub ai_opponents: Vec<AgentConfig>,
}
```

## Fixed-Point Helpers & Financial Precision

```rust
use crate::{Price, Cash, PRICE_SCALE};

pub mod price_utils {
    use super::*;
    
    pub const ZERO: Price = Price(0);
    
    /// Price precision: 4 decimal places
    /// 10000 = $1.00, 1 = $0.0001
    pub const PRECISION: u32 = 4;
    
    pub fn from_cents(cents: i64) -> Price {
        Price(cents * 100) // 100 cents * 100 = 10000 = $1.00
    }
    
    pub fn to_cents(price: Price) -> i64 {
        price.0 / 100
    }
    
    pub fn mid_price(bid: Price, ask: Price) -> Price {
        Price((bid.0 + ask.0) / 2)
    }
    
    pub fn round_cash(cash: Cash) -> Cash {
        cash.round_dp_with_strategy(CASH_PRECISION, ROUNDING_MODE)
    }
}
```

**Critical Financial Precision Notes:**
- All monetary calculations MUST use explicit rounding with `ROUNDING_MODE`
- Prices round to 4 decimal places, cash to 2 decimal places
- Never use implicit floating-point conversions for ledger updates

## Agent Tier Types

| Type | Purpose |
|------|---------|
| `TickFrequency` | `EveryTick`, `EveryN(u64)`, `Probabilistic(f64)` — f64 is probability per tick in range [0.0, 1.0] |

### TickFrequency Implementation

```rust
impl TickFrequency {
    /// Returns true if agent should run this tick.
    /// Uses provided RNG to ensure determinism.
    pub fn should_run(&self, tick: Tick, rng: &mut impl Rng) -> bool {
        match self {
            Self::EveryTick => true,
            Self::EveryN(n) => tick % n == 0,
            Self::Probabilistic(p) => rng.gen::<f64>() < *p,
        }
    }
}
```
| `WakeCondition` | `PriceCross`, `NewsEvent`, `PortfolioChange`, `TimeInterval` |
| `ReactiveStrategyType` | Enum: `ThresholdBuyer`, `ThresholdSeller`, `NewsReactor`, `MomentumFollower` |
| `ReactivePortfolio` | `SingleSymbol { symbol, quantity, avg_cost, cash }` or `Full(Portfolio)` — single-symbol variant uses ~150 bytes vs ~1KB for full |
| `MarketRegime` | `Calm`, `Normal`, `Volatile`, `Crisis` — presets for background config |

### Context Types

`StrategyContext` is the full context provided to Tier 1 smart agents:

```rust
pub struct StrategyContext<'a> {
    pub view: &'a dyn MarketView,
    pub portfolio: &'a Portfolio,
    pub indicators: &'a IndicatorCache,
    pub factors: &'a FactorScores,
    pub risk: &'a RiskMetrics,
    pub events: &'a [NewsEvent],
    pub current_tick: Tick,
}
```

`LightweightContext` is a minimal subset used by Tier 2 reactive agents:

```rust
pub struct LightweightContext<'a> {
    pub symbol: &'a Symbol,              // Primary symbol (for single-symbol agents)
    pub price: Price,                    // Current price
    pub portfolio: &'a ReactivePortfolio,
    pub triggered_by: &'a WakeCondition, // What woke this agent
    pub current_tick: Tick,
}
```

## Simulation Types

| Type | Purpose |
|------|---------|
| `SimulationPreset` | `QuickTest`, `Standard`, `Stress`, `Training`, `Custom(...)` |
| `TickMetrics` | Wall time, order counts by tier, matching time, etc. |

### SimulationPreset Configurations

| Preset | Tier 1 | Tier 2 | Tier 3 | Ticks | Latency | Warmup |
|--------|--------|--------|--------|-------|---------|--------|
| `QuickTest` | 10 | 100 | 1,000 | 1,000 | Off | 50 |
| `Standard` | 100 | 5,000 | 50,000 | 10,000 | On | 500 |
| `Stress` | 100 | 10,000 | 100,000 | 50,000 | On | 1,000 |
| `Training` | 10 | 100 | 1,000 | 100,000 | On | 500 |
| `Custom(...)` | User-defined | User-defined | User-defined | User-defined | User-defined | User-defined |

---

# Part 6: Trait Boundaries

## sim-core Exports

| Export | Kind | Methods |
|--------|------|---------|
| `MarketView` | trait | `price(&self, symbol) → Option<Price>` |
| | | `book(&self, symbol) → Option<&BookSnapshot>` |
| | | `volume(&self, symbol) → Quantity` |
| | | `last_trade(&self, symbol) → Option<&Trade>` |
| | | `candles(&self, symbol, n) → Vec<Candle>` |
| | | `symbols(&self) → Vec<Symbol>` |
| | | `current_tick(&self) → Tick` |
| `OrderBook` | struct | `new()`, `add_order()`, `cancel_order()`, `best_bid()`, `best_ask()`, `spread()`, `depth()` |
| `MatchingEngine` | struct | `match_order(&mut book, order) → MatchResult` |
| `PendingOrderQueue` | struct | `push(execute_at, order)`, `drain_ready(current_tick) → Vec<Order>` |
| `Market` | struct | `new()`, `submit_order()`, `cancel_order()`, `tick()`, `snapshot()` |
| `MatchResult` | struct | `trades: Vec<Trade>`, `status: OrderStatus` |

## quant Exports

| Export | Kind | Methods |
|--------|------|---------|
| `Indicator` | trait | `indicator_type(&self) → IndicatorType` |
| | | `calculate(&self, candles: &[Candle]) → f64` |
| | | `required_periods(&self) → usize` |
| `FactorModel` | trait | `factor_type(&self) → FactorType` |
| | | `score(&self, symbol, candles: &[Candle]) → f64` |
| `RiskCalculator` | trait | `compute(&self, portfolio, prices, history) → RiskMetrics` |
| `ExecutionAlgo` | trait | `slice(&self, order, view) → Vec<Order>` |
| `IndicatorEngine` | struct | `register()`, `compute_all(view) → IndicatorCache` |
| `IndicatorCache` | struct | `get(symbol, indicator_type) → Option<f64>`, `get_or_compute()` |

### Indicator Caching Strategy

```rust
impl IndicatorCache {
    /// Only compute if not cached for current tick.
    /// Strategies declare required_indicators(); these are computed lazily on first access.
    pub fn get_or_compute(
        &mut self, 
        symbol: &Symbol, 
        indicator: IndicatorType, 
        tick: Tick, 
        candles: &[Candle]
    ) -> f64 {
        if let Some((cached_tick, value)) = self.cache.get(&(symbol.clone(), indicator)) {
            if *cached_tick == tick {
                return *value;
            }
        }
        let value = self.compute(symbol, indicator, candles);
        self.cache.insert((symbol.clone(), indicator), (tick, value));
        value
    }
}
```
| `FactorEngine` | struct | `register()`, `score_all(view) → FactorScores` |
| `FactorScores` | struct | `get(symbol, factor_type) → Option<f64>` |

## news Exports

| Export | Kind | Methods |
|--------|------|---------|
| `NewsGenerator` | struct | `new(config, seed)`, `tick() → Vec<NewsEvent>` |
| `SectorModel` | struct | `stocks_in_sector(sector) → Vec<Symbol>` |
| | | `sector_for_stock(symbol) → Sector` |
| `NewsConfig` | struct | `macro_frequency`, `sector_frequency`, `company_frequency` |

## agents Exports

| Export | Kind | Methods |
|--------|------|---------|
| `Strategy` | trait | `name(&self) → &str` |
| | | `required_indicators(&self) → Vec<IndicatorType>` |
| | | `required_factors(&self) → Vec<FactorType>` |
| | | `decide(&mut self, ctx: &StrategyContext) → Vec<Order>` |
| | | `latency_ticks(&self) → u64` |
| | | `tick_frequency(&self) → TickFrequency` |
| `StrategyContext` | struct | `view`, `portfolio`, `indicators`, `factors`, `risk`, `events`, `current_tick` |
| `LightweightContext` | struct | `symbol`, `price`, `portfolio`, `triggered_by`, `current_tick` |
| `Agent` | struct | `id`, `strategy`, `portfolio`, `risk_limits` |
| `ReactiveAgent` | struct | `id`, `strategy_type`, `portfolio`, `wake_conditions` |
| `TieredOrchestrator` | struct | `spawn_smart()`, `spawn_reactive()`, `configure_background()`, `tick() → Vec<Order>` |
| `StrategyRegistry` | struct | `register()`, `create(name, config) → Box<dyn Strategy>` |
| `WakeConditionIndex` | struct | `register()`, `find_triggered(tick, prices, events) → Vec<AgentId>` |
| `BackgroundAgentPool` | struct | `new(config)`, `tick(context) → Vec<Order>`, `adjust_sentiment()` |
| `BackgroundPoolAccounting` | struct | `record_fill()`, `computed_pnl() → Cash`, `sanity_check() → bool` |

**Multi-Symbol Reactive Agent Behavior:** A reactive agent with `ReactivePortfolio::Full` that monitors multiple symbols is woken ONCE PER SYMBOL when wake conditions trigger. The `LightweightContext` provides the single triggered symbol. This maintains the "lightweight" promise—agents don't receive a batch of all triggered symbols at once.

### WakeConditionIndex Structure

```rust
/// Maps wake conditions to agents that should be notified.
/// Enables O(log n) lookup for triggered agents.
pub struct WakeConditionIndex {
    /// Price threshold crossings: (symbol, price) → agents to wake
    price_crosses: BTreeMap<(Symbol, OrderedPrice), Vec<AgentId>>,
    /// News event subscriptions: event_type → agents to wake
    news_subscriptions: HashMap<EventType, Vec<AgentId>>,
    /// Time-based intervals: min-heap of (wake_tick, agent_id)
    time_intervals: BinaryHeap<Reverse<(Tick, AgentId)>>,
}

impl WakeConditionIndex {
    /// Register a new wake condition for an agent
    pub fn register(&mut self, agent_id: AgentId, condition: WakeCondition);
    
    /// Remove a wake condition (agent adapts to market)
    pub fn unregister(&mut self, agent_id: AgentId, condition: &WakeCondition);
    
    /// Update a condition's parameters without full re-registration
    pub fn update_threshold(&mut self, agent_id: AgentId, old: &WakeCondition, new: WakeCondition);
}
```

### Parametric Condition Updates

Reactive agents can adapt to changing market conditions by updating their wake conditions at runtime, rather than being destroyed and recreated:

```rust
/// Deferred condition update (applied after tick completes)
pub struct ConditionUpdate {
    pub agent_id: AgentId,
    pub remove: Vec<WakeCondition>,
    pub add: Vec<WakeCondition>,
}

impl ReactiveAgent {
    /// Request condition changes (collected during tick, applied after)
    pub fn request_condition_update(&self, ctx: &LightweightContext) -> Option<ConditionUpdate> {
        // Example: price threshold became stale, update to new level
        if self.threshold_stale(ctx.price) {
            Some(ConditionUpdate {
                agent_id: self.id,
                remove: vec![self.current_condition.clone()],
                add: vec![WakeCondition::PriceCross {
                    symbol: ctx.symbol.clone(),
                    threshold: self.compute_new_threshold(ctx.price),
                    direction: CrossDirection::Below,
                }],
            })
        } else {
            None
        }
    }
}
```

## simulation Exports

| Export | Kind | Methods |
|--------|------|---------|
| `SimulationRunner` | struct | `new(config)`, `tick() → TickResult`, `run(n_ticks)`, `submit_order()` |
| `SimulationConfig` | struct | `initial_agents`, `initial_cash`, `symbols`, `news_config`, `seed` |
| `TickResult` | struct | `tick_number`, `trades`, `events`, `indicators` |
| `SimulationHook` | trait | `on_tick(&mut self, result: &TickResult)` |
| `SimulationBuilder` | struct | `market()`, `agents()`, `news()`, `hook()`, `build()` |

## gym Exports

| Export | Kind | Methods |
|--------|------|---------|
| `Env` | trait | `step(action) → StepResult`, `reset() → Observation`, `action_space()`, `observation_space()` |
| `ObservationBuilder` | trait | `build(ctx) → Vec<f64>`, `shape() → Vec<usize>`, `contract() → ObservationContract` |
| `ObservationContract` | struct | `shape`, `features: Vec<FeatureSpec>` |
| `FeatureSpec` | struct | `name`, `index`, `normalization` |
| `RewardFunction` | trait | `compute(prev, action, curr) → f64` |
| `TradingEnv` | struct | Implements `Env` |
| `TradingEnvBuilder` | struct | `simulation()`, `observation()`, `reward()`, `build()` |
| `StepResult` | struct | `observation`, `reward`, `done`, `info` |
| `Action` | enum | `Hold`, `Buy { symbol, quantity }`, `Sell { symbol, quantity }` |

## storage Exports

| Export | Kind | Methods |
|--------|------|---------|
| `TradeStore` | struct | `append()`, `query_range()`, `replay_from()` |
| `CandleStore` | struct | `write()`, `query()`, `aggregate()`, `maybe_aggregate_hourly()`, `aggregate_daily()` |
| `PortfolioStore` | struct | `get()`, `update()`, `rebuild_from_trades()` |

### PortfolioStore Rebuild Algorithm

```rust
impl PortfolioStore {
    /// Rebuilds portfolio state by replaying all trades for an agent.
    /// Uses WEIGHTED AVERAGE for cost basis calculation.
    /// 
    /// Algorithm:
    /// 1. Start with initial_cash from agents table
    /// 2. For each trade in chronological order:
    ///    - If buyer_id == agent_id: 
    ///        new_avg_cost = (old_qty * old_avg + trade_qty * trade_price) / (old_qty + trade_qty)
    ///        Subtract cost from cash, add to position
    ///    - If seller_id == agent_id:
    ///        Add proceeds to cash, reduce position (avg_cost unchanged)
    /// 3. Return reconstructed Portfolio
    pub fn rebuild_from_trades(&self, agent_id: AgentId) -> Portfolio;
}
```

**Cost Basis Method:** Weighted average (not FIFO/LIFO) for simplicity and consistency.
| `RiskStore` | struct | `store_metrics()`, `query_history()` |
| `SnapshotStore` | struct | `save()`, `restore()` |
| `GameSnapshotStore` | struct | `save_game()`, `load_game()`, `list_saves()` |
| `TradeLogStore` | struct | `append()`, `query_game()` |
| `Storage` | struct | Aggregates all stores |
| `PersistenceHook` | struct | Implements `SimulationHook` |

## services/common Exports

| Export | Kind | Methods |
|--------|------|---------|
| `SimulationBridge` | struct | `submit_order()`, `tick()`, `get_book()`, `spawn_agent()`, etc. |
| `SimCommand` | enum | All commands that can be sent to simulation thread |
| `run_simulation_thread` | fn | Blocking loop that processes commands |

---

# Part 7: Sync/Async Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Async Runtime (tokio)                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │exchange │ │ agents  │ │portfolio│ │ chatbot │  ...      │
│  │ service │ │ service │ │ service │ │ service │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│       └───────────┴───────────┴───────────┘                 │
│                         │                                   │
│              ┌──────────▼──────────┐                        │
│              │  SimulationBridge   │ (mpsc channels)        │
└──────────────┴──────────┬──────────┴────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              Simulation Thread (std::thread)                │
│  SimulationRunner (sync)                                    │
│    └─ TieredOrchestrator                                    │
│         ├─ Tier 1: Smart agents (parallel via rayon)        │
│         ├─ Tier 2: Reactive agents (wake index lookup)      │
│         └─ Tier 3: Background pool (statistical)            │
└─────────────────────────────────────────────────────────────┘
```

**Bridge commands:** `SubmitOrder`, `CancelOrder`, `Tick`, `GetBook`, `GetPortfolio`, `SpawnAgent`, `Snapshot`, `GetMetrics`

## Error Handling Strategy

| Crate | Error Type | Library |
|-------|------------|----------|
| All crates | Domain errors | `thiserror` for typed errors |
| Application layer | Error propagation | `anyhow` for context chaining |
| Services | HTTP errors | Custom `ApiError` with status codes |

**Pattern:**
- Each crate defines its own `Error` enum with `#[derive(thiserror::Error)]`
- Errors implement `From` for composability across crate boundaries
- Services convert domain errors to `ApiError` at handler boundaries
- Simulation thread errors sent back via `oneshot` channel as `Result<T, SimError>`

```rust
// Example: crates/sim-core/error.rs
#[derive(Debug, thiserror::Error)]
pub enum SimCoreError {
    #[error("unknown symbol: {0}")]
    UnknownSymbol(Symbol),
    #[error("insufficient liquidity for {symbol}")]
    InsufficientLiquidity { symbol: Symbol },
    #[error("order not found: {0}")]
    OrderNotFound(OrderId),
}
```

## Bridge Implementation

```rust
// crates/services/common/bridge.rs

use tokio::sync::{mpsc, oneshot};

pub enum SimCommand {
    SubmitOrder { order: Order, respond: oneshot::Sender<Result<OrderId, SimError>> },
    CancelOrder { id: OrderId, respond: oneshot::Sender<Result<bool, SimError>> },
    Tick { respond: oneshot::Sender<Result<TickResult, SimError>> },
    GetBook { symbol: Symbol, respond: oneshot::Sender<Result<BookSnapshot, SimError>> },
    GetPortfolio { agent_id: AgentId, respond: oneshot::Sender<Result<Portfolio, SimError>> },
    SpawnAgent { config: AgentConfig, respond: oneshot::Sender<Result<AgentId, SimError>> },
    Snapshot { respond: oneshot::Sender<Result<MarketSnapshot, SimError>> },
}

pub struct SimulationBridge {
    command_tx: mpsc::Sender<SimCommand>,
}

impl SimulationBridge {
    pub async fn submit_order(&self, order: Order) -> Result<OrderId, Error> {
        let (tx, rx) = oneshot::channel();
        self.command_tx.send(SimCommand::SubmitOrder { order, respond: tx }).await?;
        Ok(rx.await?)
    }
    
    // ... other methods
}

// Runs in dedicated thread
pub fn run_simulation_thread(
    mut runner: SimulationRunner,
    mut command_rx: mpsc::Receiver<SimCommand>,
) {
    while let Some(cmd) = command_rx.blocking_recv() {
        match cmd {
            SimCommand::SubmitOrder { order, respond } => {
                let id = runner.submit_order(order);
                let _ = respond.send(id);
            }
            SimCommand::Tick { respond } => {
                let result = runner.tick();
                let _ = respond.send(result);
            }
            // ... other commands
        }
    }
}
```

---

# Part 8: Observation Contract System

## Purpose

Ensure observation vectors are identical in Rust (production) and Python (training).

## Rust Implementation

```rust
// crates/gym/observation/contract.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationContract {
    pub version: u32,
    pub shape: Vec<usize>,
    pub features: Vec<FeatureSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSpec {
    pub name: String,
    pub start_index: usize,
    pub end_index: usize,
    pub normalization: Normalization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Normalization {
    None,
    MinMax { min: f64, max: f64 },
    ZScore { mean: f64, std: f64 },
    LogReturn,
}

impl ObservationContract {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }
    
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_json())?;
        Ok(())
    }
}

// Each ObservationBuilder must provide its contract
pub trait ObservationBuilder: Send + Sync {
    fn build(&self, ctx: &ObservationContext) -> Vec<f64>;
    fn contract(&self) -> ObservationContract;
}
```

## Python Implementation

```python
# python/quant_trading_gym/observation.py

import json
import numpy as np
from dataclasses import dataclass
from typing import List
from enum import Enum

class Normalization(Enum):
    NONE = "None"
    MIN_MAX = "MinMax"
    Z_SCORE = "ZScore"
    LOG_RETURN = "LogReturn"

@dataclass
class FeatureSpec:
    name: str
    start_index: int
    end_index: int
    normalization: Normalization
    norm_params: dict = None

@dataclass
class ObservationContract:
    version: int
    shape: List[int]
    features: List[FeatureSpec]
    
    @classmethod
    def load(cls, path: str) -> 'ObservationContract':
        with open(path) as f:
            data = json.load(f)
        # Parse and return contract
        ...
    
    def validate_observation(self, obs: np.ndarray) -> bool:
        """Verify observation matches contract."""
        return obs.shape == tuple(self.shape)
```

## Parity Test

```python
# tests/parity/test_observation_parity.py

import numpy as np
from quant_trading_gym import TradingEnv
from quant_trading_gym.observation import ObservationContract

def test_observation_parity():
    """Verify Rust and Python produce identical observations."""
    
    # Create env with known seed
    env = TradingEnv(agents=100, stocks=5, seed=42)
    
    # Get contract
    contract = env.observation_contract()
    
    # Reset and get observation
    obs_rust = env.reset()
    
    # Build same observation in Python using raw market data
    market_data = env.get_raw_market_data()
    obs_python = build_observation_python(market_data, contract)
    
    # Compare
    np.testing.assert_array_almost_equal(
        obs_rust, 
        obs_python, 
        decimal=10,
        err_msg="Rust and Python observations don't match!"
    )

def test_contract_schema():
    """Verify contract has all required fields."""
    env = TradingEnv(agents=100, stocks=5, seed=42)
    contract = env.observation_contract()
    
    assert contract.version >= 1
    assert len(contract.shape) > 0
    assert len(contract.features) > 0
    
    # Verify indices are contiguous
    expected_size = contract.shape[0]
    actual_size = sum(f.end_index - f.start_index for f in contract.features)
    assert expected_size == actual_size
```

---

# Part 9: Order Latency System

## Purpose

Prevent look-ahead bias in RL training by simulating realistic order processing delays.

## Implementation in sim-core

```rust
// crates/sim-core/pending_orders.rs

use std::collections::BinaryHeap;
use std::cmp::Reverse;

pub struct PendingOrderQueue {
    queue: BinaryHeap<Reverse<(Tick, Order)>>,
}

impl PendingOrderQueue {
    pub fn new() -> Self {
        Self { queue: BinaryHeap::new() }
    }
    
    pub fn push(&mut self, execute_at: Tick, order: Order) {
        self.queue.push(Reverse((execute_at, order)));
    }
    
    pub fn drain_ready(&mut self, current_tick: Tick) -> Vec<Order> {
        let mut ready = Vec::new();
        while let Some(Reverse((tick, _))) = self.queue.peek() {
            if *tick <= current_tick {
                let Reverse((_, order)) = self.queue.pop().unwrap();
                ready.push(order);
            } else {
                break;
            }
        }
        ready
    }
    
    pub fn len(&self) -> usize {
        self.queue.len()
    }
}
```

```rust
// crates/sim-core/market.rs

pub struct Market {
    books: HashMap<Symbol, OrderBook>,
    pending: PendingOrderQueue,
    current_tick: Tick,
    // ...
}

impl Market {
    pub fn submit_order(&mut self, order: Order) -> OrderId {
        let execute_at = self.current_tick + order.latency_ticks;
        
        if order.latency_ticks == 0 {
            // Immediate execution
            self.process_order(order.clone());
        } else {
            // Queue for later
            self.pending.push(execute_at, order.clone());
        }
        
        order.id
    }
    
    pub fn tick(&mut self) -> TickResult {
        self.current_tick += 1;
        
        // Process orders that are now ready
        let ready_orders = self.pending.drain_ready(self.current_tick);
        let mut all_trades = Vec::new();
        
        for order in ready_orders {
            let result = self.process_order(order);
            all_trades.extend(result.trades);
        }
        
        TickResult {
            tick: self.current_tick,
            trades: all_trades,
            // ...
        }
    }
    
    fn process_order(&mut self, order: Order) -> MatchResult {
        let book = self.books.get_mut(&order.symbol).unwrap();
        self.matching_engine.match_order(book, order)
    }
}
```

## Strategy Latency Configuration

```rust
// crates/agents/traits.rs

pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    
    /// Latency in ticks for orders from this strategy.
    /// Override to simulate different agent types.
    fn latency_ticks(&self) -> u64 {
        1  // Default: 1 tick delay
    }
    
    fn decide(&mut self, ctx: &StrategyContext) -> Vec<Order>;
    
    // ...
}

// High-frequency market maker might have low latency
impl Strategy for MarketMakerStrategy {
    fn latency_ticks(&self) -> u64 { 0 }
    // ...
}

// Retail trader has higher latency
impl Strategy for RetailStrategy {
    fn latency_ticks(&self) -> u64 { 3 }
    // ...
}
```

## Configuration

```rust
// crates/simulation/config.rs

pub struct SimulationConfig {
    // ...
    
    /// Default latency for agents that don't specify their own
    pub default_latency_ticks: u64,
    
    /// Whether to enforce latency (false = all orders instant)
    pub enable_latency: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            default_latency_ticks: 1,
            enable_latency: true,
            // ...
        }
    }
}
```

**Latency Precedence:**
1. Strategy's `latency_ticks()` method takes precedence if non-zero
2. Falls back to `SimulationConfig::default_latency_ticks`
3. If `enable_latency` is false, all orders execute immediately regardless of strategy settings

---

# Part 10: Services Layer

## Service Architecture (Consolidated)

4 services instead of 8, same API surface:

```
┌─────────────────────────────────────────────────────────────┐
│                      SIMULATION                             │
│  (sync, computes everything for agents)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ broadcast
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │    DATA     │   │    GAME     │   │   STORAGE   │
  │   SERVICE   │   │   SERVICE   │   │   SERVICE   │
  │   :8001     │   │   :8002     │   │   :8003     │
  │             │   │             │   │             │
  │ /analytics/*│   │ /game/*     │   │ /storage/*  │
  │ /portfolio/*│   │ WebSocket   │   │ Snapshots   │
  │ /risk/*     │   │ Sessions    │   │ Trade log   │
  │ /news/*     │   │ Time ctrl   │   │ Queries     │
  │             │   │ Orders      │   │             │
  └─────────────┘   └─────────────┘   └─────────────┘
         │                 │
         └────────┬────────┘
                  ▼
           ┌─────────────┐
           │   CHATBOT   │
           │   :8004     │
           │  NLP → API  │
           └─────────────┘
```

## Service Overview

| Service | Port | Responsibility | Sync/Async |
|---------|------|----------------|------------|
| data | 8001 | Analytics, portfolio, risk, news queries | Async (subscribes to sim) |
| game | 8002 | WebSocket, sessions, time control, orders, BFF | Async (bridges to sync) |
| storage | 8003 | Snapshots, trade log, historical queries | Async |
| chatbot | 8004 | Natural language interface | Async |

**Consolidation Rationale:**
- Analytics, portfolio, risk, news all have similar scaling needs (query-heavy, read-mostly)
- Single Data service with internal modules: `analytics.rs`, `portfolio.rs`, `risk.rs`, `news.rs`
- Same API surface as 8 services, fewer deployment units

**Note:** All services should be configurable via environment variables (e.g., `DATA_PORT=8001`). Consider using a shared config crate or `.env` file for service discovery.

## Data Service Internal Structure

```rust
// crates/services/data/src/
├── lib.rs
├── main.rs
├── analytics.rs   // /analytics/* handlers
├── portfolio.rs   // /portfolio/* handlers
├── risk.rs        // /risk/* handlers
└── news.rs        // /news/* handlers
```

## Game Service Architecture

```rust
// crates/services/game/main.rs

use axum::{Router, routing::{get, post}};
use std::sync::Arc;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    // Create bridge channels
    let (cmd_tx, cmd_rx) = mpsc::channel(1000);
    
    // Start simulation in dedicated thread
    let runner = SimulationBuilder::new()
        .config(config)
        .build();
    
    std::thread::spawn(move || {
        run_simulation_thread(runner, cmd_rx);
    });
    
    // Create bridge
    let bridge = Arc::new(SimulationBridge::new(cmd_tx));
    
    // Build routes
    let app = Router::new()
        .route("/order", post(handlers::submit_order))
        .route("/order/:id", delete(handlers::cancel_order))
        .route("/book/:symbol", get(handlers::get_book))
        .route("/price/:symbol", get(handlers::get_price))
        .with_state(bridge);
    
    // Run server
    axum::Server::bind(&"0.0.0.0:8001".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

## API Endpoints

### data service (/analytics/*)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | /analytics/candles/{symbol} | `?interval=1m&limit=100` | `Vec<Candle>` |
| GET | /analytics/indicators/{symbol} | `?types=RSI,MACD` | `Vec<IndicatorValue>` |
| GET | /analytics/factors | - | `Vec<FactorScore>` |
| GET | /analytics/correlation | - | `CorrelationMatrix` |

### data service (/portfolio/*)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | /portfolio/{agent_id} | - | `Portfolio` |
| GET | /portfolio/{agent_id}/pnl | - | `PnL` |
| GET | /portfolio/{agent_id}/history | `?since=timestamp` | `Vec<Trade>` |
| GET | /portfolio/leaderboard | `?limit=n` | `Vec<LeaderboardEntry>` |

### data service (/risk/*)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | /risk/metrics/{agent_id} | - | `RiskMetrics` |
| GET | /risk/exposure/{agent_id} | - | `ExposureReport` |
| POST | /risk/limits/{agent_id} | `PositionLimits` | `()` |
| GET | /risk/alerts | - | `Vec<RiskViolation>` |

### data service (/news/*)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | /news/feed | `?since=timestamp` | `Vec<NewsEvent>` |
| GET | /news/scheduled | - | `Vec<NewsEvent>` |

### game service

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | /game/order | `Order` | `OrderId` |
| DELETE | /game/order/{id} | - | `OrderStatus` |
| GET | /game/book/{symbol} | - | `BookSnapshot` |
| GET | /game/price/{symbol} | - | `Price` |
| GET | /game/dashboard | - | `DashboardSnapshot` |
| WS | /game/stream | - | Real-time tick updates |
| POST | /game/time | `{ speed: "slow" }` | `()` |
| POST | /game/step | - | `TickResult` |
| POST | /game/save | - | `SnapshotId` |
| POST | /game/session | `GameConfig` | `SessionId` |
| GET | /game/session/{id} | - | `SessionState` |

### storage service

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | /storage/snapshots | `GameSnapshot` | `SnapshotId` |
| GET | /storage/snapshots/{game_id} | - | `Vec<SnapshotMeta>` |
| GET | /storage/snapshots/{game_id}/{tick} | - | `GameSnapshot` |
| GET | /storage/trades/{game_id} | `?since=tick` | `Vec<Trade>` |
| GET | /storage/games | - | `Vec<GameMeta>` |

### chatbot service

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | /chat | `ChatRequest` | `ChatResponse` |
| GET | /chat/history | - | `Vec<Message>` |

### Health Check (All Services)

Every service MUST expose:

| Method | Path | Response |
|--------|------|----------|
| GET | /health | `{ "status": "healthy", "service": "<name>", "version": "<version>" }` |

**Health Check Requirements:**
- Must respond within **5 seconds** or return 503
- Returns **503 Service Unavailable** if simulation thread is unresponsive
- Checks channel connectivity to simulation thread
- Used by Docker health checks, load balancers, and Kubernetes probes

Health checks are essential for Docker orchestration, load balancers, and Kubernetes deployments.

---

# Part 11: Human Player Interface

## The Problem

AI agents operate at tick speeds (potentially <10ms per decision). Humans cannot compete at this pace. Without careful design, the "game" becomes unwatchable chaos where the human has already lost before comprehending the market state.

## Design Requirements

| Requirement | Purpose |
|-------------|----------|
| Time Controls | Pause, step, speed adjustment so humans can think |
| Quant Dashboard | Expose the same indicators/factors AI agents see |
| Decision Support | Order entry, position management, risk alerts |
| Information Parity | Humans must not be informationally disadvantaged vs AI |

## Time Control System

### Simulation Speed Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| Paused | Simulation frozen, full inspection | Analysis, order planning |
| Step | Advance exactly 1 tick per click | Debugging, learning |
| Slow | 1 tick per second | Comfortable human play |
| Normal | 10 ticks per second | Engaged play |
| Fast | 100 ticks per second | Skip boring periods |
| Max | As fast as possible | AI-only training |

### Implementation

```rust
// crates/simulation/time_control.rs

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimulationSpeed {
    Paused,
    Step,           // Manual advance
    Slow,           // 1 tick/sec
    Normal,         // 10 ticks/sec  
    Fast,           // 100 ticks/sec
    Max,            // Unbounded
}

pub struct TimeController {
    speed: SimulationSpeed,
    tick_interval_ms: u64,
    pending_step: bool,
}

impl TimeController {
    pub fn should_tick(&mut self) -> bool {
        match self.speed {
            SimulationSpeed::Paused => false,
            SimulationSpeed::Step => {
                let should = self.pending_step;
                self.pending_step = false;
                should
            }
            _ => true, // Timer-based in game loop
        }
    }
    
    pub fn request_step(&mut self) {
        self.pending_step = true;
    }
    
    pub fn tick_interval(&self) -> Option<Duration> {
        match self.speed {
            SimulationSpeed::Paused | SimulationSpeed::Step => None,
            SimulationSpeed::Slow => Some(Duration::from_millis(1000)),
            SimulationSpeed::Normal => Some(Duration::from_millis(100)),
            SimulationSpeed::Fast => Some(Duration::from_millis(10)),
            SimulationSpeed::Max => Some(Duration::from_millis(0)),
        }
    }
}
```

### WebSocket Commands

| Command | Effect |
|---------|--------|
| `{ "type": "pause" }` | Freeze simulation |
| `{ "type": "step" }` | Advance one tick (when paused) |
| `{ "type": "speed", "value": "slow" }` | Set speed mode |
| `{ "type": "resume" }` | Resume from pause |

## Quant Dashboard

Humans need visual access to the same quantitative data that AI agents receive in `StrategyContext`.

### Dashboard Panels

| Panel | Data Source | Update Frequency |
|-------|-------------|------------------|
| Price Chart | `MarketView::candles()` | Every tick |
| Order Book Depth | `MarketView::book()` | Every tick |
| Technical Indicators | `IndicatorCache` | Every tick |
| Factor Scores | `FactorScores` | Every tick |
| Risk Metrics | `RiskCalculator` | Every tick |
| News Feed | `NewsGenerator::active_events()` | On event |
| Portfolio Summary | `Portfolio` | On trade |
| P&L Chart | `PnL` history | Every tick |

### Indicator Display

```typescript
// frontend/src/components/IndicatorPanel.tsx

interface IndicatorPanelProps {
  indicators: {
    sma_20: number;
    sma_50: number;
    ema_12: number;
    ema_26: number;
    rsi_14: number;
    macd: { value: number; signal: number; histogram: number };
    bollinger: { upper: number; middle: number; lower: number };
    atr_14: number;
  };
  factors: {
    momentum: number;    // -1 to +1
    value: number;       // -1 to +1
    volatility: number;  // 0 to +1
    mean_reversion: number; // -1 to +1
  };
}
```

### Signal Summary

Aggregate indicators into human-readable signals:

| Signal | Color | Meaning |
|--------|-------|----------|
| Strong Buy | Green | Multiple indicators bullish |
| Buy | Light Green | Net bullish |
| Neutral | Gray | Mixed signals |
| Sell | Light Red | Net bearish |
| Strong Sell | Red | Multiple indicators bearish |

## Decision Support UI

### Order Entry Panel

| Feature | Description |
|---------|-------------|
| Quick buttons | Buy/Sell 10%, 25%, 50%, 100% of buying power |
| Limit price helper | Click on order book level to set price |
| Risk preview | Show position size, new exposure before confirm |
| Bracket orders | Set stop-loss and take-profit with single entry |

### Position Management

| Feature | Description |
|---------|-------------|
| One-click flatten | Close all positions instantly |
| Scale out buttons | Reduce position by 25%, 50% |
| Break-even line | Visual line on chart showing avg entry |
| P&L watermark | Unrealized P&L overlaid on chart |

### Risk Alerts

| Alert | Trigger | Action |
|-------|---------|--------|
| Drawdown Warning | Drawdown > 5% | Yellow banner |
| Drawdown Critical | Drawdown > 10% | Red banner, audio |
| Position Limit | Near max position | Prevent over-buying |
| Buying Power Low | Cash < 10% of initial | Warning banner |

## Information Parity Contract

**Principle:** Humans must have access to ALL information that AI agents can observe.

| AI Observation | Human Equivalent |
|----------------|------------------|
| `StrategyContext.view` | Price chart, order book display |
| `StrategyContext.indicators` | Indicator panel with values |
| `StrategyContext.factors` | Factor score gauges |
| `StrategyContext.risk` | Risk metrics panel |
| `StrategyContext.events` | News feed with sentiment |
| `StrategyContext.portfolio` | Portfolio summary |
| Order book depth (N levels) | Visual depth chart |
| Historical candles | Scrollable chart history |

**What humans get that AI doesn't:**
- Visual pattern recognition on charts
- Intuition from real-world knowledge
- Ability to pause and think indefinitely

**What AI gets that humans don't:**
- Perfect recall of all historical data
- Instant calculation (no mental math)
- Consistent execution (no fat fingers)

## Game Service as Dashboard BFF (Backend-For-Frontend)

The frontend shouldn't call 5+ services directly. The `services/game` crate acts as an **aggregation layer** for human player data:

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │TimeCtrl  │ │Indicators│ │  Risk    │ │Portfolio │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       └────────────┴────────────┴────────────┘              │
│                         │                                   │
│              WebSocket + REST                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              Game Service (BFF) :8008                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  GET /game/dashboard  →  DashboardSnapshot          │   │
│  │  WS  /game/stream     →  Real-time updates          │   │
│  └─────────────────────────────────────────────────────┘   │
│       │           │           │           │                 │
│   analytics    risk      portfolio    exchange             │
│   (internal)  (internal)  (internal)  (internal)           │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard Aggregation Endpoint

| Endpoint | Method | Purpose |
|----------|--------|----------|
| `/game/dashboard` | GET | **Aggregated snapshot** of all dashboard data |
| `/game/stream` | WS | Real-time dashboard updates (push on each tick) |
| `/game/time` | POST | Set simulation speed |
| `/game/step` | POST | Advance one tick (when paused) |

### DashboardSnapshot Response

```rust
// crates/services/game/dashboard.rs

#[derive(Serialize)]
pub struct DashboardSnapshot {
    pub tick: Tick,
    pub timestamp: Timestamp,
    
    // Price data
    pub prices: HashMap<Symbol, Price>,
    pub book_snapshots: HashMap<Symbol, BookSnapshot>,
    
    // Quant data (from analytics service)
    pub indicators: HashMap<Symbol, IndicatorSet>,
    pub factors: Vec<FactorScore>,
    
    // Risk data (from risk service)
    pub risk_metrics: RiskMetrics,
    pub alerts: Vec<RiskViolation>,
    
    // Portfolio data (from portfolio service)
    pub portfolio: Portfolio,
    pub pnl: PnL,
    
    // News (from news service)
    pub active_events: Vec<NewsEvent>,
    
    // Aggregated signals
    pub signals: HashMap<Symbol, Signal>,
    
    // Time control state
    pub speed: SimulationSpeed,
    pub paused: bool,
}

#[derive(Serialize)]
pub struct IndicatorSet {
    pub sma_20: Option<f64>,
    pub sma_50: Option<f64>,
    pub ema_12: Option<f64>,
    pub ema_26: Option<f64>,
    pub rsi_14: Option<f64>,
    pub macd: Option<MacdValue>,
    pub bollinger: Option<BollingerValue>,
    pub atr_14: Option<f64>,
}
```

### WebSocket Stream Messages

```typescript
// Frontend receives these on /game/stream

type DashboardUpdate = 
  | { type: 'tick', data: DashboardSnapshot }
  | { type: 'trade', data: Trade }
  | { type: 'news', data: NewsEvent }
  | { type: 'alert', data: RiskViolation }
  | { type: 'speed_changed', data: SimulationSpeed };
```

### Why BFF Pattern?

| Benefit | Explanation |
|---------|-------------|
| Single connection | Frontend opens 1 WebSocket, not 5 |
| Consistent tick | All data from same simulation tick |
| Reduced latency | Internal service calls, not network |
| Simplified frontend | No aggregation logic in React |
| CORS simplicity | Single origin to configure |

## Frontend Components (Phase 22 G Extension)

| Component | Purpose |
|-----------|----------|
| `TimeControls.tsx` | Pause/play/step/speed buttons |
| `IndicatorPanel.tsx` | Technical indicator values |
| `FactorGauges.tsx` | Visual factor score display |
| `RiskDashboard.tsx` | VaR, drawdown, exposure |
| `SignalSummary.tsx` | Aggregated buy/sell/hold signal |
| `QuickTradeButtons.tsx` | % of portfolio quick entry |
| `AlertBanner.tsx` | Risk warnings, news alerts |

---

# Part 12: Storage Design

## Database Split

| Database | Engine | Use Case |
|----------|--------|----------|
| DuckDB | Columnar | Trades (append-only), candles, factor/risk history |
| SQLite | Row | Agents, portfolios, leaderboard cache, snapshots |

## Key Optimizations

| Optimization | Rationale |
|--------------|-----------|
| Trade buffering | Batch writes every N ticks, reduces I/O |
| Leaderboard cache | Avoid O(n log n) sort per request; Tier 1/2 only |
| Decimal as TEXT | Preserves precision in DB |
| Background pool excluded from leaderboard | No individual P&L to track |

## DuckDB Tables

```sql
-- Append-only trade log (Price stored as TEXT for Decimal)
trades (
    id BIGINT PRIMARY KEY,   -- Sequential OrderId (u64)
    symbol TEXT,
    buyer_id BIGINT,         -- Sequential AgentId (u64)
    seller_id BIGINT,        -- Sequential AgentId (u64)
    price TEXT,              -- Decimal as string
    quantity BIGINT,
    timestamp BIGINT,
    tick BIGINT
);

-- Required indexes for trades table
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX idx_trades_buyer ON trades(buyer_id);
CREATE INDEX idx_trades_seller ON trades(seller_id);
CREATE INDEX idx_trades_tick ON trades(tick);

-- Materialized candles
candles_1m (
    symbol TEXT,
    open TEXT,           -- Decimal
    high TEXT,           -- Decimal
    low TEXT,            -- Decimal
    close TEXT,          -- Decimal
    volume BIGINT,
    timestamp BIGINT,
    PRIMARY KEY (symbol, timestamp)
);

-- Required indexes for candles
CREATE INDEX idx_candles_1m_symbol ON candles_1m(symbol);

candles_1h (...);
candles_1d (...);

-- Event log
events (
    id TEXT PRIMARY KEY,
    event_type TEXT,
    target TEXT,
    sentiment DOUBLE,    -- f64 OK, not money
    duration_ticks INT,
    timestamp BIGINT
);

CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_type ON events(event_type);

-- Factor history
factor_snapshots (
    timestamp BIGINT,
    symbol TEXT,
    factor_type TEXT,
    score DOUBLE,        -- f64 OK, statistical
    PRIMARY KEY (timestamp, symbol, factor_type)
);

-- Risk history
risk_snapshots (
    timestamp BIGINT,
    agent_id BIGINT,     -- Sequential AgentId (u64)
    var_95 DOUBLE,
    var_99 DOUBLE,
    sharpe DOUBLE,
    sortino DOUBLE,
    max_drawdown DOUBLE,
    PRIMARY KEY (timestamp, agent_id)
);

CREATE INDEX idx_risk_agent ON risk_snapshots(agent_id);

-- Indicator cache
indicator_cache (
    timestamp BIGINT,
    symbol TEXT,
    indicator_type TEXT,
    value DOUBLE,
    PRIMARY KEY (timestamp, symbol, indicator_type)
);

CREATE INDEX idx_indicator_symbol ON indicator_cache(symbol, indicator_type);
```

## SQLite Tables

```sql
-- Agent configuration
agents (
    id INTEGER PRIMARY KEY, -- Sequential AgentId (u64)
    strategy_name TEXT,
    config_json TEXT,
    latency_ticks INTEGER,
    created_at INTEGER
);

-- Portfolio state (quantities as INTEGER, prices as TEXT for Decimal)
portfolios (
    agent_id INTEGER,       -- Sequential AgentId (u64)
    symbol TEXT,
    quantity INTEGER,
    avg_cost TEXT,          -- Decimal as string
    PRIMARY KEY (agent_id, symbol)
);

CREATE INDEX idx_portfolios_agent ON portfolios(agent_id);

-- Cash balances
cash (
    agent_id INTEGER PRIMARY KEY,  -- Sequential AgentId (u64)
    balance TEXT                   -- Decimal as string
);

-- Risk limits
risk_limits (
    agent_id INTEGER PRIMARY KEY,  -- Sequential AgentId (u64)
    max_position INTEGER,
    max_sector_exposure REAL,
    max_drawdown REAL
);

-- State snapshots
snapshots (
    id INTEGER PRIMARY KEY, -- Sequential snapshot ID
    state_blob BLOB,
    timestamp INTEGER
);

CREATE INDEX idx_snapshots_timestamp ON snapshots(timestamp);

-- Player accounts (Game Mode)
players (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

-- Player sessions (Game Mode)
sessions (
    token TEXT PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES players(id),
    expires_at INTEGER NOT NULL
);

CREATE INDEX idx_sessions_player ON sessions(player_id);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);

-- Game sessions (Game Mode)
game_sessions (
    id TEXT PRIMARY KEY,
    config_json TEXT,
    status TEXT NOT NULL,  -- 'waiting', 'active', 'completed'
    winner_id TEXT REFERENCES players(id),
    start_tick INTEGER,
    end_tick INTEGER,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_game_sessions_status ON game_sessions(status);

-- Game participants (Game Mode)
game_participants (
    game_id TEXT REFERENCES game_sessions(id),
    player_id TEXT REFERENCES players(id),
    agent_id INTEGER NOT NULL,  -- Link to agents table
    final_pnl TEXT,             -- Decimal as string
    rank INTEGER,
    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX idx_game_participants_player ON game_participants(player_id);
```

## Price Storage

```rust
// crates/storage/price_serde.rs

use crate::{Price, Cash, PRICE_SCALE};

pub fn price_to_storage(p: &Price) -> i64 {
    p.0
}

pub fn price_from_storage(v: i64) -> Price {
    Price(v)
}

// For display/JSON: convert to float string
pub fn price_to_display(p: &Price) -> String {
    format!("{:.4}", p.to_float())
}
```

## Trade Buffer & Flush Configuration

```rust
// crates/storage/persistence_hook.rs

impl PersistenceHook {
    /// Flush buffered trades to DB every N ticks
    pub const FLUSH_INTERVAL_TICKS: u64 = 100;
    
    /// Maximum trades to buffer before forced flush
    pub const MAX_BUFFER_SIZE: usize = 10_000;
}
```

**Flush Triggers (any condition triggers write):**
1. `FLUSH_INTERVAL_TICKS` reached since last flush
2. Buffer size exceeds `MAX_BUFFER_SIZE`
3. Simulation shutdown/pause
4. Explicit `flush()` call (e.g., before snapshot)

---

# Part 13: Simulation Flow

## Tick Sequence

1. Generate news events
2. Compute indicators (cached)
3. **Tier 1:** Run all smart agents (parallel)
4. **Tier 2:** Find triggered agents via wake index, run them
5. **Tier 3:** Pool reacts to events, generates orders
6. Submit all orders to market (with latency)
7. Process pending orders whose latency expired
8. Run matching engine
9. Update portfolios from fills
10. Decay background pool state
11. Fire hooks (persistence, metrics)

## Warm-up Phase

Before smart agents activate:
1. Run N warm-up ticks (configurable, e.g., 500)
2. Only Tier 2 and Tier 3 participate (Tier 3 alone can bootstrap liquidity via statistical generation)
3. Populates order books with realistic depth
4. Prevents cold-start issues for smart agents

**Warm-up Behavior Details:**

| Question | Answer |
|----------|--------|
| Do indicators compute during warm-up? | Yes — required for Tier 2 agents with indicator-based wake conditions |
| Are warm-up trades persisted? | Configurable via `WarmupConfig::persist_warmup_candles` (default: true) |
| Can agents query warm-up history? | Configurable via `WarmupConfig::provide_warmup_history` (default: true) |
| Can Tier 3 alone bootstrap liquidity? | Yes — statistical order generation doesn't depend on existing book depth |
| How are initial prices set? | Via `SymbolConfig::initial_price` in simulation config |

---

# Part 14: Observation System

## Observation Builders

| Builder | Features |
|---------|----------|
| `PriceObservation` | Current prices, returns |
| `BookObservation` | Bid/ask depth, spread |
| `IndicatorObservation` | RSI, MACD, etc. |
| `PortfolioObservation` | Holdings, P&L |
| `MicrostructureObservation` | Order imbalance, depth ratio, trade rate |
| `CompositeObservation` | Combines multiple builders |

## Observation Contract

Each builder exports a contract specifying:
- Shape (dimensions)
- Feature names and indices
- Normalization method per feature

Contract ensures Rust == Python for training parity.

**Contract Versioning:**
- `version: u32` increments on breaking changes to observation shape or normalization
- Migration strategy: contracts with different versions are incompatible; retrain models after version bump
- Include version check in `TradingEnv::load_model()` to fail fast on mismatch

---

# Part 15: Project Phases

**Phase Structure:**

After Phase 12 (Scale Testing), two independent tracks proceed in parallel with mirrored numbering:
- **RL Track (Phases 13-18 RL):** gym → PyO3 → Training → RL Agent Strategy
- **Game Track (Phases 13-22 G):** Services → Game/Frontend

These tracks do NOT depend on each other. The RL Agent (Phase 18 RL) is a Strategy in the `agents` crate that loads ONNX models—it doesn't require services.

**Phase Naming Convention:** 
- `(RL)` = RL track only
- `(G)` = Game track only
- No suffix = Core infrastructure (both tracks need it)

| Phase | Name | Track | Effort | Deliverables |
|-------|------|-------|--------|--------------|
| 1 | Types | Core | 2 days | Core types, sequential IDs, constants |
| 2 | Sim-Core | Core | 1.5 wks | Order book, matching, latency queue |
| 3 | Quant Foundation | Core | 2 wks | Indicators, risk, factors, execution algos |
| 4 | News | Core | 4 days | Event generator, sector model |
| 5 | Agents Foundation | Core | 1 wk | Strategy trait, Tier 1 agents, registry |
| 6 | Agent Scaling | Core | 2 wks | Tier 2, Tier 3, orchestrator, wake index |
| 7 | Technical Strategies | Core | 5 days | RSI, MACD, Bollinger strategies |
| 8 | Statistical Strategies | Core | 1 wk | Pairs, factor, VWAP strategies |
| 9 | Simulation | Core | 1.5 wks | Runner, presets, warm-up, metrics |
| 10 | Storage | Core | 2 wks | Buffered writes, DuckDB/SQLite stores |
| 11 | Integration Testing | Core | 4 days | Docker, E2E tests across core crates |
| 12 | Scale Testing | Core | 1 wk | 100k benchmarks, profiling, validation |
| **--- Parallel tracks start below (13+) ---** ||||
| 13 (RL) | Gym Foundation | RL | 4 days | Env trait, builder |
| 14 (RL) | Gym Observations | RL | 5 days | All observation builders |
| 15 (RL) | Gym Rewards | RL | 3 days | P&L, Sharpe, drawdown rewards |
| 16 (RL) | PyO3 Bindings | RL | 5 days | Python module, parity tests |
| 17 (RL) | Training Scripts | RL | 1 wk | DQN, PPO, ONNX export |
| 18 (RL) | RL Agent Strategy | RL | 5 days | ONNX runtime, adds Strategy to agents |
| 13 (G) | Services Foundation | Game | 4 days | Bridge, middleware, telemetry |
| 14 (G) | Data Service | Game | 1 wk | Analytics, portfolio, risk, news APIs |
| 15 (G) | Game Service | Game | 1 wk | WebSocket, sessions, time control, orders, BFF |
| 16 (G) | Storage Service | Game | 3 days | Snapshots, trade log, queries |
| 17 (G) | Chatbot Service | Game | 1 wk | LLM function calling |
| 18 (G) | Game Frontend | Game | 2.5 wks | React UI, real-time dashboard |
| 19 | RL Game Integration | Both | 1 wk | RL agents as game opponents |

**Parallel Development:**
```
                              ┌─── RL Track (13-18 RL) ─────────────────┐
                              │                                          │
Phase 12 (Scale) ────────────┤                                          ├─── Phase 19 (RL Game Integration)
                              │                                          │    (requires both tracks)
                              └─── Game Track (13-18 G) ────────────────┘
```

---

# Part 16: Architectural Considerations

## Multi-Symbol Support

Multi-symbol trading infrastructure should be added in **V2** alongside agent scaling:

### Market Structure for Multiple Symbols

```rust
// crates/sim-core/market.rs
pub struct Market {
    books: HashMap<Symbol, OrderBook>,
    pending: PendingOrderQueue,  // Shared across symbols
    order_id_counter: AtomicU64,
    current_tick: Tick,
}

impl Market {
    pub fn symbols(&self) -> Vec<Symbol> {
        self.books.keys().cloned().collect()
    }
    
    pub fn book(&self, symbol: &Symbol) -> Option<&OrderBook> {
        self.books.get(symbol)
    }
}
```

### Configuration

```rust
// crates/simulation/config.rs
pub struct SimulationConfig {
    pub symbols: Vec<SymbolConfig>,  // Multiple symbols
    pub sectors: SectorConfig,        // Symbol-to-sector mapping
    // ...
}
```

### Why V2?
1. `TieredOrchestrator` already needs agent-symbol relationships
2. `WakeConditionIndex` benefits from symbol-scoped indexing
3. Background pool sentiment should be per-sector
4. Pairs trading strategy (Phase 8) requires correlated symbols

---

## Short-Selling and Position Limits

### Problem Statement

Unrestricted short-selling leads to unrealistic scenarios:
- Agents accumulate positions of -1000+ shares without borrowing
- No margin requirements or borrow costs
- Infinite liquidity for shorting

### Realistic Position Model

**Long positions** are constrained naturally by:
1. **Cash available** — can't buy what you can't afford
2. **Shares outstanding** — can't buy more shares than exist in the market

**Short positions** require explicit infrastructure:

```rust
// crates/types/lib.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortSellingConfig {
    /// Whether short selling is allowed at all
    pub enabled: bool,
    /// Annual borrow rate in basis points (e.g., 50 = 0.5%/year)
    pub borrow_rate_bps: u32,
    /// Require explicit locate before shorting
    pub locate_required: bool,
    /// Maximum short position per agent (risk limit)
    pub max_short_per_agent: Quantity,
}

#[derive(Debug, Clone)]
pub struct BorrowLedger {
    /// Available shares to borrow per symbol (derived from shares_outstanding * borrow_fraction)
    available: HashMap<Symbol, Quantity>,
    /// Borrowed positions: agent_id → symbol → BorrowPosition
    borrows: HashMap<AgentId, HashMap<Symbol, BorrowPosition>>,
}

#[derive(Debug, Clone)]
pub struct BorrowPosition {
    pub symbol: Symbol,
    pub quantity: Quantity,
    pub borrow_rate_bps: u32,
    pub borrowed_at: Tick,
}
```

### Order Validation

```rust
// crates/agents/src/portfolio.rs
impl Portfolio {
    /// Validate order against constraints before submission
    pub fn validate_order(
        &self,
        order: &Order,
        symbol_config: &SymbolConfig,
        short_config: &ShortSellingConfig,
        borrow_ledger: &BorrowLedger,
        total_held: Quantity,  // Sum of all agents' long positions
    ) -> Result<(), RiskViolation> {
        let current_pos = self.position(&order.symbol);
        let projected = match order.side {
            OrderSide::Buy => current_pos + order.quantity as i64,
            OrderSide::Sell => current_pos - order.quantity as i64,
        };
        
        // Check shares outstanding limit (long positions)
        if projected > 0 {
            let new_total = total_held + order.quantity;
            if new_total > symbol_config.shares_outstanding {
                return Err(RiskViolation::InsufficientShares);
            }
        }
        
        // Check short limit (negative position)
        if projected < -(short_config.max_short_per_agent as i64) {
            return Err(RiskViolation::ShortLimitExceeded);
        }
        
        // Check borrow availability for shorts
        if projected < 0 {
            let needed_borrow = (-projected) as Quantity;
            let existing_borrow = borrow_ledger.borrowed(self.agent_id, &order.symbol);
            let additional_needed = needed_borrow.saturating_sub(existing_borrow);
            
            if !borrow_ledger.can_borrow(&order.symbol, additional_needed) {
                return Err(RiskViolation::NoBorrowAvailable);
            }
        }
        
        Ok(())
    }
}
```

### Default Configuration

| Scenario | `shares_outstanding` | `max_short` | Borrow Pool (% of float) | Use Case |
|----------|---------------------|-------------|-------------------------|----------|
| Training | 1,000,000 | 0 | 0% | Long-only RL agents |
| Realistic | 10,000,000 | 10,000 | 15% | Balanced simulation |
| Liquid | 100,000,000 | 50,000 | 25% | Large cap stocks |

---

## Borrow-Checking and Data-Race Pitfalls

### V2 Pitfalls (Agent Scaling)

**Pitfall 1: Parallel Agent Execution with Shared Market State**

```rust
// PROBLEM: Agents read market, produce orders simultaneously
agents.par_iter_mut().map(|agent| {
    agent.decide(&context)  // context borrows market immutably
}).collect::<Vec<_>>();
```

**Solution:** Two-phase tick architecture:
1. **Read phase:** All agents read `&MarketView` (immutable, safe to parallelize via rayon)
2. **Write phase:** Collected orders applied sequentially to `&mut Market`

```rust
impl TieredOrchestrator {
    pub fn tick(&mut self, market: &Market) -> Vec<Order> {
        // Phase 1: Read (parallel-safe)
        let tier1_orders = self.run_tier1_parallel(market);
        let tier2_orders = self.run_tier2_triggered(market);
        let tier3_orders = self.pool.generate(market);
        
        // Phase 2: Collect (sequential merge)
        [tier1_orders, tier2_orders, tier3_orders].concat()
        // Orders applied to &mut Market by Simulation, not Orchestrator
    }
}
```

**Pitfall 2: WakeConditionIndex Updates During Tick**

```rust
// PROBLEM: Agent wakes, decides to change its own wake conditions
fn on_wake(&mut self, ctx: &LightweightContext, index: &mut WakeConditionIndex) {
    // Can't mutate index while iterating triggered agents!
}
```

**Solution:** Deferred condition updates (see Parametric Condition Updates section above).

**Pitfall 3: Background Pool Accounting**

```rust
// PROBLEM: Pool generates orders, fills come back asynchronously
impl BackgroundAgentPool {
    fn tick(&mut self, ctx: &PoolContext) -> Vec<Order>;  // Generates orders
    fn on_fill(&mut self, fill: &Fill);  // Called later when matched
}
```

**Solution:** `BackgroundPoolAccounting` is append-only—fills are recorded but never read during order generation.

### V3 Pitfalls (Persistence & Events)

**Pitfall 4: SimulationHook Borrows**

```rust
// PROBLEM: Multiple hooks need &mut self
for hook in &mut self.hooks {
    hook.on_tick(&result);  // Each hook gets exclusive &mut self sequentially
}
```

**Solution:** Hooks are called sequentially (no parallelism for hooks).

**Pitfall 5: Snapshot During Active Simulation**

**Solution:** Snapshots only at tick boundaries:
```rust
impl Simulation {
    pub fn step(&mut self) -> TickResult {
        // ... complete tick atomically ...
        if self.should_snapshot() {
            self.snapshot_buffer = Some(self.to_snapshot());
        }
    }
}
```

### V4 Pitfalls (RL/Game Tracks)

**V4-RL: PyO3 GIL Considerations**

```rust
// Release GIL during Rust computation
#[pyfunction]
fn step(py: Python, env: &mut TradingEnv, action: i32) -> PyResult<StepResult> {
    py.allow_threads(|| env.step_internal(action))  // Pure Rust, no GIL
}
```

**V4-Game: Async Bridge**

Channel-based communication between async services and sync simulation (see Part 7: Sync/Async Architecture).

---

## Error Handling Strategy

### Unified Error Chain

```rust
// crates/simulation/error.rs
#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    #[error("agent error: {0}")]
    Agent(#[from] AgentError),
    #[error("market error: {0}")]
    Market(#[from] SimCoreError),
    #[error("quant error: {0}")]
    Quant(#[from] QuantError),
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
}
```

### Configuration Validation

```rust
impl SimulationConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.tier1_count > 1000 {
            warn!("Tier 1 count {} exceeds recommended max 100", self.tier1_count);
        }
        if self.tier2_count > 0 && self.symbols.is_empty() {
            return Err(ConfigError::NoSymbolsForReactiveAgents);
        }
        if self.short_selling.enabled && self.short_selling.borrow_pool_size == 0 {
            return Err(ConfigError::ShortSellingWithNoBorrowPool);
        }
        Ok(())
    }
}
```

---

## Observability & Metrics

### TickMetrics for Debugging

```rust
pub struct TickMetrics {
    pub tick_number: Tick,
    pub tier1_decision_time_us: u64,
    pub tier2_wakeups: usize,
    pub tier3_orders_generated: usize,
    pub matching_time_us: u64,
    pub total_tick_time_us: u64,
    pub orders_submitted: usize,
    pub trades_executed: usize,
}
```

### Memory Budget Tracking

```rust
pub struct MemoryStats {
    pub tier1_bytes: usize,
    pub tier2_bytes: usize,
    pub tier3_bytes: usize,
    pub book_bytes: usize,
    pub cache_bytes: usize,
    pub total_bytes: usize,
}

impl Simulation {
    pub fn memory_stats(&self) -> MemoryStats {
        // Implementation measures actual heap usage
    }
}
```

---

## Containerization & Deployment (V4-Game+)

### Overview

For environment-agnostic deployment, all services are containerized with Docker and orchestrated via Docker Compose (dev/staging) or Kubernetes (production).

### Container Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Docker Compose / K8s                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   nginx     │  │    data     │  │    game     │  │   storage   │   │
│  │   :80/443   │  │   :8001     │  │   :8002     │  │   :8003     │   │
│  │  (reverse   │  │             │  │  + WS :8082 │  │             │   │
│  │   proxy)    │  │             │  │             │  │             │   │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         │                                                               │
│  ┌──────┴──────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │  frontend   │  │   chatbot   │  │ simulation  │                     │
│  │   :3000     │  │   :8004     │  │  (no port)  │                     │
│  │  (React)    │  │             │  │  internal   │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                     │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐                                      │
│  │  postgres   │  │    redis    │  (optional: session store, cache)    │
│  │   :5432     │  │   :6379     │                                      │
│  └─────────────┘  └─────────────┘                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dockerfile Strategy

**Multi-stage builds** for minimal image size:

```dockerfile
# Rust services (data, game, storage, chatbot, simulation)
FROM rust:1.75-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin <service>

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/<service> /usr/local/bin/
CMD ["<service>"]
```

```dockerfile
# Frontend (React)
FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
```

### Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'
services:
  simulation:
    build:
      context: .
      dockerfile: docker/simulation.Dockerfile
    environment:
      - RUST_LOG=info
    volumes:
      - sim-data:/data
    
  data:
    build:
      context: .
      dockerfile: docker/data.Dockerfile
    ports:
      - "8001:8001"
    depends_on:
      - simulation
    environment:
      - SIMULATION_BRIDGE_ADDR=simulation:9000
      
  game:
    build:
      context: .
      dockerfile: docker/game.Dockerfile
    ports:
      - "8002:8002"
      - "8082:8082"  # WebSocket
    depends_on:
      - simulation
      - data
      
  storage:
    build:
      context: .
      dockerfile: docker/storage.Dockerfile
    ports:
      - "8003:8003"
    volumes:
      - db-data:/data
      
  chatbot:
    build:
      context: .
      dockerfile: docker/chatbot.Dockerfile
    ports:
      - "8004:8004"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - game

volumes:
  sim-data:
  db-data:
```

### Environment Configuration

```bash
# .env.example
RUST_LOG=info
DATABASE_URL=sqlite:///data/trading.db
SIMULATION_SEED=42
OPENAI_API_KEY=sk-...

# Production overrides
RUST_LOG=warn
ENABLE_METRICS=true
METRICS_ENDPOINT=http://prometheus:9090
```

### Health Checks

All services expose `/health` endpoint (see Part 10: Services):
- Used by Docker `HEALTHCHECK`
- Used by Kubernetes liveness/readiness probes
- Returns service status, uptime, and dependency health

### Kubernetes (Production)

For production deployment, Helm charts or Kustomize manifests:

```yaml
# k8s/deployment-game.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: game-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: game
  template:
    spec:
      containers:
      - name: game
        image: quant-trading-gym/game:latest
        ports:
        - containerPort: 8002
        - containerPort: 8082
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 10
          periodSeconds: 30
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### CI/CD Integration

```yaml
# .github/workflows/docker.yml
name: Build and Push
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

### Deployment Phases

| Phase | Deployment | Use Case |
|-------|------------|----------|
| V0-V3 | Local binary | Development, testing |
| V4-Game | Docker Compose | Local multi-service, demo |
| Production | Kubernetes | Scalable, cloud deployment |

---

# Part 17: Phase Details

## Phase 1: Types

**Goal:** Shared vocabulary compiles

**Context Required:** None (greenfield)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/types/Cargo.toml` | Dependencies (serde) |
| `crates/types/lib.rs` | All type definitions |
| `crates/types/constants.rs` | `BACKGROUND_POOL_ID` and other sentinel values |

**Key Types:**
- `Price = Decimal` (not f64)
- `Cash = Decimal` (not f64)
- `Order` with `latency_ticks: u64`

**Exit Criteria:**
- `cargo build -p types` succeeds
- All types derive `Debug`, `Clone`, `Serialize`, `Deserialize`
- Decimal types used for all monetary values

**Effort:** 2 days

---

## Phase 2: Sim-Core

**Goal:** Order book and matching engine with latency support

**Context Required:**
- `types` crate (read-only reference)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/sim-core/Cargo.toml` | Dependencies |
| `crates/sim-core/order_book.rs` | `OrderBook` struct |
| `crates/sim-core/matching.rs` | `MatchingEngine` with price-time priority |
| `crates/sim-core/pending_orders.rs` | `PendingOrderQueue` for latency |
| `crates/sim-core/market.rs` | `Market` struct, `MarketView` trait |
| `crates/sim-core/lib.rs` | Module exports |

**Key Features:**
- Decimal arithmetic for prices
- Latency queue delays order processing
- Orders queued at `current_tick + latency_ticks`
- **Self-trade prevention:** Matching engine MUST check `buyer_id != seller_id`
- **Background pool self-trade handling:** When both buyer and seller are `BACKGROUND_POOL_ID`, skip the match:

```rust
// In matching engine
if buyer_id == BACKGROUND_POOL_ID && seller_id == BACKGROUND_POOL_ID {
    // Skip - background pool doesn't trade with itself
    continue;
}
```

**Exit Criteria:**
- Limit orders match correctly with Decimal prices
- Market orders execute at best price
- Partial fills work
- Orders with `latency_ticks > 0` are delayed
- No floating point used for prices
- Self-trades rejected (agent cannot trade with itself)

**Tests:**
- `order_book_matches_limit_orders_by_price_time_priority`
- `market_order_executes_at_best_available_price`
- `partial_fill_leaves_remainder_on_book`
- `price_time_priority_respected_for_same_price_orders`
- `cancelled_order_removed_from_book`
- `latency_queue_delays_order_by_configured_ticks`
- `decimal_precision_preserved_through_matching`
- `self_trade_prevention_rejects_same_agent_orders`
- `background_pool_orders_do_not_match_each_other`

**Test Naming Convention:** Use descriptive names without `test_` prefix. The name should describe the expected behavior, not just the scenario.

**Effort:** 1.5 weeks

---

## Phase 3: Quant Foundation

**Goal:** Indicators and risk calculations work

**Context Required:**
- `types` crate (read-only reference)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/quant/Cargo.toml` | Dependencies |
| `crates/quant/indicators.rs` | `Indicator` trait, SMA, EMA, RSI, MACD, Bollinger, ATR |
| `crates/quant/risk.rs` | `RiskCalculator` trait, VaR, drawdown, Sharpe, Sortino |
| `crates/quant/factors.rs` | `FactorModel` trait, momentum, volatility factors |
| `crates/quant/execution.rs` | `ExecutionAlgo` trait, VWAP, TWAP |
| `crates/quant/stats.rs` | Correlation, cointegration, returns |
| `crates/quant/engine.rs` | `IndicatorEngine`, `FactorEngine` |
| `crates/quant/lib.rs` | Module exports |

**Note:** Uses `f64` for statistical values (not money). Converts from `Decimal` prices as needed.

**Exit Criteria:**
- All indicators produce correct values against known test data
- Risk metrics compute correctly
- Engines batch-compute for multiple symbols

**Tests:**
- `test_sma_calculation`
- `test_rsi_boundaries`
- `test_macd_crossover_detection`
- `test_var_calculation`
- `test_sharpe_ratio`

**Effort:** 2 weeks

---

## Phase 4: News

**Goal:** Event generation works

**Context Required:**
- `types` crate (read-only reference)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/news/Cargo.toml` | Dependencies |
| `crates/news/sectors.rs` | `SectorModel`, stock-to-sector mapping |
| `crates/news/generator.rs` | `NewsGenerator`, configurable frequencies |
| `crates/news/lib.rs` | Module exports |

**Exit Criteria:**
- Events generated at configured frequencies
- Sector events affect correct stocks
- Deterministic with seed

**Tests:**
- `test_event_frequency`
- `test_sector_propagation`
- `test_deterministic_generation`

**Effort:** 4 days

---

## Phase 5: Agents Foundation

**Goal:** Modular strategy framework with basic strategies

**Context Required:**
- `types` crate
- `sim-core` traits (`MarketView`)
- `quant` traits and caches
- `news` types (`NewsEvent`)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/agents/Cargo.toml` | Dependencies |
| `crates/agents/traits.rs` | `Strategy` trait with `latency_ticks()`, `StrategyContext` |
| `crates/agents/registry.rs` | `StrategyRegistry` |
| `crates/agents/orchestrator.rs` | `AgentOrchestrator` (Tier 1 only at this phase) |
| `crates/agents/strategies/mod.rs` | Strategy registration |
| `crates/agents/strategies/noise_trader.rs` | `NoiseTraderStrategy` |
| `crates/agents/strategies/market_maker.rs` | `MarketMakerStrategy` |
| `crates/agents/lib.rs` | Module exports |

**Key Features:**
- Strategies declare their latency via `latency_ticks()`
- Orders created with strategy's latency
- `AgentOrchestrator` handles Tier 1 agents only at this phase

**Exit Criteria:**
- Strategies declare indicator/factor dependencies
- Strategies declare latency requirements
- Orchestrator runs agents in parallel
- Registry creates strategies by name

**Tests:**
- `test_strategy_registration`
- `test_parallel_execution`
- `test_context_population`
- `test_latency_propagation`

**Effort:** 1 week

---

## Phase 6: Agent Scaling

**Goal:** Tier 2 (reactive) and Tier 3 (background) agents; evolve orchestrator

**Deliverables:**
- `ReactiveAgent` with enum dispatch
- `ReactivePortfolio` (single-symbol option)
- `LightweightContext` for Tier 2 agents
- `WakeConditionIndex` with BTreeMap for O(log n) lookup
- `BackgroundAgentPool` with sentiment/volatility reactivity
- `BackgroundConfig` with regime presets
- `BackgroundPoolAccounting` for sanity checks
- Evolve `AgentOrchestrator` → `TieredOrchestrator` integrating all three tiers
- Rename file `orchestrator.rs` to reflect `TieredOrchestrator` as the main export
- **Portfolio validation:** Reject orders that would result in negative cash or short positions beyond limits

**Critical Safety Features:**
- Portfolio updates MUST check `cash >= 0` after buy orders
- Portfolio updates MUST check `quantity >= 0` after sell orders (unless short selling allowed)
- Return `RiskViolation::InsufficientCash` or `RiskViolation::NegativePosition` on violation

**Exit Criteria:**
- 100k agents initialize in <1 second
- 100k agents tick in <10ms
- Memory usage <500 MB for 100k agents
- Background pool reacts to events
- Wake conditions trigger correctly
- Negative balance/position prevented

**Tests:** `test_reactive_agent_wake`, `test_wake_index_efficiency`, `test_background_pool_generation`, `test_100k_agents_memory` (feature-gated), `test_100k_agents_throughput` (feature-gated), `test_negative_cash_prevention`, `test_negative_position_prevention`

**Effort:** 2 weeks

---

## Phase 7: Technical Strategies

**Goal:** Indicator-based strategies work

**Context Required:**
- `agents` traits (`Strategy`, `StrategyContext`)
- `quant` indicator types
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/agents/strategies/rsi_momentum.rs` | RSI oversold/overbought |
| `crates/agents/strategies/macd_crossover.rs` | MACD signal crossover |
| `crates/agents/strategies/bollinger_reversion.rs` | Band mean reversion |
| `crates/agents/strategies/trend_following.rs` | SMA/EMA crossover |

**Exit Criteria:**
- Each strategy produces orders matching expected logic
- Strategies correctly declare required indicators

**Tests:**
- `test_rsi_buy_signal`
- `test_macd_crossover_signal`
- `test_bollinger_reversion_signal`

**Effort:** 5 days

---

## Phase 8: Statistical Strategies

**Goal:** Factor and pairs strategies work

**Context Required:**
- `agents` traits
- `quant` factor and stats modules
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/agents/strategies/pairs_trading.rs` | Cointegration-based pairs |
| `crates/agents/strategies/factor_long_short.rs` | Long high-factor, short low-factor |
| `crates/agents/strategies/vwap_executor.rs` | VWAP execution |
| `crates/agents/strategies/news_reactive.rs` | Event-driven trading |

**Exit Criteria:**
- Pairs trading identifies and trades spread divergence
- Factor strategy ranks and trades universe
- VWAP slices orders correctly

**Tests:**
- `test_pairs_spread_signal`
- `test_factor_ranking`
- `test_vwap_slicing`

**Effort:** 1 week

---

## Phase 9: Simulation

**Goal:** Complete simulation loop with tiered orchestrator and warm-up

**Context Required:**
- `sim-core` (`Market`)
- `quant` (`IndicatorEngine`, `FactorEngine`)
- `agents` (`TieredOrchestrator`) — evolved from `AgentOrchestrator` in Phase 6
- `news` (`NewsGenerator`)
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/simulation/Cargo.toml` | Dependencies |
| `crates/simulation/config.rs` | `SimulationConfig` with `enable_latency`, `default_latency_ticks` |
| `crates/simulation/runner.rs` | `SimulationRunner`, tick loop |
| `crates/simulation/hooks.rs` | `SimulationHook` trait, logging hook |
| `crates/simulation/builder.rs` | `SimulationBuilder` |
| `crates/simulation/lib.rs` | Module exports |
| `crates/simulation/metrics.rs` | `TickMetrics` |
| `crates/simulation/warmup.rs` | Warm-up logic |

**Key Features:**
- All code is synchronous (no async)
- Deterministic with seed
- Latency processing integrated into tick loop
- Warm-up phase populates books

**Warm-up Configuration:**
```rust
pub struct WarmupConfig {
    /// Number of ticks before smart agents activate (default: 500)
    pub warmup_ticks: u64,
    /// Whether to store candles generated during warm-up (default: true)
    pub persist_warmup_candles: bool,
    /// Whether smart agents receive warm-up history on activation (default: true)
    pub provide_warmup_history: bool,
    /// Minimum book depth (bids + asks) required to end warm-up early (default: None)
    pub min_book_depth: Option<usize>,
}
```

**Warm-up Progress Monitoring:**
```rust
pub struct WarmupProgress {
    pub current_tick: Tick,
    pub total_warmup_ticks: u64,
    pub book_depth: HashMap<Symbol, usize>,
    pub percent_complete: f64,
}

impl SimulationRunner {
    /// Returns warmup progress if still in warmup phase, None otherwise.
    pub fn warmup_progress(&self) -> Option<WarmupProgress>;
}
```

**Exit Criteria:**
- Full tick loop with all three tiers
- Warm-up populates order books before smart agents activate
- Metrics capture timing breakdown by tier
- Presets work correctly
- Same seed produces identical simulation output

**Tests:**
- `test_full_tick_cycle`
- `test_hook_invocation`
- `test_10k_ticks_no_panic`
- `test_deterministic_output` — **Critical:** Run same simulation twice with same seed, verify identical final state
- `test_latency_integration`
- `test_empty_order_book_handling` — Edge case: what happens when book is empty?
- `test_cash_exhaustion` — Edge case: what happens when agent runs out of cash?

**Effort:** 1.5 weeks

---

## Phase 10: Storage

**Goal:** Persistence layer works

**Context Required:**
- `types` crate
- `simulation` (`SimulationHook` trait)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/storage/Cargo.toml` | Dependencies (duckdb, rusqlite) |
| `crates/storage/decimal_serde.rs` | Decimal to/from string |
| `crates/storage/schema.rs` | Table definitions |
| `crates/storage/connection.rs` | DuckDB + SQLite connections |
| `crates/storage/stores/trades.rs` | `TradeStore` |
| `crates/storage/stores/candles.rs` | `CandleStore` with aggregation |
| `crates/storage/stores/portfolios.rs` | `PortfolioStore` with event sourcing |
| `crates/storage/stores/risk.rs` | `RiskStore` |
| `crates/storage/stores/snapshots.rs` | `SnapshotStore` (market snapshots) |
| `crates/storage/stores/game_snapshots.rs` | `GameSnapshotStore` (save/resume) |
| `crates/storage/stores/trade_log.rs` | `TradeLogStore` (append-only game log) |
| `crates/storage/persistence_hook.rs` | `PersistenceHook` |
| `crates/storage/lib.rs` | Module exports |

**Key Features:**
- Decimal values stored as TEXT strings
- Proper decimal parsing on read
- Game snapshots for save/resume (auto-save every N ticks, manual save)
- Trade log for post-game analysis

**Exit Criteria:**
- Trades append and query correctly
- Candles aggregate on write
- Portfolios rebuild from trades
- Snapshots save/restore market state
- **Game snapshots save/load correctly**
- **Trade log appends and queries by game**
- Decimal precision preserved

**Tests:**
- `test_trade_append_query`
- `test_candle_aggregation`
- `test_portfolio_rebuild`
- `test_snapshot_restore`
- `test_game_snapshot_roundtrip`
- `test_trade_log_query`
- `test_decimal_roundtrip`

**Effort:** 2 weeks

---

## Phase 11: Integration Testing

**Goal:** All core crates work together before track split

**Context Required:**
- All core crates (1-10)
- Docker compose setup

**Deliverables:**

| File | Contents |
|------|----------|
| `docker-compose.yml` | Service orchestration |
| `tests/integration/` | End-to-end tests |
| `scripts/run_local.sh` | Local dev setup |

**Exit Criteria:**
- All crates integrate correctly
- Full workflow test passes (types → sim → agents → storage)
- Core simulation runs without issues
- Determinism verified across integration

**Tests:**
- `test_full_integration_workflow`
- `test_storage_persistence_roundtrip`
- `test_deterministic_replay`

**Effort:** 4 days

---

## Phase 12: Scale Testing

**Goal:** Verify and optimize 100k agent performance

**Deliverables:**

| File | Contents |
|------|----------|
| `tests/scale/test_100k_agents.rs` | Scale benchmarks (feature-gated) |
| `benches/tick_throughput.rs` | Criterion benchmarks |
| `scripts/profile.sh` | Profiling scripts |

**Exit Criteria:**
- 100k agents tick in <10ms consistently
- Memory stays under 2GB
- No performance regression from baseline
- Bottlenecks identified and documented

**Tests:**
- `test_100k_agents_10ms`
- `test_memory_under_2gb`
- `bench_tick_throughput`

**Effort:** 1 week

---

# Part 15.1: RL Track Phases (13-18 RL)

## Phase 13 (RL): Gym Foundation

**Goal:** RL environment core works

**Context Required:**
- `simulation` (`SimulationRunner`)
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/gym/Cargo.toml` | Dependencies |
| `crates/gym/env.rs` | `Env` trait, `TradingEnv` |
| `crates/gym/observation/traits.rs` | `ObservationBuilder` trait |
| `crates/gym/observation/contract.rs` | `ObservationContract`, `FeatureSpec` |
| `crates/gym/reward/traits.rs` | `RewardFunction` trait |
| `crates/gym/builder.rs` | `TradingEnvBuilder` |
| `crates/gym/lib.rs` | Module exports |

**Key Features:**
- Contract system for observation parity
- Sync-only implementation

**Exit Criteria:**
- `step()` advances simulation and returns observation
- `reset()` returns to initial state
- `contract()` returns observation schema
- Builder pattern works

**Tests:**
- `test_step_advances_tick`
- `test_reset_restores_state`
- `test_done_on_terminal`
- `test_contract_generation`

**Effort:** 4 days

---

## Phase 14 (RL): Gym Observations

**Goal:** Modular observation builders with contracts

**Context Required:**
- `gym` traits (`ObservationBuilder`, `ObservationContract`)
- `types` crate
- `quant` cache types

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/gym/observation/price.rs` | `PriceObservation` |
| `crates/gym/observation/book.rs` | `BookObservation` |
| `crates/gym/observation/indicators.rs` | `IndicatorObservation` |
| `crates/gym/observation/portfolio.rs` | `PortfolioObservation` |
| `crates/gym/observation/composite.rs` | `CompositeObservation` |
| `crates/gym/observation/mod.rs` | Module exports |

**Key Features:**
- Each builder provides its contract
- Composite merges contracts

**Exit Criteria:**
- Each observation builder outputs correct shape
- Each builder provides accurate contract
- Composite combines multiple builders
- Values normalized appropriately

**Tests:**
- `test_price_observation_shape`
- `test_composite_observation`
- `test_normalization`
- `test_contract_accuracy`

**Effort:** 4 days

---

## Phase 15 (RL): Gym Rewards

**Goal:** Modular reward functions work

**Context Required:**
- `gym` traits (`RewardFunction`)
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/gym/reward/pnl.rs` | `PnLReward` |
| `crates/gym/reward/sharpe.rs` | `SharpeReward` |
| `crates/gym/reward/drawdown.rs` | `DrawdownPenaltyReward` |
| `crates/gym/reward/composite.rs` | `CompositeReward` |
| `crates/gym/reward/mod.rs` | Module exports |

**Exit Criteria:**
- Each reward function computes correctly
- Composite applies weights
- Works with Decimal portfolio values

**Tests:**
- `test_pnl_reward`
- `test_sharpe_reward`
- `test_composite_weights`

**Effort:** 3 days

---

## Phase 16 (RL): PyO3 Bindings

**Goal:** Python can use TradingEnv with parity verification

**Context Required:**
- `gym` crate (all)
- PyO3 documentation

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/gym/pyo3.rs` | Python class wrappers |
| `python/quant_trading_gym/__init__.py` | Package init |
| `python/quant_trading_gym/observation.py` | Python observation contract |
| `python/pyproject.toml` | Build config (maturin) |
| `tests/parity/test_observation_parity.py` | Parity verification |

**Key Features:**
- Contract exported to Python
- Parity test infrastructure

**Exit Criteria:**

```python
from quant_trading_gym import TradingEnv

env = TradingEnv(agents=1000, stocks=10, seed=42)
contract = env.observation_contract()
obs = env.reset()
obs, reward, done, info = env.step(action)
```

- Parity test passes

**Tests:**
- `test_python_import`
- `test_gymnasium_compatible`
- `test_observation_parity`
- `test_contract_schema`

**Effort:** 5 days

---

## Phase 17 (RL): Training Scripts

**Goal:** RL training works end-to-end

**Context Required:**
- PyO3 module (as library)
- Stable-Baselines3 / RLlib documentation

**Deliverables:**

| File | Contents |
|------|----------|
| `python/train_dqn.py` | DQN training script |
| `python/train_ppo.py` | PPO training script |
| `python/evaluate.py` | Policy evaluation |
| `python/export_onnx.py` | ONNX export script |
| `python/notebooks/training_analysis.ipynb` | Training visualization |

**Key Features:**
- Uses observation contract for validation
- Exports to ONNX format
- Verifies observation shape before training

**Exit Criteria:**
- Training runs without error
- Loss decreases
- Trained policy beats random baseline
- ONNX export succeeds

**Effort:** 1 week

---

## Phase 18 (RL): RL Agent

**Goal:** Trained policy runs in Rust simulation

**Context Required:**
- `agents` traits (`Strategy`)
- `gym` (`ObservationContract`)
- ONNX runtime (`ort` crate)
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/agents/strategies/rl_agent.rs` | `RLAgentStrategy` |
| `crates/agents/strategies/onnx_runtime.rs` | ONNX model loading |
| `crates/agents/strategies/observation_builder.rs` | Builds observation matching contract |

**Key Features:**
- Loads ONNX model
- Builds observation vector matching Python exactly
- Validates against contract before inference

**ONNX Model Version Validation:**
```python
# In export_onnx.py - add metadata to model
import json
model.metadata_props = {
    "observation_contract_version": str(contract.version),
    "observation_shape": json.dumps(contract.shape),
}
```

```rust
// In Rust - validate on load
impl RLAgentStrategy {
    pub fn load(model_path: &Path, expected_contract: &ObservationContract) -> Result<Self, Error> {
        let model = ort::Session::new(model_path)?;
        let metadata = model.metadata()?;
        
        let model_version: u32 = metadata.get("observation_contract_version")
            .ok_or(Error::MissingMetadata)?
            .parse()?;
        
        if model_version != expected_contract.version {
            return Err(Error::ContractVersionMismatch {
                model: model_version,
                expected: expected_contract.version,
            });
        }
        
        Ok(Self { model, contract: expected_contract.clone() })
    }
}
```

**Exit Criteria:**
- Trained policy loads in Rust
- `RLAgentStrategy` produces actions matching Python inference
- Contract validation passes

**Tests:**
- `test_policy_loading`
- `test_inference_matches_python`
- `test_contract_validation`

**Effort:** 5 days

---

# Part 15.2: Game Track Phases (13-22 G)

## Phase 13 (G): Services Foundation

**Goal:** HTTP service infrastructure with sync/async bridge

**Context Required:**
- axum documentation
- tokio documentation
- `types` crate
- `simulation` (`SimulationRunner`)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/services/common/Cargo.toml` | Dependencies (axum, tokio, tracing) |
| `crates/services/common/config.rs` | `ServiceConfig` with configurable ports |
| `crates/services/common/telemetry.rs` | Tracing, metrics infrastructure |
| `crates/services/common/errors.rs` | Error types, responses |
| `crates/services/common/middleware.rs` | Logging, CORS |
| `crates/services/common/bridge.rs` | `SimulationBridge`, `SimCommand`, channel setup |
| `crates/services/common/state.rs` | Shared application state |
| `crates/services/common/lib.rs` | Module exports |

**Key Features:**
- `SimulationBridge` with mpsc channels
- `run_simulation_thread` blocking function
- Clean async/sync separation

**ServiceConfig:**
```rust
pub struct ServiceConfig {
    pub host: String,
    pub port: u16,
    /// For future service discovery integration
    pub service_registry_url: Option<String>,
}

impl ServiceConfig {
    pub fn from_env(service_name: &str) -> Self {
        Self {
            host: std::env::var(format!("{}_HOST", service_name.to_uppercase()))
                .unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var(format!("{}_PORT", service_name.to_uppercase()))
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8000),
            service_registry_url: std::env::var("SERVICE_REGISTRY_URL").ok(),
        }
    }
}
```

**Observability Infrastructure:**
```rust
// crates/services/common/telemetry.rs
pub mod telemetry {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    
    /// Initialize tracing subscriber with JSON output
    pub fn init_tracing(service_name: &str) {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().json())
            .init();
        tracing::info!(service = service_name, "Service starting");
    }
    
    /// Prometheus metrics endpoint router
    pub fn metrics_router() -> axum::Router {
        // Returns /metrics endpoint with Prometheus format
    }
}
```

**Price JSON Serialization Pattern:**
```rust
// Price/Cash are i64 newtypes - serialize directly as integers
// For human-readable APIs, convert to float in response DTOs
#[derive(Serialize, Deserialize)]
pub struct PriceResponse {
    pub price_raw: i64,           // Raw fixed-point value
    pub price_display: String,    // "100.5000" for humans
}

impl From<Price> for PriceResponse {
    fn from(p: Price) -> Self {
        Self {
            price_raw: p.0,
            price_display: format!("{:.4}", p.to_float()),
        }
    }
}
```

**Exit Criteria:**
- Bridge compiles and handles all command types
- Simulation thread runs independently
- Commands execute and return results
- No async in simulation code

**Tests:**
- `test_bridge_order_submission`
- `test_bridge_tick`
- `test_thread_communication`

**Effort:** 4 days

---

## Phase 14 (G): Data Service

**Goal:** Consolidated analytics, portfolio, risk, news APIs work

**Context Required:**
- `services/common` (bridge)
- `quant` (`IndicatorEngine`, `FactorEngine`, `RiskCalculator`)
- `storage` (`CandleStore`, `PortfolioStore`, `TradeStore`, `RiskStore`)
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/services/data/Cargo.toml` | Dependencies |
| `crates/services/data/main.rs` | Service entry |
| `crates/services/data/analytics.rs` | `/analytics/*` handlers - candles, indicators, factors |
| `crates/services/data/portfolio.rs` | `/portfolio/*` handlers - holdings, P&L, leaderboard |
| `crates/services/data/risk.rs` | `/risk/*` handlers - metrics, exposure, limits |
| `crates/services/data/news.rs` | `/news/*` handlers - feed, scheduled events |

**Exit Criteria:**
- Analytics: Candle queries work with intervals, indicators compute on demand
- Portfolio: Holdings return with Decimal values, P&L calculates correctly
- Risk: Metrics query works, exposure breakdown works, limits can be set
- News: Feed returns recent events

**Tests:**
- `test_candle_queries`
- `test_indicator_calculation`
- `test_portfolio_pnl`
- `test_risk_metrics`

**Effort:** 1 week

---

## Phase 15 (G): Game Service

**Goal:** WebSocket, sessions, time control, orders, BFF for frontend

**Context Required:**
- `services/common` (bridge)
- `sim-core` (`Market`, `SimulationEngine`)
- `agents` (`AgentOrchestrator`)
- `storage` (`SnapshotStore`)
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/services/game/Cargo.toml` | Dependencies |
| `crates/services/game/main.rs` | Service entry |
| `crates/services/game/websocket.rs` | Real-time tick streaming |
| `crates/services/game/session.rs` | Session management, GameConfig |
| `crates/services/game/time_control.rs` | Speed control, step, pause |
| `crates/services/game/orders.rs` | Order submission via bridge |
| `crates/services/game/dashboard.rs` | BFF aggregation endpoint |

**Exit Criteria:**
- WebSocket streams tick updates to frontend
- Sessions create/resume with GameConfig
- Time control: play, pause, step, speed adjustment
- Orders submit via bridge and match
- Dashboard endpoint aggregates from Data + Storage services

**Tests:**
- `test_websocket_connection`
- `test_session_lifecycle`
- `test_time_control`
- `test_order_submission`
- `test_dashboard_aggregation`

**Effort:** 1 week

---

## Phase 16 (G): Storage Service

**Goal:** Snapshots, trade log, historical queries work

**Context Required:**
- `services/common`
- `storage` (`SnapshotStore`, `TradeStore`)
- `types` crate

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/services/storage/Cargo.toml` | Dependencies |
| `crates/services/storage/main.rs` | Service entry |
| `crates/services/storage/snapshots.rs` | Save/load game snapshots |
| `crates/services/storage/trades.rs` | Trade log queries |
| `crates/services/storage/games.rs` | Game metadata listing |

**Exit Criteria:**
- Snapshots save and load correctly
- Trade log queries filter by tick range
- Game listing returns metadata

**Tests:**
- `test_snapshot_save_load`
- `test_trade_queries`
- `test_game_listing`

**Effort:** 3 days

---

## Phase 17 (G): Chatbot Service

**Goal:** Natural language interface works

**Context Required:**
- `services/common`
- Data, Game, Storage service APIs (as HTTP clients)
- LLM API documentation (Claude/OpenAI)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/services/chatbot/Cargo.toml` | Dependencies |
| `crates/services/chatbot/main.rs` | Service entry |
| `crates/services/chatbot/routes.rs` | Chat endpoint |
| `crates/services/chatbot/llm.rs` | LLM client |
| `crates/services/chatbot/functions.rs` | Function schemas |
| `crates/services/chatbot/handlers.rs` | Function execution |

**Exit Criteria:**
- Natural language maps to function calls
- Multi-turn conversation works
- Errors handled gracefully

**Example Interactions:**
- "Buy 100 ACME" → game service order
- "What's my portfolio?" → data service portfolio query
- "Show RSI for TECH" → data service analytics query
- "Deploy my RL bot with 1 tick latency" → game service agent spawn
- "Save the game" → game service save → storage service snapshot

**Effort:** 1 week

---

## Phase 18 (G): Game Frontend

**Goal:** Playable trading game UI against AI opponents

**Context Required:**
- Phase 14-17 (G) services (Data, Game, Storage, Chatbot)
- React/TypeScript

**Deliverables:**

| File | Contents |
|------|----------|
| `frontend/package.json` | Dependencies (React, TypeScript, WebSocket) |
| `frontend/src/App.tsx` | Main application |
| `frontend/src/components/OrderBook.tsx` | Real-time order book display |
| `frontend/src/components/Chart.tsx` | Price chart with indicators |
| `frontend/src/components/Portfolio.tsx` | Holdings, P&L display |
| `frontend/src/components/OrderEntry.tsx` | Trade submission form |
| `frontend/src/components/Leaderboard.tsx` | Rankings display |
| `frontend/src/components/SaveControls.tsx` | Save/Save & Exit buttons |
| `frontend/src/components/ResumeDialog.tsx` | Resume game selection in lobby |
| `frontend/src/components/TimeControls.tsx` | Play/Pause/Step/Speed controls |
| `frontend/src/components/RiskPanel.tsx` | VaR, exposure, drawdown display |
| `frontend/src/components/NewsPanel.tsx` | Real-time news feed |
| `frontend/src/hooks/useWebSocket.ts` | WebSocket connection management |
| `frontend/src/api/client.ts` | REST API client |

**AI Opponents (using existing strategies):**
- Market makers provide liquidity (Phase 5)
- Technical strategies (RSI, MACD) as "medium" difficulty (Phase 7)
- Statistical strategies (pairs, factor) as "hard" difficulty (Phase 8)

**Exit Criteria:**
- Players can trade against AI opponents
- Real-time price updates via WebSocket
- Full dashboard: order book, chart, portfolio, risk, news, leaderboard
- Time controls work (pause, step, speed adjustment)
- Auto-save and manual save work
- Resume from snapshot works

**Tests:**
- `player_can_submit_order_via_ui`
- `websocket_updates_chart_in_realtime`
- `leaderboard_shows_rankings`
- `time_controls_pause_resume`
- `save_resume_workflow`

**Effort:** 2.5 weeks

---

## Phase 19: RL Game Integration

**Goal:** Add trained RL agents as premium opponents in game mode

**Context Required:**
- Phase 18 (RL) (RL Agent Strategy with ONNX)
- Phase 18 (G) (Game Frontend)
- All game services (Data, Game, Storage, Chatbot)

**Deliverables:**

| File | Contents |
|------|----------|
| `crates/services/game/rl_opponents.rs` | RL agent opponent tier configuration |
| `frontend/src/components/DifficultySelector.tsx` | Opponent difficulty selection |
| `frontend/src/components/PerformanceComparison.tsx` | Player vs RL agent metrics |

**Features:**
- RL agent as "expert" difficulty tier
- Model selection in game config
- Performance comparison between player and RL agent
- Leaderboard shows RL agent rankings

**Exit Criteria:**
- Players can select RL agents as opponents
- RL agents load and execute correctly in game context
- Performance metrics compare player vs trained agent

**Tests:**
- `rl_agent_loads_in_game_context`
- `expert_difficulty_uses_rl_model`
- `performance_comparison_displays_correctly`

**Effort:** 1 week

---

# Part 18: Timeline Summary

**Phase Structure:** Core (1-12) → Parallel tracks (13+ RL/G) → Integration (19)

| Phase | Name | Track | Effort | Track Cumulative |
|-------|------|-------|--------|------------------|
| 1 | Types | Core | 2 days | 2 days |
| 2 | Sim-Core | Core | 1.5 wks | 2 wks |
| 3 | Quant Foundation | Core | 2 wks | 4 wks |
| 4 | News | Core | 4 days | 4.5 wks |
| 5 | Agents Foundation | Core | 1 wk | 5.5 wks |
| 6 | Agent Scaling | Core | 2 wks | 7.5 wks |
| 7 | Technical Strategies | Core | 5 days | 8.5 wks |
| 8 | Statistical Strategies | Core | 1 wk | 9.5 wks |
| 9 | Simulation | Core | 1.5 wks | 11 wks |
| 10 | Storage | Core | 2 wks | 13 wks |
| 11 | Integration Testing | Core | 4 days | 13.8 wks |
| 12 | Scale Testing | Core | 1 wk | 14.8 wks |
| **Core Total** | | | **14.8 wks** ||
| **--- Parallel tracks start at 13 ---** |||||
| 13 (RL) | Gym Foundation | RL | 4 days | 4 days |
| 14 (RL) | Gym Observations | RL | 5 days | 1.8 wks |
| 15 (RL) | Gym Rewards | RL | 3 days | 2.4 wks |
| 16 (RL) | PyO3 Bindings | RL | 5 days | 3.4 wks |
| 17 (RL) | Training Scripts | RL | 1 wk | 4.4 wks |
| 18 (RL) | RL Agent Strategy | RL | 5 days | 5.4 wks |
| **RL Track Total** | | | **5.4 wks** ||
| 13 (G) | Services Foundation | Game | 4 days | 4 days |
| 14 (G) | Data Service | Game | 1 wk | 1.8 wks |
| 15 (G) | Game Service | Game | 1 wk | 2.8 wks |
| 16 (G) | Storage Service | Game | 3 days | 3.4 wks |
| 17 (G) | Chatbot Service | Game | 1 wk | 4.4 wks |
| 18 (G) | Game Frontend | Game | 2.5 wks | 6.9 wks |
| **Game Track Total** | | | **6.9 wks** ||
| 19 | RL Game Integration | Both | 1 wk | 1 wk |
| **Phase 19 Total** | | | **1 wk** ||

**Duration Totals:**

| Component | Duration |
|-----------|----------|
| Core (1-12) | 14.8 wks |
| RL Track (13-18 RL) | 5.4 wks |
| Game Track (13-18 G) | 6.9 wks |
| RL Game Integration (19) | 1 wk |

**Timeline Scenarios:**

| Scenario | Components | Duration | Demo |
|----------|------------|----------|------|
| Core Only | Core | **14.8 wks** | Validated simulation with scale testing |
| RL Only | Core + RL | **20.2 wks** | Train RL agents, no game UI |
| Game Only | Core + Game | **21.7 wks** | Full playable game vs AI strategies |
| Full (parallel dev) | Core + max(RL, Game) + 19 | **22.7 wks** | Everything (RL & Game in parallel) |
| Full (sequential) | Core + RL + Game + 19 | **28.1 wks** | If one person does everything sequentially |

---

# Part 19: MVP Milestones

| MVP | Phases | Demo | Portfolio Value |
|-----|--------|------|-----------------|
| MVP1 | 1-9 | "100k agents trade with realistic latency" | Solid |
| MVP2 | 1-10 | "Persistent market with risk management" | Strong |
| MVP3 | 1-12 | "Scale-tested simulation ready for extensions" | Strong+ |
| MVP4-RL | 1-12, 13-18 (RL) | "RL agent trained with observation parity" | Very Strong |
| MVP4-Game | 1-12, 13-18 (G) | "Playable trading game against AI strategies" | Very Strong |
| Full | All phases | "Play against trained RL quant bots" | Flagship |

**MVP Independence:**
- **MVP4-RL** and **MVP4-Game** are completely independent—choose based on priority
- Both start from the same MVP3 (validated, scale-tested core)
- **Full** requires both tracks for RL agents as game opponents

---

# Part 20: Context Requirements Per Phase

| Phase | Files to Load | Est. Lines |
|-------|---------------|------------|
| 1 | None | 0 |
| 2 | `types/lib.rs` | ~400 |
| 3 | `types/lib.rs` | ~400 |
| 4 | `types/lib.rs` | ~400 |
| 5 | `types`, `sim-core` traits, `quant` traits | ~700 |
| 6 | `agents/traits.rs`, tier structs | ~600 |
| 7-8 | `agents/traits.rs`, `types` | ~500 |
| 9 | All crate traits (not impls) | ~900 |
| 10 | `types`, `simulation/hooks.rs` | ~500 |
| 11 | Docker, all core crate interfaces | ~700 |
| 12 | Benchmarks, profiling tools | ~400 |
| **RL Track** |||
| 13 (RL) | `simulation/runner.rs`, `types` | ~600 |
| 14 (RL) | `gym/observation/traits.rs`, `contract.rs` | ~500 |
| 15 (RL) | `gym/reward/traits.rs`, `types` | ~400 |
| 16 (RL) | `gym` crate, PyO3 docs | ~700 |
| 17 (RL) | PyO3 module only | ~400 |
| 18 (RL) | `agents/traits.rs`, `gym/observation/contract.rs`, ONNX docs | ~600 |
| **Game Track** |||
| 13 (G) | `simulation/runner.rs`, tokio/axum docs | ~700 |
| 14 (G) | `services/common`, `quant`, `storage` public APIs | ~600 |
| 15 (G) | `services/common`, `sim-core`, `agents`, WebSocket docs | ~700 |
| 16 (G) | `services/common`, `storage` public API | ~400 |
| 17 (G) | `services/common`, LLM API docs, Data/Game/Storage endpoints | ~600 |
| 18 (G) | All game services, React docs | ~900 |
| **Final** |||
| 19 | Phase 18 (RL) outputs, Phase 18 (G) services | ~600 |

Each phase fits in one LLM context window.

---

# Part 21: Technical Talking Points

| Topic | What You Built |
|-------|----------------|
| Financial precision | `i64` fixed-point for all monetary values, no floating-point errors |
| Rust systems | Matching engine, order book, latency queue |
| Sync/async separation | Dedicated sim thread, channel bridge to async services |
| Trait-based design | Modular strategies, observations, rewards |
| Agent scaling | Tiered architecture for 100k+ agents in <10ms |
| Quant strategies | RSI, MACD, pairs trading, factor models |
| Risk management | VaR, position sizing, drawdown limits |
| Execution algos | VWAP, TWAP |
| Realistic simulation | Order latency prevents look-ahead bias |
| Parallelism | Rayon agent orchestration |
| Cross-language FFI | PyO3 Python bindings |
| Training parity | Observation contracts, bit-exact verification |
| RL environment design | Modular observation/reward composition |
| Reinforcement learning | DQN/PPO in multi-agent market |
| Microservices | Independent deployable services |
| Persistence patterns | WAL, event sourcing, checkpointing |
| Embedded databases | DuckDB + SQLite hybrid |
| LLM integration | Function-calling chatbot |

---

# Part 22: Key Technical Decisions Summary

| Decision | Rationale |
|----------|-----------|
| `i64` fixed-point over `f64` | No floating-point errors, fast integer ops, `Ord` for BTreeMap |
| Sync simulation, async services | Determinism + performance for sim, I/O efficiency for HTTP |
| Channel bridge | Clean separation without async pollution |
| Order latency queue | Prevents look-ahead bias in RL training |
| Observation contracts | Ensures Rust == Python for training parity |
| Tiered agent architecture | 100k+ agents with O(log n) wake conditions |
| Statistical background pool | Eliminates 90k+ individual agent instances |
| Sequential IDs over UUIDs | Memory efficiency and cache locality |
| ONNX for model exchange | Standard format, `ort` crate mature |
| DuckDB for analytics | Columnar storage optimal for OHLCV queries |
| SQLite for operational | Simple, reliable for key-value patterns |
| Decimal as TEXT in DB | Preserves precision, avoids DB float issues |
| Rayon for agents | Simple parallel iteration, no async overhead |

---
