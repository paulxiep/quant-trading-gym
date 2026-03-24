# Quant Trading Gym — Technical Summary

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    simulation crate                      │
│  (tick loop, event processing, order routing)           │
├─────────────┬─────────────┬─────────────┬───────────────┤
│   agents    │  sim-core   │    news     │     quant     │
│  (traits,   │ (order book,│ (events,    │ (indicators,  │
│  strategies)│  matching)  │ fundamentals)│  risk metrics)│
├─────────────┴─────────────┴─────────────┴───────────────┤
│                      types crate                         │
│        (Order, Trade, Price, Symbol, Sector)            │
└─────────────────────────────────────────────────────────┘
│  tui crate  ││  storage crate  ││     server crate      │
│ (terminal)  ││ (SQLite persist)││ (Axum REST/WebSocket) │
└─────────────┘└─────────────────┘└───────────────────────┘
                                  │    React Frontend     │
                                  │ (Vite/Tailwind/TypeScript)
                                  └───────────────────────┘
```

## Crate Responsibilities

| Crate | Purpose | Key Files |
|-------|---------|-----------|
| `types` | Shared types, no logic | `order.rs`, `config.rs` |
| `sim-core` | Order book, matching engine, batch auction | `order_book.rs`, `batch_auction.rs` |
| `agents` | Agent trait + tiered strategies | `strategies/`, `tier2/`, `tier3/` |
| `news` | Events, fundamentals, sectors | `generator.rs`, `fundamentals.rs` |
| `quant` | Indicators, risk metrics | `indicators/`, `tracker.rs` |
| `simulation` | Tick loop, parallel execution, hooks | `runner.rs`, `parallel.rs`, `hooks.rs` |
| `storage` | SQLite persistence, candle aggregation | `schema.rs`, `candles.rs`, `hook.rs` |
| `tui` | Terminal UI | `app.rs`, `widgets/` |
| `server` | Axum HTTP/WebSocket server | `app.rs`, `routes/`, `hooks.rs` |
| `frontend` | React web dashboard | `pages/`, `components/`, `hooks/` |

## Simulation Loop (V3 Two-Phase Architecture)

```
for tick in 0..max_ticks:
    Phase 1: Collection (parallel)
      1. Generate news events (probabilistic)
      2. Apply permanent fundamentals changes
      3. Build StrategyContext per symbol
      4. Collect orders from T1/T2/T3 agents in parallel (rayon)
      5. Validate against position limits

    Phase 2: Auction (parallel per-symbol)
      6. Reference-price auction: match crossing orders at volume-weighted seller price
      7. Notify agents of fills
      8. Update candles, indicators, risk metrics
      9. Invoke SimulationHooks (storage, metrics, TUI)
```

## Agent Strategies

| Strategy | Tier | Signal | Behavior |
|----------|------|--------|----------|
| MarketMaker | T1 | Always | Two-sided quotes around mid |
| NoiseTrader | T1 | Random | Random buys/sells near fair value |
| Momentum | T1 | RSI < 30 / > 70 | Buy oversold, sell overbought |
| TrendFollower | T1 | SMA crossover | Buy golden cross, sell death cross |
| MacdCrossover | T1 | MACD/Signal | Buy bullish cross, sell bearish |
| BollingerReversion | T1 | Band touch | Buy lower band, sell upper band |
| VwapExecutor | T1 | Time-sliced | Execute target qty over horizon |
| PairsTrader | T1 | Spread z-score | Long/short cointegrated pairs |
| EnsembleAgent (ML) | T1 | Model ensemble | RandomForest + LinearModel + SVM (28 features) |
| SectorRotator | T2 | Sector sentiment | Rotate allocation on news events |
| ThresholdTrader | T2 | Price threshold | Wake on price cross |
| BackgroundPool | T3 | Statistical | 45k+ agents via aggregate modeling |

## Key Design Decisions

1. **Reference-price auction**: Statistically equivalent to individual matching; volume-weighted price is deterministic, not arrival-order dependent
2. **Tiered agent architecture**: T1 (full logic), T2 (reactive/wake conditions), T3 (statistical background pool)
3. **Mean-reverting prices**: Realistic for tick-level liquid markets; momentum strategies struggle (as in real HFT)
4. **Fixed-point arithmetic**: `Price` and `Cash` use i64 with implicit decimals for financial precision
5. **Event-value-first generation**: Events generate magnitude before selecting symbol to prevent seed-based bias
6. **Growth cap at 7%**: Prevents Gordon Growth Model breakdown when g ≥ r
7. **ML ensemble agents (V6)**: SHAP-validated 28 canonical features; ensemble models (RandomForest, LinearModel, SVM); training-serving parity via `FeatureExtractor` trait; semantic neutral values for imputation

## Build & Run

### Web Dashboard (full visualization)
```bash
docker compose up                         # Production (localhost:80)
docker compose -f docker-compose.dev.yaml up  # Development with hot reload (localhost:5173)
```

### Terminal UI via Docker (no Rust needed)
```bash
docker compose -f docker-compose.tui.yaml up
```
Open http://localhost:7681 for web-based terminal.

### Terminal UI (requires Rust)
```bash
cargo build --release
cargo run --release                              # TUI (Space to start)
cargo run --release -- --headless --ticks 10000  # Headless benchmark
```

### Development Commands
```bash
cargo test --all                     # Run all tests
docker compose -f docker-compose.frontend.yaml run --rm typecheck        # TypeScript check
docker compose -f docker-compose.frontend.yaml run --rm integration-test # API tests
```

## TUI Controls

| Key | Action |
|-----|--------|
| `Space` | Start/Pause |
| `Tab`/`1-4` | Switch symbol |
| `O` | Overlay mode |
| `↑↓` | Scroll |
| `q` | Quit |
