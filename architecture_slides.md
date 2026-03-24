# Quant Trading Gym — Architecture Slides

---

## Slide 1: System Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                QUANT TRADING GYM — SYSTEM OVERVIEW                               │
│                                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐             │
│  │                          RUST SIMULATION ENGINE                                  │             │
│  │                                                                                  │             │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────────┐  ┌──────────┐  │             │
│  │  │  types   │  │  sim-core │  │  quant   │  │    agents         │  │   news   │  │             │
│  │  │ ──────── │  │ ────────  │  │ ──────── │  │ ───────────────── │  │ ──────── │  │             │
│  │  │ Order    │  │ OrderBook │  │ SMA, EMA │  │ T1: 9 strategies  │  │ Earnings │  │             │
│  │  │ Trade    │  │ Market    │  │ RSI,MACD │  │ T2: Reactive      │  │ Guidance │  │             │
│  │  │ Price    │  │ Batch     │  │ Bollinger│  │ T3: 45k+ pool     │  │ Rates    │  │             │
│  │  │ Features │  │ Auction   │  │ ATR, VaR │  │ ML Agent+Models   │  │ Sector   │  │             │
│  │  └──────────┘  └───────────┘  └──────────┘  └───────────────────┘  └──────────┘  │             │
│  │                                                                                  │             │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────────┐  ┌────────────┐ │             │
│  │  │simulation│  │  storage  │  │ parallel │  │     server        │  │tui(ratatui)│ │             │
│  │  │ ──────── │  │ ───────── │  │ ──────── │  │ ───────────────── │  │ ────────── │ │             │
│  │  │ 14-phase │  │ SQLite    │  │ rayon    │  │ Axum REST API     │  │ Terminal   │ │             │
│  │  │ tick loop│  │ Parquet   │  │ abstrac- │  │ WebSocket ticks   │  │ crossbeam  │ │             │
│  │  │ Hooks    │  │ Recording │  │ tion     │  │ ~15 endpoints     │  │ channels   │ │             │
│  │  └──────────┘  └─────┬─────┘  └──────────┘  └────────┬──────────┘  └────────────┘ │             │
│  │                      │ .parquet        │ WS/REST                                 │             │
│  └──────────────────────┼─────────────────┼─────────────────────────────────────────┘             │
│                         │                 │                                                       │
│                         ▼                 ▼                                                       │
│  ┌────────────────────────────────────────────────────┐   ┌─────────────────────────────────────┐ │
│  │           PYTHON ML TRAINING PIPELINE              │   │         REACT + TS FRONTEND         │ │
│  │  feature_schema ─► feature_engineering             │   │  Dashboard │ Charts │ Portfolio     │ │
│  │  feature_selection (SHAP) ─► train_models          │   │  Risk metrics │ News event feed    │ │
│  │  Trees / Linear / SVM / NaiveBayes / Ensemble      │   └─────────────────────────────────────┘ │
│  │           │                                        │                                           │
│  │           └──── .json models ──► agents crate      │                                           │
│  └────────────────────────────────────────────────────┘                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- 10 Rust crates with strict separation of concerns — no god objects
- Trait-based polymorphism: `Agent`, `MlModel`, `FeatureExtractor`, `SimulationHook`
- Fixed-point arithmetic (`Price`/`Cash` as i64, 10000 units = $1) — no float rounding
- Python trains models offline → JSON export → Rust inference at runtime
- React dashboard connects via Axum WebSocket for live tick streaming

---

## Slide 2: Tick Loop (14 Phases)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            SIMULATION TICK LOOP (14 PHASES)                             │
│                                                                                         │
│  ═══ COLLECTION ══════════════════════════════════════════════════════════════════════  │
│                                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │ Phase 0  │  │ Phase 0b │  │    Phase 1      │  │         Phase 2                  │  │
│  │ News     ├─►│ Fair val ├─►│ Hook:           ├─►│  Determine active agents         │  │
│  │ events   │  │ drift    │  │ on_tick_start   │  │  T1: always  T2: wake-condition  │  │
│  └──────────┘  └──────────┘  └─────────────────┘  └──────────────┬───────────────────┘  │
│                                                                  ▼                      │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────────────────────┐  │
│  │ Phase 3                     │  │ Phase 3b                                         │  │
│  │ Build StrategyContext       ├─►│ Extract features ─► impute NaN ─► ML pred cache  │  │
│  │ (candles, indicators, etc.) │  │ (same features go to hooks for recording parity) │  │
│  └─────────────────────────────┘  └──────────────────────┬───────────────────────────┘  │
│                                                          ▼                              │
│  ┌───────────────────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ Phase 4  [PARALLEL]       │  │  Phase 5    │  │ Phase 5b     │  │   Phase 6      │   │
│  │ T1+T2 Agent.on_tick()     ├─►│  Collect    ├─►│ T3 pool      ├─►│ Hook:          │   │
│  │ → AgentAction (orders)    │  │  all orders │  │ statistical  │  │ orders_collect │   │
│  └───────────────────────────┘  └─────────────┘  └──────────────┘  └───────┬────────┘   │
│                                                                            ▼            │
│  ═══ AUCTION ═════════════════════════════════════════════════════════════════════════  │
│                                                                                         │
│  ┌─────────────────────────────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Phase 7  [PARALLEL per-symbol]      │  │  Phase 8     │  │  Phase 9               │  │
│  │ Reference-Price Batch Auction       ├─►│  T3 pool     ├─►│  Hook: on_trades       │  │
│  │ WAP clearing price, pro-rata fill   │  │  accounting  │  │                        │  │
│  └─────────────────────────────────────┘  └──────────────┘  └───────────┬────────────┘  │
│                                                                         ▼               │
│  ═══ POST-AUCTION ════════════════════════════════════════════════════════════════════  │
│                                                                                         │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌────────────┐ ┌──────────┐  │
│  │  Phase 10    │  │  Phase 11        │  │  Phase 12    │  │ Phase 13   │ │ Phase 14 │  │
│  │  Update      ├─►│  Agent.on_fill() ├─►│  Risk        ├─►│ Hook:      ├►│ Finalize │  │
│  │  candles,    │  │  [PARALLEL]      │  │  tracking    │  │ on_tick_end│ │ tick     │  │
│  │  trades      │  │  + wake conds    │  │  [PARALLEL]  │  │            │ │          │  │
│  └──────────────┘  └──────────────────┘  └──────────────┘  └────────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

- 14 sequential phases with explicit data dependencies
- Parallelism via rayon at 4 points: agent collection, auctions, fill notifications, risk tracking
- Feature extraction happens once (Phase 3b), shared between ML inference and recording hooks — guarantees training-serving parity
- Batch auction clears per-symbol independently (parallel), uses WAP clearing price with pro-rata fill on oversubscribed side

---

## Slide 3: Tiered Agent & ML Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        TIERED AGENT ARCHITECTURE + ML PIPELINE                           │
│                                                                                          │
│  ┌─ TIER 1: Smart (every tick, ~800 agents) ──────────────────────────────────────────┐  │
│  │  NoiseTrader │ MarketMaker │ MomentumTrader │ TrendFollower │ MacdCrossover        │  │
│  │  BollingerReversion │ VwapExecutor │ PairsTrading │ MlAgent                        │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│  ┌─ TIER 2: Reactive (wake-on-condition, ~5k agents) ─────────────────────────────────┐  │
│  │  ReactiveAgent (generic) │ SectorRotator (news-triggered rebalance)                │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│  ┌─ TIER 3: Background Pool (statistical model, 45k-100k+ implicit agents) ───────────┐  │
│  │  NOT individually simulated — aggregate order generation from statistical model    │  │
│  │  Log-normal sizes │ sentiment-biased direction │ regime-dependent frequency        │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│  ═══ ML INFERENCE (Rust) ═══════════        ═══ ML TRAINING (Python) ═══════════════     │
│                                                                                          │
│  ┌──────────────────────────────┐          ┌──────────────────────────────────────┐      │
│  │ FeatureExtractor             │          │  Parquet data (6 runs × 40k rows)    │      │
│  │  Minimal: 42 features (V5)   │          │           ▼                          │      │
│  │  Full:    55 features (V6.1) │          │  feature_schema.py (group defs)      │      │
│  │  Canon:   28 features (V6.3) │          │           ▼                          │      │
│  └─────────────┬────────────────┘          │  feature_selection.py (SHAP trim)    │      │
│                ▼                           │  55 → 28 canonical features          │      │
│  ┌─────────────────────────────┐           │           ▼                          │      │
│  │ MlModel trait (ensemble)    │           │  feature_engineering.py              │      │
│  │  DecisionTree               │           │  (interactions, squares, ratios)     │      │
│  │  RandomForest               │◄──.json───│           ▼                          │      │
│  │  GradientBoosted            │           │  train_trees / linear / svm / nb     │      │
│  │  LinearPredictor            │           │           ▼                          │      │
│  │  GaussianNBPredictor        │           │  Accuracy-weighted ensemble config   │      │
│  │  EnsembleModel (weighted)   │           └──────────────────────────────────────┘      │
│  └─────────────┬───────────────┘                                                         │
│                ▼                                                                         │
│  ┌──────────────────────────────┐                                                        │
│  │ MlAgent decision logic       │                                                        │
│  │ [p_sell, p_hold, p_buy]      │                                                        │
│  │    ─► threshold ─► Order     │                                                        │
│  └──────────────────────────────┘                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

- Tiered scaling: 800 smart + 5k reactive + 45k+ statistical = realistic market without O(n) cost
- T1 runs full strategy logic each tick; T2 only wakes on indexed conditions; T3 never individually simulated
- 3 feature registries (42/55/28) with semantic neutral values for NaN imputation — single source of truth in `types` crate
- SHAP analysis trimmed 55 → 28 canonical features at 96%+ accuracy retention
- Python trains offline → JSON model export → Rust `ModelRegistry` loads at startup

---

## Slide 4: Data & Observability

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                          DATA PERSISTENCE & OBSERVABILITY                                 │
│                                                                                          │
│  ┌─ SimulationHook (plugin observer pattern) ────────────────────────────────────────┐   │
│  │                                                                                    │   │
│  │  MetricsHook ──► order/trade counts, tick history (lock-free atomics)              │   │
│  │  StorageHook ──► SQLite: trades│candles│snapshots (batched writes)                 │   │
│  │  RecordingHook ► Parquet: per-tick features for ML training                        │   │
│  │  BroadcastHook ► tokio broadcast channel → WebSocket clients                       │   │
│  │                                                                                    │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌─ Server Endpoints (Axum) ──────────────────┐  ┌─ Frontend (React + TS) ───────────┐  │
│  │  /api/status          sim state             │  │  Live dashboard (WebSocket)       │  │
│  │  /api/analytics/*     candles, indicators,  │  │  OHLCV charts                    │  │
│  │                       factors, order dist   │  │  Portfolio & agent P&L            │  │
│  │  /api/portfolio/*     agent positions, P&L  │  │  Risk metrics (Sharpe, VaR)      │  │
│  │  /api/risk/*          Sharpe, VaR, drawdown │  │  News event feed                 │  │
│  │  /api/news/active     live events           │  │                                  │  │
│  │  /ws                  tick stream (JSON)    ├──►                                  │  │
│  │  /api/command         Start│Pause│Stop      │  │                                  │  │
│  └────────────────────────────────────────────┘  └──────────────────────────────────┘   │
│                                                                                          │
│  ┌─ TUI (ratatui) ───────────────────────────────────────────────────────────────────┐   │
│  │  Terminal visualization — separate thread, crossbeam channels to simulation       │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

- Hook pattern decouples observation from simulation — add new observers without changing core
- SQLite for runtime persistence (trades, candles, snapshots); Parquet for ML training data export
- Server bridges sync Rust simulation to async WebSocket via tokio broadcast channel
- ~15 REST endpoints cover analytics, portfolio, risk, news, and simulation control
- CLI supports 20+ args including 8 granular parallelization toggles for profiling

---

## Summary

- **Rust core** — 10 crates, fixed-point arithmetic, rayon parallelism
- **100k+ agent scale** — 3-tier architecture (smart / reactive / statistical) keeps cost at O(k) not O(n)
- **Reference-price batch auction** — deterministic WAP clearing, parallel per-symbol
- **ML ensemble** — 5 model types, SHAP-validated 28-feature canonical set, training-serving parity
- **Plugin observability** — hooks for metrics, storage, recording, WebSocket broadcast
- **Full-stack** — React dashboard + Axum REST/WS API + terminal TUI
