# Quant Trading Gym — Executive Summary

## What It Is

A high-performance trading simulation that models realistic market microstructure with 100,000 agents trading across 10 symbols. Built in Rust for speed and reliability.

## Why It Matters

- **Strategy Testing**: Safe environment to test trading strategies before deploying real capital
- **Market Research**: Study price discovery, liquidity dynamics, and agent behavior
- **ML/RL Training**: Train reinforcement learning models against realistic market conditions

## Key Features

- **Multi-Symbol Trading**: currently 10 symbols across 5 different sectors
- **12 Agent Strategies**: Market makers, momentum, MACD, Bollinger, VWAP, pairs trading, sector rotation, ML ensemble agents
- **ML Agents (V6)**: Ensemble agents (RandomForest, LinearModel, SVM) with 28 SHAP-validated features
- **Tiered Architecture**: T1 (full logic), T2 (reactive), T3 (45k+ background agents)
- **News Events**: Earnings surprises, guidance changes, rate decisions, sector news
- **Fair Value Model**: Gordon Growth Model anchors prices to fundamentals
- **Dual Visualization**: Terminal UI (TUI) and Web Dashboard with real-time charts
- **Web Frontend**: React dashboard with candlestick charts, indicators, agent explorer, risk metrics

## Quick Start

**Option A: Web Dashboard** (full visualization)
```bash
docker compose up
```
Open http://localhost:80 — real-time charts, agent explorer, risk metrics.

**Option B: Terminal UI via Docker** (no Rust needed)
```bash
docker compose -f docker-compose.tui.yaml up
```
Open http://localhost:7681 for web-based terminal.

**Option C: Terminal UI** (requires Rust)
```bash
cargo build --release
cargo run --release          # Press Space to start simulation
```

## Current State

**V6 Complete** — Feature engineering with SHAP-validated 28 canonical features, ensemble ML agents (RandomForest, LinearModel, SVM), training-serving parity via feature extractor traits.

**V7 Planned** — Gym environment + PyO3 bindings (V7.1) for RL training infrastructure, then RL training with PPO/A2C (V7.2).

## Technology

- **Language**: Rust (100% safe, no external dependencies for core)
- **Precision**: Fixed-point arithmetic for financial accuracy
- **Architecture**: Modular crates (types, sim-core, agents, news, quant, simulation, tui, storage, server)
- **Parallelism**: Rayon-based parallel agent execution with batch auction
- **Persistence**: SQLite for trade history, candle aggregation, portfolio snapshots
- **Web Stack**: Axum 0.8 (async server), React 19, Vite 6, Tailwind CSS, TypeScript
- **Real-time**: WebSocket tick streaming, REST API for analytics/portfolio/risk data
