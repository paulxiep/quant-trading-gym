# Multi-stage Dockerfile for quant-trading-gym (headless mode)
# V3.7: Distroless production image for benchmarks and CI

# ─────────────────────────────────────────────────────────────────────────────
# Builder stage: compile the Rust application
# ─────────────────────────────────────────────────────────────────────────────
FROM rust:1.92-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy everything and build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY crates ./crates

RUN cargo build --release

# ─────────────────────────────────────────────────────────────────────────────
# Runtime stage: minimal distroless image
# ─────────────────────────────────────────────────────────────────────────────
FROM gcr.io/distroless/cc-debian13:nonroot

# Copy the compiled binary
COPY --from=builder /build/target/release/quant-trading-gym /app/quant-trading-gym

WORKDIR /app

# Default to headless mode with 10k ticks
ENV SIM_HEADLESS=true
ENV SIM_TICKS=10000

# Run the simulation
ENTRYPOINT ["/app/quant-trading-gym"]
