# Multi-stage Dockerfile for quant-trading-gym (server mode)
# V4.2: HTTP/WebSocket server for React frontend

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

# Default to server mode
ENV SIM_SERVER=true
ENV SIM_SERVER_PORT=8001

# Expose the server port
EXPOSE 8001

# Run the simulation in server mode
ENTRYPOINT ["/app/quant-trading-gym"]
