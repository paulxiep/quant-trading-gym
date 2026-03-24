# Dockerfile for quant-trading-gym with ttyd (browser-accessible TUI)
# V3.7: Run TUI in browser without requiring local Rust installation

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
# Runtime stage: Debian with ttyd for web-based terminal
# ─────────────────────────────────────────────────────────────────────────────
FROM debian:trixie-slim

# Install ttyd from GitHub releases (auto-detect architecture)
ARG TARGETARCH
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && TTYD_ARCH=$(case ${TARGETARCH} in amd64) echo "x86_64";; arm64) echo "aarch64";; *) echo "x86_64";; esac) \
    && curl -sSL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.${TTYD_ARCH} -o /usr/local/bin/ttyd \
    && chmod +x /usr/local/bin/ttyd \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary
COPY --from=builder /build/target/release/quant-trading-gym /app/quant-trading-gym

WORKDIR /app

# Default configuration (TUI mode)
ENV SIM_TICKS=5000
ENV TERM=xterm-256color

# Expose ttyd web port
EXPOSE 7681

# Run ttyd serving the TUI application
# -W: writable (allows input)
# -t: terminal options for proper color support
CMD ["ttyd", "-W", "-t", "titleFixed=Quant Trading Gym", "-t", "fontSize=14", "/app/quant-trading-gym"]
