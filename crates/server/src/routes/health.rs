//! Health check endpoints (V4.2).
//!
//! Provides liveness and readiness probes for the server.
//!
//! # Endpoints
//!
//! - `GET /health` - Liveness probe (always 200 if server is up)
//! - `GET /health/ready` - Readiness probe (200 if simulation is running)
//!
//! # Design Principles
//!
//! - **Declarative**: Response types define the contract
//! - **Modular**: Health checks independent of other routes
//! - **SoC**: Handlers only read state, don't modify

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::state::ServerState;

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Health status.
    pub status: &'static str,
    /// Current simulation tick.
    pub tick: u64,
    /// Total agent count.
    pub agents: u64,
    /// Server uptime in seconds.
    pub uptime_secs: u64,
    /// Active WebSocket connections.
    pub ws_connections: u64,
}

/// Readiness check response.
#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    /// Whether server is ready.
    pub ready: bool,
    /// Readiness reason.
    pub reason: &'static str,
    /// Simulation running state.
    pub sim_running: bool,
    /// Simulation finished state.
    pub sim_finished: bool,
}

/// Liveness probe: `GET /health`
///
/// Returns 200 if the server is running.
pub async fn health(State(state): State<ServerState>) -> Json<HealthResponse> {
    let metrics = &state.metrics;

    Json(HealthResponse {
        status: "healthy",
        tick: metrics.tick(),
        agents: metrics.agents(),
        uptime_secs: state.uptime_secs(),
        ws_connections: metrics.ws_count(),
    })
}

/// Readiness probe: `GET /health/ready`
///
/// Returns 200 with ready=true if simulation is available.
pub async fn ready(State(state): State<ServerState>) -> Json<ReadyResponse> {
    let metrics = &state.metrics;
    let running = metrics.is_running();
    let finished = metrics.is_finished();

    let (ready, reason) = if finished {
        (true, "simulation finished")
    } else if running {
        (true, "simulation running")
    } else if metrics.tick() > 0 {
        (true, "simulation paused")
    } else {
        (false, "waiting for simulation")
    };

    Json(ReadyResponse {
        ready,
        reason,
        sim_running: running,
        sim_finished: finished,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy",
            tick: 100,
            agents: 25000,
            uptime_secs: 60,
            ws_connections: 5,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"tick\":100"));
    }

    #[test]
    fn test_ready_response_serialization() {
        let response = ReadyResponse {
            ready: true,
            reason: "simulation running",
            sim_running: true,
            sim_finished: false,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"ready\":true"));
    }
}
