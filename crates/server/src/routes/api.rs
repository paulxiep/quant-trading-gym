//! REST API endpoints (V4.2 foundation, V4.3 full implementation).
//!
//! Placeholder module for REST endpoints.
//! V4.2 provides structure; V4.3 adds full analytics/portfolio/risk APIs.
//!
//! # Planned Endpoints (V4.3)
//!
//! - `GET /api/status` - Current simulation state
//! - `POST /api/command` - Send command to simulation
//! - `GET /api/presets` - List config presets
//! - `GET /api/presets/:name` - Get specific preset
//! - `POST /api/presets` - Save custom preset
//!
//! # Design Principles
//!
//! - **Declarative**: Each endpoint handler is a pure function
//! - **Modular**: Endpoints grouped by domain (status, presets, etc.)
//! - **SoC**: Handlers extract state, return responses

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::bridge::SimCommand;
use crate::error::{AppError, AppResult};
use crate::state::ServerState;

/// Simulation status response.
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    /// Current tick.
    pub tick: u64,
    /// Whether simulation is running.
    pub running: bool,
    /// Whether simulation has finished.
    pub finished: bool,
    /// Total agent count.
    pub agents: u64,
}

/// Get simulation status: `GET /api/status`
pub async fn get_status(State(state): State<ServerState>) -> Json<StatusResponse> {
    let metrics = &state.metrics;

    Json(StatusResponse {
        tick: metrics.tick(),
        running: metrics.is_running(),
        finished: metrics.is_finished(),
        agents: metrics.agents(),
    })
}

/// Command request body.
#[derive(Debug, serde::Deserialize)]
pub struct CommandRequest {
    /// Command to send.
    pub command: SimCommand,
}

/// Command response.
#[derive(Debug, Serialize)]
pub struct CommandResponse {
    /// Whether command was sent.
    pub ok: bool,
}

/// Send command to simulation: `POST /api/command`
pub async fn post_command(
    State(state): State<ServerState>,
    Json(req): Json<CommandRequest>,
) -> AppResult<Json<CommandResponse>> {
    state
        .send_command(req.command)
        .map_err(|_| AppError::Unavailable("Simulation not connected".into()))?;

    Ok(Json(CommandResponse { ok: true }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_response_serialization() {
        let response = StatusResponse {
            tick: 500,
            running: true,
            finished: false,
            agents: 25000,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"tick\":500"));
        assert!(json.contains("\"running\":true"));
    }

    #[test]
    fn test_command_request_parsing() {
        let json = r#"{"command": "Start"}"#;
        let req: CommandRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.command, SimCommand::Start));
    }
}
