//! Unified error handling for the server (V4.2).
//!
//! Provides a single error type that maps to HTTP responses.
//!
//! # Design Principles
//!
//! - **Declarative**: Each error variant declares its HTTP status code
//! - **Modular**: Error type is self-contained with IntoResponse impl
//! - **SoC**: Error handling separate from business logic

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Application error type with HTTP response mapping.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// Resource not found (404).
    #[error("Not found: {0}")]
    NotFound(String),

    /// Invalid request data (400).
    #[error("Bad request: {0}")]
    BadRequest(String),

    /// Internal server error (500).
    #[error("Internal error: {0}")]
    Internal(String),

    /// Service unavailable (503).
    #[error("Service unavailable: {0}")]
    Unavailable(String),

    /// Simulation not ready (503).
    #[error("Simulation not ready")]
    SimulationNotReady,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::Unavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg.clone()),
            AppError::SimulationNotReady => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Simulation not ready".into(),
            ),
        };

        let body = axum::Json(json!({
            "error": message,
            "status": status.as_u16()
        }));

        (status, body).into_response()
    }
}

/// Result type alias for handlers.
pub type AppResult<T> = Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AppError::NotFound("preset xyz".into());
        assert_eq!(err.to_string(), "Not found: preset xyz");
    }

    #[test]
    fn test_error_variants() {
        let _ = AppError::NotFound("test".into());
        let _ = AppError::BadRequest("test".into());
        let _ = AppError::Internal("test".into());
        let _ = AppError::Unavailable("test".into());
        let _ = AppError::SimulationNotReady;
    }
}
