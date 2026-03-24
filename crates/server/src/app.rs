//! Axum application builder (V4.2, extended V4.3).
//!
//! Configures routes, middleware, and state for the server.
//!
//! # Design Principles
//!
//! - **Declarative**: Routes declared via Axum's type-safe Router
//! - **Modular**: App builder separate from handlers
//! - **SoC**: Configuration here, logic in route modules
//!
//! # V4.3 Routes
//!
//! Data Service endpoints:
//! - `GET /api/analytics/candles` - OHLCV candles
//! - `GET /api/analytics/indicators` - Technical indicators
//! - `GET /api/analytics/factors` - Factor scores
//! - `GET /api/portfolio/agents` - Agent list with P&L
//! - `GET /api/portfolio/agents/:agent_id` - Agent details
//! - `GET /api/risk/:agent_id` - Risk metrics
//! - `GET /api/news/active` - Active news events

use axum::Router;
use axum::routing::{get, post};
use std::time::Duration;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::routes::{api, data, health, ws};
use crate::state::ServerState;

/// Create the Axum application with all routes.
pub fn create_app(state: ServerState) -> Router {
    // CORS layer for frontend development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .max_age(Duration::from_secs(3600));

    Router::new()
        // Health endpoints
        .route("/health", get(health::health))
        .route("/health/ready", get(health::ready))
        // WebSocket endpoint
        .route("/ws", get(ws::ws_handler))
        // API endpoints (V4.2)
        .route("/api/status", get(api::get_status))
        .route("/api/command", post(api::post_command))
        // V4.5 Data Service: Symbols
        .route("/api/symbols", get(data::get_symbols))
        // V4.3 Data Service: Analytics
        .route("/api/analytics/candles", get(data::get_candles))
        .route("/api/analytics/indicators", get(data::get_indicators))
        .route("/api/analytics/factors", get(data::get_factors))
        // V4.3 Data Service: Portfolio
        .route("/api/portfolio/agents", get(data::get_agents))
        .route(
            "/api/portfolio/agents/{agent_id}",
            get(data::get_agent_portfolio),
        )
        // V4.3 Data Service: Risk
        .route("/api/risk/{agent_id}", get(data::get_risk_metrics))
        // V4.4 Data Service: Aggregate Risk
        .route("/api/risk/aggregate", get(data::get_aggregate_risk))
        // V4.4 Data Service: Order Distribution
        .route(
            "/api/analytics/order-distribution",
            get(data::get_order_distribution),
        )
        // V4.3 Data Service: News
        .route("/api/news/active", get(data::get_active_news))
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        // State
        .with_state(state)
}

/// Server configuration.
pub struct ServerConfig {
    /// Port to listen on.
    pub port: u16,
    /// Host to bind to.
    pub host: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8001,
            host: "0.0.0.0".into(),
        }
    }
}

impl ServerConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let port = std::env::var("SIM_SERVER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8001);

        let host = std::env::var("SIM_SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".into());

        Self { port, host }
    }

    /// Get bind address.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8001);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.bind_addr(), "0.0.0.0:8001");
    }

    #[test]
    fn test_create_app() {
        let (tick_tx, _) = broadcast::channel(16);
        let (cmd_tx, _) = crossbeam_channel::unbounded();
        let state = ServerState::new(tick_tx, cmd_tx);

        let _app = create_app(state);
        // App created successfully
    }
}
