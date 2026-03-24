//! Route handlers for the server (V4.2, extended V4.3).
//!
//! # Modules
//!
//! - [`health`]: Health and readiness endpoints
//! - [`ws`]: WebSocket handlers for real-time updates
//! - [`api`]: REST API endpoints for simulation control
//! - [`data`]: V4.3 Data Service endpoints (analytics, portfolio, risk, news)

pub mod api;
pub mod data;
pub mod health;
pub mod ws;
