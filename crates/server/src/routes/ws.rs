//! WebSocket handlers for real-time updates (V4.2).
//!
//! Broadcasts tick data to connected clients.
//!
//! # Endpoints
//!
//! - `GET /ws` - WebSocket upgrade for tick stream
//!
//! # Protocol
//!
//! After connection, client receives JSON-serialized TickData on each tick.
//! Client can send SimCommand messages to control simulation.
//!
//! # Design Principles
//!
//! - **Declarative**: Message types define protocol
//! - **Modular**: WebSocket logic isolated from HTTP routes
//! - **SoC**: Handler manages connection, state provides channels

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures::{SinkExt, StreamExt};
use tracing::{debug, warn};

use crate::bridge::SimCommand;
use crate::state::ServerState;

/// WebSocket upgrade handler: `GET /ws`
///
/// Upgrades HTTP connection to WebSocket for real-time tick stream.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle a WebSocket connection.
async fn handle_socket(socket: WebSocket, state: ServerState) {
    state.metrics.ws_connect();
    debug!("WebSocket client connected");

    let (mut sender, mut receiver) = socket.split();

    // Subscribe to tick updates
    let mut tick_rx = state.subscribe_ticks();

    // Spawn task to forward tick updates to client
    let send_task = tokio::spawn(async move {
        loop {
            match tick_rx.recv().await {
                Ok(tick_data) => {
                    match serde_json::to_string(&tick_data) {
                        Ok(json) => {
                            if sender.send(Message::Text(json.into())).await.is_err() {
                                break; // Client disconnected
                            }
                        }
                        Err(e) => {
                            warn!("Failed to serialize tick data: {}", e);
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    debug!("WebSocket client lagged by {} messages", n);
                    // Continue, client will get next message
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    break; // Channel closed
                }
            }
        }
    });

    // Handle incoming messages (commands from client)
    let cmd_tx = state.cmd_tx.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(cmd) = serde_json::from_str::<SimCommand>(&text) {
                        if cmd_tx.send(cmd).is_err() {
                            break; // Simulation disconnected
                        }
                    } else {
                        debug!("Invalid command from client: {}", text);
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    warn!("WebSocket error: {}", e);
                    break;
                }
                _ => {} // Ignore ping/pong/binary
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    state.metrics.ws_disconnect();
    debug!("WebSocket client disconnected");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_command_parsing() {
        let json = r#""Start""#;
        let cmd: SimCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, SimCommand::Start));

        let json = r#""Pause""#;
        let cmd: SimCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, SimCommand::Pause));
    }
}
