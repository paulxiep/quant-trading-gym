//! TUI crate: Terminal User Interface for the Quant Trading Gym.
//!
//! This crate provides real-time visualization of the trading simulation:
//! - Live price chart (line graph)
//! - Order book depth visualization
//! - Agent P&L summary table
//! - Simulation statistics
//!
//! # Architecture
//!
//! The TUI runs in a separate thread from the simulation, communicating via channels:
//!
//! ```text
//! ┌────────────────┐     SimUpdate      ┌────────────────┐
//! │   Simulation   │ ────────────────►  │      TUI       │
//! │   (Thread A)   │   (channel)        │   (Thread B)   │
//! │                │ ◄────────────────  │                │
//! └────────────────┘     SimCommand     └────────────────┘
//! ```
//!
//! This prevents slow terminal rendering from blocking the matching engine.
//! The TUI can send commands (start/stop) back to the simulation.
//!
//! # Usage
//!
//! ```ignore
//! use tui::{TuiApp, SimUpdate, SimCommand};
//! use crossbeam_channel::unbounded;
//!
//! // Create channels
//! let (update_tx, update_rx) = unbounded();
//! let (cmd_tx, cmd_rx) = unbounded();
//!
//! // Start TUI in main thread
//! let app = TuiApp::new(update_rx).with_command_sender(cmd_tx);
//! app.run();
//!
//! // Simulation checks cmd_rx for start/stop commands
//! ```

mod app;
mod widgets;

pub use app::{SimpleTui, TuiApp, check_quit};
pub use widgets::{AgentInfo, RiskInfo, SimUpdate};

/// Commands sent from TUI to simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimCommand {
    /// Start or resume the simulation.
    Start,
    /// Pause the simulation.
    Pause,
    /// Toggle between running and paused.
    Toggle,
    /// Quit the simulation.
    Quit,
}
