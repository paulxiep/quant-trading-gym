//! Borrow ledger for tracking short-selling borrows.
//!
//! The `BorrowLedger` tracks which agents have borrowed shares for short selling,
//! how many shares are available in the borrow pool, and manages the lifecycle
//! of borrow positions.
//!
//! # Architecture
//!
//! The borrow pool is derived from `SymbolConfig::borrow_pool_size()`, which is
//! a fraction of shares_outstanding. When an agent opens a short position, they
//! must first borrow shares from this pool. The ledger ensures:
//!
//! 1. Agents cannot borrow more than available
//! 2. Agents cannot exceed their individual max_short limit
//! 3. Borrows are tracked for cost calculation
//!
//! # Example
//!
//! ```ignore
//! use agents::BorrowLedger;
//! use types::{AgentId, Quantity, Tick};
//!
//! let mut ledger = BorrowLedger::new(Quantity(1_000_000)); // 1M shares available
//!
//! // Agent borrows 1000 shares
//! ledger.borrow(AgentId(1), "AAPL", Quantity(1000), 0).unwrap();
//!
//! // Check available
//! assert_eq!(ledger.available("AAPL"), Quantity(999_000));
//!
//! // Return shares when closing short
//! ledger.return_shares(AgentId(1), "AAPL", Quantity(1000));
//! ```

use std::collections::HashMap;
use types::{AgentId, Quantity, Symbol, Tick};

/// A single borrow position held by an agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowPosition {
    /// The symbol borrowed.
    pub symbol: Symbol,
    /// Number of shares currently borrowed.
    pub quantity: Quantity,
    /// Tick when the borrow was initiated.
    pub borrowed_at: Tick,
}

impl BorrowPosition {
    /// Create a new borrow position.
    pub fn new(symbol: Symbol, quantity: Quantity, tick: Tick) -> Self {
        Self {
            symbol,
            quantity,
            borrowed_at: tick,
        }
    }

    /// Calculate borrow cost in ticks (for basis point rate calculation).
    ///
    /// Cost = quantity * borrow_rate_bps * ticks_held / (10000 * ticks_per_year)
    /// Returns raw ticks held; actual cost calculation depends on tick duration.
    pub fn ticks_held(&self, current_tick: Tick) -> u64 {
        current_tick.saturating_sub(self.borrowed_at)
    }
}

/// Ledger tracking all borrows across agents and symbols.
///
/// The ledger maintains:
/// - Per-symbol borrow pool availability
/// - Per-agent borrow positions
/// - Aggregate statistics
#[derive(Debug, Clone)]
pub struct BorrowLedger {
    /// Available shares to borrow per symbol.
    /// Initialized from SymbolConfig::borrow_pool_size().
    available: HashMap<Symbol, Quantity>,

    /// Borrowed positions: agent_id → symbol → BorrowPosition.
    borrows: HashMap<AgentId, HashMap<Symbol, BorrowPosition>>,
}

impl BorrowLedger {
    /// Create a new empty borrow ledger.
    pub fn new() -> Self {
        Self {
            available: HashMap::new(),
            borrows: HashMap::new(),
        }
    }

    /// Initialize the borrow pool for a symbol.
    ///
    /// Should be called once during simulation setup for each symbol.
    pub fn init_symbol(&mut self, symbol: impl Into<Symbol>, pool_size: Quantity) {
        self.available.insert(symbol.into(), pool_size);
    }

    /// Get the available shares to borrow for a symbol.
    pub fn available(&self, symbol: &str) -> Quantity {
        self.available
            .get(symbol)
            .copied()
            .unwrap_or(Quantity::ZERO)
    }

    /// Check if the requested quantity can be borrowed.
    pub fn can_borrow(&self, symbol: &str, quantity: Quantity) -> bool {
        self.available(symbol) >= quantity
    }

    /// Get the current borrowed quantity for an agent-symbol pair.
    pub fn borrowed(&self, agent_id: AgentId, symbol: &str) -> Quantity {
        self.borrows
            .get(&agent_id)
            .and_then(|m| m.get(symbol))
            .map(|p| p.quantity)
            .unwrap_or(Quantity::ZERO)
    }

    /// Get the total borrowed quantity for a symbol across all agents.
    pub fn total_borrowed(&self, symbol: &str) -> Quantity {
        self.borrows
            .values()
            .filter_map(|m| m.get(symbol))
            .map(|p| p.quantity)
            .sum()
    }

    /// Borrow shares for short selling.
    ///
    /// Returns `Ok(())` if successful, `Err(needed)` if insufficient availability.
    pub fn borrow(
        &mut self,
        agent_id: AgentId,
        symbol: impl Into<Symbol>,
        quantity: Quantity,
        tick: Tick,
    ) -> Result<(), Quantity> {
        let symbol = symbol.into();

        // Check availability
        let avail = self
            .available
            .entry(symbol.clone())
            .or_insert(Quantity::ZERO);
        if *avail < quantity {
            return Err(quantity - *avail);
        }

        // Deduct from pool
        *avail = avail.saturating_sub(quantity);

        // Add to agent's borrows
        let agent_borrows = self.borrows.entry(agent_id).or_default();
        let position = agent_borrows
            .entry(symbol.clone())
            .or_insert_with(|| BorrowPosition::new(symbol, Quantity::ZERO, tick));

        position.quantity += quantity;

        Ok(())
    }

    /// Return borrowed shares (when closing a short position).
    ///
    /// Returns the actual quantity returned (capped at borrowed amount).
    pub fn return_shares(
        &mut self,
        agent_id: AgentId,
        symbol: &str,
        quantity: Quantity,
    ) -> Quantity {
        // Get agent's current borrow
        let borrowed = self.borrowed(agent_id, symbol);
        if borrowed.is_zero() {
            return Quantity::ZERO;
        }

        // Cap at what was actually borrowed
        let to_return = quantity.min(borrowed);

        // Return to pool
        if let Some(avail) = self.available.get_mut(symbol) {
            *avail += to_return;
        }

        // Update agent's position
        if let Some(agent_borrows) = self.borrows.get_mut(&agent_id) {
            if let Some(position) = agent_borrows.get_mut(symbol) {
                position.quantity = position.quantity.saturating_sub(to_return);

                // Remove position if fully closed
                if position.quantity.is_zero() {
                    agent_borrows.remove(symbol);
                }
            }

            // Remove agent entry if no borrows remain
            if agent_borrows.is_empty() {
                self.borrows.remove(&agent_id);
            }
        }

        to_return
    }

    /// Get all borrow positions for an agent.
    pub fn agent_positions(&self, agent_id: AgentId) -> Vec<&BorrowPosition> {
        self.borrows
            .get(&agent_id)
            .map(|m| m.values().collect())
            .unwrap_or_default()
    }

    /// Get total number of agents with active borrows.
    pub fn borrower_count(&self) -> usize {
        self.borrows.len()
    }

    /// Clear all borrows (for simulation reset).
    pub fn clear(&mut self) {
        // Return all borrowed shares to pools
        for agent_borrows in self.borrows.values() {
            for (symbol, position) in agent_borrows {
                if let Some(avail) = self.available.get_mut(symbol) {
                    *avail += position.quantity;
                }
            }
        }
        self.borrows.clear();
    }
}

impl Default for BorrowLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ledger_init_and_availability() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        assert_eq!(ledger.available("AAPL"), Quantity(1_000_000));
        assert_eq!(ledger.available("UNKNOWN"), Quantity::ZERO);
    }

    #[test]
    fn test_borrow_success() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        let result = ledger.borrow(AgentId(1), "AAPL", Quantity(1_000), 0);
        assert!(result.is_ok());

        assert_eq!(ledger.available("AAPL"), Quantity(999_000));
        assert_eq!(ledger.borrowed(AgentId(1), "AAPL"), Quantity(1_000));
    }

    #[test]
    fn test_borrow_insufficient() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(500));

        let result = ledger.borrow(AgentId(1), "AAPL", Quantity(1_000), 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Quantity(500)); // Need 500 more
    }

    #[test]
    fn test_multiple_borrows_same_agent() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        ledger
            .borrow(AgentId(1), "AAPL", Quantity(1_000), 0)
            .unwrap();
        ledger
            .borrow(AgentId(1), "AAPL", Quantity(500), 10)
            .unwrap();

        assert_eq!(ledger.borrowed(AgentId(1), "AAPL"), Quantity(1_500));
        assert_eq!(ledger.available("AAPL"), Quantity(998_500));
    }

    #[test]
    fn test_multiple_agents() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        ledger
            .borrow(AgentId(1), "AAPL", Quantity(1_000), 0)
            .unwrap();
        ledger
            .borrow(AgentId(2), "AAPL", Quantity(2_000), 0)
            .unwrap();

        assert_eq!(ledger.borrowed(AgentId(1), "AAPL"), Quantity(1_000));
        assert_eq!(ledger.borrowed(AgentId(2), "AAPL"), Quantity(2_000));
        assert_eq!(ledger.total_borrowed("AAPL"), Quantity(3_000));
        assert_eq!(ledger.borrower_count(), 2);
    }

    #[test]
    fn test_return_shares() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        ledger
            .borrow(AgentId(1), "AAPL", Quantity(1_000), 0)
            .unwrap();
        let returned = ledger.return_shares(AgentId(1), "AAPL", Quantity(600));

        assert_eq!(returned, Quantity(600));
        assert_eq!(ledger.borrowed(AgentId(1), "AAPL"), Quantity(400));
        assert_eq!(ledger.available("AAPL"), Quantity(999_600));
    }

    #[test]
    fn test_return_all_shares() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        ledger
            .borrow(AgentId(1), "AAPL", Quantity(1_000), 0)
            .unwrap();
        let returned = ledger.return_shares(AgentId(1), "AAPL", Quantity(1_000));

        assert_eq!(returned, Quantity(1_000));
        assert_eq!(ledger.borrowed(AgentId(1), "AAPL"), Quantity::ZERO);
        assert_eq!(ledger.borrower_count(), 0);
    }

    #[test]
    fn test_return_more_than_borrowed() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        ledger
            .borrow(AgentId(1), "AAPL", Quantity(1_000), 0)
            .unwrap();
        // Try to return more than borrowed
        let returned = ledger.return_shares(AgentId(1), "AAPL", Quantity(2_000));

        // Should only return what was borrowed
        assert_eq!(returned, Quantity(1_000));
        assert_eq!(ledger.available("AAPL"), Quantity(1_000_000));
    }

    #[test]
    fn test_clear() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000_000));

        ledger
            .borrow(AgentId(1), "AAPL", Quantity(1_000), 0)
            .unwrap();
        ledger
            .borrow(AgentId(2), "AAPL", Quantity(2_000), 0)
            .unwrap();

        ledger.clear();

        assert_eq!(ledger.borrower_count(), 0);
        assert_eq!(ledger.available("AAPL"), Quantity(1_000_000));
    }

    #[test]
    fn test_can_borrow() {
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("AAPL", Quantity(1_000));

        assert!(ledger.can_borrow("AAPL", Quantity(500)));
        assert!(ledger.can_borrow("AAPL", Quantity(1_000)));
        assert!(!ledger.can_borrow("AAPL", Quantity(1_001)));
    }

    #[test]
    fn test_borrow_position_ticks_held() {
        let position = BorrowPosition::new("AAPL".to_string(), Quantity(100), 10);
        assert_eq!(position.ticks_held(100), 90);
        assert_eq!(position.ticks_held(10), 0);
    }
}
