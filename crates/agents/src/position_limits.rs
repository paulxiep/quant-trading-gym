//! Position limits validation for order submission.
//!
//! This module provides the `PositionValidator` which checks orders against
//! position constraints before they are submitted to the matching engine.
//!
//! # Constraints Validated
//!
//! 1. **Cash sufficiency** - Agent must have enough cash to buy
//! 2. **Shares outstanding** - Aggregate long positions cannot exceed total shares
//! 3. **Short limit per agent** - Individual short positions are capped
//! 4. **Borrow availability** - Short sales require available borrows
//!
//! # Architecture
//!
//! The validator is a pure function that takes:
//! - The order being validated
//! - Current agent position (from AgentState or Agent trait)
//! - Current agent cash (from AgentState or Agent trait)
//! - Symbol configuration (shares_outstanding)
//! - Short-selling configuration (limits, enabled)
//! - Borrow ledger (current availability)
//! - Total held shares across all agents
//!
//! It returns `Ok(())` if the order is valid, or `Err(RiskViolation)` describing
//! why the order was rejected.
//!
//! # Example
//!
//! ```ignore
//! use agents::{PositionValidator, BorrowLedger};
//! use types::{Order, OrderSide, Price, Quantity, AgentId, SymbolConfig, ShortSellingConfig, Cash};
//!
//! let validator = PositionValidator::new(
//!     SymbolConfig::default(),
//!     ShortSellingConfig::disabled(),
//! );
//!
//! let order = Order::limit(
//!     AgentId(1),
//!     "SIM",
//!     OrderSide::Buy,
//!     Price::from_float(100.0),
//!     Quantity(100),
//! );
//!
//! let result = validator.validate_order(
//!     &order,
//!     0,                           // current position
//!     Cash::from_float(10_000.0),  // current cash
//!     &BorrowLedger::new(),
//!     Quantity(500_000),           // total held by all agents///     false,                       // exempt from short limit (true for market makers)//! );
//! assert!(result.is_ok());
//! ```

use crate::BorrowLedger;
use types::{
    Cash, Order, OrderSide, Price, Quantity, RiskViolation, ShortSellingConfig, SymbolConfig,
};

/// Validates orders against position limits and constraints.
///
/// The validator holds configuration that is typically constant for a simulation,
/// while per-order state (position, cash, borrow availability) is passed to
/// each validation call.
#[derive(Debug, Clone)]
pub struct PositionValidator {
    /// Symbol configuration with shares_outstanding.
    symbol_config: SymbolConfig,
    /// Short-selling rules.
    short_config: ShortSellingConfig,
}

impl PositionValidator {
    /// Create a new position validator with the given configurations.
    pub fn new(symbol_config: SymbolConfig, short_config: ShortSellingConfig) -> Self {
        Self {
            symbol_config,
            short_config,
        }
    }

    /// Create a validator with defaults (no short selling).
    pub fn with_defaults() -> Self {
        Self::new(SymbolConfig::default(), ShortSellingConfig::disabled())
    }

    /// Get a reference to the symbol configuration.
    pub fn symbol_config(&self) -> &SymbolConfig {
        &self.symbol_config
    }

    /// Get a reference to the short-selling configuration.
    pub fn short_config(&self) -> &ShortSellingConfig {
        &self.short_config
    }

    /// Validate an order against position limits.
    ///
    /// # Arguments
    /// * `order` - The order to validate
    /// * `current_position` - Agent's current position (positive = long, negative = short)
    /// * `current_cash` - Agent's current cash balance
    /// * `borrow_ledger` - Current borrow state
    /// * `total_held` - Sum of all agents' long positions in this symbol
    /// * `exempt_from_short_limit` - If true, skip the max_short_per_agent check (for market makers)
    ///
    /// # Returns
    /// * `Ok(())` if the order passes all checks
    /// * `Err(RiskViolation)` describing why the order was rejected
    pub fn validate_order(
        &self,
        order: &Order,
        current_position: i64,
        current_cash: Cash,
        borrow_ledger: &BorrowLedger,
        total_held: Quantity,
        exempt_from_short_limit: bool,
    ) -> Result<(), RiskViolation> {
        // Calculate projected position after order fills
        let order_qty = order.quantity.raw() as i64;
        let projected_position = match order.side {
            OrderSide::Buy => current_position + order_qty,
            OrderSide::Sell => current_position - order_qty,
        };

        // Check based on order type
        match order.side {
            OrderSide::Buy => {
                self.validate_buy(order, current_cash, total_held, projected_position)?;
            }
            OrderSide::Sell => {
                self.validate_sell(
                    order,
                    current_position,
                    projected_position,
                    borrow_ledger,
                    exempt_from_short_limit,
                )?;
            }
        }

        Ok(())
    }

    /// Validate a buy order.
    fn validate_buy(
        &self,
        order: &Order,
        current_cash: Cash,
        total_held: Quantity,
        projected_position: i64,
    ) -> Result<(), RiskViolation> {
        // Check cash sufficiency for limit orders
        // Market orders have unknown price, so we skip this check for them
        if let Some(price) = order.limit_price() {
            let cost = price * order.quantity;
            if cost.raw() > current_cash.raw() {
                return Err(RiskViolation::InsufficientCash);
            }
        }

        // Check shares outstanding limit only if position would be positive (long)
        if projected_position > 0 {
            // Calculate new total held if this order fills
            let new_total = total_held.raw() + order.quantity.raw();
            if new_total > self.symbol_config.shares_outstanding.raw() {
                return Err(RiskViolation::InsufficientShares);
            }
        }

        Ok(())
    }

    /// Validate a sell order.
    fn validate_sell(
        &self,
        order: &Order,
        current_position: i64,
        projected_position: i64,
        borrow_ledger: &BorrowLedger,
        exempt_from_short_limit: bool,
    ) -> Result<(), RiskViolation> {
        // If projected position is negative, this is a short sale
        if projected_position < 0 {
            // Check if short selling is enabled
            if !self.short_config.enabled {
                return Err(RiskViolation::ShortSellingDisabled);
            }

            // Check agent's short limit (unless exempt, e.g., market makers)
            let short_size = (-projected_position) as u64;
            if !exempt_from_short_limit
                && self.short_config.max_short_per_agent.raw() > 0
                && short_size > self.short_config.max_short_per_agent.raw()
            {
                return Err(RiskViolation::ShortLimitExceeded);
            }

            // Check borrow availability if locate is required
            if self.short_config.locate_required {
                // Calculate additional borrow needed
                let current_short = if current_position < 0 {
                    (-current_position) as u64
                } else {
                    0
                };
                let needed_short = short_size;
                let additional_borrow = needed_short.saturating_sub(current_short);

                // Check against already borrowed + available
                let already_borrowed = borrow_ledger.borrowed(order.agent_id, &order.symbol).raw();
                let additional_needed = additional_borrow.saturating_sub(already_borrowed);

                if additional_needed > 0 {
                    let available = borrow_ledger.available(&order.symbol).raw();
                    if additional_needed > available {
                        return Err(RiskViolation::NoBorrowAvailable);
                    }
                }
            }
        }

        Ok(())
    }

    /// Estimate the maximum quantity an agent can buy given their cash and current state.
    ///
    /// This is useful for agents that want to go "all in" on a position.
    pub fn max_buyable(&self, price: Price, current_cash: Cash, total_held: Quantity) -> Quantity {
        if price.raw() <= 0 {
            return Quantity::ZERO;
        }

        // Cash-limited quantity
        let cash_limited = (current_cash.raw() / price.raw()) as u64;

        // Shares-outstanding limited quantity
        let shares_limited = self
            .symbol_config
            .shares_outstanding
            .raw()
            .saturating_sub(total_held.raw());

        Quantity(cash_limited.min(shares_limited))
    }

    /// Estimate the maximum quantity an agent can short given constraints.
    pub fn max_shortable(
        &self,
        current_position: i64,
        borrow_ledger: &BorrowLedger,
        symbol: &str,
    ) -> Quantity {
        if !self.short_config.enabled {
            return Quantity::ZERO;
        }

        // Current short position (0 if long)
        let current_short = if current_position < 0 {
            (-current_position) as u64
        } else {
            0
        };

        // Max short limit
        let limit_remaining = self
            .short_config
            .max_short_per_agent
            .raw()
            .saturating_sub(current_short);

        // Borrow availability
        let borrow_available = if self.short_config.locate_required {
            let already_borrowed = borrow_ledger.borrowed(types::AgentId(0), symbol).raw(); // Placeholder
            borrow_ledger.available(symbol).raw() + already_borrowed
        } else {
            u64::MAX // No locate required, unlimited by borrow
        };

        // Also need to account for current long position that can be sold first
        let can_sell_long = current_position.max(0) as u64;

        Quantity(limit_remaining.min(borrow_available) + can_sell_long)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{AgentId, OrderType};

    fn make_buy_order(agent_id: u64, price: f64, quantity: u64) -> Order {
        Order::limit(
            AgentId(agent_id),
            "SIM",
            OrderSide::Buy,
            Price::from_float(price),
            Quantity(quantity),
        )
    }

    fn make_sell_order(agent_id: u64, price: f64, quantity: u64) -> Order {
        Order::limit(
            AgentId(agent_id),
            "SIM",
            OrderSide::Sell,
            Price::from_float(price),
            Quantity(quantity),
        )
    }

    fn make_market_sell_order(agent_id: u64, quantity: u64) -> Order {
        Order {
            id: types::OrderId(0),
            agent_id: AgentId(agent_id),
            symbol: "SIM".to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Market,
            quantity: Quantity(quantity),
            remaining_quantity: Quantity(quantity),
            timestamp: 0,
            latency_ticks: 0,
            status: types::OrderStatus::Pending,
        }
    }

    #[test]
    fn test_buy_order_sufficient_cash() {
        let validator = PositionValidator::with_defaults();
        let ledger = BorrowLedger::new();

        let order = make_buy_order(1, 100.0, 10);
        let result = validator.validate_order(
            &order,
            0,
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(0),
            false,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_buy_order_insufficient_cash() {
        let validator = PositionValidator::with_defaults();
        let ledger = BorrowLedger::new();

        let order = make_buy_order(1, 100.0, 1000); // $100,000 needed
        let result = validator.validate_order(
            &order,
            0,
            Cash::from_float(10_000.0), // Only $10,000
            &ledger,
            Quantity(0),
            false,
        );

        assert_eq!(result, Err(RiskViolation::InsufficientCash));
    }

    #[test]
    fn test_buy_order_exceeds_shares_outstanding() {
        let symbol_config = SymbolConfig::new("SIM", Quantity(1_000), Price::from_float(100.0));
        let validator = PositionValidator::new(symbol_config, ShortSellingConfig::disabled());
        let ledger = BorrowLedger::new();

        // Try to buy 600 when 500 already held (total would be 1100, exceeding 1000)
        let order = make_buy_order(1, 100.0, 600);
        let result = validator.validate_order(
            &order,
            0,
            Cash::from_float(100_000.0),
            &ledger,
            Quantity(500),
            false,
        );

        assert_eq!(result, Err(RiskViolation::InsufficientShares));
    }

    #[test]
    fn test_sell_order_long_position() {
        let validator = PositionValidator::with_defaults();
        let ledger = BorrowLedger::new();

        // Selling from a long position is always OK
        let order = make_sell_order(1, 100.0, 50);
        let result = validator.validate_order(
            &order,
            100, // Have 100 shares
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(100),
            false,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_sell_order_short_disabled() {
        let validator = PositionValidator::with_defaults(); // Short selling disabled
        let ledger = BorrowLedger::new();

        // Try to sell more than we have (would go short)
        let order = make_sell_order(1, 100.0, 150);
        let result = validator.validate_order(
            &order,
            100, // Have 100, selling 150 would go -50
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(100),
            false,
        );

        assert_eq!(result, Err(RiskViolation::ShortSellingDisabled));
    }

    #[test]
    fn test_sell_order_short_enabled_no_borrow() {
        let short_config = ShortSellingConfig::enabled_default();
        let validator = PositionValidator::new(SymbolConfig::default(), short_config);
        let ledger = BorrowLedger::new(); // No shares available to borrow

        // Try to go short without any borrow available
        let order = make_sell_order(1, 100.0, 100);
        let result = validator.validate_order(
            &order,
            0, // No position, selling would go -100
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(0),
            false,
        );

        assert_eq!(result, Err(RiskViolation::NoBorrowAvailable));
    }

    #[test]
    fn test_sell_order_short_with_borrow() {
        let short_config = ShortSellingConfig::enabled_default();
        let validator = PositionValidator::new(SymbolConfig::default(), short_config);
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("SIM", Quantity(100_000)); // 100k shares available

        // Now short selling should work
        let order = make_sell_order(1, 100.0, 100);
        let result = validator.validate_order(
            &order,
            0,
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(0),
            false,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_sell_order_exceeds_short_limit() {
        let short_config = ShortSellingConfig::enabled_default().with_max_short(Quantity(50)); // Max 50 short
        let validator = PositionValidator::new(SymbolConfig::default(), short_config);
        let mut ledger = BorrowLedger::new();
        ledger.init_symbol("SIM", Quantity(100_000));

        // Try to go -100 when max is 50
        let order = make_sell_order(1, 100.0, 100);
        let result = validator.validate_order(
            &order,
            0,
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(0),
            false,
        );

        assert_eq!(result, Err(RiskViolation::ShortLimitExceeded));
    }

    #[test]
    fn test_sell_order_no_locate_required() {
        let short_config = ShortSellingConfig::enabled_default().with_locate_required(false); // No locate needed
        let validator = PositionValidator::new(SymbolConfig::default(), short_config);
        let ledger = BorrowLedger::new(); // No shares available, but locate not required

        let order = make_sell_order(1, 100.0, 100);
        let result = validator.validate_order(
            &order,
            0,
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(0),
            false,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_max_buyable() {
        let symbol_config = SymbolConfig::new("SIM", Quantity(10_000), Price::from_float(100.0));
        let validator = PositionValidator::new(symbol_config, ShortSellingConfig::disabled());

        // Cash-limited: $10,000 / $100 = 100 shares
        // Shares-limited: 10,000 - 5,000 = 5,000 shares
        // Min = 100
        let max = validator.max_buyable(
            Price::from_float(100.0),
            Cash::from_float(10_000.0),
            Quantity(5_000),
        );
        assert_eq!(max, Quantity(100));

        // Cash-limited: $1,000,000 / $100 = 10,000 shares
        // Shares-limited: 10,000 - 5,000 = 5,000 shares
        // Min = 5,000
        let max = validator.max_buyable(
            Price::from_float(100.0),
            Cash::from_float(1_000_000.0),
            Quantity(5_000),
        );
        assert_eq!(max, Quantity(5_000));
    }

    #[test]
    fn test_market_sell_order() {
        // Market orders should still be validated for short constraints
        let validator = PositionValidator::with_defaults();
        let ledger = BorrowLedger::new();

        let order = make_market_sell_order(1, 150);
        let result = validator.validate_order(
            &order,
            100, // Have 100, selling 150 would go -50
            Cash::from_float(10_000.0),
            &ledger,
            Quantity(100),
            false,
        );

        assert_eq!(result, Err(RiskViolation::ShortSellingDisabled));
    }
}
