//! Risk manager subsystem.
//!
//! Manages position tracking, borrow ledger, and agent risk metrics.

use std::collections::HashMap;

use agents::{BorrowLedger, PositionValidator};
use quant::{AgentRiskSnapshot, AgentRiskTracker};
use types::{
    AgentId, Cash, Order, Quantity, RiskViolation, ShortSellingConfig, Symbol, SymbolConfig, Tick,
    Trade,
};

use crate::traits::{PositionTracker, RiskTracker};

/// Manages risk tracking and position validation.
///
/// Owns the borrow ledger, position validator, risk tracker, and
/// total shares held tracking.
pub struct RiskManager {
    /// Borrow ledger for short-selling tracking.
    borrow_ledger: BorrowLedger,

    /// Cache of total shares held per symbol (sum of all long positions).
    total_shares_held: HashMap<Symbol, Quantity>,

    /// Per-agent risk tracking.
    risk_tracker: AgentRiskTracker,

    /// Position validator for order validation.
    position_validator: PositionValidator,
}

impl RiskManager {
    /// Create a new risk manager.
    pub fn new(
        symbols: &[Symbol],
        borrow_pool_sizes: &HashMap<Symbol, Quantity>,
        primary_config: SymbolConfig,
        short_config: ShortSellingConfig,
    ) -> Self {
        let mut borrow_ledger = BorrowLedger::new();
        let mut total_shares_held = HashMap::new();

        for symbol in symbols {
            let pool_size = borrow_pool_sizes
                .get(symbol)
                .copied()
                .unwrap_or(Quantity::ZERO);
            borrow_ledger.init_symbol(symbol, pool_size);
            total_shares_held.insert(symbol.clone(), Quantity::ZERO);
        }

        Self {
            borrow_ledger,
            total_shares_held,
            risk_tracker: AgentRiskTracker::with_defaults(),
            position_validator: PositionValidator::new(primary_config, short_config),
        }
    }

    /// Process a trade, updating borrow ledger and total shares held.
    pub fn process_trade(
        &mut self,
        trade: &Trade,
        seller_position_before: i64,
        buyer_position_before: i64,
        tick: Tick,
    ) {
        self.update_borrow_ledger(trade, seller_position_before, buyer_position_before, tick);
        self.update_total_shares_held(trade, seller_position_before, buyer_position_before);
    }

    /// Update borrow ledger after a trade.
    fn update_borrow_ledger(
        &mut self,
        trade: &Trade,
        seller_position_before: i64,
        buyer_position_before: i64,
        tick: Tick,
    ) {
        // Check if seller is going short (position becoming more negative)
        let seller_position_after = seller_position_before - trade.quantity.raw() as i64;
        if seller_position_after < 0 && seller_position_before >= seller_position_after {
            // Calculate additional borrow needed
            let was_short = seller_position_before.min(0).unsigned_abs();
            let now_short = seller_position_after.unsigned_abs();
            let additional_borrow = now_short.saturating_sub(was_short);

            if additional_borrow > 0 {
                let _ = self.borrow_ledger.borrow(
                    trade.seller_id,
                    &trade.symbol,
                    Quantity(additional_borrow),
                    tick,
                );
            }
        }

        // Check if buyer is covering a short (position becoming less negative)
        let buyer_position_after = buyer_position_before + trade.quantity.raw() as i64;
        if buyer_position_before < 0 && buyer_position_after > buyer_position_before {
            // Calculate shares to return
            let was_short = buyer_position_before.unsigned_abs();
            let now_short = buyer_position_after.min(0).unsigned_abs();
            let to_return = was_short.saturating_sub(now_short);

            if to_return > 0 {
                self.borrow_ledger.return_shares(
                    trade.buyer_id,
                    &trade.symbol,
                    Quantity(to_return),
                );
            }
        }
    }

    /// Update total shares held after a trade.
    fn update_total_shares_held(
        &mut self,
        trade: &Trade,
        seller_position_before: i64,
        buyer_position_before: i64,
    ) {
        let seller_position_after = seller_position_before - trade.quantity.raw() as i64;
        let buyer_position_after = buyer_position_before + trade.quantity.raw() as i64;

        // Seller's long position change (only count positive positions)
        let seller_long_before = seller_position_before.max(0) as u64;
        let seller_long_after = seller_position_after.max(0) as u64;

        // Buyer's long position change
        let buyer_long_before = buyer_position_before.max(0) as u64;
        let buyer_long_after = buyer_position_after.max(0) as u64;

        // Net change
        let delta = (buyer_long_after + seller_long_after) as i64
            - (buyer_long_before + seller_long_before) as i64;

        let symbol_shares = self
            .total_shares_held
            .entry(trade.symbol.clone())
            .or_insert(Quantity::ZERO);
        if delta > 0 {
            *symbol_shares += Quantity(delta as u64);
        } else {
            *symbol_shares = symbol_shares.saturating_sub(Quantity((-delta) as u64));
        }
    }

    /// Update risk tracking with current equities.
    pub fn update_equities(&mut self, equities: &[(AgentId, f64)]) {
        for &(agent_id, equity) in equities {
            self.risk_tracker.record_equity(agent_id, equity);
        }
    }

    /// Get the number of shares an agent has borrowed for a symbol.
    pub fn borrowed_shares(&self, agent_id: AgentId, symbol: &Symbol) -> Quantity {
        self.borrow_ledger.borrowed(agent_id, symbol)
    }

    /// Get reference to position validator.
    pub fn position_validator(&self) -> &PositionValidator {
        &self.position_validator
    }
}

impl PositionTracker for RiskManager {
    fn borrow_ledger(&self) -> &BorrowLedger {
        &self.borrow_ledger
    }

    fn total_shares_held_for(&self, symbol: &Symbol) -> Quantity {
        self.total_shares_held
            .get(symbol)
            .copied()
            .unwrap_or(Quantity::ZERO)
    }

    fn all_total_shares(&self) -> &HashMap<Symbol, Quantity> {
        &self.total_shares_held
    }

    fn validate_order(
        &self,
        order: &Order,
        agent_position: i64,
        agent_cash: Cash,
        is_market_maker: bool,
        enforce_limits: bool,
    ) -> Result<(), RiskViolation> {
        if !enforce_limits {
            return Ok(());
        }

        let total_held = self.total_shares_held_for(&order.symbol);

        self.position_validator.validate_order(
            order,
            agent_position,
            agent_cash,
            &self.borrow_ledger,
            total_held,
            is_market_maker,
        )
    }
}

impl RiskTracker for RiskManager {
    fn record_equity(&mut self, agent_id: AgentId, equity: f64) {
        self.risk_tracker.record_equity(agent_id, equity);
    }

    fn compute_all_metrics(&self) -> HashMap<AgentId, AgentRiskSnapshot> {
        self.risk_tracker.compute_all_metrics()
    }

    fn compute_metrics(&self, agent_id: AgentId) -> AgentRiskSnapshot {
        self.risk_tracker.compute_metrics(agent_id)
    }
}
