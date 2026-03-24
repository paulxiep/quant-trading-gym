//! ML Agent for cached-prediction trading (V6.2).
//!
//! Non-generic agent that reads ML predictions from the centralized
//! [`MlPredictionCache`] and generates orders based on configurable thresholds.
//!
//! # Architecture
//!
//! Since V5.6, prediction computation is centralized in [`ModelRegistry`]:
//! the runner extracts features once, the registry predicts for all models,
//! and agents simply read cached probabilities by model name.
//!
//! ```text
//! Runner -> extract features -> cache -> ModelRegistry.predict_from_cache()
//!   -> cache.insert_prediction(model_name, symbol, probs)
//!   -> MlAgent reads from cache -> threshold logic -> orders
//! ```
//!
//! # Decision Logic
//!
//! For each tick and each watched symbol:
//! 1. Read `[p_sell, p_hold, p_buy]` from prediction cache
//! 2. Add small random jitter to decorrelate agents sharing the same model
//! 3. Compare to thresholds -> generate order for the stronger signal
//!
//! # Usage
//!
//! ```ignore
//! let agent = MlAgent::new(
//!     AgentId(100),
//!     "RandomForest_small".to_string(),
//!     MlAgentConfig::default(),
//! );
//! ```

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Symbol, Trade};

use super::ClassProbabilities;

/// Configuration for an ML prediction-reading agent.
#[derive(Debug, Clone)]
pub struct MlAgentConfig {
    /// Symbols to trade.
    pub symbols: Vec<Symbol>,
    /// Probability threshold to trigger a buy (e.g., 0.55 = 55% confidence).
    pub buy_threshold: f64,
    /// Probability threshold to trigger a sell.
    pub sell_threshold: f64,
    /// Order size in shares.
    pub order_size: u64,
    /// Maximum long position per symbol.
    pub max_long_position: i64,
    /// Maximum short position per symbol (as positive number).
    pub max_short_position: i64,
    /// Initial cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market data unavailable.
    pub initial_price: Price,
}

impl Default for MlAgentConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["ACME".to_string()],
            buy_threshold: 0.55,
            sell_threshold: 0.55,
            order_size: 50,
            max_long_position: 1000,
            max_short_position: 200,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
        }
    }
}

/// ML agent that reads cached predictions by model name.
///
/// This agent does NOT own a model instance -- predictions are computed
/// centrally by [`ModelRegistry`] and stored in [`MlPredictionCache`].
/// The agent holds only the model name for cache lookup.
pub struct MlAgent {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: MlAgentConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Name of the model whose predictions this agent reads from cache.
    model_name: String,
}

impl MlAgent {
    /// Create a new ML agent that reads predictions for `model_name` from cache.
    pub fn new(id: AgentId, model_name: String, config: MlAgentConfig) -> Self {
        let state = AgentState::with_symbols(config.initial_cash, config.symbols.clone());
        Self {
            id,
            config,
            state,
            model_name,
        }
    }

    /// Get the model name this agent reads predictions from.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get reference price for a symbol.
    fn get_reference_price(&self, symbol: &Symbol, ctx: &StrategyContext<'_>) -> Price {
        ctx.mid_price(symbol)
            .or(ctx.last_price(symbol))
            .unwrap_or(self.config.initial_price)
    }

    /// Check if we can buy more of a symbol.
    fn can_buy(&self, symbol: &Symbol) -> bool {
        self.state.position_for(symbol) < self.config.max_long_position
    }

    /// Check if we can sell/short more of a symbol.
    fn can_sell(&self, symbol: &Symbol) -> bool {
        self.state.position_for(symbol) > -self.config.max_short_position
    }

    /// Generate a buy order for a symbol.
    fn generate_buy_order(&self, symbol: &Symbol, mid_price: Price) -> Order {
        // Bid slightly below mid to qualify in batch auction
        let order_price = Price::from_float(floor_price(mid_price.to_float() * 0.999));
        Order::limit(
            self.id,
            symbol,
            OrderSide::Buy,
            order_price,
            Quantity(self.config.order_size),
        )
    }

    /// Generate a sell order for a symbol.
    fn generate_sell_order(&self, symbol: &Symbol, mid_price: Price) -> Order {
        // Ask slightly above mid to qualify in batch auction
        let order_price = Price::from_float(floor_price(mid_price.to_float() * 1.001));
        Order::limit(
            self.id,
            symbol,
            OrderSide::Sell,
            order_price,
            Quantity(self.config.order_size),
        )
    }
}

impl Agent for MlAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        let mut orders: Vec<Order> = Vec::new();

        // Evaluate each symbol independently - can place up to 1 order per symbol
        for symbol in &self.config.symbols {
            // Read centralized cached prediction (computed in Phase 3 of tick loop)
            let probs: ClassProbabilities = match ctx.get_ml_prediction(&self.model_name, symbol) {
                Some(p) => p,
                None => continue,
            };

            // Small random jitter to decorrelate agents reading the same cached prediction
            let p_sell = probs[0] + rand::random::<f64>() * 0.005;
            let p_hold: f64 = probs[1];
            let p_buy = probs[2] + rand::random::<f64>() * 0.005;

            let mid_price = self.get_reference_price(symbol, ctx);

            // Independent decision per symbol: stronger signal wins if above threshold
            let buy_signal =
                p_buy > self.config.buy_threshold && p_buy >= p_hold && self.can_buy(symbol);
            let sell_signal =
                p_sell > self.config.sell_threshold && p_sell >= p_hold && self.can_sell(symbol);

            match (sell_signal, buy_signal) {
                (true, true) if p_buy >= p_sell => {
                    orders.push(self.generate_buy_order(symbol, mid_price));
                    self.state.record_order();
                }
                (true, true) => {
                    orders.push(self.generate_sell_order(symbol, mid_price));
                    self.state.record_order();
                }
                (true, false) => {
                    orders.push(self.generate_sell_order(symbol, mid_price));
                    self.state.record_order();
                }
                (false, true) => {
                    orders.push(self.generate_buy_order(symbol, mid_price));
                    self.state.record_order();
                }
                (false, false) => {}
            }
        }

        if orders.is_empty() {
            AgentAction::none()
        } else {
            AgentAction::multiple(orders)
        }
    }

    fn on_fill(&mut self, trade: &Trade) {
        // Use separate if blocks (not else if) to handle self-trades correctly.
        if trade.buyer_id == self.id {
            self.state
                .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
        }
        if trade.seller_id == self.id {
            self.state
                .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
        }
    }

    fn name(&self) -> &str {
        &self.model_name
    }

    fn is_ml_agent(&self) -> bool {
        true
    }

    fn state(&self) -> &AgentState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_agent_creation() {
        let config = MlAgentConfig::default();
        let agent = MlAgent::new(AgentId(1), "RandomForest_small".to_string(), config);

        assert_eq!(agent.id(), AgentId(1));
        assert_eq!(agent.model_name(), "RandomForest_small");
        assert_eq!(agent.name(), "RandomForest_small");
    }

    #[test]
    fn test_ml_agent_default_config() {
        let config = MlAgentConfig::default();
        assert_eq!(config.buy_threshold, 0.55);
        assert_eq!(config.sell_threshold, 0.55);
        assert_eq!(config.order_size, 50);
        assert_eq!(config.max_long_position, 1000);
        assert_eq!(config.max_short_position, 200);
    }

    // Price change and log return tests remain in types crate where
    // the utility functions are defined.
}
