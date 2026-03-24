//! Portfolio scope for Tier 2 reactive agents.
//!
//! Defines what symbols an agent trades. V3.2 supports single-symbol only;
//! multi-symbol portfolios are deferred to V3.3.

use serde::{Deserialize, Serialize};
use types::Symbol;

/// Portfolio scope for a reactive agent.
///
/// Determines which symbols the agent monitors and trades.
/// Currently only `SingleSymbol` is supported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReactivePortfolio {
    /// Agent trades a single symbol.
    ///
    /// This is the only supported variant in V3.2.
    /// Multi-symbol support is planned for V3.3.
    SingleSymbol(Symbol),
    // Future variants (V3.3):
    // MultiSymbol(SmallVec<[Symbol; 4]>),
    // Sector { sector: String, max_symbols: usize },
}

impl ReactivePortfolio {
    /// Get the primary symbol for this portfolio.
    ///
    /// For single-symbol portfolios, returns that symbol.
    /// For future multi-symbol portfolios, returns the first symbol.
    pub fn primary_symbol(&self) -> &Symbol {
        match self {
            Self::SingleSymbol(s) => s,
        }
    }

    /// Get all symbols in this portfolio.
    pub fn symbols(&self) -> Vec<&Symbol> {
        match self {
            Self::SingleSymbol(s) => vec![s],
        }
    }

    /// Returns true if this portfolio includes the given symbol.
    pub fn contains(&self, symbol: &Symbol) -> bool {
        match self {
            Self::SingleSymbol(s) => s == symbol,
        }
    }
}

impl From<Symbol> for ReactivePortfolio {
    fn from(symbol: Symbol) -> Self {
        Self::SingleSymbol(symbol)
    }
}

impl From<&str> for ReactivePortfolio {
    fn from(s: &str) -> Self {
        Self::SingleSymbol(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_symbol() {
        let portfolio = ReactivePortfolio::SingleSymbol("ACME".to_string());

        assert_eq!(portfolio.primary_symbol(), "ACME");
        assert!(portfolio.contains(&"ACME".to_string()));
        assert!(!portfolio.contains(&"OTHER".to_string()));
        assert_eq!(portfolio.symbols().len(), 1);
    }

    #[test]
    fn test_from_conversions() {
        let p1: ReactivePortfolio = "ACME".into();
        let p2: ReactivePortfolio = "ACME".to_string().into();

        assert_eq!(p1, p2);
        assert_eq!(p1.primary_symbol(), "ACME");
    }
}
