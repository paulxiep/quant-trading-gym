//! Sector classification and symbol-to-sector mapping (V2.4).
//!
//! This module provides [`SectorModel`] for mapping symbols to their industry sectors.
//! Used for sector-level news events and portfolio grouping.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use types::{Sector, Symbol};

// =============================================================================
// SectorModel
// =============================================================================

/// Maps symbols to their industry sectors.
///
/// Provides lookups for sector-level operations like:
/// - Applying sector news sentiment to all symbols in a sector
/// - Grouping portfolio positions by sector
/// - Computing sector exposures
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SectorModel {
    /// Symbol to sector mapping.
    symbol_to_sector: HashMap<Symbol, Sector>,

    /// Sector to symbols mapping (reverse index).
    sector_to_symbols: HashMap<Sector, Vec<Symbol>>,
}

impl SectorModel {
    /// Create an empty sector model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symbol-sector mapping.
    pub fn add(&mut self, symbol: impl Into<Symbol>, sector: Sector) {
        let symbol = symbol.into();
        self.symbol_to_sector.insert(symbol.clone(), sector);
        self.sector_to_symbols
            .entry(sector)
            .or_default()
            .push(symbol);
    }

    /// Get the sector for a symbol.
    pub fn sector(&self, symbol: &Symbol) -> Option<Sector> {
        self.symbol_to_sector.get(symbol).copied()
    }

    /// Get all symbols in a sector.
    pub fn symbols_in_sector(&self, sector: Sector) -> &[Symbol] {
        self.sector_to_symbols
            .get(&sector)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Check if a symbol belongs to a sector.
    pub fn is_in_sector(&self, symbol: &Symbol, sector: Sector) -> bool {
        self.symbol_to_sector.get(symbol) == Some(&sector)
    }

    /// Get all mapped symbols.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.symbol_to_sector.keys()
    }

    /// Get all sectors that have symbols.
    pub fn active_sectors(&self) -> impl Iterator<Item = &Sector> {
        self.sector_to_symbols.keys()
    }

    /// Get the number of mapped symbols.
    pub fn len(&self) -> usize {
        self.symbol_to_sector.len()
    }

    /// Check if the model is empty.
    pub fn is_empty(&self) -> bool {
        self.symbol_to_sector.is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sector_model_add_and_lookup() {
        let mut model = SectorModel::new();
        model.add("AAPL", Sector::Tech);
        model.add("MSFT", Sector::Tech);

        assert_eq!(model.sector(&"AAPL".to_string()), Some(Sector::Tech));
        assert_eq!(model.sector(&"GOOG".to_string()), None);
    }

    #[test]
    fn test_symbols_in_sector() {
        let mut model = SectorModel::new();
        model.add("AAPL", Sector::Tech);
        model.add("MSFT", Sector::Tech);
        model.add("GOOG", Sector::Tech);

        let tech_symbols = model.symbols_in_sector(Sector::Tech);
        assert_eq!(tech_symbols.len(), 3);
        assert!(tech_symbols.contains(&"AAPL".to_string()));
        assert!(tech_symbols.contains(&"MSFT".to_string()));
    }

    #[test]
    fn test_is_in_sector() {
        let mut model = SectorModel::new();
        model.add("AAPL", Sector::Tech);

        assert!(model.is_in_sector(&"AAPL".to_string(), Sector::Tech));
        assert!(!model.is_in_sector(&"AAPL".to_string(), Sector::Utilities));
        assert!(!model.is_in_sector(&"XOM".to_string(), Sector::Utilities));
    }
}
