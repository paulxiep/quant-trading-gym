//! Fundamentals provider trait for news and fundamental data.

use types::{Price, Symbol, Tick};

use crate::hooks::NewsEventSnapshot;

/// Provides news events and fundamental data.
///
/// This trait abstracts news/fundamentals, enabling:
/// - Deterministic news in tests (seeded generator)
/// - Scripted news scenarios
/// - Decoupling price dynamics from simulation core
pub trait FundamentalsProvider {
    /// Process news and fundamentals for the current tick.
    ///
    /// This includes:
    /// - Generating new events
    /// - Applying permanent fundamental changes
    /// - Pruning expired events
    /// - Applying fair value drift
    fn process_tick(&mut self, tick: Tick, verbose: bool);

    /// Get currently active news events.
    fn active_events(&self) -> &[news::NewsEvent];

    /// Get symbol fundamentals.
    fn fundamentals(&self) -> &news::SymbolFundamentals;

    /// Get fair value for a symbol.
    fn fair_value(&self, symbol: &Symbol) -> Option<Price>;

    /// Get active news events as snapshots for hooks.
    fn get_news_snapshots(&self, tick: Tick) -> Vec<NewsEventSnapshot>;
}
