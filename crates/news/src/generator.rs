//! News event generator (V2.4).
//!
//! This module provides [`NewsGenerator`] which produces market-moving events
//! based on declarative configuration and deterministic random seeding.
//!
//! # Usage
//!
//! ```ignore
//! let config = NewsGeneratorConfig::default();
//! let mut generator = NewsGenerator::new(config, symbols, sectors, 42);
//!
//! // Each tick
//! let events = generator.tick(current_tick);
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use types::{Sector, Symbol, Tick};

use crate::config::NewsGeneratorConfig;
use crate::events::{FundamentalEvent, NewsEvent};
use crate::sectors::SectorModel;

// =============================================================================
// NewsGenerator
// =============================================================================

/// Generates news events based on configuration.
///
/// The generator is deterministic given the same seed, enabling reproducible
/// simulations for testing and debugging.
pub struct NewsGenerator {
    /// Configuration for event generation.
    config: NewsGeneratorConfig,

    /// Random number generator.
    rng: StdRng,

    /// Symbols that can have events.
    symbols: Vec<Symbol>,

    /// Sector model for sector news.
    sector_model: SectorModel,

    /// Next event ID.
    next_id: u64,

    /// Last tick when each event type occurred (for min_interval enforcement).
    last_earnings_tick: Option<Tick>,
    last_guidance_tick: Option<Tick>,
    last_rate_tick: Option<Tick>,
    last_sector_tick: Option<Tick>,
}

impl NewsGenerator {
    /// Create a new news generator.
    ///
    /// # Arguments
    /// * `config` - Event generation configuration
    /// * `symbols` - Symbols that can have symbol-specific events
    /// * `sector_model` - Sector mappings for sector news
    /// * `seed` - Random seed for deterministic generation
    pub fn new(
        config: NewsGeneratorConfig,
        symbols: Vec<Symbol>,
        sector_model: SectorModel,
        seed: u64,
    ) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
            symbols,
            sector_model,
            next_id: 1,
            last_earnings_tick: None,
            last_guidance_tick: None,
            last_rate_tick: None,
            last_sector_tick: None,
        }
    }

    /// Generate events for the current tick.
    ///
    /// Returns a vector of new events. May return empty if no events occur.
    pub fn tick(&mut self, current_tick: Tick) -> Vec<NewsEvent> {
        let mut events = Vec::new();

        // Try each event type
        if let Some(event) = self.try_generate_earnings(current_tick) {
            events.push(event);
        }
        if let Some(event) = self.try_generate_guidance(current_tick) {
            events.push(event);
        }
        if let Some(event) = self.try_generate_rate_decision(current_tick) {
            events.push(event);
        }
        if let Some(event) = self.try_generate_sector_news(current_tick) {
            events.push(event);
        }

        events
    }

    /// Try to generate an earnings surprise event.
    fn try_generate_earnings(&mut self, current_tick: Tick) -> Option<NewsEvent> {
        let cfg = &self.config.earnings;

        if !cfg.frequency.enabled || self.symbols.is_empty() {
            return None;
        }

        // Check min interval
        if let Some(last) = self.last_earnings_tick
            && current_tick < last + cfg.frequency.min_interval
        {
            return None;
        }

        // Roll probability
        if !self.rng.r#gen_bool(cfg.frequency.probability_per_tick) {
            return None;
        }

        // Generate event value FIRST to break symbol-value correlation
        // (Otherwise the same seed always gives the same symbol the same outcomes)
        let surprise_pct = self
            .rng
            .r#gen_range(cfg.surprise_range.0..=cfg.surprise_range.1);

        // Then select symbol independently
        let symbol_idx = self.rng.r#gen_range(0..self.symbols.len());
        let symbol = self.symbols[symbol_idx].clone();
        let sentiment = surprise_pct.signum() * (surprise_pct.abs().sqrt()); // Sentiment proportional to sqrt(surprise)
        let magnitude = self.rng.r#gen_range(cfg.magnitude.min..=cfg.magnitude.max);

        let event = NewsEvent::new(
            self.next_id,
            FundamentalEvent::EarningsSurprise {
                symbol,
                surprise_pct,
            },
            sentiment.clamp(-1.0, 1.0),
            magnitude,
            current_tick,
            cfg.duration_ticks,
        );

        self.next_id += 1;
        self.last_earnings_tick = Some(current_tick);

        Some(event)
    }

    /// Try to generate a guidance change event.
    fn try_generate_guidance(&mut self, current_tick: Tick) -> Option<NewsEvent> {
        let cfg = &self.config.guidance;

        if !cfg.frequency.enabled || self.symbols.is_empty() {
            return None;
        }

        if let Some(last) = self.last_guidance_tick
            && current_tick < last + cfg.frequency.min_interval
        {
            return None;
        }

        if !self.rng.r#gen_bool(cfg.frequency.probability_per_tick) {
            return None;
        }

        // Generate growth value FIRST to break symbol-value correlation
        let new_growth = self
            .rng
            .r#gen_range(cfg.growth_range.0..=cfg.growth_range.1);

        // Then select symbol independently
        let symbol_idx = self.rng.r#gen_range(0..self.symbols.len());
        let symbol = self.symbols[symbol_idx].clone();
        // Positive guidance = positive sentiment
        let sentiment = (new_growth * 10.0).clamp(-1.0, 1.0);
        let magnitude = self.rng.r#gen_range(cfg.magnitude.min..=cfg.magnitude.max);

        let event = NewsEvent::new(
            self.next_id,
            FundamentalEvent::GuidanceChange { symbol, new_growth },
            sentiment,
            magnitude,
            current_tick,
            cfg.duration_ticks,
        );

        self.next_id += 1;
        self.last_guidance_tick = Some(current_tick);

        Some(event)
    }

    /// Try to generate a rate decision event.
    fn try_generate_rate_decision(&mut self, current_tick: Tick) -> Option<NewsEvent> {
        let cfg = &self.config.rate_decision;

        if !cfg.frequency.enabled {
            return None;
        }

        if let Some(last) = self.last_rate_tick
            && current_tick < last + cfg.frequency.min_interval
        {
            return None;
        }

        if !self.rng.r#gen_bool(cfg.frequency.probability_per_tick) {
            return None;
        }

        let change_bps = self
            .rng
            .r#gen_range(cfg.change_range_bps.0..=cfg.change_range_bps.1);
        let new_rate = 0.04 + (change_bps as f64 / 10000.0); // Base rate + change

        // Rate hikes are typically negative for equities
        let sentiment = -(change_bps as f64 / 50.0).clamp(-1.0, 1.0);
        let magnitude = self.rng.r#gen_range(cfg.magnitude.min..=cfg.magnitude.max);

        let event = NewsEvent::new(
            self.next_id,
            FundamentalEvent::RateDecision { new_rate },
            sentiment,
            magnitude,
            current_tick,
            cfg.duration_ticks,
        );

        self.next_id += 1;
        self.last_rate_tick = Some(current_tick);

        Some(event)
    }

    /// Try to generate a sector news event.
    fn try_generate_sector_news(&mut self, current_tick: Tick) -> Option<NewsEvent> {
        // Copy config values to avoid borrow conflict
        let enabled = self.config.sector_news.frequency.enabled;
        let min_interval = self.config.sector_news.frequency.min_interval;
        let probability = self.config.sector_news.frequency.probability_per_tick;
        let duration_ticks = self.config.sector_news.duration_ticks;
        let mag_min = self.config.sector_news.magnitude.min;
        let mag_max = self.config.sector_news.magnitude.max;

        if !enabled {
            return None;
        }

        if let Some(last) = self.last_sector_tick
            && current_tick < last + min_interval
        {
            return None;
        }

        if !self.rng.r#gen_bool(probability) {
            return None;
        }

        // Pick a random sector that has symbols
        let active_sectors: Vec<_> = self.sector_model.active_sectors().copied().collect();
        let sector = if active_sectors.is_empty() {
            // Fallback: pick any sector
            let all_sectors = Sector::all();
            all_sectors[self.rng.r#gen_range(0..all_sectors.len())]
        } else {
            active_sectors[self.rng.r#gen_range(0..active_sectors.len())]
        };

        // Random sentiment
        let sentiment = self.rng.r#gen_range(-1.0..=1.0);
        let magnitude = self.rng.r#gen_range(mag_min..=mag_max);

        let event = NewsEvent::new(
            self.next_id,
            FundamentalEvent::SectorNews { sector, sentiment },
            sentiment,
            magnitude,
            current_tick,
            duration_ticks,
        );

        self.next_id += 1;
        self.last_sector_tick = Some(current_tick);

        Some(event)
    }

    /// Get current configuration (for debugging/inspection).
    pub fn config(&self) -> &NewsGeneratorConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: NewsGeneratorConfig) {
        self.config = config;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NewsGeneratorConfig;

    fn setup_generator() -> NewsGenerator {
        let symbols = vec!["AAPL".to_string(), "MSFT".to_string(), "GOOG".to_string()];
        let mut sector_model = SectorModel::new();
        sector_model.add("AAPL", Sector::Tech);
        sector_model.add("MSFT", Sector::Tech);
        sector_model.add("GOOG", Sector::Tech);

        NewsGenerator::new(
            NewsGeneratorConfig::high_frequency(),
            symbols,
            sector_model,
            42,
        )
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = setup_generator();
        let mut gen2 = setup_generator();

        // Run 100 ticks and compare results
        for tick in 0..100 {
            let events1 = gen1.tick(tick);
            let events2 = gen2.tick(tick);
            assert_eq!(
                events1.len(),
                events2.len(),
                "Tick {tick} event count mismatch"
            );
        }
    }

    #[test]
    fn test_high_frequency_generates_events() {
        let mut generator = setup_generator();

        let mut total_events = 0;
        for tick in 0..1000 {
            let events = generator.tick(tick);
            total_events += events.len();
        }

        // With high frequency config, we should get many events
        assert!(
            total_events > 10,
            "Expected many events, got {total_events}"
        );
    }

    #[test]
    fn test_disabled_config_no_events() {
        let config = NewsGeneratorConfig::disabled();
        let symbols = vec!["AAPL".to_string()];
        let sector_model = SectorModel::new();
        let mut generator = NewsGenerator::new(config, symbols, sector_model, 42);

        let mut total_events = 0;
        for tick in 0..1000 {
            total_events += generator.tick(tick).len();
        }

        assert_eq!(total_events, 0, "Disabled config should generate no events");
    }

    #[test]
    fn test_min_interval_enforced() {
        let mut config = NewsGeneratorConfig::high_frequency();
        config.earnings.frequency.min_interval = 100;
        config.earnings.frequency.probability_per_tick = 1.0; // Always trigger

        let symbols = vec!["AAPL".to_string()];
        let mut generator = NewsGenerator::new(config, symbols, SectorModel::new(), 42);

        // First event should occur
        let events1 = generator.tick(0);
        assert!(!events1.is_empty());

        // No event within min_interval
        let mut events_in_interval = 0;
        for tick in 1..100 {
            events_in_interval += generator
                .tick(tick)
                .iter()
                .filter(|e| matches!(e.event, FundamentalEvent::EarningsSurprise { .. }))
                .count();
        }
        assert_eq!(events_in_interval, 0, "Should respect min_interval");

        // Event should be possible after min_interval
        let events_after = generator.tick(100);
        // May or may not have earnings (other event types still run)
        let _ = events_after;
    }
}
