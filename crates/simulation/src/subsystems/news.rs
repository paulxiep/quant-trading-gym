//! News engine subsystem.
//!
//! Manages news generation, fundamental data, and fair value drift.

use std::collections::HashMap;

use rand::SeedableRng;
use types::{Price, Sector, Symbol, Tick};

use crate::hooks::NewsEventSnapshot;
use crate::traits::FundamentalsProvider;

/// Manages news events and fundamentals.
///
/// Owns the news generator, active events, fundamentals, and drift RNG.
pub struct NewsEngine {
    /// News event generator.
    news_generator: news::NewsGenerator,

    /// Currently active news events.
    active_events: Vec<news::NewsEvent>,

    /// Symbol fundamentals for fair value calculation.
    fundamentals: news::SymbolFundamentals,

    /// RNG for fair value drift.
    drift_rng: rand::rngs::StdRng,

    /// Fair value drift configuration.
    drift_config: news::config::FairValueDriftConfig,

    /// Verbose logging flag.
    verbose: bool,

    /// Cached symbol-to-sector mapping.
    symbol_sectors: HashMap<Symbol, Sector>,
}

/// Configuration for creating a NewsEngine.
pub struct NewsEngineConfig {
    pub news_config: news::config::NewsGeneratorConfig,
    pub symbols: Vec<Symbol>,
    pub sector_model: news::SectorModel,
    pub initial_fundamentals: news::SymbolFundamentals,
    pub drift_config: news::config::FairValueDriftConfig,
    pub symbol_sectors: HashMap<Symbol, Sector>,
    pub seed: u64,
    pub verbose: bool,
}

impl NewsEngine {
    /// Create a new news engine.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        news_config: news::config::NewsGeneratorConfig,
        symbols: Vec<Symbol>,
        sector_model: news::SectorModel,
        initial_fundamentals: news::SymbolFundamentals,
        drift_config: news::config::FairValueDriftConfig,
        symbol_sectors: HashMap<Symbol, Sector>,
        seed: u64,
        verbose: bool,
    ) -> Self {
        Self::from_config(NewsEngineConfig {
            news_config,
            symbols,
            sector_model,
            initial_fundamentals,
            drift_config,
            symbol_sectors,
            seed,
            verbose,
        })
    }

    /// Create from config struct.
    pub fn from_config(cfg: NewsEngineConfig) -> Self {
        let news_generator =
            news::NewsGenerator::new(cfg.news_config, cfg.symbols, cfg.sector_model, cfg.seed);
        let drift_rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(1));

        Self {
            news_generator,
            active_events: Vec::new(),
            fundamentals: cfg.initial_fundamentals,
            drift_rng,
            drift_config: cfg.drift_config,
            verbose: cfg.verbose,
            symbol_sectors: cfg.symbol_sectors,
        }
    }

    /// Get mutable reference to fundamentals (for initialization).
    pub fn fundamentals_mut(&mut self) -> &mut news::SymbolFundamentals {
        &mut self.fundamentals
    }

    /// Get symbol-to-sector mapping.
    pub fn symbol_sectors(&self) -> &HashMap<Symbol, Sector> {
        &self.symbol_sectors
    }

    /// Apply a permanent fundamental event.
    fn apply_fundamental_event(&mut self, event: &news::NewsEvent, tick: Tick) {
        match &event.event {
            news::FundamentalEvent::EarningsSurprise {
                symbol,
                surprise_pct,
            } => {
                if let Some(fundamentals) = self.fundamentals.get_mut(symbol) {
                    fundamentals.apply_earnings_surprise(*surprise_pct);
                    if self.verbose {
                        eprintln!(
                            "[Tick {}] Earnings surprise for {}: {:.1}% → EPS now ${:.2}",
                            tick,
                            symbol,
                            surprise_pct * 100.0,
                            fundamentals.eps.to_float()
                        );
                    }
                }
            }
            news::FundamentalEvent::GuidanceChange { symbol, new_growth } => {
                if let Some(fundamentals) = self.fundamentals.get_mut(symbol) {
                    fundamentals.apply_guidance_change(*new_growth);
                    if self.verbose {
                        eprintln!(
                            "[Tick {}] Guidance change for {}: growth now {:.1}%",
                            tick,
                            symbol,
                            new_growth * 100.0
                        );
                    }
                }
            }
            news::FundamentalEvent::RateDecision { new_rate } => {
                self.fundamentals.macro_env.apply_rate_decision(*new_rate);
                if self.verbose {
                    eprintln!(
                        "[Tick {}] Rate decision: risk-free rate now {:.2}%",
                        tick,
                        new_rate * 100.0
                    );
                }
            }
            news::FundamentalEvent::SectorNews { .. } => {
                // Sector news is temporary sentiment, not a permanent fundamental change
            }
        }
    }
}

impl FundamentalsProvider for NewsEngine {
    fn process_tick(&mut self, tick: Tick, verbose: bool) {
        // Generate new events
        let new_events = self.news_generator.tick(tick);

        // Apply permanent fundamental changes before adding to active list
        for event in &new_events {
            if event.is_permanent() {
                self.apply_fundamental_event(event, tick);
            }
        }

        // Add new events to active list
        self.active_events.extend(new_events);

        // Prune expired events
        self.active_events.retain(|e| e.is_active(tick));

        // Apply fair value drift
        self.fundamentals
            .apply_drift(&self.drift_config, &mut self.drift_rng);

        // Update verbose flag if changed
        self.verbose = verbose;
    }

    fn active_events(&self) -> &[news::NewsEvent] {
        &self.active_events
    }

    fn fundamentals(&self) -> &news::SymbolFundamentals {
        &self.fundamentals
    }

    fn fair_value(&self, symbol: &Symbol) -> Option<Price> {
        self.fundamentals.fair_value(symbol)
    }

    fn get_news_snapshots(&self, tick: Tick) -> Vec<NewsEventSnapshot> {
        self.active_events
            .iter()
            .filter(|e| e.is_active(tick))
            .map(|e| NewsEventSnapshot {
                id: e.id,
                event: e.event.clone(),
                sentiment: e.sentiment,
                magnitude: e.magnitude,
                start_tick: e.start_tick,
                duration_ticks: e.duration_ticks,
            })
            .collect()
    }
}
