//! Candle aggregation logic
//!
//! **SoC:** This module ONLY handles candle OHLCV calculation, no storage

use std::collections::HashMap;
use types::{Price, Quantity, Symbol};

/// Candle data (OHLCV)
#[derive(Debug, Clone)]
pub struct Candle {
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Quantity,
}

impl Candle {
    pub fn new(price: Price, quantity: Quantity) -> Self {
        Self {
            open: price,
            high: price,
            low: price,
            close: price,
            volume: quantity,
        }
    }

    pub fn update(&mut self, price: Price, quantity: Quantity) {
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.close = price;
        self.volume += quantity;
    }
}

/// Candle aggregator (stateful, in-memory buffer)
#[derive(Debug)]
pub struct CandleAggregator {
    /// Timeframe in ticks
    timeframe: u64,
    /// Current candles: symbol -> (tick_start, candle)
    current: HashMap<Symbol, (u64, Candle)>,
    /// Completed candles ready for flush: (symbol, tick_start, candle)
    completed: Vec<(Symbol, u64, Candle)>,
}

impl CandleAggregator {
    pub fn new(timeframe: u64) -> Self {
        Self {
            timeframe,
            current: HashMap::new(),
            completed: Vec::new(),
        }
    }

    /// Process a trade, updating current candle
    pub fn process_trade(&mut self, tick: u64, symbol: Symbol, price: Price, quantity: Quantity) {
        let tick_start = (tick / self.timeframe) * self.timeframe;

        self.current
            .entry(symbol.clone())
            .and_modify(|(start, candle)| {
                if *start == tick_start {
                    // Same candle, update OHLCV
                    candle.update(price, quantity);
                } else {
                    // New candle period, complete old candle
                    self.completed
                        .push((symbol.clone(), *start, candle.clone()));
                    *start = tick_start;
                    *candle = Candle::new(price, quantity);
                }
            })
            .or_insert_with(|| (tick_start, Candle::new(price, quantity)));
    }

    /// Get completed candles and clear buffer
    pub fn flush(&mut self) -> Vec<(Symbol, u64, Candle)> {
        std::mem::take(&mut self.completed)
    }

    pub fn timeframe(&self) -> u64 {
        self.timeframe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_creation() {
        let price = Price::from(100_0000); // $100.00
        let qty = Quantity(10);
        let candle = Candle::new(price, qty);
        assert_eq!(candle.open, price);
        assert_eq!(candle.high, price);
        assert_eq!(candle.low, price);
        assert_eq!(candle.close, price);
        assert_eq!(candle.volume, qty);
    }

    #[test]
    fn test_candle_update() {
        let mut candle = Candle::new(Price::from(100_0000), Quantity(10));
        candle.update(Price::from(105_0000), Quantity(5));
        candle.update(Price::from(98_0000), Quantity(3));

        assert_eq!(candle.open, Price::from(100_0000));
        assert_eq!(candle.high, Price::from(105_0000));
        assert_eq!(candle.low, Price::from(98_0000));
        assert_eq!(candle.close, Price::from(98_0000));
        assert_eq!(candle.volume, Quantity(18));
    }

    #[test]
    fn test_aggregator_single_period() {
        let mut agg = CandleAggregator::new(60); // 1-minute candles
        let symbol = Symbol::from("AAPL");

        agg.process_trade(0, symbol.clone(), Price::from(100_0000), Quantity(10));
        agg.process_trade(30, symbol.clone(), Price::from(105_0000), Quantity(5));
        agg.process_trade(59, symbol.clone(), Price::from(98_0000), Quantity(3));

        // All trades in same period, no completed candles yet
        assert_eq!(agg.flush().len(), 0);
    }

    #[test]
    fn test_aggregator_multiple_periods() {
        let mut agg = CandleAggregator::new(60);
        let symbol = Symbol::from("AAPL");

        agg.process_trade(0, symbol.clone(), Price::from(100_0000), Quantity(10));
        agg.process_trade(59, symbol.clone(), Price::from(105_0000), Quantity(5));
        agg.process_trade(60, symbol.clone(), Price::from(98_0000), Quantity(3)); // New period

        let completed = agg.flush();
        assert_eq!(completed.len(), 1);

        let (sym, tick_start, candle) = &completed[0];
        assert_eq!(sym, &symbol);
        assert_eq!(*tick_start, 0);
        assert_eq!(candle.open, Price::from(100_0000));
        assert_eq!(candle.close, Price::from(105_0000));
        assert_eq!(candle.volume, Quantity(15));
    }
}
