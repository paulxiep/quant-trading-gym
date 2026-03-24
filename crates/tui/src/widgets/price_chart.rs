//! Price chart widget - displays price history as a line graph.
//!
//! # V2.3 Multi-Symbol Support
//!
//! The chart supports two modes:
//! - Single symbol: Shows one price line (cyan)
//! - Multi-symbol overlay: Shows multiple price lines with different colors
//!
//! # V3.7 Smoothing
//!
//! Prices can be smoothed with a simple moving average for better visualization.

use std::collections::HashMap;

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    symbols::Marker,
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Widget},
};

use types::Symbol;

/// Default smoothing window size (number of ticks to average).
const DEFAULT_SMOOTHING_WINDOW: usize = 1;

/// Colors for multi-symbol overlay mode.
const OVERLAY_COLORS: &[Color] = &[
    Color::Cyan,
    Color::Green,
    Color::Yellow,
    Color::Magenta,
    Color::Red,
    Color::Blue,
    Color::LightCyan,
    Color::LightGreen,
];

/// Apply centered moving average smoothing to price data.
/// Each point is the average of `window` points centered around it.
fn smooth_prices(prices: &[f64], window: usize) -> Vec<f64> {
    if window <= 1 || prices.len() < window {
        return prices.to_vec();
    }

    let half = window / 2;
    let mut smoothed = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        // Center the window around index i
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(prices.len());
        let slice = &prices[start..end];
        let avg = slice.iter().sum::<f64>() / slice.len() as f64;
        smoothed.push(avg);
    }

    smoothed
}

/// Type alias for multi-symbol price data references.
type MultiPriceData<'a> = (&'a HashMap<Symbol, Vec<f64>>, &'a [Symbol]);

/// Price chart widget displaying price history as a sparkline.
pub struct PriceChart<'a> {
    /// Single symbol price data.
    prices: &'a [f64],
    /// Multi-symbol price data (for overlay mode).
    multi_prices: Option<MultiPriceData<'a>>,
    /// Chart title.
    title: &'a str,
    /// Smoothing window size (1 = no smoothing).
    smoothing: usize,
    /// Current tick (for X axis labeling).
    current_tick: u64,
}

impl<'a> PriceChart<'a> {
    /// Create a new price chart widget for a single symbol.
    pub fn new(prices: &'a [f64]) -> Self {
        Self {
            prices,
            multi_prices: None,
            title: "Price",
            smoothing: DEFAULT_SMOOTHING_WINDOW,
            current_tick: 0,
        }
    }

    /// Create a new price chart widget for multiple symbols (overlay mode).
    pub fn multi(prices: &'a HashMap<Symbol, Vec<f64>>, symbols: &'a [Symbol]) -> Self {
        Self {
            prices: &[],
            multi_prices: Some((prices, symbols)),
            title: "Price (Overlay)",
            smoothing: DEFAULT_SMOOTHING_WINDOW,
            current_tick: 0,
        }
    }

    /// Set the chart title.
    pub fn title(mut self, title: &'a str) -> Self {
        self.title = title;
        self
    }

    /// Set the smoothing window size (1 = no smoothing).
    #[allow(dead_code)]
    pub fn smoothing(mut self, window: usize) -> Self {
        self.smoothing = window.max(1);
        self
    }

    /// Set the current tick for X axis labels.
    pub fn tick(mut self, tick: u64) -> Self {
        self.current_tick = tick;
        self
    }
}

impl Widget for PriceChart<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if let Some((prices_map, symbols)) = self.multi_prices {
            self.render_multi(prices_map, symbols, area, buf);
        } else {
            self.render_single(area, buf);
        }
    }
}

impl PriceChart<'_> {
    /// Render single symbol chart.
    fn render_single(self, area: Rect, buf: &mut Buffer) {
        if self.prices.is_empty() {
            // Render empty chart with "No data" message
            let block = Block::default()
                .title(self.title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray));
            block.render(area, buf);
            return;
        }

        // Apply smoothing
        let smoothed = smooth_prices(self.prices, self.smoothing);

        // Calculate bounds for Y axis
        let min_price = smoothed.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = smoothed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Ensure minimum Y range to prevent visual noise when prices are stable
        // Use 2% of mid price or $2, whichever is larger
        let mid_price = (min_price + max_price) / 2.0;
        let min_range = (mid_price * 0.02).max(2.0);
        let actual_range = max_price - min_price;

        let (y_min, y_max) = if actual_range < min_range {
            // Expand range symmetrically around midpoint
            let half_range = min_range / 2.0;
            (mid_price - half_range, mid_price + half_range)
        } else {
            // Add 10% padding to actual range
            let y_padding = actual_range * 0.1;
            (min_price - y_padding, max_price + y_padding)
        };

        // Prepare data points: (x, y) where x is the index
        let data: Vec<(f64, f64)> = smoothed
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as f64, p))
            .collect();

        let x_max = (smoothed.len().saturating_sub(1)) as f64;

        let dataset = Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&data);

        let y_labels: Vec<Line> = vec![
            Line::from(format!("{:.2}", y_min)),
            Line::from(format!("{:.2}", (y_min + y_max) / 2.0)),
            Line::from(format!("{:.2}", y_max)),
        ];

        // X axis tick labels showing actual tick numbers
        let history_len = smoothed.len() as u64;
        let start_tick = self
            .current_tick
            .saturating_sub(history_len.saturating_sub(1));
        let mid_tick = (start_tick + self.current_tick) / 2;
        let x_labels: Vec<Line> = vec![
            Line::from(format!("{}", start_tick)),
            Line::from(format!("{}", mid_tick)),
            Line::from(format!("{}", self.current_tick)),
        ];

        let chart = Chart::new(vec![dataset])
            .block(
                Block::default()
                    .title(self.title)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White)),
            )
            .x_axis(
                Axis::default()
                    .title("Tick")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, x_max.max(1.0)])
                    .labels(x_labels),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(y_labels),
            );

        chart.render(area, buf);
    }

    /// Render multi-symbol overlay chart.
    fn render_multi(
        self,
        prices_map: &HashMap<Symbol, Vec<f64>>,
        symbols: &[Symbol],
        area: Rect,
        buf: &mut Buffer,
    ) {
        // Collect and smooth all price data
        let smoothed_prices: Vec<Vec<f64>> = symbols
            .iter()
            .filter_map(|s| prices_map.get(s))
            .filter(|v| !v.is_empty())
            .map(|v| smooth_prices(v, self.smoothing))
            .collect();

        if smoothed_prices.is_empty() {
            let block = Block::default()
                .title(self.title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray));
            block.render(area, buf);
            return;
        }

        // Calculate global Y bounds across all symbols
        let min_price = smoothed_prices
            .iter()
            .flat_map(|v| v.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_price = smoothed_prices
            .iter()
            .flat_map(|v| v.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Ensure minimum Y range to prevent visual noise when prices are stable
        let mid_price = (min_price + max_price) / 2.0;
        let min_range = (mid_price * 0.05).max(5.0);
        let actual_range = max_price - min_price;

        let (y_min, y_max) = if actual_range < min_range {
            let half_range = min_range / 2.0;
            (mid_price - half_range, mid_price + half_range)
        } else {
            let y_padding = actual_range * 0.1;
            (min_price - y_padding, max_price + y_padding)
        };

        // Find max X
        let x_max = smoothed_prices.iter().map(|v| v.len()).max().unwrap_or(1) as f64 - 1.0;

        // Build datasets with owned data
        let data_vecs: Vec<Vec<(f64, f64)>> = smoothed_prices
            .iter()
            .map(|prices| {
                prices
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i as f64, p))
                    .collect()
            })
            .collect();

        let datasets: Vec<Dataset> = data_vecs
            .iter()
            .enumerate()
            .map(|(i, data)| {
                let color = OVERLAY_COLORS[i % OVERLAY_COLORS.len()];
                Dataset::default()
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(color))
                    .data(data)
            })
            .collect();

        let y_labels: Vec<Line> = vec![
            Line::from(format!("{:.2}", y_min)),
            Line::from(format!("{:.2}", (y_min + y_max) / 2.0)),
            Line::from(format!("{:.2}", y_max)),
        ];

        // X axis tick labels showing actual tick numbers
        let history_len = (x_max as u64) + 1;
        let start_tick = self
            .current_tick
            .saturating_sub(history_len.saturating_sub(1));
        let mid_tick = (start_tick + self.current_tick) / 2;
        let x_labels: Vec<Line> = vec![
            Line::from(format!("{}", start_tick)),
            Line::from(format!("{}", mid_tick)),
            Line::from(format!("{}", self.current_tick)),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title(self.title)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White)),
            )
            .x_axis(
                Axis::default()
                    .title("Tick")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, x_max.max(1.0)])
                    .labels(x_labels),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(y_labels),
            );

        chart.render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_chart_empty() {
        let prices: Vec<f64> = vec![];
        let chart = PriceChart::new(&prices);
        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        chart.render(area, &mut buf);
        // Should not panic
    }

    #[test]
    fn test_price_chart_with_data() {
        let prices = vec![100.0, 101.0, 99.5, 102.0, 100.5];
        let chart = PriceChart::new(&prices).title("Test Price");
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        chart.render(area, &mut buf);
        // Should render without panic
    }

    #[test]
    fn test_price_chart_multi_symbol() {
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), vec![150.0, 151.0, 149.0]);
        prices.insert("GOOG".to_string(), vec![2800.0, 2810.0, 2790.0]);
        let symbols = vec!["AAPL".to_string(), "GOOG".to_string()];
        let chart = PriceChart::multi(&prices, &symbols);
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        chart.render(area, &mut buf);
        // Should render without panic
    }
}
