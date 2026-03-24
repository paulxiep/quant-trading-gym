//! Stats panel widget - displays simulation statistics.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use types::Price;

/// Simulation statistics panel widget.
pub struct StatsPanel {
    /// Current tick.
    pub tick: u64,
    /// Last trade price.
    pub last_price: Option<Price>,
    /// Total trades executed.
    pub total_trades: u64,
    /// Total orders submitted.
    pub total_orders: u64,
    /// Number of Tier 1 agents.
    pub tier1_count: usize,
    /// Number of Tier 2 agents.
    pub tier2_count: usize,
    /// Number of Tier 3 background pool agents.
    pub tier3_count: usize,
    /// T3 orders generated this tick.
    pub t3_orders: usize,
    /// Agents called this tick.
    pub agents_called: usize,
    /// T2 agents triggered this tick.
    pub t2_triggered: usize,
}

impl StatsPanel {
    /// Create a new stats panel.
    pub fn new() -> Self {
        Self {
            tick: 0,
            last_price: None,
            total_trades: 0,
            total_orders: 0,
            tier1_count: 0,
            tier2_count: 0,
            tier3_count: 0,
            t3_orders: 0,
            agents_called: 0,
            t2_triggered: 0,
        }
    }

    /// Set the current tick.
    pub fn tick(mut self, tick: u64) -> Self {
        self.tick = tick;
        self
    }

    /// Set the last price.
    pub fn last_price(mut self, price: Option<Price>) -> Self {
        self.last_price = price;
        self
    }

    /// Set total trades.
    pub fn total_trades(mut self, trades: u64) -> Self {
        self.total_trades = trades;
        self
    }

    /// Set total orders.
    pub fn total_orders(mut self, orders: u64) -> Self {
        self.total_orders = orders;
        self
    }

    /// Set Tier 1 agent count.
    pub fn tier1_count(mut self, count: usize) -> Self {
        self.tier1_count = count;
        self
    }

    /// Set Tier 2 agent count.
    pub fn tier2_count(mut self, count: usize) -> Self {
        self.tier2_count = count;
        self
    }

    /// Set Tier 3 agent count.
    pub fn tier3_count(mut self, count: usize) -> Self {
        self.tier3_count = count;
        self
    }

    /// Set T3 orders this tick.
    pub fn t3_orders(mut self, count: usize) -> Self {
        self.t3_orders = count;
        self
    }

    /// Set agents called this tick.
    pub fn agents_called(mut self, count: usize) -> Self {
        self.agents_called = count;
        self
    }

    /// Set T2 agents triggered this tick.
    pub fn t2_triggered(mut self, count: usize) -> Self {
        self.t2_triggered = count;
        self
    }
}

impl Default for StatsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for StatsPanel {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let price_str = match self.last_price {
            Some(p) => format!("${:.2}", p.to_float()),
            None => "—".to_string(),
        };

        // Compact layout: combine related stats on same line
        let lines = vec![
            Line::from(vec![
                Span::styled("Tick: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.tick),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("Price: ", Style::default().fg(Color::Gray)),
                Span::styled(price_str, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Trades: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.total_trades),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::styled("Agents: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(
                        "{}T1 + {}T2 + {}T3",
                        self.tier1_count, self.tier2_count, self.tier3_count
                    ),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
            Line::from(vec![
                Span::styled("Called: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.agents_called),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw("  "),
                Span::styled("T2 Wake: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.t2_triggered),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(vec![
                Span::styled("T3 Orders: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.t3_orders),
                    Style::default().fg(Color::Blue),
                ),
            ]),
        ];

        let para = Paragraph::new(lines).block(
            Block::default()
                .title("Simulation")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        );

        para.render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_panel_default() {
        let panel = StatsPanel::default();
        let area = Rect::new(0, 0, 30, 10);
        let mut buf = Buffer::empty(area);
        panel.render(area, &mut buf);
    }

    #[test]
    fn test_stats_panel_with_data() {
        use types::Price;

        let panel = StatsPanel::new()
            .tick(500)
            .last_price(Some(Price::from_float(100.25)))
            .total_trades(42)
            .total_orders(150)
            .tier1_count(12)
            .tier2_count(4000);

        let area = Rect::new(0, 0, 30, 10);
        let mut buf = Buffer::empty(area);
        panel.render(area, &mut buf);
    }
}
