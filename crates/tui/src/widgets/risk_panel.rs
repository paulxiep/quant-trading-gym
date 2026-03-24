//! Risk panel widget - displays per-agent risk metrics.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use serde::{Deserialize, Serialize};

/// Risk snapshot for display in TUI.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskInfo {
    /// Agent name.
    pub name: String,
    /// Sharpe ratio.
    pub sharpe: Option<f64>,
    /// Maximum drawdown percentage.
    pub max_drawdown: f64,
    /// Total return percentage.
    pub total_return: f64,
    /// Value at Risk (95%).
    pub var_95: Option<f64>,
    /// Current equity.
    pub equity: f64,
    /// Whether this is a market maker (for sorting to bottom).
    pub is_market_maker: bool,
}

/// Risk panel widget showing aggregate and per-agent risk metrics.
pub struct RiskPanel {
    /// Risk info per agent (top agents by equity).
    pub agents: Vec<RiskInfo>,
    /// Aggregate metrics across all agents.
    pub aggregate_sharpe: Option<f64>,
    /// Aggregate max drawdown.
    pub aggregate_max_drawdown: f64,
    /// Scroll offset for the agent list.
    pub scroll_offset: usize,
}

impl RiskPanel {
    /// Create a new risk panel.
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            aggregate_sharpe: None,
            aggregate_max_drawdown: 0.0,
            scroll_offset: 0,
        }
    }

    /// Set agent risk info.
    pub fn agents(mut self, agents: Vec<RiskInfo>) -> Self {
        self.agents = agents;
        self
    }

    /// Set aggregate sharpe.
    pub fn aggregate_sharpe(mut self, sharpe: Option<f64>) -> Self {
        self.aggregate_sharpe = sharpe;
        self
    }

    /// Set aggregate max drawdown.
    pub fn aggregate_max_drawdown(mut self, dd: f64) -> Self {
        self.aggregate_max_drawdown = dd;
        self
    }

    /// Set scroll offset.
    pub fn scroll_offset(mut self, offset: usize) -> Self {
        self.scroll_offset = offset;
        self
    }
}

impl Default for RiskPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a percentage value with sign.
fn format_pct(val: f64) -> String {
    if val >= 0.0 {
        format!("+{:.2}%", val * 100.0)
    } else {
        format!("{:.2}%", val * 100.0)
    }
}

/// Format Sharpe ratio.
fn format_sharpe(val: Option<f64>) -> String {
    match val {
        Some(s) if s.is_finite() => format!("{:.2}", s),
        _ => "—".to_string(),
    }
}

/// Get color for return value.
fn return_color(val: f64) -> Color {
    if val >= 0.01 {
        Color::Green
    } else if val <= -0.01 {
        Color::Red
    } else {
        Color::Yellow
    }
}

/// Get color for drawdown value.
fn drawdown_color(val: f64) -> Color {
    if val < 0.05 {
        Color::Green
    } else if val < 0.10 {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Get color for Sharpe ratio.
fn sharpe_color(val: Option<f64>) -> Color {
    match val {
        Some(s) if s >= 1.0 => Color::Green,
        Some(s) if s >= 0.0 => Color::Yellow,
        Some(_) => Color::Red,
        None => Color::Gray,
    }
}

impl Widget for RiskPanel {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Calculate available lines for agents (area height - border - header lines)
        // Border: 2 lines (top + bottom), Header: 5 lines (title, blank, agg metrics, blank, column headers)
        let header_lines = 5;
        let available_lines = area.height.saturating_sub(2 + header_lines as u16) as usize;

        // Header line with aggregate metrics
        let mut lines = vec![
            Line::from(vec![
                Span::styled("═══ ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "RISK METRICS",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ═══", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Agg Sharpe: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format_sharpe(self.aggregate_sharpe),
                    Style::default().fg(sharpe_color(self.aggregate_sharpe)),
                ),
                Span::styled("  Max DD: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.2}%", self.aggregate_max_drawdown * 100.0),
                    Style::default().fg(drawdown_color(self.aggregate_max_drawdown)),
                ),
            ]),
            Line::from(""),
        ];

        // Per-agent risk (scrollable)
        if !self.agents.is_empty() {
            // Header row
            lines.push(Line::from(vec![
                Span::styled(
                    "Agent        ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Return   ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "MaxDD    ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Sharpe",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));

            // Calculate visible range with scroll offset
            let total_agents = self.agents.len();
            let scroll_offset = self.scroll_offset.min(total_agents.saturating_sub(1));
            let visible_count = available_lines;

            // Show agents from scroll_offset
            for agent in self.agents.iter().skip(scroll_offset).take(visible_count) {
                let name = if agent.name.len() > 10 {
                    format!("{}…", &agent.name[..9])
                } else {
                    format!("{:<10}", agent.name)
                };

                lines.push(Line::from(vec![
                    Span::styled(format!("{} ", name), Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{:<8} ", format_pct(agent.total_return)),
                        Style::default().fg(return_color(agent.total_return)),
                    ),
                    Span::styled(
                        format!("{:<8} ", format!("{:.2}%", agent.max_drawdown * 100.0)),
                        Style::default().fg(drawdown_color(agent.max_drawdown)),
                    ),
                    Span::styled(
                        format_sharpe(agent.sharpe),
                        Style::default().fg(sharpe_color(agent.sharpe)),
                    ),
                ]));
            }
        } else {
            lines.push(Line::from(vec![Span::styled(
                "Collecting data...",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )]));
        }

        let paragraph = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    " Risk ",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                )),
        );

        paragraph.render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_pct() {
        assert_eq!(format_pct(0.05), "+5.00%");
        assert_eq!(format_pct(-0.03), "-3.00%");
        assert_eq!(format_pct(0.0), "+0.00%");
    }

    #[test]
    fn test_format_sharpe() {
        assert_eq!(format_sharpe(Some(1.5)), "1.50");
        assert_eq!(format_sharpe(Some(-0.5)), "-0.50");
        assert_eq!(format_sharpe(None), "—");
        assert_eq!(format_sharpe(Some(f64::NAN)), "—");
    }

    #[test]
    fn test_risk_panel_builder() {
        let panel = RiskPanel::new()
            .agents(vec![RiskInfo {
                name: "01-NoiseTrader".to_string(),
                sharpe: Some(1.2),
                max_drawdown: 0.05,
                total_return: 0.10,
                var_95: Some(0.02),
                equity: 10500.0,
                is_market_maker: false,
            }])
            .aggregate_sharpe(Some(0.8))
            .aggregate_max_drawdown(0.03);

        assert_eq!(panel.agents.len(), 1);
        assert_eq!(panel.aggregate_sharpe, Some(0.8));
        assert_eq!(panel.aggregate_max_drawdown, 0.03);
    }
}
