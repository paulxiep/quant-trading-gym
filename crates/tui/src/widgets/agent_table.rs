//! Agent P&L table widget - displays agent positions and profits.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Cell, Row, Table, Widget},
};

use super::update::AgentInfo;

/// Agent P&L summary table widget.
///
/// Sorts ML agents at top, market makers at bottom, others by P&L (descending).
pub struct AgentTable {
    /// Agent information to display (sorted).
    agents: Vec<AgentInfo>,
    /// Scroll offset for the agent list.
    scroll_offset: usize,
    /// Symbol to display position for (None = aggregate).
    symbol: Option<String>,
}

impl AgentTable {
    /// Create a new agent table widget.
    ///
    /// Agents are sorted: ML agents at top, market makers at bottom, others by P&L (desc).
    pub fn new(agents: &[AgentInfo]) -> Self {
        let mut sorted = agents.to_vec();
        sorted.sort_by(|a, b| {
            // ML agents always at top
            match (a.is_ml_agent, b.is_ml_agent) {
                (true, false) => return std::cmp::Ordering::Less,
                (false, true) => return std::cmp::Ordering::Greater,
                _ => {}
            }
            // Market makers always at bottom
            match (a.is_market_maker, b.is_market_maker) {
                (true, false) => return std::cmp::Ordering::Greater,
                (false, true) => return std::cmp::Ordering::Less,
                _ => {}
            }
            // Within same category, sort by total P&L (descending)
            b.total_pnl
                .partial_cmp(&a.total_pnl)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            agents: sorted,
            scroll_offset: 0,
            symbol: None,
        }
    }

    /// Set scroll offset.
    pub fn scroll_offset(mut self, offset: usize) -> Self {
        self.scroll_offset = offset;
        self
    }

    /// Set symbol to display position for.
    pub fn symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }
}

impl Widget for AgentTable {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let header_cells = ["Agent", "Position", "Cash", "Total P&L"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().add_modifier(Modifier::BOLD)));
        let header = Row::new(header_cells)
            .style(Style::default().fg(Color::Yellow))
            .height(1);

        // Calculate visible rows (area height - border - header)
        let visible_rows = (area.height.saturating_sub(3)) as usize;
        let scroll_offset = self.scroll_offset.min(self.agents.len().saturating_sub(1));

        let rows = self
            .agents
            .iter()
            .skip(scroll_offset)
            .take(visible_rows)
            .map(|agent| {
                // Use per-symbol position if symbol specified, otherwise aggregate
                let position = match &self.symbol {
                    Some(sym) => agent.position(sym),
                    None => agent.net_position(),
                };

                // Color position based on direction
                let position_style = if position > 0 {
                    Style::default().fg(Color::Green)
                } else if position < 0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Gray)
                };

                // Color P&L based on profit/loss
                let pnl_value = agent.total_pnl.to_float();
                let pnl_style = if pnl_value > 0.0 {
                    Style::default().fg(Color::Green)
                } else if pnl_value < 0.0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Gray)
                };

                Row::new(vec![
                    Cell::from(agent.name.clone()),
                    Cell::from(format!("{:>8}", position)).style(position_style),
                    Cell::from(format!("${:>10.2}", agent.cash.to_float())),
                    Cell::from(format!("${:>10.2}", pnl_value)).style(pnl_style),
                ])
            });

        let table = Table::new(
            rows,
            [
                Constraint::Min(15),    // Agent name
                Constraint::Length(10), // Position
                Constraint::Length(14), // Cash
                Constraint::Length(14), // P&L
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(Span::styled(
                    " Agent P&L ",
                    Style::default().fg(Color::White),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        );

        Widget::render(table, area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use types::Cash;

    fn make_positions(position: i64) -> HashMap<String, i64> {
        let mut positions = HashMap::new();
        positions.insert("TEST".to_string(), position);
        positions
    }

    #[test]
    fn test_agent_table_empty() {
        let agents: Vec<AgentInfo> = vec![];
        let widget = AgentTable::new(&agents);
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_agent_table_with_data() {
        let agents = vec![
            AgentInfo {
                name: "01-MarketMaker".to_string(),
                positions: make_positions(50),
                total_pnl: Cash::from_float(125.50),
                cash: Cash::from_float(10_125.50),
                is_market_maker: true,
                is_ml_agent: false,
                equity: 10_125.50 + 50.0 * 100.0,
            },
            AgentInfo {
                name: "02-NoiseTrader".to_string(),
                positions: make_positions(-20),
                total_pnl: Cash::from_float(-45.00),
                cash: Cash::from_float(9_955.00),
                is_market_maker: false,
                is_ml_agent: false,
                equity: 9_955.00 - 20.0 * 100.0,
            },
        ];
        let widget = AgentTable::new(&agents);
        let area = Rect::new(0, 0, 60, 10);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }
}
