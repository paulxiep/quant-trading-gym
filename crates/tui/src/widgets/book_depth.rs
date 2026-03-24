//! Order book depth widget - displays bid/ask levels as horizontal bars.
//!
//! NOTE: Currently unused in batch auction mode (book cleared each tick).
//! Kept for potential future use with "orders this tick" display.

#![allow(dead_code)]

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use types::BookLevel;

/// Order book depth visualization widget.
///
/// Displays bid and ask levels side-by-side with quantity bars.
pub struct BookDepth<'a> {
    /// Bid levels (highest price first).
    bids: &'a [BookLevel],
    /// Ask levels (lowest price first).
    asks: &'a [BookLevel],
    /// Maximum levels to display.
    max_levels: usize,
}

impl<'a> BookDepth<'a> {
    /// Create a new book depth widget.
    pub fn new(bids: &'a [BookLevel], asks: &'a [BookLevel]) -> Self {
        Self {
            bids,
            asks,
            max_levels: 10,
        }
    }

    /// Set the maximum number of levels to display.
    pub fn max_levels(mut self, levels: usize) -> Self {
        self.max_levels = levels;
        self
    }

    /// Render bid side (left column).
    fn render_bids(&self, area: Rect, buf: &mut Buffer, max_qty: u64) {
        let block = Block::default()
            .title("Bids")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green));
        let inner = block.inner(area);
        block.render(area, buf);

        if self.bids.is_empty() {
            let text = Paragraph::new("No bids");
            text.render(inner, buf);
            return;
        }

        for (i, level) in self.bids.iter().take(self.max_levels).enumerate() {
            if i as u16 >= inner.height {
                break;
            }

            let y = inner.y + i as u16;
            let bar_width = if max_qty > 0 {
                ((level.quantity.0 as f64 / max_qty as f64) * (inner.width as f64 / 2.0)) as u16
            } else {
                0
            };

            // Price label (right-aligned)
            let price_str = format!("{:>8.2}", level.price.to_float());
            let qty_str = format!("{:<6}", level.quantity.0);

            // Draw bar from right to left
            let bar_start = inner.x + inner.width.saturating_sub(bar_width);
            for x in bar_start..(inner.x + inner.width) {
                if x < buf.area.width {
                    buf[(x, y)].set_bg(Color::Green);
                }
            }

            // Draw price and quantity
            let line = Line::from(vec![
                Span::styled(price_str, Style::default().fg(Color::White)),
                Span::raw(" "),
                Span::styled(qty_str, Style::default().fg(Color::DarkGray)),
            ]);
            let text = Paragraph::new(line);
            let text_area = Rect::new(inner.x, y, inner.width, 1);
            text.render(text_area, buf);
        }
    }

    /// Render ask side (right column).
    fn render_asks(&self, area: Rect, buf: &mut Buffer, max_qty: u64) {
        let block = Block::default()
            .title("Asks")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Red));
        let inner = block.inner(area);
        block.render(area, buf);

        if self.asks.is_empty() {
            let text = Paragraph::new("No asks");
            text.render(inner, buf);
            return;
        }

        for (i, level) in self.asks.iter().take(self.max_levels).enumerate() {
            if i as u16 >= inner.height {
                break;
            }

            let y = inner.y + i as u16;
            let bar_width = if max_qty > 0 {
                ((level.quantity.0 as f64 / max_qty as f64) * (inner.width as f64 / 2.0)) as u16
            } else {
                0
            };

            // Price label
            let price_str = format!("{:<8.2}", level.price.to_float());
            let qty_str = format!("{:>6}", level.quantity.0);

            // Draw bar from left to right
            for x in inner.x..(inner.x + bar_width) {
                if x < buf.area.width {
                    buf[(x, y)].set_bg(Color::Red);
                }
            }

            // Draw price and quantity
            let line = Line::from(vec![
                Span::styled(price_str, Style::default().fg(Color::White)),
                Span::raw(" "),
                Span::styled(qty_str, Style::default().fg(Color::DarkGray)),
            ]);
            let text = Paragraph::new(line);
            let text_area = Rect::new(inner.x, y, inner.width, 1);
            text.render(text_area, buf);
        }
    }
}

impl Widget for BookDepth<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Split into two columns: bids (left) and asks (right)
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Calculate max quantity for bar scaling
        let max_bid_qty = self.bids.iter().map(|l| l.quantity.0).max().unwrap_or(1);
        let max_ask_qty = self.asks.iter().map(|l| l.quantity.0).max().unwrap_or(1);
        let max_qty = max_bid_qty.max(max_ask_qty);

        self.render_bids(chunks[0], buf, max_qty);
        self.render_asks(chunks[1], buf, max_qty);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{Price, Quantity};

    #[test]
    fn test_book_depth_empty() {
        let bids: Vec<BookLevel> = vec![];
        let asks: Vec<BookLevel> = vec![];
        let widget = BookDepth::new(&bids, &asks);
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_book_depth_with_data() {
        let bids = vec![
            BookLevel {
                price: Price::from_float(99.50),
                quantity: Quantity(100),
                order_count: 1,
            },
            BookLevel {
                price: Price::from_float(99.00),
                quantity: Quantity(200),
                order_count: 2,
            },
        ];
        let asks = vec![
            BookLevel {
                price: Price::from_float(100.50),
                quantity: Quantity(150),
                order_count: 1,
            },
            BookLevel {
                price: Price::from_float(101.00),
                quantity: Quantity(50),
                order_count: 1,
            },
        ];
        let widget = BookDepth::new(&bids, &asks).max_levels(5);
        let area = Rect::new(0, 0, 60, 15);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }
}
