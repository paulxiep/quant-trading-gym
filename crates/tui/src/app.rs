//! Main TUI application - composes widgets and handles rendering loop.
//!
//! # V2.3 Multi-Symbol Support
//!
//! The TUI now supports multiple symbols with tab-based navigation:
//! - `Tab`/`→`: Next symbol
//! - `Shift+Tab`/`←`: Previous symbol  
//! - `1`-`9`: Jump to symbol by number
//! - `O`: Toggle price overlay mode (show all symbols on chart)
//!
//! # Start/Stop Control
//!
//! The simulation starts paused. Use `Space` to start/stop:
//! - `Space`: Toggle simulation running state

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
        MouseButton, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};

use crate::SimCommand;
use crate::widgets::{AgentTable, PriceChart, RiskPanel, SimUpdate, StatsPanel};

/// TUI application state.
pub struct TuiApp {
    /// Channel receiver for simulation updates.
    receiver: Receiver<SimUpdate>,
    /// Channel sender for commands to simulation.
    command_sender: Option<Sender<SimCommand>>,
    /// Latest simulation state.
    state: SimUpdate,
    /// Whether the simulation has finished.
    finished: bool,
    /// Whether the simulation is currently running.
    running: bool,
    /// Target frame rate.
    frame_rate: u64,
    /// Risk panel scroll offset.
    risk_scroll: usize,
    /// Agent P&L scroll offset.
    agent_scroll: usize,
    /// Last known risk panel area (for mouse detection).
    risk_area: Option<Rect>,
    /// Last known agent panel area (for mouse detection).
    agent_area: Option<Rect>,
    /// Currently selected symbol index (V2.3).
    selected_symbol: usize,
    /// Overlay mode: show all symbols on chart (V2.3).
    overlay_mode: bool,
}

impl TuiApp {
    /// Create a new TUI app with the given channel receiver.
    ///
    /// The simulation starts **paused**. Press Space to start.
    pub fn new(receiver: Receiver<SimUpdate>) -> Self {
        Self {
            receiver,
            command_sender: None,
            state: SimUpdate::default(),
            finished: false,
            running: false, // Start paused
            frame_rate: 60, // 60 FPS is smooth enough for visualization
            risk_scroll: 0,
            agent_scroll: 0,
            risk_area: None,
            agent_area: None,
            selected_symbol: 0,
            overlay_mode: false,
        }
    }

    /// Set the command sender for controlling the simulation.
    pub fn with_command_sender(mut self, sender: Sender<SimCommand>) -> Self {
        self.command_sender = Some(sender);
        self
    }

    /// Set the target frame rate (frames per second).
    pub fn frame_rate(mut self, fps: u64) -> Self {
        self.frame_rate = fps;
        self
    }

    /// Run the TUI event loop.
    ///
    /// Blocks until the user presses 'q' or the simulation finishes.
    pub fn run(mut self) -> io::Result<()> {
        // Setup terminal with mouse support
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.run_loop(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    /// Main event loop.
    fn run_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        let tick_rate = Duration::from_millis(1000 / self.frame_rate);
        let mut last_tick = Instant::now();

        loop {
            // V3.8: Drain all pending updates BEFORE drawing
            // This ensures TUI shows latest state and never blocks simulation
            self.poll_updates();

            // Draw current state
            terminal.draw(|f| self.draw(f))?;

            // Handle input with timeout
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if event::poll(timeout)? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if self.handle_key_event(key.code, key.modifiers) {
                            return Ok(());
                        }
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse_event(mouse);
                    }
                    _ => {}
                }
            }

            // Rate limit frames
            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }

            // Exit if simulation finished and user has seen it
            if self.finished {
                // Keep showing until user presses q
            }
        }
    }

    /// Handle keyboard input. Returns true if should quit.
    fn handle_key_event(&mut self, code: KeyCode, modifiers: KeyModifiers) -> bool {
        match code {
            // Quit
            KeyCode::Char('q') | KeyCode::Esc => {
                // Send quit command to simulation
                if let Some(ref sender) = self.command_sender {
                    let _ = sender.send(SimCommand::Quit);
                }
                return true;
            }

            // Start/Stop toggle
            KeyCode::Char(' ') => {
                if !self.finished {
                    self.running = !self.running;
                    if let Some(ref sender) = self.command_sender {
                        let _ = sender.send(SimCommand::Toggle);
                    }
                }
            }

            // Symbol navigation: Tab/Right = next, Shift+Tab/Left = previous
            KeyCode::Tab if modifiers.contains(KeyModifiers::SHIFT) => {
                self.select_previous_symbol();
            }
            KeyCode::Tab | KeyCode::Right => {
                self.select_next_symbol();
            }
            KeyCode::Left => {
                self.select_previous_symbol();
            }

            // Number keys 1-9 jump to symbol
            KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                let idx = (c as usize) - ('1' as usize);
                if idx < self.state.symbols.len() {
                    self.selected_symbol = idx;
                    self.state.selected_symbol = self.selected_symbol;
                }
            }

            // Toggle overlay mode
            KeyCode::Char('o') | KeyCode::Char('O') => {
                self.overlay_mode = !self.overlay_mode;
            }

            _ => {}
        }
        false
    }

    /// Select the next symbol (wraps around).
    fn select_next_symbol(&mut self) {
        if !self.state.symbols.is_empty() {
            self.selected_symbol = (self.selected_symbol + 1) % self.state.symbols.len();
            self.state.selected_symbol = self.selected_symbol;
        }
    }

    /// Select the previous symbol (wraps around).
    fn select_previous_symbol(&mut self) {
        if !self.state.symbols.is_empty() {
            self.selected_symbol = if self.selected_symbol == 0 {
                self.state.symbols.len() - 1
            } else {
                self.selected_symbol - 1
            };
            self.state.selected_symbol = self.selected_symbol;
        }
    }

    /// Handle mouse events for scrolling.
    fn handle_mouse_event(&mut self, mouse: crossterm::event::MouseEvent) {
        let x = mouse.column;
        let y = mouse.row;

        match mouse.kind {
            MouseEventKind::ScrollUp => {
                // Check which panel the mouse is over
                if let Some(area) = self.risk_area
                    && x >= area.x
                    && x < area.x + area.width
                    && y >= area.y
                    && y < area.y + area.height
                {
                    self.risk_scroll = self.risk_scroll.saturating_sub(1);
                    return;
                }
                if let Some(area) = self.agent_area
                    && x >= area.x
                    && x < area.x + area.width
                    && y >= area.y
                    && y < area.y + area.height
                {
                    self.agent_scroll = self.agent_scroll.saturating_sub(1);
                }
            }
            MouseEventKind::ScrollDown => {
                // Check which panel the mouse is over
                if let Some(area) = self.risk_area
                    && x >= area.x
                    && x < area.x + area.width
                    && y >= area.y
                    && y < area.y + area.height
                {
                    let max_scroll = self.state.risk_metrics.len().saturating_sub(1);
                    self.risk_scroll = (self.risk_scroll + 1).min(max_scroll);
                    return;
                }
                if let Some(area) = self.agent_area
                    && x >= area.x
                    && x < area.x + area.width
                    && y >= area.y
                    && y < area.y + area.height
                {
                    let max_scroll = self.state.agents.len().saturating_sub(1);
                    self.agent_scroll = (self.agent_scroll + 1).min(max_scroll);
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                // Handle scrollbar click-to-scroll
                self.handle_scrollbar_click(x, y);
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                // Handle scrollbar drag
                self.handle_scrollbar_click(x, y);
            }
            _ => {}
        }
    }

    /// Handle scrollbar click/drag for direct position jumping.
    fn handle_scrollbar_click(&mut self, x: u16, y: u16) {
        // Check if click is on the right edge of risk panel (scrollbar area)
        if let Some(area) = self.risk_area {
            let scrollbar_x = area.x + area.width - 1;
            if x == scrollbar_x && y > area.y && y < area.y + area.height - 1 {
                let total = self.state.risk_metrics.len();
                if total > 0 {
                    let scrollbar_height = (area.height - 2) as usize;
                    let click_pos = (y - area.y - 1) as usize;
                    let new_scroll = (click_pos * total) / scrollbar_height.max(1);
                    self.risk_scroll = new_scroll.min(total.saturating_sub(1));
                }
                return;
            }
        }

        // Check if click is on the right edge of agent panel (scrollbar area)
        if let Some(area) = self.agent_area {
            let scrollbar_x = area.x + area.width - 1;
            if x == scrollbar_x && y > area.y && y < area.y + area.height - 1 {
                let total = self.state.agents.len();
                if total > 0 {
                    let scrollbar_height = (area.height - 2) as usize;
                    let click_pos = (y - area.y - 1) as usize;
                    let new_scroll = (click_pos * total) / scrollbar_height.max(1);
                    self.agent_scroll = new_scroll.min(total.saturating_sub(1));
                }
            }
        }
    }

    /// Poll for updates from the simulation channel (non-blocking).
    fn poll_updates(&mut self) {
        // Drain all currently available updates, keep the latest
        // try_iter() is non-blocking - returns only items currently in the channel
        for update in self.receiver.try_iter() {
            if update.finished {
                self.finished = true;
            }
            // Keep selected_symbol in bounds
            if self.selected_symbol >= update.symbols.len() && !update.symbols.is_empty() {
                self.selected_symbol = 0;
            }
            self.state = update;
        }
        // Sync TuiApp's selected_symbol to SimUpdate so helper methods work
        self.state.selected_symbol = self.selected_symbol;
    }

    /// Draw the UI.
    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();

        // Main layout: header + content
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header
                Constraint::Length(1), // Symbol tabs (V2.3)
                Constraint::Min(0),    // Content
                Constraint::Length(1), // Footer
            ])
            .split(area);

        // Render header
        self.draw_header(frame, main_chunks[0]);

        // Render symbol tabs (V2.3)
        self.draw_symbol_tabs(frame, main_chunks[1]);

        // Content layout: left panel (stats + book + risk) + right panel (chart + agents)
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
            .split(main_chunks[2]);

        // Left panel: stats + risk panel (book removed - batch auction clears each tick)
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(9), // Stats
                Constraint::Min(10),   // Risk panel (expanded)
            ])
            .split(content_chunks[0]);

        // Right panel: price chart + agent table
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content_chunks[1]);

        // Store areas for mouse detection
        self.risk_area = Some(left_chunks[1]);
        self.agent_area = Some(right_chunks[1]);

        // Draw widgets
        self.draw_stats(frame, left_chunks[0]);
        self.draw_risk_panel(frame, left_chunks[1]);
        self.draw_price_chart(frame, right_chunks[0]);
        self.draw_agent_table(frame, right_chunks[1]);

        // Render footer
        self.draw_footer(frame, main_chunks[3]);
    }

    /// Draw the header bar.
    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let status = if self.finished {
            Span::styled(
                " FINISHED ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        } else if self.running {
            Span::styled(
                " RUNNING ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(
                " PAUSED ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            )
        };

        let overlay_indicator = if self.overlay_mode {
            Span::styled(
                " OVERLAY ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::raw("")
        };

        let title = Line::from(vec![
            Span::styled(
                "Quant Trading Gym",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" │ "),
            status,
            Span::raw(" "),
            overlay_indicator,
        ]);

        let header = Paragraph::new(title).style(Style::default().bg(Color::DarkGray));
        frame.render_widget(header, area);
    }

    /// Draw the symbol tabs (V2.3).
    fn draw_symbol_tabs(&self, frame: &mut Frame, area: Rect) {
        if self.state.symbols.is_empty() {
            // Single symbol mode or no symbols yet
            let current = self.state.current_symbol().cloned().unwrap_or_default();
            let tabs = Paragraph::new(Line::from(vec![Span::styled(
                format!(" [{}] ", current),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]))
            .style(Style::default().bg(Color::DarkGray));
            frame.render_widget(tabs, area);
            return;
        }

        let mut spans: Vec<Span> = vec![Span::raw(" ")];
        for (i, symbol) in self.state.symbols.iter().enumerate() {
            let style = if i == self.selected_symbol {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            spans.push(Span::styled(format!("[{}:{}]", i + 1, symbol), style));
            spans.push(Span::raw(" "));
        }

        let tabs = Paragraph::new(Line::from(spans)).style(Style::default().bg(Color::DarkGray));
        frame.render_widget(tabs, area);
    }

    /// Draw the footer bar.
    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let start_stop_hint = if self.finished {
            Span::raw("")
        } else if self.running {
            Span::styled("Space", Style::default().fg(Color::Yellow))
        } else {
            Span::styled("Space", Style::default().fg(Color::Green))
        };

        let start_stop_label = if self.finished {
            Span::raw("")
        } else if self.running {
            Span::raw(" Pause  │ ")
        } else {
            Span::raw(" Start  │ ")
        };

        let footer = Paragraph::new(Line::from(vec![
            Span::styled(" q", Style::default().fg(Color::Yellow)),
            Span::raw(" Quit  │ "),
            start_stop_hint,
            start_stop_label,
            Span::styled("Tab/←→", Style::default().fg(Color::Cyan)),
            Span::raw(" Symbol  │ "),
            Span::styled("o", Style::default().fg(Color::Cyan)),
            Span::raw(" Overlay  │ "),
            Span::styled("🖱 Scroll", Style::default().fg(Color::Cyan)),
            Span::raw(" Mouse wheel"),
        ]))
        .style(Style::default().bg(Color::DarkGray));
        frame.render_widget(footer, area);
    }

    /// Draw the stats panel.
    fn draw_stats(&self, frame: &mut Frame, area: Rect) {
        let stats = StatsPanel::new()
            .tick(self.state.tick)
            .last_price(self.state.current_last_price())
            .total_trades(self.state.total_trades)
            .total_orders(self.state.total_orders)
            .tier1_count(self.state.tier1_count)
            .tier2_count(self.state.tier2_count)
            .tier3_count(self.state.tier3_count)
            .t3_orders(self.state.t3_orders)
            .agents_called(self.state.agents_called)
            .t2_triggered(self.state.t2_triggered);

        frame.render_widget(stats, area);
    }

    /// Draw the price chart.
    fn draw_price_chart(&self, frame: &mut Frame, area: Rect) {
        if self.overlay_mode && self.state.symbols.len() > 1 {
            // Overlay mode: show all symbols
            let title = "Price (Overlay)".to_string();
            let chart = PriceChart::multi(&self.state.price_history, &self.state.symbols)
                .title(&title)
                .tick(self.state.tick);
            frame.render_widget(chart, area);
        } else {
            // Single symbol mode
            let price_history = self.state.current_price_history();
            let title = match self.state.current_last_price() {
                Some(p) => format!("Price: ${:.2}", p.to_float()),
                None => "Price".to_string(),
            };
            let chart = PriceChart::new(price_history)
                .title(&title)
                .tick(self.state.tick);
            frame.render_widget(chart, area);
        }
    }

    /// Draw the agent P&L table.
    fn draw_agent_table(&mut self, frame: &mut Frame, area: Rect) {
        let total = self.state.agents.len();
        let selected_symbol = self.state.symbols.get(self.selected_symbol).cloned();
        let mut table = AgentTable::new(&self.state.agents).scroll_offset(self.agent_scroll);
        if let Some(sym) = selected_symbol {
            table = table.symbol(sym);
        }
        frame.render_widget(table, area);

        // Render scrollbar if there are items
        if total > 0 {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("▲"))
                .end_symbol(Some("▼"))
                .track_symbol(Some("│"))
                .thumb_symbol("█");

            let mut scrollbar_state = ScrollbarState::new(total).position(self.agent_scroll);

            // Render scrollbar in the inner area (inside the border)
            let scrollbar_area = Rect {
                x: area.x,
                y: area.y,
                width: area.width,
                height: area.height,
            };
            frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
        }
    }

    /// Draw the risk metrics panel.
    fn draw_risk_panel(&self, frame: &mut Frame, area: Rect) {
        // Sort: noise traders by total return (desc), market makers at bottom
        let mut sorted_metrics = self.state.risk_metrics.clone();
        sorted_metrics.sort_by(|a, b| {
            // Market makers always go to bottom
            match (a.is_market_maker, b.is_market_maker) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => {
                    // Within same category, sort by total return (descending)
                    b.total_return
                        .partial_cmp(&a.total_return)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        });

        // Calculate aggregate metrics from noise traders only (exclude MMs)
        let noise_traders: Vec<_> = sorted_metrics
            .iter()
            .filter(|r| !r.is_market_maker)
            .collect();

        let aggregate_sharpe = if !noise_traders.is_empty() {
            let valid_sharpes: Vec<f64> = noise_traders
                .iter()
                .filter_map(|r| r.sharpe)
                .filter(|s| s.is_finite())
                .collect();
            if valid_sharpes.is_empty() {
                None
            } else {
                Some(valid_sharpes.iter().sum::<f64>() / valid_sharpes.len() as f64)
            }
        } else {
            None
        };

        let aggregate_max_drawdown = noise_traders
            .iter()
            .map(|r| r.max_drawdown)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let risk_panel = RiskPanel::new()
            .agents(sorted_metrics.clone())
            .aggregate_sharpe(aggregate_sharpe)
            .aggregate_max_drawdown(aggregate_max_drawdown)
            .scroll_offset(self.risk_scroll);

        frame.render_widget(risk_panel, area);

        // Render scrollbar if there are items
        let total = sorted_metrics.len();
        if total > 0 {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("▲"))
                .end_symbol(Some("▼"))
                .track_symbol(Some("│"))
                .thumb_symbol("█");

            let mut scrollbar_state = ScrollbarState::new(total).position(self.risk_scroll);

            frame.render_stateful_widget(scrollbar, area, &mut scrollbar_state);
        }
    }
}

/// A simpler TUI that doesn't use channels - for direct integration.
pub struct SimpleTui {
    /// Current state to display.
    state: SimUpdate,
    /// Risk panel scroll offset.
    risk_scroll: usize,
    /// Agent P&L scroll offset.
    agent_scroll: usize,
    /// Currently selected symbol index.
    selected_symbol: usize,
    /// Overlay mode.
    overlay_mode: bool,
}

impl SimpleTui {
    /// Create a new simple TUI.
    pub fn new() -> Self {
        Self {
            state: SimUpdate::default(),
            risk_scroll: 0,
            agent_scroll: 0,
            selected_symbol: 0,
            overlay_mode: false,
        }
    }

    /// Update the display state.
    pub fn update(&mut self, state: SimUpdate) {
        self.state = state;
    }

    /// Draw a single frame to the given terminal.
    pub fn draw(&mut self, frame: &mut Frame) {
        // Create a dummy channel just for drawing (never used)
        let (_tx, rx) = crossbeam_channel::unbounded::<SimUpdate>();
        let mut app = TuiApp {
            receiver: rx,
            state: self.state.clone(),
            finished: self.state.finished,
            frame_rate: 60, // 60 FPS is smooth enough for visualization
            risk_scroll: self.risk_scroll,
            agent_scroll: self.agent_scroll,
            risk_area: None,
            agent_area: None,
            selected_symbol: self.selected_symbol,
            overlay_mode: self.overlay_mode,
            command_sender: None,
            running: true, // SimpleTui assumes running for display
        };
        app.draw(frame);
    }

    /// Initialize the terminal for TUI rendering.
    pub fn init() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        Terminal::new(backend)
    }

    /// Restore the terminal to normal state.
    pub fn restore(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        Ok(())
    }
}

impl Default for SimpleTui {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a key was pressed (non-blocking).
pub fn check_quit() -> io::Result<bool> {
    if event::poll(Duration::from_millis(0))?
        && let Event::Key(key) = event::read()?
        && key.kind == KeyEventKind::Press
        && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
    {
        return Ok(true);
    }
    Ok(false)
}
