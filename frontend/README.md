# Quant Trading Gym - Frontend

React/TypeScript frontend for the Quant Trading Gym simulation.

## Development

### Prerequisites
- Node.js 20+ 
- npm or pnpm

### Setup
```bash
cd frontend
npm install
```

### Run Development Server
```bash
npm run dev
```
Opens at http://localhost:5173 with hot module replacement.

### Build for Production
```bash
npm run build
```
Output in `dist/` directory.

## Architecture

Following **Declarative, Modular, SoC** principles:

```
src/
├── api/           # API client functions (SoC: communication layer)
├── components/    # Reusable UI components (Modular: self-contained)
│   ├── ui/        # Generic UI primitives (Button, Input, Accordion)
│   └── config/    # Config-specific components (form sections)
├── config/        # Default configs and presets (Declarative: data)
├── pages/         # Route pages (SoC: page-level orchestration)
└── types/         # TypeScript types (Declarative: shape definitions)
```

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing | Hero + Quick Start / Configure buttons |
| `/config` | Config | Full SimConfig editor with presets |
| `/sim` | Simulation | Dashboard (placeholder for V4.4) |

## Tech Stack

- **Vite** - Build tool with fast HMR
- **React 19** - UI library
- **React Router 7** - Client-side routing
- **TypeScript 5.7** - Type safety
- **Tailwind CSS 3.4** - Utility-first styling
