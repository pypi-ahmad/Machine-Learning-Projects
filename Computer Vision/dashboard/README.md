# CV Projects Dashboard

A premium, SaaS-grade, metadata-driven dashboard for the Computer Vision Projects repository. Browse, search, run inference on, and manage all 78+ CV projects from one unified interface — with a glassmorphism design system, animated transitions, and a professional visual polish.

## Quick Start

```bash
# Install dependencies
cd dashboard
npm install

# Start development server
npm run dev
# → http://localhost:5173

# Optional: Start the backend API for live inference
python dashboard/api/server.py
# → http://localhost:8042/api/docs
```

## Features

| Feature | Description |
|---------|-------------|
| **Dashboard** | Gradient hero, donut chart, stats with glow effects, category breakdown |
| **Command Palette** | ⌘K search for instant access to any page or project |
| **Projects Explorer** | Searchable catalog with glass filter pills, stagger animations, shimmer loading |
| **Project Detail** | Gradient banner, glass info cards, capability pills, CLI quick-start |
| **Run Inference** | Upload → run model → annotated output + JSON, glass panels |
| **Training** | Config form with premium inputs, CLI generation, dataset status |
| **Datasets & Models** | Premium table with glass stat cards, status indicators |
| **Settings** | System status, backend connectivity, keyboard shortcuts, theme palette |
| **Help & Docs** | Architecture overview, step-by-step guides, Python API reference |

## Design System

The dashboard uses a custom premium design system built on Tailwind CSS v4:

| Token / Utility | Description |
|----------------|-------------|
| `primary-*` | Indigo-shifted palette (`#6366f1` primary-500) |
| `accent-*` | Green accent for status indicators |
| `surface-*` | Dark theme surface scale (900–950) with `-850` intermediate |
| `.glass` | Glassmorphism — `backdrop-blur-md` + rgba backgrounds + glow border |
| `.glass-subtle` | Lighter glass variant for hover states |
| `.gradient-border` | CSS mask technique for animated gradient borders |
| `.glow-sm/md/accent` | Box-shadow glow effects for cards and interactive elements |
| `.shimmer` | Loading skeleton animation (replaces `animate-pulse`) |
| `.animate-fade-in` | Smooth page/section entrance animation |
| `.stagger-children` | Cascaded entry animations for grids (8 slots × 50ms) |
| `.noise` | Subtle noise texture overlay for hero sections |

### Component Library

- **GlassCard** — Reusable glass-effect card with optional gradient border
- **DonutChart** — SVG donut with animated segments and center total
- **StatCard** — Glow-on-hover stat with subtitle and change tracking
- **Badge** — Glass variant, dot indicator, size options
- **CommandPalette** — ⌘K overlay with keyboard navigation (↑↓ Enter Esc)
- **Button** — Five variants (primary/secondary/outline/ghost/danger) with shadow accents
- **Sidebar** — Glassmorphism shell, section labels, active indicator pill, search trigger

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   React Frontend (Vite)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │Dashboard │ │ Explorer │ │  Detail  │ │ Run/Train  │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
│                       │                                   │
│              ┌────────┴────────┐                          │
│              │  services/api   │   (typed API client)     │
│              └────────┬────────┘                          │
└───────────────────────┼───────────────────────────────────┘
                        │  /api/*
                        ▼
┌───────────────────────────────────────────────────────────┐
│              FastAPI Backend (optional)                     │
│  /api/health  /api/projects  /api/run/{key}               │
│  /api/system  /api/registry  /api/visualize/{key}         │
│  /api/datasets/{key}         /api/train/{key}             │
└───────────────────────┬───────────────────────────────────┘
                        │
┌───────────────────────┼───────────────────────────────────┐
│              CV Projects Core Framework                     │
│  core/         — CVProject ABC, @register, runner          │
│  */modern.py   — 78 registered projects                    │
│  models/       — Three-tier weight resolution              │
│  configs/      — Dataset YAML configs                      │
│  utils/        — Shared utilities                          │
└───────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Metadata-driven**: The frontend renders all projects dynamically from `projects.json`—no hardcoded per-project pages
2. **Graceful degradation**: Works in read-only mode without the backend API; live inference requires the API server
3. **Adapter pattern**: `services/api.ts` provides a typed client that wraps all backend calls with proper error handling
4. **Progressive disclosure**: Simple overview first, detailed info on demand (project detail → run → train)

## Tech Stack

- **React 19** + **TypeScript** — UI framework
- **Vite 8** — Build tool (337KB JS gzipped to 97KB, 56KB CSS gzipped to 9.5KB)
- **Tailwind CSS v4** — Styling with custom dark theme tokens + glassmorphism utilities
- **React Router v7** — Client-side routing
- **Lucide React** — Icon system (100+ icons used)
- **FastAPI** — Optional Python backend (inference, system info, dataset management)

## Project Structure

```
dashboard/
├── api/
│   └── server.py              # FastAPI backend (optional)
├── public/
│   └── data/
│       └── projects.json      # Pre-generated project manifest
├── scripts/
│   ├── scan_projects.py       # Manifest generator
│   └── generate_notebooks.py
├── src/
│   ├── components/
│   │   ├── ui/                # Reusable primitives
│   │   │   ├── Badge.tsx      # Glass/solid variants, dot indicator
│   │   │   ├── Button.tsx     # 5 variants with shadow accents
│   │   │   ├── DonutChart.tsx # SVG donut with animated segments
│   │   │   ├── FileUpload.tsx # Drag-and-drop with preview
│   │   │   ├── GlassCard.tsx  # Glassmorphism card container
│   │   │   ├── LogPanel.tsx   # Terminal-style log viewer
│   │   │   ├── OutputViewer.tsx
│   │   │   ├── ProgressBar.tsx
│   │   │   ├── SearchBar.tsx  # Rounded with premium focus ring
│   │   │   ├── StatCard.tsx   # Glow-on-hover with subtitle
│   │   │   ├── Tabs.tsx       # Pill-style tab bar
│   │   │   ├── BackendBadge.tsx # Dot indicator for API status
│   │   │   └── EmptyState.tsx
│   │   ├── CommandPalette.tsx  # ⌘K search overlay
│   │   ├── Sidebar.tsx        # Glassmorphism sidebar
│   │   ├── ProjectCard.tsx    # Gradient accent stripe
│   │   └── FilterPanel.tsx    # Glass filter pills
│   ├── hooks/
│   │   ├── useProjects.ts     # Manifest, filters, favorites
│   │   └── useApi.ts          # Backend connectivity, API calls
│   ├── services/
│   │   └── api.ts             # Typed API client
│   ├── pages/
│   │   ├── DashboardPage.tsx  # Gradient hero + donut chart
│   │   ├── ProjectsPage.tsx   # Stagger grid with shimmer
│   │   ├── ProjectDetailPage.tsx # Glass info cards
│   │   ├── RunProjectPage.tsx
│   │   ├── TrainProjectPage.tsx
│   │   ├── DatasetsPage.tsx   # Premium table
│   │   ├── SettingsPage.tsx
│   │   └── HelpPage.tsx
│   ├── types.ts               # All TypeScript interfaces
│   ├── index.css              # Premium design tokens & utilities
│   ├── App.tsx                # Router + CommandPalette
│   └── main.tsx               # Entry point
└── package.json
```

## How Project Metadata Works

1. `scan_projects.py` walks every `*/Source Code/modern.py` via AST parsing
2. Extracts `@register("key")` decorators and class attributes (`project_type`, `description`, etc.)
3. Checks for `train.py`, `infer.py`, `config.py`, `README.md` existence
4. Merges dataset info from `configs/datasets/*.yaml`
5. Infers input modes and tags from source code heuristics
6. Outputs `dashboard/public/data/projects.json`

The frontend reads this manifest at runtime—**no build step needed** to add projects.

## How to Add a New Project

1. Create `MyProject/Source Code/modern.py` with `@register("my_project")` class extending `CVProject`
2. Set class attributes: `project_type`, `description`, `modern_tech`, `legacy_tech`
3. Implement `load()` and `predict()` methods
4. Optionally add: `train.py`, `infer.py`, `configs/datasets/my_project.yaml`
5. Run: `python dashboard/scripts/scan_projects.py --pretty`
6. The project appears automatically in the dashboard

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/system` | GPU, Python, PyTorch info |
| GET | `/api/projects` | Full project manifest |
| GET | `/api/projects/{key}` | Single project metadata |
| GET | `/api/registry` | Live project registry |
| POST | `/api/run/{key}` | Run inference (upload image) |
| POST | `/api/visualize/{key}` | Run inference + visualization |
| GET | `/api/datasets/{key}` | Dataset status |
| POST | `/api/datasets/{key}/download` | Trigger dataset download |
| POST | `/api/train/{key}` | Start training (scaffold) |
| GET | `/api/train/{key}/status` | Training status |

## Extending the Adapter Layer

The `services/api.ts` module provides typed functions for all API calls. To add a new endpoint:

1. Add the response type to `types.ts`
2. Add the API function to `services/api.ts`
3. Create or update a hook in `hooks/useApi.ts`
4. Use in components via the hook

The adapter pattern ensures the frontend never makes raw `fetch` calls—all API access goes through the typed service layer.

## Development

```bash
npm run dev       # Start dev server with HMR
npm run build     # Production build
npm run preview   # Preview production build
npm run lint      # Lint with ESLint
```

## Known Integration Gaps

- **Training is CLI-first**: Full async job management for training is scaffolded but not yet fully implemented. The UI generates CLI commands and provides a scaffold for future live training.
- **Webcam/video input**: The upload UI supports images; webcam and video streaming are marked as input modes but require WebRTC integration for live use.
- **Dataset download**: The download endpoint runs synchronously—large datasets may timeout. A background job system is planned.

These gaps are clearly marked in the UI with appropriate messaging rather than faked functionality.
