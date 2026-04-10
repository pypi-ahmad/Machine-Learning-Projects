import {
  Terminal, FolderPlus, Server, RefreshCw,
  Layers, Search,
} from 'lucide-react';

export function HelpPage() {
  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-surface-50">Help & Documentation</h1>
        <p className="text-sm text-surface-500">Architecture, guides, and reference for the dashboard and CV Projects repo</p>
      </div>

      {/* Architecture */}
      <HelpSection title="Architecture Overview" icon={<Layers className="h-4 w-4" />}>
        <p>The dashboard is a metadata-driven frontend for the Computer Vision Projects repository.</p>
        <pre className="mt-3 overflow-x-auto rounded-xl bg-surface-950/80 p-4 text-xs text-surface-400">{`
┌──────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │Dashboard │ │ Explorer │ │  Detail  │ │ Datasets │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│                       ▲                                  │
│              projects.json (manifest)                    │
└──────────────────────────────────────────────────────────┘
         ▲                              ▲
  scan_projects.py                 FastAPI (optional)
  (build-time scan)                (server.py)
         ▲                              ▲
┌──────────────────────────────────────────────────────────┐
│                  CV Projects Repo                        │
│  core/         — CVProject ABC, registry, runner         │
│  */Source Code/modern.py — @register("key") projects     │
│  configs/datasets/*.yaml — dataset source configs        │
│  models/registry.py      — YOLO/pretrained registry     │
│  utils/                  — shared utilities              │
└──────────────────────────────────────────────────────────┘`}
        </pre>
      </HelpSection>

      {/* How metadata works */}
      <HelpSection title="How Project Metadata Is Discovered" icon={<Search className="h-4 w-4" />}>
        <ol className="list-decimal list-inside space-y-2 text-surface-400">
          <li>
            <code className="text-surface-300">scan_projects.py</code> scans every{' '}
            <code className="text-surface-300">*/Source Code/modern.py</code> file via AST parsing.
          </li>
          <li>
            Extracts <code className="text-surface-300">@register("key")</code> decorators plus class-level attributes:
            <code className="text-surface-300"> project_type</code>, <code className="text-surface-300">description</code>,{' '}
            <code className="text-surface-300">legacy_tech</code>, <code className="text-surface-300">modern_tech</code>.
          </li>
          <li>
            Checks for <code className="text-surface-300">train.py</code>,{' '}
            <code className="text-surface-300">infer.py</code>, <code className="text-surface-300">config.py</code>,{' '}
            <code className="text-surface-300">README.md</code> existence.
          </li>
          <li>
            Merges dataset info from <code className="text-surface-300">configs/datasets/*.yaml</code>.
          </li>
          <li>
            Infers input modes by scanning source code for video/webcam/folder keywords.
          </li>
          <li>
            Assigns tags and model family labels from keyword heuristics.
          </li>
          <li>
            Outputs <code className="text-surface-300">dashboard/public/data/projects.json</code>.
          </li>
        </ol>
      </HelpSection>

      {/* Adding new projects */}
      <HelpSection title="How to Add a New Project to the Dashboard" icon={<FolderPlus className="h-4 w-4" />}>
        <ol className="list-decimal list-inside space-y-2 text-surface-400">
          <li>Create the project folder: <code className="text-surface-300">MyProject/Source Code/</code></li>
          <li>
            Add <code className="text-surface-300">modern.py</code> with a{' '}
            <code className="text-surface-300">@register("my_project")</code> class extending{' '}
            <code className="text-surface-300">CVProject</code>.
          </li>
          <li>
            Set class attributes: <code className="text-surface-300">project_type</code>,{' '}
            <code className="text-surface-300">description</code>,{' '}
            <code className="text-surface-300">modern_tech</code>,{' '}
            <code className="text-surface-300">legacy_tech</code>.
          </li>
          <li>
            Optionally add <code className="text-surface-300">train.py</code>,{' '}
            <code className="text-surface-300">infer.py</code>,{' '}
            <code className="text-surface-300">config.py</code>,{' '}
            <code className="text-surface-300">README.md</code>.
          </li>
          <li>
            Optionally add <code className="text-surface-300">configs/datasets/my_project.yaml</code>.
          </li>
          <li>
            Re-run: <code className="text-surface-300">python dashboard/scripts/scan_projects.py --pretty</code>
          </li>
          <li>The project appears automatically in the dashboard.</li>
        </ol>
      </HelpSection>

      {/* Python API */}
      <HelpSection title="Python API Quick Reference" icon={<Terminal className="h-4 w-4" />}>
        <pre className="overflow-x-auto rounded-xl bg-surface-950/80 p-4 text-xs text-surface-400">{`from core import discover_projects, run, benchmark, list_projects

# Discover all projects (auto-imports modern.py files)
discover_projects()

# List all registered project keys
print(list_projects())

# Run inference
result = run("pedestrian_detection", "path/to/image.jpg")

# Run with visualization
annotated = run("pedestrian_detection", frame, visualize=True)

# Benchmark performance
stats = benchmark("pedestrian_detection", frame, n_runs=20)
# → { mean_latency_s, fps, load_time_s, ... }`}
        </pre>
      </HelpSection>

      {/* Backend API */}
      <HelpSection title="Extending the Backend Adapter" icon={<Server className="h-4 w-4" />}>
        <p className="text-surface-400">
          The optional FastAPI backend (<code className="text-surface-300">dashboard/api/server.py</code>) provides
          REST endpoints for live inference and project management. It wraps
          <code className="text-surface-300"> core.runner</code> with HTTP endpoints.
        </p>
        <p className="mt-2 text-surface-400">
          To add backend support for a new project, ensure it has a working{' '}
          <code className="text-surface-300">modern.py</code> with{' '}
          <code className="text-surface-300">load()</code> and{' '}
          <code className="text-surface-300">predict()</code> methods.
          The adapter layer handles the rest.
        </p>
      </HelpSection>

      {/* Refresh */}
      <HelpSection title="Refreshing Project Data" icon={<RefreshCw className="h-4 w-4" />}>
        <pre className="overflow-x-auto rounded-xl bg-surface-950/80 p-4 text-xs text-surface-400">{`# Re-scan all projects and regenerate the manifest
python dashboard/scripts/scan_projects.py --pretty

# The frontend reads from dashboard/public/data/projects.json
# No server restart needed — just refresh the browser`}
        </pre>
      </HelpSection>
    </div>
  );
}

function HelpSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-6">
      <h2 className="mb-4 flex items-center gap-2 text-base font-semibold text-surface-100">{icon} {title}</h2>
      <div className="text-sm leading-relaxed">{children}</div>
    </section>
  );
}
