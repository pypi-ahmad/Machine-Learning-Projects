import { useParams, Link } from 'react-router-dom';
import {
  Star, Play, GraduationCap, Database, FileText, Code2,
  Cpu, Monitor, FolderOpen, Terminal, ChevronRight,
} from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import { GlassCard } from '../components/ui/GlassCard';
import { useProject, useFavorites } from '../hooks/useProjects';
import { CATEGORY_META, PROJECT_TYPE_LABELS } from '../types';
import { INPUT_MODE_ICONS, INPUT_MODE_FALLBACK_ICON } from '../constants';

export function ProjectDetailPage() {
  const { key } = useParams<{ key: string }>();
  const { project, loading, error } = useProject(key);
  const { toggle, isFavorite } = useFavorites();

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 rounded-xl bg-surface-800/20 shimmer" />
        <div className="h-72 rounded-3xl bg-surface-800/20 shimmer" />
        <div className="grid grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => <div key={i} className="h-48 rounded-2xl bg-surface-800/20 shimmer" style={{ animationDelay: `${i * 80}ms` }} />)}
        </div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center animate-fade-in">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-surface-800/40">
          <FolderOpen className="h-7 w-7 text-surface-500" />
        </div>
        <h2 className="mt-4 text-lg font-semibold text-surface-300">Project Not Found</h2>
        <p className="mt-1 text-sm text-surface-500">No project with key &ldquo;{key}&rdquo;</p>
        <Link to="/projects" className="mt-4 text-sm text-primary-400 hover:text-primary-300 transition-colors">
          ← Back to projects
        </Link>
      </div>
    );
  }

  const catMeta = CATEGORY_META[project.category] || CATEGORY_META['Other'];
  const fav = isFavorite(project.key);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-[13px] text-surface-500">
        <Link to="/projects" className="flex items-center gap-1 transition-colors hover:text-surface-300">
          Projects
        </Link>
        <ChevronRight className="h-3 w-3 text-surface-600" />
        <span className="text-surface-300">{project.name}</span>
      </div>

      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-3xl border border-surface-800/40">
        <div className="absolute inset-0 bg-gradient-to-br from-surface-900/90 via-surface-900/70 to-surface-950/90" />
        <div className="absolute -right-20 -top-20 h-60 w-60 rounded-full opacity-[0.08] blur-[60px]" style={{ backgroundColor: catMeta.color }} />
        <div className="absolute -left-10 -bottom-10 h-40 w-40 rounded-full opacity-[0.05] blur-[40px]" style={{ backgroundColor: catMeta.color }} />

        <div className="relative p-7 lg:p-8">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-5">
              <div
                className="flex h-16 w-16 items-center justify-center rounded-2xl text-xl font-bold shadow-lg"
                style={{ backgroundColor: `${catMeta.color}15`, color: catMeta.color, boxShadow: `0 8px 30px -8px ${catMeta.color}20` }}
              >
                {project.category.charAt(0)}
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-surface-50">{project.name}</h1>
                <div className="mt-1 flex items-center gap-2">
                  <code className="rounded-lg bg-surface-800/50 px-2 py-0.5 text-xs text-surface-400 font-mono">{project.key}</code>
                  {project.aliases.length > 0 && (
                    <span className="text-[11px] text-surface-600">
                      aliases: {project.aliases.join(', ')}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <button
              onClick={() => toggle(project.key)}
              className="rounded-xl p-2.5 text-surface-500 transition-all hover:bg-surface-800/40 hover:text-amber-400"
            >
              <Star className={`h-5 w-5 ${fav ? 'fill-amber-400 text-amber-400' : ''}`} />
            </button>
          </div>

          <p className="mt-5 max-w-2xl text-sm leading-relaxed text-surface-400">{project.description}</p>

          {/* Badges */}
          <div className="mt-5 flex flex-wrap gap-2">
            <Badge color={catMeta.color} variant="glass" size="md" dot>{project.category}</Badge>
            <Badge variant="glass" size="md" color={catMeta.color}>{PROJECT_TYPE_LABELS[project.projectType] || project.projectType}</Badge>
            {project.tags.map((t) => <Badge key={t} variant="glass" size="md" color="#10b981">{t}</Badge>)}
            {project.modelFamily.map((m) => <Badge key={m} variant="glass" size="md">{m}</Badge>)}
          </div>

          {/* Action buttons */}
          <div className="mt-6 flex flex-wrap gap-3 border-t border-surface-800/40 pt-6">
            {project.hasInference && (
              <Link
                to={`/projects/${project.key}/run`}
                className="group inline-flex items-center gap-2 rounded-xl bg-primary-600 px-5 py-2.5 text-sm font-medium text-white shadow-lg shadow-primary-600/20 transition-all hover:bg-primary-500 hover:shadow-primary-600/30"
              >
                <Play className="h-4 w-4" /> Run Inference
              </Link>
            )}
            {project.hasTraining && (
              <Link
                to={`/projects/${project.key}/train`}
                className="inline-flex items-center gap-2 rounded-xl border border-surface-700/50 bg-surface-800/30 px-5 py-2.5 text-sm font-medium text-surface-300 transition-all hover:border-surface-600 hover:bg-surface-800/50 hover:text-surface-100"
              >
                <GraduationCap className="h-4 w-4" /> Train Model
              </Link>
            )}
          </div>
        </div>
      </div>

      {/* Info cards grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 stagger-children">
        <GlassCard hover>
          <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
            <Cpu className="h-4 w-4 text-surface-400" /> Technology
          </h3>
          <dl className="space-y-3 text-sm">
            <DL label="Modern Stack" value={project.modernTech} />
            <DL label="Legacy Stack" value={project.legacyTech} />
            <DL label="Type" value={PROJECT_TYPE_LABELS[project.projectType] || project.projectType} />
          </dl>
        </GlassCard>

        <GlassCard hover>
          <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
            <Monitor className="h-4 w-4 text-surface-400" /> Supported Inputs
          </h3>
          <div className="flex flex-wrap gap-2">
            {project.inputModes.map((mode) => {
              const Icon = INPUT_MODE_ICONS[mode] || INPUT_MODE_FALLBACK_ICON;
              return (
                <div key={mode} className="flex items-center gap-1.5 rounded-xl bg-surface-800/40 px-3 py-2 text-xs text-surface-300">
                  <Icon className="h-3.5 w-3.5 text-surface-400" />
                  {mode}
                </div>
              );
            })}
          </div>
        </GlassCard>

        <GlassCard hover>
          <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
            <Database className="h-4 w-4 text-surface-400" /> Dataset
          </h3>
          <dl className="space-y-3 text-sm">
            <DL
              label="Status"
              value={
                project.dataset.ready
                  ? <span className="text-emerald-400">Ready</span>
                  : project.dataset.configured
                    ? <span className="text-amber-400">Configured (not downloaded)</span>
                    : <span className="text-surface-500">Not configured</span>
              }
            />
            {project.dataset.type && <DL label="Source" value={project.dataset.type} />}
            {project.dataset.id && <DL label="ID" value={<span className="break-all text-xs text-surface-500 font-mono">{project.dataset.id}</span>} />}
          </dl>
        </GlassCard>

        <GlassCard hover>
          <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
            <Play className="h-4 w-4 text-surface-400" /> Capabilities
          </h3>
          <div className="space-y-2.5">
            <Capability label="Inference CLI" available={project.hasInference} icon={<Terminal className="h-3.5 w-3.5" />} />
            <Capability label="Training Script" available={project.hasTraining} icon={<GraduationCap className="h-3.5 w-3.5" />} />
            <Capability label="Config File" available={project.hasConfig} icon={<Code2 className="h-3.5 w-3.5" />} />
            <Capability label="README" available={project.hasReadme} icon={<FileText className="h-3.5 w-3.5" />} />
          </div>
        </GlassCard>

        <GlassCard hover>
          <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
            <FolderOpen className="h-4 w-4 text-surface-400" /> Project Files
          </h3>
          <div className="space-y-1.5 text-xs font-mono text-surface-500">
            <p className="text-surface-300">{project.folderPath}/</p>
            <p className="pl-4">└── Source Code/</p>
            <p className="pl-8">├── modern.py</p>
            {project.hasConfig && <p className="pl-8">├── config.py</p>}
            {project.hasTraining && <p className="pl-8">├── train.py</p>}
            {project.hasInference && <p className="pl-8">├── infer.py</p>}
            <p className="pl-8">└── ...</p>
          </div>
        </GlassCard>

        <GlassCard hover>
          <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
            <Terminal className="h-4 w-4 text-surface-400" /> Quick Start
          </h3>
          <div className="space-y-3">
            <CodeBlock label="Python API" code={`from core import discover_projects, run\ndiscover_projects()\nresult = run("${project.key}", "image.jpg")`} />
            {project.hasInference && (
              <CodeBlock label="Inference" code={`cd "${project.folderPath}/Source Code"\npython infer.py --source image.jpg`} />
            )}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}

function DL({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-xs text-surface-500">{label}</dt>
      <dd className="text-sm text-surface-300 text-right">{value}</dd>
    </div>
  );
}

function Capability({ label, available, icon }: { label: string; available: boolean; icon: React.ReactNode }) {
  return (
    <div className={`flex items-center gap-2.5 text-xs ${available ? 'text-surface-300' : 'text-surface-600'}`}>
      <div className={`flex h-6 w-6 items-center justify-center rounded-lg ${available ? 'bg-emerald-500/10 text-emerald-400' : 'bg-surface-800/40 text-surface-600'}`}>
        {icon}
      </div>
      {label}
      {available ? (
        <span className="ml-auto text-[10px] font-medium text-emerald-400">Available</span>
      ) : (
        <span className="ml-auto text-[10px] text-surface-600">—</span>
      )}
    </div>
  );
}

function CodeBlock({ label, code }: { label: string; code: string }) {
  return (
    <div>
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-surface-500">{label}</p>
      <pre className="overflow-x-auto rounded-xl bg-surface-950/80 p-3 text-[11px] leading-relaxed text-surface-400">
        <code>{code}</code>
      </pre>
    </div>
  );
}
