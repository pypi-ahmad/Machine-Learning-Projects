import { Link } from 'react-router-dom';
import {
  FolderOpen, GraduationCap, Database, Play,
  ArrowRight, Star, Zap, Activity, TrendingUp, Sparkles,
} from 'lucide-react';
import { StatCard } from '../components/ui/StatCard';
import { GlassCard } from '../components/ui/GlassCard';
import { DonutChart } from '../components/ui/DonutChart';
import { ProjectCard } from '../components/ProjectCard';
import { useManifest, useFavorites } from '../hooks/useProjects';
import { useBackendStatus } from '../hooks/useApi';
import { CATEGORY_META } from '../types';

export function DashboardPage() {
  const { manifest, loading, error } = useManifest();
  const { favorites, toggle, isFavorite } = useFavorites();
  const { available } = useBackendStatus();

  if (loading) return <PageSkeleton />;
  if (error || !manifest) {
    return (
      <div className="flex h-96 items-center justify-center text-surface-400">
        <p>Failed to load project data: {error}</p>
      </div>
    );
  }

  const { stats, projects } = manifest;
  const favoriteProjects = projects.filter((p) => favorites.includes(p.key)).slice(0, 6);
  const recentProjects = projects.slice(0, 8);
  const categoryEntries = Object.entries(stats.categories).sort((a, b) => b[1] - a[1]);

  const chartSegments = categoryEntries.map(([cat, count]) => ({
    label: cat,
    value: count,
    color: (CATEGORY_META[cat] || CATEGORY_META['Other']).color,
  }));

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-3xl border border-surface-800/40 p-8 lg:p-10">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-900/20 via-surface-900/80 to-surface-950" />
        <div className="absolute -right-20 -top-20 h-80 w-80 rounded-full bg-primary-600/[0.06] blur-[80px]" />
        <div className="absolute -left-10 -bottom-20 h-60 w-60 rounded-full bg-accent-500/[0.04] blur-[60px]" />
        <div className="absolute inset-0 noise" />

        <div className="relative">
          <div className="flex items-start justify-between">
            <div className="max-w-2xl">
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-primary-500/20 bg-primary-600/10 px-3 py-1 text-[11px] font-medium text-primary-400">
                <Sparkles className="h-3 w-3" /> AI Platform · {stats.totalProjects} Projects
              </div>
              <h1 className="text-3xl font-bold tracking-tight text-surface-50 lg:text-4xl">
                Computer Vision
                <span className="bg-gradient-to-r from-primary-400 to-accent-400 bg-clip-text text-transparent"> Projects</span>
              </h1>
              <p className="mt-3 max-w-xl text-sm leading-relaxed text-surface-400">
                Production-quality detection, segmentation, classification, OCR, pose estimation,
                tracking, and retrieval — unified under one framework.
              </p>
            </div>
            <div className="hidden items-center gap-3 lg:flex">
              <div className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-xs font-medium transition-all ${
                available
                  ? 'border border-emerald-500/20 bg-emerald-500/10 text-emerald-400'
                  : 'border border-surface-700/40 bg-surface-800/40 text-surface-500'
              }`}>
                <span className={`h-2 w-2 rounded-full ${available ? 'bg-emerald-400 animate-pulse-dot' : 'bg-surface-600'}`} />
                {available ? 'API Online' : 'API Offline'}
              </div>
            </div>
          </div>
          <div className="mt-6 flex gap-3">
            <Link
              to="/projects"
              className="group inline-flex items-center gap-2 rounded-xl bg-primary-600 px-5 py-2.5 text-sm font-medium text-white shadow-lg shadow-primary-600/20 transition-all hover:bg-primary-500 hover:shadow-primary-600/30"
            >
              <FolderOpen className="h-4 w-4" /> Browse Projects
              <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              to="/run"
              className="inline-flex items-center gap-2 rounded-xl border border-surface-700/50 bg-surface-800/30 px-5 py-2.5 text-sm font-medium text-surface-300 backdrop-blur-sm transition-all hover:border-surface-600 hover:bg-surface-800/50 hover:text-surface-100"
            >
              <Zap className="h-4 w-4" /> Quick Inference
            </Link>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4 stagger-children">
        <StatCard
          label="Projects"
          value={stats.totalProjects}
          icon={<FolderOpen className="h-5 w-5" />}
          accentColor="#6366f1"
          subtitle="Total in repository"
        />
        <StatCard
          label="Trainable"
          value={stats.trainable}
          icon={<GraduationCap className="h-5 w-5" />}
          accentColor="#8b5cf6"
          subtitle={`${Math.round((stats.trainable / stats.totalProjects) * 100)}% of projects`}
        />
        <StatCard
          label="Inference"
          value={stats.withInference}
          icon={<Play className="h-5 w-5" />}
          accentColor="#10b981"
          subtitle="Ready to run"
        />
        <StatCard
          label="Datasets"
          value={stats.withDataset}
          icon={<Database className="h-5 w-5" />}
          accentColor="#f59e0b"
          subtitle={`${stats.dataReady} fully ready`}
        />
      </div>

      {/* Categories + Chart */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SectionHeader title="Categories" to="/projects" />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {categoryEntries.map(([cat, count]) => {
              const meta = CATEGORY_META[cat] || CATEGORY_META['Other'];
              return (
                <Link
                  key={cat}
                  to={`/projects?category=${encodeURIComponent(cat)}`}
                  className="group flex items-center gap-3 rounded-2xl border border-surface-800/40 bg-surface-900/30 p-4 transition-all duration-300 hover:border-surface-700/50 hover:bg-surface-900/50"
                >
                  <div
                    className="flex h-10 w-10 items-center justify-center rounded-xl text-sm font-bold transition-transform duration-300 group-hover:scale-110"
                    style={{ backgroundColor: `${meta.color}12`, color: meta.color }}
                  >
                    {cat.charAt(0)}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-surface-200 transition-colors group-hover:text-primary-400">
                      {cat}
                    </p>
                    <p className="text-[11px] text-surface-500">{count} projects</p>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>

        {/* Distribution chart */}
        <GlassCard className="flex flex-col items-center justify-center" padding="lg">
          <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-surface-500">Distribution</h3>
          <DonutChart segments={chartSegments} size={140} thickness={16} />
          <div className="mt-4 grid w-full grid-cols-2 gap-x-4 gap-y-1">
            {chartSegments.slice(0, 6).map((seg) => (
              <div key={seg.label} className="flex items-center gap-2 text-[11px]">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: seg.color }} />
                <span className="truncate text-surface-400">{seg.label}</span>
                <span className="ml-auto text-surface-500">{seg.value}</span>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Favorites */}
      {favoriteProjects.length > 0 && (
        <section>
          <SectionHeader title="Favorites" icon={<Star className="h-4 w-4 text-amber-400" />} />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 stagger-children">
            {favoriteProjects.map((p) => (
              <ProjectCard key={p.key} project={p} isFavorite={true} onToggleFavorite={toggle} compact />
            ))}
          </div>
        </section>
      )}

      {/* Featured */}
      <section>
        <SectionHeader title="Featured Projects" to="/projects" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 stagger-children">
          {recentProjects.map((p) => (
            <ProjectCard key={p.key} project={p} isFavorite={isFavorite(p.key)} onToggleFavorite={toggle} compact />
          ))}
        </div>
      </section>
    </div>
  );
}

function SectionHeader({ title, to, icon }: { title: string; to?: string; icon?: React.ReactNode }) {
  return (
    <div className="mb-5 flex items-center justify-between">
      <h2 className="flex items-center gap-2 text-base font-semibold text-surface-100">
        {icon}
        {title}
      </h2>
      {to && (
        <Link to={to} className="group flex items-center gap-1 text-xs font-medium text-surface-500 transition-colors hover:text-primary-400">
          View all <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" />
        </Link>
      )}
    </div>
  );
}

function PageSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-48 rounded-3xl bg-surface-800/20 shimmer" />
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => <div key={i} className="h-28 rounded-2xl bg-surface-800/20 shimmer" style={{ animationDelay: `${i * 70}ms` }} />)}
      </div>
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3, 4, 5, 6].map((i) => <div key={i} className="h-40 rounded-2xl bg-surface-800/20 shimmer" style={{ animationDelay: `${i * 70}ms` }} />)}
      </div>
    </div>
  );
}
