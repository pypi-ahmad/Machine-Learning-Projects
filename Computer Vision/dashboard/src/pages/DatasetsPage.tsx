import { Database, Download, CheckCircle2, AlertCircle, HardDrive } from 'lucide-react';
import { SearchBar, EmptyState } from '../components/ui';
import { Badge } from '../components/ui/Badge';
import { useManifest } from '../hooks/useProjects';
import { CATEGORY_META } from '../types';
import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';

export function DatasetsPage() {
  const { manifest, loading, error } = useManifest();
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<'all' | 'configured' | 'ready' | 'missing'>('all');

  const projects = useMemo(() => {
    if (!manifest) return [];
    let list = manifest.projects;

    if (search) {
      const q = search.toLowerCase();
      list = list.filter((p) =>
        `${p.name} ${p.key} ${p.dataset.type} ${p.dataset.id}`.toLowerCase().includes(q)
      );
    }

    if (filter === 'configured') list = list.filter((p) => p.dataset.configured);
    if (filter === 'ready') list = list.filter((p) => p.dataset.ready);
    if (filter === 'missing') list = list.filter((p) => !p.dataset.configured);

    return list;
  }, [manifest, search, filter]);

  if (loading) return <div className="h-96 rounded-3xl bg-surface-800/20 shimmer" />;
  if (error || !manifest) return <div className="text-surface-400">Error loading data</div>;

  const stats = {
    total: manifest.projects.length,
    configured: manifest.projects.filter((p) => p.dataset.configured).length,
    ready: manifest.projects.filter((p) => p.dataset.ready).length,
    missing: manifest.projects.filter((p) => !p.dataset.configured).length,
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-surface-50">Datasets & Models</h1>
        <p className="text-sm text-surface-500">Overview of dataset and model availability across all projects</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MiniStat label="Total Projects" value={stats.total} icon={<HardDrive className="h-4 w-4" />} color="#3b82f6" />
        <MiniStat label="Dataset Configured" value={stats.configured} icon={<Database className="h-4 w-4" />} color="#8b5cf6" />
        <MiniStat label="Data Ready" value={stats.ready} icon={<CheckCircle2 className="h-4 w-4" />} color="#10b981" />
        <MiniStat label="No Dataset Config" value={stats.missing} icon={<AlertCircle className="h-4 w-4" />} color="#f59e0b" />
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3">
        <SearchBar value={search} onChange={setSearch} placeholder="Search datasets..." className="flex-1" />
        <div className="flex gap-1">
          {(['all', 'configured', 'ready', 'missing'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-xl px-3 py-2 text-xs font-medium transition-colors ${
                filter === f ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/20' : 'bg-surface-800/40 text-surface-400 hover:bg-surface-800/60'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-surface-800/50">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b border-surface-800/50 bg-surface-900/60">
              <th className="px-4 py-3 text-left text-xs font-semibold tracking-wide text-surface-500">Project</th>
              <th className="px-4 py-3 text-left text-xs font-semibold tracking-wide text-surface-500">Category</th>
              <th className="px-4 py-3 text-left text-xs font-semibold tracking-wide text-surface-500">Model</th>
              <th className="px-4 py-3 text-left text-xs font-semibold tracking-wide text-surface-500">Dataset</th>
              <th className="px-4 py-3 text-left text-xs font-semibold tracking-wide text-surface-500">Status</th>
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => {
              const catMeta = CATEGORY_META[p.category] || CATEGORY_META['Other'];
              return (
                <tr key={p.key} className="border-b border-surface-800/30 hover:bg-surface-900/40 transition-colors">
                  <td className="px-4 py-3">
                    <Link to={`/projects/${p.key}`} className="text-surface-200 hover:text-primary-400 transition-colors font-medium">
                      {p.name}
                    </Link>
                    <p className="text-[11px] text-surface-600">{p.key}</p>
                  </td>
                  <td className="px-4 py-3">
                    <Badge color={catMeta.color} variant="glass">{p.category}</Badge>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs text-surface-400">{p.modelFamily.join(', ')}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs text-surface-500">{p.dataset.type || '—'}</span>
                  </td>
                  <td className="px-4 py-3">
                    {p.dataset.ready ? (
                      <span className="flex items-center gap-1 text-xs text-emerald-500">
                        <CheckCircle2 className="h-3.5 w-3.5" /> Ready
                      </span>
                    ) : p.dataset.configured ? (
                      <span className="flex items-center gap-1 text-xs text-amber-400">
                        <Download className="h-3.5 w-3.5" /> Configured
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-xs text-surface-600">
                        <AlertCircle className="h-3.5 w-3.5" /> Not configured
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MiniStat({ label, value, icon, color }: { label: string; value: number; icon: React.ReactNode; color: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-surface-800/50 bg-surface-900/40 p-4">
      <div className="flex h-9 w-9 items-center justify-center rounded-xl" style={{ backgroundColor: `${color}12`, color }}>
        {icon}
      </div>
      <div>
        <p className="text-lg font-semibold text-surface-50">{value}</p>
        <p className="text-[11px] text-surface-500">{label}</p>
      </div>
    </div>
  );
}
