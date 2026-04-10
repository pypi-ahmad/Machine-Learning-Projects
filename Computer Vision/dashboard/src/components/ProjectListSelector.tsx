import { useState } from 'react';
import { Link } from 'react-router-dom';
import { CATEGORY_META } from '../types';
import type { Project, ProjectManifest } from '../types';
import { useDebouncedValue } from '../hooks/useDebouncedValue';

const MAX_VISIBLE_RESULTS = 80;

interface ProjectListSelectorProps {
  manifest: ProjectManifest | null;
  filter: (project: Project) => boolean;
  linkTo: (project: Project) => string;
  title: string;
  subtitle: (count: number) => string;
  icon: React.ComponentType<{ className?: string }>;
  searchPlaceholder?: string;
}

export function ProjectListSelector({
  manifest,
  filter,
  linkTo,
  title,
  subtitle,
  icon: Icon,
  searchPlaceholder = 'Search projects...',
}: ProjectListSelectorProps) {
  const [search, setSearch] = useState('');
  const debouncedSearch = useDebouncedValue(search, 150);
  const projects = manifest?.projects ?? [];
  const eligible = projects.filter(filter);

  const filtered = debouncedSearch
    ? eligible.filter((p) =>
        `${p.name} ${p.key} ${p.category} ${p.tags.join(' ')} ${p.modelFamily.join(' ')}`
          .toLowerCase()
          .includes(debouncedSearch.toLowerCase()),
      )
    : eligible;

  const visible = filtered.slice(0, MAX_VISIBLE_RESULTS);

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-surface-50">{title}</h1>
        <p className="text-sm text-surface-500">
          {subtitle(eligible.length)}
        </p>
      </div>

      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder={searchPlaceholder}
        className="w-full rounded-xl border border-surface-800/50 bg-surface-900/40 py-2.5 px-4 text-sm text-surface-100 placeholder:text-surface-500 outline-none focus:border-primary-500/40 focus:ring-2 focus:ring-primary-500/20"
      />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {visible.map((p) => {
          const catMeta = CATEGORY_META[p.category] || CATEGORY_META['Other'];
          return (
            <Link
              key={p.key}
              to={linkTo(p)}
              className="group flex items-center gap-3 rounded-2xl border border-surface-800/50 bg-surface-900/40 p-4 transition-all hover:border-surface-700 hover:bg-surface-900/60"
            >
              <div
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-sm font-bold"
                style={{ backgroundColor: `${catMeta.color}15`, color: catMeta.color }}
              >
                <Icon className="h-4 w-4" />
              </div>
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-surface-200 transition-colors group-hover:text-primary-400">
                  {p.name}
                </p>
                <p className="text-xs text-surface-500">{p.category} · {p.modelFamily.join(', ')}</p>
              </div>
            </Link>
          );
        })}
      </div>

      {filtered.length > MAX_VISIBLE_RESULTS && (
        <p className="text-center text-xs text-surface-500">
          Showing {MAX_VISIBLE_RESULTS} of {filtered.length} — refine your search to see more
        </p>
      )}

      {filtered.length === 0 && (
        <div className="py-16 text-center text-sm text-surface-500 animate-fade-in">
          No matching projects found
        </div>
      )}
    </div>
  );
}
