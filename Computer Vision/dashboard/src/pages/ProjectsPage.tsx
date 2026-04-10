import { useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { FolderOpen, SlidersHorizontal, Grid3x3, LayoutList } from 'lucide-react';
import { SearchBar, EmptyState } from '../components/ui';
import { ProjectCard } from '../components/ProjectCard';
import { FilterPanel } from '../components/FilterPanel';
import {
  useManifest, useFilteredProjects, useFavorites,
  useAllCategories, useAllTags, useAllModelFamilies,
} from '../hooks/useProjects';
import { useDebouncedValue } from '../hooks/useDebouncedValue';
import type { Filters, SortField, SortDir } from '../types';
import { EMPTY_FILTERS } from '../types';

const PAGE_SIZE = 48;

export function ProjectsPage() {
  const { manifest, loading, error } = useManifest();
  const { toggle, isFavorite } = useFavorites();
  const [searchParams] = useSearchParams();
  const [showFilters, setShowFilters] = useState(true);
  const [sortField, setSortField] = useState<SortField>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const initialCat = searchParams.get('category');
  const [filters, setFilters] = useState<Filters>({
    ...EMPTY_FILTERS,
    categories: initialCat ? [initialCat] : [],
  });
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  const updateFilters = useCallback((next: Filters) => {
    setFilters(next);
    setVisibleCount(PAGE_SIZE);
  }, []);

  const projects = manifest?.projects ?? [];
  const debouncedFilters = useDebouncedValue(filters, 150);
  const filtered = useFilteredProjects(projects, debouncedFilters, sortField, sortDir);
  const categories = useAllCategories(projects);
  const tags = useAllTags(projects);
  const modelFamilies = useAllModelFamilies(projects);

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="h-10 rounded-xl bg-surface-800/20 shimmer" />
        <div className="grid grid-cols-3 gap-4">
          {Array.from({ length: 9 }).map((_, i) => <div key={i} className="h-52 rounded-2xl bg-surface-800/20 shimmer" style={{ animationDelay: `${i * 70}ms` }} />)}
        </div>
      </div>
    );
  }

  if (error) return <div className="text-surface-400">Error: {error}</div>;

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-surface-50">Projects</h1>
          <p className="mt-1 text-sm text-surface-500">
            <span className="font-medium text-surface-300">{filtered.length}</span> of {projects.length} projects
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={`${sortField}-${sortDir}`}
            onChange={(e) => {
              const [f, d] = e.target.value.split('-') as [SortField, SortDir];
              setSortField(f);
              setSortDir(d);
            }}
            className="rounded-xl border border-surface-800/50 bg-surface-900/40 px-3 py-2 text-xs text-surface-300 outline-none focus:border-primary-500/50 transition-colors"
          >
            <option value="name-asc">Name A–Z</option>
            <option value="name-desc">Name Z–A</option>
            <option value="category-asc">Category A–Z</option>
            <option value="category-desc">Category Z–A</option>
          </select>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`rounded-xl border p-2.5 transition-all ${
              showFilters
                ? 'border-primary-500/30 bg-primary-600/10 text-primary-400'
                : 'border-surface-800/50 text-surface-400 hover:bg-surface-800/40'
            }`}
          >
            <SlidersHorizontal className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Search */}
      <SearchBar
        value={filters.search}
        onChange={(search) => updateFilters({ ...filters, search })}
        placeholder="Search by name, key, description, model, tag..."
      />

      {/* Content */}
      <div className="flex gap-6">
        {/* Filters sidebar */}
        {showFilters && (
          <div className="hidden w-56 shrink-0 lg:block">
            <FilterPanel
              filters={filters}
              onChange={updateFilters}
              categories={categories}
              tags={tags}
              modelFamilies={modelFamilies}
            />
          </div>
        )}

        {/* Project grid */}
        <div className="flex-1">
          {filtered.length === 0 ? (
            <EmptyState
              icon={<FolderOpen className="h-12 w-12" />}
              title="No projects match your filters"
              description="Try adjusting your search or filter criteria"
              action={
                <button
                  onClick={() => updateFilters(EMPTY_FILTERS)}
                  className="rounded-xl bg-primary-600 px-4 py-2 text-sm font-medium text-white shadow-lg shadow-primary-600/20 hover:bg-primary-500 transition-all"
                >
                  Clear all filters
                </button>
              }
            />
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 stagger-children">
                {filtered.slice(0, visibleCount).map((p) => (
                  <ProjectCard
                    key={p.key}
                    project={p}
                    isFavorite={isFavorite(p.key)}
                    onToggleFavorite={toggle}
                  />
                ))}
              </div>
              {filtered.length > visibleCount && (
                <div className="mt-6 flex justify-center">
                  <button
                    onClick={() => setVisibleCount((c) => c + PAGE_SIZE)}
                    className="rounded-xl border border-surface-700/50 bg-surface-800/30 px-6 py-2.5 text-sm font-medium text-surface-300 transition-all hover:border-surface-600 hover:bg-surface-800/50 hover:text-surface-100"
                  >
                    Show more ({filtered.length - visibleCount} remaining)
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
