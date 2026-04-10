import { useState, useEffect, useMemo, useCallback } from 'react';
import type { ProjectManifest, Project, Filters, SortField, SortDir } from '../types';
import { EMPTY_FILTERS } from '../types';

let _cache: ProjectManifest | null = null;

export async function fetchManifest(): Promise<ProjectManifest> {
  if (_cache) return _cache;
  const res = await fetch('/data/projects.json');
  if (!res.ok) throw new Error(`Failed to load manifest: ${res.status}`);
  _cache = await res.json();
  return _cache!;
}

export function useManifest() {
  const [manifest, setManifest] = useState<ProjectManifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchManifest()
      .then(setManifest)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { manifest, loading, error };
}

export function useProject(key: string | undefined) {
  const { manifest, loading, error } = useManifest();
  const project = useMemo(() => {
    if (!manifest || !key) return null;
    return manifest.projects.find(
      (p) => p.key === key || p.aliases.includes(key)
    ) ?? null;
  }, [manifest, key]);
  return { project, loading, error };
}

function matchesFilter(project: Project, filters: Filters): boolean {
  if (filters.search) {
    const q = filters.search.toLowerCase();
    const haystack = `${project.name} ${project.key} ${project.description} ${project.tags.join(' ')} ${project.modelFamily.join(' ')}`.toLowerCase();
    if (!haystack.includes(q)) return false;
  }
  if (filters.categories.length && !filters.categories.includes(project.category)) return false;
  if (filters.tags.length && !filters.tags.some((t) => project.tags.includes(t))) return false;
  if (filters.modelFamilies.length && !filters.modelFamilies.some((m) => project.modelFamily.includes(m))) return false;
  if (filters.inputModes.length && !filters.inputModes.some((m) => project.inputModes.includes(m))) return false;
  if (filters.hasTraining === true && !project.hasTraining) return false;
  if (filters.hasTraining === false && project.hasTraining) return false;
  if (filters.hasInference === true && !project.hasInference) return false;
  if (filters.datasetReady === true && !project.dataset.ready) return false;
  return true;
}

export function useFilteredProjects(
  projects: Project[],
  filters: Filters,
  sortField: SortField = 'name',
  sortDir: SortDir = 'asc',
) {
  return useMemo(() => {
    let result = projects.filter((p) => matchesFilter(p, filters));

    result.sort((a, b) => {
      const av = a[sortField] ?? '';
      const bv = b[sortField] ?? '';
      const cmp = String(av).localeCompare(String(bv));
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return result;
  }, [projects, filters, sortField, sortDir]);
}

export function useAllTags(projects: Project[]) {
  return useMemo(() => {
    const set = new Set<string>();
    for (const p of projects) p.tags.forEach((t) => set.add(t));
    return Array.from(set).sort();
  }, [projects]);
}

export function useAllModelFamilies(projects: Project[]) {
  return useMemo(() => {
    const set = new Set<string>();
    for (const p of projects) p.modelFamily.forEach((m) => set.add(m));
    return Array.from(set).sort();
  }, [projects]);
}

export function useAllCategories(projects: Project[]) {
  return useMemo(() => {
    const set = new Set<string>();
    for (const p of projects) set.add(p.category);
    return Array.from(set).sort();
  }, [projects]);
}

// Favorites stored in localStorage
const FAV_KEY = 'cv-dashboard-favorites';

export function useFavorites() {
  const [favorites, setFavorites] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(FAV_KEY) || '[]');
    } catch { return []; }
  });

  const favSet = useMemo(() => new Set(favorites), [favorites]);

  const toggle = useCallback((key: string) => {
    setFavorites((prev) => {
      const next = prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key];
      localStorage.setItem(FAV_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const isFavorite = useCallback((key: string) => favSet.has(key), [favSet]);

  return { favorites, toggle, isFavorite };
}
