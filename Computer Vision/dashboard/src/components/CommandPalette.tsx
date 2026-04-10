import { useState, useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, FolderOpen, Play, GraduationCap, LayoutDashboard, Database, Settings, HelpCircle, ArrowRight } from 'lucide-react';
import { useManifest } from '../hooks/useProjects';
import { CATEGORY_META, MAX_COMMAND_PALETTE_RESULTS } from '../types';

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

const PAGES = [
  { label: 'Dashboard', path: '/', icon: LayoutDashboard },
  { label: 'All Projects', path: '/projects', icon: FolderOpen },
  { label: 'Run Inference', path: '/run', icon: Play },
  { label: 'Training', path: '/train', icon: GraduationCap },
  { label: 'Datasets & Models', path: '/datasets', icon: Database },
  { label: 'Settings', path: '/settings', icon: Settings },
  { label: 'Help & Docs', path: '/help', icon: HelpCircle },
];

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { manifest } = useManifest();

  useEffect(() => {
    if (open) {
      setQuery('');
      setSelected(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  const results = useMemo(() => {
    const items: { label: string; sub: string; path: string; icon: typeof Play; color?: string }[] = [];
    const q = query.toLowerCase();

    // Pages
    PAGES.forEach((p) => {
      if (!q || p.label.toLowerCase().includes(q)) {
        items.push({ label: p.label, sub: 'Page', path: p.path, icon: p.icon });
      }
    });

    // Projects
    if (manifest) {
      const matching = manifest.projects.filter((p) =>
        !q || `${p.name} ${p.key} ${p.category} ${p.tags.join(' ')}`.toLowerCase().includes(q),
      );
      matching.slice(0, MAX_COMMAND_PALETTE_RESULTS).forEach((p) => {
        const meta = CATEGORY_META[p.category] || CATEGORY_META['Other'];
        items.push({
          label: p.name,
          sub: `${p.category} · ${p.modelFamily.join(', ')}`,
          path: `/projects/${p.key}`,
          icon: FolderOpen,
          color: meta.color,
        });
      });
    }

    return items;
  }, [query, manifest]);

  useEffect(() => {
    setSelected(0);
  }, [query]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSelected((s) => Math.min(s + 1, results.length - 1)); }
    if (e.key === 'ArrowUp') { e.preventDefault(); setSelected((s) => Math.max(s - 1, 0)); }
    if (e.key === 'Enter' && results[selected]) {
      navigate(results[selected].path);
      onClose();
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]" onClick={onClose}>
      <div className="absolute inset-0 bg-surface-950/70 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-xl animate-fade-in overflow-hidden rounded-2xl border border-surface-700/50 bg-surface-900/95 shadow-2xl backdrop-blur-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 border-b border-surface-800 px-5 py-4">
          <Search className="h-5 w-5 text-surface-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Search projects, pages, commands..."
            className="flex-1 bg-transparent text-sm text-surface-100 placeholder:text-surface-500 outline-none"
          />
          <kbd className="rounded border border-surface-700 bg-surface-800 px-1.5 py-0.5 text-[10px] text-surface-500">ESC</kbd>
        </div>

        {/* Results */}
        <div className="max-h-[340px] overflow-y-auto p-2">
          {results.length === 0 && (
            <div className="py-10 text-center text-sm text-surface-500">
              No results found
            </div>
          )}
          {results.map((item, i) => (
            <button
              key={`${item.path}-${i}`}
              onClick={() => { navigate(item.path); onClose(); }}
              onMouseEnter={() => setSelected(i)}
              className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left transition-colors ${
                i === selected ? 'bg-primary-600/10 text-surface-100' : 'text-surface-400 hover:bg-surface-800/60'
              }`}
            >
              <div
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
                style={{
                  backgroundColor: item.color ? `${item.color}15` : 'rgba(99,102,241,0.1)',
                  color: item.color || '#818cf8',
                }}
              >
                <item.icon className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{item.label}</p>
                <p className="truncate text-xs text-surface-500">{item.sub}</p>
              </div>
              {i === selected && <ArrowRight className="h-3.5 w-3.5 text-primary-400" />}
            </button>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-4 border-t border-surface-800 px-5 py-2.5 text-[10px] text-surface-500">
          <span className="flex items-center gap-1"><kbd className="rounded border border-surface-700 bg-surface-800 px-1 py-0.5">↑↓</kbd> Navigate</span>
          <span className="flex items-center gap-1"><kbd className="rounded border border-surface-700 bg-surface-800 px-1 py-0.5">↵</kbd> Open</span>
          <span className="flex items-center gap-1"><kbd className="rounded border border-surface-700 bg-surface-800 px-1 py-0.5">Esc</kbd> Close</span>
        </div>
      </div>
    </div>
  );
}
