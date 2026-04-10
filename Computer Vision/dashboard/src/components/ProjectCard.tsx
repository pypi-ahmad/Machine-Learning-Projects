import { Link } from 'react-router-dom';
import { Star, Play, GraduationCap, Database, ArrowUpRight } from 'lucide-react';
import { Badge } from './ui/Badge';
import type { Project } from '../types';
import { CATEGORY_META } from '../types';

interface ProjectCardProps {
  project: Project;
  isFavorite: boolean;
  onToggleFavorite: (key: string) => void;
  compact?: boolean;
}

export function ProjectCard({ project, isFavorite, onToggleFavorite, compact }: ProjectCardProps) {
  const catMeta = CATEGORY_META[project.category] || CATEGORY_META['Other'];

  return (
    <Link
      to={`/projects/${project.key}`}
      className="group relative flex flex-col overflow-hidden rounded-2xl border border-surface-800/50 bg-surface-900/40 transition-all duration-300 hover:border-surface-700/60 hover:bg-surface-900/60 hover:shadow-xl hover:shadow-surface-950/40"
    >
      {/* Gradient accent stripe */}
      <div className="h-[3px]" style={{ background: `linear-gradient(90deg, ${catMeta.color}, ${catMeta.color}40)` }} />

      <div className="flex flex-1 flex-col p-5">
        {/* Favorite button */}
        <button
          onClick={(e) => { e.preventDefault(); e.stopPropagation(); onToggleFavorite(project.key); }}
          className="absolute right-3.5 top-5 p-1.5 text-surface-600 transition-all hover:scale-110 hover:text-amber-400"
          aria-label={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
        >
          <Star className={`h-3.5 w-3.5 ${isFavorite ? 'fill-amber-400 text-amber-400' : ''}`} />
        </button>

        {/* Header */}
        <div className="mb-3 flex items-start gap-3">
          <div
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-sm font-bold"
            style={{ backgroundColor: `${catMeta.color}12`, color: catMeta.color }}
          >
            {project.category.charAt(0)}
          </div>
          <div className="min-w-0 flex-1 pr-6">
            <h3 className="truncate text-sm font-semibold text-surface-100 transition-colors group-hover:text-primary-400">
              {project.name}
            </h3>
            <p className="text-[11px] text-surface-500 font-mono">{project.key}</p>
          </div>
        </div>

        {/* Description */}
        {!compact && (
          <p className="mb-4 line-clamp-2 text-[12px] leading-relaxed text-surface-400">
            {project.description}
          </p>
        )}

        {/* Badges */}
        <div className="mb-4 flex flex-wrap gap-1.5">
          <Badge color={catMeta.color} variant="glass" size="sm">{project.category}</Badge>
          {project.modelFamily.slice(0, 2).map((m) => (
            <Badge key={m} variant="glass" size="sm">{m}</Badge>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-auto flex items-center gap-2.5 border-t border-surface-800/40 pt-3.5 text-[11px] text-surface-500">
          {project.hasInference && (
            <span className="flex items-center gap-1 text-emerald-400/80">
              <Play className="h-3 w-3" /> Infer
            </span>
          )}
          {project.hasTraining && (
            <span className="flex items-center gap-1 text-primary-400/80">
              <GraduationCap className="h-3 w-3" /> Train
            </span>
          )}
          {project.dataset.configured && (
            <span className="flex items-center gap-1 text-purple-400/80">
              <Database className="h-3 w-3" /> Data
            </span>
          )}
          <ArrowUpRight className="ml-auto h-3.5 w-3.5 text-surface-600 transition-all group-hover:text-primary-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </div>
      </div>
    </Link>
  );
}
