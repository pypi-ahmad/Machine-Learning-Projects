import type { Filters } from '../types';
import { CATEGORY_META } from '../types';

interface FilterPanelProps {
  filters: Filters;
  onChange: (filters: Filters) => void;
  categories: string[];
  tags: string[];
  modelFamilies: string[];
}

export function FilterPanel({ filters, onChange, categories, tags, modelFamilies }: FilterPanelProps) {
  const toggleArray = (arr: string[], val: string) =>
    arr.includes(val) ? arr.filter((v) => v !== val) : [...arr, val];

  return (
    <div className="space-y-5">
      {/* Categories */}
      <Section title="Category">
        <div className="flex flex-wrap gap-1.5">
          {categories.map((cat) => {
            const active = filters.categories.includes(cat);
            const meta = CATEGORY_META[cat];
            return (
              <button
                key={cat}
                onClick={() => onChange({ ...filters, categories: toggleArray(filters.categories, cat) })}
                className={`rounded-full px-2.5 py-1 text-xs font-medium transition-all ${
                  active
                    ? 'text-white shadow-md'
                    : 'bg-surface-800/40 text-surface-400 hover:bg-surface-800/60'
                }`}
                style={active ? { backgroundColor: meta?.color || CATEGORY_META['Other'].color } : undefined}
              >
                {cat}
              </button>
            );
          })}
        </div>
      </Section>

      {/* Model Family */}
      <Section title="Model Family">
        <div className="flex flex-wrap gap-1.5">
          {modelFamilies.map((fam) => {
            const active = filters.modelFamilies.includes(fam);
            return (
              <button
                key={fam}
                onClick={() => onChange({ ...filters, modelFamilies: toggleArray(filters.modelFamilies, fam) })}
                className={`rounded-full px-2.5 py-1 text-xs font-medium transition-all ${
                  active
                    ? 'bg-primary-600 text-white shadow-md shadow-primary-600/20'
                    : 'bg-surface-800/40 text-surface-400 hover:bg-surface-800/60'
                }`}
              >
                {fam}
              </button>
            );
          })}
        </div>
      </Section>

      {/* Tags */}
      <Section title="Domain Tags">
        <div className="flex flex-wrap gap-1.5">
          {tags.map((tag) => {
            const active = filters.tags.includes(tag);
            return (
              <button
                key={tag}
                onClick={() => onChange({ ...filters, tags: toggleArray(filters.tags, tag) })}
                className={`rounded-full px-2.5 py-1 text-xs font-medium transition-all ${
                  active
                    ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/20'
                    : 'bg-surface-800/40 text-surface-400 hover:bg-surface-800/60'
                }`}
              >
                {tag}
              </button>
            );
          })}
        </div>
      </Section>

      {/* Capabilities */}
      <Section title="Capabilities">
        <div className="space-y-2">
          <Toggle
            label="Trainable"
            active={filters.hasTraining === true}
            onClick={() => onChange({ ...filters, hasTraining: filters.hasTraining === true ? null : true })}
          />
          <Toggle
            label="Has Inference CLI"
            active={filters.hasInference === true}
            onClick={() => onChange({ ...filters, hasInference: filters.hasInference === true ? null : true })}
          />
        </div>
      </Section>

      {/* Clear all */}
      {hasActiveFilters(filters) && (
        <button
          onClick={() => onChange({
            search: filters.search,
            categories: [],
            tags: [],
            modelFamilies: [],
            inputModes: [],
            hasTraining: null,
            hasInference: null,
            datasetReady: null,
          })}
          className="w-full rounded-xl border border-surface-800/50 py-2 text-xs font-medium text-surface-400 hover:bg-surface-800/40 hover:text-surface-200 transition-colors"
        >
          Clear all filters
        </button>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-surface-500">{title}</h4>
      {children}
    </div>
  );
}

function Toggle({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className="flex items-center gap-2 text-xs text-surface-400 hover:text-surface-200 transition-colors">
      <div className={`h-4 w-4 rounded border transition-colors ${active ? 'border-primary-500 bg-primary-600 shadow-sm shadow-primary-600/30' : 'border-surface-600'}`}>
        {active && (
          <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>
      {label}
    </button>
  );
}

function hasActiveFilters(f: Filters): boolean {
  return f.categories.length > 0 || f.tags.length > 0 || f.modelFamilies.length > 0 ||
    f.inputModes.length > 0 || f.hasTraining !== null || f.hasInference !== null || f.datasetReady !== null;
}
