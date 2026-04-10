import { Search as SearchIcon, X } from 'lucide-react';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export function SearchBar({ value, onChange, placeholder = 'Search projects...', className = '' }: SearchBarProps) {
  return (
    <div className={`relative ${className}`}>
      <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-500" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-xl border border-surface-800/50 bg-surface-900/40 py-2.5 pl-11 pr-10 text-sm text-surface-100 placeholder:text-surface-500 outline-none transition-all focus:border-primary-500/40 focus:bg-surface-900/60 focus:ring-1 focus:ring-primary-500/20"
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className="absolute right-3 top-1/2 -translate-y-1/2 rounded-lg p-1 text-surface-500 transition-colors hover:bg-surface-800/40 hover:text-surface-300"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
