import { useRef, useEffect } from 'react';
import { Terminal, X } from 'lucide-react';

interface LogPanelProps {
  lines: string[];
  title?: string;
  maxHeight?: string;
  onClear?: () => void;
  className?: string;
}

export function LogPanel({
  lines,
  title = 'Logs',
  maxHeight = '300px',
  onClear,
  className = '',
}: LogPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines.length]);

  return (
    <div className={`rounded-2xl border border-surface-800/50 bg-surface-950/80 ${className}`}>
      <div className="flex items-center justify-between border-b border-surface-800/50 px-4 py-2">
        <span className="flex items-center gap-2 text-xs font-medium text-surface-400">
          <Terminal className="h-3.5 w-3.5" />
          {title}
          {lines.length > 0 && (
            <span className="rounded-full bg-surface-800/60 px-1.5 py-0.5 text-[10px]">
              {lines.length}
            </span>
          )}
        </span>
        {onClear && lines.length > 0 && (
          <button
            onClick={onClear}
            className="text-surface-600 hover:text-surface-400 transition-colors"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      <div
        ref={scrollRef}
        className="overflow-auto p-3 font-mono text-xs leading-relaxed"
        style={{ maxHeight }}
      >
        {lines.length === 0 ? (
          <p className="text-surface-600">No logs yet...</p>
        ) : (
          lines.map((line, i) => (
            <div key={i} className={`${getLogColor(line)} whitespace-pre-wrap break-all`}>
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function getLogColor(line: string): string {
  const lower = line.toLowerCase();
  if (lower.includes('error') || lower.includes('failed') || lower.includes('exception'))
    return 'text-red-400';
  if (lower.includes('warn'))
    return 'text-amber-400';
  if (lower.includes('success') || lower.includes('completed') || lower.includes('✓') || lower.includes('ok'))
    return 'text-emerald-400';
  if (lower.includes('info'))
    return 'text-blue-400';
  return 'text-surface-400';
}
