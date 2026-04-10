import { useState } from 'react';
import { ChevronDown, ChevronRight, Image as ImageIcon, Copy, Check } from 'lucide-react';

interface OutputViewerProps {
  image?: string | null;
  data?: Record<string, unknown> | null;
  className?: string;
}

export function OutputViewer({ image, data, className = '' }: OutputViewerProps) {
  const [showJson, setShowJson] = useState(false);
  const [copied, setCopied] = useState(false);

  const hasImage = !!image;
  const hasData = data && Object.keys(data).length > 0;

  if (!hasImage && !hasData) {
    return (
      <div className={`flex items-center justify-center rounded-2xl border border-dashed border-surface-700/50 bg-surface-900/20 py-16 ${className}`}>
        <div className="text-center">
          <ImageIcon className="mx-auto h-10 w-10 text-surface-600" />
          <p className="mt-2 text-sm text-surface-500">Output will appear here</p>
        </div>
      </div>
    );
  }

  const copyJson = () => {
    if (data) {
      navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {hasImage && (
        <div className="overflow-hidden rounded-2xl border border-surface-800/50 bg-surface-950/80">
          <img
            src={image!}
            alt="Visualization output"
            className="mx-auto max-h-[500px] object-contain"
          />
        </div>
      )}

      {hasData && (
        <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40">
          <div className="flex w-full items-center justify-between px-4 py-3">
            <button
              onClick={() => setShowJson(!showJson)}
              className="flex items-center gap-2 text-sm font-medium text-surface-300 hover:text-surface-100 transition-colors"
            >
              {showJson ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Prediction Results
            </button>
            <button
              onClick={copyJson}
              className="flex items-center gap-1 rounded-xl px-2 py-1 text-xs text-surface-500 hover:bg-surface-800/40 hover:text-surface-300"
            >
              {copied ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>

          {showJson && (
            <div className="border-t border-surface-800/50 px-4 py-3">
              <pre className="max-h-80 overflow-auto text-xs leading-relaxed text-surface-400">
                <code>{JSON.stringify(data, null, 2)}</code>
              </pre>
            </div>
          )}

          {!showJson && (
            <div className="border-t border-surface-800/50 px-4 py-3">
              <ResultSummary data={data} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ResultSummary({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data).slice(0, 10);

  return (
    <div className="space-y-1.5">
      {entries.map(([key, value]) => (
        <div key={key} className="flex items-start gap-2 text-xs">
          <span className="shrink-0 font-medium text-surface-400">{key}:</span>
          <span className="text-surface-300 break-all">
            {typeof value === 'object' && value !== null
              ? Array.isArray(value)
                ? `[${(value as unknown[]).length} items]`
                : `{${Object.keys(value as Record<string, unknown>).length} keys}`
              : String(value)}
          </span>
        </div>
      ))}
      {Object.keys(data).length > 10 && (
        <p className="text-[11px] text-surface-600">...and {Object.keys(data).length - 10} more fields</p>
      )}
    </div>
  );
}
