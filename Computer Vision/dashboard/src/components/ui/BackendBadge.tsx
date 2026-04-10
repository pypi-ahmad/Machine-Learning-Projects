import { Wifi, WifiOff, Loader2 } from 'lucide-react';
import { useBackendStatus } from '../../hooks/useApi';

export function BackendBadge() {
  const { available, checking, recheck } = useBackendStatus();

  if (checking) {
    return (
      <span className="flex items-center gap-1.5 rounded-xl bg-surface-800/40 px-3 py-1.5 text-[10px] text-surface-500">
        <Loader2 className="h-3 w-3 animate-spin" /> Checking...
      </span>
    );
  }

  return (
    <button
      onClick={recheck}
      className={`flex items-center gap-2 rounded-xl px-3 py-1.5 text-[11px] font-medium transition-all ${
        available
          ? 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15 border border-emerald-500/15'
          : 'bg-surface-800/40 text-surface-500 hover:bg-surface-800/60 border border-surface-800/50'
      }`}
      title={available ? 'Backend connected — click to recheck' : 'Backend offline — click to retry'}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${available ? 'bg-emerald-400 animate-pulse-dot' : 'bg-surface-600'}`} />
      {available ? 'API Connected' : 'API Offline'}
    </button>
  );
}
