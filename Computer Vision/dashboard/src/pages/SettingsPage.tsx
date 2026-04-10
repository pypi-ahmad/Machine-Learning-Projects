import { Settings, Palette, RefreshCw, Monitor, Activity, Server } from 'lucide-react';
import { useManifest } from '../hooks/useProjects';
import { useBackendStatus, useSystemInfo } from '../hooks/useApi';

export function SettingsPage() {
  const { manifest } = useManifest();
  const { available, checking, recheck } = useBackendStatus();
  const { info: systemInfo, loading: sysLoading } = useSystemInfo();

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-surface-50">Settings</h1>
        <p className="text-sm text-surface-500">Dashboard configuration, system status, and preferences</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* System Status */}
        <SettingsCard title="System Status" icon={<Activity className="h-4 w-4" />}>
          <div className="space-y-3 text-sm">
            <SettingsRow
              label="Backend API"
              value={
                checking ? 'Checking...' :
                available ? 'Online' : 'Offline'
              }
              status={checking ? 'neutral' : available ? 'ok' : 'error'}
            />
            {systemInfo && (
              <>
                <SettingsRow
                  label="GPU"
                  value={systemInfo.gpu_available ? systemInfo.gpu_name : 'Not available'}
                  status={systemInfo.gpu_available ? 'ok' : 'error'}
                />
                {systemInfo.gpu_available && (
                  <SettingsRow label="GPU Memory" value={`${systemInfo.gpu_memory_gb} GB`} />
                )}
                {systemInfo.cuda_version && (
                  <SettingsRow label="CUDA Version" value={systemInfo.cuda_version} />
                )}
                <SettingsRow label="PyTorch" value={systemInfo.torch_version || '—'} />
                <SettingsRow label="Python" value={systemInfo.python_version} />
                <SettingsRow label="Discovered Projects" value={String(systemInfo.projects_discovered)} />
              </>
            )}
            {!systemInfo && !sysLoading && available === false && (
              <p className="text-xs text-surface-500">Start the backend API to see system info</p>
            )}
            <button
              onClick={recheck}
              className="mt-2 rounded-lg border border-surface-700 px-3 py-1.5 text-xs font-medium text-surface-400 hover:bg-surface-800 hover:text-surface-200 transition-colors"
            >
              <RefreshCw className="mr-1 inline-block h-3 w-3" /> Refresh
            </button>
          </div>
        </SettingsCard>

        {/* General */}
        <SettingsCard title="General" icon={<Settings className="h-4 w-4" />}>
          <dl className="space-y-3 text-sm">
            <SettingsRow label="Dashboard Version" value="2.0.0" />
            <SettingsRow label="Repo Version" value={manifest?.repoVersion ?? '—'} />
            <SettingsRow label="Manifest Generated" value={manifest?.generatedAt ?? '—'} />
            <SettingsRow label="Total Projects" value={String(manifest?.stats.totalProjects ?? 0)} />
            <SettingsRow label="Trainable Projects" value={String(manifest?.stats.trainable ?? 0)} />
            <SettingsRow label="With Inference" value={String(manifest?.stats.withInference ?? 0)} />
          </dl>
        </SettingsCard>

        {/* Appearance */}
        <SettingsCard title="Appearance" icon={<Palette className="h-4 w-4" />}>
          <p className="text-xs text-surface-500">
            The dashboard uses a dark theme optimized for extended use.
            Theming is applied via CSS custom properties in <code className="text-surface-400">index.css</code>.
          </p>
          <div className="mt-3 grid grid-cols-5 gap-2">
            {['primary', 'surface', 'emerald', 'amber', 'red'].map((color) => (
              <div key={color} className="text-center">
                <div className={`mx-auto h-8 w-8 rounded-lg ${
                  color === 'primary' ? 'bg-primary-600' :
                  color === 'surface' ? 'bg-surface-600' :
                  color === 'emerald' ? 'bg-emerald-600' :
                  color === 'amber' ? 'bg-amber-500' :
                  'bg-red-600'
                }`} />
                <p className="mt-1 text-[10px] text-surface-500">{color}</p>
              </div>
            ))}
          </div>
        </SettingsCard>

        {/* Data Refresh */}
        <SettingsCard title="Data Refresh" icon={<RefreshCw className="h-4 w-4" />}>
          <p className="text-xs text-surface-400 mb-3">
            Re-scan the repository to update the project manifest. This runs
            the Python scanner and regenerates <code className="text-surface-300">projects.json</code>.
          </p>
          <code className="block rounded-xl bg-surface-950/80 p-3 text-[11px] text-surface-400">
            python dashboard/scripts/scan_projects.py --pretty
          </code>
        </SettingsCard>

        {/* Backend */}
        <SettingsCard title="API Backend" icon={<Server className="h-4 w-4" />}>
          <p className="text-xs text-surface-400 mb-3">
            The FastAPI backend enables live inference, training triggers, and dataset management.
            The dashboard works in read-only mode without the backend.
          </p>
          <div className="space-y-2">
            <code className="block rounded-xl bg-surface-950/80 p-3 text-[11px] text-surface-400">
              python dashboard/api/server.py
            </code>
            <p className="text-[11px] text-surface-500">
              Runs on <code className="text-surface-400">localhost:8042</code> — Swagger docs at <code className="text-surface-400">/api/docs</code>
            </p>
          </div>
        </SettingsCard>

        {/* Keyboard Shortcuts */}
        <SettingsCard title="Keyboard Shortcuts" icon={<Monitor className="h-4 w-4" />}>
          <div className="space-y-2 text-xs">
            <ShortcutRow keys={['⌘', 'K']} action="Command palette" />
            <ShortcutRow keys={['/']} action="Focus search" />
            <ShortcutRow keys={['G', 'D']} action="Go to Dashboard" />
            <ShortcutRow keys={['G', 'P']} action="Go to Projects" />
          </div>
        </SettingsCard>
      </div>
    </div>
  );
}

function SettingsCard({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">{icon} {title}</h3>
      {children}
    </div>
  );
}

function SettingsRow({ label, value, status }: { label: string; value: string; status?: 'ok' | 'error' | 'neutral' }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-surface-400">{label}</span>
      <span className="flex items-center gap-1.5 text-surface-200 font-mono text-xs">
        {status === 'ok' && <span className="h-2 w-2 rounded-full bg-emerald-400" />}
        {status === 'error' && <span className="h-2 w-2 rounded-full bg-red-400" />}
        {value}
      </span>
    </div>
  );
}

function ShortcutRow({ keys, action }: { keys: string[]; action: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-surface-400">{action}</span>
      <div className="flex items-center gap-1">
        {keys.map((k, i) => (
          <span key={i}>
            <kbd className="rounded-md border border-surface-700/60 bg-surface-800/60 px-1.5 py-0.5 text-[10px] font-mono text-surface-300">
              {k}
            </kbd>
            {i < keys.length - 1 && <span className="mx-0.5 text-surface-600">+</span>}
          </span>
        ))}
      </div>
    </div>
  );
}
