import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  GraduationCap, Play, Terminal, AlertTriangle,
  CheckCircle2, Settings as SettingsIcon, Database,
  Copy, Check, FolderOpen, Download,
} from 'lucide-react';
import { Button, LogPanel } from '../components/ui';
import { Badge } from '../components/ui/Badge';
import { ProjectListSelector } from '../components/ProjectListSelector';
import { useProject, useManifest } from '../hooks/useProjects';
import { useBackendStatus, useApiCall } from '../hooks/useApi';
import { startTraining } from '../services/api';
import { CATEGORY_META } from '../types';

export function TrainProjectPage() {
  const { key } = useParams<{ key: string }>();
  const { manifest } = useManifest();

  if (!key) {
    return (
      <ProjectListSelector
        manifest={manifest}
        filter={(p) => p.hasTraining}
        linkTo={(p) => `/projects/${p.key}/train`}
        title="Training"
        subtitle={(n) => `Select a project — ${n} with training support`}
        icon={GraduationCap}
        searchPlaceholder="Search trainable projects..."
      />
    );
  }

  return <TrainInterface projectKey={key} />;
}

function TrainInterface({ projectKey }: { projectKey: string }) {
  const { project, loading, error } = useProject(projectKey);
  const { available } = useBackendStatus();
  const trainApi = useApiCall<{ status: string; message: string; cli_command: string }>();
  const [copied, setCopied] = useState(false);

  const [config, setConfig] = useState({
    epochs: 50,
    batch_size: 16,
    image_size: 640,
    learning_rate: 0.01,
    device: 'auto',
  });

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 rounded-xl bg-surface-800/20 shimmer" />
        <div className="h-72 rounded-3xl bg-surface-800/20 shimmer" />
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center animate-fade-in">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-surface-800/40">
          <FolderOpen className="h-7 w-7 text-surface-500" />
        </div>
        <h2 className="mt-4 text-lg font-semibold text-surface-300">Project Not Found</h2>
        <Link to="/projects" className="mt-4 text-sm text-primary-400 hover:text-primary-300">
          ← Back to projects
        </Link>
      </div>
    );
  }

  if (!project.hasTraining) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-1.5 text-[13px] text-surface-500">
          <Link to={`/projects/${project.key}`} className="transition-colors hover:text-surface-300">
            {project.name}
          </Link>
          <span className="text-surface-600">/</span>
          <span className="text-surface-300">Training</span>
        </div>
        <div className="flex flex-col items-center justify-center py-24 text-center animate-fade-in">
          <AlertTriangle className="h-12 w-12 text-amber-400/50" />
          <h2 className="mt-4 text-lg font-semibold text-surface-300">Training Not Available</h2>
          <p className="mt-1 text-sm text-surface-500">This project does not have a training script.</p>
          <Link to={`/projects/${project.key}`} className="mt-4 text-sm text-primary-400 hover:text-primary-300">
            ← Back to project
          </Link>
        </div>
      </div>
    );
  }

  const catMeta = CATEGORY_META[project.category] || CATEGORY_META['Other'];
  const cliCmd = `cd "${project.folderPath}/Source Code"\npython train.py --epochs ${config.epochs} --batch ${config.batch_size} --imgsz ${config.image_size} --lr ${config.learning_rate} --device ${config.device}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(cliCmd.replace(/\n/g, ' && '));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleTrain = async () => {
    await trainApi.execute(() => startTraining(projectKey, config));
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-[13px] text-surface-500">
        <Link to={`/projects/${project.key}`} className="transition-colors hover:text-surface-300">
          {project.name}
        </Link>
        <span className="text-surface-600">/</span>
        <span className="text-surface-300">Training</span>
      </div>

      {/* Header */}
      <div className="flex items-center gap-4">
        <div
          className="flex h-12 w-12 items-center justify-center rounded-2xl text-lg font-bold shadow-lg"
          style={{ backgroundColor: `${catMeta.color}12`, color: catMeta.color, boxShadow: `0 8px 30px -8px ${catMeta.color}15` }}
        >
          <GraduationCap className="h-5 w-5" />
        </div>
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-surface-50">Train {project.name}</h1>
          <div className="mt-1 flex items-center gap-2">
            <Badge color={catMeta.color} variant="glass">{project.category}</Badge>
            {project.modelFamily.map((m) => <Badge key={m} variant="glass">{m}</Badge>)}
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: Config */}
        <div className="space-y-4">
          {/* Training config */}
          <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
              <SettingsIcon className="h-4 w-4" /> Training Configuration
            </h3>
            <div className="space-y-4">
              <FormField label="Epochs" hint="Number of training epochs">
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: Number(e.target.value) })}
                  min={1} max={1000}
                  className="w-full rounded-xl border border-surface-800/50 bg-surface-900/40 px-3 py-2 text-sm text-surface-200 outline-none focus:border-primary-500/40 focus:ring-2 focus:ring-primary-500/20"
                />
              </FormField>
              <FormField label="Batch Size" hint="Images per batch">
                <select
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: Number(e.target.value) })}
                  className="w-full rounded-xl border border-surface-800/50 bg-surface-900/40 px-3 py-2 text-sm text-surface-200 outline-none focus:border-primary-500/40 focus:ring-2 focus:ring-primary-500/20"
                >
                  {[4, 8, 16, 32, 64].map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </FormField>
              <FormField label="Image Size" hint="Training image resolution">
                <select
                  value={config.image_size}
                  onChange={(e) => setConfig({ ...config, image_size: Number(e.target.value) })}
                  className="w-full rounded-xl border border-surface-800/50 bg-surface-900/40 px-3 py-2 text-sm text-surface-200 outline-none focus:border-primary-500/40 focus:ring-2 focus:ring-primary-500/20"
                >
                  {[320, 416, 512, 640, 800, 1024].map((v) => <option key={v} value={v}>{v}px</option>)}
                </select>
              </FormField>
              <FormField label="Learning Rate" hint="Initial learning rate">
                <input
                  type="number"
                  value={config.learning_rate}
                  onChange={(e) => setConfig({ ...config, learning_rate: Number(e.target.value) })}
                  step={0.001} min={0.0001} max={1}
                  className="w-full rounded-xl border border-surface-800/50 bg-surface-900/40 px-3 py-2 text-sm text-surface-200 outline-none focus:border-primary-500/40 focus:ring-2 focus:ring-primary-500/20"
                />
              </FormField>
              <FormField label="Device" hint="GPU/CPU selection">
                <select
                  value={config.device}
                  onChange={(e) => setConfig({ ...config, device: e.target.value })}
                  className="w-full rounded-xl border border-surface-800/50 bg-surface-900/40 px-3 py-2 text-sm text-surface-200 outline-none focus:border-primary-500/40 focus:ring-2 focus:ring-primary-500/20"
                >
                  <option value="auto">Auto (GPU if available)</option>
                  <option value="0">GPU 0</option>
                  <option value="cpu">CPU</option>
                </select>
              </FormField>
            </div>
          </div>

          {/* Dataset info */}
          <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-surface-200">
              <Database className="h-4 w-4" /> Dataset
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-surface-400">Status</span>
                {project.dataset.ready ? (
                  <span className="flex items-center gap-1.5 text-emerald-400">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Ready
                  </span>
                ) : project.dataset.configured ? (
                  <span className="flex items-center gap-1.5 text-amber-400">
                    <Download className="h-3.5 w-3.5" /> Needs download
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 text-surface-500">
                    <AlertTriangle className="h-3.5 w-3.5" /> Not configured
                  </span>
                )}
              </div>
              {project.dataset.type && (
                <div className="flex items-center justify-between">
                  <span className="text-surface-400">Source</span>
                  <span className="text-surface-300">{project.dataset.type}</span>
                </div>
              )}
              {project.dataset.id && (
                <div className="flex items-center justify-between">
                  <span className="text-surface-400">ID</span>
                  <span className="text-xs text-surface-500 break-all">{project.dataset.id}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Actions & CLI */}
        <div className="space-y-4">
          {/* Launch training */}
          <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-surface-200">
              <Terminal className="h-4 w-4" /> Launch Training
            </h3>

            <div className="mb-4 rounded-xl bg-amber-500/[0.06] border border-amber-500/20 p-3 text-xs text-amber-300">
              <p className="font-medium">Note: Training is best run via CLI</p>
              <p className="mt-1 text-amber-400/70">
                Full training support with live progress streaming is planned for a future release.
                For now, use the CLI command below.
              </p>
            </div>

            <div className="relative">
              <pre className="overflow-x-auto rounded-xl bg-surface-950/80 p-4 pr-12 text-xs leading-relaxed text-surface-400">
                <code>{cliCmd}</code>
              </pre>
              <button
                onClick={handleCopy}
                className="absolute right-2 top-2 rounded-lg p-1.5 text-surface-500 hover:bg-surface-800 hover:text-surface-300 transition-colors"
              >
                {copied ? <Check className="h-4 w-4 text-emerald-500" /> : <Copy className="h-4 w-4" />}
              </button>
            </div>

            <div className="mt-4 flex gap-3">
              <Button
                onClick={handleTrain}
                loading={trainApi.loading}
                disabled={available === false}
                icon={<Play className="h-4 w-4" />}
              >
                Start Training
              </Button>
              {project.hasInference && (
                <Link
                  to={`/projects/${project.key}/run`}
                  className="inline-flex items-center gap-2 rounded-lg border border-surface-700 px-4 py-2 text-sm font-medium text-surface-400 hover:bg-surface-800 hover:text-surface-200 transition-colors"
                >
                  Run Inference →
                </Link>
              )}
            </div>

            {trainApi.data && (
              <div className="mt-4 rounded-lg bg-surface-800/50 p-3 text-xs text-surface-400">
                <p>{trainApi.data.message}</p>
              </div>
            )}

            {trainApi.error && (
              <div className="mt-4 rounded-lg bg-red-500/10 p-3 text-xs text-red-300">
                {trainApi.error}
              </div>
            )}
          </div>

          {/* Training output */}
          <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
            <h3 className="mb-3 text-sm font-semibold text-surface-200">Training Output</h3>
            <div className="flex items-center justify-center py-12 text-center">
              <div>
                <Terminal className="mx-auto h-10 w-10 text-surface-700" />
                <p className="mt-2 text-sm text-surface-500">Run training via the CLI command above</p>
                <p className="mt-1 text-xs text-surface-600">Live training metrics are not yet available in the dashboard</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function FormField({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-surface-400">{label}</label>
      {children}
      {hint && <p className="mt-1 text-[11px] text-surface-600">{hint}</p>}
    </div>
  );
}
