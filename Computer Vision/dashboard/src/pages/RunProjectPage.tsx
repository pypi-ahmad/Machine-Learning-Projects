import { useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Play, AlertTriangle, Loader2, CheckCircle2, XCircle, Zap, FolderOpen,
} from 'lucide-react';
import { FileUpload, OutputViewer, LogPanel, Button } from '../components/ui';
import { Badge } from '../components/ui/Badge';
import { ProjectListSelector } from '../components/ProjectListSelector';
import { useProject, useManifest } from '../hooks/useProjects';
import { useBackendStatus, useApiCall } from '../hooks/useApi';
import { runVisualize, runInference } from '../services/api';
import { CATEGORY_META } from '../types';
import { INPUT_MODE_ICONS, INPUT_MODE_FALLBACK_ICON } from '../constants';
import type { VisualizationResult, PredictionResult, RunStatus } from '../types';

export function RunProjectPage() {
  const { key } = useParams<{ key: string }>();
  const { manifest } = useManifest();
  const { available } = useBackendStatus();

  if (!key) {
    return (
      <ProjectListSelector
        manifest={manifest}
        filter={(p) => p.hasInference}
        linkTo={(p) => `/projects/${p.key}/run`}
        title="Run Inference"
        subtitle={(n) => `Select a project — ${n} with inference support`}
        icon={Play}
        searchPlaceholder="Search inference-ready projects..."
      />
    );
  }

  return <RunInterface projectKey={key} backendAvailable={available} />;
}

function RunInterface({ projectKey, backendAvailable }: { projectKey: string; backendAvailable: boolean | null }) {
  const { project, loading, error } = useProject(projectKey);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [mode, setMode] = useState<'visualize' | 'predict'>('visualize');
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<RunStatus>('idle');

  const visualizeCall = useApiCall<VisualizationResult>();
  const predictCall = useApiCall<PredictionResult>();

  const handleFiles = useCallback((files: File[]) => {
    if (files.length > 0) {
      setSelectedFile(files[0]);
      setStatus('idle');
      visualizeCall.reset();
      predictCall.reset();
      setLogs([]);
    }
  }, [visualizeCall, predictCall]);

  const handleRun = useCallback(async () => {
    if (!selectedFile || !projectKey) return;

    setStatus('uploading');
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Uploading ${selectedFile.name}...`]);

    setStatus('processing');
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Running ${mode} on ${projectKey}...`]);

    const result = mode === 'visualize'
      ? await visualizeCall.execute(() => runVisualize(projectKey, selectedFile))
      : await predictCall.execute(() => runInference(projectKey, selectedFile));

    if (result.error) {
      setStatus('error');
      setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ✗ Failed — ${result.error}`]);
    } else {
      setStatus('done');
      setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ✓ Completed`]);
    }
  }, [selectedFile, projectKey, mode, visualizeCall, predictCall]);

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
        <p className="mt-1 text-sm text-surface-500">No project with key "{projectKey}"</p>
        <Link to="/projects" className="mt-4 text-sm text-primary-400 hover:text-primary-300">
          ← Back to projects
        </Link>
      </div>
    );
  }

  const catMeta = CATEGORY_META[project.category] || CATEGORY_META['Other'];
  const isRunning = status === 'uploading' || status === 'processing';

  const resultImage = visualizeCall.data?.image || null;
  const resultData = mode === 'visualize'
    ? visualizeCall.data?.prediction || null
    : predictCall.data || null;
  const resultError = visualizeCall.error || predictCall.error;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-[13px] text-surface-500">
        <Link to={`/projects/${project.key}`} className="transition-colors hover:text-surface-300">
          {project.name}
        </Link>
        <span className="text-surface-600">/</span>
        <span className="text-surface-300">Run Inference</span>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div
            className="flex h-12 w-12 items-center justify-center rounded-2xl text-lg font-bold shadow-lg"
            style={{ backgroundColor: `${catMeta.color}12`, color: catMeta.color, boxShadow: `0 8px 30px -8px ${catMeta.color}15` }}
          >
            <Play className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-surface-50">Run {project.name}</h1>
            <div className="mt-1 flex items-center gap-2">
              <Badge color={catMeta.color} variant="glass">{project.category}</Badge>
              {project.modelFamily.map((m) => <Badge key={m} variant="glass">{m}</Badge>)}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {project.hasTraining && (
            <Link
              to={`/projects/${project.key}/train`}
              className="rounded-lg border border-surface-700 px-3 py-2 text-xs font-medium text-surface-400 hover:bg-surface-800 hover:text-surface-200 transition-colors"
            >
              Train →
            </Link>
          )}
        </div>
      </div>

      {/* Backend warning */}
      {backendAvailable === false && (
        <div className="flex items-center gap-3 rounded-2xl border border-amber-500/20 bg-amber-500/[0.06] px-5 py-4">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400" />
          <div className="text-sm">
            <p className="font-medium text-amber-300">Backend API is offline</p>
            <p className="text-amber-400/60">
              Start the API server: <code className="rounded-lg bg-surface-900/60 px-1.5 py-0.5 text-xs">python dashboard/api/server.py</code>
            </p>
          </div>
        </div>
      )}

      {/* Main content: two columns */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: Input */}
        <div className="space-y-4">
          <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
            <h3 className="mb-4 text-sm font-semibold text-surface-200">Input</h3>

            {/* Input modes */}
            <div className="mb-4 flex flex-wrap gap-2">
              {project.inputModes.map((mode) => {
                const Icon = INPUT_MODE_ICONS[mode] || INPUT_MODE_FALLBACK_ICON;
                return (
                  <div key={mode} className="flex items-center gap-1.5 rounded-xl bg-surface-800/40 px-3 py-1.5 text-xs text-surface-300">
                    <Icon className="h-3.5 w-3.5 text-surface-400" />
                    {mode}
                  </div>
                );
              })}
            </div>

            <FileUpload
              accept="image/*"
              onFiles={handleFiles}
              label="Upload an image"
              hint="Drag & drop or click to browse • JPG, PNG, WebP"
            />
          </div>

          {/* Mode selection + Run button */}
          <div className="flex items-center gap-3">
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as 'visualize' | 'predict')}
              className="rounded-lg border border-surface-700 bg-surface-900 px-3 py-2 text-xs text-surface-300 outline-none focus:border-primary-500"
            >
              <option value="visualize">Visualize (annotated image + results)</option>
              <option value="predict">Predict only (JSON results)</option>
            </select>
            <Button
              onClick={handleRun}
              disabled={!selectedFile || backendAvailable === false}
              loading={isRunning}
              icon={isRunning ? undefined : <Zap className="h-4 w-4" />}
              className="ml-auto"
            >
              {isRunning ? 'Running...' : 'Run Inference'}
            </Button>
          </div>

          {/* Logs */}
          <LogPanel
            lines={logs}
            title="Execution Log"
            onClear={() => setLogs([])}
          />
        </div>

        {/* Right: Output */}
        <div className="space-y-4">
          <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-surface-200">Output</h3>
              {status === 'done' && (
                <span className="flex items-center gap-1 text-xs text-emerald-400">
                  <CheckCircle2 className="h-3.5 w-3.5" /> Success
                </span>
              )}
              {status === 'error' && (
                <span className="flex items-center gap-1 text-xs text-red-400">
                  <XCircle className="h-3.5 w-3.5" /> Error
                </span>
              )}
            </div>

            {isRunning && (
              <div className="flex items-center justify-center py-16">
                <div className="text-center">
                  <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary-400" />
                  <p className="mt-3 text-sm text-surface-400">
                    {status === 'uploading' ? 'Uploading...' : 'Processing...'}
                  </p>
                </div>
              </div>
            )}

            {resultError && !isRunning && (
              <div className="rounded-lg bg-red-500/10 p-4 text-sm text-red-300">
                {resultError}
              </div>
            )}

            {!isRunning && !resultError && (
              <OutputViewer image={resultImage} data={resultData as Record<string, unknown> | null} />
            )}
          </div>

          {/* Quick start code */}
          <div className="rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5">
            <h3 className="mb-3 text-sm font-semibold text-surface-200">CLI Alternative</h3>
            <pre className="overflow-x-auto rounded-xl bg-surface-950/80 p-3 text-[11px] leading-relaxed text-surface-400">
              <code>{`from core import discover_projects, run\ndiscover_projects()\n\n# Prediction only\nresult = run("${project.key}", "image.jpg")\n\n# With visualization\nvisualized = run("${project.key}", "image.jpg", visualize=True)`}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
