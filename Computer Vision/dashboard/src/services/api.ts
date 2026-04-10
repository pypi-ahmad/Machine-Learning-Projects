import type {
  ApiHealth,
  RegistryStatus,
  SystemInfo,
  PredictionResult,
  VisualizationResult,
  ProjectManifest,
  Project,
  TrainStatus,
  DatasetStatus,
} from '../types';

const API_BASE = '/api';

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      msg = body.detail || body.message || msg;
    } catch { /* ignore parse errors */ }
    throw new ApiError(res.status, msg);
  }
  return res.json();
}

// ── Health & System ────────────────────────────────────────

export async function getHealth(): Promise<ApiHealth> {
  return request<ApiHealth>('/health');
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return request<SystemInfo>('/system');
}

export async function getRegistryStatus(): Promise<RegistryStatus> {
  return request<RegistryStatus>('/registry');
}

// ── Projects ───────────────────────────────────────────────

export async function getManifest(): Promise<ProjectManifest> {
  return request<ProjectManifest>('/projects');
}

export async function getProject(key: string): Promise<Project> {
  return request<Project>(`/projects/${encodeURIComponent(key)}`);
}

// ── Inference ──────────────────────────────────────────────

export async function runInference(key: string, file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', file);
  return request<PredictionResult>(`/run/${encodeURIComponent(key)}`, {
    method: 'POST',
    body: formData,
  });
}

export async function runVisualize(key: string, file: File): Promise<VisualizationResult> {
  const formData = new FormData();
  formData.append('file', file);
  return request<VisualizationResult>(`/visualize/${encodeURIComponent(key)}`, {
    method: 'POST',
    body: formData,
  });
}

// ── Training ───────────────────────────────────────────────

export async function startTraining(
  key: string,
  config: Record<string, unknown>,
): Promise<{ status: string; message: string; cli_command: string }> {
  return request<{ status: string; message: string; cli_command: string }>(`/train/${encodeURIComponent(key)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
}

export async function getTrainStatus(key: string): Promise<TrainStatus> {
  return request<TrainStatus>(`/train/${encodeURIComponent(key)}/status`);
}

// ── Datasets ───────────────────────────────────────────────

export async function getDatasetStatus(key: string): Promise<DatasetStatus> {
  return request<DatasetStatus>(`/datasets/${encodeURIComponent(key)}`);
}

export async function downloadDataset(key: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/datasets/${encodeURIComponent(key)}/download`, {
    method: 'POST',
  });
}

// ── Connectivity Check ─────────────────────────────────────

export async function checkBackendAvailable(): Promise<boolean> {
  try {
    await getHealth();
    return true;
  } catch {
    return false;
  }
}

export { ApiError };
