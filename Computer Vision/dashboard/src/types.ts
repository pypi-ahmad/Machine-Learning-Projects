// ── Project & Manifest Types ───────────────────────────────

export interface ProjectDataset {
  configured: boolean;
  ready: boolean;
  type: string;
  id: string;
}

export interface Project {
  key: string;
  aliases: string[];
  name: string;
  className: string;
  category: string;
  projectType: string;
  description: string;
  legacyTech: string;
  modernTech: string;
  tags: string[];
  modelFamily: string[];
  inputModes: string[];
  hasTraining: boolean;
  hasInference: boolean;
  hasConfig: boolean;
  hasReadme: boolean;
  folderPath: string;
  sourcePath: string;
  dataset: ProjectDataset;
}

export interface ManifestStats {
  totalProjects: number;
  trainable: number;
  withInference: number;
  withDataset: number;
  dataReady: number;
  categories: Record<string, number>;
}

export interface ProjectManifest {
  version: string;
  generatedAt: string;
  repoVersion: string;
  stats: ManifestStats;
  projects: Project[];
}

// ── Filter & Sort Types ────────────────────────────────────

export type SortField = 'name' | 'category' | 'projectType';
export type SortDir = 'asc' | 'desc';

export interface Filters {
  search: string;
  categories: string[];
  tags: string[];
  modelFamilies: string[];
  inputModes: string[];
  hasTraining: boolean | null;
  hasInference: boolean | null;
  datasetReady: boolean | null;
}

export const EMPTY_FILTERS: Filters = {
  search: '',
  categories: [],
  tags: [],
  modelFamilies: [],
  inputModes: [],
  hasTraining: null,
  hasInference: null,
  datasetReady: null,
};

// ── API Response Types ─────────────────────────────────────

export interface ApiHealth {
  status: string;
  timestamp: number;
}

export interface RegistryStatus {
  registered: string[];
  count: number;
}

export interface PredictionResult {
  [key: string]: unknown;
}

export interface VisualizationResult {
  image: string;
  prediction: PredictionResult;
}

export interface SystemInfo {
  gpu_available: boolean;
  gpu_name: string;
  gpu_memory_gb: number;
  cuda_version: string;
  python_version: string;
  torch_version: string;
  projects_discovered: number;
}

export interface TrainConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  image_size: number;
  device: string;
}

export interface TrainStatus {
  project: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  epoch: number;
  total_epochs: number;
  metrics: Record<string, number>;
  log: string[];
}

export interface DatasetStatus {
  project: string;
  configured: boolean;
  ready: boolean;
  type: string;
  id: string;
  size_mb: number | null;
  path: string;
}

export type RunStatus = 'idle' | 'uploading' | 'processing' | 'done' | 'error';

// ── UI Metadata ────────────────────────────────────────────

export interface CategoryMeta {
  label: string;
  color: string;
  icon: string;
}

export const CATEGORY_META: Record<string, CategoryMeta> = {
  'Detection':          { label: 'Detection',          color: '#3b82f6', icon: 'Crosshair' },
  'Classification':     { label: 'Classification',     color: '#8b5cf6', icon: 'Tags' },
  'Segmentation':       { label: 'Segmentation',       color: '#06b6d4', icon: 'Layers' },
  'Tracking':           { label: 'Tracking',           color: '#f59e0b', icon: 'Route' },
  'Pose & Landmarks':   { label: 'Pose & Landmarks',   color: '#10b981', icon: 'Accessibility' },
  'OCR & Document AI':  { label: 'OCR & Document AI',  color: '#ec4899', icon: 'FileText' },
  'Retrieval & Search':  { label: 'Retrieval & Search',  color: '#f97316', icon: 'Search' },
  'Anomaly Detection':   { label: 'Anomaly Detection',   color: '#ef4444', icon: 'ShieldAlert' },
  'OpenCV Utilities':    { label: 'OpenCV Utilities',    color: '#6b7280', icon: 'Wrench' },
  'Other':               { label: 'Other',               color: '#a855f7', icon: 'Box' },
};

export const PROJECT_TYPE_LABELS: Record<string, string> = {
  detection: 'Object Detection',
  classification: 'Image Classification',
  segmentation: 'Image Segmentation',
  pose: 'Pose Estimation',
  ocr: 'OCR / Text Recognition',
  tracking: 'Object Tracking',
  retrieval: 'Image Retrieval',
  anomaly: 'Anomaly Detection',
  opencv: 'OpenCV Utility',
  other: 'Other',
};

// ── Skeleton / animation constants ─────────────────────────

export const SKELETON_STAGGER_MS = 70;
export const MAX_COMMAND_PALETTE_RESULTS = 8;
