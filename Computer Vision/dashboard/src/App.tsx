import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useState, useEffect, lazy, Suspense } from 'react';
import { Sidebar } from './components/Sidebar';
import { CommandPalette } from './components/CommandPalette';

const DashboardPage = lazy(() => import('./pages/DashboardPage').then(m => ({ default: m.DashboardPage })));
const ProjectsPage = lazy(() => import('./pages/ProjectsPage').then(m => ({ default: m.ProjectsPage })));
const ProjectDetailPage = lazy(() => import('./pages/ProjectDetailPage').then(m => ({ default: m.ProjectDetailPage })));
const RunProjectPage = lazy(() => import('./pages/RunProjectPage').then(m => ({ default: m.RunProjectPage })));
const TrainProjectPage = lazy(() => import('./pages/TrainProjectPage').then(m => ({ default: m.TrainProjectPage })));
const DatasetsPage = lazy(() => import('./pages/DatasetsPage').then(m => ({ default: m.DatasetsPage })));
const SettingsPage = lazy(() => import('./pages/SettingsPage').then(m => ({ default: m.SettingsPage })));
const HelpPage = lazy(() => import('./pages/HelpPage').then(m => ({ default: m.HelpPage })));

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [cmdOpen, setCmdOpen] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); setCmdOpen(true); }
      if (e.key === 'Escape') setCmdOpen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-surface-950">
        <Sidebar collapsed={sidebarCollapsed} onToggleCollapse={() => setSidebarCollapsed((c) => !c)} onOpenCommand={() => setCmdOpen(true)} />
        <main className={`flex-1 transition-all duration-300 ease-out ${sidebarCollapsed ? 'ml-[72px]' : 'ml-[260px]'}`}>
          <div className="mx-auto max-w-[1400px] px-6 py-6 lg:px-10 lg:py-8">
            <Suspense fallback={<div className="h-64 rounded-3xl bg-surface-800/20 shimmer" />}>
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/projects" element={<ProjectsPage />} />
                <Route path="/projects/:key" element={<ProjectDetailPage />} />
                <Route path="/projects/:key/run" element={<RunProjectPage />} />
                <Route path="/projects/:key/train" element={<TrainProjectPage />} />
                <Route path="/run" element={<RunProjectPage />} />
                <Route path="/train" element={<TrainProjectPage />} />
                <Route path="/datasets" element={<DatasetsPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="/help" element={<HelpPage />} />
              </Routes>
            </Suspense>
          </div>
        </main>
        <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} />
      </div>
    </BrowserRouter>
  );
}
