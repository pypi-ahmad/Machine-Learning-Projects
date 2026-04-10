import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, FolderOpen, Database, Settings, HelpCircle,
  Eye, ChevronLeft, ChevronRight, Play, GraduationCap, Search,
} from 'lucide-react';
import { BackendBadge } from './ui/BackendBadge';

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard', end: true },
  { to: '/projects', icon: FolderOpen, label: 'Projects', end: false },
  { to: '/run', icon: Play, label: 'Inference', end: false },
  { to: '/train', icon: GraduationCap, label: 'Training', end: false },
  { to: '/datasets', icon: Database, label: 'Datasets', end: false },
];

const BOTTOM_NAV = [
  { to: '/settings', icon: Settings, label: 'Settings', end: false },
  { to: '/help', icon: HelpCircle, label: 'Help', end: false },
];

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  onOpenCommand: () => void;
}

export function Sidebar({ collapsed, onToggleCollapse, onOpenCommand }: SidebarProps) {
  return (
    <aside
      className={`fixed inset-y-0 left-0 z-30 flex flex-col border-r border-surface-800/60 bg-surface-950/80 backdrop-blur-xl transition-all duration-300 ease-out ${
        collapsed ? 'w-[72px]' : 'w-[260px]'
      }`}
    >
      {/* Logo */}
      <div className="flex h-[68px] items-center gap-3 border-b border-surface-800/40 px-5">
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg shadow-primary-600/20">
          <Eye className="h-4 w-4 text-white" />
          <div className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full border-2 border-surface-950 bg-accent-400 animate-pulse-dot" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <h1 className="truncate text-sm font-bold tracking-tight text-surface-50">
              CV <span className="text-primary-400">Projects</span>
            </h1>
            <p className="text-[10px] font-medium text-surface-500">AI Platform</p>
          </div>
        )}
      </div>

      {/* Quick search trigger */}
      {!collapsed && (
        <div className="px-4 pt-4 pb-2">
          <button
            onClick={onOpenCommand}
            className="flex w-full items-center gap-2.5 rounded-xl border border-surface-800/60 bg-surface-900/50 px-3 py-2.5 text-xs text-surface-500 transition-all hover:border-surface-700 hover:bg-surface-800/50 hover:text-surface-400"
          >
            <Search className="h-3.5 w-3.5" />
            <span className="flex-1 text-left">Search...</span>
            <kbd className="rounded border border-surface-700/60 bg-surface-800/60 px-1.5 py-0.5 text-[9px] font-medium text-surface-500">⌘K</kbd>
          </button>
        </div>
      )}

      {/* Main nav */}
      <nav className="flex-1 space-y-0.5 px-3 py-3">
        <div className={`${!collapsed ? 'mb-1 px-3 text-[10px] font-semibold uppercase tracking-widest text-surface-600' : 'sr-only'}`}>
          Main
        </div>
        {NAV_ITEMS.map((item) => (
          <SidebarLink key={item.to} item={item} collapsed={collapsed} />
        ))}

        <div className={`${!collapsed ? 'mb-1 mt-5 px-3 text-[10px] font-semibold uppercase tracking-widest text-surface-600' : 'mt-4 sr-only'}`}>
          System
        </div>
        {BOTTOM_NAV.map((item) => (
          <SidebarLink key={item.to} item={item} collapsed={collapsed} />
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-surface-800/40 p-3">
        {!collapsed && (
          <div className="mb-3 flex justify-center">
            <BackendBadge />
          </div>
        )}
        <button
          onClick={onToggleCollapse}
          className="flex h-9 w-full items-center justify-center rounded-xl text-surface-500 transition-all hover:bg-surface-800/40 hover:text-surface-300"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>
    </aside>
  );
}

function SidebarLink({
  item,
  collapsed,
}: {
  item: { to: string; icon: typeof Play; label: string; end: boolean };
  collapsed: boolean;
}) {
  return (
    <NavLink
      to={item.to}
      end={item.end}
      className={({ isActive }) =>
        `group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13px] font-medium transition-all duration-200 ${
          isActive
            ? 'bg-primary-600/12 text-primary-400'
            : 'text-surface-400 hover:bg-surface-800/40 hover:text-surface-200'
        } ${collapsed ? 'justify-center' : ''}`
      }
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <div className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-primary-500" />
          )}
          <item.icon className={`h-[18px] w-[18px] shrink-0 transition-colors ${isActive ? 'text-primary-400' : 'text-surface-500 group-hover:text-surface-300'}`} />
          {!collapsed && <span>{item.label}</span>}
        </>
      )}
    </NavLink>
  );
}
