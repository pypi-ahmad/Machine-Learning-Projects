import { type ReactNode } from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  className?: string;
  accentColor?: string;
  change?: string;
  subtitle?: string;
}

export function StatCard({ label, value, icon, className = '', accentColor, change, subtitle }: StatCardProps) {
  return (
    <div className={`group relative overflow-hidden rounded-2xl border border-surface-800/50 bg-surface-900/40 p-5 transition-all duration-300 hover:border-surface-700/60 hover:shadow-lg hover:shadow-surface-950/50 ${className}`}>
      {/* Subtle gradient glow */}
      <div
        className="absolute -right-8 -top-8 h-24 w-24 rounded-full opacity-[0.07] blur-2xl transition-opacity duration-300 group-hover:opacity-[0.12]"
        style={{ backgroundColor: accentColor || '#6366f1' }}
      />

      <div className="relative flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-xs font-medium uppercase tracking-wider text-surface-500">{label}</p>
          <p className="text-2xl font-bold tracking-tight text-surface-50">{value}</p>
          {subtitle && <p className="text-[11px] text-surface-500">{subtitle}</p>}
          {change && (
            <p className={`text-[11px] font-medium ${change.startsWith('+') ? 'text-accent-400' : 'text-surface-500'}`}>
              {change}
            </p>
          )}
        </div>
        {icon && (
          <div
            className="flex h-11 w-11 items-center justify-center rounded-xl transition-transform duration-300 group-hover:scale-110"
            style={{ backgroundColor: accentColor ? `${accentColor}12` : undefined, color: accentColor }}
          >
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
