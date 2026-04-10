import { type ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  gradient?: boolean;
  hover?: boolean;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

export function GlassCard({
  children,
  className = '',
  gradient = false,
  hover = false,
  padding = 'md',
}: GlassCardProps) {
  const pad = { none: '', sm: 'p-4', md: 'p-5', lg: 'p-6' };

  return (
    <div
      className={`
        relative overflow-hidden rounded-2xl border border-surface-800/50
        bg-surface-900/40 backdrop-blur-sm
        ${gradient ? 'gradient-border' : ''}
        ${hover ? 'transition-all duration-300 hover:border-surface-700/60 hover:bg-surface-900/60 hover:shadow-lg hover:shadow-surface-950/50' : ''}
        ${pad[padding]}
        ${className}
      `}
    >
      {children}
    </div>
  );
}
