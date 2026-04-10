import { type ReactNode } from 'react';

interface BadgeProps {
  children: ReactNode;
  color?: string;
  variant?: 'solid' | 'outline' | 'subtle' | 'glass';
  size?: 'sm' | 'md';
  dot?: boolean;
  className?: string;
}

export function Badge({ children, color, variant = 'subtle', size = 'sm', dot = false, className = '' }: BadgeProps) {
  const sizeClass = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';
  const base = `inline-flex items-center gap-1.5 rounded-full font-medium whitespace-nowrap ${sizeClass}`;

  if (variant === 'solid' && color) {
    return (
      <span className={`${base} text-white ${className}`} style={{ backgroundColor: color }}>
        {dot && <span className="h-1.5 w-1.5 rounded-full bg-white/70" />}
        {children}
      </span>
    );
  }
  if (variant === 'outline' && color) {
    return (
      <span className={`${base} border ${className}`} style={{ borderColor: `${color}40`, color }}>
        {dot && <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />}
        {children}
      </span>
    );
  }
  if (variant === 'glass') {
    return (
      <span
        className={`${base} border border-white/[0.06] ${className}`}
        style={{
          backgroundColor: color ? `${color}10` : 'rgba(255,255,255,0.05)',
          color: color || '#94a3b8',
        }}
      >
        {dot && <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color || '#94a3b8' }} />}
        {children}
      </span>
    );
  }
  return (
    <span className={`${base} bg-surface-800/60 text-surface-300 ${className}`}>
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-surface-500" />}
      {children}
    </span>
  );
}
