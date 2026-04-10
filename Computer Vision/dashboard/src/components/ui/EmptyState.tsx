import { type ReactNode } from 'react';

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-3xl border border-dashed border-surface-800/50 bg-surface-900/20 px-8 py-20 text-center animate-fade-in">
      {icon && <div className="mb-5 text-surface-600">{icon}</div>}
      <h3 className="text-base font-semibold text-surface-300">{title}</h3>
      {description && <p className="mt-2 max-w-sm text-sm text-surface-500">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
