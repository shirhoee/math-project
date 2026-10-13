import React from 'react';
import { Loader2 } from 'lucide-react';

interface ProgressBarProps {
  progress: number;
  status: string | null;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ progress, status }) => {
  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs font-medium text-slate-300">
        <div className="flex items-center gap-2">
          <Loader2 size={14} className="text-indigo-400 animate-spin" />
          <span>{status || 'Processing job...'}</span>
        </div>
        <span className="font-mono text-indigo-400">{progress}%</span>
      </div>

      <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden p-0.5">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${Math.max(4, Math.min(100, progress))}%` }}
        />
      </div>
    </div>
  );
};
