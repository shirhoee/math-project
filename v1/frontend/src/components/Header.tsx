import React from 'react';
import { Box, Sparkles, Activity } from 'lucide-react';

interface HeaderProps {
  serverStatus: 'connected' | 'connecting' | 'disconnected';
}

export const Header: React.FC<HeaderProps> = ({ serverStatus }) => {
  return (
    <header className="w-full border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40 px-6 py-3.5 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-500/20">
          <Box size={22} />
        </div>
        <div>
          <h1 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            3D View Synthesis Studio
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-900/50 text-indigo-300 border border-indigo-700/50 font-normal">
              v1.0
            </span>
          </h1>
          <p className="text-xs text-slate-400">
            Multi-Plane Image (MPI) & Point Splatting Novel Perspective Engine
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
          <Activity
            size={14}
            className={
              serverStatus === 'connected'
                ? 'text-emerald-400 animate-pulse'
                : serverStatus === 'connecting'
                ? 'text-amber-400 animate-spin'
                : 'text-rose-400'
            }
          />
          <span className="text-slate-300 capitalize font-medium">
            API: {serverStatus}
          </span>
        </div>

        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-950/40 border border-indigo-800/40 text-indigo-300 text-xs font-medium">
          <Sparkles size={13} className="text-indigo-400" />
          <span>60 FPS Parallax</span>
        </div>
      </div>
    </header>
  );
};
