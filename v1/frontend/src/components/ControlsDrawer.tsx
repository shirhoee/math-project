import React, { useState } from 'react';
import { Play, Settings2, ChevronDown, ChevronUp } from 'lucide-react';
import type { RenderParams } from '../api/types';

interface ControlsDrawerProps {
  params: Omit<RenderParams, 'file'>;
  onChange: (updated: Partial<Omit<RenderParams, 'file'>>) => void;
  onSubmit: () => void;
  isSubmitting: boolean;
  disabled: boolean;
}

export const ControlsDrawer: React.FC<ControlsDrawerProps> = ({
  params,
  onChange,
  onSubmit,
  isSubmitting,
  disabled,
}) => {
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl flex flex-col gap-3">
      {/* Primary Action Button */}
      <button
        onClick={onSubmit}
        disabled={disabled || isSubmitting}
        className="w-full flex items-center justify-center gap-2 py-3 px-5 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-semibold text-sm shadow-lg shadow-indigo-950/40 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer active:scale-[0.99]"
      >
        {isSubmitting ? (
          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
        ) : (
          <Play size={15} fill="currentColor" />
        )}
        <span>{isSubmitting ? 'Synthesizing 3D Novel Views...' : '⚡ Synthesize 3D Novel Views'}</span>
      </button>

      {/* Optional Advanced Settings Toggle */}
      <button
        onClick={() => setShowAdvanced((prev) => !prev)}
        type="button"
        className="flex items-center justify-between text-xs text-slate-400 hover:text-slate-200 transition-colors py-1 px-1 cursor-pointer"
      >
        <div className="flex items-center gap-1.5">
          <Settings2 size={13} className="text-indigo-400" />
          <span>Advanced Tuning</span>
          <span className="text-[11px] text-slate-500 font-mono">
            ({params.motionPreset}, {params.nLayers || 16}L)
          </span>
        </div>
        {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {/* Collapsible Advanced Parameters */}
      {showAdvanced && (
        <div className="pt-2 border-t border-slate-800 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          {/* Motion Baseline Preset */}
          <div>
            <label className="block text-slate-400 font-medium mb-1 text-[11px]">Motion Range</label>
            <select
              value={params.motionPreset}
              onChange={(e) =>
                onChange({
                  motionPreset: e.target.value as
                    | 'Subtle (0.04)'
                    | 'Normal (0.07)'
                    | 'Dramatic (0.11)',
                })
              }
              disabled={disabled}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500 text-xs"
            >
              <option value="Subtle (0.04)">Subtle (0.04 - Minimum Distortion)</option>
              <option value="Normal (0.07)">Normal (0.07 - Standard)</option>
              <option value="Dramatic (0.11)">Dramatic (0.11 - Wide Parallax)</option>
            </select>
          </div>

          {/* Layer Count */}
          <div>
            <div className="flex justify-between text-slate-400 font-medium mb-1 text-[11px]">
              <span>Layers</span>
              <span className="font-mono text-indigo-400">{params.nLayers || 16} planes</span>
            </div>
            <select
              value={params.nLayers || 16}
              onChange={(e) => onChange({ nLayers: parseInt(e.target.value) })}
              disabled={disabled}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500 text-xs"
            >
              <option value={8}>8 Layers (Fastest)</option>
              <option value={16}>16 Layers (Balanced)</option>
              <option value={24}>24 Layers (Detailed)</option>
            </select>
          </div>

          {/* Near Bound */}
          <div>
            <div className="flex justify-between text-slate-400 font-medium mb-1 text-[11px]">
              <span>Near Bound (Z_near)</span>
              <span className="font-mono text-indigo-400">{(params.zNear || 1.0).toFixed(1)}m</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.5"
              step="0.1"
              value={params.zNear || 1.0}
              onChange={(e) => onChange({ zNear: parseFloat(e.target.value) })}
              disabled={disabled}
              className="w-full accent-indigo-500 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>

          {/* Far Bound */}
          <div>
            <div className="flex justify-between text-slate-400 font-medium mb-1 text-[11px]">
              <span>Far Bound (Z_far)</span>
              <span className="font-mono text-indigo-400">{(params.zFar || 4.0).toFixed(1)}m</span>
            </div>
            <input
              type="range"
              min="2.0"
              max="6.0"
              step="0.2"
              value={params.zFar || 4.0}
              onChange={(e) => onChange({ zFar: parseFloat(e.target.value) })}
              disabled={disabled}
              className="w-full accent-indigo-500 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>
        </div>
      )}
    </div>
  );
};
