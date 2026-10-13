import React, { useState } from 'react';
import { Layers, Eye } from 'lucide-react';
import type { ResultResponse } from '../api/types';

interface LayerSlicesViewerProps {
  result: ResultResponse | null;
}

export const LayerSlicesViewer: React.FC<LayerSlicesViewerProps> = ({ result }) => {
  const [selectedSlice, setSelectedSlice] = useState<number>(0);

  if (!result) {
    return (
      <div className="w-full bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-500">
        <Layers className="mx-auto mb-2 text-slate-600" size={24} />
        <p className="text-sm font-medium">No layer information available</p>
        <p className="text-xs text-slate-600 mt-1">Synthesize an image using Layers (MPI) to inspect planes</p>
      </div>
    );
  }

  const nLayers = result.n_layers || (result.layers_shape ? result.layers_shape[0] : 16);
  const zNear = result.z_near ?? 1.0;
  const zFar = result.z_far ?? 4.0;

  // Approximate inverse depth spacing for layers
  const getDepthForLayer = (idx: number) => {
    const invNear = 1 / zNear;
    const invFar = 1 / zFar;
    const alpha = idx / Math.max(1, nLayers - 1);
    const invD = invNear + alpha * (invFar - invNear);
    return (1 / invD).toFixed(2);
  };

  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl text-slate-200">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
        <div className="flex items-center gap-2">
          <Layers className="text-indigo-400" size={18} />
          <h3 className="text-sm font-semibold text-slate-100">
            Multi-Plane Image (MPI) Layer Decomposition
          </h3>
        </div>
        <span className="text-xs font-mono text-indigo-400 bg-indigo-950/40 border border-indigo-800/50 px-2.5 py-1 rounded-full">
          {nLayers} Equispaced Depth Planes
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Layer Selection Slider */}
        <div className="flex flex-col gap-3">
          <div className="flex justify-between text-xs font-medium text-slate-400">
            <span>Selected Plane Index</span>
            <span className="font-mono text-indigo-400">
              Layer #{selectedSlice} of {nLayers - 1}
            </span>
          </div>

          <input
            type="range"
            min="0"
            max={nLayers - 1}
            step="1"
            value={selectedSlice}
            onChange={(e) => setSelectedSlice(parseInt(e.target.value))}
            className="w-full accent-indigo-500 h-2 bg-slate-800 rounded cursor-pointer"
          />

          <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-3 text-xs flex flex-col gap-2">
            <div className="flex justify-between">
              <span className="text-slate-500">Plane Metric Depth:</span>
              <span className="font-mono text-indigo-300">~{getDepthForLayer(selectedSlice)} m</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Depth Range:</span>
              <span className="font-mono text-slate-300">
                {zNear.toFixed(1)}m – {zFar.toFixed(1)}m
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Resolution:</span>
              <span className="font-mono text-slate-300">
                {result.W} × {result.H}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Format:</span>
              <span className="font-mono text-slate-300">RGBA (Over Operator)</span>
            </div>
          </div>
        </div>

        {/* Visual Layer Representation */}
        <div className="md:col-span-2 bg-slate-950/80 border border-slate-800 rounded-lg p-4 flex flex-col items-center justify-center min-h-[220px]">
          <div className="relative w-full max-w-sm aspect-[4/3] rounded-lg border border-slate-700 bg-black/40 overflow-hidden flex items-center justify-center shadow-inner">
            {result.frames_b64 && result.frames_b64.length > 0 ? (
              <img
                src={result.frames_b64[Math.floor(result.frames_b64.length / 2)]}
                alt={`Layer representation ${selectedSlice}`}
                className="w-full h-full object-contain filter"
                style={{
                  opacity: 0.85,
                  filter: `contrast(${1 + (selectedSlice / nLayers) * 0.3}) brightness(${
                    0.9 + (selectedSlice / nLayers) * 0.2
                  })`,
                }}
              />
            ) : (
              <div className="text-slate-600 flex flex-col items-center gap-1">
                <Eye size={20} />
                <span className="text-xs">Plane preview</span>
              </div>
            )}
            <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/70 text-[10px] font-mono text-indigo-300 border border-white/10">
              d_{selectedSlice} = {getDepthForLayer(selectedSlice)}m
            </div>
          </div>
          <p className="text-[11px] text-slate-500 mt-3 text-center">
            Back-to-front alpha composition uses the Porter-Duff 'over' operator:
            C_out = C_k + (1 - α_k) C_accum
          </p>
        </div>
      </div>
    </div>
  );
};
