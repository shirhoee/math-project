import React, { useMemo, useState } from 'react';
import katex from 'katex';
import {
  createRotationMatrix,
  createIntrinsics,
  computeHomography,
  matrixToLatex,
  determinant,
  orthogonalityError,
} from '../utils/math';
import { CheckCircle2, Cpu, Activity } from 'lucide-react';

interface MathMatrixInspectorProps {
  W?: number;
  H?: number;
  liveYaw?: number;
  livePitch?: number;
  isHovering?: boolean;
}

export const MathMatrixInspector: React.FC<MathMatrixInspectorProps> = ({
  W = 640,
  H = 480,
  liveYaw,
  livePitch,
  isHovering = false,
}) => {
  const [manualYaw, setManualYaw] = useState<number>(0);
  const [manualPitch, setManualPitch] = useState<number>(0);
  const [depth, setDepth] = useState<number>(2.0);

  // If live mouse tracking is active, use live angles; otherwise use manual/idle angles
  const yaw = isHovering && liveYaw !== undefined ? liveYaw : (liveYaw ?? manualYaw);
  const pitch = isHovering && livePitch !== undefined ? livePitch : (livePitch ?? manualPitch);

  const fx = W * 0.9;
  const fy = W * 0.9;
  const cx = W / 2;
  const cy = H / 2;

  const K = useMemo(() => createIntrinsics(fx, fy, cx, cy), [fx, fy, cx, cy]);
  const R = useMemo(() => createRotationMatrix(yaw, pitch, 0), [yaw, pitch]);
  const det = useMemo(() => determinant(R), [R]);
  const orthErr = useMemo(() => orthogonalityError(R), [R]);

  const homography = useMemo(() => {
    const t: [number, number, number] = [0.04 * Math.sin((yaw * Math.PI) / 180), 0.02 * Math.sin((pitch * Math.PI) / 180), 0];
    const n: [number, number, number] = [0, 0, 1];
    return computeHomography(K, R, t, n, depth);
  }, [K, R, yaw, pitch, depth]);

  const renderKatex = (latex: string) => {
    try {
      return (
        <span
          dangerouslySetInnerHTML={{
            __html: katex.renderToString(latex, { throwOnError: false, displayMode: true }),
          }}
        />
      );
    } catch {
      return <span className="font-mono text-xs">{latex}</span>;
    }
  };

  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl text-slate-200 flex flex-col gap-4">
      {/* Header with live tracking indicator */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-3 gap-2">
        <div className="flex items-center gap-2">
          <Cpu className="text-indigo-400" size={17} />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-100">
            Camera Math & Homography Inspector
          </h3>
        </div>

        <div className="flex items-center gap-2">
          {isHovering ? (
            <div className="flex items-center gap-1.5 text-[11px] text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 px-2.5 py-0.5 rounded-full font-mono">
              <Activity size={12} className="animate-spin" />
              <span>Live Sync: θ={yaw >= 0 ? '+' : ''}{yaw.toFixed(1)}°, ϕ={pitch >= 0 ? '+' : ''}{pitch.toFixed(1)}°</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-[11px] text-slate-400 bg-slate-800/60 px-2.5 py-0.5 rounded-full font-mono">
              <span>Resting (θ=0°, ϕ=0°)</span>
            </div>
          )}

          <div className="hidden sm:flex items-center gap-1 text-[11px] text-indigo-400 bg-indigo-950/40 border border-indigo-800/50 px-2 py-0.5 rounded-full">
            <CheckCircle2 size={12} />
            <span>det(R)={det.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* Manual Sliders for Testing / Idle Exploration */}
      {!isHovering && (
        <div className="grid grid-cols-3 gap-2.5 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-xs">
          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Yaw (θ)</span>
              <span className="font-mono text-indigo-400">{manualYaw}°</span>
            </div>
            <input
              type="range"
              min="-12"
              max="12"
              step="0.5"
              value={manualYaw}
              onChange={(e) => setManualYaw(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Pitch (ϕ)</span>
              <span className="font-mono text-indigo-400">{manualPitch}°</span>
            </div>
            <input
              type="range"
              min="-8"
              max="8"
              step="0.5"
              value={manualPitch}
              onChange={(e) => setManualPitch(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Layer Depth (z_k)</span>
              <span className="font-mono text-indigo-400">{depth.toFixed(1)}m</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="5.0"
              step="0.1"
              value={depth}
              onChange={(e) => setDepth(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>
        </div>
      )}

      {/* Matrix Cards Stack */}
      <div className="flex flex-col gap-3">
        {/* Rotation Matrix R */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              3D Rotation Matrix R(θ, ϕ) ∈ SO(3)
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              Lie error: {orthErr.toExponential(1)}
            </span>
          </div>
          <div className="overflow-x-auto text-[11px] py-1 text-indigo-200">
            {renderKatex(`R = ${matrixToLatex(R, 3)}`)}
          </div>
        </div>

        {/* Camera Intrinsics Matrix K */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Camera Pinhole Intrinsics K
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              fx={fx.toFixed(0)}, cx={cx.toFixed(0)}
            </span>
          </div>
          <div className="overflow-x-auto text-[11px] py-1 text-indigo-200">
            {renderKatex(`K = ${matrixToLatex(K, 0)}`)}
          </div>
        </div>

        {/* Plane Induced Homography H_k */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400">
              Planar Homography H_k
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              H_k = K (R + \frac{'{'}t n^T{'}'}{'{'}z_k{'}'}) K^{'-1'}
            </span>
          </div>
          <div className="overflow-x-auto text-[11px] py-1 text-emerald-300">
            {renderKatex(`H_k = ${matrixToLatex(homography, 3)}`)}
          </div>
          <p className="text-[10px] text-slate-500 mt-1">
            Warps slice k at depth {depth.toFixed(1)}m from canonical frame to novel view.
          </p>
        </div>
      </div>
    </div>
  );
};
