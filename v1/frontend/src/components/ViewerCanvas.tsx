import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Maximize, Sliders, RotateCcw, Compass } from 'lucide-react';
import { decodeFrames } from '../utils/imageDecoder';

interface ViewerCanvasProps {
  framesB64: string[];
  nYaw?: number;
  nPitch?: number;
  width?: number;
  height?: number;
  maxYawDeg?: number;
  maxPitchDeg?: number;
  onAngleChange?: (yawDeg: number, pitchDeg: number, isHovering: boolean) => void;
}

export const ViewerCanvas: React.FC<ViewerCanvasProps> = ({
  framesB64,
  nYaw = 21,
  nPitch = 3,
  width = 640,
  height = 480,
  maxYawDeg = 16.0,
  maxPitchDeg = 5.0,
  onAngleChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const decodedFramesRef = useRef<(ImageBitmap | HTMLImageElement)[]>([]);

  // Sensitivity & UI states
  const [sensitivity, setSensitivity] = useState<number>(1.0);
  const [isDecoding, setIsDecoding] = useState<boolean>(false);
  const [decodeProgress, setDecodeProgress] = useState<number>(0);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [displayAngles, setDisplayAngles] = useState<{ yaw: number; pitch: number; isHovering: boolean }>({
    yaw: 0,
    pitch: 0,
    isHovering: false,
  });

  // Position references for RAF loop
  const posRef = useRef({
    targetX: 0.5,
    targetY: 0.5,
    currentX: 0.5,
    currentY: 0.5,
    isHovering: false,
  });

  // Batch decode frames when framesB64 changes
  useEffect(() => {
    if (!framesB64 || framesB64.length === 0) {
      decodedFramesRef.current = [];
      return;
    }

    let isMounted = true;
    setIsDecoding(true);
    setDecodeProgress(0);

    decodeFrames(framesB64, (loaded, total) => {
      if (isMounted) {
        setDecodeProgress(Math.round((loaded / total) * 100));
      }
    }).then((decoded) => {
      if (isMounted) {
        decodedFramesRef.current = decoded;
        setIsDecoding(false);
      }
    });

    return () => {
      isMounted = false;
    };
  }, [framesB64]);

  // Handle Fullscreen change
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  // Reset to Center
  const resetCenter = useCallback(() => {
    posRef.current.targetX = 0.5;
    posRef.current.targetY = 0.5;
  }, []);

  // Main Render Loop (60-120 FPS via requestAnimationFrame)
  useEffect(() => {
    let animationFrameId: number;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    let lastNotifyTime = 0;

    const render = (time: number) => {
      const state = posRef.current;

      // Smooth cursor lerp: when hovering follow mouse; when not hovering ease back to center
      const lerpSpeed = state.isHovering ? 0.22 : 0.12;
      state.currentX += (state.targetX - state.currentX) * lerpSpeed;
      state.currentY += (state.targetY - state.currentY) * lerpSpeed;

      // Compute current live angles in degrees
      const liveYaw = (state.currentX - 0.5) * 2 * maxYawDeg;
      const livePitch = (state.currentY - 0.5) * 2 * maxPitchDeg;

      // Throttle notifying parent for matrix updates ~30 FPS
      if (time - lastNotifyTime > 30) {
        lastNotifyTime = time;
        if (onAngleChange) {
          onAngleChange(liveYaw, livePitch, state.isHovering);
        }
        setDisplayAngles({
          yaw: liveYaw,
          pitch: livePitch,
          isHovering: state.isHovering,
        });
      }

      const frames = decodedFramesRef.current;
      if (frames.length > 0) {
        const cWidth = canvas.width;
        const cHeight = canvas.height;

        // Clamp between 0 and 1
        const clampedX = Math.max(0, Math.min(1, state.currentX));
        const clampedY = Math.max(0, Math.min(1, state.currentY));

        const yNorm = clampedX * (nYaw - 1);
        const pNorm = clampedY * (nPitch - 1);

        // Crisp frame indexing avoids double-exposure ghosting and layer transparency blur
        const nearestY = Math.max(0, Math.min(nYaw - 1, Math.round(yNorm)));
        const nearestP = Math.max(0, Math.min(nPitch - 1, Math.round(pNorm)));
        const idx = Math.min(nearestP * nYaw + nearestY, frames.length - 1);

        const frame = frames[idx];
        if (frame) {
          ctx.drawImage(frame, 0, 0, cWidth, cHeight);
        }
      }

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [nYaw, nPitch, maxYawDeg, maxPitchDeg, onAngleChange]);

  // Pointer Movement Handlers
  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const rawX = (e.clientX - rect.left) / rect.width;
    const rawY = (e.clientY - rect.top) / rect.height;

    // Apply sensitivity centered around 0.5
    const sensX = 0.5 + (rawX - 0.5) * sensitivity;
    const sensY = 0.5 + (rawY - 0.5) * sensitivity;

    posRef.current.targetX = Math.max(0, Math.min(1, sensX));
    posRef.current.targetY = Math.max(0, Math.min(1, sensY));
    posRef.current.isHovering = true;
  };

  const handlePointerEnter = () => {
    posRef.current.isHovering = true;
  };

  const handlePointerLeave = () => {
    posRef.current.isHovering = false;
    resetCenter();
  };

  return (
    <div
      ref={containerRef}
      className={`relative flex flex-col items-center bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl transition-all ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none' : 'w-full mx-auto'
      }`}
    >
      {/* Canvas Viewport with Maximized Field of View */}
      <div
        className="relative w-full bg-black flex items-center justify-center overflow-hidden"
        style={{
          aspectRatio: width && height ? `${width} / ${height}` : '4 / 3',
          maxHeight: isFullscreen ? '100vh' : '70vh',
        }}
      >
        {isDecoding && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/75 backdrop-blur-sm">
            <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-3"></div>
            <p className="text-sm font-medium text-slate-300">
              Decoding 3D Atlas ({decodeProgress}%)
            </p>
          </div>
        )}

        {framesB64.length === 0 ? (
          <div className="text-center p-8 text-slate-500 flex flex-col items-center gap-2">
            <Compass size={32} className="text-slate-600 mb-1" />
            <p className="text-sm font-medium text-slate-400">Interactive 3D Viewport</p>
            <p className="text-xs text-slate-600">Select an image & click Synthesize to render novel views</p>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            onPointerMove={handlePointerMove}
            onPointerEnter={handlePointerEnter}
            onPointerLeave={handlePointerLeave}
            className="w-full h-full object-contain cursor-crosshair touch-none select-none"
          />
        )}

        {/* Live Tracking Status Badge */}
        {framesB64.length > 0 && (
          <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-black/65 backdrop-blur-md text-[11px] font-mono text-slate-300 border border-white/10 pointer-events-none flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                displayAngles.isHovering ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'
              }`}
            />
            <span>
              {displayAngles.isHovering ? 'Mouse Active' : 'Centered'}: Yaw {displayAngles.yaw >= 0 ? '+' : ''}
              {displayAngles.yaw.toFixed(1)}° | Pitch {displayAngles.pitch >= 0 ? '+' : ''}
              {displayAngles.pitch.toFixed(1)}°
            </span>
          </div>
        )}
      </div>

      {/* Clean Control Bar */}
      {framesB64.length > 0 && (
        <div className="w-full px-4 py-2.5 bg-slate-900/90 backdrop-blur border-t border-slate-800 flex items-center justify-between gap-4 text-xs text-slate-300">
          <div className="flex items-center gap-2">
            <button
              onClick={resetCenter}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
              title="Recenter Camera"
            >
              <RotateCcw size={13} />
              <span>Center</span>
            </button>
            <span className="text-[11px] text-slate-500 hidden sm:inline">
              Move mouse over image to rotate perspective
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Sliders size={13} className="text-slate-400" />
              <span className="text-[11px] text-slate-400">Sens:</span>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={sensitivity}
                onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                className="w-16 accent-indigo-500 h-1.5 bg-slate-700 rounded-lg cursor-pointer"
              />
              <span className="font-mono text-[11px] text-slate-400">{sensitivity.toFixed(1)}x</span>
            </div>

            <button
              onClick={toggleFullscreen}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
              title="Fullscreen"
            >
              <Maximize size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
