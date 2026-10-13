import { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { UploadZone } from './components/UploadZone';
import { PresetGallery } from './components/PresetGallery';
import { ControlsDrawer } from './components/ControlsDrawer';
import { ViewerCanvas } from './components/ViewerCanvas';
import { ProgressBar } from './components/ProgressBar';
import { MathMatrixInspector } from './components/MathMatrixInspector';
import { LayerSlicesViewer } from './components/LayerSlicesViewer';
import { submitRenderJob, fetchJobStatus, fetchJobResult, fetchSamples } from './api/client';
import type { RenderParams, ResultResponse } from './api/types';
import { AlertCircle, Layers, ChevronDown, ChevronUp } from 'lucide-react';

export function App() {
  // Backend status
  const [serverStatus, setServerStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');

  // Input state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Parameters (Default to Subtle 0.04 motion for minimum distortion and maximum stability)
  const [params, setParams] = useState<Omit<RenderParams, 'file'>>({
    renderer: 'Layers (MPI)',
    motionPreset: 'Subtle (0.04)',
    zNear: 1.0,
    zFar: 4.0,
    nLayers: 16,
    sharpenAmount: 0.35,
  });

  // Job & Result state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<number>(0);
  const [jobStatusText, setJobStatusText] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<ResultResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Live angle streaming from mouse hover over 3D canvas
  const [liveAngles, setLiveAngles] = useState<{
    yaw: number;
    pitch: number;
    isHovering: boolean;
  }>({
    yaw: 0,
    pitch: 0,
    isHovering: false,
  });

  // Optional collapsible drawer for MPI layer slices
  const [showLayerSlices, setShowLayerSlices] = useState<boolean>(false);

  const pollIntervalRef = useRef<number | null>(null);

  // Check backend connectivity on mount
  useEffect(() => {
    fetchSamples()
      .then(() => setServerStatus('connected'))
      .catch(() => setServerStatus('disconnected'));
  }, []);

  // Handle file select
  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setErrorMessage(null);
  };

  // Clear selected file
  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
  };

  // Submit Job
  const handleSubmit = async () => {
    if (!selectedFile) {
      setErrorMessage('Please select or upload an image first.');
      return;
    }

    try {
      setIsSubmitting(true);
      setErrorMessage(null);
      setJobProgress(5);
      setJobStatusText('Initializing rendering pipeline...');

      const jobResp = await submitRenderJob({
        ...params,
        file: selectedFile,
      });

      setActiveJobId(jobResp.job_id);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to submit render job.');
      setIsSubmitting(false);
    }
  };

  // Polling loop
  useEffect(() => {
    if (!activeJobId) return;

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const statusResp = await fetchJobStatus(activeJobId);
        setJobProgress(statusResp.progress);
        if (statusResp.status) {
          setJobStatusText(statusResp.status);
        }

        if (statusResp.result) {
          // Completed with attached result
          setResult(statusResp.result);
          setIsSubmitting(false);
          setActiveJobId(null);
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        } else if (statusResp.progress === 100) {
          // Progress reached 100%, fetch result endpoint
          const res = await fetchJobResult(activeJobId);
          setResult(res);
          setIsSubmitting(false);
          setActiveJobId(null);
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 450);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [activeJobId]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Header serverStatus={serverStatus} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 flex flex-col gap-6">
        {/* Error Alert */}
        {errorMessage && (
          <div className="flex items-center gap-2 p-3.5 bg-rose-950/60 border border-rose-800 rounded-xl text-rose-200 text-xs">
            <AlertCircle size={16} className="text-rose-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Top Control Bar: Upload, Presets & Synthesize */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-lg flex flex-col gap-4">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
            {/* Left Upload / Dropzone */}
            <div className="lg:col-span-5">
              <UploadZone
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                onFileSelect={handleFileSelect}
                onClear={handleClear}
                disabled={isSubmitting}
              />
            </div>

            {/* Center Preset Samples */}
            <div className="lg:col-span-4">
              <PresetGallery
                onSelectSample={handleFileSelect}
                disabled={isSubmitting}
              />
            </div>

            {/* Right Synthesize & Parameters Action */}
            <div className="lg:col-span-3">
              <ControlsDrawer
                params={params}
                onChange={(upd) => setParams((p) => ({ ...p, ...upd }))}
                onSubmit={handleSubmit}
                isSubmitting={isSubmitting}
                disabled={!selectedFile || isSubmitting}
              />
            </div>
          </div>

          {/* Progress Bar when processing */}
          {isSubmitting && (
            <div className="pt-2 border-t border-slate-800">
              <ProgressBar progress={jobProgress} status={jobStatusText} />
            </div>
          )}
        </div>

        {/* Side-by-Side Main Window: 3D Viewport & Real-Time Camera Math */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left / Center Viewport (7 columns) */}
          <div className="lg:col-span-7 flex flex-col gap-3">
            <ViewerCanvas
              framesB64={result?.frames_b64 || []}
              nYaw={result?.n_yaw ?? 21}
              nPitch={result?.n_pitch ?? 3}
              width={result?.W ?? 640}
              height={result?.H ?? 480}
              maxYawDeg={result?.max_yaw ?? 16.0}
              maxPitchDeg={result?.max_pitch ?? 5.0}
              onAngleChange={(yaw, pitch, isHovering) =>
                setLiveAngles({ yaw, pitch, isHovering })
              }
            />

            {/* Viewport Meta Footer */}
            {result && (
              <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-2 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-slate-500">Method:</span>
                  <span className="font-medium text-slate-200">{result.renderer}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-500">Grid:</span>
                  <span className="font-mono text-indigo-400">
                    {result.n_yaw ?? 21}×{result.n_pitch ?? 3} ({result.frames_b64?.length ?? 0} views)
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-500">Resolution:</span>
                  <span className="font-mono text-slate-300">
                    {result.W}×{result.H}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Live Camera Math & Homographies (5 columns) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <MathMatrixInspector
              W={result?.W ?? 640}
              H={result?.H ?? 480}
              liveYaw={liveAngles.yaw}
              livePitch={liveAngles.pitch}
              isHovering={liveAngles.isHovering}
            />
          </div>
        </div>

        {/* Collapsible Layer Slices Viewer (Optional Inspection) */}
        {result && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-3">
            <button
              onClick={() => setShowLayerSlices((prev) => !prev)}
              className="flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-white transition-colors cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <Layers size={15} className="text-indigo-400" />
                <span>Multi-Plane Image (MPI) Layer Decomposition</span>
                <span className="text-slate-500 text-[11px] font-mono">
                  ({result.n_layers || 16} Depth Planes)
                </span>
              </div>
              {showLayerSlices ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
            </button>

            {showLayerSlices && (
              <div className="pt-3 border-t border-slate-800">
                <LayerSlicesViewer result={result} />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
