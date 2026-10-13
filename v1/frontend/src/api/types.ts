export interface JobResponse {
  job_id: string;
}

export interface StatusResponse {
  status: string | null;
  progress: number;
  result?: ResultResponse | null;
}

export interface ResultResponse {
  renderer: string;
  H: number;
  W: number;
  frames_b64: string[];
  n_yaw?: number;
  n_pitch?: number;
  max_yaw?: number;
  max_pitch?: number;
  z_pivot?: number;
  baseline_x?: number;
  baseline_y?: number;
  z_near?: number;
  z_far?: number;
  n_layers?: number;
  cache_key?: string;
  settings_str?: string;
  layers_shape?: number[];
  [key: string]: unknown;
}

export interface SampleItem {
  id: string;
  name: string;
  filename: string;
  url: string;
}

export interface RenderParams {
  file: File;
  renderer: 'Layers (MPI)' | 'Point Splatting';
  motionPreset: 'Subtle (0.04)' | 'Normal (0.07)' | 'Dramatic (0.11)';
  zNear?: number;
  zFar?: number;
  nLayers?: number;
  sharpenAmount?: number;
}
