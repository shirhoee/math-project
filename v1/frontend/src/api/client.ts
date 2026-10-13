import type { JobResponse, RenderParams, ResultResponse, SampleItem, StatusResponse } from './types';

const API_BASE = '/api/v1';

export async function submitRenderJob(params: RenderParams): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', params.file);
  formData.append('renderer', params.renderer);
  formData.append('motion_preset_str', params.motionPreset);
  if (params.zNear !== undefined) formData.append('z_near', params.zNear.toString());
  if (params.zFar !== undefined) formData.append('z_far', params.zFar.toString());
  if (params.nLayers !== undefined) formData.append('n_layers', params.nLayers.toString());
  if (params.sharpenAmount !== undefined) formData.append('sharpen_amount', params.sharpenAmount.toString());

  const response = await fetch(`${API_BASE}/render`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to submit job: ${response.status} - ${errorText}`);
  }

  return response.json();
}

export async function fetchJobStatus(jobId: string): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchJobResult(jobId: string): Promise<ResultResponse> {
  const response = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/result`);
  if (!response.ok) {
    throw new Error(`Failed to fetch result: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchSamples(): Promise<SampleItem[]> {
  try {
    const response = await fetch(`${API_BASE}/samples`);
    if (!response.ok) {
      return [];
    }
    return response.json();
  } catch (err) {
    console.warn('Failed to load samples:', err);
    return [];
  }
}

export async function sampleToFile(sample: SampleItem): Promise<File> {
  const response = await fetch(sample.url);
  const blob = await response.blob();
  return new File([blob], sample.filename, { type: blob.type || 'image/jpeg' });
}
