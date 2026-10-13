import React, { useEffect, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { fetchSamples, sampleToFile } from '../api/client';
import type { SampleItem } from '../api/types';

interface PresetGalleryProps {
  onSelectSample: (file: File) => void;
  disabled?: boolean;
}

export const PresetGallery: React.FC<PresetGalleryProps> = ({
  onSelectSample,
  disabled = false,
}) => {
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [loadingSampleId, setLoadingSampleId] = useState<string | null>(null);

  useEffect(() => {
    fetchSamples().then((items) => {
      setSamples(items);
    });
  }, []);

  const handleSelect = async (sample: SampleItem) => {
    if (disabled || loadingSampleId) return;
    try {
      setLoadingSampleId(sample.id);
      const file = await sampleToFile(sample);
      onSelectSample(file);
    } catch (err) {
      console.error('Failed to load sample:', err);
    } finally {
      setLoadingSampleId(null);
    }
  };

  if (samples.length === 0) {
    return null;
  }

  return (
    <div className="w-full">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
        <Sparkles size={13} className="text-amber-400" />
        <span>Sample Presets</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
        {samples.slice(0, 5).map((sample) => (
          <button
            key={sample.id}
            onClick={() => handleSelect(sample)}
            disabled={disabled || !!loadingSampleId}
            className="group relative flex flex-col items-center bg-slate-900 border border-slate-800 hover:border-indigo-500 rounded-lg p-1.5 transition-all text-left overflow-hidden disabled:opacity-50"
          >
            <div className="w-full h-16 rounded overflow-hidden bg-slate-800 flex items-center justify-center">
              <img
                src={sample.url}
                alt={sample.name}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                loading="lazy"
              />
            </div>
            <span className="text-[11px] font-medium text-slate-300 mt-1 truncate w-full text-center">
              {loadingSampleId === sample.id ? 'Loading...' : sample.name}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
};
