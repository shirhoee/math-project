import React, { useRef, useState } from 'react';
import { UploadCloud, FolderOpen, X, RefreshCw } from 'lucide-react';

interface UploadZoneProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  onClear: () => void;
  disabled?: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  selectedFile,
  previewUrl,
  onFileSelect,
  onClear,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        onFileSelect(file);
      }
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        onFileSelect(file);
      }
    }
    // Reset so selecting the same file again works
    e.target.value = '';
  };

  const validateFile = (file: File): boolean => {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid image file (JPEG, PNG, WebP).');
      return false;
    }
    return true;
  };

  return (
    <div className="w-full flex flex-col gap-3">
      {/* Hidden standard file input */}
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/jpg"
        onChange={handleInputChange}
        disabled={disabled}
        className="hidden"
      />

      {/* Prominent Separate Button to Select Image from PC */}
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Source Image
        </span>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="flex items-center gap-2 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium shadow-md shadow-indigo-950/40 transition-all cursor-pointer"
          title="Browse your computer to upload a custom image"
        >
          <FolderOpen size={14} />
          <span>Upload Image from PC</span>
        </button>
      </div>

      {previewUrl ? (
        <div className="relative group rounded-xl overflow-hidden border border-slate-700 bg-slate-900/50 shadow-md">
          <img
            src={previewUrl}
            alt="Upload Preview"
            className="w-full h-52 object-cover rounded-xl"
          />
          <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={disabled}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition-colors shadow cursor-pointer"
            >
              <RefreshCw size={13} />
              <span>Choose Another</span>
            </button>
            <button
              type="button"
              onClick={onClear}
              disabled={disabled}
              className="flex items-center gap-1.5 p-1.5 px-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-medium transition-colors shadow cursor-pointer"
              title="Remove image"
            >
              <X size={15} />
              <span>Clear</span>
            </button>
          </div>
          <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded bg-black/70 text-[11px] font-mono text-slate-200 border border-white/10 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span className="truncate max-w-[200px]">{selectedFile?.name || 'Custom Image'}</span>
          </div>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && inputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl transition-all cursor-pointer ${
            isDragging
              ? 'border-indigo-500 bg-indigo-950/30 scale-[0.99]'
              : 'border-slate-700 hover:border-indigo-500/80 bg-slate-900/40 hover:bg-slate-900/70'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <div className="p-3 bg-indigo-600/10 text-indigo-400 rounded-full mb-3">
            <UploadCloud size={28} />
          </div>
          <p className="text-sm font-semibold text-slate-200">
            Click here or drag & drop an image
          </p>
          <p className="text-xs text-slate-400 mt-1">
            JPEG, PNG, or WebP from your computer
          </p>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
            disabled={disabled}
            className="mt-3 flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium border border-slate-700 hover:border-slate-600 transition-colors"
          >
            <FolderOpen size={13} className="text-indigo-400" />
            <span>Browse Files</span>
          </button>
        </div>
      )}
    </div>
  );
};
