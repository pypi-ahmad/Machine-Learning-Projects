import { useCallback, useState, useRef } from 'react';
import { Upload, X, Image as ImageIcon, FileVideo, File } from 'lucide-react';

interface FileUploadProps {
  accept?: string;
  multiple?: boolean;
  onFiles: (files: File[]) => void;
  maxSizeMb?: number;
  label?: string;
  hint?: string;
  className?: string;
  preview?: boolean;
}

export function FileUpload({
  accept = 'image/*',
  multiple = false,
  onFiles,
  maxSizeMb = 50,
  label = 'Upload file',
  hint,
  className = '',
  preview = true,
}: FileUploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0) return;
      const files = Array.from(fileList);
      const maxBytes = maxSizeMb * 1024 * 1024;
      const valid = files.filter((f) => f.size <= maxBytes);
      if (valid.length === 0) return;

      setFileName(valid[0].name);

      if (preview && valid[0].type.startsWith('image/')) {
        const url = URL.createObjectURL(valid[0]);
        setPreviewUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      } else {
        setPreviewUrl(null);
      }

      onFiles(multiple ? valid : [valid[0]]);
    },
    [maxSizeMb, multiple, onFiles, preview],
  );

  const clear = useCallback(() => {
    setFileName(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = '';
  }, [previewUrl]);

  const FileTypeIcon = accept?.includes('video') ? FileVideo : accept?.includes('image') ? ImageIcon : File;

  return (
    <div className={className}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
        className={`relative cursor-pointer rounded-xl border-2 border-dashed transition-all ${
          dragOver
            ? 'border-primary-500 bg-primary-600/10'
            : 'border-surface-700 bg-surface-900/40 hover:border-surface-500 hover:bg-surface-900/60'
        } ${previewUrl ? 'p-2' : 'p-8'}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={(e) => handleFiles(e.target.files)}
          className="hidden"
        />

        {previewUrl ? (
          <div className="relative">
            <img
              src={previewUrl}
              alt="Preview"
              className="mx-auto max-h-64 rounded-lg object-contain"
            />
            <button
              onClick={(e) => { e.stopPropagation(); clear(); }}
              className="absolute right-1 top-1 rounded-full bg-surface-900/80 p-1 text-surface-400 hover:text-white transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
            <p className="mt-2 truncate text-center text-xs text-surface-500">{fileName}</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-surface-800 text-surface-400">
              {dragOver ? <Upload className="h-6 w-6 text-primary-400" /> : <FileTypeIcon className="h-6 w-6" />}
            </div>
            <div>
              <p className="text-sm font-medium text-surface-300">{label}</p>
              <p className="mt-1 text-xs text-surface-500">
                {hint || `Drag & drop or click to browse • Max ${maxSizeMb}MB`}
              </p>
            </div>
            {fileName && (
              <div className="flex items-center gap-2 rounded-lg bg-surface-800 px-3 py-1.5 text-xs text-surface-300">
                <File className="h-3.5 w-3.5" />
                {fileName}
                <button
                  onClick={(e) => { e.stopPropagation(); clear(); }}
                  className="text-surface-500 hover:text-white"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
