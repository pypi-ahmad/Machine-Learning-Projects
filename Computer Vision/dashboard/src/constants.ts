import { Image as ImageIcon, Video, Camera, Folder, FileType, Upload } from 'lucide-react';

export const INPUT_MODE_ICONS: Record<string, typeof Video> = {
  image: ImageIcon,
  video: Video,
  webcam: Camera,
  folder: Folder,
  pdf: FileType,
};

export const INPUT_MODE_FALLBACK_ICON = Upload;
