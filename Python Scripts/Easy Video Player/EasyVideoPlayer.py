"""Find and play a video file with OpenCV and ffpyplayer."""

import argparse
import os
from pathlib import Path

import cv2
from ffpyplayer.player import MediaPlayer


def find_video(file_name: str, directory_name: str) -> Path:
    """Return the first matching video file below a directory."""
    for path, subdirs, files in os.walk(directory_name):
        for name in files:
            if file_name == name:
                return Path(path, name)

    raise FileNotFoundError(f'No file named {file_name!r} was found in {directory_name!r}.')


def play_video(video_path: Path) -> None:
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')

    player = MediaPlayer(str(video_path))

    while True:
        grabbed, frame = video.read()
        if not grabbed:
            print('End of video')
            break

        player.get_frame()
        cv2.imshow('Video', frame)
        if cv2.waitKey(28) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description='Find and play a video file.')
    parser.add_argument('filename', nargs='?', help='Video filename to find')
    parser.add_argument('directory', nargs='?', help='Directory to search')
    args = parser.parse_args()

    filename = args.filename or input('Name of the video file that you want to play: ')
    directory = args.directory or input('Directory that may contain the video: ')
    play_video(find_video(filename, directory))


if __name__ == '__main__':
    main()
