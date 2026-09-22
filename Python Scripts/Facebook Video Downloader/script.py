"""Download public Facebook videos through a small Tkinter interface."""

import argparse
from pathlib import Path
import queue
import re
from threading import Thread
import tkinter as tk
from tkinter import ttk
from urllib.parse import unquote, urlparse, urlunparse

import requests

REQUEST_TIMEOUT = 15
OUTPUT_PATH = Path(__file__).with_name('video.mp4')


def get_download_link(url: str) -> str:
    """Return a direct video URL from a public Facebook page URL."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or '').lower()
    if parsed.scheme not in {'http', 'https'} or (
        hostname != 'facebook.com' and not hostname.endswith('.facebook.com')
    ):
        raise ValueError('Enter a valid Facebook video URL.')

    mobile_url = urlunparse(parsed._replace(netloc='mbasic.facebook.com'))
    response = requests.get(mobile_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    match = re.search(r'[?&]src=([^"&]+)', response.text)
    if match is None:
        raise ValueError('No downloadable video was found at this URL.')

    return unquote(match.group(1))


class VideoDownload(Thread):
    """Download a video and send progress updates to the Tkinter thread."""

    def __init__(self, url: str, updates: queue.Queue[tuple[str, object]]) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.updates = updates

    def run(self) -> None:
        try:
            response = requests.get(self.url, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with OUTPUT_PATH.open('wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        self.updates.put(('progress', downloaded * 100 / total_size))

            self.updates.put(('complete', OUTPUT_PATH))
        except (OSError, requests.RequestException) as error:
            self.updates.put(('error', str(error)))


class DownloaderApp:
    """Tkinter controls and state for one download at a time."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.updates: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: VideoDownload | None = None

        root.title('Facebook Video Downloader')
        root.geometry('400x300')

        tk.Label(root, text='Enter Facebook Video URL:').pack()
        self.url_value = tk.StringVar()
        tk.Entry(root, textvariable=self.url_value, font=('Calibri', 9)).place(
            x=25, y=50, width=350
        )
        self.download_button = tk.Button(root, text='Download', command=self.start_download)
        self.download_button.place(x=100, y=100, width=200)
        self.progress = ttk.Progressbar(root, length=350, mode='determinate')
        self.progress.place(y=200, width=350, x=25)
        self.status = tk.Label(
            root, text='Ready', fg='blue', font=('Calibri', 9), bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def start_download(self) -> None:
        try:
            link = get_download_link(self.url_value.get().strip())
        except (requests.RequestException, ValueError) as error:
            self.set_status(str(error), 'red')
            return

        self.progress['value'] = 0
        self.download_button['state'] = tk.DISABLED
        self.set_status('Downloading', 'green')
        self.worker = VideoDownload(link, self.updates)
        self.worker.start()
        self.monitor_download()

    def monitor_download(self) -> None:
        try:
            event, value = self.updates.get_nowait()
        except queue.Empty:
            event = None
        else:
            if event == 'progress':
                self.progress['value'] = value
            elif event == 'complete':
                self.progress['value'] = 100
                self.set_status(f'Finished: {value}', 'green')
                self.download_button['state'] = tk.NORMAL
                self.worker = None
            elif event == 'error':
                self.set_status(f'Download failed: {value}', 'red')
                self.download_button['state'] = tk.NORMAL
                self.worker = None

        if self.worker is not None:
            self.root.after(50, self.monitor_download)

    def set_status(self, message: str, color: str) -> None:
        self.status['text'] = message
        self.status['fg'] = color


def main() -> None:
    parser = argparse.ArgumentParser(description='Open the Facebook video downloader GUI.')
    parser.parse_args()

    root = tk.Tk()
    DownloaderApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
