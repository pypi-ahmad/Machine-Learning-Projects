"""Download a YouTube video to the current working directory with Tkinter."""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from pytubefix import YouTube


def download_video(url: str, output_directory: Path) -> str:
    """Download the highest-resolution progressive stream and return its title."""
    video = YouTube(url)
    stream = video.streams.get_highest_resolution()
    if stream is None:
        raise RuntimeError("No progressive video stream is available for this URL.")

    stream.download(output_path=str(output_directory))
    return video.title


def main() -> None:
    """Create and run the downloader window."""
    root = tk.Tk()
    root.geometry("700x300")
    root.resizable(False, False)
    root.title("YouTube Video Downloader")

    tk.Label(
        root,
        text="Paste the YouTube video link you want to download",
        font="arial 15 bold",
    ).pack(pady=(20, 0))

    link = tk.StringVar()
    status = tk.StringVar(value="Downloads save to the current working directory.")

    tk.Label(root, text="Paste link here:", font="arial 15 bold").place(x=270, y=75)
    tk.Entry(root, width=80, textvariable=link).place(x=32, y=110)

    def download() -> None:
        url = link.get().strip()
        if not url:
            messagebox.showerror("Missing URL", "Paste a YouTube video URL before downloading.")
            return

        status.set("Downloading...")
        root.update_idletasks()
        try:
            title = download_video(url, Path.cwd())
        except Exception as error:
            status.set("Download failed.")
            messagebox.showerror("Download failed", str(error))
            return

        status.set(f"Downloaded: {title}")

    tk.Button(
        root,
        text="DOWNLOAD",
        font="arial 15 bold",
        bg="white",
        padx=2,
        command=download,
    ).place(x=280, y=165)
    tk.Label(root, textvariable=status, font="arial 11").place(x=32, y=240)

    root.mainloop()


if __name__ == "__main__":
    main()
