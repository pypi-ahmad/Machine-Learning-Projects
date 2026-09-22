"""Convert one PNG image to a Windows ICO file."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, UnidentifiedImageError


ICON_SIZES = ((16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256))


def convert_png(source: Path, destination: Path) -> None:
    """Write a multi-resolution ICO file without overwriting an existing file."""
    if source.suffix.lower() != ".png" or not source.is_file():
        raise ValueError("Input must be an existing PNG file")
    if destination.suffix.lower() != ".ico":
        raise ValueError("Output file must use the .ico extension")
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose another output path")
    with Image.open(source) as image:
        image.save(destination, format="ICO", sizes=ICON_SIZES)


def launch_gui() -> None:
    """Launch a minimal file-picker interface for one conversion."""
    root = tk.Tk()
    root.title("PNG to ICO Converter")
    root.geometry("420x180")
    selected: Path | None = None
    status = tk.StringVar(value="Choose a PNG file.")

    def choose_source() -> None:
        nonlocal selected
        chosen = filedialog.askopenfilename(filetypes=[("PNG image", "*.png")])
        if chosen:
            selected = Path(chosen)
            status.set(selected.name)

    def save_icon() -> None:
        if selected is None:
            messagebox.showerror("No input file", "Choose a PNG file first.", parent=root)
            return
        chosen = filedialog.asksaveasfilename(defaultextension=".ico", filetypes=[("Icon file", "*.ico")])
        if not chosen:
            return
        try:
            convert_png(selected, Path(chosen))
        except (FileExistsError, UnidentifiedImageError, OSError, ValueError) as error:
            messagebox.showerror("Conversion failed", str(error), parent=root)
            return
        messagebox.showinfo("Conversion complete", f"Saved icon to {chosen}", parent=root)

    frame = tk.Frame(root, padx=18, pady=18)
    frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(frame, textvariable=status).pack(pady=(0, 12))
    tk.Button(frame, text="Choose PNG", command=choose_source).pack(side=tk.LEFT, padx=(70, 8))
    tk.Button(frame, text="Save ICO", command=save_icon).pack(side=tk.LEFT)
    root.mainloop()


def main() -> None:
    """Parse CLI arguments or launch the file-picker interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="existing PNG input")
    parser.add_argument("destination", nargs="?", type=Path, help="new ICO output")
    parser.add_argument("--gui", action="store_true", help="open the file-picker interface")
    args = parser.parse_args()
    if args.gui:
        if args.source or args.destination:
            parser.error("--gui does not accept source or destination paths")
        launch_gui()
        return
    if args.source is None or args.destination is None:
        parser.error("source and destination are required unless --gui is used")
    try:
        convert_png(args.source, args.destination)
    except (FileExistsError, UnidentifiedImageError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved icon to {args.destination}")


if __name__ == "__main__":
    main()
