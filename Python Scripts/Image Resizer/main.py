"""Image Resizer — CLI tool.

Resize images by percentage, fixed dimensions, or max width/height.
Supports batch processing of a directory.
Requires Pillow: pip install Pillow

Usage:
    python main.py image.jpg --width 800
    python main.py images/ --percent 50
    python main.py image.jpg --max-side 1024 --output resized.jpg
"""

import argparse
import sys
from pathlib import Path


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


def resize_image(src: Path, dst: Path, width: int | None = None,
                 height: int | None = None, percent: float | None = None,
                 max_side: int | None = None, quality: int = 90):
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    img = Image.open(src)
    ow, oh = img.size

    if percent is not None:
        nw = int(ow * percent / 100)
        nh = int(oh * percent / 100)
    elif max_side is not None:
        ratio = min(max_side / ow, max_side / oh)
        nw = int(ow * ratio)
        nh = int(oh * ratio)
    elif width and height:
        nw, nh = width, height
    elif width:
        nw = width
        nh = int(oh * (width / ow))
    elif height:
        nh = height
        nw = int(ow * (height / oh))
    else:
        nw, nh = ow, oh

    resized = img.resize((nw, nh), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        if resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

    resized.save(dst, **save_kwargs)
    return ow, oh, nw, nh


def main():
    parser = argparse.ArgumentParser(description="Image Resizer")
    parser.add_argument("input",            help="Image file or directory")
    parser.add_argument("--width",  "-W",   type=int, default=None)
    parser.add_argument("--height", "-H",   type=int, default=None)
    parser.add_argument("--percent","-p",   type=float, default=None)
    parser.add_argument("--max-side","-m",  type=int, default=None)
    parser.add_argument("--output", "-o",   default=None)
    parser.add_argument("--quality","-q",   type=int, default=90)
    parser.add_argument("--suffix",         default="_resized")
    args = parser.parse_args()

    src = Path(args.input)

    if not any([args.width, args.height, args.percent, args.max_side]):
        # Interactive mode
        print("Image Resizer")
        print("─────────────────────────────")
        src_str  = input("Image file or folder: ").strip()
        src      = Path(src_str)
        mode     = input("Mode: [p]ercent / [w]idth / [h]eight / [m]ax-side [p]: ").strip().lower() or "p"
        if mode == "p":
            args.percent = float(input("Percent (e.g. 50): ").strip() or "50")
        elif mode == "w":
            args.width = int(input("Width (px): ").strip())
        elif mode == "h":
            args.height = int(input("Height (px): ").strip())
        elif mode == "m":
            args.max_side = int(input("Max side (px): ").strip())
        args.quality = int(input("JPEG quality [90]: ").strip() or "90")
        args.suffix  = "_resized"

    if src.is_dir():
        images = [p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED]
        if not images:
            print("  No supported images found.")
            return
        out_dir = Path(args.output) if args.output else src / "resized"
        out_dir.mkdir(exist_ok=True)
        print(f"  Resizing {len(images)} image(s) → {out_dir}")
        for img_path in images:
            dst = out_dir / img_path.name
            ow, oh, nw, nh = resize_image(img_path, dst,
                args.width, args.height, args.percent, args.max_side, args.quality)
            print(f"  {img_path.name}: {ow}×{oh} → {nw}×{nh}")
        print("  Done.")
    elif src.is_file():
        if args.output:
            dst = Path(args.output)
        else:
            dst = src.with_stem(src.stem + args.suffix)
        ow, oh, nw, nh = resize_image(src, dst,
            args.width, args.height, args.percent, args.max_side, args.quality)
        print(f"  {src.name}: {ow}×{oh} → {nw}×{nh}  saved to {dst}")
    else:
        print(f"  Not found: {src}")
        sys.exit(1)


if __name__ == "__main__":
    main()
