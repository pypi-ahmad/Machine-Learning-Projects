"""Fix broken PaddleOCR sample image URLs in OCR pipeline files."""
import os
from pathlib import Path

OLD_BLOCK = '''SAMPLE_URLS = [
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img_12.jpg",
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img623.jpg",
]


def download_samples():
    save_dir = Path(SAVE_DIR) / "ocr_samples"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            urllib.request.urlretrieve(url, str(fname))
        paths.append(fname)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        paths.extend([p for p in Path(SAVE_DIR).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available for OCR")
    return paths'''

NEW_BLOCK = '''SAMPLE_URLS = [
    # WikiMedia Commons public domain OCR test images (reliable)
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Atomist_quote_from_Democritus.png/320px-Atomist_quote_from_Democritus.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Handwriting_of_Shivaji.jpg/320px-Handwriting_of_Shivaji.jpg",
]


def _generate_synthetic_image(save_dir, idx=0):
    """Create synthetic OCR test image using PIL when downloads fail."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        lines = [
            "Hello World OCR Test",
            f"Sample Image {idx + 1}",
            "Python Machine Learning",
            "Image Text Extraction",
        ]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill=(0, 0, 0))
            y += 40
        out = save_dir / f"synthetic_{idx}.png"
        img.save(str(out))
        return out
    except Exception:
        return None


def download_samples():
    save_dir = Path(SAVE_DIR) / "ocr_samples"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            try:
                urllib.request.urlretrieve(url, str(fname))
            except Exception as e:
                print(f"  Warning: could not download {url}: {e}")
                continue
        if fname.exists():
            paths.append(fname)
    if not paths:
        print("  No sample images downloaded; generating synthetic test images...")
        for i in range(3):
            p = _generate_synthetic_image(save_dir, i)
            if p:
                paths.append(p)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        paths.extend([p for p in Path(SAVE_DIR).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available for OCR")
    return paths'''

files = [
    "Computer Vision/Captcha Recognition/pipeline.py",
    "Computer Vision/Document Word Detection/pipeline.py",
    "Computer Vision/Image to Text Conversion - OCR/pipeline.py",
    "Computer Vision/QR Code Readability/pipeline.py",
]

root = Path(__file__).parent
fixed = 0
for rel in files:
    f = root / rel
    if not f.exists():
        print(f"NOT FOUND: {rel}")
        continue
    content = f.read_text("utf-8")
    if OLD_BLOCK in content:
        content = content.replace(OLD_BLOCK, NEW_BLOCK)
        f.write_text(content, "utf-8")
        print(f"Fixed: {rel}")
        fixed += 1
    else:
        print(f"Pattern not found (may already be fixed): {rel}")

print(f"\nTotal fixed: {fixed}")
