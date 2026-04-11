"""Fix OCR pipeline files to use synthetic PIL images instead of broken URLs."""
from pathlib import Path

# Templates with broken URLs (matches all variants)
OLD_BLOCKS = [
    # Image Text Extraction variant  
    '''SAMPLE_URLS = [
    # PaddleOCR repo sample images (main branch)
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/doc/imgs_en/img_12.jpg",
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/doc/imgs_en/img623.jpg",
    # WikiMedia Commons public domain fallbacks
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Atomist_quote_from_Democritus.png/320px-Atomist_quote_from_Democritus.png",
]


def _generate_synthetic_image(save_dir, idx=0):
    """Create a synthetic OCR test image using PIL when network downloads fail."""
    try:
        from PIL import Image, ImageDraw, ImageFont
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
    # If no downloads succeeded, generate synthetic images
    if not paths:
        print("  No sample images downloaded; generating synthetic test images...")
        for i in range(3):
            p = _generate_synthetic_image(save_dir, i)
            if p:
                paths.append(p)
    # Also pick up any local images in the project folder
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        paths.extend([p for p in Path(SAVE_DIR).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available for OCR")
    return paths''',

    # Other OCR pipelines variant
    '''SAMPLE_URLS = [
    # WikiMedia Commons public domain OCR test images (reliable)
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Atomist_quote_from_Democritus.png/320px-Atomist_quote_from_Democritus.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Handwriting_of_Shivaji.jpg/320px-Handwriting_of_Shivaji.jpg",
]


def _generate_synthetic_image(save_dir, idx=0):
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        lines = ["Hello World OCR Test", f"Sample Image {idx + 1}",
                 "Python Machine Learning", "Image Text Extraction"]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill=(0, 0, 0)); y += 40
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
    return paths''',
]

NEW_BLOCK = '''def _generate_synthetic_image(save_dir, idx=0):
    """Create a synthetic OCR test image using PIL."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        lines = [
            f"OCR Test Image {idx + 1}",
            "Hello World Machine Learning",
            "Python Image Text Extraction",
            "Production Grade OCR Pipeline",
            "Sample Document Processing",
        ]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill=(0, 0, 0))
            y += 50
        out = save_dir / f"synthetic_{idx}.png"
        img.save(str(out))
        return out
    except Exception as e:
        print(f"  Warning: could not generate synthetic image: {e}")
        return None


def download_samples():
    """Generate synthetic OCR test images (no network download needed)."""
    save_dir = Path(SAVE_DIR) / "ocr_samples"
    save_dir.mkdir(exist_ok=True)
    paths = []
    # Generate synthetic test images always available
    for i in range(5):
        p = save_dir / f"synthetic_{i}.png"
        if not p.exists():
            p = _generate_synthetic_image(save_dir, i)
        if p and p.exists():
            paths.append(p)
    # Also pick up any local images in the project folder
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        for found in Path(SAVE_DIR).rglob(f"*{ext}"):
            if found not in paths:
                paths.append(found)
    print(f"{len(paths)} images available for OCR")
    return paths'''

root = Path(__file__).parent
files = sorted(root.rglob("pipeline.py"))
files = [f for f in files if "venv" not in str(f)]

fixed = 0
for f in files:
    try:
        content = f.read_text("utf-8", errors="replace")
        changed = False
        for old_block in OLD_BLOCKS:
            if old_block in content:
                content = content.replace(old_block, NEW_BLOCK)
                changed = True
                break
        if changed:
            f.write_text(content, "utf-8")
            print(f"Fixed: {f.relative_to(root)}")
            fixed += 1
    except Exception as e:
        print(f"ERROR {f}: {e}")

print(f"\nTotal fixed: {fixed}")
