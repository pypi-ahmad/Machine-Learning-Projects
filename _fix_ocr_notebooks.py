"""Fix broken PaddleOCR URLs in OCR notebook files by updating cells directly."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# New SAMPLE_URLS section (removes broken URL variable entirely)
NEW_IMPORTS_SUFFIX = '''
warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
'''

# The new download_samples function (no network needed)
NEW_DOWNLOAD_SAMPLES = '''def _generate_synthetic_image(save_dir, idx=0):
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
            "Sample Document Processing 2026",
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
    for i in range(5):
        p = save_dir / f"synthetic_{i}.png"
        if not p.exists():
            p = _generate_synthetic_image(save_dir, i)
        if p and p.exists():
            paths.append(p)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        for found in Path(SAVE_DIR).rglob(f"*{ext}"):
            if found not in paths:
                paths.append(found)
    print(f"{len(paths)} images available for OCR")
    return paths'''


BROKEN_IMPORTS = '''
SAMPLE_URLS = [
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img_12.jpg",
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img623.jpg",
]

'''

OLD_DOWNLOAD_BODY = '''def download_samples():
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


root = Path(__file__).parent
broken_nbs = [
    root / "Computer Vision/Captcha Recognition/Captcha Recognition.ipynb",
    root / "Computer Vision/Document Word Detection/Document Word Detection.ipynb",
    root / "Computer Vision/Image Text Extraction - OCR/Image Text Extraction - OCR.ipynb",
    root / "Computer Vision/Image to Text Conversion - OCR/Image to Text Conversion - OCR.ipynb",
    root / "Computer Vision/QR Code Readability/QR Code Readability.ipynb",
]

fixed = 0
for nb_path in broken_nbs:
    if not nb_path.exists():
        print(f"NOT FOUND: {nb_path}")
        continue
    
    try:
        nb = json.loads(nb_path.read_text("utf-8", errors="replace"))
        changed = False
        
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            
            # Fix the imports cell (remove SAMPLE_URLS)
            if "SAMPLE_URLS" in src and "refs/heads/main" in src:
                new_src = src.replace(BROKEN_IMPORTS, "\n")
                # Clean up double newlines
                while "\n\n\n" in new_src:
                    new_src = new_src.replace("\n\n\n", "\n\n")
                cell["source"] = [new_src]
                changed = True
                print(f"  Fixed imports cell in {nb_path.name}")
            
            # Fix the download_samples function cell
            if "def download_samples():" in src and "urllib.request.urlretrieve(url" in src:
                # Find and replace just the download_samples function
                if OLD_DOWNLOAD_BODY in src:
                    new_src = src.replace(OLD_DOWNLOAD_BODY, NEW_DOWNLOAD_SAMPLES)
                else:
                    # Find the function start and replace everything until next `def`
                    lines = src.split("\n")
                    new_lines = []
                    i = 0
                    inserted = False
                    while i < len(lines):
                        if lines[i].startswith("def download_samples():") and not inserted:
                            new_lines.append(NEW_DOWNLOAD_SAMPLES)
                            inserted = True
                            # Skip until next top-level def or end
                            i += 1
                            while i < len(lines):
                                line = lines[i]
                                if line.startswith("def ") or line.startswith("class "):
                                    new_lines.append(line)
                                    i += 1
                                    break
                                i += 1
                        else:
                            new_lines.append(lines[i])
                            i += 1
                    new_src = "\n".join(new_lines)
                cell["source"] = [new_src]
                changed = True
                print(f"  Fixed download_samples cell in {nb_path.name}")
        
        if changed:
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), "utf-8")
            print(f"Saved: {nb_path.name}")
            fixed += 1
        else:
            print(f"No changes needed: {nb_path.name}")
    
    except Exception as e:
        print(f"ERROR {nb_path}: {e}")

print(f"\nTotal notebooks fixed: {fixed}")
