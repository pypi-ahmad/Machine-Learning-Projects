"""
Modern OCR Pipeline (April 2026)
Model: PaddleOCR (GPU, multilingual)
Data: Auto-downloads sample document images at runtime
"""
import os, json, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

SAMPLE_URLS = [
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img_12.jpg",
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img623.jpg",
]


def download_samples():
    save_dir = Path(os.path.dirname(__file__)) / "ocr_samples"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            urllib.request.urlretrieve(url, str(fname))
        paths.append(fname)
    # Also gather any local images
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        paths.extend([p for p in Path(os.path.dirname(__file__)).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available for OCR")
    return paths


def run_ocr(files):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
    results = []
    for f in files[:30]:
        result = ocr.ocr(str(f), cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append({"text": line[1][0], "confidence": line[1][1]})
        full_text = " ".join(t["text"] for t in texts)
        results.append({"file": f.name, "full_text": full_text, "lines": texts, "n_lines": len(texts)})
        print(f"  ✓ {f.name}: {len(texts)} lines — '{full_text[:80]}...'")
    return results


def main():
    print("=" * 60)
    print("OCR — PaddleOCR (GPU)")
    print("=" * 60)
    files = download_samples()
    results = run_ocr(files)
    out_path = os.path.join(os.path.dirname(__file__), "ocr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved to {out_path}: {len(results)} files, {sum(r['n_lines'] for r in results)} lines")


if __name__ == "__main__":
    main()
