"""OCR template: PaddleOCR — April 2026"""
import textwrap


def generate(project_path, config):
    return textwrap.dedent('''\
        """
        Modern OCR Pipeline (April 2026)
        Model: PaddleOCR (multilingual, high-accuracy)
        """
        import os, json, warnings
        from pathlib import Path

        warnings.filterwarnings("ignore")


        def find_images():
            data_dir = Path(os.path.dirname(__file__))
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf")
            files = []
            for ext in exts:
                files.extend(data_dir.rglob(f"*{ext}"))
            return files


        def run_ocr(files):
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
            results = []

            for f in files[:30]:
                result = ocr.ocr(str(f), cls=True)
                texts = []
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        conf = line[1][1]
                        texts.append({"text": text, "confidence": conf})

                full_text = " ".join(t["text"] for t in texts)
                results.append({
                    "file": f.name,
                    "full_text": full_text,
                    "lines": texts,
                    "n_lines": len(texts),
                })
                print(f"  ✓ {f.name}: {len(texts)} lines — '{full_text[:80]}...'")

            return results


        def main():
            print("=" * 60)
            print("MODERN OCR PIPELINE")
            print("Model: PaddleOCR (GPU)")
            print("=" * 60)

            files = find_images()
            print(f"Found {len(files)} image/PDF files")

            if not files:
                print("⚠ No image files found in project directory.")
                return

            results = run_ocr(files)

            # Save results
            out_path = os.path.join(os.path.dirname(__file__), "ocr_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\\n✓ OCR results saved to {out_path}")
            print(f"  Processed {len(results)} files, {sum(r['n_lines'] for r in results)} total lines")


        if __name__ == "__main__":
            main()
    ''')
