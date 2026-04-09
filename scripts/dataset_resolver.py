#!/usr/bin/env python3
"""
Dataset Resolver — Deep Learning Projects Monorepo
====================================================
Scans all projects, extracts dataset URLs from documentation (.txt, .pdf, .docx),
and ensures each project has a working data/download_dataset.py script.

Usage:
    python scripts/dataset_resolver.py              # Scan & report
    python scripts/dataset_resolver.py --generate   # Generate missing download scripts
    python scripts/dataset_resolver.py --execute    # Run all download scripts
"""

import argparse
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s | %(message)s",
)
logger = logging.getLogger("dataset_resolver")

# ── Constants ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = [
    "Anomaly detection and fraud detection",
    "Associate Rule Learning",
    "Chat bot",
    "GANS",
    "Recommendation Systems",
    "Reinforcement Learning",
    "Speech and Audio processing",
]

URL_PATTERN = re.compile(
    r'https?://[^\s<>"\')\]},;]+',
    re.IGNORECASE,
)
KAGGLE_PATTERN = re.compile(
    r'kaggle\.com/(?:datasets?/)?([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)',
    re.IGNORECASE,
)
GDRIVE_PATTERN = re.compile(
    r'drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([a-zA-Z0-9_-]+)',
    re.IGNORECASE,
)


# ── Text Extraction ─────────────────────────────────────────
def extract_text_from_txt(filepath: Path) -> str:
    """Read plain text file."""
    try:
        return filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Cannot read %s: %s", filepath, e)
        return ""


def extract_text_from_pdf(filepath: Path) -> str:
    """Extract text from PDF using PyPDF2 or pdfplumber."""
    text = ""
    # Try PyPDF2 first
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(filepath))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except ImportError:
        pass
    except Exception as e:
        logger.debug("PyPDF2 failed for %s: %s", filepath, e)

    # Fallback: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(filepath)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        logger.warning("Install PyPDF2 or pdfplumber to extract PDF text: pip install PyPDF2")
    except Exception as e:
        logger.debug("pdfplumber failed for %s: %s", filepath, e)

    return text


def extract_text_from_docx(filepath: Path) -> str:
    """Extract text from docx using python-docx."""
    try:
        import docx
        doc = docx.Document(str(filepath))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        logger.warning("Install python-docx to extract .docx text: pip install python-docx")
    except Exception as e:
        logger.warning("Cannot read %s: %s", filepath, e)
    return ""


# ── URL Extraction ──────────────────────────────────────────
def extract_urls(text: str) -> dict:
    """Extract categorized URLs from text."""
    all_urls = URL_PATTERN.findall(text)
    # Clean trailing punctuation
    cleaned = []
    for url in all_urls:
        url = url.rstrip(".,;:!?)")
        cleaned.append(url)

    kaggle_datasets = []
    gdrive_ids = []
    other_urls = []

    for url in cleaned:
        km = KAGGLE_PATTERN.search(url)
        gm = GDRIVE_PATTERN.search(url)
        if km:
            kaggle_datasets.append(km.group(1))
        elif gm:
            gdrive_ids.append(gm.group(1))
        else:
            other_urls.append(url)

    return {
        "kaggle": list(set(kaggle_datasets)),
        "gdrive": list(set(gdrive_ids)),
        "urls": list(set(other_urls)),
    }


# ── Project Scanner ─────────────────────────────────────────
def discover_projects() -> list:
    """Find all project directories."""
    projects = []
    for category in CATEGORIES:
        cat_path = ROOT / category
        if not cat_path.is_dir():
            continue
        for entry in sorted(cat_path.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                projects.append({
                    "category": category,
                    "name": entry.name,
                    "path": entry,
                })
    return projects


def scan_project(project: dict) -> dict:
    """Scan a single project for documentation and dataset info."""
    path = project["path"]
    info = {
        **project,
        "docs": [],
        "has_code_py": (path / "run.py").exists(),
        "has_notebook": False,
        "notebook_file": None,
        "has_download_script": (path / "data" / "download_dataset.py").exists(),
        "extracted_urls": {"kaggle": [], "gdrive": [], "urls": []},
    }

    # Find notebooks
    notebooks = list(path.glob("*.ipynb"))
    if notebooks:
        info["has_notebook"] = True
        info["notebook_file"] = notebooks[0].name

    # Scan documentation files
    doc_extensions = {".txt", ".pdf", ".docx"}
    for f in path.iterdir():
        if f.suffix.lower() in doc_extensions:
            info["docs"].append(f.name)

            # Extract text
            text = ""
            if f.suffix.lower() == ".txt":
                text = extract_text_from_txt(f)
            elif f.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(f)
            elif f.suffix.lower() == ".docx":
                text = extract_text_from_docx(f)

            # Extract URLs
            if text:
                urls = extract_urls(text)
                for key in urls:
                    info["extracted_urls"][key].extend(urls[key])

    # Deduplicate
    for key in info["extracted_urls"]:
        info["extracted_urls"][key] = list(set(info["extracted_urls"][key]))

    return info


# ── Download Script Generator ────────────────────────────────
def generate_download_script(project_info: dict) -> str:
    """Generate a download_dataset.py script for the project."""
    name = project_info["name"]
    urls = project_info["extracted_urls"]
    lines = [
        '#!/usr/bin/env python3',
        '"""',
        f'Dataset downloader for: {name}',
        'Auto-generated by scripts/dataset_resolver.py',
        '"""',
        '',
        'from pathlib import Path',
        'import logging',
        'import os',
        'import sys',
        '',
        'logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")',
        'logger = logging.getLogger(__name__)',
        '',
        'DATA_DIR = Path(__file__).resolve().parent',
        '',
        '',
        'def _extract(filepath: Path):',
        '    """Extract archive if applicable."""',
        '    name = filepath.name.lower()',
        '    if name.endswith(".zip"):',
        '        import zipfile',
        '        with zipfile.ZipFile(filepath, "r") as zf:',
        '            zf.extractall(filepath.parent)',
        '        logger.info("Extracted ZIP → %s", filepath.parent)',
        '    elif name.endswith((".tar.gz", ".tgz")):',
        '        import tarfile',
        '        with tarfile.open(filepath, "r:gz") as tf:',
        '            tf.extractall(filepath.parent)',
        '        logger.info("Extracted TAR.GZ → %s", filepath.parent)',
        '    elif name.endswith(".tar"):',
        '        import tarfile',
        '        with tarfile.open(filepath, "r:") as tf:',
        '            tf.extractall(filepath.parent)',
        '        logger.info("Extracted TAR → %s", filepath.parent)',
        '    elif name.endswith(".gz") and not name.endswith(".tar.gz"):',
        '        import gzip, shutil',
        '        out = filepath.with_suffix("")',
        '        with gzip.open(filepath, "rb") as f_in, open(out, "wb") as f_out:',
        '            shutil.copyfileobj(f_in, f_out)',
        '        logger.info("Decompressed GZ → %s", out)',
        '',
        '',
    ]

    # Kaggle download
    if urls.get("kaggle"):
        ds = urls["kaggle"][0]
        lines.extend([
            'def main():',
            f'    """Download dataset from Kaggle: {ds}"""',
            '    try:',
            '        import opendatasets as od',
            '    except ImportError:',
            '        logger.error("Install opendatasets: pip install opendatasets")',
            '        sys.exit(1)',
            '',
            f'    logger.info("Downloading Kaggle dataset: {ds}")',
            '    logger.info("You may need Kaggle API credentials (~/.kaggle/kaggle.json)")',
            f'    od.download("https://www.kaggle.com/datasets/{ds}", data_dir=str(DATA_DIR))',
            '    logger.info("Download complete → %s", DATA_DIR)',
        ])
    # Google Drive download
    elif urls.get("gdrive"):
        file_id = urls["gdrive"][0]
        lines.extend([
            'def main():',
            f'    """Download dataset from Google Drive: {file_id}"""',
            '    try:',
            '        import gdown',
            '    except ImportError:',
            '        logger.error("Install gdown: pip install gdown")',
            '        sys.exit(1)',
            '',
            f'    url = "https://drive.google.com/uc?id={file_id}"',
            '    dest = DATA_DIR / "dataset.zip"',
            '',
            '    if dest.exists():',
            '        logger.info("Already downloaded: %s", dest)',
            '    else:',
            '        logger.info("Downloading from Google Drive...")',
            '        gdown.download(url, str(dest), quiet=False)',
            '',
            '    _extract(dest)',
            '    logger.info("Download complete → %s", DATA_DIR)',
        ])
    # Direct URL download
    elif urls.get("urls"):
        url = urls["urls"][0]
        filename = url.split("/")[-1].split("?")[0] or "dataset.zip"
        lines.extend([
            'def main():',
            '    """Download dataset from URL."""',
            '    import urllib.request',
            '',
            f'    url = "{url}"',
            f'    dest = DATA_DIR / "{filename}"',
            '',
            '    if dest.exists():',
            '        logger.info("Already downloaded: %s", dest)',
            '    else:',
            '        logger.info("Downloading %s ...", url)',
            '        urllib.request.urlretrieve(url, str(dest))',
            '',
            '    _extract(dest)',
            '    logger.info("Download complete → %s", DATA_DIR)',
        ])
    else:
        lines.extend([
            'def main():',
            '    """',
            '    No dataset URL found in project documentation.',
            '    Please add the dataset manually to this data/ directory,',
            '    or update this script with the correct download URL.',
            '    """',
            f'    logger.info("Project: {name}")',
            '    logger.info("No automatic download URL found.")',
            '    logger.info("Check the project PDF/DOCX for dataset source.")',
            '    logger.info("Data directory: %s", DATA_DIR)',
        ])

    lines.extend([
        '',
        '',
        'if __name__ == "__main__":',
        '    main()',
        '',
    ])

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dataset Resolver for DL Projects Monorepo")
    parser.add_argument("--generate", action="store_true", help="Generate missing download scripts")
    parser.add_argument("--execute", action="store_true", help="Execute all download scripts")
    parser.add_argument("--json", action="store_true", help="Output scan results as JSON")
    args = parser.parse_args()

    logger.info("Scanning workspace: %s", ROOT)
    projects = discover_projects()
    logger.info("Found %d projects across %d categories", len(projects), len(CATEGORIES))

    results = []
    for proj in projects:
        info = scan_project(proj)
        results.append(info)

    # Report
    print("\n" + "=" * 70)
    print("  DATASET RESOLVER — SCAN RESULTS")
    print("=" * 70)

    has_download = sum(1 for r in results if r["has_download_script"])
    has_urls = sum(1 for r in results if any(r["extracted_urls"][k] for k in r["extracted_urls"]))

    print(f"\n  Total projects     : {len(results)}")
    print(f"  With download.py   : {has_download}")
    print(f"  With extracted URLs : {has_urls}")
    print()

    for r in results:
        status = "✓" if r["has_download_script"] else "✗"
        url_count = sum(len(v) for v in r["extracted_urls"].values())
        print(f"  [{status}] {r['category']}/{r['name']}")
        if r["docs"]:
            print(f"      docs: {', '.join(r['docs'])}")
        if url_count > 0:
            for k, v in r["extracted_urls"].items():
                if v:
                    print(f"      {k}: {v}")

    # Generate missing scripts
    if args.generate:
        print("\n" + "=" * 70)
        print("  GENERATING DOWNLOAD SCRIPTS")
        print("=" * 70)
        for r in results:
            data_dir = r["path"] / "data"
            script_path = data_dir / "download_dataset.py"
            if not script_path.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                script_content = generate_download_script(r)
                script_path.write_text(script_content, encoding="utf-8")
                logger.info("Generated: %s", script_path.relative_to(ROOT))
            else:
                logger.debug("Exists: %s", script_path.relative_to(ROOT))

    # Execute download scripts
    if args.execute:
        print("\n" + "=" * 70)
        print("  EXECUTING DOWNLOAD SCRIPTS")
        print("=" * 70)
        for r in results:
            script_path = r["path"] / "data" / "download_dataset.py"
            if script_path.exists():
                logger.info("Running: %s", script_path.relative_to(ROOT))
                try:
                    subprocess.run(
                        [sys.executable, str(script_path)],
                        cwd=str(r["path"]),
                        timeout=300,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    logger.warning("Timeout: %s", script_path.relative_to(ROOT))
                except Exception as e:
                    logger.error("Error running %s: %s", script_path.relative_to(ROOT), e)

    # JSON output
    if args.json:
        serializable = []
        for r in results:
            serializable.append({
                "category": r["category"],
                "name": r["name"],
                "path": str(r["path"]),
                "has_code_py": r["has_code_py"],
                "has_notebook": r["has_notebook"],
                "notebook_file": r["notebook_file"],
                "has_download_script": r["has_download_script"],
                "docs": r["docs"],
                "extracted_urls": r["extracted_urls"],
            })
        print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
