"""
PHASE 3 — DATASET RESOLUTION SCRIPT
For every project: validate local data, attempt downloads, classify status.
Standardize layout: /data/project_slug/

NO FIXES TO NOTEBOOKS — ONLY DATA RESOLUTION.
"""

import os
import csv
import json
import shutil
import re
import urllib.request
import ssl
from pathlib import Path

ROOT = Path(r"d:\Workspace\Github\Machine-Learning-Projects")
DATA_ROOT = ROOT / "data"
AUDIT_DIR = ROOT / "audit_phase3"
AUDIT_DIR.mkdir(exist_ok=True)
DATA_ROOT.mkdir(exist_ok=True)

DATASET_EXT = {'.csv', '.tsv', '.xlsx', '.xls', '.json', '.parquet', '.feather',
               '.pkl', '.pickle', '.npy', '.npz', '.data', '.txt', '.rar', '.zip',
               '.gz', '.tar', '.h5', '.hdf5'}

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}

SKIP_DIRS = {'.git', 'audit_phase1', 'audit_phase2', 'audit_phase3', 'venv',
             '__pycache__', 'data', '.ipynb_checkpoints'}

# ─── CLASSIFICATION OVERRIDES (from deep inspection) ────────────────────────

BUILT_IN_DATASET_PROJECTS = {
    "Machine Learning Project 16 - Flower Species Classification": "seaborn.load_dataset('iris')",
    "Machine Learning Project 19 - Fashion Mnist Data Analysis Using ML": "torchvision.FashionMNIST(download=True)",
    "Machine Learning Project 33 - (Conceptual Analysis) Select Right Threhold Value using ROC curve": "sklearn.make_classification()",
    "Machine Learning Project 38 - TIme Series Forecasting": "inline_list_data",
    "Machine Learning Project 40 - Titanic Survival Prediction": "seaborn.load_dataset('titanic')",
    "Machine Learning Project 46 - Breast Cancer Detection": "sklearn.load_breast_cancer()",
    "Machine Learning Projects  91 - Text Summarization using Word Frequency - NLP": "inline_string_data",
}

# Projects where dataset IS present but phase1 scanner missed it
LOCAL_OVERRIDE = {
    "Machine Learning Project 35 - Spam Classifier": "smsspamcollection/SMSSpamCollection",
}

# Projects with .rar archives that need extraction
RAR_ARCHIVES = {
    "Machine Learning Project 55 - Million Songs Dataset - Recommendation Engine": ["song_data.rar", "triplets_file.rar"],
}

# Link-only projects with hardcoded author paths but no data shipped
HARDCODED_PATH_MISSING_DATA = {
    "Machine Learning Project 11 - Creadi Card Fraud Problem Handling Imbalanced Dataset": "creditcard.csv (Kaggle: credit card fraud dataset)",
    "Machine Learning Project 24 - Dog Vs Cat Classification": "Dog vs Cat image dataset (Kaggle: dogs-vs-cats)",
    "Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning": "Indian classical dance image dataset",
    "Machine Learning Project 25 - Advanced Credit Card fraud Detection": "creditcard.csv (Kaggle: credit card fraud detection)",
    "Machine Learning Project 30 - Pneumonia Classification": "chest_xray images (Kaggle: chest-xray-pneumonia)",
}

# Projects where data is loaded programmatically via APIs/built-in but have Kaggle links
KAGGLE_LINK_PROJECTS = {
    "Machine Learning Project 12 - Hand Digit Recognition Using ML": "kaggle: dczerniawko/fifa19-analysis",
    "Machine Learning Project 20 - Face Mask Detection using ML": "kaggle: ashishjangra27/face-mask-12k-images-dataset",
    "Machine Learning Project 21 - Face Expression Identifier": "kaggle: manishshah120/facial-expression-recog-image-ver-of-fercdataset",
    "Machine Learning Project 29 - Plant Diseases Recognition": "kaggle: vipoooool/new-plant-diseases-dataset",
    "Machine Learning Project 42 - Traffic_Sign_Recognition": "kaggle: meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
    "Machine Learning Projects 147 - Language Translation Model using ML": "kaggle: anilreddy8989/word-guess-with-rnn",
}

# Downloadable URLs (cleaned from link extraction)
DOWNLOADABLE_URLS = {
    "Machine Learning Project 3 - Boston Housing Analysis": [
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", "housing.data"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names", "housing.names"),
    ],
    "Machine Learning Project 41 - Tracking of Covid-19": [
        ("https://www.mohfw.gov.in/data/datanew.json", "datanew.json"),
    ],
    "Machine Learning Projects  87 - Clustering - KMeans Clustering for Imagery Analysis": [
        ("https://s3.amazonaws.com/img-datasets/mnist.npz", "mnist.npz"),
    ],
}

# Projects with truly missing external data (not in repo, not downloadable easily)
EXTERNAL_FILE_MISSING = {
    "Machine Learning Projects  90 - SMS Spam Detection Analysis - NLP": "spam.csv",
    "Machine Learning Projects 109 - Mercari Price Suggestion Lightgbm": "train.tsv (Kaggle: mercari-price-suggestion-challenge)",
    "Machine Learning Projects 124 - (Conceptual) Text Classification Keras": "Consumer_Complaints.csv",
    "Machine Learning Projects 125 - Text Classification keras_consumer_complaints": "Consumer_Complaints.csv + glove.6B.50d.txt.gz",
    "Machine Learning Projects 137 - Consumer_complaints": "Consumer_Complaints.csv",
    "Machine Learning Projects 144 - LSTM Time Series Power Consumption": "household_power_consumption.txt (UCI ML Repo)",
    "Machine Learning Projects 149 - Twitter Sentiment Analysis - NLP": "Twitter Sentiments.csv",
    "Machine Learning Projects 86 - Autoencoder Fashion MNIST": "Fashion MNIST gz files (can use keras.datasets instead)",
    "Machine Learning Projects 92 - Movie Recommendation Engine": "movies.csv + ratings.csv (from data/movie_dataset/)",
    "Machine Learning Project 34 - Live Smile Detector": "frontal_face.xml (OpenCV cascade, can be resolved from opencv data)",
}


def slugify(name):
    """Create a clean folder-safe slug from project name."""
    slug = re.sub(r'^Machine Learning Project[s]?\s*\d+\s*[-–]\s*', '', name)
    slug = re.sub(r'^Project\s*\d+\s*[-–]\s*', '', slug)
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s]+', '_', slug).strip('_')
    return slug.lower()[:80]


def find_dataset_files(proj_dir):
    """Find all dataset files in a project directory (non-recursive into known skip dirs)."""
    files = []
    for item in proj_dir.rglob('*'):
        if item.is_file():
            # Skip .ipynb_checkpoints and __pycache__
            parts = item.parts
            if any(skip in parts for skip in ('.ipynb_checkpoints', '__pycache__')):
                continue
            ext = item.suffix.lower()
            if ext in DATASET_EXT or ext in IMAGE_EXT:
                files.append(item)
    return files


def find_dataset_dirs(proj_dir):
    """Find directories that likely contain image datasets (train/test/val structure)."""
    dataset_dirs = []
    for d in proj_dir.rglob('*'):
        if d.is_dir() and d.name.lower() in ('train', 'test', 'val', 'validation', 'data', 'dataset', 'images'):
            dataset_dirs.append(d)
    return dataset_dirs


def attempt_download(url, dest_path, timeout=30):
    """Attempt to download a URL. Returns (success, error_msg)."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(resp.read())
        return True, None
    except Exception as e:
        return False, str(e)[:200]


def copy_datasets_to_data_dir(proj_name, proj_dir, data_dest):
    """Copy dataset files from project dir to centralized data dir."""
    data_dest.mkdir(parents=True, exist_ok=True)
    copied = []

    dataset_files = find_dataset_files(proj_dir)
    dataset_dirs = find_dataset_dirs(proj_dir)

    # Check if there's a structured image dataset (train/test/val)
    has_image_structure = any(d.name.lower() in ('train', 'test', 'val', 'validation')
                             for d in dataset_dirs)

    if has_image_structure:
        # Find the parent of train/test/val and copy the whole structure
        parents_seen = set()
        for d in dataset_dirs:
            if d.name.lower() in ('train', 'test', 'val', 'validation'):
                parent = d.parent
                if parent not in parents_seen:
                    parents_seen.add(parent)
                    dest_sub = data_dest / d.name
                    if not dest_sub.exists():
                        try:
                            shutil.copytree(d, dest_sub)
                            img_count = sum(1 for _ in dest_sub.rglob('*') if _.is_file())
                            copied.append(f"{d.name}/ ({img_count} files)")
                        except Exception as e:
                            copied.append(f"FAILED: {d.name}/ -> {str(e)[:80]}")
    else:
        # Copy individual dataset files
        for f in dataset_files:
            rel = f.relative_to(proj_dir)
            # Skip if it's a model artifact or README-related
            if f.suffix.lower() in {'.pkl', '.pickle', '.h5', '.hdf5', '.pt', '.pth'}:
                # Only skip if it's clearly a model, not data
                if 'model' in f.name.lower() or 'tokenizer' in f.name.lower():
                    continue
            dest_file = data_dest / rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if not dest_file.exists():
                try:
                    shutil.copy2(f, dest_file)
                    copied.append(str(rel))
                except Exception as e:
                    copied.append(f"FAILED: {rel} -> {str(e)[:80]}")

    return copied


# ─── MAIN RESOLUTION ─────────────────────────────────────────────────────────

def main():
    # Load Phase 1 inventory
    with open(ROOT / 'audit_phase1' / 'project_inventory.csv', encoding='utf-8') as f:
        inventory = list(csv.DictReader(f))

    proj_dirs = sorted([
        d for d in ROOT.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and d.name != 'data'
    ])

    results = []
    print(f"Phase 3: Resolving datasets for {len(proj_dirs)} projects...\n")

    for idx, proj_dir in enumerate(proj_dirs, 1):
        proj_name = proj_dir.name
        slug = slugify(proj_name)
        data_dest = DATA_ROOT / slug

        inv_row = next((r for r in inventory if r['project'] == proj_name), None)
        dataset_source = inv_row['dataset_source'] if inv_row else 'unknown'

        status = 'UNKNOWN'
        detail = ''
        files_moved = []

        print(f"  [{idx:03d}] {proj_name[:70]}")

        # ── CASE: Built-in dataset (no file needed) ──
        if proj_name in BUILT_IN_DATASET_PROJECTS:
            status = 'OK_BUILTIN'
            detail = BUILT_IN_DATASET_PROJECTS[proj_name]
            print(f"        -> BUILTIN: {detail}")

        # ── CASE: Local override (was misclassified as none) ──
        elif proj_name in LOCAL_OVERRIDE:
            dataset_source = 'local'
            detail = LOCAL_OVERRIDE[proj_name]
            status = 'OK_LOCAL'
            files_moved = copy_datasets_to_data_dir(proj_name, proj_dir, data_dest)
            print(f"        -> LOCAL (override): {detail}")

        # ── CASE: Has .rar archives needing extraction ──
        elif proj_name in RAR_ARCHIVES:
            status = 'BLOCKED_RAR'
            archives = RAR_ARCHIVES[proj_name]
            detail = f"Has RAR archives ({', '.join(archives)}) but cannot auto-extract without unrar"
            print(f"        -> BLOCKED_RAR: {detail}")

        # ── CASE: External file missing (known) ──
        elif proj_name in EXTERNAL_FILE_MISSING:
            status = 'BLOCKED_MISSING'
            detail = EXTERNAL_FILE_MISSING[proj_name]
            print(f"        -> BLOCKED_MISSING: {detail}")

        # ── CASE: Hardcoded path, data not shipped ──
        elif proj_name in HARDCODED_PATH_MISSING_DATA:
            status = 'BLOCKED_MISSING'
            detail = HARDCODED_PATH_MISSING_DATA[proj_name]
            print(f"        -> BLOCKED_MISSING (hardcoded path): {detail}")

        # ── CASE: Kaggle-only links (need API/manual) ──
        elif proj_name in KAGGLE_LINK_PROJECTS:
            status = 'BLOCKED_KAGGLE'
            detail = KAGGLE_LINK_PROJECTS[proj_name]
            print(f"        -> BLOCKED_KAGGLE: {detail}")

        # ── CASE: Downloadable URL ──
        elif proj_name in DOWNLOADABLE_URLS:
            urls = DOWNLOADABLE_URLS[proj_name]
            all_ok = True
            downloaded = []
            for url, filename in urls:
                dest_file = data_dest / filename
                if dest_file.exists():
                    downloaded.append(f"{filename} (already exists)")
                    continue
                print(f"        -> Downloading {filename}...")
                ok, err = attempt_download(url, dest_file)
                if ok:
                    downloaded.append(f"{filename} (downloaded)")
                else:
                    downloaded.append(f"{filename} FAILED: {err}")
                    all_ok = False
            if all_ok:
                status = 'DOWNLOADED'
            else:
                status = 'DOWNLOAD_PARTIAL'
            detail = '; '.join(downloaded)
            files_moved = downloaded
            print(f"        -> {'DOWNLOADED' if all_ok else 'PARTIAL'}: {detail[:100]}")

        # ── CASE A: Local dataset exists ──
        elif dataset_source == 'local':
            dataset_files = find_dataset_files(proj_dir)
            if dataset_files:
                status = 'OK_LOCAL'
                files_moved = copy_datasets_to_data_dir(proj_name, proj_dir, data_dest)
                detail = f"{len(dataset_files)} files found, {len(files_moved)} copied to data/{slug}/"
                print(f"        -> OK_LOCAL: {detail}")
            else:
                status = 'OK_LOCAL_NODATA'
                detail = 'Phase1 said local but no dataset ext files found on re-scan'
                print(f"        -> WARN: {detail}")

        # ── CASE B: Link only (not in our special lists — generic handler) ──
        elif dataset_source == 'link_only':
            # Already handled special cases above; anything remaining is unresolvable
            status = 'BLOCKED_LINK_ONLY'
            detail = 'Link-only but no downloadable URL identified'
            print(f"        -> BLOCKED_LINK_ONLY: {detail}")

        # ── CASE C: No dataset detected at all ──
        elif dataset_source == 'none_detected':
            status = 'BLOCKED_MISSING'
            detail = 'No dataset file or link detected'
            print(f"        -> BLOCKED: {detail}")

        # ── CASE: Non-project directories (audit_phase2, etc) ──
        else:
            status = 'SKIP'
            detail = 'Not a ML project directory'
            print(f"        -> SKIP: {detail}")

        results.append({
            'project': proj_name,
            'slug': slug,
            'original_source': dataset_source,
            'status': status,
            'detail': detail,
            'files_standardized': '|'.join(str(f) for f in files_moved[:10]) if files_moved else '',
            'data_path': str(data_dest) if status in ('OK_LOCAL', 'DOWNLOADED', 'OK_BUILTIN') else '',
        })

    # ─── Write Results ────────────────────────────────────────────────────────
    output_csv = AUDIT_DIR / 'phase3_dataset_status.csv'
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ─── Summary ──────────────────────────────────────────────────────────────
    status_counts = {}
    for r in results:
        s = r['status']
        status_counts[s] = status_counts.get(s, 0) + 1

    blocked = [r for r in results if r['status'].startswith('BLOCKED')]

    print(f"\n{'='*60}")
    print("PHASE 3 COMPLETE")
    print(f"{'='*60}")
    print("\nDataset Status Summary:")
    for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:30s}: {c}")
    print(f"\nTotal BLOCKED (need user input): {len(blocked)}")
    print(f"\nArtifacts:")
    print(f"  {output_csv}")
    print(f"  {DATA_ROOT}/")

    # Write blocked list for user
    blocked_path = AUDIT_DIR / 'phase3_blocked.csv'
    if blocked:
        with open(blocked_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['project', 'status', 'detail'])
            w.writeheader()
            for r in blocked:
                w.writerow({'project': r['project'], 'status': r['status'], 'detail': r['detail']})
        print(f"  {blocked_path}")


if __name__ == '__main__':
    main()
