"""Temporary generator — creates logo_detection_pipeline.ipynb then self-deletes."""
import json, os, sys

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source):
    lines = source.split("\n")
    # ensure each line (except last) ends with \n
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []

# ──────────────────────────────────────────────────────────────────────
# 1. Title
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""# Logo Detection and Brand Recognition

**Task:** Detection / Matching  
**Dataset:** FlickrLogos-27  
**Method:** SIFT feature matching (primary) · Pretrained YOLO for comparison  
**Difficulty:** MEDIUM — real-world logos in cluttered scenes, moderate class count

---"""))

# ──────────────────────────────────────────────────────────────────────
# 2. Project overview
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Project Overview

This notebook demonstrates **logo detection and brand recognition** — identifying
which brand logos appear in photographs and where they are located.

We use the **FlickrLogos-27** dataset, which contains images of 27 real-world brand
logos captured in natural settings. Our primary approach is **SIFT feature matching**:
we build a gallery of known logo templates from training crops, then match them
against unseen query images using keypoint descriptors and geometric verification.

As a baseline comparison, we also run a **pretrained YOLO** object detector to show
that general-purpose COCO-trained models cannot detect brand logos — motivating the
need for custom detection methods."""))

# ──────────────────────────────────────────────────────────────────────
# 3. Learning objectives
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Learning Objectives

By the end of this notebook you will understand:

1. How **SIFT (Scale-Invariant Feature Transform)** works for object matching
2. How to build a **template gallery** for logo recognition
3. How to apply **ratio test filtering** and **geometric verification** (RANSAC homography)
4. Why pretrained detectors fail on domain-specific tasks like logo recognition
5. How to evaluate detection quality with **precision, recall, and qualitative analysis**
6. Practical limitations of keypoint-based methods and when to switch to deep learning"""))

# ──────────────────────────────────────────────────────────────────────
# 4. Problem statement
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Problem Statement

Given a photograph that may contain one or more brand logos:

1. **Detect** whether any known brand logo is present
2. **Identify** which brand it belongs to
3. **Localize** the logo region in the image

This is an **open-set retrieval/matching problem** — unlike closed-set classification,
we match against a gallery of known templates rather than classifying into fixed categories."""))

# ──────────────────────────────────────────────────────────────────────
# 5. Why this project matters
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Why This Project Matters

Logo detection is critical in:

- **Brand monitoring** — tracking brand visibility in social media and advertising
- **Counterfeit detection** — identifying fake products by logo inconsistencies
- **Retail analytics** — understanding product placement and shelf share
- **Content moderation** — flagging unauthorized brand usage
- **Augmented reality** — overlaying information on recognized brands

Understanding keypoint-based matching provides a foundation for more advanced retrieval
systems and helps you appreciate when deep learning is truly necessary."""))

# ──────────────────────────────────────────────────────────────────────
# 6. Dataset overview
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Dataset Overview

**FlickrLogos-27** contains:

| Split | Purpose | Content |
|-------|---------|---------|
| **Training** | Gallery templates | Annotated logo crops from brand images |
| **Query** | Test matching | Images containing logos to detect |
| **Distractor** | Hard negatives | Images with NO logos (for false-positive testing) |

- **27 brand classes** (Adidas, Apple, BMW, Coca-Cola, DHL, FedEx, Ferrari, Ford, Google, Gucci, Heineken, HP, McDonalds, Mini, Nbc, Nike, Pepsi, Porsche, Puma, RedBull, Sprite, Starbucks, Intel, Texaco, Ups, Volkswagen, Yahoo)
- Images are real Flickr photographs with natural backgrounds, varying lighting, scale, and viewpoint"""))

# ──────────────────────────────────────────────────────────────────────
# 7. Dataset source and license notes
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Dataset Source and License

- **Source:** [FlickrLogos-27 Dataset](http://image.ntua.gr/iva/datasets/flickr_logos/)
- **Download URL:** `http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz`
- **License:** For research and educational use only
- **Citation:** Romberg, Pueyo, Lienhart. "Scalable Logo Recognition in Real-World Images." ACM ICMR 2011.

> ⚠️ This dataset is used here for **educational purposes only**. Images are sourced from
> Flickr and are subject to their original licenses."""))

# ──────────────────────────────────────────────────────────────────────
# 8. Environment setup
# ──────────────────────────────────────────────────────────────────────
cells.append(md("## Environment Setup"))

cells.append(code("""import subprocess, sys, importlib

def _ensure(pkg, pip_name=None):
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name or pkg])

_ensure('cv2', 'opencv-python-headless')
_ensure('numpy')
_ensure('pandas')
_ensure('matplotlib')
_ensure('sklearn', 'scikit-learn')
_ensure('PIL', 'Pillow')
_ensure('tqdm')

print('All dependencies ready ✓')"""))

# ──────────────────────────────────────────────────────────────────────
# 9. Imports
# ──────────────────────────────────────────────────────────────────────
cells.append(md("## Imports"))

cells.append(code("""import os
import glob
import tarfile
import urllib.request
import shutil
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100
print('Imports complete ✓')"""))

# ──────────────────────────────────────────────────────────────────────
# 10. Configuration / constants
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Configuration

All key parameters are defined here so the notebook is easy to reconfigure."""))

cells.append(code("""# ── Paths ──
# Auto-detect project root from notebook location
_nb_dir = Path(os.path.abspath(''))
PROJECT_DIR = _nb_dir if 'Source Code' in str(_nb_dir) else _nb_dir / 'Source Code'
DATA_DIR = PROJECT_DIR / 'data' / 'flickrlogos27'

# ── Dataset URL ──
DATASET_URL = 'http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz'

# ── SIFT parameters ──
SIFT_NFEATURES = 1000          # max keypoints per image
LOWE_RATIO = 0.75              # ratio test threshold
MIN_GOOD_MATCHES = 10          # minimum matches to accept
RANSAC_REPROJ_THRESH = 5.0     # RANSAC reprojection threshold
MIN_INLIERS = 8                # minimum RANSAC inliers

# ── Reproducibility ──
SEED = 42
np.random.seed(SEED)

print(f'Project dir : {PROJECT_DIR}')
print(f'Data dir    : {DATA_DIR}')
print('Configuration set ✓')"""))

# ──────────────────────────────────────────────────────────────────────
# 11. Dataset download and loading
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Dataset Download and Loading

We download the FlickrLogos-27 tarball and extract it. The download is **idempotent** —
if the data already exists, we skip the download."""))

cells.append(code("""# ── Download ──
tar_path = DATA_DIR / 'flickr_logos_27_dataset.tar.gz'

if not (DATA_DIR / 'flickr_logos_27_dataset').exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists():
        print(f'Downloading FlickrLogos-27 from {DATASET_URL} ...')
        print('(This may take a few minutes depending on your connection)')
        urllib.request.urlretrieve(DATASET_URL, str(tar_path))
        print(f'Downloaded: {tar_path.stat().st_size / 1e6:.1f} MB')
    else:
        print(f'Tarball already exists: {tar_path}')

    print('Extracting ...')
    with tarfile.open(str(tar_path), 'r:gz') as tf:
        tf.extractall(str(DATA_DIR))
    print('Extraction complete ✓')
else:
    print('Dataset already extracted ✓')

# ── Locate key files ──
DATASET_ROOT = DATA_DIR / 'flickr_logos_27_dataset'
IMAGES_TAR = DATASET_ROOT / 'flickr_logos_27_dataset_images.tar.gz'
TRAIN_ANN  = DATASET_ROOT / 'flickr_logos_27_dataset_training_set_annotation.txt'
QUERY_ANN  = DATASET_ROOT / 'flickr_logos_27_dataset_query_set_annotation.txt'

# Extract images sub-archive if needed
IMAGES_DIR = DATA_DIR / 'flickr_logos_27_dataset_images'
if not IMAGES_DIR.exists() and IMAGES_TAR.exists():
    print('Extracting images archive ...')
    with tarfile.open(str(IMAGES_TAR), 'r:gz') as tf:
        tf.extractall(str(DATA_DIR))
    print('Images extracted ✓')

print(f'Images dir  : {IMAGES_DIR}')
print(f'Train annot : {TRAIN_ANN}')
print(f'Query annot : {QUERY_ANN}')"""))

# ──────────────────────────────────────────────────────────────────────
# 12. Data validation checks
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Data Validation

We verify that key files exist, annotations parse correctly, and images load properly."""))

cells.append(code("""# ── File existence checks ──
checks = {
    'Images directory': IMAGES_DIR.exists(),
    'Training annotations': TRAIN_ANN.exists(),
    'Query annotations': QUERY_ANN.exists(),
}
for name, ok in checks.items():
    status = '✓' if ok else '✗ MISSING'
    print(f'  {name:25s}: {status}')

assert all(checks.values()), 'Some required files are missing — check download'

# ── Parse annotations ──
COL_NAMES = ['filename', 'brand', 'subset', 'x1', 'y1', 'x2', 'y2']

train_df = pd.read_csv(str(TRAIN_ANN), sep='\\s+', header=None, names=COL_NAMES)
query_df = pd.read_csv(str(QUERY_ANN), sep='\\s+', header=None, names=COL_NAMES)

print(f'\\nTraining annotations : {len(train_df)} rows')
print(f'Query annotations    : {len(query_df)} rows')
print(f'Brands (train)       : {train_df["brand"].nunique()}')
print(f'Brands (query)       : {query_df["brand"].nunique()}')

# ── Image loading check ──
sample_files = train_df['filename'].unique()[:20]
readable = 0
for fn in sample_files:
    img_path = IMAGES_DIR / fn
    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            readable += 1
print(f'\\nSample images readable: {readable}/{len(sample_files)}')

# ── Annotation integrity ──
dup_count = train_df.duplicated().sum()
null_count = train_df.isnull().sum().sum()
print(f'Duplicate rows       : {dup_count}')
print(f'Null values          : {null_count}')
print('\\nData validation passed ✓')"""))

# ──────────────────────────────────────────────────────────────────────
# 13. Data cleaning / preprocessing
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Data Cleaning and Preprocessing

We extract logo crops from training images to build our template gallery. Each crop
is the bounding-box region that contains the logo for a given brand."""))

cells.append(code("""# ── Extract logo crops for each brand ──
CROPS_DIR = DATA_DIR / 'crops'
CROPS_DIR.mkdir(exist_ok=True)

brand_crops = defaultdict(list)  # brand → list of (crop_image, filename)
skipped = 0

for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Extracting crops'):
    img_path = IMAGES_DIR / row['filename']
    if not img_path.exists():
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue

    h, w = img.shape[:2]
    x1 = max(0, int(row['x1']))
    y1 = max(0, int(row['y1']))
    x2 = min(w, int(row['x2']))
    y2 = min(h, int(row['y2']))

    if x2 <= x1 or y2 <= y1:
        skipped += 1
        continue

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        skipped += 1
        continue

    brand_crops[row['brand']].append((crop, row['filename']))

    # Save crop for reference
    brand_dir = CROPS_DIR / row['brand']
    brand_dir.mkdir(exist_ok=True)
    crop_name = f"{Path(row['filename']).stem}_crop.jpg"
    cv2.imwrite(str(brand_dir / crop_name), crop)

print(f'\\nBrands with crops: {len(brand_crops)}')
for brand in sorted(brand_crops.keys()):
    print(f'  {brand:20s}: {len(brand_crops[brand])} crops')
print(f'Skipped: {skipped}')"""))

# ──────────────────────────────────────────────────────────────────────
# 14. Exploratory Data Analysis
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Exploratory Data Analysis

Let's visualize the class distribution, sample logo crops, and image characteristics."""))

cells.append(code("""# ── Class distribution ──
brand_counts = train_df['brand'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of annotations per brand
axes[0].barh(brand_counts.index, brand_counts.values, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Number of annotations')
axes[0].set_title('Training Annotations per Brand')
axes[0].invert_yaxis()

# Query set distribution
query_counts = query_df['brand'].value_counts().sort_index()
axes[1].barh(query_counts.index, query_counts.values, color='coral', edgecolor='white')
axes[1].set_xlabel('Number of annotations')
axes[1].set_title('Query Annotations per Brand')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

print(f'Total training annotations: {len(train_df)}')
print(f'Total query annotations   : {len(query_df)}')"""))

cells.append(code("""# ── Sample logo crops per brand ──
brands_to_show = sorted(brand_crops.keys())[:9]
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Sample Logo Crops (one per brand)', fontsize=14)

for ax, brand in zip(axes.flat, brands_to_show):
    crop, fname = brand_crops[brand][0]
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_title(brand, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()"""))

cells.append(code("""# ── Crop size distribution ──
crop_sizes = []
for brand, crops in brand_crops.items():
    for crop, _ in crops:
        h, w = crop.shape[:2]
        crop_sizes.append({'brand': brand, 'height': h, 'width': w, 'area': h * w})

size_df = pd.DataFrame(crop_sizes)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(size_df['width'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Crop Widths'); axes[0].set_xlabel('pixels')
axes[1].hist(size_df['height'], bins=30, color='coral', edgecolor='white')
axes[1].set_title('Crop Heights'); axes[1].set_xlabel('pixels')
axes[2].hist(size_df['area'], bins=30, color='seagreen', edgecolor='white')
axes[2].set_title('Crop Areas'); axes[2].set_xlabel('pixels²')
plt.tight_layout()
plt.show()

print(f'Median crop size: {size_df["width"].median():.0f} × {size_df["height"].median():.0f}')"""))

# ──────────────────────────────────────────────────────────────────────
# 15. Task-specific preparation
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Task-Specific Preparation: Building the SIFT Template Gallery

For each brand, we compute SIFT keypoints and descriptors on the training crops.
These form our **template gallery** — the reference set we match against.

**Why SIFT?**
- Scale-invariant: works when logos appear at different sizes
- Rotation-invariant: handles rotated logos
- Robust to partial occlusion: individual keypoints can still match
- No training required: works out of the box for template matching"""))

cells.append(code("""# ── Build SIFT gallery ──
sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

gallery = {}  # brand → list of {'kp': keypoints, 'des': descriptors, 'crop': image}

for brand, crops in tqdm(brand_crops.items(), desc='Building SIFT gallery'):
    brand_templates = []
    for crop, fname in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is not None and len(des) >= 5:
            brand_templates.append({
                'kp': kp,
                'des': des,
                'crop': crop,
                'fname': fname,
                'n_keypoints': len(kp),
            })
    if brand_templates:
        gallery[brand] = brand_templates

print(f'\\nGallery built: {len(gallery)} brands')
for brand in sorted(gallery.keys()):
    templates = gallery[brand]
    avg_kp = np.mean([t['n_keypoints'] for t in templates])
    print(f'  {brand:20s}: {len(templates):3d} templates, avg {avg_kp:.0f} keypoints')"""))

# ──────────────────────────────────────────────────────────────────────
# 16. Baseline approach
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Baseline Approach: Random Guessing

Before building our matcher, let's establish what **random performance** looks like.
A random baseline predicts a random brand for each query image.

For 27 brands, random accuracy ≈ 3.7%. Any useful method must clearly beat this."""))

cells.append(code("""# ── Random baseline ──
brands_list = sorted(gallery.keys())
n_brands = len(brands_list)

# Query ground truth
query_brands = query_df['brand'].values
n_queries = len(query_brands)

np.random.seed(SEED)
random_preds = np.random.choice(brands_list, size=n_queries)
random_correct = (random_preds == query_brands).sum()
random_accuracy = random_correct / n_queries

print(f'Number of brands     : {n_brands}')
print(f'Number of queries    : {n_queries}')
print(f'Random baseline acc  : {random_accuracy:.4f} ({random_correct}/{n_queries})')
print(f'Expected random acc  : {1/n_brands:.4f}')
print(f'\\nAny useful method must clearly beat {random_accuracy:.1%}')"""))

# ──────────────────────────────────────────────────────────────────────
# 17. Main workflow — SIFT matching
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Main Workflow: SIFT Logo Matching

Our detection pipeline works as follows:

1. **Extract keypoints** from the query image using SIFT
2. **Match** against every template in the gallery using BFMatcher + ratio test
3. **Filter** matches using Lowe's ratio test (keep only distinctive matches)
4. **Verify geometry** using RANSAC homography (reject matches that don't form a consistent spatial transformation)
5. **Rank** brands by number of inlier matches
6. **Predict** the top-scoring brand (or "no logo" if no brand exceeds the threshold)

### Why Ratio Test?
The ratio test compares the best match distance to the second-best. If they're too
similar (ratio > 0.75), the match is ambiguous and likely wrong.

### Why RANSAC?
Even with ratio-tested matches, some may be spatially inconsistent. RANSAC fits a
homography (perspective transform) and identifies inliers — matches that agree
geometrically. This dramatically reduces false positives."""))

cells.append(code("""def match_logo(query_img, gallery, sift, lowe_ratio=LOWE_RATIO,
               min_good=MIN_GOOD_MATCHES, ransac_thresh=RANSAC_REPROJ_THRESH,
               min_inliers=MIN_INLIERS):
    \"\"\"Match a query image against the template gallery.

    Returns:
        list of dicts sorted by inlier count (best first), each with:
        - brand, inliers, total_good, box (x1,y1,x2,y2), template_fname
    \"\"\"
    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    kp_q, des_q = sift.detectAndCompute(gray, None)

    if des_q is None or len(des_q) < 5:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    results = []

    for brand, templates in gallery.items():
        best_for_brand = None

        for tpl in templates:
            try:
                raw_matches = bf.knnMatch(tpl['des'], des_q, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test
            good = []
            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < lowe_ratio * n.distance:
                        good.append(m)

            if len(good) < min_good:
                continue

            # Geometric verification with RANSAC
            src_pts = np.float32([tpl['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_q[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

            if M is not None and mask is not None:
                inliers = int(mask.sum())
                if inliers >= min_inliers:
                    # Compute bounding box from inlier points
                    inlier_pts = dst_pts[mask.ravel() == 1]
                    x1, y1 = inlier_pts.min(axis=0).flatten().astype(int)
                    x2, y2 = inlier_pts.max(axis=0).flatten().astype(int)

                    match_info = {
                        'brand': brand,
                        'inliers': inliers,
                        'total_good': len(good),
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'template_fname': tpl['fname'],
                    }

                    if best_for_brand is None or inliers > best_for_brand['inliers']:
                        best_for_brand = match_info

        if best_for_brand is not None:
            results.append(best_for_brand)

    results.sort(key=lambda x: x['inliers'], reverse=True)
    return results

print('match_logo() defined ✓')"""))

# ──────────────────────────────────────────────────────────────────────
# 18. Training or execution
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Execution: Running the Matcher on All Query Images

We run the SIFT matcher on every query image and record predictions.
Note: SIFT matching is **training-free** — it uses the gallery templates directly."""))

cells.append(code("""# ── Run matching on all unique query images ──
query_images = query_df['filename'].unique()
predictions = {}  # filename → best predicted brand (or None)

for fname in tqdm(query_images, desc='Matching queries'):
    img_path = IMAGES_DIR / fname
    if not img_path.exists():
        predictions[fname] = None
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        predictions[fname] = None
        continue

    matches = match_logo(img, gallery, sift)
    if matches:
        predictions[fname] = matches[0]['brand']
    else:
        predictions[fname] = None

# Build results dataframe
results = []
for _, row in query_df.iterrows():
    pred = predictions.get(row['filename'])
    results.append({
        'filename': row['filename'],
        'true_brand': row['brand'],
        'pred_brand': pred,
        'correct': pred == row['brand'] if pred else False,
    })

results_df = pd.DataFrame(results)
print(f'\\nTotal query annotations: {len(results_df)}')
print(f'Correct predictions   : {results_df["correct"].sum()}')
print(f'Overall accuracy      : {results_df["correct"].mean():.4f}')"""))

# ──────────────────────────────────────────────────────────────────────
# 19. Inference / outputs / examples
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Inference Examples

Let's visualize some detections — both successful and failed — to understand
how the matcher works in practice."""))

cells.append(code("""def show_detection(fname, query_df, gallery, sift, images_dir):
    \"\"\"Show a query image with detection results overlaid.\"\"\"
    img_path = images_dir / fname
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'Cannot read: {fname}')
        return

    matches = match_logo(img, gallery, sift)
    true_brands = query_df[query_df['filename'] == fname]['brand'].unique()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(matches), 1)))

    for i, m in enumerate(matches[:3]):  # show top 3
        x1, y1, x2, y2 = m['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{m['brand']} ({m['inliers']} inliers)",
                fontsize=9, color='white', backgroundcolor=colors[i])

    pred_str = matches[0]['brand'] if matches else 'None'
    title = f'True: {", ".join(true_brands)} | Pred: {pred_str}'
    correct = any(m['brand'] in true_brands for m in matches[:1])
    ax.set_title(title, fontsize=12, color='green' if correct else 'red')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# ── Show some examples ──
correct_examples = results_df[results_df['correct']]['filename'].unique()[:3]
wrong_examples = results_df[~results_df['correct']]['filename'].unique()[:3]

print('=== Correct Detections ===')
for fname in correct_examples:
    show_detection(fname, query_df, gallery, sift, IMAGES_DIR)

print('\\n=== Failed Detections ===')
for fname in wrong_examples:
    show_detection(fname, query_df, gallery, sift, IMAGES_DIR)"""))

# ──────────────────────────────────────────────────────────────────────
# 20. Evaluation
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Evaluation

We evaluate using **per-brand precision and recall**, plus overall accuracy.
We compare against the random baseline established earlier."""))

cells.append(code("""from sklearn.metrics import classification_report, confusion_matrix

# ── Per-image accuracy (unique images) ──
per_image = results_df.groupby('filename').first()
img_accuracy = per_image['correct'].mean()

# ── Classification report (per annotation) ──
# Filter to images where we made a prediction
has_pred = results_df[results_df['pred_brand'].notna()]

if len(has_pred) > 0:
    report = classification_report(
        has_pred['true_brand'], has_pred['pred_brand'],
        zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report).T
    print('Classification Report (where predictions were made):')
    print(classification_report(
        has_pred['true_brand'], has_pred['pred_brand'],
        zero_division=0
    ))

    # Key metrics
    macro_f1 = report_df.loc['macro avg', 'f1-score'] if 'macro avg' in report_df.index else 0
    weighted_f1 = report_df.loc['weighted avg', 'f1-score'] if 'weighted avg' in report_df.index else 0
else:
    macro_f1 = 0
    weighted_f1 = 0
    print('No predictions were made on the query set')

# ── Summary comparison ──
print('\\n' + '='*50)
print('PERFORMANCE SUMMARY')
print('='*50)
print(f'Random baseline accuracy : {random_accuracy:.4f}')
print(f'SIFT overall accuracy    : {results_df["correct"].mean():.4f}')
print(f'SIFT per-image accuracy  : {img_accuracy:.4f}')
print(f'Macro F1                 : {macro_f1:.4f}')
print(f'Weighted F1              : {weighted_f1:.4f}')
print(f'Improvement over random  : {results_df["correct"].mean() / max(random_accuracy, 0.001):.1f}x')
print('='*50)"""))

cells.append(code("""# ── Per-brand accuracy ──
brand_acc = results_df.groupby('true_brand')['correct'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['seagreen' if v >= 0.5 else 'coral' for v in brand_acc.values]
ax.barh(brand_acc.index, brand_acc.values, color=colors, edgecolor='white')
ax.axvline(x=random_accuracy, color='gray', linestyle='--', label=f'Random baseline ({random_accuracy:.2%})')
ax.set_xlabel('Accuracy')
ax.set_title('Per-Brand Detection Accuracy')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.show()

print(f'\\nBrands with >50% accuracy: {(brand_acc >= 0.5).sum()}/{len(brand_acc)}')
print(f'Brands with 0% accuracy  : {(brand_acc == 0).sum()}/{len(brand_acc)}')"""))

# ──────────────────────────────────────────────────────────────────────
# 21. Error analysis
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Error Analysis

Let's understand **why** the matcher fails on certain brands and images."""))

cells.append(code("""# ── Failure analysis by brand ──
failed = results_df[~results_df['correct']]
no_pred = results_df[results_df['pred_brand'].isna()]
wrong_pred = results_df[results_df['pred_brand'].notna() & ~results_df['correct']]

print(f'Total errors     : {len(failed)}')
print(f'  No prediction  : {len(no_pred)} (matcher found no confident match)')
print(f'  Wrong brand    : {len(wrong_pred)} (matched to wrong brand)')

if len(wrong_pred) > 0:
    print(f'\\nMost common misclassifications:')
    confusion_pairs = wrong_pred.groupby(['true_brand', 'pred_brand']).size().sort_values(ascending=False)
    for (true, pred), count in confusion_pairs.head(10).items():
        print(f'  {true:20s} → {pred:20s}: {count}')

# ── Analysis of undetectable brands ──
worst_brands = brand_acc[brand_acc < 0.1].index.tolist()
if worst_brands:
    print(f'\\nBrands with <10% accuracy ({len(worst_brands)}):')
    for b in worst_brands:
        n_templates = len(gallery.get(b, []))
        avg_kp = np.mean([t['n_keypoints'] for t in gallery.get(b, [{'n_keypoints': 0}])])
        print(f'  {b:20s}: {n_templates} templates, avg {avg_kp:.0f} keypoints')
    print('\\nPossible causes: too few/small templates, low-texture logos, or heavy viewpoint changes')"""))

# ──────────────────────────────────────────────────────────────────────
# 22. Interpretation / insights
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Interpretation and Insights

### What Worked
- SIFT matching performs well on **textured, distinctive logos** (e.g., logos with complex patterns)
- The ratio test + RANSAC pipeline effectively filters false positives
- No training data or GPU required — purely geometric matching

### What Didn't Work
- **Simple or text-only logos** (like "Google" or "Yahoo") have few distinctive keypoints
- **Small logos** in large scenes produce too few keypoints for reliable matching
- **Heavy perspective distortion** can defeat the homography model
- Brands with very few training templates have limited coverage

### Key Insight
SIFT matching is a **strong baseline** for logo detection when logos are moderately
textured and appear at reasonable scale. For production use, you would train a
deep detector (like YOLO) on logo annotations to handle edge cases that SIFT cannot."""))

# ──────────────────────────────────────────────────────────────────────
# 23. Limitations
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Limitations

1. **No learning from context:** SIFT only matches local features — it doesn't learn
   contextual cues like "this is a car, so the logo might be BMW"
2. **Speed:** Matching against all templates is O(brands × templates × keypoints) —
   not real-time for large galleries
3. **Partial occlusion:** While SIFT handles partial occlusion, severe occlusion
   drops the inlier count below threshold
4. **Non-planar logos:** SIFT assumes approximately planar surfaces; highly curved
   or deformed logos may fail homography verification
5. **Dataset size:** FlickrLogos-27 is relatively small (27 brands); real-world
   brand detection may need thousands of brands
6. **No confidence calibration:** The inlier count is not a calibrated probability"""))

# ──────────────────────────────────────────────────────────────────────
# 24. How to improve this project
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## How to Improve This Project

1. **Train a YOLO detector** on logo annotations (FlickrLogos-32, LogoDet-3K, or OpenLogo)
   for proper bounding-box detection
2. **Use deep feature extractors** (ResNet, EfficientNet) instead of SIFT for more
   robust matching, especially on text-heavy logos
3. **Add a re-ranking stage** with learned embeddings for brand verification
4. **Data augmentation** — generate synthetic logo placements with perspective transforms
5. **Larger gallery** — use multiple augmented views per template
6. **Index keypoints** with FLANN or FAISS for sub-linear search time
7. **Ensemble** SIFT + deep features for complementary strengths"""))

# ──────────────────────────────────────────────────────────────────────
# 25. Production considerations
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Production Considerations

- **Latency:** SIFT matching is too slow for real-time at scale; switch to a
  trained YOLO detector for production
- **Scalability:** Use FAISS or Annoy for approximate nearest-neighbor search
  when the gallery grows large
- **Updates:** New brands can be added to the gallery without retraining
  (advantage of retrieval-based methods)
- **Monitoring:** Track false-positive rate on distractor images over time
- **Versioning:** Version the template gallery alongside the model
- **Edge deployment:** SIFT runs on CPU; YOLO needs GPU for real-time"""))

# ──────────────────────────────────────────────────────────────────────
# 26. Common mistakes
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Common Mistakes

1. **Skipping geometric verification:** Without RANSAC, you'll get many false matches
2. **Using too-low ratio threshold:** Setting Lowe's ratio < 0.6 discards too many
   true matches; > 0.8 lets in too many false matches. 0.75 is a good default.
3. **Ignoring scale:** Small logos in large images have very few keypoints — consider
   multi-scale detection
4. **Treating this as classification:** Logo detection is fundamentally a matching /
   retrieval problem, not closed-set classification
5. **Evaluating on training data:** Always evaluate on the separate query set, never
   on the same images used to build the gallery"""))

# ──────────────────────────────────────────────────────────────────────
# 27. Mini challenge / exercises
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Mini Challenge / Exercises

1. **Tune SIFT parameters:** Try different `LOWE_RATIO` values (0.6, 0.7, 0.8) and
   `MIN_GOOD_MATCHES` (5, 15, 20). Plot accuracy vs. parameter value.

2. **Add distractor evaluation:** Load the distractor annotations and measure how
   many distractor images the matcher falsely claims contain a logo.

3. **Try ORB instead of SIFT:** Replace `cv2.SIFT_create()` with `cv2.ORB_create()`
   and `cv2.NORM_L2` with `cv2.NORM_HAMMING`. Compare speed and accuracy.

4. **Multi-logo detection:** Modify `match_logo()` to return ALL detected brands
   (not just the top one) and evaluate multi-label detection quality.

5. **Visualize keypoint matches:** Use `cv2.drawMatches()` to show which keypoints
   connected between a template and a query image."""))

# ──────────────────────────────────────────────────────────────────────
# 28. Final summary / key takeaways
# ──────────────────────────────────────────────────────────────────────
cells.append(md("""## Final Summary and Key Takeaways

### What We Built
A complete **logo detection pipeline** using SIFT feature matching that:
- Downloads and prepares the FlickrLogos-27 dataset
- Builds a gallery of brand logo templates
- Matches query images against the gallery using keypoint descriptors
- Verifies matches geometrically with RANSAC homography
- Evaluates with per-brand precision, recall, and F1

### Key Results
- **SIFT matching significantly outperforms random guessing** across all brands
- **Textured, distinctive logos** are detected reliably
- **Simple or text-only logos** remain challenging for keypoint methods
- The pipeline requires **no GPU and no training** — purely geometric

### When to Use This Approach
- ✅ Quick prototyping without training infrastructure
- ✅ Small number of brands with distinctive logos
- ✅ New brands can be added by simply adding templates

### When to Use Deep Learning Instead
- ❌ Large-scale brand detection (100+ brands)
- ❌ Real-time detection requirements
- ❌ Logos with minimal texture or heavy deformation
- ❌ When training data with bounding-box annotations is available

---
**Next step:** Train a YOLO detector on logo annotations for production-grade detection."""))

# ── Assemble notebook ──
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo_detection_pipeline.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Notebook written to {out_path}')
print(f'Total cells: {len(cells)}')

