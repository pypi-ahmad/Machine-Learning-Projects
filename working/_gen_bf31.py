import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-001", "language": "markdown"},
            "source": [
                "# Building Footprint Segmentation (Inria Aerial)\n",
                "\n",
                "Task family: building footprint segmentation per-pixel.\n",
                "\n",
                "Dataset source: https://www.kaggle.com/datasets/sagar100rathod/inria-aerial-image-labeling-dataset"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-002", "language": "markdown"},
            "source": [
                "## Why This Method is Correct\n",
                "\n",
                "Building footprints require dense per-pixel segmentation masks. YOLO segmentation mode (YOLO26m-seg preferred, with fallback) is the correct model family for this task. We extract building regions via binary mask conversion to polygon labels."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-003", "language": "markdown"},
            "source": [
                "## Environment Setup"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-004", "language": "python"},
            "source": [
                "import importlib, subprocess, sys\n",
                "\n",
                "def ensure_pkg(import_name: str, pip_name: str | None = None):\n",
                "    try:\n",
                "        importlib.import_module(import_name)\n",
                "    except Exception:\n",
                "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name or import_name])\n",
                "\n",
                "ensure_pkg('kagglehub')\n",
                "ensure_pkg('numpy')\n",
                "ensure_pkg('pandas')\n",
                "ensure_pkg('PIL', 'Pillow')\n",
                "ensure_pkg('matplotlib')\n",
                "ensure_pkg('cv2', 'opencv-python')\n",
                "ensure_pkg('yaml', 'pyyaml')\n",
                "ensure_pkg('ultralytics')\n",
                "print('Dependencies ready.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-005", "language": "markdown"},
            "source": [
                "## Imports and Configuration"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-006", "language": "python"},
            "source": [
                "import json, os, random, shutil\n",
                "from pathlib import Path\n",
                "import cv2, numpy as np, yaml\n",
                "from PIL import Image\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "\n",
                "PROJECT_DIR = Path.cwd()\n",
                "WORK_DIR = PROJECT_DIR / 'working' / 'bf_seg'\n",
                "PREP_DIR = WORK_DIR / 'prepared_yolo_seg'\n",
                "ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'\n",
                "for d in [WORK_DIR, PREP_DIR, ARTIFACTS_DIR]:\n",
                "    d.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}\n",
                "MAX_TRAIN, MAX_VAL = 150, 35\n",
                "print(f'Project dir: {PROJECT_DIR}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-007", "language": "markdown"},
            "source": [
                "## Real Kaggle Dataset Download"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-008", "language": "python"},
            "source": [
                "import kagglehub\n",
                "\n",
                "if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):\n",
                "    raise EnvironmentError('Missing Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY.')\n",
                "\n",
                "dataset_root = Path(kagglehub.dataset_download('sagar100rathod/inria-aerial-image-labeling-dataset'))\n",
                "print(f'Dataset root: {dataset_root}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-009", "language": "markdown"},
            "source": [
                "## Discover and Validate Building Footprint Pairs"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-010", "language": "python"},
            "source": [
                "def gather_files(root: Path) -> list[Path]:\n",
                "    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]\n",
                "\n",
                "all_images = gather_files(dataset_root)\n",
                "if len(all_images) == 0:\n",
                "    raise RuntimeError('No images found after Kaggle download.')\n",
                "\n",
                "def mask_score(p: Path) -> int:\n",
                "    s = str(p).lower()\n",
                "    score = 0\n",
                "    if any(x in s for x in ['mask', 'label', 'gt', 'ground']):\n",
                "        score += 2\n",
                "    if any(x in s for x in ['image', 'img', 'rgb', 'photo']):\n",
                "        score -= 1\n",
                "    return score\n",
                "\n",
                "mask_cands = [p for p in all_images if mask_score(p) > 0]\n",
                "img_cands = [p for p in all_images if mask_score(p) <= 0]\n",
                "\n",
                "if len(mask_cands) == 0:\n",
                "    sample = all_images[: min(600, len(all_images))]\n",
                "    for p in sample:\n",
                "        arr = np.array(Image.open(p).convert('L'))\n",
                "        if len(np.unique(arr)) <= 6:\n",
                "            mask_cands.append(p)\n",
                "        else:\n",
                "            img_cands.append(p)\n",
                "\n",
                "if len(img_cands) == 0 or len(mask_cands) == 0:\n",
                "    raise RuntimeError('Could not separate image and mask candidates.')\n",
                "\n",
                "mask_by_stem = {m.stem: m for m in mask_cands}\n",
                "pairs = []\n",
                "for img in img_cands:\n",
                "    key = img.stem\n",
                "    m = mask_by_stem.get(key) or mask_by_stem.get(key.replace('_image', '_mask')) or mask_by_stem.get(key.replace('_img', '_mask'))\n",
                "    if m is not None:\n",
                "        pairs.append((img, m))\n",
                "\n",
                "if len(pairs) < 20:\n",
                "    raise RuntimeError(f'Too few pairs: {len(pairs)}')\n",
                "\n",
                "valid_pairs = []\n",
                "for img, m in pairs:\n",
                "    mask_arr = np.array(Image.open(m).convert('L'))\n",
                "    if mask_arr.shape[0] < 16 or mask_arr.shape[1] < 16:\n",
                "        continue\n",
                "    if int(mask_arr.max()) <= int(mask_arr.min()):\n",
                "        continue\n",
                "    valid_pairs.append((img, m))\n",
                "\n",
                "if len(valid_pairs) < 20:\n",
                "    raise RuntimeError(f'Not enough valid pairs: {len(valid_pairs)}')\n",
                "\n",
                "rng = random.Random(SEED)\n",
                "rng.shuffle(valid_pairs)\n",
                "split_idx = max(1, int(0.81 * len(valid_pairs)))\n",
                "train_pairs = valid_pairs[:split_idx][:MAX_TRAIN]\n",
                "val_pairs = valid_pairs[split_idx:][:MAX_VAL]\n",
                "if len(val_pairs) == 0:\n",
                "    val_pairs = train_pairs[-1:]\n",
                "    train_pairs = train_pairs[:-1]\n",
                "\n",
                "if len(train_pairs) == 0 or len(val_pairs) == 0:\n",
                "    raise RuntimeError('Empty train/val after split.')\n",
                "\n",
                "print(f'Candidates: imgs={len(img_cands)} masks={len(mask_cands)} | Valid pairs: {len(valid_pairs)}')\n",
                "print(f'Train: {len(train_pairs)} | Val: {len(val_pairs)}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-011", "language": "python"},
            "source": [
                "img0, m0 = train_pairs[0]\n",
                "img_arr = np.array(Image.open(img0).convert('RGB'))\n",
                "mask_arr = np.array(Image.open(m0).convert('L'))\n",
                "\n",
                "fig, ax = plt.subplots(1, 2, figsize=(10, 4))\n",
                "ax[0].imshow(img_arr)\n",
                "ax[0].set_title('Aerial Image')\n",
                "ax[0].axis('off')\n",
                "ax[1].imshow(mask_arr, cmap='gray')\n",
                "ax[1].set_title('Building Footprint Mask')\n",
                "ax[1].axis('off')\n",
                "plt.tight_layout()\n",
                "pair_preview = ARTIFACTS_DIR / 'building_footprint_sample.png'\n",
                "plt.savefig(pair_preview, dpi=140)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-012", "language": "markdown"},
            "source": [
                "## Convert Masks to YOLO Segmentation Labels"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-013", "language": "python"},
            "source": [
                "for split in ['train', 'val']:\n",
                "    (PREP_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)\n",
                "    (PREP_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "def mask_to_yolo_segs(mask_path: Path) -> list[str]:\n",
                "    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)\n",
                "    if m is None:\n",
                "        return []\n",
                "    h, w = m.shape\n",
                "    _, bin_m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n",
                "    contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
                "    rows = []\n",
                "    for c in contours:\n",
                "        if cv2.contourArea(c) < 50:\n",
                "            continue\n",
                "        c = c.squeeze(1)\n",
                "        if len(c.shape) != 2 or c.shape[0] < 3:\n",
                "            continue\n",
                "        pts = []\n",
                "        for x, y in c:\n",
                "            pts.extend([f'{x/w:.6f}', f'{y/h:.6f}'])\n",
                "        rows.append('0 ' + ' '.join(pts))\n",
                "    return rows\n",
                "\n",
                "def process_split(pairs: list, split: str) -> tuple[int, int]:\n",
                "    img_n, lbl_n = 0, 0\n",
                "    for idx, (img_p, mask_p) in enumerate(pairs):\n",
                "        new_name = f'{split}_{idx:05d}{img_p.suffix.lower()}'\n",
                "        dst_img = PREP_DIR / 'images' / split / new_name\n",
                "        dst_lbl = PREP_DIR / 'labels' / split / f'{Path(new_name).stem}.txt'\n",
                "        shutil.copy2(img_p, dst_img)\n",
                "        segs = mask_to_yolo_segs(mask_p)\n",
                "        with open(dst_lbl, 'w', encoding='utf-8') as f:\n",
                "            if segs:\n",
                "                f.write('\\n'.join(segs))\n",
                "        img_n += 1\n",
                "        if segs:\n",
                "            lbl_n += 1\n",
                "    return img_n, lbl_n\n",
                "\n",
                "tr_n, tr_lbl = process_split(train_pairs, 'train')\n",
                "va_n, va_lbl = process_split(val_pairs, 'val')\n",
                "\n",
                "if tr_n == 0 or va_n == 0:\n",
                "    raise RuntimeError('Empty split.')\n",
                "if tr_lbl == 0 or va_lbl == 0:\n",
                "    raise RuntimeError('No labeled images in split.')\n",
                "\n",
                "print(f'Train: {tr_n} imgs, {tr_lbl} labeled')\n",
                "print(f'Val: {va_n} imgs, {va_lbl} labeled')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-014", "language": "python"},
            "source": [
                "data_yaml = PREP_DIR / 'data.yaml'\n",
                "with open(data_yaml, 'w', encoding='utf-8') as f:\n",
                "    yaml.safe_dump({\n",
                "        'path': str(PREP_DIR),\n",
                "        'train': 'images/train',\n",
                "        'val': 'images/val',\n",
                "        'names': {0: 'building_footprint'}\n",
                "    }, f, sort_keys=False)\n",
                "\n",
                "print(f'Data yaml: {data_yaml}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-015", "language": "markdown"},
            "source": [
                "## Train YOLO26m-seg"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-016", "language": "python"},
            "source": [
                "from ultralytics import YOLO\n",
                "\n",
                "weights = ['yolo26m-seg.pt', 'yolo11m-seg.pt', 'yolov8m-seg.pt']\n",
                "selected_w = None\n",
                "model = None\n",
                "for w in weights:\n",
                "    try:\n",
                "        model = YOLO(w)\n",
                "        selected_w = w\n",
                "        break\n",
                "    except Exception:\n",
                "        pass\n",
                "\n",
                "if selected_w is None:\n",
                "    raise RuntimeError('No YOLO segmentation checkpoint available.')\n",
                "\n",
                "print(f'Using: {selected_w}')\n",
                "\n",
                "_ = model.train(\n",
                "    data=str(data_yaml),\n",
                "    epochs=2,\n",
                "    imgsz=512,\n",
                "    batch=4,\n",
                "    project=str(ARTIFACTS_DIR / 'yolo_runs'),\n",
                "    name='building_footprint_seg',\n",
                "    seed=SEED,\n",
                "    verbose=False\n",
                ")\n",
                "\n",
                "best_model_path = Path(model.trainer.best)\n",
                "print(f'Best model: {best_model_path}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-017", "language": "markdown"},
            "source": [
                "## Real Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-018", "language": "python"},
            "source": [
                "best = YOLO(str(best_model_path))\n",
                "val_res = best.val(data=str(data_yaml), split='val', imgsz=512, batch=4, verbose=False)\n",
                "\n",
                "map50 = float(val_res.seg.map50)\n",
                "map5095 = float(val_res.seg.map)\n",
                "prec = float(val_res.seg.mp)\n",
                "rec = float(val_res.seg.mr)\n",
                "\n",
                "print(f'Building Footprint Segmentation Results:')\n",
                "print(f'  mAP50: {map50:.4f}')\n",
                "print(f'  mAP50-95: {map5095:.4f}')\n",
                "print(f'  Precision: {prec:.4f}')\n",
                "print(f'  Recall: {rec:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-019", "language": "markdown"},
            "source": [
                "## Qualitative Predictions on Validation Set"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-020", "language": "python"},
            "source": [
                "val_imgs = sorted((PREP_DIR / 'images' / 'val').iterdir())\n",
                "sample_indices = list(range(min(6, len(val_imgs))))\n",
                "preds = best.predict([str(val_imgs[i]) for i in sample_indices], imgsz=512, conf=0.25, verbose=False)\n",
                "\n",
                "fig, ax = plt.subplots(2, 3, figsize=(15, 9))\n",
                "for i in range(6):\n",
                "    a = ax.flatten()[i]\n",
                "    if i >= len(preds):\n",
                "        a.axis('off')\n",
                "        continue\n",
                "    rendered = preds[i].plot()[:, :, ::-1]\n",
                "    a.imshow(rendered)\n",
                "    a.set_title(val_imgs[i].name)\n",
                "    a.axis('off')\n",
                "plt.tight_layout()\n",
                "qual_path = ARTIFACTS_DIR / 'building_footprint_qualitative.png'\n",
                "plt.savefig(qual_path, dpi=140)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-021", "language": "markdown"},
            "source": [
                "## Save Real Outputs"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "bf31-022", "language": "python"},
            "source": [
                "metrics = {\n",
                "    'dataset': 'sagar100rathod/inria-aerial-image-labeling-dataset',\n",
                "    'dataset_url': 'https://www.kaggle.com/datasets/sagar100rathod/inria-aerial-image-labeling-dataset',\n",
                "    'task': 'building_footprint_segmentation',\n",
                "    'train_pairs': len(train_pairs),\n",
                "    'val_pairs': len(val_pairs),\n",
                "    'selected_weights': selected_w,\n",
                "    'best_model_path': str(best_model_path),\n",
                "    'seg_map50': map50,\n",
                "    'seg_map50_95': map5095,\n",
                "    'seg_precision': prec,\n",
                "    'seg_recall': rec\n",
                "}\n",
                "\n",
                "metrics_path = ARTIFACTS_DIR / 'metrics.json'\n",
                "with open(metrics_path, 'w', encoding='utf-8') as f:\n",
                "    json.dump(metrics, f, indent=2)\n",
                "\n",
                "manifest = {\n",
                "    'pair_preview_png': str(pair_preview),\n",
                "    'qualitative_predictions_png': str(qual_path),\n",
                "    'metrics_json': str(metrics_path),\n",
                "    'data_yaml': str(data_yaml),\n",
                "    'best_model': str(best_model_path)\n",
                "}\n",
                "manifest_path = ARTIFACTS_DIR / 'artifact_manifest.json'\n",
                "with open(manifest_path, 'w', encoding='utf-8') as f:\n",
                "    json.dump(manifest, f, indent=2)\n",
                "\n",
                "print('Saved:')\n",
                "print(f'  Metrics: {metrics_path}')\n",
                "print(f'  Manifest: {manifest_path}')\n",
                "print(f'  Preview: {pair_preview}')\n",
                "print(f'  Qualitative: {qual_path}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "bf31-023", "language": "markdown"},
            "source": [
                "## Limitations\n",
                "\n",
                "- Contour-based label conversion can miss very small footprints.",
                "- Training is intentionally short for local runtime.",
                "- Extend epochs, tune augmentation, and increase dataset size for production use."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = Path(r'e:\Github\Machine-Learning-Projects\Computer Vision\Building Footprint Segmentation\Source Code\building_footprint_segmentation_pipeline.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'Wrote: {out_path}')
