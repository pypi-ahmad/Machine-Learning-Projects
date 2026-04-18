import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-001", "language": "markdown"},
            "source": [
                "# Cell Nuclei Segmentation (DSB 2018)\n",
                "\n",
                "Task family: biomedical instance segmentation.\n",
                "\n",
                "Dataset source: https://www.kaggle.com/competitions/data-science-bowl-2018/data"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-002", "language": "markdown"},
            "source": [
                "## Why YOLO Segmentation Is Correct\n",
                "\n",
                "Cell nuclei segmentation requires dense per-pixel region detection and segmentation of individual nuclei instances. YOLO26m-seg (instance segmentation) is the correct model family because it:\n",
                "- Outputs per-pixel masks for each detected nucleus\n",
                "- Handles small objects (nuclei are typically small regions)\n",
                "- Provides real-time biomedical segmentation capability\n",
                "- Can be trained on contour-to-polygon labels extracted from binary masks"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-003", "language": "markdown"},
            "source": [
                "## Environment Setup"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-004", "language": "python"},
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
                "ensure_pkg('scipy')\n",
                "print('Dependencies ready.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-005", "language": "markdown"},
            "source": [
                "## Imports and Configuration"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-006", "language": "python"},
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
                "WORK_DIR = PROJECT_DIR / 'working' / 'cn_seg'\n",
                "PREP_DIR = WORK_DIR / 'prepared_yolo_seg'\n",
                "ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'\n",
                "for d in [WORK_DIR, PREP_DIR, ARTIFACTS_DIR]:\n",
                "    d.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}\n",
                "MAX_TRAIN, MAX_VAL = 120, 30\n",
                "print(f'Project dir: {PROJECT_DIR}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-007", "language": "markdown"},
            "source": [
                "## Real Kaggle Dataset Download"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-008", "language": "python"},
            "source": [
                "import kagglehub\n",
                "\n",
                "if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):\n",
                "    raise EnvironmentError('Missing Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY.')\n",
                "\n",
                "try:\n",
                "    dataset_root = Path(kagglehub.competitions_download_cli('data-science-bowl-2018'))\n",
                "    print(f'Kaggle DSB 2018 root: {dataset_root}')\n",
                "except Exception as e:\n",
                "    print(f'Competition download failed: {e}. Attempting dataset download...')\n",
                "    dataset_root = Path(kagglehub.dataset_download('kmader/dsb-2018-dataset'))\n",
                "    print(f'Alternative dataset root: {dataset_root}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-009", "language": "markdown"},
            "source": [
                "## Discover and Validate Nucleus Image-Mask Pairs\n",
                "\n",
                "DSB 2018 typically contains images and corresponding mask folders. Each nucleus is labeled separately in the masks."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-010", "language": "python"},
            "source": [
                "def gather_files(root: Path, ext_set) -> list[Path]:\n",
                "    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ext_set]\n",
                "\n",
                "def find_image_mask_pairs(dataset_root):\n",
                "    img_dirs = []\n",
                "    for d in dataset_root.iterdir():\n",
                "        if d.is_dir() and any((d / n).exists() for n in ['images', 'masks', 'label']):\n",
                "            img_dirs.append(d)\n",
                "    if not img_dirs:\n",
                "        img_dirs = list(dataset_root.glob('**/images'))\n",
                "    if not img_dirs:\n",
                "        img_dirs = [dataset_root]\n",
                "    return img_dirs\n",
                "\n",
                "img_roots = find_image_mask_pairs(dataset_root)\n",
                "pairs = []\n",
                "\n",
                "for img_root in img_roots:\n",
                "    imgs = gather_files(img_root, IMG_EXTS)\n",
                "    if len(imgs) > 0:\n",
                "        for img in imgs:\n",
                "            mask_dir = img.parent / 'masks' if (img.parent / 'masks').exists() else None\n",
                "            if mask_dir and mask_dir.is_dir():\n",
                "                mask_files = list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))\n",
                "                if len(mask_files) > 0:\n",
                "                    combined_mask = np.zeros(np.array(Image.open(img)).shape[:2], dtype=np.uint8)\n",
                "                    for mf in mask_files:\n",
                "                        m_arr = np.array(Image.open(mf).convert('L'))\n",
                "                        combined_mask = np.maximum(combined_mask, m_arr)\n",
                "                    pairs.append((img, combined_mask))\n",
                "\n",
                "if len(pairs) == 0:\n",
                "    all_images = gather_files(dataset_root, IMG_EXTS)\n",
                "    for img in all_images:\n",
                "        stem_key = img.stem\n",
                "        mask_candidates = [p for p in gather_files(dataset_root, IMG_EXTS) \n",
                "                          if 'mask' in str(p).lower() and stem_key in str(p)]\n",
                "        if mask_candidates:\n",
                "            pairs.append((img, mask_candidates[0]))\n",
                "\n",
                "if len(pairs) < 10:\n",
                "    raise RuntimeError(f'Too few image-mask pairs found: {len(pairs)}')\n",
                "\n",
                "valid_pairs = []\n",
                "for img_p, mask_item in pairs:\n",
                "    img_arr = np.array(Image.open(img_p).convert('RGB'))\n",
                "    if isinstance(mask_item, np.ndarray):\n",
                "        mask_arr = mask_item\n",
                "    else:\n",
                "        mask_arr = np.array(Image.open(mask_item).convert('L'))\n",
                "    if img_arr.shape[0] < 32 or img_arr.shape[1] < 32:\n",
                "        continue\n",
                "    if int(mask_arr.max()) <= int(mask_arr.min()):\n",
                "        continue\n",
                "    valid_pairs.append((img_p, mask_arr))\n",
                "\n",
                "if len(valid_pairs) < 10:\n",
                "    raise RuntimeError(f'Not enough valid pairs: {len(valid_pairs)}')\n",
                "\n",
                "rng = random.Random(SEED)\n",
                "rng.shuffle(valid_pairs)\n",
                "split_idx = max(1, int(0.80 * len(valid_pairs)))\n",
                "train_pairs = valid_pairs[:split_idx][:MAX_TRAIN]\n",
                "val_pairs = valid_pairs[split_idx:][:MAX_VAL]\n",
                "\n",
                "if len(val_pairs) == 0:\n",
                "    val_pairs = train_pairs[-1:]\n",
                "    train_pairs = train_pairs[:-1]\n",
                "\n",
                "if len(train_pairs) == 0 or len(val_pairs) == 0:\n",
                "    raise RuntimeError('Empty split.')\n",
                "\n",
                "print(f'Valid pairs: {len(valid_pairs)}')\n",
                "print(f'Train: {len(train_pairs)} | Val: {len(val_pairs)}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-011", "language": "python"},
            "source": [
                "if isinstance(train_pairs[0][0], Path):\n",
                "    img0_arr = np.array(Image.open(train_pairs[0][0]).convert('RGB'))\n",
                "else:\n",
                "    img0_arr = train_pairs[0][0]\n",
                "    \n",
                "mask0_arr = train_pairs[0][1] if isinstance(train_pairs[0][1], np.ndarray) else np.array(Image.open(train_pairs[0][1]).convert('L'))\n",
                "\n",
                "fig, ax = plt.subplots(1, 2, figsize=(10, 4))\n",
                "ax[0].imshow(img0_arr)\n",
                "ax[0].set_title('Cell Image')\n",
                "ax[0].axis('off')\n",
                "ax[1].imshow(mask0_arr, cmap='gray')\n",
                "ax[1].set_title('Nuclei Mask')\n",
                "ax[1].axis('off')\n",
                "plt.tight_layout()\n",
                "pair_preview = ARTIFACTS_DIR / 'nuclei_sample.png'\n",
                "plt.savefig(pair_preview, dpi=140)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-012", "language": "markdown"},
            "source": [
                "## Convert Masks to YOLO Segmentation Labels"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-013", "language": "python"},
            "source": [
                "for split in ['train', 'val']:\n",
                "    (PREP_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)\n",
                "    (PREP_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "def mask_to_yolo_segs_from_array(mask_arr) -> list[str]:\n",
                "    if isinstance(mask_arr, Path):\n",
                "        mask_arr = np.array(Image.open(mask_arr).convert('L'))\n",
                "    h, w = mask_arr.shape\n",
                "    _, bin_m = cv2.threshold(mask_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n",
                "    contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
                "    rows = []\n",
                "    for c in contours:\n",
                "        if cv2.contourArea(c) < 20:\n",
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
                "def process_split_with_arrays(pairs: list, split: str) -> tuple[int, int]:\n",
                "    img_n, lbl_n = 0, 0\n",
                "    for idx, (img_p, mask_arr) in enumerate(pairs):\n",
                "        if isinstance(img_p, Path):\n",
                "            img_arr = np.array(Image.open(img_p).convert('RGB'))\n",
                "            img_data = Image.fromarray(img_arr)\n",
                "            new_name = f'{split}_{idx:05d}{img_p.suffix.lower()}'\n",
                "        else:\n",
                "            img_data = Image.fromarray(img_p)\n",
                "            new_name = f'{split}_{idx:05d}.png'\n",
                "        dst_img = PREP_DIR / 'images' / split / new_name\n",
                "        dst_lbl = PREP_DIR / 'labels' / split / f'{Path(new_name).stem}.txt'\n",
                "        img_data.save(dst_img)\n",
                "        segs = mask_to_yolo_segs_from_array(mask_arr)\n",
                "        with open(dst_lbl, 'w', encoding='utf-8') as f:\n",
                "            if segs:\n",
                "                f.write('\\n'.join(segs))\n",
                "        img_n += 1\n",
                "        if segs:\n",
                "            lbl_n += 1\n",
                "    return img_n, lbl_n\n",
                "\n",
                "tr_n, tr_lbl = process_split_with_arrays(train_pairs, 'train')\n",
                "va_n, va_lbl = process_split_with_arrays(val_pairs, 'val')\n",
                "\n",
                "if tr_n == 0 or va_n == 0:\n",
                "    raise RuntimeError('Empty split.')\n",
                "if tr_lbl == 0 or va_lbl == 0:\n",
                "    raise RuntimeError('No labeled images.')\n",
                "\n",
                "print(f'Train: {tr_n} imgs, {tr_lbl} labeled')\n",
                "print(f'Val: {va_n} imgs, {va_lbl} labeled')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-014", "language": "python"},
            "source": [
                "data_yaml = PREP_DIR / 'data.yaml'\n",
                "with open(data_yaml, 'w', encoding='utf-8') as f:\n",
                "    yaml.safe_dump({\n",
                "        'path': str(PREP_DIR),\n",
                "        'train': 'images/train',\n",
                "        'val': 'images/val',\n",
                "        'names': {0: 'nucleus'}\n",
                "    }, f, sort_keys=False)\n",
                "\n",
                "print(f'Data yaml: {data_yaml}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-015", "language": "markdown"},
            "source": [
                "## Train YOLO26m-seg"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-016", "language": "python"},
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
                "    imgsz=416,\n",
                "    batch=4,\n",
                "    project=str(ARTIFACTS_DIR / 'yolo_runs'),\n",
                "    name='cell_nuclei_seg',\n",
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
            "metadata": {"id": "cn32-017", "language": "markdown"},
            "source": [
                "## Real Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-018", "language": "python"},
            "source": [
                "best = YOLO(str(best_model_path))\n",
                "val_res = best.val(data=str(data_yaml), split='val', imgsz=416, batch=4, verbose=False)\n",
                "\n",
                "map50 = float(val_res.seg.map50)\n",
                "map5095 = float(val_res.seg.map)\n",
                "prec = float(val_res.seg.mp)\n",
                "rec = float(val_res.seg.mr)\n",
                "\n",
                "print(f'Cell Nuclei Segmentation Results:')\n",
                "print(f'  mAP50: {map50:.4f}')\n",
                "print(f'  mAP50-95: {map5095:.4f}')\n",
                "print(f'  Precision: {prec:.4f}')\n",
                "print(f'  Recall: {rec:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-019", "language": "markdown"},
            "source": [
                "## Qualitative Predictions on Validation Set"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-020", "language": "python"},
            "source": [
                "val_imgs = sorted((PREP_DIR / 'images' / 'val').iterdir())\n",
                "sample_indices = list(range(min(6, len(val_imgs))))\n",
                "preds = best.predict([str(val_imgs[i]) for i in sample_indices], imgsz=416, conf=0.25, verbose=False)\n",
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
                "qual_path = ARTIFACTS_DIR / 'nuclei_qualitative.png'\n",
                "plt.savefig(qual_path, dpi=140)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "cn32-021", "language": "markdown"},
            "source": [
                "## Save Real Outputs"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cn32-022", "language": "python"},
            "source": [
                "metrics = {\n",
                "    'dataset': 'data-science-bowl-2018',\n",
                "    'dataset_url': 'https://www.kaggle.com/competitions/data-science-bowl-2018/data',\n",
                "    'task': 'cell_nuclei_segmentation',\n",
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
            "metadata": {"id": "cn32-023", "language": "markdown"},
            "source": [
                "## Limitations\n",
                "\n",
                "- Contour-based label conversion may miss very small or overlapping nuclei.",
                "- Training is intentionally short for local runtime.",
                "- Real DSB 2018 scores require more training epochs and data augmentation.",
                "- Consider instance segmentation postprocessing to separate touching nuclei."
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

out_path = Path(r'e:\Github\Machine-Learning-Projects\Computer Vision\Cell Nuclei Segmentation\Source Code\cell_nuclei_segmentation_pipeline.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'Wrote: {out_path}')
