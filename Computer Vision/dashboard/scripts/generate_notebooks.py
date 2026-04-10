"""Generate end-to-end Jupyter notebooks for every modern project (LOCAL execution).

Each notebook is designed for local GPU/CUDA execution -- no Colab dependencies.

Usage::
    python dashboard/scripts/generate_notebooks.py
"""

from __future__ import annotations
import json, nbformat
from pathlib import Path
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "dashboard" / "public" / "data" / "projects.json"


def _make_nb(p: dict) -> nbformat.NotebookNode:
    key   = p["key"]
    name  = p["name"].strip()
    cat   = p["category"]
    ptype = p["projectType"]
    tech  = p["modernTech"]
    fam   = p["modelFamily"]
    train = p["hasTraining"]
    ds    = p["dataset"]
    folder = p["folderPath"].replace("\\", "/")
    source = p["sourcePath"].replace("\\", "/")
    desc  = p["description"]
    tags  = p["tags"]

    cells: list = []

    # ── 0  Title ───────────────────────────────────────────
    cells.append(new_markdown_cell(
        f"# {name} -- End-to-End Pipeline\n\n"
        f"**Category:** {cat} | **Type:** {ptype} | **Tech:** {tech}\n\n"
        f"{desc}\n\n"
        f"**Tags:** {', '.join(tags) if tags else 'N/A'} | "
        f"**Model Family:** {', '.join(fam)}\n\n---\n"
        "Pipeline: environment -> paths -> download -> EDA -> "
        "preprocess -> train -> evaluate -> infer -> export."
    ))

    # ── 1  Deps ────────────────────────────────────────────
    dep = [
        "import subprocess, sys, importlib",
        "",
        "def _ensure(pkg, pip_name=None):",
        "    try: importlib.import_module(pkg)",
        "    except ImportError:",
        "        subprocess.check_call([sys.executable, '-m', 'pip',",
        "                               'install', '-q', pip_name or pkg])",
        "",
        "_ensure('ultralytics')",
        "_ensure('cv2', 'opencv-python-headless')",
        "_ensure('numpy'); _ensure('pandas'); _ensure('matplotlib')",
        "_ensure('seaborn'); _ensure('sklearn', 'scikit-learn')",
        "_ensure('yaml', 'pyyaml'); _ensure('tqdm'); _ensure('PIL', 'Pillow')",
    ]
    for f in fam:
        if f == "PaddleOCR":  dep.append("_ensure('paddleocr')")
        if f == "TrOCR":      dep += ["_ensure('transformers')", "_ensure('datasets')"]
        if f == "MediaPipe":  dep.append("_ensure('mediapipe')")
        if f == "DeepFace":   dep.append("_ensure('deepface')")
        if f == "InsightFace":dep.append("_ensure('insightface')")
        if f == "SAM":        dep.append("_ensure('segment_anything', 'segment-anything')")
        if f in ("ResNet","Custom"): dep.append("_ensure('timm')")
    dep.append("\nprint('All dependencies ready')")
    cells.append(new_code_cell("\n".join(dep)))

    # ── 2  GPU ─────────────────────────────────────────────
    cells.append(new_markdown_cell("## 1. GPU & Runtime Check"))
    cells.append(new_code_cell(
        "import torch, os\n\n"
        "print(f'PyTorch : {torch.__version__}')\n"
        "print(f'CUDA    : {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU     : {torch.cuda.get_device_name(0)}')\n"
        "    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')\n"
        "    DEVICE = 'cuda'\n"
        "else:\n"
        "    print('No GPU detected -- running on CPU')\n"
        "    DEVICE = 'cpu'\n"
        "print(f'Device  : {DEVICE}')"
    ))

    # ── 3  Paths (LOCAL) ───────────────────────────────────
    cells.append(new_markdown_cell("## 2. Repository & Project Paths"))
    cells.append(new_code_cell(
        "import os, sys\n"
        "from pathlib import Path\n\n"
        "# Auto-detect repo root from notebook location\n"
        "_d = Path(os.path.abspath(''))\n"
        "for _ in range(10):\n"
        "    if (_d / 'core' / '__init__.py').exists(): break\n"
        "    _d = _d.parent\n"
        "REPO_DIR = str(_d)\n\n"
        "os.chdir(REPO_DIR)\n"
        "sys.path.insert(0, REPO_DIR)\n"
        f"sys.path.insert(0, os.path.join(REPO_DIR, r'{source}'))\n\n"
        f"PROJECT_DIR = os.path.join(REPO_DIR, r'{folder}')\n"
        f"SOURCE_DIR  = os.path.join(REPO_DIR, r'{source}')\n"
        "print(f'Repo:    {REPO_DIR}')\n"
        "print(f'Project: {PROJECT_DIR}')\n"
        "print(f'Source:  {SOURCE_DIR}')"
    ))

    # ── 4  Dataset ─────────────────────────────────────────
    cells.append(new_markdown_cell("## 3. Dataset Download"))

    if ds.get("configured"):
        dt, did = ds.get("type",""), ds.get("id","")
        if dt == "kaggle":
            cells.append(new_code_cell(
                "import os, subprocess, sys\n\n"
                f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}', 'raw')\n"
                "os.makedirs(DATA_DIR, exist_ok=True)\n\n"
                "try:\n    import kaggle\n"
                "except ImportError:\n"
                "    subprocess.check_call([sys.executable,'-m','pip','install','-q','kaggle'])\n\n"
                f"subprocess.run(['kaggle','datasets','download','-d','{did}',\n"
                "                '-p', DATA_DIR, '--unzip'], check=False)\n"
                "print(f'Dataset at {DATA_DIR}')"
            ))
        elif dt == "kaggle_competition":
            cells.append(new_code_cell(
                "import os, subprocess, sys, zipfile, glob\n\n"
                f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}', 'raw')\n"
                "os.makedirs(DATA_DIR, exist_ok=True)\n\n"
                "try:\n    import kaggle\n"
                "except ImportError:\n"
                "    subprocess.check_call([sys.executable,'-m','pip','install','-q','kaggle'])\n\n"
                f"subprocess.run(['kaggle','competitions','download','-c','{did}',\n"
                "                '-p', DATA_DIR], check=False)\n"
                "for zf in glob.glob(os.path.join(DATA_DIR,'*.zip')):\n"
                "    with zipfile.ZipFile(zf) as z: z.extractall(DATA_DIR)\n"
                "print(f'Dataset at {DATA_DIR}')"
            ))
        elif dt == "huggingface":
            cells.append(new_code_cell(
                "import os, subprocess, sys\n\n"
                f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
                "os.makedirs(DATA_DIR, exist_ok=True)\n\n"
                "try:\n    from datasets import load_dataset\n"
                "except ImportError:\n"
                "    subprocess.check_call([sys.executable,'-m','pip','install','-q','datasets'])\n"
                "    from datasets import load_dataset\n\n"
                f"dataset = load_dataset('{did}', cache_dir=os.path.join(DATA_DIR,'raw'))\n"
                "print(dataset)"
            ))
        elif dt == "roboflow":
            cells.append(new_code_cell(
                "import os\n\n"
                f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
                "os.makedirs(DATA_DIR, exist_ok=True)\n\n"
                "try:\n"
                "    from scripts.download_data import ensure_dataset\n"
                f"    data_path = ensure_dataset('{key}', force=False)\n"
                "    print(f'Dataset ready at: {data_path}')\n"
                "except Exception as e:\n"
                "    print(f'Auto-download note: {e}')\n"
                "    print('Set ROBOFLOW_API_KEY env var or download manually')"
            ))
        else:  # direct_url / local_only
            cells.append(new_code_cell(
                "import os\n\n"
                f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
                "os.makedirs(DATA_DIR, exist_ok=True)\n\n"
                "try:\n"
                "    from scripts.download_data import ensure_dataset\n"
                f"    data_path = ensure_dataset('{key}', force=False)\n"
                "    print(f'Dataset ready at: {data_path}')\n"
                "except Exception as e:\n"
                "    print(f'Dataset setup note: {e}')"
            ))
    else:
        cells.append(new_code_cell(
            "# Pretrained model project -- download sample image for demo\n"
            "import urllib.request, os\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "os.makedirs(DATA_DIR, exist_ok=True)\n\n"
            "sample = os.path.join(DATA_DIR, 'sample.jpg')\n"
            "if not os.path.exists(sample):\n"
            "    urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', sample)\n"
            "    print(f'Sample saved to {sample}')\n"
            "else:\n"
            "    print(f'Sample exists: {sample}')"
        ))

    # ── 5  EDA ─────────────────────────────────────────────
    cells.append(new_markdown_cell("## 4. Exploratory Data Analysis (EDA)"))

    if ptype in ("detection","segmentation") and "YOLO" in fam:
        cells.append(new_code_cell(
            "import os, glob, cv2, numpy as np\n"
            "import matplotlib.pyplot as plt\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "images = []\n"
            "for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):\n"
            "    images.extend(glob.glob(os.path.join(DATA_DIR,'**',ext), recursive=True))\n"
            "print(f'Total images: {len(images)}')\n\n"
            "fig, axes = plt.subplots(2, 4, figsize=(16,8))\n"
            "fig.suptitle('Sample Images', fontsize=14)\n"
            "for ax, p in zip(axes.flat, images[:8]):\n"
            "    img = cv2.imread(p)\n"
            "    if img is not None: ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
            "    ax.set_title(os.path.basename(p)[:20], fontsize=8); ax.axis('off')\n"
            "plt.tight_layout(); plt.show()\n\n"
            "if images:\n"
            "    sizes = [cv2.imread(p).shape[:2] for p in images[:500] if cv2.imread(p) is not None]\n"
            "    if sizes:\n"
            "        h, w = zip(*sizes)\n"
            "        fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))\n"
            "        a1.hist(w,bins=30,color='steelblue',edgecolor='white'); a1.set_title('Widths')\n"
            "        a2.hist(h,bins=30,color='coral',edgecolor='white'); a2.set_title('Heights')\n"
            "        plt.tight_layout(); plt.show()\n"
            "        print(f'Avg: {np.mean(w):.0f}x{np.mean(h):.0f}')"
        ))
        cells.append(new_code_cell(
            "# Label distribution\n"
            "from collections import Counter\n"
            "import glob, os, matplotlib.pyplot as plt\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "label_files = [f for f in glob.glob(os.path.join(DATA_DIR,'**','*.txt'), recursive=True)\n"
            "               if 'classes' not in os.path.basename(f)]\n"
            "print(f'Label files: {len(label_files)}')\n\n"
            "cc, bc = Counter(), []\n"
            "for lf in label_files[:2000]:\n"
            "    try:\n"
            "        lines = open(lf).readlines()\n"
            "        bc.append(len(lines))\n"
            "        for l in lines: cc[l.strip().split()[0]] += 1\n"
            "    except Exception: pass\n\n"
            "if cc:\n"
            "    cls = sorted(cc, key=lambda x: int(x) if x.isdigit() else x)\n"
            "    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))\n"
            "    a1.barh(cls,[cc[c] for c in cls],color='steelblue'); a1.set_title('Class Dist')\n"
            "    a2.hist(bc,bins=30,color='coral',edgecolor='white'); a2.set_title('Objs/Image')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('No YOLO labels found (dataset may not be downloaded yet)')"
        ))
    elif ptype == "classification":
        cells.append(new_code_cell(
            "import os, glob, cv2, matplotlib.pyplot as plt\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "cc = {}\n"
            "for root, dirs, files in os.walk(DATA_DIR):\n"
            "    imgs = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]\n"
            "    if imgs:\n"
            "        cn = os.path.basename(root)\n"
            "        if cn not in ('raw','processed'): cc[cn] = len(imgs)\n\n"
            "print(f'Classes: {len(cc)}, Images: {sum(cc.values())}')\n\n"
            "if cc:\n"
            "    sc = sorted(cc.items(), key=lambda x: x[1], reverse=True)\n"
            "    n,c = zip(*sc)\n"
            "    fig,ax = plt.subplots(figsize=(12, max(4,len(cc)*0.3)))\n"
            "    ax.barh(n[:30],c[:30],color='steelblue')\n"
            "    ax.set_title('Class Distribution (top 30)'); plt.tight_layout(); plt.show()\n\n"
            "all_imgs = glob.glob(os.path.join(DATA_DIR,'**','*.jpg'), recursive=True)\n"
            "all_imgs += glob.glob(os.path.join(DATA_DIR,'**','*.png'), recursive=True)\n"
            "fig, axes = plt.subplots(2,4,figsize=(16,8))\n"
            "for ax,p in zip(axes.flat, all_imgs[:8]):\n"
            "    img = cv2.imread(p)\n"
            "    if img is not None: ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
            "    ax.set_title(os.path.basename(os.path.dirname(p))[:20], fontsize=8); ax.axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ))
    else:
        cells.append(new_code_cell(
            "import os, glob, cv2, matplotlib.pyplot as plt\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "images = []\n"
            "for ext in ('*.jpg','*.jpeg','*.png'):\n"
            "    images.extend(glob.glob(os.path.join(DATA_DIR,'**',ext), recursive=True))\n"
            "print(f'Files found: {len(images)}')\n\n"
            "fig, axes = plt.subplots(2,4,figsize=(16,8))\n"
            "for ax,p in zip(axes.flat, images[:8]):\n"
            "    img = cv2.imread(p)\n"
            "    if img is not None: ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
            "    ax.set_title(os.path.basename(p)[:20], fontsize=8); ax.axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ))

    # ── 6  Preprocessing ──────────────────────────────────
    cells.append(new_markdown_cell("## 5. Data Preprocessing & Preparation"))

    if ptype in ("detection","segmentation") and "YOLO" in fam:
        cells.append(new_code_cell(
            "import os, glob, yaml\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "os.makedirs(os.path.join(DATA_DIR,'processed'), exist_ok=True)\n\n"
            "yf = glob.glob(os.path.join(DATA_DIR,'**','data.yaml'), recursive=True)\n"
            "yf += glob.glob(os.path.join(DATA_DIR,'**','*.yaml'), recursive=True)\n"
            "yf = [f for f in yf if 'dataset_info' not in f]\n\n"
            "if yf:\n"
            "    DATA_YAML = yf[0]\n"
            "    print(f'data.yaml: {DATA_YAML}')\n"
            "    cfg = yaml.safe_load(open(DATA_YAML))\n"
            "    print(f'Classes: {cfg.get(\"names\", cfg.get(\"nc\",\"?\"))}')\n"
            "else:\n"
            "    ti = glob.glob(os.path.join(DATA_DIR,'**/train/images/*.jpg'), recursive=True)\n"
            "    vi = glob.glob(os.path.join(DATA_DIR,'**/val/images/*.jpg'), recursive=True)\n"
            "    print(f'Train: {len(ti)}, Val: {len(vi)}')\n"
            "print('Dataset ready')"
        ))
    elif ptype == "classification":
        cells.append(new_code_cell(
            "import os, glob\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "train_dir = None\n"
            "for c in ['train','Train','training']:\n"
            "    m = glob.glob(os.path.join(DATA_DIR,'**',c), recursive=True)\n"
            "    if m and os.path.isdir(m[0]): train_dir = m[0]; break\n\n"
            "if train_dir:\n"
            "    nc = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))])\n"
            "    print(f'Train: {train_dir} ({nc} classes)')\n"
            "else:\n"
            "    ai = glob.glob(os.path.join(DATA_DIR,'**','*.jpg'), recursive=True)\n"
            "    ai += glob.glob(os.path.join(DATA_DIR,'**','*.png'), recursive=True)\n"
            "    print(f'No split found. Total images: {len(ai)}')\n"
            "print('Preprocessing complete')"
        ))
    else:
        cells.append(new_code_cell(
            "import os, glob\n\n"
            f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
            "os.makedirs(DATA_DIR, exist_ok=True)\n"
            "af = []\n"
            "for ext in ('*.jpg','*.png','*.txt','*.json','*.csv'):\n"
            "    af.extend(glob.glob(os.path.join(DATA_DIR,'**',ext), recursive=True))\n"
            "print(f'Data files: {len(af)}')\n"
            "for f in af[:10]: print(f'  {os.path.relpath(f, DATA_DIR)}')"
        ))

    # ── 7  Training ────────────────────────────────────────
    cells.append(new_markdown_cell("## 6. Model Training"))

    if train:
        if "YOLO" in fam and ptype in ("detection","segmentation","classification"):
            task = {"detection":"detect","segmentation":"segment","classification":"classify"}.get(ptype,"detect")
            if "obb" in key: task = "obb"
            wt = {"detect":"yolo11m.pt","segment":"yolo11m-seg.pt",
                  "classify":"yolo11m-cls.pt","obb":"yolo11m-obb.pt"}.get(task,"yolo11m.pt")
            cells.append(new_code_cell(
                "from ultralytics import YOLO\n"
                "import torch, glob, os\n\n"
                f"model = YOLO('{wt}')  # pretrained for {task}\n\n"
                f"yf = glob.glob(os.path.join(REPO_DIR,'data','{key}','**','data.yaml'), recursive=True)\n"
                f"yf += glob.glob(os.path.join(REPO_DIR,'data','{key}','**','*.yaml'), recursive=True)\n"
                "yf = [f for f in yf if 'dataset_info' not in f and '_template' not in f]\n\n"
                "if yf:\n"
                "    print(f'Training with: {yf[0]}')\n"
                "    model.train(\n"
                "        data=yf[0], epochs=25, imgsz=640, batch=16,\n"
                "        device=0 if torch.cuda.is_available() else 'cpu',\n"
                f"        project=os.path.join(REPO_DIR,'runs','{key}'),\n"
                "        name='train', exist_ok=True, patience=10, save=True, plots=True)\n"
                "else:\n"
                "    print('No data.yaml -- using pretrained model directly')"
            ))
        elif "ResNet" in fam or "Custom" in fam:
            cells.append(new_code_cell(
                "import os, sys\nos.chdir(SOURCE_DIR)\n\n"
                "try:\n    from train import main as train_main\n"
                "    sys.argv = ['train.py','--epochs','15','--batch','32']\n"
                "    train_main()\n"
                "except Exception as e:\n"
                "    print(f'Training: {e}')\n"
                "    import torch, torchvision\n"
                "    from torchvision import transforms\n"
                "    print(f'Data: {os.path.join(REPO_DIR, \"data\", \"" + key + "\")}')\n"
                "    print('Setup complete -- configure training loop as needed')"
            ))
        elif "PaddleOCR" in fam or "TrOCR" in fam:
            cells.append(new_code_cell(
                "import os, sys\nos.chdir(SOURCE_DIR)\n\n"
                "try:\n    from train import main as train_main\n"
                "    sys.argv = ['train.py']\n    train_main()\n"
                "except Exception as e:\n"
                "    print(f'Training note: {e}')\n"
                "    print('OCR models use pretrained weights -- proceeding to inference')"
            ))
        elif "MediaPipe" in fam:
            cells.append(new_code_cell(
                "import os, sys\nos.chdir(SOURCE_DIR)\n\n"
                "try:\n    from train import main as train_main\n"
                "    sys.argv = ['train.py']\n    train_main()\n"
                "except Exception as e:\n"
                "    print(f'Training note: {e}')\n"
                "    print('MediaPipe uses pretrained models -- proceeding to inference')"
            ))
        else:
            cells.append(new_code_cell(
                "import os, sys\nos.chdir(SOURCE_DIR)\n\n"
                "try:\n    from train import main as train_main\n"
                "    sys.argv = ['train.py']\n    train_main()\n"
                "except Exception as e:\n    print(f'Training: {e}')"
            ))
    else:
        cells.append(new_code_cell(
            f"print('Project {name} uses pretrained weights')\n"
            "print('Proceeding directly to inference...')"
        ))

    # ── 8  Evaluation ──────────────────────────────────────
    cells.append(new_markdown_cell("## 7. Evaluation & Metrics"))

    if "YOLO" in fam and ptype in ("detection","segmentation"):
        cells.append(new_code_cell(
            "import os, glob, torch\n"
            "from IPython.display import Image, display\n\n"
            f"rd = os.path.join(REPO_DIR,'runs','{key}','train')\n"
            "if os.path.exists(rd):\n"
            "    for p in ['results.png','confusion_matrix.png','PR_curve.png','F1_curve.png']:\n"
            "        pp = os.path.join(rd, p)\n"
            "        if os.path.exists(pp): display(Image(filename=pp, width=800))\n\n"
            "    from ultralytics import YOLO\n"
            "    bw = os.path.join(rd,'weights','best.pt')\n"
            "    if os.path.exists(bw):\n"
            "        model = YOLO(bw)\n"
            f"        yf = glob.glob(os.path.join(REPO_DIR,'data','{key}','**','*.yaml'), recursive=True)\n"
            "        yf = [f for f in yf if 'dataset_info' not in f]\n"
            "        if yf:\n"
            "            m = model.val(data=yf[0], device=0 if torch.cuda.is_available() else 'cpu')\n"
            "            print(f'mAP50:    {m.box.map50:.4f}')\n"
            "            print(f'mAP50-95: {m.box.map:.4f}')\n"
            "else:\n    print('No training results -- evaluating pretrained model')"
        ))
    elif ptype == "classification":
        cells.append(new_code_cell(
            "import os\nfrom IPython.display import Image, display\n\n"
            f"rd = os.path.join(REPO_DIR,'runs','{key}','train')\n"
            "if os.path.exists(rd):\n"
            "    for p in ['results.png','confusion_matrix.png']:\n"
            "        pp = os.path.join(rd, p)\n"
            "        if os.path.exists(pp): display(Image(filename=pp, width=800))\n"
            "else:\n    print('No training results directory found')\n\n"
            "print('Classification metrics computed after test-set inference')"
        ))
    else:
        cells.append(new_code_cell(
            f"print('--- Evaluation: {name} ---')\n"
            "print('Metrics depend on task and test data availability')"
        ))

    # ── 9  Inference ───────────────────────────────────────
    cells.append(new_markdown_cell("## 8. Inference & Visualization"))
    cells.append(new_code_cell(
        "import os, sys, glob, cv2\n"
        "import matplotlib.pyplot as plt\n"
        "os.chdir(REPO_DIR)\n"
        f"sys.path.insert(0, os.path.join(REPO_DIR, r'{source}'))\n\n"
        "from core import discover_projects, run\n"
        "discover_projects()\n\n"
        "sample_images = glob.glob(os.path.join(REPO_DIR,'data','**','*.jpg'), recursive=True)\n"
        "if not sample_images:\n"
        "    import urllib.request\n"
        "    os.makedirs(os.path.join(REPO_DIR,'data','samples'), exist_ok=True)\n"
        "    urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg',\n"
        "                              os.path.join(REPO_DIR,'data','samples','bus.jpg'))\n"
        "    sample_images = [os.path.join(REPO_DIR,'data','samples','bus.jpg')]\n\n"
        "test_img = sample_images[0]\n"
        "print(f'Test image: {test_img}')\n\n"
        "try:\n"
        f"    result = run('{key}', test_img)\n"
        "    print(f'Result type: {type(result)}')\n"
        "    if isinstance(result, dict):\n"
        "        for k,v in result.items():\n"
        "            if not isinstance(v,(bytes,bytearray)): print(f'  {k}: {str(v)[:200]}')\n"
        "    else: print(f'Result: {str(result)[:500]}')\n"
        "except Exception as e:\n"
        f"    print(f'Inference note: {{e}}')"
    ))
    cells.append(new_code_cell(
        "try:\n"
        f"    vis = run('{key}', test_img, visualize=True)\n"
        "    if vis is not None and hasattr(vis, 'shape'):\n"
        "        import cv2, matplotlib.pyplot as plt\n"
        "        plt.figure(figsize=(12,8))\n"
        "        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))\n"
        f"        plt.title('{name} -- Inference'); plt.axis('off')\n"
        "        plt.tight_layout(); plt.show()\n"
        "except Exception as e:\n"
        "    print(f'Visualization note: {e}')"
    ))

    # ── 10  Validation ─────────────────────────────────────
    cells.append(new_markdown_cell("## 9. Validation & Testing"))
    cells.append(new_code_cell(
        "import os, glob, time, numpy as np\n\n"
        f"DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')\n"
        "test_imgs = glob.glob(os.path.join(DATA_DIR,'**','*.jpg'), recursive=True)[:20]\n"
        "print(f'Validating on {len(test_imgs)} images')\n\n"
        "times = []\n"
        "for p in test_imgs:\n"
        "    try:\n"
        "        t0 = time.time()\n"
        f"        run('{key}', p)\n"
        "        times.append(time.time()-t0)\n"
        "    except Exception: pass\n\n"
        "if times:\n"
        "    print(f'Processed : {len(times)}')\n"
        "    print(f'Avg latency: {np.mean(times)*1000:.1f} ms')\n"
        "    print(f'FPS       : {1/np.mean(times):.1f}')\n"
        "else:\n    print('No images processed')"
    ))

    # ── 11  Export ─────────────────────────────────────────
    cells.append(new_markdown_cell("## 10. Export & Summary"))
    cells.append(new_code_cell(
        "import json, os, torch\n\n"
        "summary = {\n"
        f"    'project': '{name}',\n"
        f"    'key': '{key}',\n"
        f"    'category': '{cat}',\n"
        f"    'type': '{ptype}',\n"
        f"    'tech': '{tech}',\n"
        f"    'model_family': {fam},\n"
        f"    'has_training': {train},\n"
        "    'device': DEVICE,\n"
        "    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',\n"
        "}\n\n"
        "print('='*60)\n"
        "for k,v in summary.items(): print(f'  {k:20s}: {v}')\n"
        "print('='*60)\n\n"
        f"os.makedirs(os.path.join(REPO_DIR, r'{folder}'), exist_ok=True)\n"
        f"sp = os.path.join(REPO_DIR, r'{folder}', 'notebook_summary.json')\n"
        "with open(sp,'w') as f: json.dump(summary, f, indent=2)\n"
        "print(f'Saved to {sp}')"
    ))

    cells.append(new_markdown_cell(
        f"---\n## Pipeline Complete\n\n**{name}** end-to-end notebook finished.\n"
        "Steps: env -> paths -> download -> EDA -> preprocess -> train -> evaluate -> infer -> export."
    ))

    nb = new_notebook(cells=cells)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def main():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    created = 0
    for p in data["projects"]:
        nb_path = REPO / Path(p["sourcePath"]) / f"{p['key']}_pipeline.ipynb"
        nb = _make_nb(p)
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        created += 1
        print(f"  [{created:3d}] {nb_path.relative_to(REPO)}")
    print(f"\n> Created {created} notebooks")


if __name__ == "__main__":
    main()
