#!/usr/bin/env python3
"""
Patch all 50 run.py files to integrate parse_common_args() CLI.

Adds: --smoke-test, --epochs, --batch-size, --num-workers, --device,
      --download-only, --no-amp

Usage:  python scripts/patch_cli.py
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def w(rel: str, text: str):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"  + {rel}")


# ═══════════════════════════════════════════════════════════════════════════════
# Templates with CLI support
# ═══════════════════════════════════════════════════════════════════════════════

CV_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Model   : {model} (timm)
Task    : Image Classification

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check (1 epoch, 2 batches)
    python run.py --download-only
    python run.py --epochs 5 --batch-size 64 --device cuda
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.utils import (seed_everything, get_device, dataset_prompt,
                           kaggle_download, ensure_dir, parse_common_args)
from shared.cv import (create_dataloaders, build_timm_model,
                        train_model, evaluate_model)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
MODEL   = {model_repr}
CLASSES = {num_classes}
EPOCHS  = {epochs}
BATCH   = 32


def get_data():
    """Download and prepare the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    args = parse_common_args()
    seed_everything(42)
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    data_root = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    epochs     = args.epochs or EPOCHS
    batch_size = args.batch_size or BATCH
    num_workers = args.num_workers if args.num_workers is not None else 4
    use_amp    = not args.no_amp
    max_batches = 2 if args.smoke_test else None
    if args.smoke_test:
        epochs = 1

    train_dl, val_dl, test_dl, class_names = create_dataloaders(
        data_root, img_size=224, batch_size=batch_size,
        num_workers=num_workers,
    )
    model = build_timm_model(MODEL, num_classes=CLASSES or len(class_names))
    model = train_model(
        model, train_dl, val_dl,
        epochs=epochs, lr=1e-4, device=device, output_dir=OUTPUT_DIR,
        use_amp=use_amp, max_batches=max_batches,
    )
    evaluate_model(model, test_dl, class_names, device=device,
                   output_dir=OUTPUT_DIR, max_batches=max_batches)


if __name__ == "__main__":
    main()
'''

TAB_CLS_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Task    : Tabular Classification (PyCaret AutoML)

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check
    python run.py --download-only
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, dataset_prompt,
                           kaggle_download, ensure_dir, parse_common_args,
                           save_metrics)
from shared.tabular import run_pycaret_classification

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
TARGET  = {target_repr}


def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    args = parse_common_args()
    seed_everything(42)
    ensure_dir(OUTPUT_DIR)

    df = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    print(f"  Shape: {{df.shape}}  |  Target: {{TARGET}}")
    print(f"  Classes: {{df[TARGET].nunique()}}")

    if args.smoke_test:
        df = df.head(min(200, len(df)))
        print(f"  [SMOKE] Using first {{len(df)}} rows only.")

    run_pycaret_classification(df, target=TARGET, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

TAB_REG_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Task    : Tabular Regression (PyCaret AutoML)

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check
    python run.py --download-only
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, dataset_prompt,
                           kaggle_download, ensure_dir, parse_common_args,
                           save_metrics)
from shared.tabular import run_pycaret_regression

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
TARGET  = {target_repr}


def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    args = parse_common_args()
    seed_everything(42)
    ensure_dir(OUTPUT_DIR)

    df = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    print(f"  Shape: {{df.shape}}  |  Target: {{TARGET}}")

    if args.smoke_test:
        df = df.head(min(200, len(df)))
        print(f"  [SMOKE] Using first {{len(df)}} rows only.")

    run_pycaret_regression(df, target=TARGET, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

TAB_CLUST_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Task    : Tabular Clustering (PyCaret)

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check
    python run.py --download-only
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, dataset_prompt,
                           kaggle_download, ensure_dir, parse_common_args)
from shared.tabular import run_pycaret_clustering

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}


def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    args = parse_common_args()
    seed_everything(42)
    ensure_dir(OUTPUT_DIR)

    df = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    print(f"  Shape: {{df.shape}}")

    if args.smoke_test:
        df = df.head(min(200, len(df)))
        print(f"  [SMOKE] Using first {{len(df)}} rows only.")

    run_pycaret_clustering(df, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

NLP_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Model   : {hf_model} (HuggingFace)
Task    : Text Classification

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check (4 train steps)
    python run.py --download-only
    python run.py --epochs 5 --batch-size 32 --device cuda
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, get_device, dataset_prompt,
                           kaggle_download, ensure_dir, parse_common_args)
from shared.nlp import (build_hf_classifier, tokenize_texts,
                         train_hf_classifier, evaluate_hf_classifier)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
HF_MODEL = {hf_model_repr}
NUM_LABELS = {num_labels}
EPOCHS = 3
BATCH  = 16


def get_data():
    """Download, load and split text data.

    Returns (train_texts, train_labels, test_texts, test_labels, class_names).
    """
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    args = parse_common_args()
    seed_everything(42)
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    train_texts, train_labels, test_texts, test_labels, class_names = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    print(f"  Train: {{len(train_texts)}}  |  Test: {{len(test_texts)}}  "
          f"|  Classes: {{class_names}}")

    epochs     = args.epochs or EPOCHS
    batch_size = args.batch_size or BATCH
    use_amp    = not args.no_amp
    max_steps  = 4 if args.smoke_test else -1

    if args.smoke_test:
        train_texts  = train_texts[:100]
        train_labels = train_labels[:100]
        test_texts   = test_texts[:50]
        test_labels  = test_labels[:50]
        epochs = 1

    model, tokenizer = build_hf_classifier(HF_MODEL, num_labels=NUM_LABELS or len(class_names))
    train_ds = tokenize_texts(train_texts, train_labels, tokenizer)
    test_ds  = tokenize_texts(test_texts, test_labels, tokenizer)

    trainer = train_hf_classifier(
        model, train_ds, test_ds, OUTPUT_DIR,
        epochs=epochs, batch_size=batch_size, lr=2e-5,
        use_amp=use_amp, max_steps=max_steps,
    )
    evaluate_hf_classifier(trainer, test_ds, class_names, OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

README_TPL = '''\
# Project {num} -- {title}

## Dataset
{dataset}

{links_md}

## Stack
| Component | Choice |
|-----------|--------|
| Framework | PyTorch 2.10.0 (cu130) |
| Model     | {model_desc} |
| Task      | {task_desc} |
| AutoML    | {automl_desc} |

## Usage

```bash
# 1. Install PyTorch (once)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 2. Install dependencies (once)
pip install -r requirements.txt

# 3. Run this project
python run.py

# CLI options
python run.py --smoke-test        # Quick sanity check (1 epoch, 2 batches)
python run.py --download-only     # Just download the dataset
python run.py --epochs 5          # Override number of epochs
python run.py --batch-size 64     # Override batch size
python run.py --num-workers 8     # Override data-loader workers
python run.py --device cpu        # Force CPU (default: auto-detect)
python run.py --no-amp            # Disable mixed precision
```

## Outputs
After training, check `outputs/` for:
- `best_model.pth` (or `.pkl` for tabular) -- saved model weights
- `metrics.json` -- accuracy, F1, etc.
- `metrics.md` -- same metrics as a Markdown table
- `training_curves.png` -- loss / accuracy over epochs
- `confusion_matrix.png` -- per-class performance
- `classification_report.txt` -- detailed metrics
'''


# ═══════════════════════════════════════════════════════════════════════════════
# Custom project sources (CLI-enabled)
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_P16 = '''\
#!/usr/bin/env python3
"""Project 16 -- Brain MRI Segmentation

Dataset : LGG MRI Segmentation
Model   : DeepLabV3-ResNet50 (torchvision)
Task    : Binary Segmentation

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 10 --batch-size 4 --device cuda
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from tqdm import tqdm
from shared.utils import (seed_everything, get_device, dataset_prompt,
                          kaggle_download, ensure_dir, save_metrics,
                          parse_common_args)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "mateuszbuda/lgg-mri-segmentation"
EPOCHS, LR, BATCH = 20, 1e-4, 8


class MRIDataset(Dataset):
    def __init__(self, images, masks, size=256):
        self.images, self.masks, self.size = images, masks, size
        self.tf = transforms.Compose([transforms.Resize((size, size)),
                                      transforms.ToTensor()])
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.tf(Image.open(self.images[i]).convert("RGB"))
        msk = self.tf(Image.open(self.masks[i]).convert("L"))
        return img, (msk > 0.5).float()


def get_data():
    dataset_prompt("LGG MRI Segmentation",
                   ["https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"])
    if not list(DATA_DIR.rglob("*_mask.tif")):
        kaggle_download(KAGGLE, DATA_DIR)
    imgs = sorted(DATA_DIR.rglob("*[!_mask].tif"))
    msks = sorted(DATA_DIR.rglob("*_mask.tif"))
    if not imgs:
        imgs = sorted(p for p in DATA_DIR.rglob("*.png") if "_mask" not in p.stem)
        msks = sorted(DATA_DIR.rglob("*_mask.png"))
    return imgs, msks


def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def main():
    args = parse_common_args()
    seed_everything(42)
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    imgs, msks = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    epochs      = args.epochs or EPOCHS
    batch_size  = args.batch_size or BATCH
    num_workers = args.num_workers if args.num_workers is not None else 2
    use_amp     = not args.no_amp
    max_batches = 2 if args.smoke_test else None
    if args.smoke_test:
        epochs = 1

    ds = MRIDataset(imgs, msks)
    n_val = int(0.15 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, 1, 1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_dice = 0.0

    for ep in range(epochs):
        model.train(); loss_sum = 0
        for bi, (X, y) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")):
            if max_batches and bi >= max_batches:
                break
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                out = model(X)["out"]
                out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear")
                loss = bce(out, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
        # val
        model.eval(); dsc = []
        with torch.no_grad():
            for bi, (X, y) in enumerate(val_dl):
                if max_batches and bi >= max_batches:
                    break
                X, y = X.to(device), y.to(device)
                out = torch.sigmoid(model(X)["out"])
                out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear")
                dsc.append(dice_score(out, y).item())
        mean_dice = np.mean(dsc) if dsc else 0.0
        print(f"  loss={loss_sum / max(bi + 1, 1):.4f}  dice={mean_dice:.4f}")
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    save_metrics({"best_dice": best_dice}, OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

CUSTOM_P23 = '''\
#!/usr/bin/env python3
"""Project 23 -- Sudoku Solver with Neural Network

Dataset : 1M Sudoku Games
Model   : Custom CNN (PyTorch)

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 5 --batch-size 128
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from shared.utils import (seed_everything, get_device, dataset_prompt,
                          kaggle_download, ensure_dir, save_metrics,
                          parse_common_args)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "bryanpark/sudoku"
EPOCHS, LR, BATCH = 10, 1e-3, 256


class SudokuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.head = nn.Conv2d(128, 9, 1)  # 9 classes per cell

    def forward(self, x):  # x: (B,1,9,9)
        return self.head(self.conv(x))  # (B,9,9,9)


def parse_sudoku(s):
    return np.array([int(c) for c in s]).reshape(9, 9).astype(np.float32)


def get_data(nrows=100_000):
    dataset_prompt("1M Sudoku Games", ["https://www.kaggle.com/datasets/bryanpark/sudoku"])
    csvs = list(DATA_DIR.rglob("*.csv"))
    if not csvs:
        kaggle_download(KAGGLE, DATA_DIR)
        csvs = list(DATA_DIR.rglob("*.csv"))
    df = pd.read_csv(csvs[0], nrows=nrows)
    quizzes = np.stack([parse_sudoku(q) for q in df.iloc[:, 0]])
    solutions = np.stack([parse_sudoku(s) for s in df.iloc[:, 1]])
    X = torch.from_numpy(quizzes / 9.0).unsqueeze(1)  # (N,1,9,9)
    y = torch.from_numpy(solutions).long() - 1        # classes 0-8
    n = len(X); split = int(0.9 * n)
    return (TensorDataset(X[:split], y[:split]),
            TensorDataset(X[split:], y[split:]))


def main():
    args = parse_common_args()
    seed_everything(42)
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    nrows = 1000 if args.smoke_test else 100_000
    train_ds, val_ds = get_data(nrows=nrows)
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    epochs      = args.epochs or EPOCHS
    batch_size  = args.batch_size or BATCH
    num_workers = args.num_workers if args.num_workers is not None else 2
    max_batches = 2 if args.smoke_test else None
    if args.smoke_test:
        epochs = 1

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size, num_workers=num_workers)

    model = SudokuNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0

    for ep in range(epochs):
        model.train(); total, correct = 0, 0
        for bi, (X, y) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")):
            if max_batches and bi >= max_batches:
                break
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
        # val
        model.eval(); vc, vt = 0, 0
        with torch.no_grad():
            for bi, (X, y) in enumerate(val_dl):
                if max_batches and bi >= max_batches:
                    break
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                vc += (pred == y).sum().item(); vt += y.numel()
        vacc = vc / max(vt, 1)
        print(f"  train_acc={correct / max(total, 1):.4f}  val_acc={vacc:.4f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    save_metrics({"best_cell_accuracy": best_acc}, OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

CUSTOM_P26 = '''\
#!/usr/bin/env python3
"""Project 26 -- Face Detection (OpenCV DNN)

Model  : OpenCV SSD face detector (pre-trained)
Task   : Inference-only face detection

Usage:
    python run.py --image path/to/image.jpg
    python run.py --smoke-test
"""
import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from shared.utils import ensure_dir, url_download, save_metrics

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def get_model():
    ensure_dir(DATA_DIR)
    proto = url_download(PROTO, DATA_DIR, "deploy.prototxt")
    model = url_download(MODEL, DATA_DIR, "face_model.caffemodel")
    return cv2.dnn.readNetFromCaffe(str(proto), str(model))


def detect_faces(net, image_path, conf=0.5):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104, 177, 123))
    net.setInput(blob)
    dets = net.forward()
    faces = []
    for i in range(dets.shape[2]):
        c = dets[0, 0, i, 2]
        if c > conf:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            faces.append((x1, y1, x2, y2, float(c)))
    return img, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    ensure_dir(OUTPUT_DIR)
    net = get_model()

    if args.download_only:
        print("  Model files ready. Exiting (--download-only).")
        return

    if args.smoke_test:
        # Create a tiny test image and run inference
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(OUTPUT_DIR / "smoke_test_input.jpg"), dummy)
        result, faces = detect_faces(net, OUTPUT_DIR / "smoke_test_input.jpg", conf=0.3)
        cv2.imwrite(str(OUTPUT_DIR / "smoke_test_output.jpg"), result)
        save_metrics({"faces_detected": len(faces), "smoke_test": True}, OUTPUT_DIR)
        print(f"  [SMOKE] Inference OK — {len(faces)} face(s) on dummy image.")
        return

    if args.image:
        result, faces = detect_faces(net, args.image)
        out_path = OUTPUT_DIR / "detected.jpg"
        cv2.imwrite(str(out_path), result)
        save_metrics({"faces_detected": len(faces)}, OUTPUT_DIR)
        print(f"  Found {len(faces)} face(s) -> {out_path}")
    else:
        print("  Pass --image <path> to run. Example:")
        print("    python run.py --image data/sample.jpg")


if __name__ == "__main__":
    main()
'''

CUSTOM_P41 = '''\
#!/usr/bin/env python3
"""Project 41 -- Cat vs Dog Audio Classification

Dataset: Audio Cats and Dogs
Model  : Simple CNN on mel-spectrograms (PyTorch + torchaudio)

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 10 --batch-size 16
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from tqdm import tqdm
from shared.utils import (seed_everything, get_device, dataset_prompt,
                          kaggle_download, ensure_dir, save_metrics,
                          parse_common_args)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "mmoreaux/audio-cats-and-dogs"
EPOCHS, LR, BATCH = 15, 1e-3, 32
SR = 16000


class AudioDS(Dataset):
    def __init__(self, files, labels):
        self.files, self.labels = files, labels
        self.mel = MelSpectrogram(sample_rate=SR, n_mels=64, n_fft=1024)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        wav, sr = torchaudio.load(str(self.files[i]))
        if sr != SR:
            wav = Resample(sr, SR)(wav)
        wav = wav.mean(0, keepdim=True)  # mono
        target_len = SR * 3
        if wav.shape[1] < target_len:
            wav = nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        else:
            wav = wav[:, :target_len]
        spec = self.mel(wav)  # (1, 64, T)
        return spec, self.labels[i]


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4*4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_data():
    dataset_prompt("Audio Cats and Dogs",
                   ["https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs"])
    wavs = list(DATA_DIR.rglob("*.wav"))
    if not wavs:
        kaggle_download(KAGGLE, DATA_DIR)
        wavs = list(DATA_DIR.rglob("*.wav"))
    files, labels = [], []
    for w in wavs:
        low = w.stem.lower()
        if "cat" in low or "cat" in str(w.parent).lower():
            files.append(w); labels.append(0)
        elif "dog" in low or "dog" in str(w.parent).lower():
            files.append(w); labels.append(1)
    return AudioDS(files, labels)


def main():
    args = parse_common_args()
    seed_everything(42)
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    ds = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    epochs      = args.epochs or EPOCHS
    batch_size  = args.batch_size or BATCH
    num_workers = args.num_workers if args.num_workers is not None else 2
    max_batches = 2 if args.smoke_test else None
    if args.smoke_test:
        epochs = 1

    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size, num_workers=num_workers)

    model = AudioCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    for ep in range(epochs):
        model.train()
        for bi, (X, y) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")):
            if max_batches and bi >= max_batches:
                break
            X, y = X.to(device), y.to(device)
            loss = crit(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); c, t = 0, 0
        with torch.no_grad():
            for bi, (X, y) in enumerate(val_dl):
                if max_batches and bi >= max_batches:
                    break
                X, y = X.to(device), y.to(device)
                c += (model(X).argmax(1) == y).sum().item(); t += len(y)
        acc = c / max(t, 1)
        print(f"  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    save_metrics({"best_val_accuracy": best}, OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

CUSTOM_P44 = '''\
#!/usr/bin/env python3
"""Project 44 -- Image Colorization

Dataset : VizWiz Colorization
Model   : Simple UNet autoencoder (grayscale -> color)

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 10 --batch-size 8
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from shared.utils import (seed_everything, get_device, dataset_prompt,
                          kaggle_download, ensure_dir, save_metrics,
                          parse_common_args)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "landrykezebou/vizwiz-colorization"
EPOCHS, LR, BATCH = 20, 1e-3, 16
IMG_SIZE = 128


class ColorDS(Dataset):
    def __init__(self, paths, size=128):
        self.paths, self.size = paths, size
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        color = self.tf(Image.open(self.paths[i]).convert("RGB"))
        gray  = color.mean(0, keepdim=True)
        return gray, color


class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1), nn.BatchNorm2d(co), nn.ReLU())
        self.enc1 = block(1, 64)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), block(64, 128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), block(128, 256))
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)
        self.head = nn.Conv2d(64, 3, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.head(d1))


def get_data(max_imgs=5000):
    dataset_prompt("VizWiz Colorization",
                   ["https://www.kaggle.com/datasets/landrykezebou/vizwiz-colorization"])
    imgs = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    if not imgs:
        kaggle_download(KAGGLE, DATA_DIR)
        imgs = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    return ColorDS(imgs[:max_imgs])


def main():
    args = parse_common_args()
    seed_everything(42)
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    max_imgs = 100 if args.smoke_test else 5000
    ds = get_data(max_imgs=max_imgs)
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    epochs      = args.epochs or EPOCHS
    batch_size  = args.batch_size or BATCH
    num_workers = args.num_workers if args.num_workers is not None else 2
    max_batches = 2 if args.smoke_test else None
    if args.smoke_test:
        epochs = 1

    n_val = int(0.15 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size, num_workers=num_workers)

    model = MiniUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    best_loss = float("inf")

    for ep in range(epochs):
        model.train(); total_loss = 0
        for bi, (gray, color) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")):
            if max_batches and bi >= max_batches:
                break
            gray, color = gray.to(device), color.to(device)
            pred = model(gray)
            loss = crit(pred, color)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        # val
        model.eval(); vloss = 0; vcount = 0
        with torch.no_grad():
            for bi, (gray, color) in enumerate(val_dl):
                if max_batches and bi >= max_batches:
                    break
                gray, color = gray.to(device), color.to(device)
                vloss += crit(model(gray), color).item()
                vcount += 1
        vloss /= max(vcount, 1)
        print(f"  train_loss={total_loss / max(bi + 1, 1):.4f}  val_loss={vloss:.4f}")
        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    save_metrics({"best_val_mse": best_loss}, OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

CUSTOM_P48 = '''\
#!/usr/bin/env python3
"""Project 48 -- Fashion MNIST Clothing Classification

Dataset : Fashion-MNIST (auto-downloaded via torchvision)
Model   : efficientnet_b0 (timm, pretrained)

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 5 --batch-size 128
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from shared.utils import (seed_everything, get_device, ensure_dir,
                          parse_common_args)
from shared.cv import build_timm_model, train_model, evaluate_model

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MODEL   = "efficientnet_b0.ra_in1k"
EPOCHS  = 10
BATCH   = 64


def get_data(batch_size=64, num_workers=4):
    tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485] * 3, [0.229] * 3),
    ])
    train_full = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True, transform=tf)
    test_ds    = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True, transform=tf)
    n_val = int(0.15 * len(train_full))
    train_ds, val_ds = random_split(train_full, [len(train_full) - n_val, n_val])
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, **kw),
            DataLoader(test_ds, **kw), CLASSES)


def main():
    args = parse_common_args()
    seed_everything(42)
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    epochs      = args.epochs or EPOCHS
    batch_size  = args.batch_size or BATCH
    num_workers = args.num_workers if args.num_workers is not None else 4
    use_amp     = not args.no_amp
    max_batches = 2 if args.smoke_test else None
    if args.smoke_test:
        epochs = 1

    train_dl, val_dl, test_dl, classes = get_data(
        batch_size=batch_size, num_workers=num_workers,
    )

    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    model = build_timm_model(MODEL, num_classes=10)
    model = train_model(model, train_dl, val_dl, epochs=epochs,
                        lr=1e-4, device=device, output_dir=OUTPUT_DIR,
                        use_amp=use_amp, max_batches=max_batches)
    evaluate_model(model, test_dl, classes, device=device,
                   output_dir=OUTPUT_DIR, max_batches=max_batches)


if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (same _cv_kaggle / _tab_kaggle from original generator)
# ═══════════════════════════════════════════════════════════════════════════════

def _cv_kaggle(subdir=None, competition=False):
    comp = ", competition=True" if competition else ""
    if subdir:
        return (
            f'    target = DATA_DIR / "{subdir}"\n'
            f'    if not target.exists():\n'
            f'        kaggle_download(KAGGLE, DATA_DIR{comp})\n'
            f'    return target'
        )
    return (
        f'    if not list(DATA_DIR.rglob("*.jpg")) and not list(DATA_DIR.rglob("*.png")):\n'
        f'        kaggle_download(KAGGLE, DATA_DIR{comp})\n'
        f'    for child in sorted(DATA_DIR.iterdir()):\n'
        f'        if child.is_dir() and child.name != "__MACOSX":\n'
        f'            return child\n'
        f'    return DATA_DIR'
    )


def _tab_kaggle(csv_hint=None):
    if csv_hint:
        return (
            f'    csv = DATA_DIR / "{csv_hint}"\n'
            f'    if not csv.exists():\n'
            f'        kaggle_download(KAGGLE, DATA_DIR)\n'
            f'        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            f'        csv = csvs[0] if csvs else csv\n'
            f'    return pd.read_csv(csv)'
        )
    return (
        '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
        '    if not csvs:\n'
        '        kaggle_download(KAGGLE, DATA_DIR)\n'
        '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
        '    return pd.read_csv(csvs[0])'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Project definitions (same as original, with custom_run updated)
# ═══════════════════════════════════════════════════════════════════════════════

TYPE_TO_TPL = {
    "cv": CV_TPL,
    "tabular_cls": TAB_CLS_TPL,
    "tabular_reg": TAB_REG_TPL,
    "tabular_cluster": TAB_CLUST_TPL,
    "nlp": NLP_TPL,
}

TYPE_TO_DESC = {
    "cv":              ("timm (ConvNeXt / EfficientNet / Swin)", "Image Classification", "N/A"),
    "tabular_cls":     ("PyCaret AutoML", "Tabular Classification", "PyCaret compare_models"),
    "tabular_reg":     ("PyCaret AutoML", "Tabular Regression", "PyCaret compare_models"),
    "tabular_cluster": ("PyCaret Clustering", "Unsupervised Clustering", "PyCaret KMeans"),
    "nlp":             ("HuggingFace Transformers", "Text Classification", "N/A"),
    "custom":          ("see run.py", "see run.py", "N/A"),
}

# ── All 50 projects ──────────────────────────────────────────────────────────
PROJECTS = [
    dict(num=1, folder="Deep Learning Projects 1 - Pnemonia Detection",
         title="Pneumonia Detection", type="cv",
         dataset="Chest X-Ray Pneumonia",
         links=["https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"],
         kaggle="paultimothymooney/chest-xray-pneumonia",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=2, epochs=15,
         get_data_body=_cv_kaggle("chest_xray")),

    dict(num=2, folder="Deep Learning Projects 2 - Face Mask Detection",
         title="Face Mask Detection", type="cv",
         dataset="Face Mask 12K",
         links=["https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset"],
         kaggle="ashishjangra27/face-mask-12k-images-dataset",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=2, epochs=10,
         get_data_body=_cv_kaggle("Face Mask Dataset")),

    dict(num=3, folder="Deep Learning Projects 3 - Earthquack Prediction model",
         title="Earthquake Prediction", type="tabular_reg",
         dataset="Earthquake Prediction",
         links=["https://www.kaggle.com/datasets/henryshan/earthquake-prediction"],
         kaggle="henryshan/earthquake-prediction",
         target="magnitude", get_data_body=_tab_kaggle()),

    dict(num=4, folder="Deep Learning Projects 4 - Landmark Detection Model",
         title="Landmark Detection", type="cv",
         dataset="Google Landmarks Dataset",
         links=["https://www.kaggle.com/datasets/google/google-landmarks-dataset"],
         kaggle="google/google-landmarks-dataset",
         model="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", num_classes=0, epochs=15,
         get_data_body=_cv_kaggle()),

    dict(num=5, folder="Deep Learning Projects 5 - Chatbot With Deep Learning",
         title="Chatbot Intent Classification", type="nlp",
         dataset="Chatbot Intent Recognition",
         links=["https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset"],
         kaggle="elvinagammed/chatbots-intent-recognition-dataset",
         hf_model="distilbert-base-uncased", num_labels=0,
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    text_col = [c for c in df.columns if c.lower() in ("text","query","utterance","sentence")][0]\n'
            '    label_col = [c for c in df.columns if c.lower() in ("intent","label","category","tag")][0]\n'
            '    labels_unique = sorted(df[label_col].unique())\n'
            '    lab2id = {l: i for i, l in enumerate(labels_unique)}\n'
            '    df["_label"] = df[label_col].map(lab2id)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr[text_col].tolist(), tr["_label"].tolist(),\n'
            '            te[text_col].tolist(), te["_label"].tolist(), labels_unique)'
         )),

    dict(num=6, folder="Deep Learning Projects 6 - Movies Title Prediction",
         title="Movie Genre Prediction", type="nlp",
         dataset="Wikipedia Movie Plots",
         links=["https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots"],
         kaggle="jrobischon/wikipedia-movie-plots",
         hf_model="distilbert-base-uncased", num_labels=0,
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    top = df["Genre"].value_counts().head(10).index.tolist()\n'
            '    df = df[df["Genre"].isin(top)].copy()\n'
            '    labels = sorted(top)\n'
            '    lab2id = {l: i for i, l in enumerate(labels)}\n'
            '    df["_label"] = df["Genre"].map(lab2id)\n'
            '    df["_text"] = df["Plot"].str[:512]\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["_text"].tolist(), tr["_label"].tolist(),\n'
            '            te["_text"].tolist(), te["_label"].tolist(), labels)'
         )),

    dict(num=7, folder="Deep Learning Projects 7 - Advanced Churn Modeling",
         title="Customer Churn Prediction", type="tabular_cls",
         dataset="Churn Modelling",
         links=["https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling"],
         kaggle="shrutimechlearn/churn-modelling",
         target="Exited", get_data_body=_tab_kaggle("Churn_Modelling.csv")),

    dict(num=8, folder="Deep Learning Projects 8 - Disease Prediction Model",
         title="Disease Prediction", type="tabular_cls",
         dataset="Disease Prediction Using ML",
         links=["https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning"],
         kaggle="kaushil268/disease-prediction-using-machine-learning",
         target="prognosis", get_data_body=_tab_kaggle()),

    dict(num=9, folder="Deep Learning Projects 9 - IMDB Sentiment Analysis using Deep Learning",
         title="IMDB Sentiment Analysis", type="nlp",
         dataset="IMDB 50K Movie Reviews",
         links=["https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"],
         kaggle="lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
         hf_model="distilbert-base-uncased", num_labels=2,
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    lab_map = {"positive": 1, "negative": 0}\n'
            '    df["_label"] = df["sentiment"].map(lab_map)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["review"].tolist(), tr["_label"].tolist(),\n'
            '            te["review"].tolist(), te["_label"].tolist(),\n'
            '            ["negative", "positive"])'
         )),

    dict(num=10, folder="Deep Learning Projects 10 - Advanced rsnet50",
         title="Plant Pathology (ResNet50 -> ConvNeXt)", type="cv",
         dataset="Plant Pathology 2021",
         links=["https://www.kaggle.com/c/plant-pathology-2021-fgvc8"],
         kaggle="plant-pathology-2021-fgvc8",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=0, epochs=15,
         get_data_body=_cv_kaggle(competition=True)),

    dict(num=11, folder="Deep Learning Projects 11 - Cat Vs Dog",
         title="Cat vs Dog Classification", type="cv",
         dataset="Dogs vs Cats",
         links=["https://www.kaggle.com/c/dogs-vs-cats"],
         kaggle="dogs-vs-cats",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=2, epochs=10,
         get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "organized"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR, competition=True)\n'
            '        for split, ratio in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:\n'
            '            (target / split / "cat").mkdir(parents=True, exist_ok=True)\n'
            '            (target / split / "dog").mkdir(parents=True, exist_ok=True)\n'
            '        src = DATA_DIR / "train"\n'
            '        if not src.is_dir():\n'
            '            src = DATA_DIR\n'
            '        cats = sorted(src.glob("cat*.jpg"))\n'
            '        dogs = sorted(src.glob("dog*.jpg"))\n'
            '        import random; random.seed(42)\n'
            '        random.shuffle(cats); random.shuffle(dogs)\n'
            '        for imgs, cls in [(cats,"cat"), (dogs,"dog")]:\n'
            '            n = len(imgs)\n'
            '            splits = {"train": imgs[:int(.8*n)], "val": imgs[int(.8*n):int(.9*n)], "test": imgs[int(.9*n):]}\n'
            '            for sp, files in splits.items():\n'
            '                for f in files:\n'
            '                    shutil.copy2(f, target / sp / cls / f.name)\n'
            '    return target'
         )),

    dict(num=12, folder="Deep Learning Projects 12 - Keep Babies Safe",
         title="Distracted Driver Detection", type="cv",
         dataset="State Farm Distracted Driver Detection",
         links=["https://www.kaggle.com/c/state-farm-distracted-driver-detection"],
         kaggle="state-farm-distracted-driver-detection",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=10, epochs=15,
         get_data_body=_cv_kaggle(competition=True)),

    dict(num=13, folder="Deep Learning Projects 13 - Covid 19 Drug Recovery using Deep Learning",
         title="COVID-19 Drug Recovery Analysis", type="tabular_cls",
         dataset="UNCOVER COVID-19 Challenge",
         links=["https://www.kaggle.com/datasets/roche-data-science-coalition/uncover"],
         kaggle="roche-data-science-coalition/uncover",
         target="outcome",
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    for csv in csvs:\n'
            '        df = pd.read_csv(csv, low_memory=False)\n'
            '        if TARGET in df.columns:\n'
            '            return df\n'
            '    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)\n'
            '    df = pd.read_csv(csvs[0], low_memory=False)\n'
            '    return df'
         )),

    dict(num=14, folder="Deep Learning Projects 14 - Face, Gender & Ethincity recognizer model",
         title="Face / Gender / Ethnicity Recognizer", type="cv",
         dataset="UTKFace",
         links=["https://www.kaggle.com/datasets/jangedoo/utkface-new",
                "https://www.kaggle.com/datasets/jessicali9530/fairface-dataset"],
         kaggle="jangedoo/utkface-new",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=5, epochs=15,
         get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "ethnicity"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        races = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "others"}\n'
            '        src = None\n'
            '        for d in [DATA_DIR / "UTKFace", DATA_DIR / "utkface_aligned_cropped", DATA_DIR]:\n'
            '            if d.is_dir() and list(d.glob("*.jpg")):\n'
            '                src = d; break\n'
            '        if src is None:\n'
            '            raise FileNotFoundError("No images found after Kaggle download")\n'
            '        imgs = list(src.glob("*.jpg")) + list(src.glob("*.png"))\n'
            '        import random; random.seed(42); random.shuffle(imgs)\n'
            '        n = len(imgs)\n'
            '        for i, f in enumerate(imgs):\n'
            '            parts = f.stem.split("_")\n'
            '            if len(parts) < 3: continue\n'
            '            race_id = int(parts[2]) if parts[2].isdigit() else 4\n'
            '            race = races.get(race_id, "others")\n'
            '            split = "train" if i < .8*n else ("val" if i < .9*n else "test")\n'
            '            dest = target / split / race\n'
            '            dest.mkdir(parents=True, exist_ok=True)\n'
            '            shutil.copy2(f, dest / f.name)\n'
            '    return target'
         )),

    dict(num=15, folder="Deep Learning Projects 15 - Happy house Predictor model",
         title="Boston Housing Price Prediction", type="tabular_reg",
         dataset="Boston Housing Dataset",
         links=["https://www.kaggle.com/datasets/uciml/boston-housing-dataset"],
         kaggle="uciml/boston-housing-dataset",
         target="MEDV", get_data_body=_tab_kaggle()),

    dict(num=16, folder="Deep Learning Projects 16 - Brain MRI Segmentation modling",
         title="Brain MRI Segmentation", type="custom",
         dataset="LGG MRI Segmentation",
         links=["https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"],
         kaggle="mateuszbuda/lgg-mri-segmentation",
         model="DeepLabV3-ResNet50",
         custom_run=CUSTOM_P16),

    dict(num=17, folder="Deep Learning Projects 17 - Parkension Post Estimation using deep learning",
         title="Parkinson's Disease Detection", type="tabular_cls",
         dataset="Parkinson's Disease Data Set",
         links=["https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set"],
         kaggle="vikasukani/parkinsons-disease-data-set",
         target="status", get_data_body=_tab_kaggle()),

    dict(num=18, folder="Deep Learning Projects 18 - Diabetic Retinopathy project",
         title="Diabetic Retinopathy Detection", type="cv",
         dataset="Diabetic Retinopathy Detection",
         links=["https://www.kaggle.com/c/diabetic-retinopathy-detection"],
         kaggle="diabetic-retinopathy-detection",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=5, epochs=15,
         get_data_body=_cv_kaggle(competition=True)),

    dict(num=19, folder="Deep Learning Projects 19 - Arabic character recognization using deep learning",
         title="Arabic Handwritten Character Recognition", type="cv",
         dataset="AHCD (Arabic Handwritten Characters)",
         links=["https://www.kaggle.com/datasets/mloey1/ahcd1"],
         kaggle="mloey1/ahcd1",
         model="efficientnet_b0.ra_in1k", num_classes=28, epochs=15,
         get_data_body=_cv_kaggle()),

    dict(num=20, folder="Deep Learning Projects 20 - Brain Tumor Recognization using Deep Learning",
         title="Brain Tumor Classification", type="cv",
         dataset="Brain MRI Images for Tumor Detection",
         links=["https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection"],
         kaggle="navoneel/brain-mri-images-for-brain-tumor-detection",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=0, epochs=20,
         get_data_body=_cv_kaggle()),

    dict(num=21, folder="Deep Learning Projects 21 - Image Walking or Running",
         title="Human Action Recognition (Walk / Run)", type="cv",
         dataset="HAR Dataset",
         links=["https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset"],
         kaggle="meetnagadia/human-action-recognition-har-dataset",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=0, epochs=15,
         get_data_body=_cv_kaggle()),

    dict(num=22, folder="Deep Learning Projects 22- 1957 All Space Missions",
         title="Space Missions Success Prediction", type="tabular_cls",
         dataset="All Space Missions from 1957",
         links=["https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957"],
         kaggle="agirlcoding/all-space-missions-from-1957",
         target="Status Mission", get_data_body=_tab_kaggle()),

    dict(num=23, folder="Deep Learning Projects 23 - 1 Million Suduku Solver using neural nets",
         title="Sudoku Solver with Neural Network", type="custom",
         dataset="1M Sudoku Games",
         links=["https://www.kaggle.com/datasets/bryanpark/sudoku"],
         kaggle="bryanpark/sudoku",
         model="Custom CNN",
         custom_run=CUSTOM_P23),

    dict(num=24, folder="Deep Learning Projects 24 -Electric Car Temperature Predictor using Deep Learning",
         title="Electric Motor Temperature Prediction", type="tabular_reg",
         dataset="Electric Motor Temperature",
         links=["https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature"],
         kaggle="wkirgsn/electric-motor-temperature",
         target="pm", get_data_body=_tab_kaggle()),

    dict(num=25, folder="Deep Learning Projects 25-Hourly energy demand generation and weather",
         title="Energy Demand Prediction", type="tabular_reg",
         dataset="Hourly Energy Consumption",
         links=["https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption"],
         kaggle="robikscube/hourly-energy-consumption",
         target="PJME_MW",
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    if "Datetime" in df.columns:\n'
            '        df["Datetime"] = pd.to_datetime(df["Datetime"])\n'
            '        df["hour"] = df["Datetime"].dt.hour\n'
            '        df["dayofweek"] = df["Datetime"].dt.dayofweek\n'
            '        df["month"] = df["Datetime"].dt.month\n'
            '        df.drop(columns=["Datetime"], inplace=True)\n'
            '    return df'
         )),

    dict(num=26, folder="Deep Learning Projects 26 - Caffe Face Detector (OpenCV Pre-trained Model)",
         title="Face Detection (OpenCV DNN)", type="custom",
         dataset="OpenCV DNN Model Zoo",
         links=["https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"],
         kaggle="",
         model="OpenCV DNN",
         custom_run=CUSTOM_P26),

    dict(num=27, folder="Deep Learning Projects 27- Calculate Concrete Strength",
         title="Concrete Compressive Strength Prediction", type="tabular_reg",
         dataset="Concrete Compressive Strength",
         links=["https://www.kaggle.com/datasets/uciml/concrete-compressive-strength-data-set"],
         kaggle="uciml/concrete-compressive-strength-data-set",
         target="csMPa",
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    target_candidates = [c for c in df.columns if "strength" in c.lower() or "csm" in c.lower()]\n'
            '    if target_candidates and TARGET not in df.columns:\n'
            '        df = df.rename(columns={target_candidates[0]: TARGET})\n'
            '    return df'
         )),

    dict(num=28, folder="Deep Learning Projects 28 - Stock Market Prediction",
         title="Stock Market Prediction from News", type="nlp",
         dataset="Stock News Sentiment",
         links=["https://www.kaggle.com/datasets/aaron7sun/stocknews"],
         kaggle="aaron7sun/stocknews",
         hf_model="distilbert-base-uncased", num_labels=2,
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    text_cols = [c for c in df.columns if c.startswith("Top")]\n'
            '    if text_cols:\n'
            '        df["_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)\n'
            '    else:\n'
            '        df["_text"] = df.iloc[:, 2:].fillna("").agg(" ".join, axis=1)\n'
            '    label_col = "Label" if "Label" in df.columns else df.columns[1]\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42)\n'
            '    return (tr["_text"].tolist(), tr[label_col].astype(int).tolist(),\n'
            '            te["_text"].tolist(), te[label_col].astype(int).tolist(),\n'
            '            ["down", "up"])'
         )),

    dict(num=29, folder="Deep Learning Projects 29 - Indian Startup data Analysis",
         title="Indian Startup Funding Prediction", type="tabular_reg",
         dataset="Startup Investments (CrunchBase)",
         links=["https://www.kaggle.com/datasets/ruchi798/startup-investments-crunchbase"],
         kaggle="ruchi798/startup-investments-crunchbase",
         target="funding_total_usd",
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0], low_memory=False)\n'
            '    if TARGET in df.columns:\n'
            '        df[TARGET] = pd.to_numeric(df[TARGET].astype(str).str.replace(",","").str.strip(), errors="coerce")\n'
            '        df = df.dropna(subset=[TARGET])\n'
            '        df = df[df[TARGET] > 0]\n'
            '    return df'
         )),

    dict(num=30, folder="Deep Learning Projects 30 - Amazon Stock Price Deep Analysis",
         title="Amazon Stock Price Prediction", type="tabular_reg",
         dataset="Amazon Stock Price",
         links=["https://www.kaggle.com/datasets/rohanrao/amazon-stock-price"],
         kaggle="rohanrao/amazon-stock-price",
         target="Close",
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    if "Date" in df.columns:\n'
            '        df["Date"] = pd.to_datetime(df["Date"])\n'
            '        df = df.sort_values("Date").reset_index(drop=True)\n'
            '        for lag in [1, 3, 7]:\n'
            '            df[f"close_lag{lag}"] = df["Close"].shift(lag)\n'
            '        df.dropna(inplace=True)\n'
            '        df.drop(columns=["Date"], inplace=True)\n'
            '    return df'
         )),

    dict(num=31, folder="Deep Learning Projects 31 - Indentifying Dance Form Using Deep Learning-20210724T041140Z-001",
         title="Indian Classical Dance Form Classification", type="cv",
         dataset="Indian Classical Dance",
         links=["https://www.kaggle.com/datasets/arjunbhasin2013/indian-classical-dance"],
         kaggle="arjunbhasin2013/indian-classical-dance",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=0, epochs=20,
         get_data_body=_cv_kaggle()),

    dict(num=32, folder="Deep Learning Projects 32 - Glass or No Glass Detector Model using DL",
         title="Eyeglasses Detection", type="cv",
         dataset="Eyeglasses Dataset",
         links=["https://www.kaggle.com/datasets/jehanbhathena/eyeglasses-dataset"],
         kaggle="jehanbhathena/eyeglasses-dataset",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=2, epochs=10,
         get_data_body=_cv_kaggle()),

    dict(num=33, folder="Deep Learning Projects 33 - Fingerprint Recognizer Model using DL",
         title="Fingerprint Recognition", type="cv",
         dataset="SOCOFing Fingerprint Dataset",
         links=["https://www.kaggle.com/datasets/ruizgara/socofing"],
         kaggle="ruizgara/socofing",
         model="efficientnet_b0.ra_in1k", num_classes=0, epochs=15,
         get_data_body=_cv_kaggle()),

    dict(num=34, folder="Deep Learning Projects 34 - World Currency Coin Detector Model using DL",
         title="World Currency Coin Classification", type="cv",
         dataset="Coin Images",
         links=["https://www.kaggle.com/datasets/wanderdust/coin-images"],
         kaggle="wanderdust/coin-images",
         model="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", num_classes=0, epochs=20,
         get_data_body=_cv_kaggle()),

    dict(num=35, folder="Deep Learning Projects 35 - News Category Prediction using DL",
         title="News Category Prediction", type="nlp",
         dataset="News Category Dataset",
         links=["https://www.kaggle.com/datasets/rmisra/news-category-dataset"],
         kaggle="rmisra/news-category-dataset",
         hf_model="distilbert-base-uncased", num_labels=0,
         get_data_body=(
            '    import json as _json\n'
            '    files = list(DATA_DIR.rglob("*.json"))\n'
            '    if not files:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        files = list(DATA_DIR.rglob("*.json"))\n'
            '    records = []\n'
            '    for f in files:\n'
            '        for line in f.read_text(encoding="utf-8").splitlines():\n'
            '            if line.strip():\n'
            '                records.append(_json.loads(line))\n'
            '    df = pd.DataFrame(records)\n'
            '    top = df["category"].value_counts().head(10).index.tolist()\n'
            '    df = df[df["category"].isin(top)].copy()\n'
            '    labels = sorted(top)\n'
            '    lab2id = {l: i for i, l in enumerate(labels)}\n'
            '    df["_label"] = df["category"].map(lab2id)\n'
            '    df["_text"] = df["headline"].fillna("") + " " + df["short_description"].fillna("")\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["_text"].tolist(), tr["_label"].tolist(),\n'
            '            te["_text"].tolist(), te["_label"].tolist(), labels)'
         )),

    dict(num=36, folder="Deep Learning Projects 36 - Lego Brick Code Problem",
         title="Lego Brick Classification", type="cv",
         dataset="Lego Brick Images",
         links=["https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images"],
         kaggle="joosthazelzet/lego-brick-images",
         model="efficientnet_b0.ra_in1k", num_classes=0, epochs=15,
         get_data_body=_cv_kaggle()),

    dict(num=37, folder="Deep Learning Projects 37 - Sheep Breed Classification using CNN DL",
         title="Sheep Breed Classification", type="cv",
         dataset="Sheep Face Images",
         links=["https://www.kaggle.com/datasets/warcoder/sheep-face-images"],
         kaggle="warcoder/sheep-face-images",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=4, epochs=15,
         get_data_body=_cv_kaggle()),

    dict(num=38, folder="Deep Learning Projects 38 - Campus Recruitment Success rate analysis",
         title="Campus Recruitment Prediction", type="tabular_cls",
         dataset="Campus Recruitment",
         links=["https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement"],
         kaggle="benroshan/factors-affecting-campus-placement",
         target="status", get_data_body=_tab_kaggle()),

    dict(num=39, folder="Deep Learning Projects 39 - Bank Marketing",
         title="Bank Marketing Prediction", type="tabular_cls",
         dataset="Bank Marketing",
         links=["https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing"],
         kaggle="henriqueyamahata/bank-marketing",
         target="y", get_data_body=_tab_kaggle()),

    dict(num=40, folder="Deep Learning Projects 40 - Pokemon Generation Clustering",
         title="Pokemon Generation Clustering", type="tabular_cluster",
         dataset="Pokemon Dataset",
         links=["https://www.kaggle.com/datasets/abcsds/pokemon"],
         kaggle="abcsds/pokemon",
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    df = df.select_dtypes(include="number").dropna()\n'
            '    return df'
         )),

    dict(num=41, folder="Deep Learning Projects 41 - Cat _ Dog Voice Recognizer Model",
         title="Cat vs Dog Audio Classification", type="custom",
         dataset="Audio Cats and Dogs",
         links=["https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs"],
         kaggle="mmoreaux/audio-cats-and-dogs",
         model="CNN on mel-spectrogram",
         custom_run=CUSTOM_P41),

    dict(num=42, folder="Deep Learning Projects 42 - Bottle or Cans Classifier using DL",
         title="Bottle vs Can Classification", type="cv",
         dataset="Bottles and Cans",
         links=["https://www.kaggle.com/datasets/trolukovich/bottles-and-cans"],
         kaggle="trolukovich/bottles-and-cans",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=0, epochs=15,
         get_data_body=_cv_kaggle()),

    dict(num=43, folder="Deep Learning Projects 43 - Skin Cancer Recognizer using DL",
         title="Skin Cancer (HAM10000) Classification", type="cv",
         dataset="Skin Cancer MNIST: HAM10000",
         links=["https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"],
         kaggle="kmader/skin-cancer-mnist-ham10000",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=7, epochs=20,
         get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "organized"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csv_files = list(DATA_DIR.rglob("*metadata*")) + list(DATA_DIR.rglob("*HAM*.csv"))\n'
            '        if csv_files:\n'
            '            import pandas as _pd\n'
            '            meta = _pd.read_csv(csv_files[0])\n'
            '            imgs = list(DATA_DIR.rglob("*.jpg"))\n'
            '            img_map = {p.stem: p for p in imgs}\n'
            '            import random; random.seed(42)\n'
            '            meta = meta.sample(frac=1, random_state=42)\n'
            '            n = len(meta)\n'
            '            for i, row in enumerate(meta.itertuples()):\n'
            '                split = "train" if i < .8*n else ("val" if i < .9*n else "test")\n'
            '                cls = row.dx\n'
            '                src_img = img_map.get(row.image_id)\n'
            '                if src_img:\n'
            '                    dest = target / split / cls\n'
            '                    dest.mkdir(parents=True, exist_ok=True)\n'
            '                    shutil.copy2(src_img, dest / src_img.name)\n'
            '    return target'
         )),

    dict(num=44, folder="Deep Learning Projects 44 - Image Colorization using Deep Learning",
         title="Image Colorization", type="custom",
         dataset="VizWiz Colorization",
         links=["https://www.kaggle.com/datasets/landrykezebou/vizwiz-colorization"],
         kaggle="landrykezebou/vizwiz-colorization",
         model="UNet autoencoder",
         custom_run=CUSTOM_P44),

    dict(num=45, folder="Deep Learning Projects 45 - Amazon Alexa Review Sentiment Analysis",
         title="Amazon Alexa Review Sentiment", type="nlp",
         dataset="Amazon Alexa Reviews",
         links=["https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews"],
         kaggle="sid321axn/amazon-alexa-reviews",
         hf_model="distilbert-base-uncased", num_labels=2,
         get_data_body=(
            '    files = list(DATA_DIR.rglob("*.tsv")) + list(DATA_DIR.rglob("*.csv"))\n'
            '    if not files:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        files = list(DATA_DIR.rglob("*.tsv")) + list(DATA_DIR.rglob("*.csv"))\n'
            '    f = files[0]\n'
            '    df = pd.read_csv(f, sep="\\t" if f.suffix == ".tsv" else ",", encoding="latin-1")\n'
            '    text_col = [c for c in df.columns if "review" in c.lower() or "verified" in c.lower() or "text" in c.lower()]\n'
            '    text_col = text_col[0] if text_col else df.columns[-2]\n'
            '    label_col = "feedback" if "feedback" in df.columns else "rating"\n'
            '    if df[label_col].nunique() > 2:\n'
            '        df["_label"] = (df[label_col] >= 4).astype(int)\n'
            '    else:\n'
            '        df["_label"] = df[label_col].astype(int)\n'
            '    df["_text"] = df[text_col].astype(str)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["_text"].tolist(), tr["_label"].tolist(),\n'
            '            te["_text"].tolist(), te["_label"].tolist(),\n'
            '            ["negative", "positive"])'
         )),

    dict(num=46, folder="Deep Learning Projects 46 - Build_ChatBot_using_Neural_Network",
         title="Chatbot Intent Classification", type="nlp",
         dataset="Chatbot Intent Recognition",
         links=["https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset"],
         kaggle="elvinagammed/chatbots-intent-recognition-dataset",
         hf_model="distilbert-base-uncased", num_labels=0,
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    text_col = [c for c in df.columns if c.lower() in ("text","query","utterance","sentence")][0]\n'
            '    label_col = [c for c in df.columns if c.lower() in ("intent","label","category","tag")][0]\n'
            '    labels_unique = sorted(df[label_col].unique())\n'
            '    lab2id = {l: i for i, l in enumerate(labels_unique)}\n'
            '    df["_label"] = df[label_col].map(lab2id)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr[text_col].tolist(), tr["_label"].tolist(),\n'
            '            te[text_col].tolist(), te["_label"].tolist(), labels_unique)'
         )),

    dict(num=47, folder="Deep Learning Projects 47 - Cactus or Not Cactus Ariel Image Recognizer",
         title="Aerial Cactus Identification", type="cv",
         dataset="Aerial Cactus Identification",
         links=["https://www.kaggle.com/c/aerial-cactus-identification"],
         kaggle="aerial-cactus-identification",
         model="efficientnet_b0.ra_in1k", num_classes=2, epochs=10,
         get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "organized"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR, competition=True)\n'
            '        csv_files = list(DATA_DIR.rglob("*.csv"))\n'
            '        import pandas as _pd\n'
            '        df = _pd.read_csv(csv_files[0]) if csv_files else _pd.DataFrame()\n'
            '        img_dir = DATA_DIR / "train"\n'
            '        if not img_dir.is_dir():\n'
            '            img_dir = DATA_DIR\n'
            '        if "has_cactus" in df.columns:\n'
            '            import random; random.seed(42)\n'
            '            df = df.sample(frac=1, random_state=42)\n'
            '            n = len(df)\n'
            '            for i, row in enumerate(df.itertuples()):\n'
            '                split = "train" if i < .8*n else ("val" if i < .9*n else "test")\n'
            '                cls = "cactus" if row.has_cactus else "no_cactus"\n'
            '                dest = target / split / cls\n'
            '                dest.mkdir(parents=True, exist_ok=True)\n'
            '                src = img_dir / row.id\n'
            '                if src.exists():\n'
            '                    shutil.copy2(src, dest / row.id)\n'
            '        else:\n'
            '            return DATA_DIR\n'
            '    return target'
         )),

    dict(num=48, folder="Deep Learning Projects 48 -  Build_Clothing_Prediction_Flask_Web_App",
         title="Fashion MNIST Clothing Classification", type="custom",
         dataset="Fashion-MNIST (torchvision)",
         links=["https://github.com/zalandoresearch/fashion-mnist"],
         kaggle="",
         model="efficientnet_b0.ra_in1k",
         custom_run=CUSTOM_P48),

    dict(num=49, folder="Deep Learning Projects 49 - Build_Sentiment_Analysis_Flask_Web_App",
         title="IMDB Sentiment Analysis", type="nlp",
         dataset="IMDB 50K Movie Reviews",
         links=["https://ai.stanford.edu/~amaas/data/sentiment/",
                "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"],
         kaggle="lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
         hf_model="distilbert-base-uncased", num_labels=2,
         get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    lab_map = {"positive": 1, "negative": 0}\n'
            '    df["_label"] = df["sentiment"].map(lab_map)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["review"].tolist(), tr["_label"].tolist(),\n'
            '            te["review"].tolist(), te["_label"].tolist(),\n'
            '            ["negative", "positive"])'
         )),

    dict(num=50, folder="Deep Learning Projects 50 - COVID-19 Lung CT Scans",
         title="COVID-19 Lung CT Scan Classification", type="cv",
         dataset="SARS-CoV-2 CT Scan Dataset",
         links=["https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset"],
         kaggle="plameneduardo/sarscov2-ctscan-dataset",
         model="convnext_tiny.fb_in22k_ft_in1k", num_classes=2, epochs=15,
         get_data_body=_cv_kaggle()),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════════

def render_run_py(c):
    if c["type"] == "custom":
        return c["custom_run"]
    tpl = TYPE_TO_TPL[c["type"]]
    kwargs = dict(
        num=c["num"], title=c["title"],
        dataset=c.get("dataset", ""), model=c.get("model", ""),
        hf_model=c.get("hf_model", "distilbert-base-uncased"),
        dataset_repr=repr(c.get("dataset", "")),
        links_repr=repr(c.get("links", [])),
        kaggle_repr=repr(c.get("kaggle", "")),
        model_repr=repr(c.get("model", "")),
        hf_model_repr=repr(c.get("hf_model", "distilbert-base-uncased")),
        num_classes=c.get("num_classes", 0),
        num_labels=c.get("num_labels", 0),
        epochs=c.get("epochs", 15),
        target_repr=repr(c.get("target", "")),
        get_data_body=c.get("get_data_body", "    return DATA_DIR"),
    )
    return tpl.format(**kwargs)


def render_readme(c):
    links_md = "\n".join(f"- {l}" for l in c.get("links", []))
    model_desc, task_desc, automl_desc = TYPE_TO_DESC.get(
        c["type"], ("custom", "custom", "N/A"))
    return README_TPL.format(
        num=c["num"], title=c["title"],
        dataset=c.get("dataset", ""),
        links_md=links_md,
        model_desc=c.get("model", model_desc),
        task_desc=task_desc,
        automl_desc=automl_desc,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Patch CLI — regenerate run.py + README.md with CLI flags")
    print("=" * 60)

    ok, skip = 0, 0
    for c in PROJECTS:
        folder = ROOT / c["folder"]
        if not folder.is_dir():
            print(f"  [SKIP] {c['folder']}")
            skip += 1
            continue
        try:
            w(f"{c['folder']}/run.py", render_run_py(c))
            w(f"{c['folder']}/README.md", render_readme(c))
            ok += 1
        except Exception as exc:
            print(f"  [FAIL] P{c['num']}: {exc}")
            skip += 1

    print(f"\n  Done: {ok} generated, {skip} skipped")


if __name__ == "__main__":
    main()
