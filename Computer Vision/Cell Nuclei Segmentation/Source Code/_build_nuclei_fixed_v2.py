#!/usr/bin/env python
"""Build Project 20: Cell Nuclei Segmentation Notebook - FIXED VERSION v2"""
import nbformat as nbf
from pathlib import Path

# Create notebook
nb = nbf.v4.new_notebook()

# === SECTION 1: Title ===
nb.cells.append(nbf.v4.new_markdown_cell("""# Cell Nuclei Segmentation
## Instance Segmentation with PyTorch and Segmentation Models

**Dataset:** Data Science Bowl 2018 (synthetic for demonstration)  
**Task:** Binary and instance-level segmentation of cell nuclei in microscopy images  
**Methods:** U-Net + Watershed + Dice/IoU metrics  
**Author:** Deep Learning Series  
**Date:** April 2026"""))

# === SECTION 2: Project Overview ===
nb.cells.append(nbf.v4.new_markdown_cell("""## Project Overview

This notebook builds a nuclei segmentation model using:
- **Architecture:** U-Net with EfficientNet-B1 encoder
- **Loss:** Dice + Focal loss combination
- **Post-processing:** Watershed algorithm for instance separation
- **Metrics:** Binary Dice/IoU + Instance-level Average Precision
- **Dataset:** Data Science Bowl 2018 (microscopy images)
"""))

# === SECTION 3-7: Other markdown sections (same) ===
for md_cell in [
    """## Learning Objectives

By the end of this project, you will understand:
1. Instance segmentation vs semantic segmentation
2. U-Net architecture with modern encoders
3. Dice loss and focal loss for imbalanced segmentation
4. Watershed algorithm for splitting overlapping objects
5. Instance-level evaluation metrics
6. Qualitative mask analysis and failure modes
""",
    """## Problem Statement

Medical and biological image analysis often requires identifying individual cell nuclei.
However, nuclei frequently overlap, making this an instance segmentation problem:
- **Input:** Microscopy images (512×512 pixels)
- **Output:** Individual nuclei masks with boundaries
- **Challenge:** Overlapping objects, varying sizes, touching boundaries
- **Solution:** Combined binary segmentation + watershed-based instance separation
""",
    """## Why This Project Matters

- **Medical Image Analysis:** Automated cell counting and morphology analysis
- **High-Throughput Screening:** Process thousands of images efficiently
- **Instance Segmentation Skills:** Foundational for medical imaging tasks
- **Post-Processing:** Learn how to refine raw model outputs for better results
- **Generalization:** Techniques applicable to object detection in microscopy
""",
    """## Dataset Overview

**Data Science Bowl 2018:**
- 670 training images with annotated nuclei
- 65 test images for validation
- Variable image sizes (256×256 to 1024×1024)
- Grayscale or RGB format
- Multiple nuclei per image (instance labels)
- Official train/test split provided

**This Notebook:** Using synthetic data that mirrors DSB2018 structure for demonstration
""",
    """## Dataset Source and License

**Official Source:** https://www.kaggle.com/competitions/data-science-bowl-2018/data

**Kaggle Dataset ID:** `carlolepelaars/data-science-bowl-2018`

**This notebook uses synthetic data for demonstration.** To use the real DSB2018 dataset:
```python
import kagglehub
dataset_path = kagglehub.dataset_download("carlolepelaars/data-science-bowl-2018")
```

**Known Characteristics of DSB2018:**
- Mix of fluorescence microscopy (DAPI-stained nuclei)
- Some images with cell bodies visible
- Annotations are conservative (cautious labeling)
- Overlapping nuclei are challenging cases
""",
    """## Environment Setup

Install required packages:
- `torch` and `torchvision` (deep learning framework)
- `segmentation-models-pytorch` (pre-trained segmentation models)
- `albumentations` (image augmentation)
- `scikit-image` (image processing utilities)
- `opencv-python` (image I/O and watershed)
- `matplotlib` and `seaborn` (visualization)
"""
]:
    nb.cells.append(nbf.v4.new_markdown_cell(md_cell))

# === SECTION 9: Imports ===
nb.cells.append(nbf.v4.new_code_cell("""import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation
from scipy import ndimage
from sklearn.metrics import jaccard_score

print('✓ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
"""))

# === SECTION 10: Configuration ===
nb.cells.append(nbf.v4.new_code_cell("""# Configuration
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dataset
DATASET_PATH = Path("./dsb2018_data")

# Training
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
MODEL_NAME = "smp.Unet(encoder_name='efficientnet-b1', encoder_weights='imagenet', in_channels=3, classes=1)"
LOSS_NAME = "Dice + Focal"

# Output
SAVE_DIR = Path(os.getcwd())
CHECKPOINT_DIR = SAVE_DIR / 'checkpoint'
CHECKPOINT_DIR.mkdir(exist_ok=True)

print(f"Configuration:")
print(f"  Image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Device: {DEVICE}")
print(f"  Save directory: {SAVE_DIR}")
"""))

# === SECTION 11: Dataset Generation ===
nb.cells.append(nbf.v4.new_code_cell("""# Create synthetic nuclei dataset (mimics DSB2018 structure)
print("Creating synthetic nuclei dataset...")
DATASET_PATH.mkdir(exist_ok=True)

# Create directories
train_path = DATASET_PATH / 'stage1_train'
test_path = DATASET_PATH / 'stage1_test'
train_path.mkdir(exist_ok=True)
test_path.mkdir(exist_ok=True)

# Generate synthetic nuclei images
def generate_nuclei_image(seed, num_nuclei=20, img_size=512):
    np.random.seed(seed)
    img = np.random.randint(20, 50, (img_size, img_size), dtype=np.uint8)
    
    # Add Gaussian nuclei
    for _ in range(num_nuclei):
        x = np.random.randint(50, img_size - 50)
        y = np.random.randint(50, img_size - 50)
        radius = np.random.randint(15, 35)
        
        # Draw Gaussian blob
        yy, xx = np.ogrid[:img_size, :img_size]
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        gaussian = 200 * np.exp(-dist_sq / (2 * radius ** 2))
        img = np.maximum(img.astype(float), gaussian).astype(np.uint8)
    
    return img

# Create training samples
for i in range(10):
    sample_id = f'sample_{i:04d}'
    sample_dir = train_path / sample_id
    sample_dir.mkdir(exist_ok=True)
    
    # Image
    images_dir = sample_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    img = generate_nuclei_image(i, num_nuclei=np.random.randint(15, 30), img_size=512)
    cv2.imwrite(str(images_dir / f'{sample_id}.png'), img)
    
    # Masks (individual nuclei)
    masks_dir = sample_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)
    binary_img = img > 100
    labeled, num_nuclei = ndimage.label(binary_img)
    
    for nuc_id in range(1, num_nuclei + 1):
        mask = (labeled == nuc_id).astype(np.uint8) * 255
        cv2.imwrite(str(masks_dir / f'mask_{nuc_id}.png'), mask)

# Create test samples
for i in range(3):
    sample_id = f'test_{i:04d}'
    sample_dir = test_path / sample_id
    sample_dir.mkdir(exist_ok=True)
    
    # Image
    images_dir = sample_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    img = generate_nuclei_image(1000 + i, num_nuclei=np.random.randint(10, 25), img_size=512)
    cv2.imwrite(str(images_dir / f'{sample_id}.png'), img)
    
    # Masks
    masks_dir = sample_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)
    binary_img = img > 100
    labeled, num_nuclei = ndimage.label(binary_img)
    
    for nuc_id in range(1, num_nuclei + 1):
        mask = (labeled == nuc_id).astype(np.uint8) * 255
        cv2.imwrite(str(masks_dir / f'mask_{nuc_id}.png'), mask)

print(f"✓ Synthetic dataset created")
print(f"  Train: {len(list(train_path.iterdir()))} images")
print(f"  Test: {len(list(test_path.iterdir()))} images")
"""))

# === SECTION 12: Data Validation ===
nb.cells.append(nbf.v4.new_code_cell("""# Validate dataset integrity
train_path = DATASET_PATH / 'stage1_train'
test_path = DATASET_PATH / 'stage1_test'

assert train_path.exists(), f"Training data not found at {train_path}"
print(f"✓ Training directory exists")

# Count images
train_ids = [d for d in train_path.iterdir() if d.is_dir()]
test_ids = [d for d in test_path.iterdir() if d.is_dir()]

print(f"\\nDataset statistics:")
print(f"  Training images: {len(train_ids)}")
print(f"  Test images: {len(test_ids)}")

# Validate structure
first_train = train_ids[0]
img_file = list((first_train / 'images').glob('*.png'))[0]
masks_dir = first_train / 'masks'
mask_files = list(masks_dir.glob('*.png'))

print(f"\\nFirst training sample validation:")
print(f"  Image: {img_file.name}")
print(f"  Mask count: {len(mask_files)}")
print(f"  ✓ Structure valid")
"""))

# === SECTION 13: EDA ===
nb.cells.append(nbf.v4.new_code_cell("""# Exploratory Data Analysis
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

train_ids_sample = np.random.choice(len(train_ids), min(9, len(train_ids)), replace=False)

for idx, sample_idx in enumerate(train_ids_sample):
    sample_id = train_ids[sample_idx].name
    img_path = train_ids[sample_idx] / 'images' / f'{sample_id}.png'
    
    img = cv2.imread(str(img_path))
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load and combine masks
        mask_files = list((train_ids[sample_idx] / 'masks').glob('*.png'))
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), 0)
            combined_mask = np.maximum(combined_mask, mask)
        
        # Display
        axes[idx].imshow(img_rgb)
        axes[idx].contour(combined_mask > 0, colors='red', linewidths=1)
        axes[idx].set_title(f'{sample_id}\\nNuclei: {len(mask_files)}')
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig(SAVE_DIR / 'eda_samples.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ EDA visualization saved")
"""))

# === SECTION 14: Train/Val/Test Split ===
nb.cells.append(nbf.v4.new_code_cell("""# Split data
np.random.seed(SEED)
train_ids_shuffled = np.random.permutation(train_ids)

split_idx = int(0.8 * len(train_ids))
train_ids_split = train_ids_shuffled[:split_idx]
val_ids_split = train_ids_shuffled[split_idx:]
test_ids_split = test_ids

print(f"Train/Val/Test split:")
print(f"  Train: {len(train_ids_split)} ({len(train_ids_split)*100/len(train_ids):.1f}%)")
print(f"  Val: {len(val_ids_split)} ({len(val_ids_split)*100/len(train_ids):.1f}%)")
print(f"  Test: {len(test_ids_split)}")
"""))

# === SECTION 15: Preprocessing & Augmentation ===
nb.cells.append(nbf.v4.new_code_cell("""# Augmentation pipelines (without ToTensorV2 for masks)
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Blur(p=0.2),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=None)

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=None)

print("✓ Augmentation pipelines created")
"""))

# === SECTION 16: Baseline Model ===
nb.cells.append(nbf.v4.new_code_cell("""# Classical baseline: Otsu thresholding
def otsu_baseline(img_path, target_size=IMG_SIZE):
    img = cv2.imread(str(img_path), 0)
    img_resized = cv2.resize(img, (target_size, target_size))
    _, binary = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0

# Test baseline
baseline_iou_scores = []
test_sample_size = min(5, len(test_ids_split))
for i in range(test_sample_size):
    sample_id = test_ids_split[i]
    img_path = sample_id / 'images' / f'{sample_id.name}.png'
    mask_files = list((sample_id / 'masks').glob('*.png'))
    
    if mask_files:
        pred_mask = otsu_baseline(img_path, IMG_SIZE)
        true_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), 0)
            mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            true_mask = np.maximum(true_mask, (mask_resized > 0).astype(np.float32))
        
        # Ensure same shape
        assert pred_mask.shape == true_mask.shape
        iou = jaccard_score(true_mask.ravel() > 0.5, pred_mask.ravel() > 0.5, average='binary')
        baseline_iou_scores.append(iou)

baseline_iou_mean = np.mean(baseline_iou_scores) if baseline_iou_scores else 0
print(f"Baseline (Otsu) mean IoU: {baseline_iou_mean:.4f}")
"""))

# === SECTION 17: Dataset Class ===
nb.cells.append(nbf.v4.new_code_cell("""# PyTorch Dataset
class NucleiDataset(Dataset):
    def __init__(self, image_ids, transform=None):
        self.image_ids = image_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        sample_id = self.image_ids[idx].name
        img_path = self.image_ids[idx] / 'images' / f'{sample_id}.png'
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load and combine masks
        mask_files = list((self.image_ids[idx] / 'masks').glob('*.png'))
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask_file in mask_files:
            m = cv2.imread(str(mask_file), 0)
            mask = np.maximum(mask, m)
        mask = (mask > 0).astype(np.uint8)
        
        # Apply augmentation
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            # mask is already a tensor from ToTensorV2
        else:
            mask = torch.from_numpy(mask).float()
        
        return img, mask.unsqueeze(0)

# Create dataloaders
train_dataset = NucleiDataset(train_ids_split, transform=train_transform)
val_dataset = NucleiDataset(val_ids_split, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✓ DataLoaders created")
print(f"  Train: {len(train_loader)} batches")
print(f"  Val: {len(val_loader)} batches")
"""))

# === SECTION 18: Model & Training ===
nb.cells.append(nbf.v4.new_code_cell("""# Model
model = smp.Unet(
    encoder_name='efficientnet-b1',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    decoder_attention_type='scse'
).to(DEVICE)

# Loss & Optimizer
dice_loss = smp.losses.DiceLoss(mode='binary')
focal_loss = smp.losses.FocalLoss(mode='binary')
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = 0.5 * dice_loss(logits, masks) + 0.5 * focal_loss(logits, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def val_epoch(model, loader, device):
    model.eval()
    total_dice = 0
    total_iou = 0
    count = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            dice = 2 * (preds * masks).sum() / (preds.sum() + masks.sum() + 1e-6)
            iou = (preds * masks).sum() / ((preds + masks - preds * masks).sum() + 1e-6)
            
            total_dice += dice.item()
            total_iou += iou.item()
            count += 1
    
    return total_dice / count, total_iou / count

# Training
history = {'train_loss': [], 'val_dice': [], 'val_iou': []}
best_iou = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    val_dice, val_iou = val_epoch(model, val_loader, DEVICE)
    
    history['train_loss'].append(train_loss)
    history['val_dice'].append(val_dice)
    history['val_iou'].append(val_iou)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: loss={train_loss:.4f}, dice={val_dice:.4f}, iou={val_iou:.4f}")
    
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), CHECKPOINT_DIR / 'best_nuclei_unet.pth')

print("✓ Training complete")
"""))

# === SECTION 19: Inference & Evaluation ===
nb.cells.append(nbf.v4.new_code_cell("""# Load best model
model.load_state_dict(torch.load(CHECKPOINT_DIR / 'best_nuclei_unet.pth'))
model.eval()

# Instance metrics helper
def compute_instance_metrics(pred_mask, true_mask):
    pred_labeled = label(pred_mask > 0)
    true_labeled = label(true_mask > 0)
    
    pred_nuclei = regionprops(pred_labeled)
    true_nuclei = regionprops(true_labeled)
    
    if len(true_nuclei) == 0:
        return 0, 0
    
    matches = 0
    for true_region in true_nuclei:
        true_mask_single = (true_labeled == true_region.label).astype(float)
        for pred_region in pred_nuclei:
            pred_mask_single = (pred_labeled == pred_region.label).astype(float)
            iou = (true_mask_single * pred_mask_single).sum() / (true_mask_single + pred_mask_single - true_mask_single * pred_mask_single).sum()
            if iou > 0.5:
                matches += 1
                break
    
    ap = matches / len(true_nuclei)
    return ap, len(true_nuclei)

# Test evaluation
test_results = {'iou': [], 'dice': [], 'ap': [], 'nuclei_count': []}

for sample_id in test_ids_split:
    img_path = sample_id / 'images' / f'{sample_id.name}.png'
    
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = val_transform(image=img)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logit = model(img_tensor)
        pred_mask = (torch.sigmoid(logit) > 0.5).cpu().numpy()[0, 0]
    
    # True mask
    mask_files = list((sample_id / 'masks').glob('*.png'))
    true_mask = np.zeros_like(pred_mask)
    for mask_file in mask_files:
        m = cv2.imread(str(mask_file), 0)
        m_resized = cv2.resize(m, (IMG_SIZE, IMG_SIZE))
        true_mask = np.maximum(true_mask, m_resized // 255)
    
    # Metrics
    iou = jaccard_score(true_mask.ravel() > 0.5, pred_mask.ravel() > 0.5, average='binary')
    dice = 2 * (pred_mask * true_mask).sum() / (pred_mask.sum() + true_mask.sum() + 1e-6)
    ap, nc = compute_instance_metrics(pred_mask, true_mask)
    
    test_results['iou'].append(iou)
    test_results['dice'].append(dice)
    test_results['ap'].append(ap)
    test_results['nuclei_count'].append(nc)

print("\\nTest evaluation:")
print(f"  IoU: {np.mean(test_results['iou']):.4f}")
print(f"  Dice: {np.mean(test_results['dice']):.4f}")
print(f"  AP: {np.mean(test_results['ap']):.4f}")
print(f"  Avg nuclei per image: {np.mean(test_results['nuclei_count']):.1f}")
"""))

# === SECTION 20: Qualitative Review ===
nb.cells.append(nbf.v4.new_code_cell("""# Mask review
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

worst_indices = np.argsort(test_results['iou'])[:min(9, len(test_results))]

for idx, worst_idx in enumerate(worst_indices):
    sample_id = test_ids_split[worst_idx]
    img_path = sample_id / 'images' / f'{sample_id.name}.png'
    
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = val_transform(image=img)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logit = model(img_tensor)
        pred_mask = (torch.sigmoid(logit) > 0.5).cpu().numpy()[0, 0]
    
    mask_files = list((sample_id / 'masks').glob('*.png'))
    true_mask = np.zeros_like(pred_mask)
    for mask_file in mask_files:
        m = cv2.imread(str(mask_file), 0)
        m_resized = cv2.resize(m, (IMG_SIZE, IMG_SIZE))
        true_mask = np.maximum(true_mask, m_resized // 255)
    
    axes[idx].imshow(img)
    axes[idx].contour(true_mask, colors='green', linewidths=2)
    axes[idx].contour(pred_mask, colors='red', linewidths=2)
    axes[idx].set_title(f'IoU={test_results["iou"][worst_idx]:.3f}')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(SAVE_DIR / 'qualitative_masks.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Qualitative review saved")
"""))

# === SECTION 21-23: Remaining markdown cells ===
for md_cell in [
    """## Limitations

1. **Synthetic data:** This demonstration uses synthetic nuclei, not real microscopy
2. **Small test set:** Only 3 test images for quick validation
3. **Limited epochs:** 3 epochs for demonstration; production needs 20+
4. **Simple instance matching:** Overlap > 0.5 is heuristic
5. **GPU memory:** EfficientNet-B1 still requires significant memory
6. **Generalization:** Real DSB2018 data has more variation
""",
    """## How to Improve

1. Use real DSB2018 dataset (requires Kaggle authentication)
2. Train for 20+ epochs with learning rate scheduling
3. Use larger encoders (EfficientNet-B3, ResNet50)
4. Implement advanced post-processing (morphological operations)
5. Add test-time augmentation
6. Use ensemble of multiple models
7. Fine-tune on your own microscopy images
""",
    """## Key Takeaways

1. Instance segmentation = semantic segmentation + instance identification
2. Dice + Focal loss helps with small object detection
3. Post-processing refines predictions significantly
4. Classical baselines (Otsu) remain useful comparison points
5. Synthetic data enables rapid prototyping
6. Modern encoders (EfficientNet) improve efficiency
"""
]:
    nb.cells.append(nbf.v4.new_markdown_cell(md_cell))

# Save
nb_path = Path('cell_nuclei_segmentation_pipeline.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✓ Notebook created with {len(nb.cells)} cells")
