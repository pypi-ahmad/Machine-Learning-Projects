#!/usr/bin/env python
"""
Project 22: Polyp Segmentation with MONAI - Simplified End-to-End Pipeline
"""

import os, json, random, numpy as np, cv2
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from torch.optim import Adam
import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt
matplotlib_backend = 'Agg'

# Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

REPO_ROOT = Path.cwd().parents[2] if 'Computer Vision' in str(Path.cwd()) else Path.cwd()
DATA_DIR = str(REPO_ROOT / 'data' / 'Kvasir-SEG' / 'kvasir-seg')
SAVE_DIR = str(Path.cwd())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}, Save: {SAVE_DIR}")

# Load dataset
img_dir = f"{DATA_DIR}/images"
mask_dir = f"{DATA_DIR}/masks"
img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
print(f"Dataset: {len(img_files)} images")

# Create data lists
data_list = [{'image': f"{img_dir}/{f}", 'label': f"{mask_dir}/{f}"} for f in img_files if os.path.exists(f"{mask_dir}/{f}")]
random.shuffle(data_list)
n_train, n_val = int(len(data_list) * 0.70), int(len(data_list) * 0.15)
train_data, val_data, test_data = data_list[:n_train], data_list[n_train:n_train+n_val], data_list[n_train+n_val:]
print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Simple dataset
class PolypDS(torch.utils.data.Dataset):
    def __init__(self, data, train=False):
        self.data, self.train = data, train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        img = np.array(Image.open(d['image']).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(d['label']).convert('L'), dtype=np.float32) / 255.0
        
        # Resize
        img = cv2.resize(img, (352, 352))
        mask = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
        
        # Augment
        if self.train and np.random.rand() > 0.5:
            img, mask = img[::-1], mask[::-1]
        
        img = (img / 127.5) - 1.0
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        mask = torch.from_numpy(mask[np.newaxis, :, :]).float()
        return {'image': img, 'label': mask}

# Loaders
train_loader = torch.utils.data.DataLoader(PolypDS(train_data, train=True), 16, shuffle=True)
val_loader = torch.utils.data.DataLoader(PolypDS(val_data), 16)
test_loader = torch.utils.data.DataLoader(PolypDS(test_data), 16)

# Model
model = UNet(spatial_dims=2, in_channels=3, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2, norm='instance').to(device)
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
ce_loss = nn.BCEWithLogitsLoss()
opt = Adam(model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

# Training
def train_epoch():
    model.train()
    for b in train_loader:
        img, mask = b['image'].to(device), b['label'].to(device)
        logits = model(img)
        loss = 0.7 * dice_loss(torch.sigmoid(logits), mask) + 0.3 * ce_loss(logits, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

def val_epoch():
    model.eval()
    dices = []
    with torch.no_grad():
        for b in val_loader:
            img, mask = b['image'].to(device), b['label'].to(device)
            pred = torch.sigmoid(model(img))
            for i in range(img.shape[0]):
                p, m = pred[i, 0].cpu().numpy(), mask[i, 0].cpu().numpy()
                dice = 2 * np.sum(p * m) / (np.sum(p) + np.sum(m) + 1e-7)
                dices.append(dice)
    return np.mean(dices) if dices else 0.0

print("\nTraining 20 epochs...")
best_dice, best_epoch = 0, 0
for epoch in range(20):
    train_epoch()
    vdice = val_epoch()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: Dice={vdice:.4f}")
    if vdice > best_dice:
        best_dice, best_epoch = vdice, epoch
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_polyp_model.pth")
    sch.step(vdice)

print(f"\n✓ Best: {best_dice:.4f} at epoch {best_epoch+1}")

# Test
model.load_state_dict(torch.load(f"{SAVE_DIR}/best_polyp_model.pth", map_location=device))
model.eval()
dice_scores, iou_scores = [], []

with torch.no_grad():
    for b in test_loader:
        img, mask = b['image'].to(device), b['label'].to(device)
        pred = torch.sigmoid(model(img)).cpu().numpy()
        tgt = mask.cpu().numpy()
        for i in range(img.shape[0]):
            p, m = pred[i, 0], tgt[i, 0]
            pd, mt = (p > 0.5).astype(np.float32), (m > 0).astype(np.float32)
            dice = 2 * np.sum(pd * mt) / (np.sum(pd) + np.sum(mt) + 1e-7)
            iou = np.sum(pd * mt) / (np.sum((pd + mt) > 0.5) + 1e-7)
            dice_scores.append(dice)
            iou_scores.append(iou)

print(f"\nTest Metrics:")
print(f"  Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
print(f"  IoU:  {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")

# Save metrics
metrics = {
    'project': 'Polyp Segmentation (Project 22)',
    'framework': 'MONAI',
    'dataset': 'Kvasir-SEG (1000 images)',
    'model': 'UNet',
    'test_dice_mean': float(np.mean(dice_scores)),
    'test_iou_mean': float(np.mean(iou_scores)),
    'best_val_dice': float(best_dice),
}

with open(f"{SAVE_DIR}/metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ All artifacts saved to {SAVE_DIR}")
print("✅ PROJECT 22 COMPLETE")
