"""
Modern Image Classification Pipeline (April 2026)
Model: DINOv3 (primary backbone) + ConvNeXt V2 (fine-tuning backbone)
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 224, 32, 10, 1e-4


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from datasets import load_dataset as _hf_load
    hf_ds = _hf_load("Antoinegg1/fingerprint", split="train")
    # Convert HF image dataset to torchvision-style
    class HFImageDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.ds = hf_dataset; self.transform = transform
            img_col = next((c for c in hf_dataset.column_names if "image" in c.lower()), hf_dataset.column_names[0])
            lbl_col = next((c for c in hf_dataset.column_names if "label" in c.lower()), hf_dataset.column_names[-1])
            self.img_col, self.lbl_col = img_col, lbl_col
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img = self.ds[i][self.img_col].convert("RGB") if hasattr(self.ds[i][self.img_col], "convert") else Image.open(self.ds[i][self.img_col]).convert("RGB")
            lbl = self.ds[i][self.lbl_col]
            return self.transform(img) if self.transform else img, lbl
    train_ds = HFImageDataset(hf_ds, transform=get_transforms(True))
    n_classes = len(set(hf_ds[next(c for c in hf_ds.column_names if "label" in c.lower())]))

    val_size = max(1, int(0.2 * len(train_ds)))
    train_sub, val_sub = random_split(train_ds, [len(train_ds) - val_size, val_size])
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, num_workers=0)

    backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vits14", pretrained=True)
    embed_dim = 384  # ViT-S/14

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 256),
                                      nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
            for p in self.backbone.parameters(): p.requires_grad = False
        def forward(self, x):
            feat = self.backbone(x)
            return self.head(feat)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=0.01)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels); loss.backward()
            opt.step(); opt.zero_grad(); total_loss += loss.item()
        if epoch == 2:
            for p in model.backbone.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=0.01)
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds.extend(torch.argmax(model(imgs.to(device)), dim=-1).cpu().numpy())
                gts.extend(labels.numpy())
        val_acc = accuracy_score(gts, preds)
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f} — Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "best_model.pth"))

    print(f"\n🏆 DINOv3 Best Val Accuracy: {best_acc:.4f}")

    # ConvNeXt V2 (alternative fine-tuning backbone)
    try:
        import timm
        convnext = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=n_classes).to(device)
        convnext_opt = torch.optim.AdamW(convnext.parameters(), lr=LR * 0.5, weight_decay=0.01)
        for epoch in range(3):
            convnext.train(); total_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                loss = criterion(convnext(imgs), labels); loss.backward()
                convnext_opt.step(); convnext_opt.zero_grad(); total_loss += loss.item()
        convnext.eval(); cv_preds, cv_gts = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                cv_preds.extend(torch.argmax(convnext(imgs.to(device)), dim=-1).cpu().numpy())
                cv_gts.extend(labels.numpy())
        cv_acc = accuracy_score(cv_gts, cv_preds)
        print(f"✓ ConvNeXt V2 Val Accuracy: {cv_acc:.4f}")
    except Exception as e:
        print(f"✗ ConvNeXt V2: {e}")


def main():
    print("=" * 60)
    print("IMAGE CLASSIFICATION — DINOv3 + ConvNeXt V2")
    print("=" * 60)
    train_model()


if __name__ == "__main__":
    main()
