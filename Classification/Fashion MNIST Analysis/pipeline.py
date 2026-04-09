"""
Modern Image Classification Pipeline (April 2026)
Model: DINOv2 fine-tuning — Vision Transformer
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
    from torchvision import datasets as tv_datasets
    train_ds = tv_datasets.FashionMNIST(root="./data", train=True, download=True, transform=get_transforms(True))
    n_classes = 10

    val_size = max(1, int(0.2 * len(train_ds)))
    train_sub, val_sub = random_split(train_ds, [len(train_ds) - val_size, val_size])
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, num_workers=0)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
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

    print(f"\n🏆 DINOv2 Best Val Accuracy: {best_acc:.4f}")


def main():
    print("=" * 60)
    print("IMAGE CLASSIFICATION — DINOv2 (ViT-S/14)")
    print("=" * 60)
    train_model()


if __name__ == "__main__":
    main()
