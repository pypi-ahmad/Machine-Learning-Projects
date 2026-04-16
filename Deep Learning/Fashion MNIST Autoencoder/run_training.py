#!/usr/bin/env python
"""
Fashion MNIST Convolutional Autoencoder
Dataset: FashionMNIST from https://github.com/zalandoresearch/fashion-mnist
Trains a convolutional autoencoder for image reconstruction.
"""
import subprocess, sys

def install_if_missing(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

install_if_missing("torch")
install_if_missing("torchvision")
install_if_missing("scikit-image", "skimage")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

print(f"PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}")

# Configuration
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
METRICS_FILE = PROJECT_ROOT / "metrics.json"
DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5 if torch.cuda.is_available() else 1
BOTTLENECK_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")

# Download FashionMNIST from https://github.com/zalandoresearch/fashion-mnist
print("Downloading FashionMNIST from https://github.com/zalandoresearch/fashion-mnist...")
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = FashionMNIST(root=str(DATA_DIR), train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root=str(DATA_DIR), train=False, transform=transform, download=True)
print(f"Train: {len(train_dataset)}; Test: {len(test_dataset)}")

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Split: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")

# ConvAutoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=32):
        super().__init__()
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_encode = nn.Linear(32 * 7 * 7, bottleneck_dim)
        self.fc_decode = nn.Linear(bottleneck_dim, 32 * 7 * 7)
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.ReLU()
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc_conv1(x)
        h = self.enc_conv2(h)
        return self.fc_encode(h.view(h.size(0), -1))

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 32, 7, 7)
        h = self.dec_conv1(h)
        return self.dec_conv2(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

model = ConvAutoencoder(BOTTLENECK_DIM).to(DEVICE)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        reconstructed, _ = model(x.to(device))
        loss = criterion(reconstructed, x.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, _ in loader:
            reconstructed, _ = model(x.to(device))
            total_loss += criterion(reconstructed, x.to(device)).item()
    return total_loss / len(loader)

best_val_loss = float("inf")
best_model_path = CHECKPOINT_DIR / "best_autoencoder.pth"
print(f"Training {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = eval_epoch(model, val_loader, criterion, DEVICE)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), str(best_model_path))
    print(f"  Epoch {epoch+1}: train={train_loss:.6f} val={val_loss:.6f}")

model.load_state_dict(torch.load(str(best_model_path)))
print(f"Best val: {best_val_loss:.6f}")

# Evaluation
def compute_metrics(model, loader, device):
    model.eval()
    mse_data = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            reconstructed, _ = model(x)
            mse_data.extend(torch.mean((reconstructed - x) ** 2, dim=(1, 2, 3)).cpu().numpy())
    return np.array(mse_data)

print("Computing metrics...")
test_mse = compute_metrics(model, test_loader, DEVICE)
print(f"Test MSE: {test_mse.mean():.6f} +/- {test_mse.std():.6f}")

# Export metrics
results = {
    "dataset": "FashionMNIST",
    "dataset_source": "https://github.com/zalandoresearch/fashion-mnist",
    "device": str(DEVICE),
    "epochs": NUM_EPOCHS,
    "bottleneck_dim": BOTTLENECK_DIM,
    "test_mse_mean": float(test_mse.mean()),
    "test_mse_std": float(test_mse.std()),
    "best_val_loss": float(best_val_loss)
}
with open(METRICS_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"Metrics exported to {METRICS_FILE}")
print("EXECUTION_COMPLETE")
