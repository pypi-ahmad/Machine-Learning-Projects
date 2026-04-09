"""Image Classification pipeline template: DINOv3/ConvNeXt V2 — April 2026"""
import textwrap


def generate(project_path, config):
    model_name = config.get("model", "facebook/dinov2-base")
    n_classes = config.get("n_classes", 10)

    return textwrap.dedent(f'''\
        """
        Modern Image Classification Pipeline (April 2026)
        Models: DINOv2 (vision transformer) + ConvNeXt V2 fine-tuning
        """
        import os, warnings
        import numpy as np
        from pathlib import Path
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image
        from sklearn.metrics import accuracy_score, classification_report
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        warnings.filterwarnings("ignore")

        MODEL_NAME = "{model_name}"
        N_CLASSES = {n_classes}
        BATCH_SIZE = 32
        EPOCHS = 10
        LR = 1e-4
        IMG_SIZE = 224


        class ImageFolderDataset(Dataset):
            """Load images from folder structure: root/class_name/image.jpg"""
            def __init__(self, root_dir, transform=None):
                self.root = Path(root_dir)
                self.transform = transform
                self.samples = []
                self.class_names = []

                # Find image directories
                for class_dir in sorted(self.root.iterdir()):
                    if class_dir.is_dir():
                        self.class_names.append(class_dir.name)
                        for img_path in class_dir.glob("*"):
                            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                                self.samples.append((str(img_path), len(self.class_names) - 1))

                if not self.samples:
                    # Try flat directory with filename-based labels
                    for img_path in self.root.glob("**/*"):
                        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                            self.samples.append((str(img_path), 0))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label


        def get_transforms(train=True):
            if train:
                return transforms.Compose([
                    transforms.RandomResizedCrop(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])


        def find_data_dir():
            """Find image data directory in the project."""
            project_dir = Path(os.path.dirname(__file__))
            # Common patterns
            for name in ["train", "data", "dataset", "images", "Training", "Train"]:
                p = project_dir / name
                if p.exists() and p.is_dir():
                    return str(p)
            # Any subdirectory with images
            for d in project_dir.iterdir():
                if d.is_dir() and any(d.glob("**/*.jpg")) or any(d.glob("**/*.png")):
                    return str(d)
            return str(project_dir)


        def train_dino():
            """Fine-tune DINOv2 for image classification."""
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {{device}}")

            data_dir = find_data_dir()
            print(f"Data directory: {{data_dir}}")

            train_ds = ImageFolderDataset(data_dir, get_transforms(train=True))
            n_classes = max(len(train_ds.class_names), N_CLASSES)
            print(f"Found {{len(train_ds)}} images, {{n_classes}} classes")
            print(f"Classes: {{train_ds.class_names[:10]}}...")

            if len(train_ds) == 0:
                print("⚠ No images found. Ensure data is in folder structure: data/class_name/image.jpg")
                return None

            # Split train/val
            val_size = max(1, int(0.2 * len(train_ds)))
            train_size = len(train_ds) - val_size
            train_subset, val_subset = torch.utils.data.random_split(
                train_ds, [train_size, val_size]
            )

            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, num_workers=0)

            # Load DINOv2 backbone
            try:
                from transformers import AutoModel
                backbone = AutoModel.from_pretrained(MODEL_NAME)
                embed_dim = backbone.config.hidden_size
            except Exception:
                backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
                embed_dim = 768

            # Classification head
            class DinoClassifier(nn.Module):
                def __init__(self, backbone, embed_dim, n_classes):
                    super().__init__()
                    self.backbone = backbone
                    self.head = nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, 256),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, n_classes),
                    )
                    # Freeze backbone initially
                    for param in self.backbone.parameters():
                        param.requires_grad = False

                def forward(self, x):
                    features = self.backbone(x)
                    if hasattr(features, "last_hidden_state"):
                        cls_token = features.last_hidden_state[:, 0]
                    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                        cls_token = features.pooler_output
                    else:
                        cls_token = features
                    return self.head(cls_token)

            model = DinoClassifier(backbone, embed_dim, n_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=0.01)

            # Train
            best_acc = 0
            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()

                # Unfreeze backbone after epoch 2
                if epoch == 2:
                    for param in model.backbone.parameters():
                        param.requires_grad = True
                    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=0.01)

                # Validate
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(device)
                        outputs = model(imgs)
                        preds = torch.argmax(outputs, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.numpy())

                val_acc = accuracy_score(all_labels, all_preds)
                print(f"  Epoch {{epoch+1}}/{{EPOCHS}}, Loss: {{total_loss/len(train_loader):.4f}}, Val Acc: {{val_acc:.4f}}")

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "best_model.pth"))

            print(f"\\n✓ DINOv2 Best Val Accuracy: {{best_acc:.4f}}")
            return {{"accuracy": best_acc, "model": model}}


        def main():
            print("=" * 60)
            print("MODERN IMAGE CLASSIFICATION PIPELINE")
            print(f"Model: {{MODEL_NAME}} (DINOv2)")
            print("=" * 60)
            result = train_dino()
            if result:
                print(f"\\n🏆 Best Accuracy: {{result['accuracy']:.4f}}")


        if __name__ == "__main__":
            main()
    ''')
