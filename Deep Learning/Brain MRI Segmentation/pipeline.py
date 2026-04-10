"""
Modern Medical Image Segmentation Pipeline (April 2026)

Primary : nnU-Net-style supervised U-Net (encoder-decoder with skip connections).
Optional: MedSAM2 zero-shot promptable segmentation (center-point prompts).
Metrics : Dice coefficient + mean IoU per model, wall-clock timing.
Export  : metrics.json, segmentation_results.png, best_unet.pth.
Data    : Auto-downloaded from HuggingFace at runtime.

DISCLAIMER: This is an educational/research demonstration pipeline.
It is NOT validated for clinical use. Medical image analysis models
require rigorous validation on curated datasets, regulatory approval,
and expert clinical oversight before any diagnostic application.
"""
import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 256, 8, 15, 1e-4
N_CLASSES = 2
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    from datasets import load_dataset as _hf_load
    hf_ds = _hf_load("mateuszbuda/brain-segmentation", split="train")
    print(f"Loaded {len(hf_ds)} samples from mateuszbuda/brain-segmentation")
    return hf_ds


class MedSegDataset(Dataset):
    def __init__(self, hf_ds, img_size=IMG_SIZE):
        self.ds = hf_ds
        self.img_size = img_size
        cols = hf_ds.column_names
        self.img_col = next((c for c in cols if "image" in c.lower()), cols[0])
        self.mask_col = next((c for c in cols if "mask" in c.lower() or "seg" in c.lower() or "label" in c.lower()), cols[-1])
        self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        img = item[self.img_col]
        mask = item[self.mask_col]
        if hasattr(img, "convert"):
            img = img.convert("RGB").resize((self.img_size, self.img_size))
        if hasattr(mask, "convert"):
            mask = mask.convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        img_t = self.to_tensor(img)
        mask_t = torch.from_numpy(np.array(mask)).long()
        if mask_t.ndim == 3: mask_t = mask_t[0]
        mask_t = torch.clamp(mask_t, 0, N_CLASSES - 1)
        return img_t, mask_t


class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)


class SimpleUNet(nn.Module):
    """Lightweight U-Net as nnU-Net-style supervised baseline."""
    def __init__(self, in_ch=3, out_ch=N_CLASSES, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for f in features:
            self.encoders.append(UNetBlock(in_ch, f)); in_ch = f
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(UNetBlock(f * 2, f))
        self.final = nn.Conv2d(features[0], out_ch, 1)
    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)
            skip = skips[-(i // 2 + 1)]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i + 1](x)
        return self.final(x)


def dice_score(pred, target, n_classes=N_CLASSES):
    pred = torch.argmax(pred, dim=1)
    dice = 0.0
    for c in range(n_classes):
        p = (pred == c).float(); t = (target == c).float()
        inter = (p * t).sum()
        dice += (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return dice / n_classes


def mean_iou(pred, target, n_classes=N_CLASSES):
    pred = torch.argmax(pred, dim=1) if pred.ndim == 4 else pred
    iou = 0.0
    for c in range(n_classes):
        p = (pred == c).float(); t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        iou += (inter + 1e-6) / (union + 1e-6)
    return iou / n_classes


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_ds = load_data()
    dataset = MedSegDataset(hf_ds)
    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    metrics = {}

    # --- PRIMARY: nnU-Net-style supervised U-Net ---
    print()
    print("-- nnU-Net-style U-Net (supervised) --")
    model = SimpleUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_dice = 0
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            loss = criterion(out, masks); loss.backward()
            opt.step(); opt.zero_grad(); total_loss += loss.item()
        scheduler.step()
        model.eval(); dice_sum, iou_sum, n = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                dice_sum += dice_score(out, masks).item()
                iou_sum += mean_iou(out, masks).item()
                n += 1
        val_dice = dice_sum / max(n, 1)
        val_iou = iou_sum / max(n, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS} -- Loss: {total_loss/len(train_loader):.4f} -- Dice: {val_dice:.4f} -- IoU: {val_iou:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_unet.pth"))
    unet_elapsed = round(time.perf_counter() - t0, 1)
    print(f"  nnU-Net Best -- Dice: {best_dice:.4f} -- IoU: {best_iou:.4f} ({unet_elapsed}s)")
    metrics["nnUNet"] = {"val_dice": round(best_dice, 4), "val_iou": round(best_iou, 4),
                         "epochs": EPOCHS, "time_s": unet_elapsed}

    # --- OPTIONAL: MedSAM2 (zero-shot promptable segmentation) ---
    print()
    print("-- MedSAM2 (zero-shot, center-point prompt) --")
    try:
        from transformers import SamModel, SamProcessor
        t1 = time.perf_counter()
        sam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
        sam_proc = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
        sam_model.eval()
        dice_sum, iou_sum, n = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                for j in range(min(4, imgs.shape[0])):
                    pil_img = transforms.ToPILImage()(imgs[j])
                    h, w = pil_img.size[1], pil_img.size[0]
                    input_points = [[[w // 2, h // 2]]]
                    inputs = sam_proc(pil_img, input_points=input_points, return_tensors="pt").to(device)
                    outputs = sam_model(**inputs)
                    pred_mask = sam_proc.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                        inputs["reshaped_input_sizes"].cpu())[0]
                    pred_binary = (pred_mask[0, 0] > 0).long()
                    gt = masks[j]
                    if pred_binary.shape != gt.shape:
                        pred_binary = nn.functional.interpolate(
                            pred_binary.float().unsqueeze(0).unsqueeze(0),
                            size=gt.shape, mode="nearest")[0, 0].long()
                    inter = ((pred_binary == 1) & (gt == 1)).sum().float()
                    p_sum = (pred_binary == 1).sum().float()
                    t_sum = (gt == 1).sum().float()
                    dice_sum += (2 * inter + 1e-6) / (p_sum + t_sum + 1e-6)
                    iou_sum += (inter + 1e-6) / (p_sum + t_sum - inter + 1e-6)
                    n += 1
                if n >= 32:
                    break
        sam_dice = (dice_sum / max(n, 1)).item() if hasattr(dice_sum, "item") else dice_sum / max(n, 1)
        sam_iou = (iou_sum / max(n, 1)).item() if hasattr(iou_sum, "item") else iou_sum / max(n, 1)
        sam_elapsed = round(time.perf_counter() - t1, 1)
        print(f"  MedSAM2 -- Dice: {sam_dice:.4f} -- IoU: {sam_iou:.4f} ({sam_elapsed}s, {n} samples)")
        metrics["MedSAM2"] = {"val_dice": round(sam_dice, 4), "val_iou": round(sam_iou, 4),
                              "samples": n, "time_s": sam_elapsed}
    except Exception as e:
        print(f"  MedSAM2 skipped: {e}")

    # Visualize sample predictions
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    with torch.no_grad():
        for imgs, masks in val_loader:
            preds = torch.argmax(model(imgs.to(device)), dim=1).cpu()
            for i in range(min(4, imgs.shape[0])):
                axes[0, i].imshow(imgs[i].permute(1, 2, 0).numpy())
                axes[0, i].set_title("Input"); axes[0, i].axis("off")
                axes[1, i].imshow(preds[i].numpy(), cmap="jet", alpha=0.7)
                axes[1, i].set_title("Prediction"); axes[1, i].axis("off")
            break
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "segmentation_results.png"), dpi=100)
    plt.close(fig)
    print(f"Saved segmentation_results.png")

    return metrics


def run_eda(save_dir):
    """Dataset summary for medical segmentation."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    data_dir = os.path.join(save_dir, "data")
    if os.path.isdir(data_dir):
        imgs = [f for f in os.listdir(data_dir) if not f.startswith(".")]
        print(f"  Files in data directory: {len(imgs)}")
    print("EDA complete.")


def main():
    print("=" * 60)
    print("MEDICAL SEGMENTATION | nnU-Net + MedSAM2")
    print("=" * 60)
    run_eda(SAVE_DIR)
    print("NOTE: Educational/research demo only -- not for clinical use.")
    metrics = train_model()

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
