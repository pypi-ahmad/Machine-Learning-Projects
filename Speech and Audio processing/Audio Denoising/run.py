#!/usr/bin/env python3
"""
Audio Denoising — U-Net Spectrogram Denoiser (PyTorch)
=======================================================
Train a U-Net model to denoise audio by operating on mel-spectrogram
representations.  Clean audio is corrupted with synthetic additive
Gaussian noise to create training pairs.

Dataset
-------
* Kaggle: https://www.kaggle.com/datasets/sayuksh/denoising-audio-collection

Run
---
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    get_device,
    ensure_dir,
    dataset_prompt,
    parse_common_args,
    save_metrics,
    run_metadata,
    dataset_fingerprint,
    write_split_manifest,
    dataset_missing_metrics,
    missing_dependency_metrics,
    safe_import_available,
    resolve_device_from_args,
    configure_cuda_allocator,
    EarlyStopping,
)

logger = logging.getLogger(__name__)

# Check torchaudio availability once (import can hard-crash the process)
_TORCHAUDIO_OK = safe_import_available("torchaudio")

# ── Configuration ────────────────────────────────────────────
KAGGLE_SLUG = "sayuksh/denoising-audio-collection"
SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
NOISE_STD = 0.05          # additive Gaussian noise σ
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3
SPEC_MAX_LEN = 128        # fixed time-frame length for batching


# ═════════════════════════════════════════════════════════════
#  Audio I/O helpers
# ═════════════════════════════════════════════════════════════

def load_audio(filepath: Path, sr: int = SAMPLE_RATE):
    """Load an audio file and return waveform tensor + sample rate."""
    if _TORCHAUDIO_OK:
        try:
            import torchaudio
            waveform, orig_sr = torchaudio.load(str(filepath))
            if orig_sr != sr:
                resampler = torchaudio.transforms.Resample(orig_sr, sr)
                waveform = resampler(waveform)
            # mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform
        except Exception:
            pass

    # Fallback: librosa
    try:
        import librosa
        y, _ = librosa.load(str(filepath), sr=sr, mono=True)
        return torch.from_numpy(y).unsqueeze(0)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", filepath, exc)
        return None


def waveform_to_mel(waveform: torch.Tensor, device: torch.device):
    """Convert a 1-channel waveform -> log-mel spectrogram (1, n_mels, T)."""
    if _TORCHAUDIO_OK:
        try:
            import torchaudio
            mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
            ).to(device)
            mel = mel_fn(waveform.to(device))
        except Exception:
            import librosa
            y = waveform.squeeze().numpy()
            S = librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            )
            mel = torch.from_numpy(S).unsqueeze(0).to(device)
    else:
        import librosa
        y = waveform.squeeze().numpy()
        S = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        mel = torch.from_numpy(S).unsqueeze(0).to(device)

    log_mel = torch.log1p(mel)
    return log_mel


# ═════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════

def discover_audio_files(root: Path):
    """Recursively find audio files under *root*."""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return sorted(files)


class DenoisingDataset(Dataset):
    """Each item returns (noisy_mel, clean_mel) both shaped (1, N_MELS, T)."""

    def __init__(self, mel_specs: list, max_len: int = SPEC_MAX_LEN):
        self.specs = mel_specs
        self.max_len = max_len

    def __len__(self):
        return len(self.specs)

    def _pad_or_crop(self, spec):
        """Ensure time dim == max_len."""
        _, _, T = spec.shape
        if T >= self.max_len:
            return spec[:, :, : self.max_len]
        pad = self.max_len - T
        return F.pad(spec, (0, pad))

    def __getitem__(self, idx):
        clean = self._pad_or_crop(self.specs[idx])
        noise = torch.randn_like(clean) * NOISE_STD
        noisy = clean + noise
        return noisy, clean


# ═════════════════════════════════════════════════════════════
#  U-Net Model
# ═════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDenoiser(nn.Module):
    """Simple 2-D U-Net operating on (B, 1, N_MELS, T) spectrograms."""

    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    @staticmethod
    def _match(x, ref):
        """Crop/pad x so spatial dims match ref."""
        dh = x.shape[2] - ref.shape[2]
        dw = x.shape[3] - ref.shape[3]
        if dh > 0 or dw > 0:
            x = x[:, :, : ref.shape[2], : ref.shape[3]]
        elif dh < 0 or dw < 0:
            x = F.pad(x, (0, -dw if dw < 0 else 0, 0, -dh if dh < 0 else 0))
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self._match(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        return self.out_conv(d1)


# ═════════════════════════════════════════════════════════════
#  Metrics
# ═════════════════════════════════════════════════════════════

def snr_improvement(clean, noisy, denoised):
    """Compute average SNR improvement in dB over the batch."""
    def _snr(signal, noise):
        s_pow = (signal ** 2).sum(dim=(1, 2, 3))
        n_pow = (noise ** 2).sum(dim=(1, 2, 3)) + 1e-8
        return 10 * torch.log10(s_pow / n_pow)

    snr_before = _snr(clean, noisy - clean)
    snr_after = _snr(clean, denoised - clean)
    return (snr_after - snr_before).mean().item()


# ═════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════

def _pad_to_power_of_2(spec, min_pow=3):
    """Pad freq & time dims to nearest multiple of 2^min_pow for U-Net."""
    factor = 2 ** min_pow
    _, _, h, w = spec.shape
    new_h = math.ceil(h / factor) * factor
    new_w = math.ceil(w / factor) * factor
    return F.pad(spec, (0, new_w - w, 0, new_h - h)), h, w


def train(device, mel_specs, output_dir, *, epochs=None, batch_size=None,
          use_amp_override=None, patience=None):
    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    dataset = DenoisingDataset(mel_specs)

    # 70 / 15 / 15 split
    n_test = max(1, int(0.15 * len(dataset)))
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    model = UNetDenoiser(in_ch=1, base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    use_amp = use_amp_override if use_amp_override is not None else (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    es = EarlyStopping(patience=patience or 5, mode="min") if patience else None

    train_losses, val_losses = [], []
    best_val = float("inf")
    avg_snr = 0.0

    logger.info("Training U-Net denoiser — %d train / %d val / %d test samples, %d epochs",
                n_train, n_val, n_test, epochs)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            noisy_pad, oh, ow = _pad_to_power_of_2(noisy)
            clean_pad, _, _ = _pad_to_power_of_2(clean)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(noisy_pad)
                pred = pred[:, :, :oh, :ow]
                loss = criterion(pred, clean_pad[:, :, :oh, :ow])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / max(len(train_dl), 1))

        # ── Validate ─────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        snr_imp_total = 0.0
        v_count = 0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                noisy_pad, oh, ow = _pad_to_power_of_2(noisy)
                pred = model(noisy_pad)[:, :, :oh, :ow]
                clean_crop = clean[:, :, :oh, :ow]
                v_loss += criterion(pred, clean_crop).item()
                snr_imp_total += snr_improvement(clean_crop, noisy[:, :, :oh, :ow], pred)
                v_count += 1

        val_losses.append(v_loss / max(v_count, 1))
        avg_snr = snr_imp_total / max(v_count, 1)

        logger.info(
            "Epoch %d/%d — train_loss=%.5f  val_loss=%.5f  SNR_imp=%.2f dB",
            epoch, epochs, train_losses[-1], val_losses[-1], avg_snr,
        )

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            torch.save(model.state_dict(), output_dir / "unet_denoiser_best.pth")

        if es and es(val_losses[-1]):
            logger.info("Early stopping at epoch %d", epoch)
            break

    # ── Evaluate on TEST split ───────────────────────────────
    # Reload best checkpoint
    best_ckpt = output_dir / "unet_denoiser_best.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()

    test_loss = 0.0
    test_snr_total = 0.0
    test_count = 0
    with torch.no_grad():
        for noisy, clean in test_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            noisy_pad, oh, ow = _pad_to_power_of_2(noisy)
            pred = model(noisy_pad)[:, :, :oh, :ow]
            clean_crop = clean[:, :, :oh, :ow]
            test_loss += criterion(pred, clean_crop).item()
            test_snr_total += snr_improvement(clean_crop, noisy[:, :, :oh, :ow], pred)
            test_count += 1

    test_loss_avg = test_loss / max(test_count, 1)
    test_snr_avg = test_snr_total / max(test_count, 1)
    logger.info("TEST — loss=%.5f  SNR_imp=%.2f dB", test_loss_avg, test_snr_avg)

    return model, train_losses, val_losses, test_loss_avg, test_snr_avg, split_counts


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def save_plots(train_losses, val_losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Audio Denoiser Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve → %s", output_dir / "loss_curve.png")


def save_spectrogram_comparison(model, dataset, device, output_dir, n=3):
    """Save side-by-side spectrogram images: noisy | denoised | clean."""
    model.eval()
    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        noisy, clean = dataset[i]
        noisy_t = noisy.unsqueeze(0).to(device)
        noisy_pad, oh, ow = _pad_to_power_of_2(noisy_t)
        with torch.no_grad():
            denoised = model(noisy_pad)[:, :, :oh, :ow]

        for j, (img, title) in enumerate([
            (noisy.squeeze().cpu().numpy(), "Noisy"),
            (denoised.squeeze().cpu().numpy(), "Denoised"),
            (clean.squeeze().cpu().numpy(), "Clean"),
        ]):
            axes[i, j].imshow(img, aspect="auto", origin="lower", cmap="magma")
            axes[i, j].set_title(title)
            axes[i, j].set_ylabel(f"Sample {i+1}")

    fig.suptitle("Spectrogram Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "spectrogram_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved spectrogram comparison → %s",
                output_dir / "spectrogram_comparison.png")


# ═════════════════════════════════════════════════════════════
#  Synthetic-only fallback
# ═════════════════════════════════════════════════════════════

def generate_synthetic_specs(n: int = 200, device=None):
    """Create synthetic mel-like spectrograms for demonstration."""
    specs = []
    for _ in range(n):
        t = torch.linspace(0, 1, SPEC_MAX_LEN).unsqueeze(0)
        f = torch.linspace(0, 1, N_MELS).unsqueeze(1)
        base = torch.sin(2 * math.pi * (torch.randint(1, 8, (1,)).item()) * t) * f
        base = base.unsqueeze(0)  # (1, N_MELS, T)
        specs.append(base)
    return specs


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging()
    args = parse_common_args("Audio Denoising — U-Net Spectrogram Denoiser")
    set_seed(args.seed)
    configure_cuda_allocator()

    paths = project_paths(__file__)
    data_dir = paths["data"]
    output_dir = ensure_dir(paths["outputs"])
    device = resolve_device_from_args(args)

    # torchaudio can crash the process at import time (native DLL issues)
    # so we use a subprocess check; functions have internal librosa fallbacks
    if not safe_import_available("torchaudio"):
        logger.warning("torchaudio not available — will use librosa/synthetic fallback")

    # Download dataset
    try:
        ds_path = download_kaggle_dataset(
            KAGGLE_SLUG, data_dir,
            dataset_name="Denoising Audio Collection",
        )
    except (SystemExit, Exception) as exc:
        logger.warning("Dataset download failed: %s — using synthetic data", exc)
        ds_path = data_dir

    if args.download_only:
        logger.info("Download complete — exiting (--download-only).")
        sys.exit(0)

    # ── CLI overrides ────────────────────────────────────────
    epochs = args.epochs or EPOCHS
    batch_size = args.batch_size or BATCH_SIZE
    use_amp = not args.no_amp and device.type == "cuda"
    patience = args.patience
    if args.mode == "smoke":
        epochs = 1

    # Discover and load audio
    audio_files = discover_audio_files(ds_path)
    logger.info("Found %d audio files under %s", len(audio_files), ds_path)

    mel_specs: list[torch.Tensor] = []
    max_files = 50 if args.mode == "smoke" else 500
    if audio_files:
        logger.info("Converting audio files to mel spectrograms …")
        for fp in audio_files[:max_files]:  # cap for memory
            wav = load_audio(fp)
            if wav is not None and wav.shape[-1] > HOP_LENGTH * 4:
                mel = waveform_to_mel(wav, device=torch.device("cpu"))
                mel_specs.append(mel)

    if len(mel_specs) < 10:
        logger.warning(
            "Only %d usable audio files found — generating synthetic spectrograms "
            "for demonstration.", len(mel_specs),
        )
        mel_specs = generate_synthetic_specs(200, device)

    logger.info("Total spectrogram samples: %d", len(mel_specs))

    # Train
    model, train_losses, val_losses, test_loss, test_snr, split_counts = train(
        device, mel_specs, output_dir,
        epochs=epochs, batch_size=batch_size, use_amp_override=use_amp,
        patience=patience,
    )

    # Save outputs
    save_plots(train_losses, val_losses, output_dir)
    dataset = DenoisingDataset(mel_specs)
    save_spectrogram_comparison(model, dataset, device, output_dir)

    # ── Split manifest ───────────────────────────────────────
    ds_fp = dataset_fingerprint(ds_path)
    write_split_manifest(
        output_dir,
        dataset_fp=ds_fp,
        split_method="random_split 70/15/15",
        seed=args.seed,
        counts=split_counts,
    )

    # Metrics
    meta = run_metadata(args)
    metrics = {
        "dataset": f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}",
        "audio_files_found": len(discover_audio_files(ds_path)) if audio_files else 0,
        "spectrograms_used": len(mel_specs),
        "epochs": epochs,
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "test_loss": float(test_loss),
        "test_snr_improvement": float(test_snr),
        "split": "test",
        "run_metadata": meta,
    }
    save_metrics(output_dir, metrics, task_type="audio", mode=args.mode)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
