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
from scipy.io import wavfile
from scipy.signal import resample as _scipy_resample
from tqdm import tqdm
from shared.utils import (
    seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir,
    save_metrics, parse_common_args, load_profile, resolve_config,
    write_split_manifest, EarlyStopping)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "mmoreaux/audio-cats-and-dogs"
EPOCHS, LR, BATCH = 15, 1e-3, 32
SR = 16000
N_MELS = 64
N_FFT  = 1024
HOP    = 512


def _load_wav(path, target_sr=SR):
    """Load wav file, resample to target_sr, return mono float32 numpy array."""
    sr, data = wavfile.read(str(path))
    if data.dtype != np.float32:
        data = data.astype(np.float32) / max(np.abs(data).max(), 1e-9)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        n_samples = int(len(data) * target_sr / sr)
        data = _scipy_resample(data, n_samples).astype(np.float32)
    return data


def _mel_spectrogram(wav, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    """Compute mel spectrogram using numpy (no torchaudio needed)."""
    # STFT
    pad = n_fft // 2
    wav_padded = np.pad(wav, (pad, pad), mode='reflect')
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(wav_padded) - n_fft) // hop
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = wav_padded[i * hop: i * hop + n_fft] * window
        stft[:, i] = np.abs(np.fft.rfft(frame).astype(np.complex64))

    # Mel filter bank
    power = stft ** 2
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 700 * (10 ** (mels / 2595) - 1)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(f_left, f_center):
            fbank[m, k] = (k - f_left) / max(f_center - f_left, 1)
        for k in range(f_center, f_right):
            fbank[m, k] = (f_right - k) / max(f_right - f_center, 1)
    mel_spec = fbank @ power  # (n_mels, n_frames)
    mel_spec = np.log(mel_spec + 1e-9)
    return mel_spec


class AudioDS(Dataset):
    def __init__(self, files, labels):
        self.files, self.labels = files, labels

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        wav = _load_wav(str(self.files[i]), SR)
        target_len = SR * 3
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]
        spec = _mel_spectrogram(wav)  # (n_mels, T)
        spec_t = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, T)
        return spec_t, self.labels[i]


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



TASK_TYPE = 'custom_audio'

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
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    ds = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    is_full     = (args.mode == 'full')
    epochs      = args.epochs or cfg.get('epochs', EPOCHS)
    batch_size  = args.batch_size or cfg.get('batch_size', BATCH)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get('num_workers', 2)
    use_amp     = not args.no_amp and cfg.get('amp', True)
    max_batches = 2 if args.smoke_test else None
    es_on       = cfg.get('early_stopping', False) if is_full else False
    es_patience = cfg.get('patience', 3)
    if args.smoke_test:
        epochs = 1

    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size, num_workers=num_workers)

    model = AudioCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=es_patience, mode='max') if es_on else None

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
        if es and es.step(mean_dice if 'mean_dice' in dir() else vacc if 'vacc' in dir() else acc):
            break

    save_metrics({"best_val_accuracy": best}, OUTPUT_DIR)
    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=KAGGLE,
            seed=cfg.get('seed', 42))


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
