#!/usr/bin/env python3
"""
Voice Cloning — SpeechT5 Fine-tuning (PyTorch)
================================================
Demonstrate voice cloning / multi-speaker TTS using Microsoft's
SpeechT5 model from Hugging Face.  The pipeline:

1. Load pretrained ``microsoft/speecht5_tts`` + vocoder.
2. Download the English Multi-Speaker Corpus for Voice Cloning.
3. Extract speaker embeddings from audio samples (x-vector via
   speechbrain, or a simple mel-based fallback).
4. Generate speech samples conditioned on different speaker
   embeddings and save the resulting WAV files.
5. Optionally fine-tune the model on a small subset.

Dataset
-------
* Kaggle: https://www.kaggle.com/datasets/mfekadu/english-multispeaker-corpus-for-voice-cloning

Run
---
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
import numpy as np
import torch
import torch.nn as nn
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
)

logger = logging.getLogger(__name__)

# Check torchaudio availability once (import can hard-crash the process)
_TORCHAUDIO_OK = safe_import_available("torchaudio")

# ── Configuration ────────────────────────────────────────────
KAGGLE_SLUG = "mfekadu/english-multispeaker-corpus-for-voice-cloning"
TTS_MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
SAMPLE_RATE = 16_000
EMBEDDING_DIM = 512
MAX_SPEAKERS = 10            # demo with up to N speakers
DEMO_TEXTS = [
    "Hello, this is a voice cloning demonstration.",
    "Deep learning has transformed speech synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "Thank you for listening to this generated audio sample.",
]
FINETUNE_EPOCHS = 2
FINETUNE_LR = 1e-5
FINETUNE_BATCH = 2


# ═════════════════════════════════════════════════════════════
#  Audio I/O
# ═════════════════════════════════════════════════════════════

def discover_audio_files(root: Path, limit: int = 0):
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
            if 0 < limit <= len(files):
                break
    return sorted(files)


def load_audio(filepath: Path, sr: int = SAMPLE_RATE):
    """Load audio, return 1-D numpy array at target sample rate."""
    if _TORCHAUDIO_OK:
        try:
            import torchaudio
            wav, orig_sr = torchaudio.load(str(filepath))
            if orig_sr != sr:
                wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            return wav.squeeze(0).numpy()
        except Exception:
            pass

    try:
        import librosa
        y, _ = librosa.load(str(filepath), sr=sr, mono=True)
        return y
    except Exception as exc:
        logger.warning("Cannot load %s: %s", filepath, exc)
        return None


def save_audio(waveform, path: Path, sr: int = SAMPLE_RATE):
    """Save a 1-D numpy or tensor waveform to WAV."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)

    if _TORCHAUDIO_OK:
        try:
            import torchaudio
            t = torch.from_numpy(waveform).unsqueeze(0)
            torchaudio.save(str(path), t, sr)
            return
        except Exception:
            pass

    try:
        import soundfile as sf
        sf.write(str(path), waveform, sr)
        return
    except Exception:
        pass

    # Minimal WAV writer
    import struct, wave
    waveform_int = np.clip(waveform, -1, 1)
    waveform_int = (waveform_int * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(waveform_int.tobytes())


# ═════════════════════════════════════════════════════════════
#  Speaker embedding extraction
# ═════════════════════════════════════════════════════════════

def extract_xvector_embedding(audio_np: np.ndarray, device):
    """Extract x-vector speaker embedding using speechbrain."""
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": str(device)},
    )
    wav_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    embedding = classifier.encode_batch(wav_tensor)
    return embedding.squeeze().cpu()


def extract_simple_embedding(audio_np: np.ndarray, dim: int = EMBEDDING_DIM):
    """Compute a simple deterministic embedding from audio statistics."""
    if _TORCHAUDIO_OK:
        try:
            import torchaudio
            wav_t = torch.from_numpy(audio_np).unsqueeze(0)
            mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80,
            )
            mel = mel_fn(wav_t).squeeze(0).numpy()
        except Exception:
            mel = None
    else:
        mel = None

    if mel is None:
        try:
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio_np, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80,
            )
        except Exception:
            # Random fallback
            return torch.randn(dim)

    # Summarize mel statistics
    stats = np.concatenate([
        mel.mean(axis=1), mel.std(axis=1),
        np.percentile(mel, 25, axis=1), np.percentile(mel, 75, axis=1),
    ])
    # Pad/crop to dim
    if len(stats) < dim:
        stats = np.pad(stats, (0, dim - len(stats)))
    else:
        stats = stats[:dim]
    # Normalize
    norm = np.linalg.norm(stats) + 1e-8
    return torch.from_numpy((stats / norm).astype(np.float32))


def get_speaker_embedding(audio_np: np.ndarray, device):
    """Try speechbrain x-vector, fall back to simple embedding."""
    try:
        return extract_xvector_embedding(audio_np, device)
    except Exception:
        return extract_simple_embedding(audio_np)


# ═════════════════════════════════════════════════════════════
#  Group audio by speaker
# ═════════════════════════════════════════════════════════════

def group_by_speaker(audio_files: list[Path]):
    """Heuristic: group by parent directory (assumed to be speaker ID)."""
    speakers: dict[str, list[Path]] = {}
    for fp in audio_files:
        speaker = fp.parent.name
        speakers.setdefault(speaker, []).append(fp)
    return speakers


# ═════════════════════════════════════════════════════════════
#  TTS generation
# ═════════════════════════════════════════════════════════════

def generate_speech(model, processor, vocoder, text, speaker_embedding, device):
    """Generate speech waveform for *text* conditioned on *speaker_embedding*."""
    inputs = processor(text=text, return_tensors="pt").to(device)

    # SpeechT5 expects speaker embeddings of shape (1, embed_dim)
    spk = speaker_embedding.unsqueeze(0).to(device)

    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)

    return speech.cpu()


# ═════════════════════════════════════════════════════════════
#  Optional fine-tuning
# ═════════════════════════════════════════════════════════════

def try_finetune(model, processor, vocoder, audio_files, device, output_dir,
                 *, epochs=None, batch_size=None, use_amp_override=None):
    """Attempt a tiny fine-tune loop on available audio + transcripts."""
    epochs = epochs or FINETUNE_EPOCHS
    # This is a simplified demonstration — real fine-tuning requires
    # aligned transcripts and careful data processing.
    logger.info("Fine-tuning demo: running %d epochs on %d samples …",
                epochs, min(len(audio_files), 20))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR)

    use_amp = use_amp_override if use_amp_override is not None else (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    sample_files = audio_files[:20]
    losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        count = 0
        for fp in sample_files:
            audio_np = load_audio(fp)
            if audio_np is None or len(audio_np) < SAMPLE_RATE:
                continue

            # Use filename (without extension) as pseudo-transcript
            pseudo_text = fp.stem.replace("_", " ").replace("-", " ")[:200]
            if len(pseudo_text.strip()) < 2:
                pseudo_text = "sample audio"

            try:
                inputs = processor(text=pseudo_text, return_tensors="pt").to(device)
                # Prepare audio target
                target = processor(
                    audio_target=torch.from_numpy(audio_np).unsqueeze(0),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                )
                labels = target.get("labels", target.get("input_values"))
                if labels is None:
                    continue
                labels = labels.to(device)

                spk = get_speaker_embedding(audio_np, device).unsqueeze(0).to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        input_ids=inputs["input_ids"],
                        speaker_embeddings=spk,
                        labels=labels,
                    )
                    loss = out.loss

                if loss is not None:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    count += 1
            except Exception as exc:
                logger.debug("Finetune step failed: %s", exc)
                continue

        avg = epoch_loss / max(count, 1)
        losses.append(avg)
        logger.info("Finetune epoch %d/%d — loss=%.4f (%d samples)",
                     epoch, epochs, avg, count)

    model.eval()
    return losses


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def plot_waveforms(waveforms: dict[str, np.ndarray], output_dir: Path):
    """Plot waveform comparisons for different speakers."""
    n = len(waveforms)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), squeeze=False)
    for i, (label, wav) in enumerate(waveforms.items()):
        t = np.arange(len(wav)) / SAMPLE_RATE
        axes[i, 0].plot(t, wav, linewidth=0.5)
        axes[i, 0].set_title(label, fontsize=9)
        axes[i, 0].set_ylabel("Amplitude")
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("Generated Speech Waveforms", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "waveform_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved waveform comparison → %s", output_dir / "waveform_comparison.png")


def plot_spectrograms(waveforms: dict[str, np.ndarray], output_dir: Path):
    """Plot mel-spectrogram for each generated sample."""
    n = len(waveforms)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)

    for i, (label, wav) in enumerate(waveforms.items()):
        mel = None
        if _TORCHAUDIO_OK:
            try:
                import torchaudio
                mel_fn = torchaudio.transforms.MelSpectrogram(
                    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80,
                )
                mel = mel_fn(torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
            except Exception:
                pass
        if mel is None:
            try:
                import librosa
                mel = librosa.feature.melspectrogram(
                    y=wav, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80,
                )
            except Exception:
                continue
        axes[i, 0].imshow(np.log1p(mel), aspect="auto", origin="lower", cmap="magma")
        axes[i, 0].set_title(label, fontsize=9)

    fig.suptitle("Generated Speech — Mel Spectrograms", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "spectrogram_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved spectrogram comparison → %s", output_dir / "spectrogram_comparison.png")


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging()
    args = parse_common_args("Voice Cloning — SpeechT5 Fine-tuning")
    set_seed(args.seed)
    configure_cuda_allocator()

    paths = project_paths(__file__)
    data_dir = paths["data"]
    output_dir = ensure_dir(paths["outputs"])
    device = resolve_device_from_args(args)

    # torchaudio can crash the process at import time (native DLL issues)
    if not safe_import_available("torchaudio"):
        logger.warning("torchaudio not available — internal functions will use librosa fallback")

    # ── Load pretrained TTS ──────────────────────────────────
    try:
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    except ImportError:
        missing_dependency_metrics(
            output_dir,
            missing=["transformers"],
            install_cmd="pip install -U transformers sentencepiece",
        )

    logger.info("Loading pretrained SpeechT5 TTS model …")
    try:
        processor = SpeechT5Processor.from_pretrained(TTS_MODEL_ID)
        model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL_ID).to(device)
        vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID).to(device)
        model.eval()
        vocoder.eval()
        logger.info("Model loaded: %s", TTS_MODEL_ID)
    except Exception as exc:
        logger.error("Failed to load SpeechT5 model: %s", exc)
        save_metrics(output_dir, {"status": "error", "error": f"SpeechT5 load failed: {str(exc)[:200]}"}, task_type="audio", mode=args.mode)
        return

    # ── Download dataset ─────────────────────────────────────
    try:
        ds_path = download_kaggle_dataset(
            KAGGLE_SLUG, data_dir,
            dataset_name="English Multi-Speaker Corpus for Voice Cloning",
        )
    except (SystemExit, Exception) as exc:
        logger.warning("Dataset download failed: %s — will use random embeddings", exc)
        ds_path = data_dir

    if args.download_only:
        logger.info("Download complete — exiting (--download-only).")
        sys.exit(0)

    # ── CLI overrides ────────────────────────────────────────
    epochs = args.epochs or FINETUNE_EPOCHS
    batch_size = args.batch_size or FINETUNE_BATCH
    use_amp = not args.no_amp and device.type == "cuda"
    if args.mode == "smoke":
        epochs = 1

    audio_limit = 100 if args.mode == "smoke" else 500
    audio_files = discover_audio_files(ds_path, limit=audio_limit)
    logger.info("Found %d audio files under %s", len(audio_files), ds_path)

    # ── Extract speaker embeddings ───────────────────────────
    speaker_groups = group_by_speaker(audio_files) if audio_files else {}
    speaker_names = list(speaker_groups.keys())[:MAX_SPEAKERS]
    logger.info("Identified %d speaker groups (using up to %d)",
                len(speaker_groups), MAX_SPEAKERS)

    speaker_embeddings: dict[str, torch.Tensor] = {}

    if speaker_names:
        for spk_name in speaker_names:
            files = speaker_groups[spk_name]
            audio_np = load_audio(files[0])
            if audio_np is not None and len(audio_np) >= SAMPLE_RATE:
                emb = get_speaker_embedding(audio_np, device)
                # Resize to 512 if needed
                if emb.shape[0] != EMBEDDING_DIM:
                    emb = torch.nn.functional.interpolate(
                        emb.unsqueeze(0).unsqueeze(0),
                        size=EMBEDDING_DIM,
                        mode="linear",
                    ).squeeze()
                speaker_embeddings[spk_name] = emb
                logger.info("  Speaker '%s' — embedding shape %s", spk_name, tuple(emb.shape))

    # Fallback: use random embeddings if none extracted
    if not speaker_embeddings:
        logger.warning("No speaker embeddings extracted — using random embeddings for demo.")
        for i in range(min(3, MAX_SPEAKERS)):
            name = f"random_speaker_{i}"
            speaker_embeddings[name] = torch.randn(EMBEDDING_DIM)

    # ── Optional fine-tuning ─────────────────────────────────
    ft_losses: list[float] = []
    if audio_files and len(audio_files) >= 5:
        try:
            ft_losses = try_finetune(
                model, processor, vocoder, audio_files, device, output_dir,
                epochs=epochs, batch_size=batch_size, use_amp_override=use_amp,
            )
        except Exception as exc:
            logger.warning("Fine-tuning skipped: %s", exc)

    # ── Generate speech samples ──────────────────────────────
    logger.info("Generating speech samples for %d speakers × %d texts …",
                len(speaker_embeddings), len(DEMO_TEXTS))

    generated_wavs: dict[str, np.ndarray] = {}
    audio_out_dir = ensure_dir(output_dir / "audio")

    for spk_name, spk_emb in speaker_embeddings.items():
        for j, text in enumerate(DEMO_TEXTS):
            try:
                wav = generate_speech(model, processor, vocoder, text, spk_emb, device)
                wav_np = wav.numpy()
                label = f"{spk_name}_text{j}"
                generated_wavs[label] = wav_np

                out_path = audio_out_dir / f"{label}.wav"
                save_audio(wav_np, out_path)
                logger.info("  Saved %s (%.1f s)", out_path.name, len(wav_np) / SAMPLE_RATE)
            except Exception as exc:
                logger.warning("  Failed for speaker=%s text=%d: %s", spk_name, j, exc)

    # ── Visualisations ───────────────────────────────────────
    if generated_wavs:
        # Show first sample per speaker
        vis_wavs = {}
        seen_speakers = set()
        for label, wav in generated_wavs.items():
            spk = label.rsplit("_text", 1)[0]
            if spk not in seen_speakers:
                vis_wavs[label] = wav
                seen_speakers.add(spk)

        plot_waveforms(vis_wavs, output_dir)
        plot_spectrograms(vis_wavs, output_dir)

    # ── Save metrics ─────────────────────────────────────────
    ds_fp = dataset_fingerprint(ds_path)
    write_split_manifest(
        output_dir,
        dataset_fp=ds_fp,
        split_method="pretrained_model (no train/test split — generation only)",
        seed=args.seed,
        counts={"audio_files": len(audio_files), "speakers_used": len(speaker_embeddings)},
        extras={"tts_model": TTS_MODEL_ID, "vocoder": VOCODER_ID},
    )

    meta = run_metadata(args)
    val_loss = ft_losses[-1] if ft_losses else None
    metrics = {
        "dataset": f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}",
        "tts_model": TTS_MODEL_ID,
        "vocoder": VOCODER_ID,
        "audio_files_found": len(audio_files),
        "speakers_detected": len(speaker_groups),
        "speakers_used": len(speaker_embeddings),
        "texts_per_speaker": len(DEMO_TEXTS),
        "samples_generated": len(generated_wavs),
        "finetune_epochs": epochs if ft_losses else 0,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "run_metadata": meta,
    }
    save_metrics(output_dir, metrics, task_type="audio", mode=args.mode)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
