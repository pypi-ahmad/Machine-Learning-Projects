"""
Modern Audio/Speech Pipeline (April 2026)

Task: classification

Model selection by task:
  - ASR / speech-to-text   -- Whisper large-v3-turbo (OpenAI)
  - Audio classification   -- Wav2Vec2-base + HuBERT-base (Meta)
  - Denoising / separation -- SpeechBrain SepFormer (speechbrain/sepformer-whamr)
  - Voice cloning / TTS    -- Coqui XTTS-v2 (multilingual, speaker-adaptive)

Compute requirements:
  - Whisper large-v3-turbo : ~2 GB VRAM, ~3s/file on GPU, ~15s/file on CPU
  - Wav2Vec2 / HuBERT      : ~1 GB VRAM, <1s/file on GPU, ~3s/file on CPU
  - SepFormer              : ~2 GB VRAM, ~5s/file on GPU, ~20s/file on CPU
  - XTTS-v2                : ~4 GB VRAM, ~10s per utterance on GPU, CPU very slow

Dependencies: transformers, torch, torchaudio, soundfile, speechbrain, TTS (Coqui)
Data: Auto-downloaded at runtime from HuggingFace
"""
import os, json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

TASK = "classification"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_audio_samples():
    """Download audio samples from HuggingFace datasets."""
    from datasets import load_dataset
    import soundfile as sf

    save_dir = os.path.join(SAVE_DIR, "audio_data")
    os.makedirs(save_dir, exist_ok=True)

    if TASK == "classification":
        try:
            from datasets import load_dataset as _hf_load
            df = _hf_load("google/speech_commands", "v0.02", split="train").to_pandas()
            if df is not None:
                # Extract audio files from HF dataset if it has an audio column
                if hasattr(df, "columns"):
                    print(f"Loaded dataset: {len(df)} samples")
                return save_dir, df
        except Exception:
            pass
        ds = load_dataset("google/speech_commands", "v0.02", split="train[:100]")
    elif TASK == "cloning":
        ds = load_dataset("google/speech_commands", split="train[:20]",
                          trust_remote_code=True)
    else:
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                          "clean", split="validation[:20]")

    paths = []
    for i, sample in enumerate(ds):
        audio = sample.get("audio", sample)
        if isinstance(audio, dict):
            arr, sr = np.array(audio["array"]), audio["sampling_rate"]
            out_path = os.path.join(save_dir, f"sample_{i:03d}.wav")
            sf.write(out_path, arr, sr)
            paths.append(out_path)

    print(f"Downloaded {len(paths)} audio samples to {save_dir}")
    return save_dir, paths


# ===========================================================
# ASR / SPEECH-TO-TEXT -- Whisper large-v3-turbo
# ===========================================================

def run_whisper(audio_dir):
    """Automatic speech recognition with Whisper large-v3-turbo."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    print(f"  Loading {model_id} on {device} ...")
    t0 = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    asr = pipeline("automatic-speech-recognition", model=model,
                   tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor,
                   torch_dtype=torch_dtype, device=device)
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

    from pathlib import Path
    audio_files = sorted(Path(audio_dir).glob("*.wav")) + sorted(Path(audio_dir).glob("*.flac"))
    results = []
    for f in audio_files[:10]:
        t1 = time.perf_counter()
        result = asr(str(f), return_timestamps=True)
        dt = time.perf_counter() - t1
        results.append({"file": f.name, "text": result["text"], "time_s": round(dt, 2)})
        print(f"  {f.name} ({dt:.1f}s): {result['text'][:80]}...")
    return results


# ===========================================================
# AUDIO CLASSIFICATION -- Wav2Vec2 + HuBERT
# ===========================================================

def run_wav2vec2_clf(audio_dir):
    """Audio classification with Wav2Vec2 and HuBERT."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import soundfile as sf
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:20]

    if not audio_files:
        print("  No .wav files found - extracting from dataset ...")
        # Try to extract audio from HF dataset objects
        ds_dir = Path(audio_dir)
        from datasets import load_dataset
        ds = load_dataset("google/speech_commands", "v0.02", split="train[:50]")
        for i, sample in enumerate(ds):
            audio = sample.get("audio", sample)
            if isinstance(audio, dict):
                arr = np.array(audio["array"])
                sr = audio["sampling_rate"]
                out_path = ds_dir / f"sample_{i:03d}.wav"
                sf.write(str(out_path), arr, sr)
        audio_files = sorted(ds_dir.glob("*.wav"))[:20]

    summary = []
    for model_name, label in [("facebook/wav2vec2-base", "Wav2Vec2"),
                               ("facebook/hubert-base-ls960", "HuBERT")]:
        try:
            t0 = time.perf_counter()
            print(f"  Loading {label} ...")
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(
                model_name, num_labels=10, ignore_mismatched_sizes=True).to(device)
            preds = []
            for f in audio_files:
                arr, sr = sf.read(str(f))
                if len(arr.shape) > 1:
                    arr = arr[:, 0]
                inputs = extractor(arr, sampling_rate=sr, return_tensors="pt",
                                   padding=True).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()
                preds.append(pred)
                print(f"    {f.name}: class {pred}")
            elapsed = time.perf_counter() - t0
            summary.append({"model": label, "n_files": len(preds),
                            "time_s": round(elapsed, 1)})
            print(f"  {label}: {len(preds)} files classified ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {label} failed: {e}")

    return summary


# ===========================================================
# DENOISING / SEPARATION -- SpeechBrain SepFormer
# ===========================================================

def run_sepformer(audio_dir):
    """Speech enhancement / denoising.

    Primary: SepFormer (speechbrain/sepformer-whamr) - neural, SOTA
    Baseline: spectral subtraction - classical signal processing
    """
    try:
        from pathlib import Path
        import torchaudio

        audio_files = sorted(Path(audio_dir).glob("*.wav"))[:10]

        # --- BASELINE: Spectral Subtraction ---
        print("  --- Baseline: Spectral Subtraction ---")
        baseline_results = []
        t_base = time.perf_counter()
        for f in audio_files:
            try:
                import soundfile as sf
                data, sr = sf.read(str(f))
                if len(data.shape) > 1:
                    data = data[:, 0]
                fl = min(400, len(data))
                hop = fl // 2
                n_frames = max(1, len(data) // hop - 1)
                frames = np.array([data[i*hop:i*hop+fl] for i in range(n_frames) if i*hop+fl <= len(data)])
                if len(frames) == 0:
                    continue
                ham = np.hamming(fl)
                windowed = frames * ham
                dft = np.fft.fft(windowed)
                mag = np.abs(dft)
                phase = np.angle(dft)
                noise_est = np.mean(mag, axis=0)
                clean_mag = np.maximum(mag - 2 * noise_est, 0)
                estimate = clean_mag * np.exp(1j * phase)
                ift = [np.fft.ifft(e).real for e in estimate]
                clean_data = list(ift[0][:hop])
                for i in range(len(ift) - 1):
                    clean_data.extend(ift[i][hop:] + ift[i+1][:hop])
                clean_data.extend(ift[-1][hop:])
                baseline_results.append({"file": f.name, "method": "spectral_subtraction"})
                print(f"    {f.name}: processed")
            except Exception as e:
                print(f"    {f.name}: failed ({e})")
        dt_base = time.perf_counter() - t_base
        print(f"  Spectral subtraction: {len(baseline_results)} files ({dt_base:.1f}s)")

        # --- PRIMARY: SepFormer ---
        print("  --- Primary: SpeechBrain SepFormer ---")
        print("  Loading SepFormer (speechbrain/sepformer-whamr) ...")
        t0 = time.perf_counter()
        try:
            from speechbrain.inference.separation import SepformerSeparation
            sep_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                savedir=os.path.join(SAVE_DIR, "sepformer_model"))
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation
            sep_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                savedir=os.path.join(SAVE_DIR, "sepformer_model"))
        print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

        out_dir = os.path.join(SAVE_DIR, "enhanced")
        os.makedirs(out_dir, exist_ok=True)
        results = []

        for f in audio_files:
            t1 = time.perf_counter()
            est_sources = sep_model.separate_file(path=str(f))
            out_path = os.path.join(out_dir, f"{f.stem}_enhanced.wav")
            torchaudio.save(out_path, est_sources[:, :, 0].cpu(), 8000)
            dt = time.perf_counter() - t1
            results.append({"file": f.name, "output": f"{f.stem}_enhanced.wav",
                            "method": "sepformer", "time_s": round(dt, 1)})
            print(f"  {f.name} -> {f.stem}_enhanced.wav ({dt:.1f}s)")

        print(f"  Enhanced audio saved to {out_dir}")
        print(f"  SepFormer: {len(results)} files | Baseline: {len(baseline_results)} files")
        return {"sepformer": results, "baseline_spectral": baseline_results}
    except Exception as e:
        print(f"  SepFormer failed: {e}")
        return {}


# ===========================================================
# VOICE CLONING / TTS -- Coqui XTTS-v2
# ===========================================================

def run_voice_cloning(audio_dir):
    """Voice cloning and text-to-speech.

    Primary: Coqui XTTS-v2 (multilingual, speaker-adaptive, ~4 GB VRAM)
    Baseline: pyttsx3 (offline, CPU-only, no cloning)
    """
    results = {"xtts": [], "baseline_pyttsx3": []}

    # --- BASELINE: pyttsx3 (offline CPU TTS) ---
    try:
        import pyttsx3
        print("  --- Baseline: pyttsx3 (offline TTS) ---")
        engine = pyttsx3.init()
        baseline_dir = os.path.join(SAVE_DIR, "tts_baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        baseline_texts = [
            "This is a baseline text to speech sample using pyttsx3.",
            "Offline synthesis is fast but lacks naturalness.",
        ]
        for i, text in enumerate(baseline_texts):
            t1 = time.perf_counter()
            out_path = os.path.join(baseline_dir, f"baseline_{i:02d}.wav")
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            dt = time.perf_counter() - t1
            results["baseline_pyttsx3"].append({
                "text": text[:60], "output": os.path.basename(out_path),
                "time_s": round(dt, 1)})
            print(f"    baseline_{i:02d}.wav ({dt:.1f}s)")
    except Exception as e:
        print(f"  pyttsx3 baseline skipped: {e}")

    # --- PRIMARY: XTTS-v2 ---
    try:
        from TTS.api import TTS
        from pathlib import Path

        print("  --- Primary: Coqui XTTS-v2 ---")
        print("  Loading XTTS-v2 ...")
        t0 = time.perf_counter()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

        out_dir = os.path.join(SAVE_DIR, "tts_output")
        os.makedirs(out_dir, exist_ok=True)

        # Find a reference speaker sample for voice cloning
        ref_files = sorted(Path(audio_dir).glob("*.wav"))
        ref_speaker = str(ref_files[0]) if ref_files else None

        texts = [
            ("Hello, this is a text to speech demonstration using XTTS version 2.", "en"),
            ("Modern voice cloning can produce remarkably natural speech.", "en"),
            ("Deep learning models now generate human-quality audio in real time.", "en"),
            ("Bonjour, ceci est une demonstration de synthese vocale multilingue.", "fr"),
        ]

        for i, (text, lang) in enumerate(texts):
            t1 = time.perf_counter()
            out_path = os.path.join(out_dir, f"tts_sample_{i:02d}.wav")
            if ref_speaker:
                tts.tts_to_file(text=text, file_path=out_path,
                               speaker_wav=ref_speaker, language=lang)
                mode = "cloned"
            else:
                tts.tts_to_file(text=text, file_path=out_path)
                mode = "default"
            dt = time.perf_counter() - t1
            results["xtts"].append({"text": text[:60], "output": os.path.basename(out_path),
                            "mode": mode, "language": lang, "time_s": round(dt, 1)})
            print(f"  [{mode}/{lang}] tts_sample_{i:02d}.wav ({dt:.1f}s)")

        print(f"  TTS output saved to {out_dir}")
        print(f"  XTTS-v2: {len(results['xtts'])} samples | Baseline: {len(results['baseline_pyttsx3'])} samples")
    except Exception as e:
        print(f"  XTTS-v2 failed: {e}")

    return results


def run_eda(save_dir):
    """Audio file summary."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    audio_files = []
    for root, dirs, files in os.walk(save_dir):
        for f in files:
            if f.lower().endswith(audio_exts):
                audio_files.append(os.path.join(root, f))
    print(f"  Audio files found: {len(audio_files)}")
    if audio_files:
        total_size = sum(os.path.getsize(f) for f in audio_files)
        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print("EDA complete.")


def validate_results(task, results, save_dir):
    """Validate audio outputs for the active task."""
    validation = {"task": task, "checks": {}}

    if task == "transcription":
        records = results.get("whisper", [])
        non_empty = sum(1 for item in records if str(item.get("text", "")).strip())
        validation["checks"]["whisper"] = {
            "records": len(records),
            "non_empty_transcripts": non_empty,
            "passed": len(records) > 0 and non_empty == len(records),
        }
    elif task == "classification":
        records = results.get("classification", [])
        passed = all(item.get("n_files", 0) > 0 for item in records) if records else False
        validation["checks"]["classification"] = {
            "models": len(records),
            "files_scored": sum(int(item.get("n_files", 0)) for item in records),
            "passed": passed,
        }
    elif task in ("denoising", "separation"):
        bundle = results.get("sepformer", {})
        sep_records = bundle.get("sepformer", []) if isinstance(bundle, dict) else []
        existing = sum(
            1 for item in sep_records
            if os.path.exists(os.path.join(save_dir, "enhanced", item.get("output", "")))
        )
        validation["checks"]["sepformer"] = {
            "outputs": len(sep_records),
            "existing_outputs": existing,
            "baseline_files": len(bundle.get("baseline_spectral", [])) if isinstance(bundle, dict) else 0,
            "passed": len(sep_records) > 0 and existing == len(sep_records),
        }
    elif task == "cloning":
        bundle = results.get("xtts", {})
        xtts_records = bundle.get("xtts", []) if isinstance(bundle, dict) else []
        existing = sum(
            1 for item in xtts_records
            if os.path.exists(os.path.join(save_dir, "tts_output", item.get("output", "")))
        )
        validation["checks"]["xtts"] = {
            "outputs": len(xtts_records),
            "existing_outputs": existing,
            "baseline_outputs": len(bundle.get("baseline_pyttsx3", [])) if isinstance(bundle, dict) else 0,
            "passed": len(xtts_records) > 0 and existing == len(xtts_records),
        }

    validation["passed"] = any(item.get("passed") for item in validation["checks"].values())
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"  Validation saved to {out_path}")
    return validation


def main():
    print("=" * 60)
    print(f"AUDIO/SPEECH | Task: {TASK}")
    print("Models: Whisper | Wav2Vec2/HuBERT | SepFormer | XTTS-v2")
    print("=" * 60)
    audio_dir, data = download_audio_samples()
    run_eda(SAVE_DIR)
    results = {}

    if TASK == "transcription":
        print()
        print("--- ASR: Whisper large-v3-turbo ---")
        asr_results = run_whisper(audio_dir)
        results["whisper"] = asr_results
        if asr_results:
            out = os.path.join(SAVE_DIR, "transcriptions.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(asr_results, f, indent=2)
            print(f"  Saved transcriptions to {out}")

    elif TASK == "classification":
        print()
        print("--- Classification: Wav2Vec2 + HuBERT ---")
        clf_results = run_wav2vec2_clf(audio_dir)
        results["classification"] = clf_results

    elif TASK == "denoising" or TASK == "separation":
        print()
        print("--- Denoising: SpeechBrain SepFormer ---")
        sep_results = run_sepformer(audio_dir)
        results["sepformer"] = sep_results

    elif TASK == "cloning":
        print()
        print("--- Voice Cloning: XTTS-v2 ---")
        tts_results = run_voice_cloning(audio_dir)
        results["xtts"] = tts_results

    else:
        print()
        print("--- ASR (default): Whisper large-v3-turbo ---")
        asr_results = run_whisper(audio_dir)
        results["whisper"] = asr_results

    # Save metrics
    validation = validate_results(TASK, results, SAVE_DIR)
    results["validation"] = validation
    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
