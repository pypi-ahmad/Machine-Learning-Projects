"""
Modern Audio/Speech Pipeline (April 2026)
Models: Whisper large-v3-turbo (ASR), Wav2Vec2 (clf), SepFormer (separation), XTTS-v2 (TTS)
Data: Auto-downloaded at runtime from HuggingFace
"""
import os, json, warnings
import numpy as np

warnings.filterwarnings("ignore")

TASK = "classification"


def download_audio_samples():
    """Download audio samples from HuggingFace datasets."""
    from datasets import load_dataset
    import soundfile as sf

    save_dir = os.path.join(os.path.dirname(__file__), "audio_data")
    os.makedirs(save_dir, exist_ok=True)

    if TASK == "classification":
        try:
            from datasets import load_dataset as _hf_load
            df = _hf_load("marsyas/gtzan", split="train").to_pandas()
            if df is not None:
                print(f"Loaded dataset: {len(df)} samples")
                return save_dir, df
        except Exception:
            pass
        ds = load_dataset("google/speech_commands", "v0.02", split="train[:100]")
    else:
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:20]")

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


def run_whisper(audio_dir):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, torch_dtype=torch_dtype, device=device)

    from pathlib import Path
    audio_files = list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.flac"))
    results = []
    for f in audio_files[:10]:
        result = asr(str(f), return_timestamps=True)
        results.append({"file": f.name, "text": result["text"]})
        print(f"  ✓ {f.name}: {result['text'][:100]}...")
    return results


def run_voice_cloning():
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        out_path = os.path.join(os.path.dirname(__file__), "tts_output.wav")
        tts.tts_to_file(text="Hello, this is a text to speech sample using XTTS version 2.", file_path=out_path)
        print(f"✓ TTS output → {out_path}")
    except Exception as e:
        print(f"✗ XTTS: {e}")


def run_wav2vec2_clf(audio_dir):
    """Audio classification with Wav2Vec2."""
    import torch
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    import soundfile as sf
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/wav2vec2-base"
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_id, num_labels=10).to(device)
        audio_files = list(Path(audio_dir).glob("*.wav"))[:10]
        for f in audio_files:
            arr, sr = sf.read(str(f))
            if len(arr.shape) > 1: arr = arr[:, 0]
            inputs = extractor(arr, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            print(f"  ✓ {f.name}: class {pred}")
        print("✓ Wav2Vec2 classification complete")
    except Exception as e:
        print(f"✗ Wav2Vec2: {e}")


def run_sepformer(audio_dir):
    """Source separation with SepFormer (SpeechBrain)."""
    try:
        from speechbrain.inference.separation import SepformerSeparation
        from pathlib import Path
        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr", savedir=os.path.join(os.path.dirname(__file__), "sepformer_model"))
        audio_files = list(Path(audio_dir).glob("*.wav"))[:5]
        save_dir = os.path.join(os.path.dirname(__file__), "separated")
        os.makedirs(save_dir, exist_ok=True)
        for f in audio_files:
            est_sources = model.separate_file(path=str(f))
            for i in range(est_sources.shape[-1]):
                out_path = os.path.join(save_dir, f"{f.stem}_source{i}.wav")
                model.save_audio(out_path, est_sources[:, :, i])
            print(f"  ✓ {f.name}: separated into {est_sources.shape[-1]} sources")
        print(f"✓ SepFormer outputs saved to {save_dir}")
    except Exception as e:
        print(f"✗ SepFormer: {e}")


def main():
    print("=" * 60)
    print(f"AUDIO/SPEECH — Task: {TASK}")
    print("=" * 60)
    audio_dir, data = download_audio_samples()
    if TASK in ("transcription", "denoising"):
        results = run_whisper(audio_dir)
        if results:
            out = os.path.join(os.path.dirname(__file__), "transcriptions.json")
            with open(out, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
            print(f"Saved to {out}")
    elif TASK == "cloning": run_voice_cloning()
    elif TASK == "classification":
        run_wav2vec2_clf(audio_dir)
    elif TASK == "separation":
        run_sepformer(audio_dir)
    else:
        results = run_whisper(audio_dir)


if __name__ == "__main__":
    main()
