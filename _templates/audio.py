"""Audio/Speech template: Whisper, Wav2Vec2, XTTS-v2 — April 2026"""
import textwrap


def generate(project_path, config):
    task = config.get("task", "transcription")  # transcription, classification, cloning, denoising

    return textwrap.dedent(f'''\
        """
        Modern Audio/Speech Pipeline (April 2026)
        Models: Whisper (ASR), Wav2Vec2 (classification), XTTS-v2 (TTS/cloning)
        """
        import os, warnings
        import numpy as np
        from pathlib import Path

        warnings.filterwarnings("ignore")

        TASK = "{task}"


        def find_audio_files():
            """Find audio files in the project directory."""
            data_dir = Path(os.path.dirname(__file__))
            audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
            files = []
            for ext in audio_exts:
                files.extend(data_dir.rglob(f"*{{ext}}"))
            return files


        def run_whisper_transcription(audio_files):
            """Transcribe audio using Whisper."""
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32

                model_id = "openai/whisper-large-v3-turbo"
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
                ).to(device)
                processor = AutoProcessor.from_pretrained(model_id)

                asr = pipeline(
                    "automatic-speech-recognition",
                    model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch_dtype, device=device,
                )

                results = []
                for f in audio_files[:10]:
                    result = asr(str(f), return_timestamps=True)
                    results.append({{"file": f.name, "text": result["text"]}})
                    print(f"  ✓ {{f.name}}: {{result['text'][:100]}}...")

                return results
            except Exception as e:
                print(f"✗ Whisper: {{e}}")
                return []


        def run_audio_classification(audio_files):
            """Classify audio using Wav2Vec2."""
            try:
                import torch
                import torchaudio
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_name = "facebook/wav2vec2-base"

                # Check for CSV with labels
                import pandas as pd
                data_dir = Path(os.path.dirname(__file__))
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    print(f"Labels file: {{csv_files[0].name}}")

                feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                model = AutoModelForAudioClassification.from_pretrained(
                    model_name, num_labels=10
                ).to(device)

                print(f"✓ Wav2Vec2 model loaded on {{device}}")
                print(f"  Found {{len(audio_files)}} audio files")
                return {{"model": model}}
            except Exception as e:
                print(f"✗ Audio Classification: {{e}}")
                return None


        def run_voice_cloning():
            """Voice cloning / TTS using XTTS-v2."""
            try:
                from TTS.api import TTS

                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

                # Find reference audio
                audio_files = find_audio_files()
                if audio_files:
                    ref_audio = str(audio_files[0])
                    out_path = os.path.join(os.path.dirname(__file__), "cloned_output.wav")
                    tts.tts_to_file(
                        text="Hello, this is a cloned voice sample using XTTS version 2.",
                        speaker_wav=ref_audio,
                        language="en",
                        file_path=out_path,
                    )
                    print(f"✓ Voice cloned → {{out_path}}")
                else:
                    out_path = os.path.join(os.path.dirname(__file__), "tts_output.wav")
                    tts.tts_to_file(
                        text="Hello, this is a text to speech sample.",
                        file_path=out_path,
                    )
                    print(f"✓ TTS output → {{out_path}}")
            except Exception as e:
                print(f"✗ XTTS: {{e}}")


        def run_denoising(audio_files):
            """Audio denoising using modern approach."""
            try:
                import torch
                import torchaudio

                device = "cuda" if torch.cuda.is_available() else "cpu"

                for f in audio_files[:5]:
                    waveform, sr = torchaudio.load(str(f))
                    waveform = waveform.to(device)

                    # Simple spectral gating denoising
                    n_fft = 2048
                    spec = torch.stft(waveform[0], n_fft=n_fft, return_complex=True)
                    mag = spec.abs()
                    phase = spec.angle()

                    # Noise floor estimation (first 0.5s)
                    noise_frames = int(0.5 * sr / (n_fft // 2))
                    noise_profile = mag[:, :noise_frames].mean(dim=1, keepdim=True)
                    mask = (mag > noise_profile * 2).float()
                    clean_spec = mag * mask * torch.exp(1j * phase)
                    clean = torch.istft(clean_spec, n_fft=n_fft)

                    out_path = f.parent / f"denoised_{{f.name}}"
                    torchaudio.save(str(out_path), clean.unsqueeze(0).cpu(), sr)
                    print(f"  ✓ Denoised: {{f.name}} → {{out_path.name}}")
            except Exception as e:
                print(f"✗ Denoising: {{e}}")


        def main():
            print("=" * 60)
            print("MODERN AUDIO/SPEECH PIPELINE")
            print(f"Task: {{TASK}}")
            print("=" * 60)

            audio_files = find_audio_files()
            print(f"Found {{len(audio_files)}} audio files")

            if TASK == "transcription":
                results = run_whisper_transcription(audio_files)
                if results:
                    import json
                    out = os.path.join(os.path.dirname(__file__), "transcriptions.json")
                    with open(out, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"\\nTranscriptions saved to {{out}}")

            elif TASK == "classification":
                run_audio_classification(audio_files)

            elif TASK == "cloning":
                run_voice_cloning()

            elif TASK == "denoising":
                run_denoising(audio_files)


        if __name__ == "__main__":
            main()
    ''')
