"""Translate typed or spoken text and optionally speak the result."""

from __future__ import annotations

import argparse
import asyncio

import pyttsx3
import sounddevice as sd
import speech_recognition as sr
from googletrans import Translator


def parse_args() -> argparse.Namespace:
    """Parse translation and audio options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Text to translate instead of recording audio")
    parser.add_argument("--source", default="en", help="Source language code (default: en)")
    parser.add_argument("--target", default="ca", help="Target language code (default: ca)")
    parser.add_argument(
        "--listen-language",
        default="en-IN",
        help="Speech-recognition language code (default: en-IN)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Microphone recording length in seconds (default: 5)",
    )
    parser.add_argument("--mute", action="store_true", help="Do not speak the translation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selected configuration without using audio or translation services",
    )
    return parser.parse_args()


def record_audio(seconds: float, sample_rate: int = 16_000) -> sr.AudioData:
    """Record mono 16-bit audio with the default microphone."""
    if seconds <= 0:
        raise ValueError("Recording length must be greater than zero.")
    print(f"Listening for {seconds:g} seconds...")
    recording = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    return sr.AudioData(recording.tobytes(), sample_rate, 2)


def recognize_speech(audio: sr.AudioData, language: str) -> str:
    """Recognize recorded speech through the configured Google service."""
    print("Recognizing...")
    return sr.Recognizer().recognize_google(audio, language=language)


def translate_text(text: str, source: str, target: str) -> str:
    """Translate text with googletrans and return the translated text."""
    translation = asyncio.run(Translator().translate(text, src=source, dest=target))
    return translation.text


def speak(text: str) -> None:
    """Speak text through the platform's default text-to-speech voice."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main() -> None:
    """Run the translator command-line interface."""
    args = parse_args()
    if args.dry_run:
        print(
            f"Would translate {args.source!r} to {args.target!r}; "
            f"audio capture: {'off' if args.text else f'{args.seconds:g} seconds'}; "
            f"speech output: {'off' if args.mute else 'on'}"
        )
        return

    try:
        text = args.text or recognize_speech(
            record_audio(args.seconds), args.listen_language
        )
        translated = translate_text(text, args.source, args.target)
    except (ValueError, sr.RequestError, sr.UnknownValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(translated)
    if not args.mute:
        speak(translated)


if __name__ == "__main__":
    main()
