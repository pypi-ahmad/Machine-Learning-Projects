"""Transcribe one microphone recording with Google Speech Recognition.

Usage:
    uv run python speech-to-text.py [--output output.txt]
"""

import argparse
from pathlib import Path

import speech_recognition as sr

DEFAULT_OUTPUT = Path(__file__).with_name("output.txt")


def transcribe(timeout: float | None, phrase_time_limit: float | None) -> str:
    """Capture one phrase and return its Google Speech Recognition transcript."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Speak now.")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError as error:
        raise RuntimeError("Speech was not understood; no transcript was saved.") from error
    except sr.RequestError as error:
        raise RuntimeError("Google Speech Recognition was unavailable; no transcript was saved.") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one microphone recording.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Transcript output path")
    parser.add_argument("--timeout", type=float, help="Seconds to wait for speech")
    parser.add_argument("--phrase-time-limit", type=float, help="Maximum recorded phrase length in seconds")
    args = parser.parse_args()
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.phrase_time_limit is not None and args.phrase_time_limit <= 0:
        parser.error("--phrase-time-limit must be greater than zero")

    try:
        transcript = transcribe(args.timeout, args.phrase_time_limit)
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"Transcription failed: {error}") from error

    args.output.write_text(transcript + "\n", encoding="utf-8")
    print(f"Saved transcript to {args.output}")


if __name__ == "__main__":
    main()
