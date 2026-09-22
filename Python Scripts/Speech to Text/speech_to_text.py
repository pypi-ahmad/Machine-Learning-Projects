"""Record one microphone phrase and transcribe it with Google Speech Recognition."""

from __future__ import annotations

import argparse
from pathlib import Path

import speech_recognition as sr


def transcribe_microphone(
    language: str, timeout: float | None, phrase_time_limit: float | None, device_index: int | None
) -> str:
    """Capture one phrase, then submit it to Google's recognition service."""
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as microphone:
        print("Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(microphone, duration=1)
        print("Listening...")
        audio = recognizer.listen(microphone, timeout=timeout, phrase_time_limit=phrase_time_limit)
    return recognizer.recognize_google(audio, language=language)


def save_transcript(text: str, output: Path, overwrite: bool) -> None:
    """Save a successful transcript without replacing an existing file by default."""
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8") as file:
        file.write(text + "\n")


def main() -> None:
    """Parse recording options, transcribe one phrase, and save it locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en-US", help="Google recognition language code")
    parser.add_argument("--timeout", type=float, default=10, help="seconds to wait for speech; use 0 for no limit")
    parser.add_argument("--phrase-time-limit", type=float, default=30, help="maximum captured phrase seconds; use 0 for no limit")
    parser.add_argument("--device-index", type=int, help="optional microphone device index")
    parser.add_argument("--output", type=Path, default=Path("you_said_this.txt"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing transcript")
    args = parser.parse_args()
    if args.timeout < 0 or args.phrase_time_limit < 0:
        parser.error("timeout values cannot be negative")
    timeout = None if args.timeout == 0 else args.timeout
    phrase_time_limit = None if args.phrase_time_limit == 0 else args.phrase_time_limit
    try:
        transcript = transcribe_microphone(args.language, timeout, phrase_time_limit, args.device_index)
        save_transcript(transcript, args.output, args.overwrite)
    except sr.WaitTimeoutError:
        raise SystemExit("No speech was detected before the listen timeout.") from None
    except sr.UnknownValueError:
        raise SystemExit("Google could not understand the recorded speech.") from None
    except sr.RequestError as error:
        raise SystemExit(f"Google Speech Recognition request failed: {error}") from error
    except (OSError, FileExistsError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved transcript to {args.output}")


if __name__ == "__main__":
    main()
