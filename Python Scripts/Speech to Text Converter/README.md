# Speech to Text Converter

A command-line utility that records one microphone phrase, sends it to Google
Speech Recognition, and writes a successful transcript to a text file.

## Run

```powershell
uv sync
uv run python speech-to-text.py
```

The default output is `output.txt` next to the script.

```powershell
uv run python speech-to-text.py --output transcript.txt --timeout 10 --phrase-time-limit 30
```

The tool calibrates for ambient noise for half a second before listening.
`--timeout` limits how long it waits for speech, and `--phrase-time-limit`
limits the recorded phrase duration.

## Privacy and limitations

- Microphone access and an active internet connection are required.
- Audio is sent to Google's recognition service. Do not use it for sensitive
  speech unless that disclosure and data handling are acceptable.
- A file is written only after a transcript is successfully returned. Failed or
  unrecognized recordings leave the existing output file unchanged.
- Recognition quality depends on the microphone, noise, language, accent, and
  service availability.

## Dependencies

The project uses SpeechRecognition and PyAudio. It pins Python to the 3.13
series for the installed Windows microphone dependency, with exact versions
locked in `uv.lock`.
