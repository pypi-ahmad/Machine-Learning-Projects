# Word Counter

Analyze typed text or a UTF-8 text file. The tool reports character, word, line, sentence, paragraph, vocabulary, reading-time, and approximate readability statistics.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The project uses only the Python standard library.

## Analyze a file

```powershell
cd "Python Scripts\Word Counter"
uv run python .\main.py .\document.txt
```

## Analyze typed text

Run without a file path:

```powershell
uv run python .\main.py
```

Choose **Analyze typed text**, enter one or more lines, and type `###` alone on a line to finish.

## Notes

- Files are read as UTF-8 with replacement for undecodable characters.
- The syllable count and Flesch reading-ease score are rough English-language estimates.
- The tool reads text locally and does not send it anywhere.
