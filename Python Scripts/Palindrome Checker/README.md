# Palindrome Checker

An interactive terminal tool for checking palindromes and finding palindromic substrings.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose a menu option to check one input, list palindromic fragments, or enter a batch of inputs. The regular checker ignores case, spaces, and punctuation; substring matching operates on the provided text.

## Behavior and limits

- The tool works entirely in memory and does not write files or use a network connection.
- Substring results are unique and sorted from longest to shortest.
- Long text can take longer to inspect because the substring search checks candidate ranges around every character.
