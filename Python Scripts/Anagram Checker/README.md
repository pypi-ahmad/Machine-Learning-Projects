# Anagram Checker

`main.py` checks whether words or phrases are anagrams, groups a list into
anagram families, and searches a UTF-8 wordlist for matches.

## Install and run

```powershell
cd "Python Scripts/Anagram Checker"
uv sync
uv run python main.py check listen silent
uv run python main.py group listen silent evil vile cat
uv run python main.py find listen C:\path\to\words.txt
```

Run `uv run python main.py` without a subcommand for the interactive menu.

The comparison is case-insensitive and ignores whitespace. Punctuation remains
significant, so `rail-safety` and `fairy tales` are not treated as anagrams.

## Commands

- `check FIRST SECOND` reports whether two inputs are anagrams and shows
  unmatched characters when they are not.
- `group WORD [WORD ...]` prints each family that contains at least two words.
- `find WORD WORDLIST` prints wordlist entries that are anagrams of `WORD`.

The project uses only the Python standard library.
