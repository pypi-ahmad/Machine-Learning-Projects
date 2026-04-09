# Document Summary Creator

> A Python script that generates an extractive text summary from a plain text file using LexRank and spaCy NLP.

## Overview

This project reads a `.txt` file, preprocesses the text using spaCy for morphological analysis and sentence segmentation, then applies the LexRank summarization algorithm (via the `sumy` library) to produce a condensed summary. The summary is saved as a new text file alongside the original.

## Features

- Reads plain text files and preprocesses content (line break removal, character normalization with `neologdn`)
- Splits text into sentences using spaCy's English NLP model (`en_core_web_sm`)
- Generates an extractive summary using the **LexRank** algorithm
- Outputs summary length as 20% (2/10) of the original sentence count
- Automatically saves the summary to `<original_name>_summary.txt`

## Project Structure

```
Document-Summary-Creator/
├── main.py                    # Entry point — reads file, calls summarizer, writes output
├── preprocessing.py           # EnglishCorpus class for NLP preprocessing
├── summary_make.py            # Summarization logic using LexRank via sumy
├── origin_text.txt            # Sample input text
├── origin_text_summary.txt    # Sample output summary
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `sumy==0.8.1`
- `spacy==2.3.2`
- `neologdn==0.4`
- spaCy English model: `en_core_web_sm`
- NLTK punkt tokenizer data

## Installation

```bash
cd Document-Summary-Creator
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

## Usage

```bash
python main.py
```

When prompted, enter the path to a `.txt` file:

```
please input text's filepath-> origin_text.txt
```

The summary will be saved as `origin_text_summary.txt` in the same directory.

## How It Works

1. **`main.py`** — Reads the input file path, loads the text, calls `summarize_sentences()`, and writes the summary to a file named `<original>_summary.txt`.
2. **`preprocessing.py`** — The `EnglishCorpus` class:
   - Loads the `en_core_web_sm` spaCy model
   - Removes newlines and normalizes special characters using `neologdn`
   - Segments text into sentences with spaCy's sentence boundary detection
   - Builds a whitespace-tokenized corpus for `sumy`
3. **`summary_make.py`** — Preprocesses the input, parses it with `sumy`'s `PlaintextParser`, then runs `LexRankSummarizer` with English stop words to extract the top 20% of sentences.

## Configuration

- **Summary length ratio**: Hardcoded as `len(corpus) * 2 // 10` (20% of sentences) in `summary_make.py`.
- **Language**: Defaults to `"english"` in `summarize_sentences()`.
- **spaCy model**: Hardcoded to `en_core_web_sm` in `preprocessing.py`.

## Limitations

- Only supports English text (hardcoded language and spaCy model).
- Input must be a `.txt` file — the output path is derived by finding `.txt` in the filename, which may break if `.txt` appears elsewhere in the path.
- The summary ratio (20%) is hardcoded and not user-configurable.
- Older pinned dependency versions (`spacy==2.3.2`, `sumy==0.8.1`) may have compatibility issues with newer Python versions.
- No error handling for missing files or empty input.

## License

Not specified.
