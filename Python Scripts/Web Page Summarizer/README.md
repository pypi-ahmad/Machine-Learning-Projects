# Web Page Summation

> A command-line utility for summarizing web pages using extractive text summarization (LexRank), with support for single URLs and bulk CSV processing.

## Overview

This project provides a CLI tool that takes a website URL and produces a text summary using the `sumy` library's LexRank algorithm. It also supports bulk processing of URLs from a CSV file. Additionally, the `utils/` directory contains an experimental seq2seq abstractive summarization model built with TensorFlow (separate from the main CLI tool).

## Features

- **Single URL summarization** via the `simple` action
- **Bulk CSV processing** via the `bulk` action — reads URLs from a CSV column named `website`, appends summaries, and outputs a new CSV
- Configurable summary length (number of sentences) and language
- Extractive summarization using LexRank (via `sumy`)
- CLI with `argparse` for structured argument parsing
- Logging to `applog.log`
- Experimental seq2seq LSTM model with attention mechanism for abstractive summarization (in `utils/`)

## Project Structure

```
Web_page_summation/
├── .gitignore
├── app.py                  # Main CLI entry point
├── README.md
├── requirements.txt
└── utils/
    ├── __init__.py         # Empty init file
    ├── comparison.py       # ROUGE and BLEU evaluation metrics
    ├── model.py            # TensorFlow seq2seq model with attention
    ├── prepare.py          # Decompresses training data
    ├── summarize.py        # Core LexRank summarization function
    ├── test.py             # Model inference/validation script
    ├── train.py            # Model training script
    └── utils.py            # Data preprocessing, dictionary building, batching
```

## Requirements

- Python 3.x
- `sumy` (0.8.1) — Extractive text summarization (LexRank)
- `nltk` (3.5) — Natural language processing / tokenization
- `numpy` (1.19.1)
- `tensorflow` (2.3.0) — Required only for the seq2seq model in `utils/`
- `gensim` (3.8.3)
- `newspaper` (0.1.0.7)
- `sumeval` (0.2.2) — ROUGE/BLEU evaluation metrics
- `wget` (3.2)
- `model` (0.6.0)
- `utils` (1.0.1)

Full list in `requirements.txt`.

## Installation

```bash
cd Web_page_summation
pip install -r requirements.txt
```

For NLTK data (required by `sumy`):

```python
import nltk
nltk.download('punkt')
```

## Usage

### Summarize a single URL

```bash
python app.py simple --url https://example.com --sentence 3 --language english
```

### Summarize with defaults (2 sentences, English)

```bash
python app.py simple --url https://example.com
```

### Bulk summarize from CSV

The CSV must contain a column named `website` with URLs:

```bash
python app.py bulk --path ./data/urls.csv --sentence 2 --language english
```

This reads the CSV, summarizes each URL, appends a `summary` column, and saves as `beneficiary.csv` (then moves it to the input path).

### CLI arguments

| Argument | Description | Default |
|---|---|---|
| `action` | `simple` (single URL) or `bulk` (CSV batch) | Required |
| `--url` | URL to summarize (for `simple` action) | — |
| `--path` | Path to CSV file (for `bulk` action) | — |
| `--sentence` | Number of sentences in summary | `2` |
| `--language` | Language for summarization | `English` |

## How It Works

### CLI / Summarization (`app.py` + `utils/summarize.py`)
1. `app.py` parses CLI arguments with `argparse`.
2. For `simple`: calls `summarize(url, language, sentence_count)`.
3. For `bulk`: reads a CSV, finds the `website` column, summarizes each URL, appends results to a new column, and writes `beneficiary.csv`.
4. `summarize()` uses `sumy`'s `HtmlParser` to fetch and parse the URL, applies `LexRankSummarizer` with stemming and stop words, and returns the concatenated sentence output.

### Seq2seq Model (`utils/model.py`, `train.py`, `test.py`)
- A bidirectional LSTM encoder-decoder with Bahdanau attention mechanism.
- Uses beam search decoding during inference.
- Training data is expected as gzipped text files in `sumdata/train/`.
- Evaluation uses ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-BE, and BLEU metrics (`comparison.py`).

## Configuration

- **Training hyperparameters** (in `utils/train.py`): `num_hidden=150`, `num_layers=2`, `beam_width=10`, `embedding_size=300`, `learning_rate=1e-3`, `batch_size=64`, `num_epochs=10`, `keep_prob=0.8`.
- **Data paths** in `utils/utils.py` and `utils/prepare.py` default to `'.'` — must be updated to point to actual data directories.
- **GloVe embeddings** path in `utils/utils.py`: expects `glove/model_glove_300.pkl`.

## Limitations

- The `summarize()` function has a bug: each sentence is appended to `result` twice (once before and once inside the `try` block).
- Bare `except` clauses throughout the codebase swallow all errors silently.
- The seq2seq model code uses TensorFlow 1.x APIs (`tf.contrib`, `tf.reset_default_graph`) but `requirements.txt` specifies TensorFlow 2.3.0 — these are incompatible.
- Data paths in utility files are hardcoded to `'.'` and require manual adjustment.
- The `bulk` action outputs to a hardcoded filename `beneficiary.csv` regardless of input.
- No input validation for URLs.

## Security Notes

No API keys or credentials in the code.

## License

Not specified.
