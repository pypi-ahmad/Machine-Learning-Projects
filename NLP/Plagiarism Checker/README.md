# Plagiarism Checker

> A Python script that detects plagiarism between text documents using TF-IDF vectorization and cosine similarity.

## Overview

This script compares all `.txt` files in the current directory against each other to compute pairwise similarity scores. It uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors and cosine similarity to measure how similar each pair of documents is.

## Features

- Automatic detection of all `.txt` files in the current directory
- TF-IDF vectorization of textual content
- Pairwise cosine similarity computation between all documents
- Outputs similarity scores as tuples: `(file_a, file_b, score)`
- Deduplicates pairs (reports each pair only once using sorted names and a set)

## Project Structure

```
Plagiarism-Checker/
├── assets/
│   └── Capture1.PNG    # Screenshot
├── file1.txt           # Sample text document
├── file2.txt           # Sample text document
├── file3.txt           # Sample text document
├── file4.txt           # Sample text document
└── plagiarism.py       # Main script
```

## Requirements

- Python 3.x
- `scikit-learn`
- `os` (standard library)

## Installation

```bash
cd "Plagiarism-Checker"
pip install scikit-learn
```

## Usage

Place `.txt` files to compare in the same directory as `plagiarism.py`, then run:

```bash
python plagiarism.py
```

### Example Output

```
('file1.txt', 'file2.txt', 0.8543)
('file1.txt', 'file3.txt', 0.2311)
('file2.txt', 'file3.txt', 0.1784)
```

The similarity score ranges from `0.0` (completely different) to `1.0` (identical).

## How It Works

1. **File discovery**: Lists all `.txt` files in the current working directory using `os.listdir()`.
2. **File reading**: Opens and reads the contents of each text file.
3. **Vectorization**: Uses `TfidfVectorizer().fit_transform()` to convert all documents into TF-IDF feature arrays.
4. **Pairing**: Zips filenames with their vectors and iterates through all unique pairs.
5. **Similarity**: Computes `cosine_similarity()` between each pair of document vectors.
6. **Deduplication**: Sorts file names in each pair and stores results in a `set` to avoid duplicate comparisons.
7. **Output**: Prints each unique pair with its similarity score.

## Configuration

No configuration needed. The script automatically processes all `.txt` files in its directory.

## Limitations

- Compares **all** `.txt` files in the directory — including any unrelated text files.
- No threshold filtering — all pairs are reported regardless of similarity score.
- No support for other file formats (PDF, DOCX, etc.).
- Reads files using `open(File).read()` without specifying encoding or closing the file handle.
- No command-line arguments for specifying files or directories.
- Output is printed to console only — no export option.

## Security Notes

No security concerns identified.

## License

Not specified.
