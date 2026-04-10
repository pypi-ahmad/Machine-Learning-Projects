# Handwritten Note to Markdown

> **Task:** Handwriting OCR + Markdown Conversion &nbsp;|&nbsp; **Key:** `handwritten_note_to_markdown` &nbsp;|&nbsp; **Framework:** TrOCR

---

## Overview

Converts handwritten notes into Markdown or plain text using Microsoft's TrOCR model. The pipeline segments a page image into individual text lines via horizontal projection profiles, recognises each line with TrOCR, then formats the output as structured Markdown with optional header/list detection and paragraph breaks. Includes per-line confidence scores and supports `.md`, `.txt`, and JSON export.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Handwriting OCR + Document Formatting |
| **Line Segmentation** | Horizontal projection profile + morphological row merging |
| **OCR Engine** | TrOCR (`microsoft/trocr-base-handwritten`) via HuggingFace Transformers |
| **Confidence** | Per-token softmax probabilities from beam search |
| **Formatting** | Heuristic header detection (line height), list detection (x-offset), paragraph gaps |
| **Dataset** | IAM Handwriting Words from Hugging Face |

## Dataset

- **Source:** Hugging Face — `nielsr/iam_handwriting_words`
- **License:** See dataset page for license terms
- **Download:** Automatic on first `python train.py` run via `DatasetResolver`
- **Force re-download:** `python train.py --force-download`

## Project Structure

```
Handwritten Note to Markdown/
└── Source Code/
    ├── config.py               # NoteConfig dataclass + YAML/JSON loader
    ├── line_segmenter.py       # Projection-profile line segmentation
    ├── ocr_engine.py           # TrOCR wrapper with confidence scores
    ├── markdown_formatter.py   # Recognised lines → Markdown / plain text
    ├── parser.py               # Pipeline orchestrator (segment → OCR → format)
    ├── validator.py            # Confidence checks + missing-text warnings
    ├── visualize.py            # Annotated overlay with line boxes + panel
    ├── export.py               # .md / .txt / JSON export
    ├── infer.py                # CLI pipeline (single image / batch directory)
    ├── modern.py               # CVProject subclass — @register("handwritten_note_to_markdown")
    ├── train.py                # Dataset download + pipeline evaluation
    ├── data_bootstrap.py       # Dataset bootstrap via scripts/download_data.py
    ├── note_config.yaml        # Sample inference configuration
    ├── requirements.txt        # Project-level dependencies
    └── README.md               # This file
```

## Quick Start

### Inference (CLI)

```bash
cd "Handwritten Note to Markdown/Source Code"

# Single image — segment, recognise, display
python infer.py --source note.jpg

# Export to Markdown
python infer.py --source note.jpg --export-md output.md --no-display

# Export to plain text
python infer.py --source note.jpg --export-txt output.txt --no-display

# Batch with JSON export + confidence annotations
python infer.py --source notes/ --export-json results.json --confidence --no-display

# Save annotated images
python infer.py --source notes/ --save-annotated --no-display

# Single-line images (skip segmentation)
python infer.py --source word.png --no-segment

# Use a different TrOCR model
python infer.py --source note.jpg --model microsoft/trocr-large-handwritten
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("handwritten_note_to_markdown", "path/to/note.jpg")

print(result["result"].markdown)          # formatted Markdown
print(result["result"].plain_text)        # plain text
print(result["result"].num_lines)         # lines found
print(result["result"].mean_confidence)   # average confidence
print(result["report"].summary())         # validation report
```

### Training / Evaluation

```bash
cd "Handwritten Note to Markdown/Source Code"
python train.py                              # download dataset + evaluate
python train.py --force-download             # re-download dataset
python train.py --max-samples 50             # evaluate on more samples
python train.py --model microsoft/trocr-large-handwritten
```

## Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌────────┐    ┌───────────┐    ┌──────────┐
│  Input Image │───▶│  Line        │───▶│ TrOCR  │───▶│ Markdown  │───▶│ Validate │
│  (page/note) │    │  Segmenter   │    │ Engine │    │ Formatter │    │ + Export  │
│              │    │ (projection  │    │        │    │ (headers, │    │          │
│              │    │  profile)    │    │        │    │  lists)   │    │          │
└─────────────┘    └──────────────┘    └────────┘    └───────────┘    └──────────┘
```

## Line Segmentation

The segmenter uses horizontal projection profiles to split a page into lines:

1. **Binarisation** — Otsu threshold to get ink pixels
2. **Horizontal projection** — sum ink pixels per row
3. **Row detection** — contiguous rows above threshold → text line
4. **Row merging** — merge rows closer than `merge_gap` pixels
5. **Padding** — add configurable padding above/below each line crop
6. **X-offset** — detect left-most ink pixel per line (for indent/list detection)

For single-line images (e.g. word crops), segmentation can be disabled via `--no-segment`.

## Markdown Formatting

| Feature | Detection Method |
|---------|-----------------|
| **Headers** | Line height ≥ 1.8× median → `## heading` |
| **List items** | X-offset ≥ 40px → `- item` (preserves existing bullets) |
| **Paragraphs** | Vertical gap ≥ 2× median gap → blank line |
| **Confidence** | Lines below threshold get `<!-- low confidence: 0.XX -->` annotations |

## TrOCR Models

| Model | Size | Quality | Use Case |
|-------|------|---------|----------|
| `microsoft/trocr-small-handwritten` | ~60M | Good | Fast inference |
| `microsoft/trocr-base-handwritten` | ~330M | Better | **Default** |
| `microsoft/trocr-large-handwritten` | ~560M | Best | Maximum accuracy |

Swap models via `--model` flag or `model_name` in config.

## Multilingual Extension

The architecture is designed for future multilingual support:

- `language` field in config (currently `"en"`)
- TrOCR model is swappable via `model_name` config
- Multilingual TrOCR variants available on HuggingFace Hub
- Line segmenter is language-agnostic (projection profiles)
- Markdown formatter is language-agnostic

## Export Formats

### Markdown (`.md`)

```markdown
## Meeting Notes

Discussed project timeline and milestones.

- Action item: update documentation
- Action item: schedule review  <!-- low confidence: 0.35 -->
```

### JSON

```json
{
  "total_notes": 1,
  "exported_at": "2026-04-09T12:00:00+00:00",
  "records": [
    {
      "source": "note.jpg",
      "num_lines": 5,
      "mean_confidence": 0.82,
      "markdown": "## Meeting Notes\n...",
      "lines": [
        {"text": "Meeting Notes", "confidence": 0.95, "height": 48},
        {"text": "Discussed project timeline", "confidence": 0.88, "height": 28}
      ]
    }
  ]
}
```

## Dependencies

```
pip install transformers torch pillow opencv-python numpy pyyaml
```

The TrOCR model is downloaded automatically on first run (~330 MB for base).
