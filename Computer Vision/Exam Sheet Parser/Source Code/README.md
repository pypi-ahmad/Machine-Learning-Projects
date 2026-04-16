# Exam Sheet Parser

> **Task:** OCR + Layout Parsing &nbsp;|&nbsp; **Key:** `exam_sheet_parser` &nbsp;|&nbsp; **Framework:** PaddleOCR-first with EasyOCR fallback

---

## Overview

Document parser that identifies and extracts exam sheet structure from scanned images: headings, numbered questions, MCQ option blocks, marks annotations, and section headers. The runtime prefers PaddleOCR when its Paddle backend is available, and falls back to EasyOCR on unsupported environments. Layout analysis stays rule-based and transparent. Outputs annotated previews and structured JSON/CSV.

## Pipeline

```
Exam image → OCR detect + recognise → layout classify → question grouping → validate → overlay → export
```

1. **OCR** — PaddleOCR is attempted first; if the local runtime cannot load Paddle, the pipeline switches to EasyOCR automatically and still returns polygon-like OCR blocks with confidence scores.
2. **Layout Classification** — Each block is classified as `heading`, `section`, `question`, `mcq_option`, `marks`, or `body` using spatial heuristics (font height ratio, position) and regex patterns (question numbers, option letters, marks annotations).
3. **Question Grouping** — Sequential layout elements are grouped into structured `QuestionBlock` objects with number, text, marks, MCQ options, and body lines.
4. **Validation** — Quality checks: no text detected, no questions found, low confidence, missing marks.
5. **Overlay** — Colour-coded bounding polygons by role, structural labels, and a summary panel.
6. **Export** — JSON (full structured extraction) and/or CSV (one row per question).

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | OCR + Layout Parsing |
| **Modern Stack** | PaddleOCR or EasyOCR + rule-based layout classification |
| **Dataset** | Public dataset when available, otherwise synthetic exam sheets generated locally |
| **Key Metrics** | Question detection rate, OCR accuracy |

## Dataset

- **Source:** The bootstrap first tries the registered public dataset path for `exam_sheet_parser`. If that helper is unavailable, it generates a small synthetic exam-sheet corpus locally.
- **Synthetic fallback:** Includes headings, section lines, numbered questions, MCQ options, and marks annotations to exercise the parser end to end.
- **Download:** Automatic on first `python train.py` or `python infer.py --force-download` run via `ensure_exam_sheet_dataset()`
- **Force re-download:** `python train.py --force-download`
- **Bootstrap:** `python data_bootstrap.py` (idempotent, uses `.ready` marker)

## Project Structure

```
Exam Sheet Parser/
└── Source Code/
    ├── config.py              # ExamSheetConfig dataclass — all tunables
    ├── ocr_engine.py          # PaddleOCR-first OCR wrapper with EasyOCR fallback
    ├── layout_parser.py       # Rule-based structural classification
    ├── parser.py              # ExamSheetPipeline — OCR → layout → questions
    ├── validator.py           # Quality checks + ValidationReport
    ├── visualize.py           # Colour-coded overlay renderer by role
    ├── export.py              # JSON / CSV exporter (context manager)
    ├── infer.py               # CLI — image / directory inference
    ├── modern.py              # CVProject subclass — @register("exam_sheet_parser")
    ├── train.py               # Evaluation entry point (OCR is pre-trained)
    ├── data_bootstrap.py      # Idempotent dataset prep + synthetic fallback
    ├── exam_config.yaml       # Sample YAML configuration
    ├── requirements.txt       # Project dependencies
    └── README.md              # This file
```

## Quick Start

### CLI Inference

```bash
# Single exam sheet image
python infer.py --source exam.jpg

# Directory of scanned sheets with JSON export
python infer.py --source scans/ --export-json results.json

# Export questions to CSV
python infer.py --source test.png --export-csv questions.csv

# Headless mode with annotated output
python infer.py --source exams/ --no-display --save-annotated

# Custom config
python infer.py --source exam.jpg --config exam_config.yaml
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("exam_sheet_parser", "path/to/exam.jpg")
for q in result["result"].questions:
    marks = f" [{q.marks}m]" if q.marks else ""
    opts = f" ({len(q.options)} options)" if q.options else ""
    print(f"Q{q.number}{marks}{opts}: {q.text[:60]}")
```

### Evaluation

```bash
cd "Exam Sheet Parser/Source Code"
python train.py                    # evaluate on dataset
python train.py --force-download   # re-download dataset
python train.py --max-samples 50   # evaluate more samples
```

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--source` | Image path or directory of exam sheet scans |
| `--config` | Path to YAML/JSON config file |
| `--lang` | OCR language (default: `en`) |
| `--gpu` | Enable GPU for OCR when the selected backend supports it |
| `--export-json` | JSON export path |
| `--export-csv` | CSV export path |
| `--save-annotated` | Save annotated images |
| `--output-dir` | Output directory (default: `output`) |
| `--no-display` | Disable GUI window |
| `--force-download` | Force dataset re-download |

## Layout Classification

The `layout_parser.py` module classifies each OCR block using these rules (in priority order):

| Rule | Role | Trigger |
|------|------|---------|
| Question number | `question` | Regex match: `Q1.`, `3)`, `Q. 12:` |
| MCQ option | `mcq_option` | Regex match: `A)`, `(B).`, `c.` |
| Section keyword | `section` | Starts with: section, part, instructions, note, answer, total, time |
| Font height | `heading` | Block height > median × 1.4 |
| All-caps short | `heading` | ≤60 chars, all uppercase, mostly alphabetic |
| Marks annotation | `marks` | Regex match: `[5 marks]`, `(3 pts)` |
| Default | `body` | Everything else |

## JSON Export Format

```json
{
  "total_sheets": 1,
  "exported_at": "2026-04-09T12:00:00+00:00",
  "records": [{
    "source": "exam.jpg",
    "headings": ["MATHEMATICS FINAL EXAM"],
    "sections": ["Section A: Answer all questions"],
    "num_questions": 5,
    "total_marks": 50,
    "questions": [{
      "number": 1,
      "text": "Q1. Evaluate the integral:",
      "marks": 10,
      "options": [],
      "option_letters": [],
      "body_lines": ["∫ x² dx from 0 to 1"],
      "confidence": 0.92
    }]
  }]
}
```

## Configuration

All tunables are defined in `ExamSheetConfig` (see [config.py](config.py)). Override via:

1. **YAML config file:** `python infer.py --config exam_config.yaml --source img.jpg`
2. **CLI flags:** `--gpu`, `--lang ch`, `--no-display`, etc.
3. **Python:** `ExamSheetConfig(ocr_lang="ch", heading_font_ratio=1.5)`.

## Features

- PaddleOCR-first OCR with automatic EasyOCR fallback on unsupported runtimes
- Rule-based layout classification (heading, question, MCQ, marks, section, body)
- Question grouping with marks and MCQ option extraction
- Colour-coded overlay with structural role labels
- Summary panel with question count, total marks, headings
- JSON and CSV structured export
- Configurable regex patterns for question/option/marks detection
- Quality validation with configurable rules
- Sample YAML configuration file
- Idempotent dataset bootstrap with `.ready` marker and synthetic fallback

## Dependencies

```bash
pip install -r requirements.txt
```

If PaddleOCR is installed without a matching Paddle runtime on your platform, the project still runs by falling back to EasyOCR automatically.
