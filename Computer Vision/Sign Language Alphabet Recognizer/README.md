# Sign Language Alphabet Recognizer

Static **ASL alphabet recognition** from hand landmarks using
MediaPipe Hands and a lightweight MLP classifier.  Supports 24
static letters (A–Y, excluding J and Z which require motion).

---

## Scope & Limitations

> **This project recognises only static ASL fingerspelling signs.**
>
> - **24 letters only.**  J and Z require motion and are excluded.
> - **Landmark-based classifier.**  Hand-landmark coordinates (not
>   raw pixels) are fed to an MLP.  Accuracy depends on MediaPipe's
>   ability to detect the hand in the image.
> - **Single hand.**  Only the first detected hand is classified.
> - **Not a full sign-language interpreter.**  This covers individual
>   letters, not words, phrases, or non-manual markers.
> - **Background and lighting matter.**  Complex backgrounds or poor
>   lighting degrade hand detection.
> - **Dataset-dependent.**  Accuracy is bounded by the training
>   dataset's diversity (skin tones, angles, backgrounds).

---

## Features

| Capability | Module | Detail |
|------------|--------|--------|
| Hand detection | `hand_detector.py` | MediaPipe Hands (21 landmarks) |
| Feature extraction | `feature_extractor.py` | Wrist-normalised, scale-invariant (42-D) |
| Classification | `classifier.py` | sklearn MLP (128→64 hidden layers) |
| Training pipeline | `trainer.py` | End-to-end: download → extract → train → evaluate |
| Confusion matrix | `evaluator.py` | Per-class precision/recall/F1 + heatmap |
| Inference | `controller.py` | Detect → extract → classify → smooth |
| Webcam support | `infer.py` | Real-time recognition with overlays |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# 1. Train the classifier (auto-downloads dataset on first run)
cd "Sign Language Alphabet Recognizer/Source Code"
python train.py

# 2. Run webcam inference (press 'q' to quit)
python infer.py --source 0

# Single image
python infer.py --source hand_sign.jpg

# Video file
python infer.py --source recording.mp4
```

## Training

The training pipeline:
1. Downloads the ASL alphabet dataset (idempotent, auto on first run)
2. Runs MediaPipe Hands on each image to extract 21 landmarks
3. Converts landmarks to 42-D normalised feature vectors
4. Trains a 2-layer MLP classifier (sklearn)
5. Evaluates on a held-out test split
6. Saves model + evaluation report

```bash
# Full training with defaults
python train.py

# Limit images per class (faster experiments)
python train.py --max-images-per-class 200

# Custom model path
python train.py --model-out model/my_model.pkl

# Re-download dataset
python train.py --force-download
```

### Training CLI

```
python train.py [options]

  --force-download         Re-download dataset
  --model-out PATH         Model save path (default: model/sign_lang_clf.pkl)
  --test-size FLOAT        Test split ratio (default: 0.2)
  --max-iter INT           MLP max iterations (default: 500)
  --max-images-per-class N Limit images per class (0 = all)
```

## Inference CLI

```
python infer.py --source SOURCE [options]

Required:
  --source            '0' for webcam, or path to video/image

Optional:
  --model             Path to trained model pickle
  --config            YAML/JSON config path
  --no-smoothing      Disable majority-vote smoothing
  --no-display        Headless mode
  --export-csv        CSV export path
  --export-json       JSON export path
  --save-annotated    Save annotated frames
  --output-dir        Output directory (default: output/)
  --force-download    Re-download dataset
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset smoothing buffer |

## Evaluation

After training, the following are generated:

- **`eval_report.json`** — accuracy, per-class precision/recall/F1, confusion matrix
- **Console output** — test accuracy, class-level metrics

The evaluator supports:
- Per-class precision, recall, F1-score, and support
- Full confusion matrix (2-D array and optional heatmap image)
- Overall accuracy

## Configuration

```yaml
# MediaPipe
max_num_hands: 1
model_complexity: 1
min_detection_confidence: 0.6
min_tracking_confidence: 0.5

# Feature extraction
normalise_to_wrist: true
scale_invariant: true

# Classifier
model_path: model/sign_lang_clf.pkl

# Smoothing
enable_smoothing: true
vote_window: 5

# Display
show_landmarks: true
show_prediction: true
show_confidence: true
```

## How It Works

### Feature Extraction

Each hand's 21 landmarks are converted to a 42-dimensional vector:

1. Collect $(x, y)$ for all 21 landmarks
2. Translate so wrist = $(0, 0)$
3. Scale so $\max \| \mathbf{p}_i \| = 1$
4. Flatten to a 1-D vector of length 42

This makes features **translation-invariant** (wrist centring) and
**scale-invariant** (distance normalisation), so the same sign at
different positions and distances produces similar features.

### Classification

A 2-layer MLP with architecture $42 \to 128 \to 64 \to 24$ is
trained on the normalised feature vectors using sklearn's
`MLPClassifier` with early stopping.

### Smoothing

During inference, a **majority vote** over the last $N$ predictions
(default $N=5$) stabilises the output label, reducing frame-to-frame
flicker.

## Project Structure

```
Sign Language Alphabet Recognizer/
├── Source Code/
│   ├── config.py              # SignLangConfig dataclass
│   ├── hand_detector.py       # MediaPipe Hands wrapper
│   ├── feature_extractor.py   # Landmark → 42-D feature vector
│   ├── classifier.py          # sklearn MLP classifier
│   ├── trainer.py             # Training pipeline
│   ├── evaluator.py           # Confusion matrix + metrics
│   ├── controller.py          # Inference pipeline orchestrator
│   ├── validator.py           # Quality-check validator
│   ├── visualize.py           # Overlay renderer
│   ├── export.py              # Per-frame CSV/JSON export
│   ├── infer.py               # CLI inference entry point
│   ├── train.py               # CLI training entry point
│   ├── modern.py              # CVProject registry entry
│   └── data_bootstrap.py      # Idempotent dataset download
├── requirements.txt
└── README.md
```

## Supported Alphabet

```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```

**Excluded:** J (requires wrist motion), Z (requires drawing motion)

## Requirements

- Python 3.10+
- MediaPipe ≥ 0.10.14
- OpenCV ≥ 4.10
- NumPy ≥ 1.26
- scikit-learn ≥ 1.4
