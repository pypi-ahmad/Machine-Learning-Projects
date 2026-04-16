# Sign Language Alphabet Recognizer

Static ASL alphabet recognition from hand landmarks using MediaPipe Hand Landmarker and a lightweight sklearn MLP. The project supports training, evaluation, webcam inference, and artifact export for the 24 static fingerspelling letters:

`A B C D E F G H I K L M N O P Q R S T U V W X Y`

`J` and `Z` are excluded because they require motion.

## What This Project Does

- Detects a single hand with MediaPipe Hand Landmarker.
- Converts 21 landmarks into a compact engineered feature vector.
- Trains a lightweight classifier on static ASL alphabet poses.
- Evaluates with JSON metrics and a confusion-matrix heatmap.
- Runs live webcam, image, or video inference with overlays.
- Exports per-frame predictions to CSV or JSON.

## Scope and Limits

- This is a static alphabet recognizer, not a full sign-language interpreter.
- It predicts only the 24 static letters listed above.
- It assumes one visible hand per frame.
- Performance depends on successful landmark detection; difficult poses, occlusion, motion blur, or poor lighting can still cause skips.
- The bundled training flow uses a bounded subset of a public dataset so the first run stays practical.

## Dataset Bootstrap

The training pipeline auto-downloads a public ASL image dataset from `EricMartinezIllamola/asl-alphabet` on first run.

- Download is automatic and idempotent.
- `--force-download` rebuilds the local dataset subset from scratch.
- The bootstrap keeps a small but useful subset by default:
  - `30` training images per class
  - `10` test images per class
- Files are stored under `Computer Vision/data/sign_language_alphabet_recognizer`.

This project trains only on images where hand landmarks can actually be extracted, so the effective sample count can be smaller than the raw downloaded count.

## Quick Start

```bash
pip install -r requirements.txt

cd "Sign Language Alphabet Recognizer/Source Code"

# Train and evaluate (downloads dataset on first run)
python train.py

# Force a fresh dataset rebuild
python train.py --force-download

# Webcam inference
python infer.py --source 0

# Single image
python infer.py --source path/to/image.jpg

# Video file
python infer.py --source path/to/video.mp4
```

## Training Pipeline

`train.py` delegates to `trainer.py` and runs this flow end to end:

1. Ensure the public dataset subset is present locally.
2. Detect 21 hand landmarks for each image.
3. Convert landmarks to a scale-normalized feature vector.
4. Train a scaled MLP classifier.
5. Evaluate on the prepared test split.
6. Save the trained model and evaluation artifacts.

### Training CLI

```text
python train.py [options]

  --force-download         Re-download and rebuild the dataset subset
  --model-out PATH         Model save path (default: model/sign_lang_clf.pkl)
  --test-size FLOAT        Fallback split ratio if no prepared test split exists
  --max-iter INT           MLP max iterations (default: 500)
  --max-images-per-class N Limit images per class during extraction (0 = all prepared)
```

## Evaluation Outputs

Training writes the following artifacts into `Source Code/`:

- `model/sign_lang_clf.pkl` - trained classifier bundle
- `eval_report.json` - accuracy, per-class metrics, confusion matrix, and sample counts
- `confusion_matrix.png` - confusion-matrix heatmap image

The evaluation report includes:

- overall accuracy
- per-class precision, recall, F1, and support
- the full confusion matrix
- train/test sample counts
- skipped-image counts during landmark extraction

## Inference

`infer.py` supports webcam, still images, and video files.

### Inference CLI

```text
python infer.py --source SOURCE [options]

Required:
  --source            0 for webcam, or a path to an image/video file

Optional:
  --model             Path to trained model pickle
  --config            YAML/JSON config path
  --no-smoothing      Disable vote smoothing
  --no-display        Headless mode
  --export-csv        CSV export path
  --export-json       JSON export path
  --save-annotated    Save annotated frames
  --output-dir        Output directory for annotated frames
  --force-download    Rebuild the dataset subset before inference
```

### Keyboard Controls

- `q` quits
- `r` clears the smoothing buffer

## Model Design

### Hand Detection

`hand_detector.py` wraps MediaPipe Hand Landmarker and auto-downloads the `hand_landmarker.task` model on first use.

### Feature Extraction

`feature_extractor.py` converts landmarks into a 77-dimensional feature vector:

- `63` normalized landmark coordinates: `(x, y, z)` for all 21 points
- `5` fingertip-to-wrist distances
- `5` fingertip-to-joint distances
- `4` adjacent fingertip spread distances

This keeps the classifier lightweight while giving it more shape information than raw 2D coordinates alone.

### Classifier

`classifier.py` uses a `StandardScaler` followed by an `MLPClassifier` with hidden layers `(128, 64)` and early stopping.

### Smoothing

During inference, the controller applies majority-vote smoothing over the recent predictions to reduce frame-to-frame flicker.

## Configuration

Default values live in `config.py`. The main knobs are:

```yaml
max_num_hands: 1
min_detection_confidence: 0.6
min_presence_confidence: 0.5
min_tracking_confidence: 0.5
model_path: model/sign_lang_clf.pkl
enable_smoothing: true
vote_window: 5
show_landmarks: true
show_prediction: true
show_confidence: true
```

## Project Structure

```text
Sign Language Alphabet Recognizer/
├── Source Code/
│   ├── config.py
│   ├── hand_detector.py
│   ├── feature_extractor.py
│   ├── classifier.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── controller.py
│   ├── validator.py
│   ├── visualize.py
│   ├── export.py
│   ├── infer.py
│   ├── train.py
│   ├── modern.py
│   └── data_bootstrap.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- MediaPipe >= 0.10.14
- OpenCV >= 4.10
- NumPy >= 1.26
- scikit-learn >= 1.4
- matplotlib >= 3.8
