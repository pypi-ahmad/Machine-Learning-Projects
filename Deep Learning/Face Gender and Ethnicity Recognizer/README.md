# Project 14 -- Face / Gender / Ethnicity Recognizer

## Dataset
UTKFace

- https://www.kaggle.com/datasets/jangedoo/utkface-new
- https://www.kaggle.com/datasets/jessicali9530/fairface-dataset

## Stack
| Component | Choice |
|-----------|--------|
| Framework | PyTorch 2.10.0 (cu130) |
| Model     | convnext_tiny.fb_in22k_ft_in1k |
| Task      | Image Classification |
| AutoML    | N/A |

## Usage

```bash
# 1. Install PyTorch (once)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 2. Install dependencies (once)
pip install -r requirements.txt

# 3. Run this project
python run.py

# CLI options
python run.py --smoke-test        # Quick sanity check (1 epoch, 2 batches)
python run.py --download-only     # Just download the dataset
python run.py --epochs 5          # Override number of epochs
python run.py --batch-size 64     # Override batch size
python run.py --num-workers 8     # Override data-loader workers
python run.py --device cpu        # Force CPU (default: auto-detect)
python run.py --no-amp            # Disable mixed precision
```

## Outputs
After training, check `outputs/` for:
- `best_model.pth` (or `.pkl` for tabular) -- saved model weights
- `metrics.json` -- accuracy, F1, etc.
- `metrics.md` -- same metrics as a Markdown table
- `training_curves.png` -- loss / accuracy over epochs
- `confusion_matrix.png` -- per-class performance
- `classification_report.txt` -- detailed metrics
