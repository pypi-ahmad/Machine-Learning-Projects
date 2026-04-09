#!/usr/bin/env python3
"""
Full pipeline for Garbage Classification

Auto-generated from: Garbage_Classification.ipynb
Project: Garbage Classification
Category: Classification | Task: image_classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
# Additional imports extracted from mixed cells
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data_dir  = './archive/Garbage classification/Garbage classification'

    classes = os.listdir(data_dir)
    print(classes)

    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms

    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    dataset = ImageFolder(data_dir, transform = transformations)

    import matplotlib.pyplot as plt

    def show_sample(img, label):
        print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
        plt.imshow(img.permute(1, 2, 0))

    img, label = dataset[12]
    show_sample(img, label)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Garbage Classification")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
