#!/usr/bin/env python3
"""
Update Dataset Link Files
=========================
Writes the canonical dataset URL(s) into each project's link file.

Usage:
    python scripts/update_dataset_links.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── (folder_name, existing_link_filename | None, content) ────────────────────
# If existing_link_filename is None, "link_to_dataset.txt" will be created.
LINKS: list[tuple[str, str | None, str]] = [
    (
        "Deep Learning Projects 1 - Pnemonia Detection",
        "Link to Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n",
    ),
    (
        "Deep Learning Projects 2 - Face Mask Detection",
        "link_to_dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset\n",
    ),
    (
        "Deep Learning Projects 3 - Earthquack Prediction model",
        "link_to_dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/henryshan/earthquake-prediction\n",
    ),
    (
        "Deep Learning Projects 4 - Landmark Detection Model",
        "link_to_dataset.txt",
        (
            "Kaggle (Landmarks): https://www.kaggle.com/datasets/google/google-landmarks-dataset\n"
            "Official (GLDv2 info): https://github.com/cvdfoundation/google-landmark\n"
        ),
    ),
    (
        "Deep Learning Projects 5 - Chatbot With Deep Learning",
        None,
        (
            "Kaggle (intents.json alternatives): https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset\n"
            "Note: If using custom intents.json, no external dataset needed.\n"
        ),
    ),
    (
        "Deep Learning Projects 6 - Movies Title Prediction",
        None,
        "Kaggle: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots\n",
    ),
    (
        "Deep Learning Projects 7 - Advanced Churn Modeling",
        "link_to_dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling\n",
    ),
    (
        "Deep Learning Projects 8 - Disease Prediction Model",
        None,
        "Kaggle: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning\n",
    ),
    (
        "Deep Learning Projects 9 - IMDB Sentiment Analysis using Deep Learning",
        "Link to the Dataset.txt",
        (
            "Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews\n"
            "Official (Large Movie Review Dataset): https://ai.stanford.edu/~amaas/data/sentiment/\n"
        ),
    ),
    (
        "Deep Learning Projects 10 - Advanced rsnet50",
        None,
        "Kaggle (competition): https://www.kaggle.com/c/plant-pathology-2021-fgvc8\n",
    ),
    (
        "Deep Learning Projects 11 - Cat Vs Dog",
        "Link to the Dataset.txt",
        "Kaggle (classic): https://www.kaggle.com/c/dogs-vs-cats\n",
    ),
    (
        "Deep Learning Projects 12 - Keep Babies Safe",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/c/state-farm-distracted-driver-detection\n",
    ),
    (
        "Deep Learning Projects 13 - Covid 19 Drug Recovery using Deep Learning",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/roche-data-science-coalition/uncover\n",
    ),
    (
        "Deep Learning Projects 14 - Face, Gender & Ethincity recognizer model",
        "Link to the Dataset.txt",
        (
            "Kaggle (FairFace): https://www.kaggle.com/datasets/jessicali9530/fairface-dataset\n"
            "Kaggle (UTKFace alternative): https://www.kaggle.com/datasets/jangedoo/utkface-new\n"
        ),
    ),
    (
        "Deep Learning Projects 15 - Happy house Predictor model",
        "link_to_dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/uciml/boston-housing-dataset\n",
    ),
    (
        "Deep Learning Projects 16 - Brain MRI Segmentation modling",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation\n",
    ),
    (
        "Deep Learning Projects 17 - Parkension Post Estimation using deep learning",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set\n",
    ),
    (
        "Deep Learning Projects 18 - Diabetic Retinopathy project",
        "Link to the Dataset.txt",
        "Kaggle (competition): https://www.kaggle.com/c/diabetic-retinopathy-detection\n",
    ),
    (
        "Deep Learning Projects 19 - Arabic character recognization using deep learning",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/mloey1/ahcd1\n",
    ),
    (
        "Deep Learning Projects 20 - Brain Tumor Recognization using Deep Learning",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection\n",
    ),
    (
        "Deep Learning Projects 21 - Image Walking or Running",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset\n",
    ),
    (
        "Deep Learning Projects 22- 1957 All Space Missions",
        "link_to_dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957\n",
    ),
    (
        "Deep Learning Projects 23 - 1 Million Suduku Solver using neural nets",
        None,
        "Kaggle: https://www.kaggle.com/datasets/bryanpark/sudoku\n",
    ),
    (
        "Deep Learning Projects 24 -Electric Car Temperature Predictor using Deep Learning",
        "link_do_dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature\n",
    ),
    (
        "Deep Learning Projects 25-Hourly energy demand generation and weather",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption\n",
    ),
    (
        "Deep Learning Projects 26 - Caffe Face Detector (OpenCV Pre-trained Model)",
        "Link to the Dataset.txt",
        "Official (OpenCV model zoo): https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector\n",
    ),
    (
        "Deep Learning Projects 27- Calculate Concrete Strength",
        None,
        "Kaggle: https://www.kaggle.com/datasets/uciml/concrete-compressive-strength-data-set\n",
    ),
    (
        "Deep Learning Projects 28 - Stock Market Prediction",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/aaron7sun/stocknews\n",
    ),
    (
        "Deep Learning Projects 29 - Indian Startup data Analysis",
        None,
        "Kaggle: https://www.kaggle.com/datasets/ruchi798/startup-investments-crunchbase\n",
    ),
    (
        "Deep Learning Projects 30 - Amazon Stock Price Deep Analysis",
        None,
        "Kaggle: https://www.kaggle.com/datasets/rohanrao/amazon-stock-price\n",
    ),
    (
        "Deep Learning Projects 31 - Indentifying Dance Form Using Deep Learning-20210724T041140Z-001",
        None,
        "Kaggle: https://www.kaggle.com/datasets/arjunbhasin2013/indian-classical-dance\n",
    ),
    (
        "Deep Learning Projects 32 - Glass or No Glass Detector Model using DL",
        None,
        "Kaggle: https://www.kaggle.com/datasets/jehanbhathena/eyeglasses-dataset\n",
    ),
    (
        "Deep Learning Projects 33 - Fingerprint Recognizer Model using DL",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/ruizgara/socofing\n",
    ),
    (
        "Deep Learning Projects 34 - World Currency Coin Detector Model using DL",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/wanderdust/coin-images\n",
    ),
    (
        "Deep Learning Projects 35 - News Category Prediction using DL",
        None,
        "Kaggle: https://www.kaggle.com/datasets/rmisra/news-category-dataset\n",
    ),
    (
        "Deep Learning Projects 36 - Lego Brick Code Problem",
        None,
        "Kaggle: https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images\n",
    ),
    (
        "Deep Learning Projects 37 - Sheep Breed Classification using CNN DL",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/warcoder/sheep-face-images\n",
    ),
    (
        "Deep Learning Projects 38 - Campus Recruitment Success rate analysis",
        None,
        "Kaggle: https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement\n",
    ),
    (
        "Deep Learning Projects 39 - Bank Marketing",
        None,
        "Kaggle: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing\n",
    ),
    (
        "Deep Learning Projects 40 - Pokemon Generation Clustering",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/abcsds/pokemon\n",
    ),
    (
        "Deep Learning Projects 41 - Cat _ Dog Voice Recognizer Model",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs\n",
    ),
    (
        "Deep Learning Projects 42 - Bottle or Cans Classifier using DL",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/trolukovich/bottles-and-cans\n",
    ),
    (
        "Deep Learning Projects 43 - Skin Cancer Recognizer using DL",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000\n",
    ),
    (
        "Deep Learning Projects 44 - Image Colorization using Deep Learning",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/landrykezebou/vizwiz-colorization\n",
    ),
    (
        "Deep Learning Projects 45 - Amazon Alexa Review Sentiment Analysis",
        None,
        "Kaggle: https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews\n",
    ),
    (
        "Deep Learning Projects 46 - Build_ChatBot_using_Neural_Network",
        None,
        "Kaggle: https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset\n",
    ),
    (
        "Deep Learning Projects 47 - Cactus or Not Cactus Ariel Image Recognizer",
        None,
        "Kaggle (competition): https://www.kaggle.com/c/aerial-cactus-identification\n",
    ),
    (
        "Deep Learning Projects 48 -  Build_Clothing_Prediction_Flask_Web_App",
        None,
        "Official (Fashion-MNIST): https://github.com/zalandoresearch/fashion-mnist\n",
    ),
    (
        "Deep Learning Projects 49 - Build_Sentiment_Analysis_Flask_Web_App",
        None,
        (
            "Official (IMDB): https://ai.stanford.edu/~amaas/data/sentiment/\n"
            "Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews\n"
        ),
    ),
    (
        "Deep Learning Projects 50 - COVID-19 Lung CT Scans",
        "Link to the Dataset.txt",
        "Kaggle: https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset\n",
    ),
]

DEFAULT_FILENAME = "link_to_dataset.txt"


def main() -> None:
    updated = 0
    created = 0
    errors = 0

    for folder, existing_file, content in LINKS:
        project_dir = ROOT / folder
        if not project_dir.is_dir():
            print(f"  [WARN] Folder not found: {folder}")
            errors += 1
            continue

        filename = existing_file if existing_file else DEFAULT_FILENAME
        filepath = project_dir / filename

        action = "Updated" if filepath.exists() else "Created"
        try:
            filepath.write_text(content, encoding="utf-8")
            if action == "Updated":
                updated += 1
            else:
                created += 1
            print(f"  [{action:7s}] {folder}/{filename}")
        except Exception as e:
            print(f"  [ERROR] {folder}/{filename}: {e}")
            errors += 1

    print(f"\nDone: {updated} updated, {created} created, {errors} errors.")


if __name__ == "__main__":
    main()
