"""Quick test: try loading a sample of the 401 HF datasets."""
from datasets import load_dataset
import sys

test_names = [
    ("aai510-group1/telecom-churn-dataset", "train", None),
    ("Zaherrr/Weather-Dataset", "train", None),
    ("reczilla/movielens-100k", "train", None),
    ("Ammok/Household_Power_Consumption", "train", None),
    ("ErenalpCet/Loan-Prediction", "train", None),
    ("leostelon/KC-House-Data", "train", None),
    ("conll2003", "train", None),
    ("financial_phrasebank", "train", "sentences_50agree"),
    ("rotten_tomatoes", "train", None),
    ("wikitext", "train", "wikitext-2-raw-v1"),
    ("wmt16", "train[:10]", "de-en"),
    ("hate_speech18", "train", None),
    ("TrainingDataPro/email-spam-classification", "train", None),
    ("scikit-learn/restaurant-reviews", "train", None),
]

for name, split, config in test_names:
    try:
        kwargs = {"split": split}
        if config:
            kwargs["name"] = config
        ds = load_dataset(name, **kwargs, trust_remote_code=True)
        print(f"  OK  {name}: {len(ds)} rows, cols={ds.column_names[:5]}")
    except Exception as e:
        err = str(e)[:120]
        print(f"  BAD {name}: {err}")
    sys.stdout.flush()
