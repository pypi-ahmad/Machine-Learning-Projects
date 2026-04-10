"""Search for modern parquet versions of legacy-script datasets."""
from datasets import load_dataset
import sys

tests = [
    # conll2003 alternatives
    ("conllpp", "train", None, "conll2003 alt"),
    ("tner/conll2003", "train", None, "conll2003 alt2"),
    # financial_phrasebank alternatives
    ("zeroshot/twitter-financial-news-sentiment", "train", None, "financial sentiment alt"),
    ("financial_phrasebank", "train", "sentences_50agree", "financial_phrasebank direct"),
    ("nickmuchi/financial-classification", "train", None, "financial clf alt"),
    # hate speech alternatives
    ("tweet_eval", "train", "hate", "hate speech alt"),
    ("Paul/hatecheck", "test", None, "hate check alt"),
    # daily dialog alternatives
    ("roskoN/dailydialog", "train", None, "daily dialog alt"),
    ("li2017dailydialog/daily_dialog", "train", None, "daily dialog alt2"),
    # NER alternatives
    ("Babelscape/multinerd", "train", None, "multinerd NER"),
    # restaurant reviews
    ("cornell-movie-review-data/rotten_tomatoes", "train", None, "rotten tomatoes"),
    # consumer complaints
    ("nikikilbertus/CFPB", "train", None, "consumer complaints alt"),
    # cyberbullying
    ("bnsapa/cyberbullying-classification", "train", None, "cyberbullying alt"),
    # Amazon reviews  
    ("McAuley-Lab/Amazon-Reviews-2023", "train", "0core_last_out_All_Beauty", "amazon reviews alt"),
    # movielens
    ("grouplens/movielens-latest-small", "train", None, "movielens alt"),
    # e-commerce
    ("Vargha/brazilian_ecommerce_dataset", "train", None, "ecommerce alt"),
    # loan
    ("utkarshx27/loan-prediction", "train", None, "loan alt"),
    # book crossing
    ("reem-alrashidi/BX-Books", "train", None, "book crossing alt"),
]

for name, split, config, desc in tests:
    try:
        kwargs = {"split": split}
        if config:
            kwargs["name"] = config
        ds = load_dataset(name, **kwargs)
        print(f"  OK  {name}{f' ({config})' if config else ''} [{desc}]: {len(ds)} rows, cols={ds.column_names[:5]}")
    except Exception as e:
        err = str(e)[:100]
        print(f"  BAD {name} [{desc}]: {err}")
    sys.stdout.flush()
