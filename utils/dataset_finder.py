"""Dataset discovery, official-link resolution, and auto-download for all 21 NLP projects.

Core function: prepare_project_data(project_dir, project_slug)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_WORKSPACE = Path(__file__).resolve().parent.parent
_DATA_ROOT = _WORKSPACE / "data"

# ======================================================================
# PROJECT REGISTRY — single source of truth for all 21 projects
# ======================================================================

PROJECT_REGISTRY: dict[str, dict] = {
    # ---- 1. Text Processing & Analysis ------------------------------------
    "e-commerce-clothing-reviews": {
        "dir": "1. Text Processing and Analysis/1. Text Processing and Analysis/E-Commerce Clothing Reviews",
        "task": "analysis",
        "data_files": ["data.csv"],
        "text_col": "Review Text",
        "target_col": None,
        "kaggle_slug": "nicapotato/womens-ecommerce-clothing-reviews",
        "official_urls": [
            "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews",
        ],
        "official_source": "Kaggle (nicapotato) — anonymized real commercial data",
        "license": "CC0: Public Domain",
    },
    "trip-advisor-hotel-reviews": {
        "dir": "1. Text Processing and Analysis/1. Text Processing and Analysis/Trip Advisor Hotel Reviews",
        "task": "analysis",
        "data_files": ["data.csv"],
        "text_col": "Review",
        "target_col": "Rating",
        "kaggle_slug": "thedevastator/tripadvisor-hotel-reviews",
        "official_urls": [
            "https://www.kaggle.com/datasets/thedevastator/tripadvisor-hotel-reviews",
        ],
        "official_source": "TripAdvisor (scraped reviews)",
        "license": "Open Data Commons",
    },
    "world-war-i-letters": {
        "dir": "1. Text Processing and Analysis/1. Text Processing and Analysis/World War I Letters",
        "task": "analysis",
        "data_files": ["data.csv", "letters.json"],
        "text_col": None,
        "target_col": None,
        "kaggle_slug": "anthaus/world-war-i-letters",
        "official_urls": [
            "https://www.kaggle.com/datasets/anthaus/world-war-i-letters",
        ],
        "official_source": "Anthaus collection — historical WWI correspondence",
        "license": "CC BY 4.0",
    },
    # ---- 2. Text Classification -------------------------------------------
    "cyberbullying-classification": {
        "dir": "2. Text Classification/2. Text Classification/Cyberbullying Classification",
        "task": "classification",
        "data_files": ["data.csv"],
        "text_col": "tweet_text",
        "target_col": "cyberbullying_type",
        "kaggle_slug": "andrewmvd/cyberbullying-classification",
        "official_urls": [
            "https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification",
        ],
        "official_source": 'J. Wang et al., "Multimodal Cyberbullying Detection"',
        "license": "CC BY 4.0",
    },
    "e-commerce-product-classification": {
        "dir": "2. Text Classification/2. Text Classification/E-commerce Product Classification",
        "task": "classification",
        "data_files": ["data.csv"],
        "text_col": "__col1__",
        "target_col": "__col0__",
        "note": "CSV has no header; col0=category, col1=description",
        "kaggle_slug": "sumedhdataaspirant/e-commerce-text-dataset",
        "official_urls": [
            "https://www.kaggle.com/datasets/sumedhdataaspirant/e-commerce-text-dataset",
        ],
        "official_source": "Kaggle (sumedhdataaspirant)",
        "license": "Unknown",
    },
    "economic-news-articles": {
        "dir": "2. Text Classification/2. Text Classification/Economic News Articles",
        "task": "classification",
        "data_files": ["data.csv"],
        "text_col": "text",
        "target_col": "positivity",
        "kaggle_slug": "adhamelkomy/news-classification-and-analysis-using-nlp",
        "official_urls": [
            "https://www.kaggle.com/datasets/adhamelkomy/news-classification-and-analysis-using-nlp",
            "https://www.figure-eight.com/",
        ],
        "official_source": "Figure Eight / CrowdFlower — economic news sentiment (crowd-labeled)",
        "license": "Open (crowdsourced annotations)",
    },
    "fake-news-detection": {
        "dir": "2. Text Classification/2. Text Classification/Fake News Detection",
        "task": "classification",
        "data_files": ["True.csv", "Fake.csv"],
        "text_col": "text",
        "target_col": "label",
        "note": "True.csv and Fake.csv must be merged with a label column (1=True, 0=Fake)",
        "kaggle_slug": "vishakhdapat/fake-news-detection",
        "official_urls": [
            "https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection",
            "https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php",
        ],
        "official_source": "ISOT Fake News Dataset — University of Victoria",
        "license": "Academic use",
    },
    "news-headline-classification": {
        "dir": "2. Text Classification/2. Text Classification/News Headline Classification",
        "task": "classification",
        "data_files": ["data.json"],
        "text_col": "headline",
        "target_col": "category",
        "kaggle_slug": "rmisra/news-category-dataset",
        "official_urls": [
            "https://www.kaggle.com/datasets/rmisra/news-category-dataset",
            "https://rishabhmisra.github.io/publications/",
        ],
        "official_source": "Rishabh Misra — HuffPost news category dataset (2012-2022)",
        "license": "CC BY 4.0",
    },
    "paper-subject-prediction": {
        "dir": "2. Text Classification/2. Text Classification/Paper Subject Prediction",
        "task": "classification",
        "data_files": ["arxiv_data.csv"],
        "text_col": "summaries",
        "target_col": "terms",
        "kaggle_slug": "sumitm004/arxiv-scientific-research-papers-dataset",
        "official_urls": [
            "https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset",
            "https://arxiv.org/",
        ],
        "official_source": "arXiv.org (metadata harvested via OAI-PMH API)",
        "license": "arXiv Terms of Use",
    },
    "review-classification": {
        "dir": "2. Text Classification/2. Text Classification/Review Classification",
        "task": "classification",
        "data_files": ["amazon.txt", "imdb.txt", "yelp.txt"],
        "text_col": "text",
        "target_col": "label",
        "note": "Tab-separated files: text<TAB>label (0/1). Three sources.",
        "kaggle_slug": "ndrianahani/imdb-yelp-and-amazon-reviews",
        "official_urls": [
            "https://www.kaggle.com/datasets/ndrianahani/imdb-yelp-and-amazon-reviews",
        ],
        "official_source": "Aggregated from Amazon, IMDb, and Yelp reviews",
        "license": "Various (Amazon ToS, IMDb ToS, Yelp ToS)",
    },
    "spam-message-detection": {
        "dir": "2. Text Classification/2. Text Classification/Spam Message Detection",
        "task": "classification",
        "data_files": ["data.csv"],
        "text_col": "text",
        "target_col": "label",
        "kaggle_slug": "uciml/sms-spam-collection-dataset",
        "official_urls": [
            "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
            "https://archive.ics.uci.edu/ml/datasets/sms+spam+collection",
        ],
        "official_source": "UCI Machine Learning Repository — SMS Spam Collection v.1",
        "license": "CC BY 4.0 (UCI)",
    },
    "toxic-comment-classification": {
        "dir": "2. Text Classification/2. Text Classification/Toxic Comment Classification",
        "task": "classification_multilabel",
        "data_files": ["train.csv", "test.csv"],
        "text_col": "comment_text",
        "target_col": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "kaggle_slug": "jigsaw-toxic-comment-classification-challenge",
        "kaggle_type": "competition",
        "official_urls": [
            "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge",
            "https://www.perspectiveapi.com/research/",
        ],
        "official_source": "Google Jigsaw / Conversation AI — Wikipedia Talk page comments",
        "license": "Competition rules (Kaggle)",
    },
    "twitter-sentiment-analysis": {
        "dir": "2. Text Classification/2. Text Classification/Twitter Sentiment Analysis",
        "task": "classification",
        "data_files": ["train.csv", "test.csv"],
        "text_col": "OriginalTweet",
        "target_col": "Sentiment",
        "kaggle_slug": "milobele/sentiment140-dataset-1600000-tweets",
        "official_urls": [
            "https://www.kaggle.com/datasets/milobele/sentiment140-dataset-1600000-tweets",
            "http://help.sentiment140.com/for-students/",
        ],
        "official_source": "Sentiment140 — Stanford (Go, Bhayani & Huang, 2009)",
        "license": "Research use",
    },
    # ---- 3. Text Generation -----------------------------------------------
    "automated-image-captioning": {
        "dir": "3. Text Generation/3. Text Generation/Automated Image Captioning",
        "task": "image_captioning",
        "data_files": ["captions.txt"],
        "data_dirs": ["images"],
        "text_col": "caption",
        "target_col": None,
        "kaggle_slug": "adityajn105/flickr8k",
        "official_urls": [
            "https://www.kaggle.com/datasets/adityajn105/flickr8k",
            "https://forms.illinois.edu/sec/1713398",
        ],
        "official_source": "Flickr8k — University of Illinois (Hodosh, Young & Hockenmaier, 2013)",
        "license": "Research use (Flickr terms)",
    },
    "bbc-articles-summarization": {
        "dir": "3. Text Generation/3. Text Generation/BBC Articles Summarization",
        "task": "summarization",
        "data_files": [],
        "text_col": "article",
        "target_col": "summary",
        "note": "Dataset NOT bundled; must download from Kaggle",
        "kaggle_slug": "pariza/bbc-news-summary",
        "official_urls": [
            "https://www.kaggle.com/datasets/pariza/bbc-news-summary",
            "http://mlg.ucd.ie/datasets/bbc.html",
        ],
        "official_source": "BBC News (D. Greene & P. Cunningham, 2006) — UCD ML Group",
        "license": "Academic research only",
    },
    "english-to-french-translation": {
        "dir": "3. Text Generation/3. Text Generation/English to French Translation",
        "task": "translation",
        "data_files": ["eng-fra.txt"],
        "text_col": "english",
        "target_col": "french",
        "kaggle_slug": "dhruvildave/en-fr-translation-dataset",
        "official_urls": [
            "https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset",
            "https://www.manythings.org/anki/",
        ],
        "official_source": "Tatoeba Project / ManyThings.org (sentence pairs)",
        "license": "CC BY 2.0 (Tatoeba)",
    },
    "name-generate-from-languages": {
        "dir": "3. Text Generation/3. Text Generation/Name Generate From Languages",
        "task": "generation",
        "data_files": [],
        "data_dirs": ["names"],
        "text_col": "name",
        "target_col": "language",
        "note": "Local names/*.txt files bundled in repo",
        "kaggle_slug": "davidam9/international-names",
        "official_urls": [
            "https://www.kaggle.com/datasets/davidam9/international-names",
            "https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html",
        ],
        "official_source": "PyTorch tutorials (char-level RNN) — names from various languages",
        "license": "Public Domain",
    },
    # ---- 4. Text Clustering & Topic Modelling -----------------------------
    "kaggle-survey-questions-clustering": {
        "dir": "4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Kaggle Survey Questions Clustering",
        "task": "clustering",
        "data_files": ["questions.csv"],
        "text_col": None,
        "target_col": None,
        "kaggle_slug": "kaggle-survey-2021",
        "kaggle_type": "competition",
        "official_urls": [
            "https://www.kaggle.com/competitions/kaggle-survey-2021",
        ],
        "official_source": "Kaggle — 2021 ML & Data Science Survey (official competition)",
        "license": "Competition rules (Kaggle)",
    },
    "medium-articles-clustering": {
        "dir": "4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Medium Articles Clustering",
        "task": "clustering",
        "data_files": ["articles.csv"],
        "text_col": "text",
        "target_col": None,
        "kaggle_slug": "arnabchaki/medium-articles-dataset",
        "official_urls": [
            "https://www.kaggle.com/datasets/arnabchaki/medium-articles-dataset",
        ],
        "official_source": "Medium.com (scraped articles)",
        "license": "Unknown (scraped data)",
    },
    "newsgroups-posts-clustering": {
        "dir": "4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Newsgroups Posts Clustering",
        "task": "clustering",
        "data_files": [],
        "text_col": "text",
        "target_col": "newsgroup",
        "note": "Uses sklearn.datasets.fetch_20newsgroups; data not bundled",
        "kaggle_slug": "crawford/20-newsgroups",
        "official_urls": [
            "https://www.kaggle.com/datasets/crawford/20-newsgroups",
            "http://qwone.com/~jason/20Newsgroups/",
        ],
        "official_source": "20 Newsgroups (Ken Lang, 1995) — CMU",
        "license": "Public Domain",
    },
    "stories-clustering": {
        "dir": "4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Stories Clustering",
        "task": "clustering",
        "data_files": [],
        "text_col": "story",
        "target_col": None,
        "note": "Dataset NOT bundled; must download from Kaggle",
        "kaggle_slug": "fareedkhan557/1000-stories-100-genres",
        "official_urls": [
            "https://www.kaggle.com/datasets/fareedkhan557/1000-stories-100-genres",
        ],
        "official_source": "1000 Stories 100 Genres (Kaggle contributor)",
        "license": "CC0: Public Domain",
    },
}


# ======================================================================
# Scan project for dataset references
# ======================================================================

def scan_project_for_references(project_dir: Path) -> dict:
    """Scan guideline.txt, markdown cells, PDFs for dataset hints."""
    refs: dict = {"urls": [], "file_names": [], "dataset_names": []}
    import re

    url_re = re.compile(r"https?://[^\s\)\]\"']+")

    # Scan text files
    for pattern in ["guideline.txt", "*.txt", "*.md", "README*"]:
        for fp in project_dir.glob(pattern):
            if fp.suffix in (".csv", ".json", ".tsv"):
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
                refs["urls"].extend(url_re.findall(text))
            except Exception:
                pass

    # Scan notebook markdown cells
    nb_path = project_dir / "code.ipynb"
    if nb_path.exists():
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "markdown":
                    text = "".join(cell.get("source", []))
                    refs["urls"].extend(url_re.findall(text))
        except Exception:
            pass

    # Deduplicate
    refs["urls"] = list(set(refs["urls"]))

    # Note data files present
    data_exts = {".csv", ".json", ".jsonl", ".txt", ".tsv"}
    for fp in project_dir.iterdir():
        if fp.is_file() and fp.suffix in data_exts and fp.name not in ("guideline.txt", "requirements.txt"):
            refs["file_names"].append(fp.name)
    for fp in project_dir.iterdir():
        if fp.is_dir():
            refs["file_names"].append(f"{fp.name}/")

    return refs


# ======================================================================
# Data preparation (download / locate / copy)
# ======================================================================

def _copy_local_files(src_dir: Path, dest_dir: Path, file_list: list[str], dir_list: list[str] | None = None) -> list[str]:
    """Copy data files from project dir to standardized data dir."""
    copied = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in file_list:
        src = src_dir / fname
        dst = dest_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(fname)
        elif dst.exists():
            copied.append(fname)
    if dir_list:
        for dname in dir_list:
            src = src_dir / dname
            dst = dest_dir / dname
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
                copied.append(f"{dname}/")
            elif dst.exists():
                copied.append(f"{dname}/")
    return copied


def _try_kaggle_download(slug: str, dest: Path, dtype: str = "dataset") -> bool:
    """Attempt to download from Kaggle. Returns True on success."""
    try:
        from utils.dataset_downloader import download_dataset

        download_dataset(slug, dest=dest)
        return True
    except Exception as exc:
        logger.warning("Kaggle download failed for '%s': %s", slug, exc)
        return False


def prepare_project_data(
    project_dir: str | Path,
    project_slug: str,
) -> dict:
    """Prepare dataset for a project: locate files, copy or download.

    Returns a structured dict with: status, raw_paths, urls_found,
    official_urls, chosen_source, notes.
    """
    project_dir = Path(project_dir)
    info = PROJECT_REGISTRY.get(project_slug)
    if not info:
        return {
            "status": "UNKNOWN_PROJECT",
            "project_slug": project_slug,
            "raw_paths": [],
            "notes": f"Project '{project_slug}' not in registry.",
        }

    raw_dir = _DATA_ROOT / project_slug / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "PENDING",
        "project_slug": project_slug,
        "task": info["task"],
        "raw_dir": str(raw_dir),
        "raw_paths": [],
        "urls_found": [],
        "official_urls": info.get("official_urls", []),
        "chosen_source": "unknown",
        "notes": info.get("note", ""),
    }

    # Scan for references
    refs = scan_project_for_references(project_dir)
    result["urls_found"] = refs["urls"]

    expected_files = info.get("data_files", [])
    expected_dirs = info.get("data_dirs", [])

    # 1) Check if raw_dir already has content
    existing = [f.name for f in raw_dir.iterdir() if f.name != ".gitkeep"]
    if existing:
        result["status"] = "READY"
        result["raw_paths"] = existing
        result["chosen_source"] = "data_dir (previously prepared)"
        return result

    # 2) Check if local files exist in project directory
    local_present = []
    for fname in expected_files:
        if (project_dir / fname).exists():
            local_present.append(fname)
    local_dirs = []
    for dname in expected_dirs:
        if (project_dir / dname).is_dir():
            local_dirs.append(dname)

    if local_present or local_dirs:
        copied = _copy_local_files(project_dir, raw_dir, local_present, local_dirs)
        result["status"] = "READY"
        result["raw_paths"] = copied
        result["chosen_source"] = "local (copied from project dir)"
        return result

    # 3) Try Kaggle download (before special-case fetchers)
    kaggle_slug = info.get("kaggle_slug", "")
    if kaggle_slug:
        ok = _try_kaggle_download(project_slug, raw_dir)
        if ok:
            downloaded = [f.name for f in raw_dir.rglob("*") if f.is_file() and f.name != ".gitkeep"]
            result["status"] = "READY"
            result["raw_paths"] = downloaded
            result["chosen_source"] = f"kaggle ({kaggle_slug})"
            return result

    # 4) Special case: 20 newsgroups (sklearn built-in fallback)
    if project_slug == "newsgroups-posts-clustering":
        try:
            from sklearn.datasets import fetch_20newsgroups
            import pandas as pd

            data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
            df = pd.DataFrame({"text": data.data, "newsgroup": [data.target_names[t] for t in data.target]})
            df.to_csv(raw_dir / "newsgroups.csv", index=False)
            result["status"] = "READY"
            result["raw_paths"] = ["newsgroups.csv"]
            result["chosen_source"] = "sklearn.datasets.fetch_20newsgroups"
            return result
        except Exception as exc:
            result["notes"] += f" sklearn fetch failed: {exc}"

    # 5) Fallback: MANUAL_REQUIRED
    result["status"] = "MANUAL_REQUIRED"
    result["notes"] += (
        f" Data not found locally and auto-download failed. "
        f"Download manually from: {info.get('official_urls', [])}"
    )
    return result


# ======================================================================
# Batch operations
# ======================================================================

def prepare_all_projects() -> dict[str, dict]:
    """Run prepare_project_data for all 21 projects."""
    results = {}
    for slug, info in sorted(PROJECT_REGISTRY.items()):
        project_dir = _WORKSPACE / info["dir"]
        logger.info("Preparing: %s", slug)
        try:
            result = prepare_project_data(project_dir, slug)
            results[slug] = result
        except Exception as exc:
            logger.error("Failed: %s — %s", slug, exc)
            results[slug] = {"status": "ERROR", "error": str(exc)}
    return results


def generate_dataset_map(results: dict[str, dict]) -> Path:
    """Save /reports/dataset_map.json."""
    out_path = _WORKSPACE / "reports" / "dataset_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Enrich with project_path
    enriched = {}
    for slug, res in results.items():
        info = PROJECT_REGISTRY.get(slug, {})
        enriched[slug] = {
            "project_slug": slug,
            "project_path": info.get("dir", ""),
            **res,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved dataset_map.json -> %s", out_path)
    return out_path


def generate_dataset_links_report() -> Path:
    """Generate /reports/dataset_links.md with official links for every project."""
    lines = [
        "# Dataset Links — Official & Mirror Sources",
        "",
        "Auto-generated by Phase 1 dataset discovery.",
        "",
        "---",
        "",
    ]
    for slug, info in sorted(PROJECT_REGISTRY.items()):
        lines.append(f"## {slug}")
        lines.append("")
        lines.append(f"**Task type:** {info['task']}")
        lines.append(f"**Official source:** {info.get('official_source', 'Unknown')}")
        lines.append(f"**License:** {info.get('license', 'Unknown')}")
        lines.append("")
        lines.append("**Links:**")
        for url in info.get("official_urls", []):
            lines.append(f"- {url}")
        kaggle_slug = info.get("kaggle_slug", "")
        if kaggle_slug:
            ktype = info.get("kaggle_type", "dataset")
            if ktype == "competition":
                lines.append(f"- Kaggle competition: https://www.kaggle.com/competitions/{kaggle_slug}")
            else:
                lines.append(f"- Kaggle mirror: https://www.kaggle.com/datasets/{kaggle_slug}")
        if info.get("note"):
            lines.append(f"**Note:** {info['note']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = _WORKSPACE / "reports" / "dataset_links.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved dataset_links.md -> %s", out_path)
    return out_path
