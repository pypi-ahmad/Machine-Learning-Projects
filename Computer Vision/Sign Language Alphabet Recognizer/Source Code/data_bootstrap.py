"""Sign Language Alphabet Recognizer -- idempotent dataset bootstrap.

Downloads a bounded public subset of a GitHub-hosted ASL alphabet dataset,
keeping the source train/test split so the project can train quickly on first run
without pulling the entire archive.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

from config import ASL_STATIC_LABELS

PROJECT_KEY = "sign_language_alphabet_recognizer"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / PROJECT_KEY

DATASET_OWNER = "EricMartinezIllamola"
DATASET_REPO = "asl-alphabet"
DATASET_REF = "main"
TRAIN_IMAGES_PER_CLASS = 30
TEST_IMAGES_PER_CLASS = 10

_USER_AGENT = "sign-language-alphabet-recognizer/1.0"
_RAW_ROOT = (
    f"https://raw.githubusercontent.com/{DATASET_OWNER}/{DATASET_REPO}/"
    f"{DATASET_REF}/asl-alphabet"
)


def ensure_sign_lang_dataset(force: bool = False) -> Path:
    """Download and prepare the public ASL alphabet subset."""
    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    processed = DATA_ROOT / "processed"
    ready_marker = processed / ".ready"

    if ready_marker.exists() and not force:
        return DATA_ROOT

    manifest: list[dict[str, str]] = []
    for label in ASL_STATIC_LABELS:
        for split, limit in (("train", TRAIN_IMAGES_PER_CLASS), ("test", TEST_IMAGES_PER_CLASS)):
            split_root = processed / split / label
            split_root.mkdir(parents=True, exist_ok=True)
            manifest.extend(_download_split(label, split, limit, split_root))

    _write_info(DATA_ROOT, processed, manifest)
    ready_marker.touch()
    return DATA_ROOT


def _download_split(
    label: str,
    split: str,
    limit: int,
    split_root: Path,
) -> list[dict[str, str]]:
    manifest_entries: list[dict[str, str]] = []
    downloaded = 0
    candidate_indices = _candidate_indices(split)

    for index in candidate_indices:
        file_name = _build_file_name(label, split, index)
        url = f"{_RAW_ROOT}/{label}/{file_name}"
        dst = split_root / file_name
        try:
            if not dst.exists():
                _download_file(url, dst)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise

        manifest_entries.append(
            {
                "split": split,
                "label": label,
                "file_name": file_name,
                "download_url": url,
                "local_path": str(dst),
            }
        )
        downloaded += 1

        if downloaded >= limit:
            break

    if downloaded < limit:
        raise RuntimeError(
            f"Only found {downloaded} files for {label}/{split}; expected {limit}."
        )

    return manifest_entries


def _build_file_name(label: str, split: str, index: int) -> str:
    if split == "test":
        return f"{label}{index:04d}_test.jpg"
    return f"{label}{index}.jpg"


def _candidate_indices(split: str) -> list[int]:
    if split == "test":
        return list(range(1, 101))

    primary = list(range(1, 1501, 50))
    primary.extend(range(2, 1501, 50))
    primary.extend(range(3, 1501, 50))
    fallback = list(range(1, 3001))

    seen: set[int] = set()
    ordered: list[int] = []
    for index in primary + fallback:
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def _download_file(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(destination)


def _write_info(data_root: Path, processed: Path, manifest: list[dict[str, str]]) -> None:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        counts[split] = {}
        split_root = processed / split
        if not split_root.exists():
            continue
        for label_dir in sorted(split_root.iterdir()):
            if label_dir.is_dir():
                counts[split][label_dir.name] = len(list(label_dir.iterdir()))

    info = {
        "project": PROJECT_KEY,
        "source": {
            "type": "github_raw_subset",
            "repo": f"{DATASET_OWNER}/{DATASET_REPO}",
            "ref": DATASET_REF,
            "dataset_path": "asl-alphabet/{label}",
            "split_rule": "train uses <label><n>.jpg; test uses <label><000n>_test.jpg",
        },
        "supported_labels": list(ASL_STATIC_LABELS),
        "downloaded_per_class": {
            "train": TRAIN_IMAGES_PER_CLASS,
            "test": TEST_IMAGES_PER_CLASS,
        },
        "per_split_counts": counts,
        "total_images": len(manifest),
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    manifest_path = processed / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys

    force = "--force-download" in sys.argv
    path = ensure_sign_lang_dataset(force=force)
    print(f"Dataset ready at: {path}")
