"""Dataset registry & resolver — deterministic download, caching, validation.

Supports five source types:

- **huggingface** – loads via ``datasets.load_dataset()`` or streaming; preferred
- **url**         – direct HTTP(S) download + auto-extract (``.zip`` / ``.tar.gz``)
- **roboflow**    – public Roboflow Universe datasets with YOLO export
- **kaggle**      – downloads via ``kaggle datasets download`` or ``kaggle competitions download``
- **manual**      – datasets that require browser login / Google-Drive consent;
                    prints clear instructions and raises on missing data

Source preference order:
  1. Hugging Face Datasets (programmatic, supports streaming)
  2. Direct public ZIP / TAR URLs
  3. Roboflow public exports (labeled CV datasets)
  4. Kaggle / kagglehub (requires credentials)

Every successful download is stamped with a ``.ready`` marker and
``dataset_info.json`` metadata so subsequent calls skip the network entirely.

Usage::

    from utils.datasets import DatasetResolver

    resolver = DatasetResolver()

    # resolve (download if needed) → returns Path
    path = resolver.resolve("emotion_recognition")

    # list everything
    resolver.list_datasets()

    # force re-download
    path = resolver.resolve("emotion_recognition", force=True)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.paths import REPO_ROOT, DATA_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("datasets")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# YAML loader (use PyYAML if available, else minimal fallback)
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    """Load a YAML file, preferring PyYAML if available."""
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        # Minimal fallback — supports flat key: value only
        result: dict = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                v = v.strip().strip('"').strip("'")
                if v.lower() == "true":
                    result[k.strip()] = True
                elif v.lower() == "false":
                    result[k.strip()] = False
                else:
                    result[k.strip()] = v
        return result


# ---------------------------------------------------------------------------
# Config directory
# ---------------------------------------------------------------------------
_CONFIGS_DIR = REPO_ROOT / "configs" / "datasets"

# ---------------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------------
# Each entry must have:
#   type        – "kaggle", "kaggle_competition", "url", or "manual"
#   target_dir  – relative to REPO_ROOT (always under data/)
#
# Optional:
#   id          – kaggle dataset/competition slug
#   url         – direct download link
#   description – human-readable note
#   instructions – message shown for "manual" datasets
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Kaggle datasets ────────────────────────────────────────────────
    "aerial_cactus": {
        "type": "kaggle_competition",
        "id": "aerial-cactus-identification",
        "target_dir": "data/aerial_cactus",
        "description": "Aerial Cactus Identification (Kaggle competition, binary)",
    },
    "aerial_imagery_seg": {
        "type": "kaggle",
        "id": "humansintheloop/semantic-segmentation-of-aerial-imagery",
        "target_dir": "data/aerial_imagery_seg",
        "description": "Semantic Segmentation of Aerial Imagery (400 patches)",
    },
    "building_footprint_seg": {
        "type": "kaggle",
        "id": "balraj98/massachusetts-buildings-dataset",
        "target_dir": "data/building_footprint_seg",
        "description": "Massachusetts Buildings Dataset for footprint segmentation",
    },
    "brain_tumour_detection": {
        "type": "kaggle",
        "id": "navoneel/brain-mri-images-for-brain-tumor-detection",
        "target_dir": "data/brain_tumour_detection",
        "description": "Brain MRI Images — binary tumor/no-tumor classification",
    },
    "celebrity_face_recognition": {
        "type": "kaggle",
        "id": "hereisburak/pins-face-recognition",
        "target_dir": "data/celebrity_face_recognition",
        "description": "105-class celebrity face recognition (PINs)",
    },
    "cell_nuclei_seg": {
        "type": "kaggle_competition",
        "id": "data-science-bowl-2018",
        "target_dir": "data/cell_nuclei_seg",
        "description": "Data Science Bowl 2018 — cell nuclei segmentation",
    },
    "emotion_recognition": {
        "type": "kaggle",
        "id": "msambare/fer2013",
        "target_dir": "data/emotion_recognition",
        "description": "FER-2013 — 7-class facial emotion recognition",
    },
    "face_emotion_recognition": {
        "type": "kaggle",
        "id": "jonathanoheix/face-expression-recognition-dataset",
        "target_dir": "data/face_emotion_recognition",
        "description": "Face Expression Recognition Dataset — 7 classes",
    },
    "food_object_detection": {
        "type": "kaggle",
        "id": "trolukovich/food11-image-dataset",
        "target_dir": "data/food_object_detection",
        "description": "Food-11 image dataset (11 categories)",
    },
    "licence_plate_detector": {
        "type": "kaggle",
        "id": "deepakat002/indian-vehicle-number-plate-yolo-annotation",
        "target_dir": "data/licence_plate_detector",
        "description": "Indian vehicle number plate — YOLO annotations",
    },
    "lung_segmentation": {
        "type": "kaggle",
        "id": "nikhilpandey360/chest-xray-masks-and-labels",
        "target_dir": "data/lung_segmentation",
        "description": "Chest X-ray masks and labels for lung segmentation",
    },
    "medical_image_seg": {
        "type": "kaggle",
        "id": "awsaf49/brats20-dataset-training-validation",
        "target_dir": "data/medical_image_seg",
        "description": "BraTS 2020 brain tumor segmentation (NIfTI format)",
    },
    "plant_disease": {
        "type": "kaggle",
        "id": "emmarex/plantdisease",
        "target_dir": "data/plant_disease",
        "description": "PlantVillage — 38-class plant disease classification",
    },
    "road_segmentation": {
        "type": "kaggle",
        "id": "dansbecker/cityscapes-image-pairs",
        "target_dir": "data/road_segmentation",
        "description": "Cityscapes image pairs for road segmentation",
    },
    "skin_cancer_detection": {
        "type": "kaggle",
        "id": "kmader/skin-cancer-mnist-ham10000",
        "target_dir": "data/skin_cancer_detection",
        "description": "HAM10000 — 7-class skin lesion classification",
    },

    # ── URL datasets (direct HTTP download) ────────────────────────────
    "food_image_recognition": {
        "type": "url",
        "url": "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
        "target_dir": "data/food_image_recognition",
        "description": "Food-101 (ETH Zurich) — 101-class food recognition",
    },
    "logo_detection": {
        "type": "url",
        "url": "http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz",
        "target_dir": "data/logo_detection",
        "description": "Flickr Logos 27 — logo detection & brand recognition",
    },

    # ── Manual datasets (require browser/auth) ─────────────────────────
    "face_anti_spoofing": {
        "type": "manual",
        "target_dir": "data/face_anti_spoofing",
        "description": "Face anti-spoofing (real vs spoof) — Google Drive hosted",
        "instructions": (
            "Download manually from Google Drive:\n"
            "  https://drive.google.com/drive/u/0/folders/1-SWLkGAi34e5ef3tZxOun2hlNk4T4Df3\n"
            "Extract into: data/face_anti_spoofing/"
        ),
    },
    "handwriting_recognition": {
        "type": "manual",
        "target_dir": "data/handwriting_recognition",
        "description": "IAM Handwriting Database — requires registration",
        "instructions": (
            "Register and download from:\n"
            "  http://www.fki.inf.unibe.ch/databases/iam-handwriting-database\n"
            "Extract into: data/handwriting_recognition/"
        ),
    },
    "sign_language_recognition": {
        "type": "manual",
        "target_dir": "data/sign_language_recognition",
        "description": "Sign language .npy sequences (A, B, C) — Google Drive hosted",
        "instructions": (
            "Download manually from Google Drive:\n"
            "  https://drive.google.com/drive/folders/1MMRRVwKaXq710mDS8NWVKA2EqsaP7Hv5\n"
            "Extract into: data/sign_language_recognition/"
        ),
    },

    # ── Local-only (bundled or webcam, no external dataset) ────────────
    "age_gender_recognition": {
        "type": "local_only",
        "target_dir": "data/age_gender_recognition",
        "description": "Inference-only — pre-trained Caffe models bundled, uses webcam/test image",
    },
    "cartoonize_image": {
        "type": "local_only",
        "target_dir": "data/cartoonize_image",
        "description": "Pure OpenCV image processing — user provides input image",
    },
    "face_landmark_detection": {
        "type": "local_only",
        "target_dir": "data/face_landmark_detection",
        "description": "Webcam-based — dlib shape predictor bundled",
    },
    "face_mask_detection": {
        "type": "local_only",
        "target_dir": "data/face_mask_detection",
        "description": "Inference-only — model + test images bundled",
    },
    "fire_smoke_detection": {
        "type": "local_only",
        "target_dir": "data/fire_smoke_detection",
        "description": "Inference-only — custom YOLOv5 weights + test images bundled",
    },
    "pedestrian_detection": {
        "type": "local_only",
        "target_dir": "data/pedestrian_detection",
        "description": "OpenCV HOG detector — sample vid.mp4 bundled",
    },
    "realtime_object_tracking": {
        "type": "local_only",
        "target_dir": "data/realtime_object_tracking",
        "description": "YOLOv5 pre-trained on COCO — Flask app for uploaded media",
    },
    "road_lane_detection": {
        "type": "local_only",
        "target_dir": "data/road_lane_detection",
        "description": "Classical OpenCV (Canny + Hough) — sample image/video bundled",
    },
    "traffic_sign_recognition": {
        "type": "local_only",
        "target_dir": "data/traffic_sign_recognition",
        "description": "GTSRB dataset partially bundled in repo",
    },
    "wildlife_classification": {
        "type": "local_only",
        "target_dir": "data/wildlife_classification",
        "description": "Wildlife camera-trap images + CSVs already committed to repo",
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_READY_FILE = ".ready"
_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between retries


# ---------------------------------------------------------------------------
# DatasetResolver
# ---------------------------------------------------------------------------
class DatasetResolver:
    """Download, cache, and resolve dataset paths deterministically."""

    def __init__(self, registry: Optional[Dict[str, Dict[str, Any]]] = None):
        self.registry = registry or DATASET_REGISTRY
        self._root = REPO_ROOT
        # Merge YAML configs from configs/datasets/ into registry
        self._load_yaml_configs()

    # -- public API ---------------------------------------------------------

    def resolve(self, name: str, *, force: bool = False) -> Path:
        """Resolve a registered dataset — downloads if missing.

        Parameters
        ----------
        name : str
            Registry key (e.g. ``"emotion_recognition"``).
        force : bool
            Re-download even if ``.ready`` marker exists.

        Returns
        -------
        Path
            Absolute path to the dataset directory.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        RuntimeError
            If download fails after retries or manual action is needed.
        """
        if name not in self.registry:
            registered = ", ".join(sorted(self.registry))
            raise KeyError(
                f"Dataset '{name}' is not registered. "
                f"Known datasets: {registered}"
            )

        entry = self.registry[name]
        target = self._root / entry["target_dir"]
        ready = target / _READY_FILE

        # Already resolved?
        if not force and ready.exists():
            log.info("[%s] cached  → %s", name, target)
            return target

        ds_type = entry["type"]

        if ds_type == "local_only":
            return self._resolve_local(name, entry, target)
        if ds_type == "manual":
            return self._resolve_manual(name, entry, target)
        if ds_type == "kaggle":
            return self._download_kaggle(name, entry, target)
        if ds_type == "kaggle_competition":
            return self._download_kaggle_competition(name, entry, target)
        if ds_type == "url":
            return self._download_url(name, entry, target)
        if ds_type == "huggingface":
            return self._download_huggingface(name, entry, target)
        if ds_type == "roboflow":
            return self._download_roboflow(name, entry, target)

        raise RuntimeError(f"Unknown dataset type '{ds_type}' for '{name}'")

    def list_datasets(self) -> List[Dict[str, str]]:
        """Return a summary list of every registered dataset."""
        rows: List[Dict[str, str]] = []
        for key, entry in sorted(self.registry.items()):
            target = self._root / entry["target_dir"]
            ready = (target / _READY_FILE).exists()
            rows.append({
                "name": key,
                "type": entry["type"],
                "target_dir": entry["target_dir"],
                "cached": "yes" if ready else "no",
                "description": entry.get("description", ""),
            })
        return rows

    def status(self) -> str:
        """Print a human-readable status table."""
        rows = self.list_datasets()
        lines = [
            f"{'Name':<35} {'Type':<20} {'Cached':<8} Description",
            "-" * 100,
        ]
        for r in rows:
            lines.append(
                f"{r['name']:<35} {r['type']:<20} {r['cached']:<8} {r['description']}"
            )
        return "\n".join(lines)

    # -- private: local_only ------------------------------------------------

    def _resolve_local(
        self, name: str, entry: Dict[str, Any], target: Path
    ) -> Path:
        """Local-only datasets — just ensure target dir exists."""
        target.mkdir(parents=True, exist_ok=True)
        log.info("[%s] local_only — no download required (%s)", name, target)
        # Don't stamp .ready — nothing was downloaded; avoid false positives
        return target

    # -- private: manual ----------------------------------------------------

    def _resolve_manual(
        self, name: str, entry: Dict[str, Any], target: Path
    ) -> Path:
        """Manual datasets — print instructions and verify presence."""
        target.mkdir(parents=True, exist_ok=True)

        # Check if there's actual content (more than just .ready/.gitkeep)
        contents = [p for p in target.iterdir() if p.name not in (_READY_FILE, ".gitkeep")]
        if contents:
            self._stamp_ready(target)
            log.info("[%s] manual dataset found at %s (%d items)", name, target, len(contents))
            return target

        instructions = entry.get("instructions", "No download instructions available.")
        raise RuntimeError(
            f"Dataset '{name}' requires manual download.\n\n"
            f"{instructions}\n\n"
            f"Once downloaded, place the files in: {target}"
        )

    # -- private: Kaggle dataset --------------------------------------------

    def _download_kaggle(
        self, name: str, entry: Dict[str, Any], target: Path
    ) -> Path:
        """Download a Kaggle dataset via the kaggle CLI."""
        self._check_kaggle_auth()
        dataset_id = entry["id"]
        target.mkdir(parents=True, exist_ok=True)

        cmd = f"kaggle datasets download -d {dataset_id} -p \"{target}\" --unzip"
        self._run_with_retry(name, cmd)

        self._validate_not_empty(name, target)
        self._stamp_ready(target)
        self._write_dataset_info(name, entry, target)
        log.info("[%s] kaggle download complete → %s", name, target)
        return target

    # -- private: Kaggle competition ----------------------------------------

    def _download_kaggle_competition(
        self, name: str, entry: Dict[str, Any], target: Path
    ) -> Path:
        """Download a Kaggle competition dataset (requires accepting rules)."""
        self._check_kaggle_auth()
        competition_id = entry["id"]
        target.mkdir(parents=True, exist_ok=True)

        cmd = f"kaggle competitions download -c {competition_id} -p \"{target}\""
        self._run_with_retry(name, cmd)

        # Competition downloads are zipped — extract
        self._extract_all_archives(target)

        self._validate_not_empty(name, target)
        self._stamp_ready(target)
        self._write_dataset_info(name, entry, target)
        log.info("[%s] kaggle competition download complete → %s", name, target)
        return target

    # -- private: URL -------------------------------------------------------

    def _download_url(
        self, name: str, entry: Dict[str, Any], target: Path
    ) -> Path:
        """Download from a direct URL and extract archives."""
        import urllib.request
        import urllib.error

        url = entry["url"]
        target.mkdir(parents=True, exist_ok=True)

        filename = url.rsplit("/", 1)[-1]
        dest_file = target / filename

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                log.info("[%s] downloading %s (attempt %d/%d)", name, url, attempt, _MAX_RETRIES)
                self._urlretrieve_with_progress(url, dest_file)
                break
            except (urllib.error.URLError, OSError) as exc:
                log.warning("[%s] attempt %d failed: %s", name, attempt, exc)
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to download '{name}' from {url} after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                time.sleep(_RETRY_DELAY * attempt)

        # Verify checksum if provided
        expected_sha256 = entry.get("sha256")
        if expected_sha256 and dest_file.exists():
            self._verify_checksum(name, dest_file, expected_sha256)

        # Extract if archive
        if dest_file.exists():
            self._extract_archive(dest_file, target)

        self._validate_not_empty(name, target)
        self._stamp_ready(target)
        self._write_dataset_info(name, entry, target)
        log.info("[%s] URL download complete → %s", name, target)
        return target

    # -- private: Hugging Face ----------------------------------------------

    def _download_huggingface(
        self, name: str, entry: Dict[str, Any], target: Path
    ) -> Path:
        """Download from Hugging Face Datasets hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "Hugging Face 'datasets' library not installed.\n"
                "Install with:  pip install datasets\n"
                f"Dataset: {entry.get('id', name)}"
            )

        dataset_id = entry["id"]
        subset = entry.get("subset")  # e.g. "train" or a config name
        split = entry.get("split")    # e.g. "train", "test", "train+test"
        target.mkdir(parents=True, exist_ok=True)

        log.info("[%s] loading Hugging Face dataset: %s", name, dataset_id)

        kwargs: Dict[str, Any] = {}
        if subset:
            kwargs["name"] = subset
        if split:
            kwargs["split"] = split

        ds = load_dataset(dataset_id, **kwargs)

        # Save to disk in Arrow format for fast reload
        save_path = target / "hf_dataset"
        if hasattr(ds, "save_to_disk"):
            ds.save_to_disk(str(save_path))
        else:
            # DatasetDict (multiple splits)
            for split_name, split_ds in ds.items():
                split_ds.save_to_disk(str(save_path / split_name))

        self._validate_not_empty(name, target)
        self._stamp_ready(target)
        self._write_dataset_info(name, entry, target)
        log.info("[%s] Hugging Face download complete → %s", name, target)
        return target

    # -- private: Roboflow --------------------------------------------------

    def _download_roboflow(
        self, name: str, entry: Dict[str, Any], target: Path
    ) -> Path:
        """Download from Roboflow Universe (public datasets)."""
        try:
            from roboflow import Roboflow
        except ImportError:
            raise RuntimeError(
                "Roboflow SDK not installed.\n"
                "Install with:  pip install roboflow\n"
                f"Dataset: {entry.get('id', name)}"
            )

        workspace = entry.get("workspace", "")
        project_id = entry.get("project", "")
        version_num = entry.get("version", 1)
        export_format = entry.get("format", "yolov8")  # yolov8 = YOLO26-compatible
        api_key = entry.get("api_key") or os.environ.get("ROBOFLOW_API_KEY", "")

        if not workspace or not project_id:
            raise RuntimeError(
                f"Roboflow dataset '{name}' requires 'workspace' and 'project' fields.\n"
                f"Find these at: https://universe.roboflow.com/"
            )

        target.mkdir(parents=True, exist_ok=True)

        log.info("[%s] downloading from Roboflow: %s/%s v%d", name, workspace, project_id, version_num)

        rf = Roboflow(api_key=api_key) if api_key else Roboflow()
        project = rf.workspace(workspace).project(project_id)
        version = project.version(version_num)
        version.download(export_format, location=str(target))

        self._validate_not_empty(name, target)
        self._stamp_ready(target)
        self._write_dataset_info(name, entry, target)
        log.info("[%s] Roboflow download complete → %s", name, target)
        return target

    # -- helpers: extraction ------------------------------------------------

    @staticmethod
    def _extract_archive(archive: Path, dest: Path) -> None:
        """Extract a .zip or .tar.gz/.tgz archive into *dest*."""
        name_lower = archive.name.lower()
        if name_lower.endswith(".zip"):
            log.info("  extracting zip: %s", archive.name)
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest)
            archive.unlink()
        elif name_lower.endswith((".tar.gz", ".tgz")):
            log.info("  extracting tar.gz: %s", archive.name)
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(dest)
            archive.unlink()
        elif name_lower.endswith(".tar"):
            log.info("  extracting tar: %s", archive.name)
            with tarfile.open(archive, "r:") as tf:
                tf.extractall(dest)
            archive.unlink()

    def _extract_all_archives(self, target: Path) -> None:
        """Extract every archive found in *target*."""
        for f in list(target.iterdir()):
            if f.is_file() and f.suffix.lower() in (".zip", ".gz", ".tgz", ".tar"):
                self._extract_archive(f, target)

    # -- helpers: Kaggle auth -----------------------------------------------

    @staticmethod
    def _check_kaggle_auth() -> None:
        """Verify ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` or ``~/.kaggle/kaggle.json``."""
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            return
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Either:\n"
            "  1. Set KAGGLE_USERNAME + KAGGLE_KEY environment variables, or\n"
            "  2. Place kaggle.json in ~/.kaggle/\n"
            "See: https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    # -- helpers: run command with retry ------------------------------------

    @staticmethod
    def _run_with_retry(name: str, cmd: str) -> None:
        """Run a shell command with retry logic."""
        import subprocess

        for attempt in range(1, _MAX_RETRIES + 1):
            log.info("[%s] running: %s (attempt %d/%d)", name, cmd, attempt, _MAX_RETRIES)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                if result.stdout:
                    log.debug("[%s] stdout: %s", name, result.stdout[:500])
                return
            log.warning(
                "[%s] command failed (exit %d): %s",
                name, result.returncode, result.stderr[:500],
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)

        raise RuntimeError(
            f"Command failed after {_MAX_RETRIES} attempts for '{name}': {cmd}\n"
            f"Last stderr: {result.stderr[:1000]}"
        )

    # -- helpers: validation ------------------------------------------------

    @staticmethod
    def _validate_not_empty(name: str, target: Path) -> None:
        """Raise if the target directory is empty or missing."""
        if not target.exists():
            raise RuntimeError(f"Target directory missing after download: {target}")
        contents = [
            p for p in target.iterdir()
            if p.name not in (_READY_FILE, ".gitkeep")
        ]
        if not contents:
            raise RuntimeError(
                f"Dataset '{name}' downloaded but target is empty: {target}"
            )

    # -- helpers: cache stamp -----------------------------------------------

    @staticmethod
    def _stamp_ready(target: Path) -> None:
        """Write a ``.ready`` marker with a timestamp."""
        marker = target / _READY_FILE
        marker.write_text(
            f"downloaded={time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
            encoding="utf-8",
        )

    # -- helpers: dataset metadata ------------------------------------------

    @staticmethod
    def _write_dataset_info(name: str, entry: Dict[str, Any], target: Path) -> None:
        """Write ``dataset_info.json`` with source metadata."""
        info = {
            "dataset_key": name,
            "source_type": entry.get("type", "unknown"),
            "description": entry.get("description", ""),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target_dir": str(target),
        }
        # Include source-specific info
        for k in ("id", "url", "workspace", "project", "version", "subset", "split"):
            if k in entry:
                info[f"source_{k}"] = entry[k]

        info_path = target / "dataset_info.json"
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        log.info("[%s] wrote dataset_info.json", name)

    # -- helpers: checksum verification -------------------------------------

    @staticmethod
    def _verify_checksum(name: str, filepath: Path, expected: str) -> None:
        """Verify SHA-256 checksum of a downloaded file."""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        actual = sha.hexdigest()
        if actual != expected:
            filepath.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for '{name}':\n"
                f"  Expected: {expected}\n"
                f"  Got:      {actual}\n"
                f"File deleted — re-run to retry download."
            )
        log.info("[%s] checksum verified (%s...)", name, actual[:12])

    # -- helpers: YAML config loading ----------------------------------------

    def _load_yaml_configs(self) -> None:
        """Load YAML configs from configs/datasets/ and merge into registry.

        YAML entries override dict entries of the same key.
        """
        if not _CONFIGS_DIR.is_dir():
            return
        for yaml_path in sorted(_CONFIGS_DIR.glob("*.yaml")):
            try:
                cfg = _load_yaml(yaml_path)
            except Exception as exc:
                log.warning("Failed to load %s: %s", yaml_path, exc)
                continue

            key = cfg.get("key") or yaml_path.stem
            ds_type = cfg.get("type")
            if not ds_type:
                continue

            entry: Dict[str, Any] = {
                "type": ds_type,
                "target_dir": cfg.get("target_dir", f"data/{key}"),
            }
            # Copy optional fields
            for field in ("id", "url", "description", "instructions",
                          "sha256", "subset", "split",
                          "workspace", "project", "version", "format",
                          "api_key"):
                if field in cfg:
                    entry[field] = cfg[field]

            self.registry[key] = entry

    # -- helpers: progress download -----------------------------------------

    @staticmethod
    def _urlretrieve_with_progress(url: str, dest: Path) -> None:
        """Download a URL to *dest* with progress logging."""
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "CVProjects/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB
            last_log = 0.0

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if total and now - last_log > 2.0:
                        pct = downloaded / total * 100
                        mb = downloaded / 1048576
                        log.info("  progress: %.1f MB / %.1f MB (%.0f%%)", mb, total / 1048576, pct)
                        last_log = now

        if total:
            log.info("  download complete: %.1f MB", total / 1048576)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------
def ensure_dataset(name: str, *, force: bool = False) -> Path:
    """Convenience wrapper: resolve a dataset, downloading if needed.

    Usage::

        from utils.datasets import ensure_dataset
        data_path = ensure_dataset("emotion_recognition")

    Parameters
    ----------
    name : str
        Dataset registry key.
    force : bool
        Force re-download even if cached.

    Returns
    -------
    Path
        Absolute path to the dataset directory.
    """
    return DatasetResolver().resolve(name, force=force)
