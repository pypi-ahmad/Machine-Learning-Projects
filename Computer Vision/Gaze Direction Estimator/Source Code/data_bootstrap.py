"""Dataset bootstrap for Gaze Direction Estimator.

Downloads and prepares a public gaze-estimation dataset for coarse
LEFT/RIGHT/UP/DOWN/CENTER evaluation.

The chosen source, MPIIFaceGaze, provides continuous screen-coordinate
targets rather than coarse classes. During preparation this bootstrap
derives coarse labels by binning each participant's recorded screen
coordinate range into left/right/up/down/center regions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("gaze.data_bootstrap")

PROJECT_KEY = "gaze_direction_estimator"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY
DATASET_URL = "https://collaborative-ai.org/files/datasets/MPIIFaceGaze.zip"
ARCHIVE_NAME = "MPIIFaceGaze.zip"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LEFT_BOUND = 0.35
RIGHT_BOUND = 0.65
MAX_IMAGES_PER_PARTICIPANT = 24


@dataclass
class GazeSample:
    image_path: str
    participant: str
    screen_x: float
    screen_y: float
    norm_x: float
    norm_y: float
    coarse_label: str


def ensure_gaze_dataset(*, force: bool = False) -> Path:
    """Download and prepare the gaze evaluation dataset."""
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info(
            "[%s] Dataset already prepared at %s -- skipping",
            PROJECT_KEY,
            DATA_ROOT,
        )
        return DATA_ROOT

    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    raw_dir = DATA_ROOT / "raw"
    processed_dir = DATA_ROOT / "processed"
    media_dir = processed_dir / "media"
    raw_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    archive_path = _download_archive(raw_dir)
    samples = _prepare_samples(archive_path, media_dir)
    if not samples:
        raise RuntimeError("Could not prepare a public gaze evaluation dataset automatically.")

    manifest_path = processed_dir / "manifest.csv"
    _write_manifest(manifest_path, samples)
    _write_info(DATA_ROOT, sample_count=len(samples), participants=len({sample.participant for sample in samples}))

    ready_marker.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8",
    )
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _download_archive(raw_dir: Path) -> Path:
    archive_path = raw_dir / ARCHIVE_NAME
    if archive_path.exists():
        return archive_path

    tmp_path = archive_path.with_suffix(".download")
    log.info("[%s] Downloading MPIIFaceGaze archive from %s", PROJECT_KEY, DATASET_URL)
    with urllib.request.urlopen(DATASET_URL) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(archive_path)
    return archive_path


def _prepare_samples(archive_path: Path, media_dir: Path) -> list[GazeSample]:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        member_set = set(members)
        annotation_members = [
            name for name in members
            if _is_annotation_member(name)
        ]

        samples: list[GazeSample] = []
        for annotation_member in annotation_members:
            participant = Path(annotation_member).stem
            rows = _read_annotation_rows(archive, annotation_member)
            if not rows:
                continue

            x_values = [row["screen_x"] for row in rows]
            y_values = [row["screen_y"] for row in rows]
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)

            collected = 0
            for row in rows:
                if collected >= MAX_IMAGES_PER_PARTICIPANT:
                    break

                member_name = _resolve_image_member(members, member_set, annotation_member, row["image_rel"])
                if member_name is None:
                    continue

                norm_x = _normalize(row["screen_x"], x_min, x_max)
                norm_y = _normalize(row["screen_y"], y_min, y_max)
                coarse_label = _coarse_label(norm_x, norm_y)
                output_name = _output_name(participant, row["image_rel"])
                output_path = media_dir / output_name
                if not output_path.exists():
                    with archive.open(member_name) as src, open(output_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                samples.append(
                    GazeSample(
                        image_path=str(Path("processed") / "media" / output_name),
                        participant=participant,
                        screen_x=row["screen_x"],
                        screen_y=row["screen_y"],
                        norm_x=norm_x,
                        norm_y=norm_y,
                        coarse_label=coarse_label,
                    ),
                )
                collected += 1

    return samples


def _is_annotation_member(member_name: str) -> bool:
    path = Path(member_name)
    return (
        path.suffix.lower() == ".txt"
        and path.stem.startswith("p")
        and path.parent.name == path.stem
    )


def _read_annotation_rows(
    archive: zipfile.ZipFile,
    annotation_member: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with archive.open(annotation_member) as handle:
        for raw_line in handle.read().decode("utf-8", errors="ignore").splitlines():
            parts = raw_line.split()
            if len(parts) < 3:
                continue
            try:
                screen_x = float(parts[1])
                screen_y = float(parts[2])
            except ValueError:
                continue
            rows.append(
                {
                    "image_rel": parts[0].replace("\\", "/"),
                    "screen_x": screen_x,
                    "screen_y": screen_y,
                },
            )
    return rows


def _resolve_image_member(
    members: list[str],
    member_set: set[str],
    annotation_member: str,
    image_rel: str,
) -> str | None:
    annotation_dir = Path(annotation_member).parent
    normalized_rel = image_rel.replace("\\", "/")
    candidates = [
        (annotation_dir / normalized_rel).as_posix(),
        (Path(annotation_dir.name) / normalized_rel).as_posix(),
        normalized_rel,
    ]
    for candidate in candidates:
        if candidate in member_set:
            return candidate

    suffix = f"/{annotation_dir.name}/{normalized_rel}"
    for member in members:
        if member.endswith(suffix):
            return member
    return None


def _normalize(value: float, low: float, high: float) -> float:
    span = high - low
    if span <= 1e-6:
        return 0.5
    return max(0.0, min(1.0, (value - low) / span))


def _coarse_label(norm_x: float, norm_y: float) -> str:
    if norm_x < LEFT_BOUND:
        return "LEFT"
    if norm_x > RIGHT_BOUND:
        return "RIGHT"
    if norm_y < LEFT_BOUND:
        return "UP"
    if norm_y > RIGHT_BOUND:
        return "DOWN"
    return "CENTER"


def _output_name(participant: str, image_rel: str) -> str:
    rel_path = image_rel.replace("\\", "/")
    rel_path = rel_path.replace("/", "_")
    return f"{participant}_{rel_path}"


def _write_manifest(manifest_path: Path, samples: list[GazeSample]) -> None:
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "participant",
                "screen_x",
                "screen_y",
                "norm_x",
                "norm_y",
                "coarse_label",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "image_path": sample.image_path,
                    "participant": sample.participant,
                    "screen_x": round(sample.screen_x, 3),
                    "screen_y": round(sample.screen_y, 3),
                    "norm_x": round(sample.norm_x, 4),
                    "norm_y": round(sample.norm_y, 4),
                    "coarse_label": sample.coarse_label,
                },
            )


def _write_info(data_path: Path, *, sample_count: int, participants: int) -> None:
    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": "direct_url",
        "dataset_url": DATASET_URL,
        "description": "MPIIFaceGaze samples with coarse direction labels derived from continuous screen-coordinate annotations.",
        "sample_count": sample_count,
        "participant_count": participants,
        "manifest": str((data_path / "processed" / "manifest.csv").relative_to(data_path)),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path = data_path / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the gaze-direction evaluation dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the dataset even if it already exists",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parser.parse_args(argv)
    data_root = ensure_gaze_dataset(force=args.force)
    print(data_root)


if __name__ == "__main__":
    main()
