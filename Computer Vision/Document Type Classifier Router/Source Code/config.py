"""Document Type Classifier Router — configuration and class mappings.

16 document types from the RVL-CDIP–derived Real World Documents
Collections dataset.  Each type maps to a downstream pipeline stub
that would handle further processing (OCR, extraction, archival, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Document type classes ─────────────────────────────────

CLASS_NAMES: list[str] = [
    "advertisement",
    "budget",
    "email",
    "file_folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news_article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific_publication",
    "scientific_report",
    "specification",
]

# ── Routing table ─────────────────────────────────────────
# Maps each document type to a downstream pipeline name.
# These are *stubs* — real deployments would replace them
# with actual pipeline entry-points / queue topics.

ROUTE_TABLE: dict[str, str] = {
    "advertisement":          "marketing_pipeline",
    "budget":                 "finance_pipeline",
    "email":                  "correspondence_pipeline",
    "file_folder":            "archive_pipeline",
    "form":                   "form_extraction_pipeline",
    "handwritten":            "ocr_handwriting_pipeline",
    "invoice":                "invoice_extraction_pipeline",
    "letter":                 "correspondence_pipeline",
    "memo":                   "correspondence_pipeline",
    "news_article":           "content_ingestion_pipeline",
    "presentation":           "content_ingestion_pipeline",
    "questionnaire":          "form_extraction_pipeline",
    "resume":                 "hr_pipeline",
    "scientific_publication": "research_pipeline",
    "scientific_report":      "research_pipeline",
    "specification":          "engineering_pipeline",
}

# Human-friendly display labels
DISPLAY_LABELS: dict[str, str] = {
    "advertisement":          "Advertisement",
    "budget":                 "Budget",
    "email":                  "Email",
    "file_folder":            "File Folder",
    "form":                   "Form",
    "handwritten":            "Handwritten",
    "invoice":                "Invoice",
    "letter":                 "Letter",
    "memo":                   "Memo",
    "news_article":           "News Article",
    "presentation":           "Presentation",
    "questionnaire":          "Questionnaire",
    "resume":                 "Resume / CV",
    "scientific_publication": "Scientific Publication",
    "scientific_report":      "Scientific Report",
    "specification":          "Specification",
}


# ── Config dataclass ──────────────────────────────────────

@dataclass
class RouterConfig:
    """All tuneable knobs for the document classifier + router."""

    # ── Model ─────────────────────────────────────────────
    model_name: str = "resnet18"
    num_classes: int = 16
    imgsz: int = 224
    device: str | None = None

    # ── Training ──────────────────────────────────────────
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.2
    num_workers: int = 4

    # ── Inference ─────────────────────────────────────────
    weights_path: str = "runs/document_cls/best_model.pt"
    confidence_threshold: float = 0.3

    # ── Routing ───────────────────────────────────────────
    route_table: dict[str, str] = field(default_factory=lambda: dict(ROUTE_TABLE))
    fallback_pipeline: str = "manual_review_pipeline"

    # ── Visualisation ─────────────────────────────────────
    font_scale: float = 0.55
    badge_color: tuple[int, int, int] = (180, 80, 40)
    route_color: tuple[int, int, int] = (40, 140, 200)
    text_color: tuple[int, int, int] = (255, 255, 255)
    grid_thumb_size: int = 200
    grid_cols: int = 4

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── Helpers ────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RouterConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> RouterConfig:
    """Load config from JSON or YAML, falling back to defaults."""
    if path is None:
        return RouterConfig()
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except ImportError:
            data = json.loads(text)
    else:
        data = json.loads(text)
    return RouterConfig.from_dict(data)
