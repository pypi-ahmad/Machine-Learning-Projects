"""Image captioning with BLIP / BLIP-2.

Runs inference on sample images, compares with reference captions,
computes BLEU / ROUGE / CIDEr-like metrics.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

from utils.logger import get_logger
from utils.training_common import (
    SEED, WORKSPACE, cleanup_gpu, ensure_output_dirs, save_json, seed_everything,
)

logger = get_logger(__name__)


# ======================================================================
# Reference caption parser
# ======================================================================

def _parse_captions(captions_path: Path) -> dict[str, list[str]]:
    """Parse captions.txt (Flickr8k / Flickr30k format).
    Expected format:  image_name,caption  or  image_name#idx\tcaption
    Returns {filename: [cap1, cap2, ...]}
    """
    caps: dict[str, list[str]] = {}
    if not captions_path.exists():
        return caps

    text = captions_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("image"):  # skip header
            continue
        # Try tab-separated: "img#idx\tcaption"
        parts = line.split("\t")
        if len(parts) >= 2:
            key = parts[0].split("#")[0].strip()
            cap = parts[1].strip()
        else:
            # Try comma-separated: "img.jpg,caption text"
            idx = line.find(",")
            if idx > 0:
                key = line[:idx].strip()
                cap = line[idx + 1:].strip()
            else:
                continue
        if key and cap:
            caps.setdefault(key, []).append(cap)

    return caps


# ======================================================================
# MAIN
# ======================================================================

def run_captioning_pipeline(
    slug: str,
    images_dir: str | Path | None = None,
    captions_path: str | Path | None = None,
    *,
    model_name: str = "Salesforce/blip-image-captioning-large",
    max_samples: int = 200,
    seed: int = SEED,
    force: bool = False,
) -> dict:
    """Generate captions for images, evaluate against references."""
    seed_everything(seed)
    dirs = ensure_output_dirs(slug)
    metrics_file = dirs["metrics"] / "phase2_metrics.json"
    if metrics_file.exists() and not force:
        logger.info("Phase 2 already done for %s", slug)
        return json.loads(metrics_file.read_text())

    # ---- Resolve paths ----------------------------------------------------
    raw_dir = WORKSPACE / "data" / slug / "raw"
    if images_dir is None:
        # Search for images directory
        for cand in [raw_dir / "images", raw_dir / "Images", raw_dir]:
            if cand.is_dir() and any(cand.glob("*.jpg")):
                images_dir = cand
                break
        if images_dir is None:
            return {"slug": slug, "status": "ERROR", "error": "No images directory found"}
    images_dir = Path(images_dir)

    if captions_path is None:
        for cand in [raw_dir / "captions.txt", raw_dir / "captions.csv", raw_dir / "token.txt", raw_dir / "results.csv"]:
            if cand.exists():
                captions_path = cand
                break
    if captions_path is not None:
        captions_path = Path(captions_path)

    ref_caps = _parse_captions(captions_path) if captions_path else {}
    logger.info("[%s] Found %d images dir, %d reference caption groups", slug, len(list(images_dir.iterdir())), len(ref_caps))

    # ---- Collect image files + references ---------------------------------
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    img_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in image_exts])
    if not img_files:
        return {"slug": slug, "status": "ERROR", "error": "No image files found"}

    random.seed(seed)
    if len(img_files) > max_samples:
        img_files = random.sample(img_files, max_samples)

    result: dict = {
        "slug": slug,
        "task": "image_captioning",
        "model_name": model_name,
        "n_images_total": len(img_files),
        "n_ref_caption_groups": len(ref_caps),
        "status": "running",
    }

    # ---- Load model -------------------------------------------------------
    import torch
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try BLIP (not BLIP-2) for 8 GB compat
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as exc:
        # Fall back to smaller model
        fallback = "Salesforce/blip-image-captioning-base"
        logger.warning("Failed to load %s (%s); falling back to %s", model_name, exc, fallback)
        from transformers import BlipForConditionalGeneration, BlipProcessor
        processor = BlipProcessor.from_pretrained(fallback)
        model = BlipForConditionalGeneration.from_pretrained(fallback).to(device)
        model.eval()
        model_name = fallback
        result["model_name"] = fallback
        result["fallback_reason"] = str(exc)

    # ---- Generate captions ------------------------------------------------
    generated: list[dict] = []
    references_matched: list[tuple[str, list[str]]] = []

    for img_path in img_files:
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, num_beams=4)
            caption = processor.decode(out[0], skip_special_tokens=True)
        except Exception as exc:
            caption = f"[ERROR: {exc}]"

        entry = {"image": img_path.name, "generated": caption}
        refs = ref_caps.get(img_path.name, [])
        if refs:
            entry["references"] = refs[:5]
            references_matched.append((caption, refs[:5]))
        generated.append(entry)

    result["n_generated"] = len(generated)
    result["sample_captions"] = generated[:20]

    # ---- Evaluate ---------------------------------------------------------
    if references_matched:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        r1s, r2s, rls = [], [], []
        bleu_scores: list[float] = []

        for gen_cap, refs in references_matched:
            # ROUGE: best match among references
            best_r1, best_r2, best_rl = 0, 0, 0
            for ref in refs:
                s = scorer.score(ref, gen_cap)
                best_r1 = max(best_r1, s["rouge1"].fmeasure)
                best_r2 = max(best_r2, s["rouge2"].fmeasure)
                best_rl = max(best_rl, s["rougeL"].fmeasure)
            r1s.append(best_r1)
            r2s.append(best_r2)
            rls.append(best_rl)

            # Simple BLEU (unigram)
            gen_tokens = gen_cap.lower().split()
            max_bleu = 0
            for ref in refs:
                ref_tokens = ref.lower().split()
                if not gen_tokens or not ref_tokens:
                    continue
                matches = sum(1 for t in gen_tokens if t in ref_tokens)
                precision = matches / len(gen_tokens) if gen_tokens else 0
                # brevity penalty
                bp = min(1.0, len(gen_tokens) / max(len(ref_tokens), 1))
                max_bleu = max(max_bleu, bp * precision)
            bleu_scores.append(max_bleu)

        result["eval_metrics"] = {
            "n_evaluated": len(references_matched),
            "rouge1": float(np.mean(r1s)),
            "rouge2": float(np.mean(r2s)),
            "rougeL": float(np.mean(rls)),
            "bleu1_approx": float(np.mean(bleu_scores)),
        }
        logger.info("[%s] ROUGE-1=%.4f  ROUGE-L=%.4f  BLEU1~=%.4f",
                    slug, np.mean(r1s), np.mean(rls), np.mean(bleu_scores))
    else:
        logger.info("[%s] No reference captions matched -- skipping evaluation", slug)
        result["eval_metrics"] = {"note": "No matched references for evaluation"}

    # ---- Save -------------------------------------------------------------
    result["status"] = "OK"
    save_json(result, metrics_file)

    # Save full generated captions
    save_json(generated, dirs["artifacts"] / "generated_captions.json")

    del model
    cleanup_gpu()
    return result


def run_project(project_slug, project_dir, raw_paths, processed_dir, outputs_dir, config, force=False):
    """Unified entry point called by the Phase 2.1 orchestrator."""
    _KEYS = {"model_name","max_samples","seed"}
    kw = {k: v for k, v in config.items() if k in _KEYS}
    r = run_captioning_pipeline(
        project_slug,
        images_dir=config.get("images_dir"),
        captions_path=config.get("captions_path"),
        force=force, **kw,
    )
    return {
        "status": r.get("status", "UNKNOWN"),
        "model_name": r.get("model_name", config.get("model_name", "")),
        "dataset_size": r.get("n_images_total", 0),
        "main_metrics": r.get("eval_metrics", {}),
        "val_metrics": {},
        "training_mode": "inference",
        "train_runtime_sec": 0,
        "notes": r.get("fallback_reason", ""),
        "full_result": r,
    }
