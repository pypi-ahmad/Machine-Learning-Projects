#!/usr/bin/env python3
"""
Image Captioning — ViT + GPT-2 (PyTorch)
==========================================
Generate captions for images using a Vision Encoder-Decoder architecture
(ViT encoder  + GPT-2 decoder).  Uses the pretrained
``nlpconnect/vit-gpt2-image-captioning`` model from Hugging Face for
inference, and optionally fine-tunes on a small COCO-2017 subset.

Dataset
-------
* Kaggle: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

Run
---
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    get_device,
    ensure_dir,
    dataset_prompt,
    parse_common_args,
    save_metrics,
    run_metadata,
    dataset_fingerprint,
    write_split_manifest,
    make_file_splits,
    dataset_missing_metrics,
    resolve_device_from_args,
    configure_cuda_allocator,
    EarlyStopping,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
KAGGLE_SLUG = "awsaf49/coco-2017-dataset"
PRETRAINED_MODEL = "nlpconnect/vit-gpt2-image-captioning"
MAX_CAPTION_LEN = 64
FINETUNE_EPOCHS = 2
FINETUNE_BATCH = 4
FINETUNE_LR = 5e-5
FINETUNE_SAMPLES = 500      # small subset for demo
NUM_DEMO_IMAGES = 8


# ═════════════════════════════════════════════════════════════
#  Dataset helpers
# ═════════════════════════════════════════════════════════════

def find_images(root: Path, limit: int = 0):
    """Recursively find JPEG / PNG images under *root*."""
    exts = {".jpg", ".jpeg", ".png"}
    imgs = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            imgs.append(p)
            if 0 < limit <= len(imgs):
                break
    return sorted(imgs)


def load_coco_annotations(root: Path):
    """Try to load COCO captions JSON and return {image_id: [captions]}."""
    caption_map: dict[int, list[str]] = {}
    id_to_file: dict[int, str] = {}

    patterns = [
        root / "annotations" / "captions_train2017.json",
        root / "captions_train2017.json",
    ]
    # Also search recursively
    for f in root.rglob("captions_train2017.json"):
        patterns.append(f)

    ann_file = None
    for p in patterns:
        if p.exists():
            ann_file = p
            break

    if ann_file is None:
        return caption_map, id_to_file

    logger.info("Loading annotations from %s", ann_file)
    with open(ann_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for img_info in data.get("images", []):
        id_to_file[img_info["id"]] = img_info["file_name"]

    for ann in data.get("annotations", []):
        iid = ann["image_id"]
        caption_map.setdefault(iid, []).append(ann["caption"])

    return caption_map, id_to_file


class CocoCaptionDataset(Dataset):
    """Thin COCO dataset that returns (pixel_values, labels) for fine-tuning."""

    def __init__(self, image_paths, captions, feature_extractor, tokenizer, max_len=MAX_CAPTION_LEN):
        self.image_paths = image_paths
        self.captions = captions
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.feature_extractor(images=img, return_tensors="pt").pixel_values.squeeze(0)

        cap = self.captions[idx]
        encoding = self.tokenizer(
            cap, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        labels = encoding.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return pixel_values, labels


# ═════════════════════════════════════════════════════════════
#  Inference
# ═════════════════════════════════════════════════════════════

def generate_caption(model, feature_extractor, tokenizer, image_path, device, max_len=MAX_CAPTION_LEN):
    """Generate a caption for a single image file."""
    img = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=max_len,
            num_beams=4,
            early_stopping=True,
        )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


# ═════════════════════════════════════════════════════════════
#  Fine-tuning (small subset)
# ═════════════════════════════════════════════════════════════

def finetune(model, feature_extractor, tokenizer, image_paths, captions, device, output_dir,
             *, epochs=None, batch_size=None, use_amp_override=None,
             val_image_paths=None, val_captions=None, patience=None):
    """Fine-tune the VisionEncoderDecoder on a small COCO subset."""
    epochs = epochs or FINETUNE_EPOCHS
    batch_size = batch_size or FINETUNE_BATCH
    ds = CocoCaptionDataset(image_paths, captions, feature_extractor, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dl = None
    if val_image_paths and val_captions:
        val_ds = CocoCaptionDataset(val_image_paths, val_captions, feature_extractor, tokenizer)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR)

    use_amp = use_amp_override if use_amp_override is not None else (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    es = EarlyStopping(patience=patience or 3, mode="min") if (patience and val_dl) else None

    losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for pv, labels in dl:
            pv, labels = pv.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(pixel_values=pv, labels=labels)
                loss = out.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg = epoch_loss / max(len(dl), 1)
        losses.append(avg)

        # ── Val loss ─────────────────────────────────────────
        v_avg = None
        if val_dl:
            model.eval()
            v_total = 0.0
            v_count = 0
            with torch.no_grad():
                for pv, labels in val_dl:
                    pv, labels = pv.to(device), labels.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = model(pixel_values=pv, labels=labels)
                    v_total += out.loss.item()
                    v_count += 1
            v_avg = v_total / max(v_count, 1)
            val_losses.append(v_avg)
            logger.info("Finetune epoch %d/%d — loss=%.4f  val_loss=%.4f", epoch, epochs, avg, v_avg)

            if es and es(v_avg):
                logger.info("Early stopping at epoch %d", epoch)
                break
        else:
            logger.info("Finetune epoch %d/%d — loss=%.4f", epoch, epochs, avg)

    model.eval()
    return losses, val_losses


# ═════════════════════════════════════════════════════════════
#  BLEU evaluation
# ═════════════════════════════════════════════════════════════

def compute_bleu(predictions: list[str], references: list[list[str]]):
    """Compute corpus BLEU. Each reference is a list of possible captions."""
    try:
        import evaluate
        bleu_metric = evaluate.load("bleu")
        result = bleu_metric.compute(predictions=predictions, references=references)
        return result
    except Exception:
        pass

    # Fallback: nltk
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tok = [[ref.split() for ref in ref_list] for ref_list in references]
        hyps_tok = [p.split() for p in predictions]
        score = corpus_bleu(refs_tok, hyps_tok,
                            smoothing_function=SmoothingFunction().method1)
        return {"bleu": score}
    except Exception:
        logger.warning("Could not compute BLEU — neither 'evaluate' nor 'nltk' available.")
        return {"bleu": None}


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def save_captioned_images(image_paths, captions, output_dir, n=NUM_DEMO_IMAGES):
    n = min(n, len(image_paths))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(n):
        img = Image.open(image_paths[i]).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(captions[i], fontsize=8, wrap=True)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Image Captioning — ViT + GPT-2", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "captioned_images.png", dpi=150)
    plt.close(fig)
    logger.info("Saved captioned images → %s", output_dir / "captioned_images.png")


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging()
    args = parse_common_args("Image Captioning — ViT + GPT-2")
    set_seed(args.seed)
    configure_cuda_allocator()

    paths = project_paths(__file__)
    data_dir = paths["data"]
    output_dir = ensure_dir(paths["outputs"])
    device = resolve_device_from_args(args)

    # ── Load pretrained model ────────────────────────────────
    from transformers import (
        VisionEncoderDecoderModel,
        ViTImageProcessor,
        AutoTokenizer,
    )

    logger.info("Loading pretrained model: %s", PRETRAINED_MODEL)
    feature_extractor = ViTImageProcessor.from_pretrained(PRETRAINED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_MODEL).to(device)
    model.eval()

    # ── Download COCO dataset ────────────────────────────────
    ds_path = download_kaggle_dataset(
        KAGGLE_SLUG, data_dir,
        dataset_name="COCO 2017 Dataset",
    )

    if args.download_only:
        logger.info("Download complete — exiting (--download-only).")
        sys.exit(0)

    # ── CLI overrides ────────────────────────────────────────
    epochs = args.epochs or FINETUNE_EPOCHS
    batch_size = args.batch_size or FINETUNE_BATCH
    use_amp = not args.no_amp and device.type == "cuda"
    patience = args.patience
    max_samples = 200 if args.mode == "smoke" else FINETUNE_SAMPLES
    if args.mode == "smoke":
        epochs = 1

    images = find_images(ds_path, limit=200 if args.mode == "smoke" else 2000)
    logger.info("Found %d images under %s", len(images), ds_path)

    caption_map, id_to_file = load_coco_annotations(ds_path)
    logger.info("Loaded annotations for %d images", len(caption_map))

    # ── Build matched image-caption pairs ────────────────────
    file_to_id = {v: k for k, v in id_to_file.items()}
    all_imgs, all_caps, all_refs = [], [], []
    for img_path in images:
        iid = file_to_id.get(img_path.name)
        if iid and iid in caption_map:
            all_imgs.append(img_path)
            all_caps.append(caption_map[iid][0])  # first caption for training
            all_refs.append(caption_map[iid])      # all refs for BLEU
            if len(all_imgs) >= max_samples:
                break

    # ── Split into train / val / test ────────────────────────
    split_counts = {"train": 0, "val": 0, "test": 0}
    train_imgs, train_caps = [], []
    val_imgs, val_caps = [], []
    test_imgs, test_caps, test_refs = [], [], []

    if len(all_imgs) >= 6:
        file_strs = [str(p) for p in all_imgs]
        sp = make_file_splits(file_strs, seed=args.seed, test_size=0.15, val_size=0.15)
        train_set = set(sp.train)
        val_set = set(sp.val)
        test_set = set(sp.test)

        for img_p, cap, refs in zip(all_imgs, all_caps, all_refs):
            s = str(img_p)
            if s in train_set:
                train_imgs.append(img_p); train_caps.append(cap)
            elif s in val_set:
                val_imgs.append(img_p); val_caps.append(cap)
            else:
                test_imgs.append(img_p); test_caps.append(cap); test_refs.append(refs)

        split_counts = {"train": len(train_imgs), "val": len(val_imgs), "test": len(test_imgs)}
        logger.info("Split: %d train / %d val / %d test image-caption pairs",
                     split_counts["train"], split_counts["val"], split_counts["test"])
    elif all_imgs:
        # Too few for split — use all for training, no test eval
        train_imgs, train_caps = all_imgs, all_caps
        split_counts = {"train": len(train_imgs), "val": 0, "test": 0}
        logger.warning("Too few matched pairs (%d) for train/val/test split", len(all_imgs))

    # ── Fine-tune on train subset (if annotations available) ─
    ft_losses: list[float] = []
    ft_val_losses: list[float] = []
    if len(train_imgs) >= batch_size:
        logger.info("Fine-tuning on %d image-caption pairs …", len(train_imgs))
        ft_losses, ft_val_losses = finetune(
            model, feature_extractor, tokenizer,
            train_imgs, train_caps, device, output_dir,
            epochs=epochs, batch_size=batch_size,
            use_amp_override=use_amp,
            val_image_paths=val_imgs if val_imgs else None,
            val_captions=val_caps if val_caps else None,
            patience=patience,
        )

    # ── Inference demo ───────────────────────────────────────
    demo_imgs = images[:NUM_DEMO_IMAGES] if images else []

    if not demo_imgs:
        logger.warning("No images found — generating demo with random noise images.")
        ensure_dir(output_dir / "demo_images")
        for i in range(NUM_DEMO_IMAGES):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            p = output_dir / "demo_images" / f"random_{i}.png"
            Image.fromarray(arr).save(p)
            demo_imgs.append(p)

    gen_captions = []
    for img_p in demo_imgs:
        cap = generate_caption(model, feature_extractor, tokenizer, img_p, device)
        gen_captions.append(cap)
        logger.info("  %s → %s", img_p.name, cap)

    save_captioned_images(demo_imgs, gen_captions, output_dir)

    # ── BLEU evaluation on TEST split only ───────────────────
    bleu_result = {"bleu": None}
    if test_imgs and test_refs:
        eval_preds = []
        for img_p in test_imgs:
            cap = generate_caption(model, feature_extractor, tokenizer, img_p, device)
            eval_preds.append(cap)

        bleu_result = compute_bleu(eval_preds, test_refs)
        logger.info("BLEU score (test set, n=%d): %s", len(eval_preds), bleu_result.get("bleu"))
    elif caption_map:
        # Fallback: evaluate on demo images
        eval_preds, eval_refs = [], []
        for img_p, cap in zip(demo_imgs, gen_captions):
            iid = file_to_id.get(img_p.name)
            if iid and iid in caption_map:
                eval_preds.append(cap)
                eval_refs.append(caption_map[iid])

        if eval_preds:
            bleu_result = compute_bleu(eval_preds, eval_refs)
            logger.info("BLEU score (demo fallback): %s", bleu_result.get("bleu"))

    # ── Write split manifest ─────────────────────────────────
    ds_fp = dataset_fingerprint(ds_path)
    write_split_manifest(
        output_dir,
        dataset_fp=ds_fp,
        split_method="make_file_splits 70/15/15",
        seed=args.seed,
        counts=split_counts,
    )

    # ── Save metrics ─────────────────────────────────────────
    meta = run_metadata(args)
    val_loss = ft_val_losses[-1] if ft_val_losses else (ft_losses[-1] if ft_losses else None)
    metrics = {
        "dataset": f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}",
        "pretrained_model": PRETRAINED_MODEL,
        "images_found": len(images),
        "annotations_loaded": len(caption_map),
        "finetune_epochs": epochs if ft_losses else 0,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "demo_images_captioned": len(gen_captions),
        "bleu": bleu_result.get("bleu"),
        "split": "test",
        "run_metadata": meta,
    }
    save_metrics(output_dir, metrics, task_type="classification", mode=args.mode)

    captions_out = [{"image": str(p.name), "caption": c}
                    for p, c in zip(demo_imgs, gen_captions)]
    (output_dir / "captions.json").write_text(json.dumps(captions_out, indent=2))

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
