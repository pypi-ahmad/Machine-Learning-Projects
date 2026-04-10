"""
Modern Image Captioning / VLM Pipeline (April 2026)

Primary    : Qwen3-VL-2B-Instruct (vision-language, bfloat16, auto device).
Alternative: Molmo-7B-D-0924 (AllenAI multimodal LLM, bfloat16).
Timing     : Wall-clock per model.
Export     : metrics.json with caption counts + avg length + timing;
             captions.json with per-image captions from each model.
Data       : Auto-downloaded at runtime.
"""
import os, json, time, warnings
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MAX_SAMPLES = 20
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("nlphuji/flickr30k", split="train").to_pandas()
    return df


def caption_images():
    df = load_data()
    run_eda(df, SAVE_DIR)
    img_col = next((c for c in df.column_names if "image" in c.lower()), df.column_names[0])
    images = [df[i][img_col] for i in range(min(MAX_SAMPLES, len(df)))]
    metrics = {}
    all_captions = {}

    # -- PRIMARY: Qwen3-VL --
    print()
    print("-- Qwen3-VL-2B-Instruct --")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        t0 = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        captions = []
        for idx, img in enumerate(images):
            pil_img = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": "Describe this image in detail."}
            ]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            vis_inp = process_vision_info(msgs)
            inputs = processor(text=[text], images=vis_inp[0], return_tensors="pt").to(model.device)
            out_ids = model.generate(**inputs, max_new_tokens=128)
            caption = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True)[0]
            captions.append(caption)
            print(f"  [{idx+1}/{len(images)}] {caption[:100]}...")
        elapsed = round(time.perf_counter() - t0, 1)
        avg_len = sum(len(c) for c in captions) / max(len(captions), 1)
        print(f"  Qwen3-VL: {len(captions)} captions, avg {avg_len:.0f} chars ({elapsed}s)")
        metrics["Qwen3-VL"] = {"captions": len(captions), "avg_length": round(avg_len, 1), "time_s": elapsed}
        all_captions["Qwen3-VL"] = captions
    except Exception as e:
        print(f"  Qwen3-VL failed: {e}")

    # -- ALTERNATIVE: Molmo 2 --
    print()
    print("-- Molmo-7B-D-0924 --")
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor as AP2
        t1 = time.perf_counter()
        molmo = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924",
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        molmo_proc = AP2.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True)
        molmo_captions = []
        for idx, img in enumerate(images[:10]):
            pil_img = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            inputs = molmo_proc.process(images=[pil_img], text="Describe this image in detail.")
            inputs = {k: v.to(molmo.device).unsqueeze(0) if hasattr(v, "to") else v for k, v in inputs.items()}
            out = molmo.generate_from_batch(inputs, max_new_tokens=128, tokenizer=molmo_proc.tokenizer)
            caption = molmo_proc.tokenizer.decode(out[0], skip_special_tokens=True)
            molmo_captions.append(caption)
            print(f"  [{idx+1}/{min(10, len(images))}] {caption[:100]}...")
        mol_elapsed = round(time.perf_counter() - t1, 1)
        mol_avg = sum(len(c) for c in molmo_captions) / max(len(molmo_captions), 1)
        print(f"  Molmo-2: {len(molmo_captions)} captions, avg {mol_avg:.0f} chars ({mol_elapsed}s)")
        metrics["Molmo-2"] = {"captions": len(molmo_captions), "avg_length": round(mol_avg, 1), "time_s": mol_elapsed}
        all_captions["Molmo-2"] = molmo_captions
    except Exception as e:
        print(f"  Molmo-2 failed: {e}")

    # Save captions
    cap_path = os.path.join(SAVE_DIR, "captions.json")
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)
    print(f"Captions saved to {cap_path}")

    validation = validate_results(all_captions, len(images), SAVE_DIR)
    metrics["validation"] = validation

    return metrics


def run_eda(df, save_dir):
    """Input data summary for captioning."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    n_rows = len(df)
    columns = list(getattr(df, "column_names", []))
    print(f"  Samples: {n_rows}")
    if columns:
        print(f"  Columns: {columns}")
    summary = {"samples": n_rows, "columns": columns}
    with open(os.path.join(save_dir, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("EDA complete.")


def validate_results(all_captions, expected_images, save_dir):
    """Validate caption outputs for completeness and diversity."""
    validation = {"expected_images": expected_images, "models": {}}
    for model_name, captions in all_captions.items():
        clean = [c.strip() for c in captions if isinstance(c, str) and c.strip()]
        validation["models"][model_name] = {
            "captions": len(captions),
            "non_empty": len(clean),
            "coverage_ratio": round(len(clean) / max(expected_images, 1), 4),
            "unique_ratio": round(len(set(clean)) / max(len(clean), 1), 4),
            "avg_chars": round(sum(len(c) for c in clean) / max(len(clean), 1), 1),
            "passed": bool(clean),
        }
    validation["passed"] = any(model.get("passed") for model in validation["models"].values())
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation saved to {out_path}")
    return validation


def main():
    print("=" * 60)
    print("IMAGE CAPTIONING / VLM | Qwen3-VL + Molmo 2")
    print("=" * 60)
    metrics = caption_images()

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
