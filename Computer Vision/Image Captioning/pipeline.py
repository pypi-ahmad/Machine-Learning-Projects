"""
Modern Image Captioning / VLM Pipeline (April 2026)
Models: Qwen3-VL (primary) + Molmo 2 (lightweight alternative)
Data: Auto-downloaded at runtime
"""
import os, warnings
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MAX_SAMPLES = 20


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("nlphuji/flickr30k", split="train").to_pandas()
    return df


def caption_images():
    df = load_data()
    # Auto-detect image column
    img_col = next((c for c in df.column_names if "image" in c.lower()), df.column_names[0])
    images = [df[i][img_col] for i in range(min(MAX_SAMPLES, len(df)))]

    # ═══ PRIMARY: Qwen3-VL ═══
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        captions = []
        for img in images:
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
            print(f"  Qwen3-VL: {caption[:100]}...")
        print(f"✓ Qwen3-VL captioned {len(captions)} images")
    except Exception as e:
        print(f"✗ Qwen3-VL: {e}")

    # ═══ ALTERNATIVE: Molmo 2 ═══
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor as AP2
        molmo = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924",
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        molmo_proc = AP2.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True)
        for img in images[:5]:
            pil_img = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            inputs = molmo_proc.process(images=[pil_img], text="Describe this image in detail.")
            inputs = {k: v.to(molmo.device).unsqueeze(0) if hasattr(v, "to") else v for k, v in inputs.items()}
            out = molmo.generate_from_batch(inputs, max_new_tokens=128, tokenizer=molmo_proc.tokenizer)
            caption = molmo_proc.tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  Molmo-2: {caption[:100]}...")
        print(f"✓ Molmo-2 captioned {min(5, len(images))} images")
    except Exception as e:
        print(f"✗ Molmo-2: {e}")


def main():
    print("=" * 60)
    print("IMAGE CAPTIONING / VLM — Qwen3-VL + Molmo 2")
    print("=" * 60)
    caption_images()


if __name__ == "__main__":
    main()
