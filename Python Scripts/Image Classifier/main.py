"""Image Classifier — Streamlit ML demo.

Classify images using a rule-based color and brightness analysis
pipeline (no external ML frameworks required). Also supports
optional classification via the Hugging Face Inference API
if an API key is provided.

Usage:
    streamlit run main.py
"""

import io
import math

import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="Image Classifier", layout="wide")
st.title("🖼️ Image Classifier")
st.caption("Local color and brightness analysis, with an optional Hugging Face model call.")


# ── Image pixel reader ────────────────────────────────────────────────────────

def read_image_pixels(data: bytes) -> tuple[list[tuple[int, int, int, int]], int, int] | None:
    """Return RGBA pixels and dimensions for a supported uploaded image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        rgba = image.convert("RGBA")
        return list(rgba.getdata()), rgba.width, rgba.height
    except (OSError, UnidentifiedImageError):
        return None


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyze_pixels(pixels: list, width: int, height: int) -> dict:
    n      = len(pixels)
    if n == 0:
        return {}
    total_r = sum(p[0] for p in pixels)
    total_g = sum(p[1] for p in pixels)
    total_b = sum(p[2] for p in pixels)
    avg_r, avg_g, avg_b = total_r / n, total_g / n, total_b / n
    brightness = (avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114)

    # Saturation estimate
    maxc = [max(p[0], p[1], p[2]) for p in pixels]
    minc = [min(p[0], p[1], p[2]) for p in pixels]
    sat  = sum((mx - mn) / (mx + 1e-9) for mx, mn in zip(maxc, minc)) / n

    # Dominant hue bucket
    hue_buckets = {"red": 0, "green": 0, "blue": 0, "neutral": 0}
    for r, g, b, _ in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx - mn < 30:
            hue_buckets["neutral"] += 1
        elif mx == r:
            hue_buckets["red"] += 1
        elif mx == g:
            hue_buckets["green"] += 1
        else:
            hue_buckets["blue"] += 1

    dominant_hue = max(hue_buckets, key=hue_buckets.get)
    hue_pcts     = {k: v / n for k, v in hue_buckets.items()}

    # Variance (image complexity estimate)
    var = sum((p[0] - avg_r) ** 2 + (p[1] - avg_g) ** 2 + (p[2] - avg_b) ** 2
              for p in pixels) / n
    std = math.sqrt(var / 3)

    return {
        "avg_r": avg_r, "avg_g": avg_g, "avg_b": avg_b,
        "brightness": brightness, "saturation": sat,
        "dominant_hue": dominant_hue, "hue_pcts": hue_pcts,
        "std": std, "width": width, "height": height, "n_pixels": n,
    }


def rule_classify(stats: dict) -> list[tuple[str, float]]:
    """Return sorted list of (label, confidence) using rule-based heuristics."""
    b   = stats["brightness"]
    sat = stats["saturation"]
    std = stats["std"]
    hue = stats["dominant_hue"]
    hp  = stats["hue_pcts"]
    w   = stats["width"]
    h   = stats["height"]

    candidates = []

    # Outdoor / nature: high green, moderate brightness
    nature_score = hp["green"] * 2 + (0.3 < b / 255 < 0.8) * 0.5 + (sat > 0.3) * 0.3
    candidates.append(("Nature / Outdoors", nature_score))

    # Sky: high blue, high brightness
    sky_score = hp["blue"] * 1.5 + (b > 150) * 0.5 + (sat > 0.2) * 0.3
    candidates.append(("Sky / Water", sky_score))

    # Portrait / person: skin tone (red-dominant, medium brightness)
    skin_score = hp["red"] * 1.2 + (100 < b < 200) * 0.4 + (sat < 0.4) * 0.2
    candidates.append(("Portrait / Person", skin_score))

    # Night / dark: low brightness
    dark_score = (b < 80) * 1.5 + (std > 40) * 0.4
    candidates.append(("Night / Dark Scene", dark_score))

    # Document / text: very high brightness, very low saturation
    doc_score  = (b > 200) * 1.2 + (sat < 0.1) * 1.0 + (std < 30) * 0.5
    candidates.append(("Document / Text", doc_score))

    # Abstract / colorful: high saturation, high variance
    art_score  = (sat > 0.5) * 1.2 + (std > 60) * 0.8
    candidates.append(("Abstract / Colorful", art_score))

    total = sum(s for _, s in candidates) or 1
    normed = [(label, score / total) for label, score in candidates]
    normed.sort(key=lambda x: -x[1])
    return normed


# ── Optional HF Inference API ─────────────────────────────────────────────────

def hf_classify(image_bytes: bytes, api_key: str) -> list[dict] | None:
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api-inference.huggingface.co/models/google/vit-base-patch16-224",
            data=image_bytes,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json
            return json.loads(resp.read())
    except Exception:
        return None


# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Classify Image", "About"])

with tab1:
    uploaded = st.file_uploader("Upload an image (PNG or JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption=uploaded.name, use_container_width=False, width=400)

        st.subheader("Rule-Based Analysis")
        parsed = None
        parsed = read_image_pixels(img_bytes)

        if parsed:
            pixels, width, height = parsed
            stats = analyze_pixels(pixels, width, height)
            labels = rule_classify(stats)

            c1, c2, c3 = st.columns(3)
            c1.metric("Brightness",   f"{stats['brightness']:.0f}/255")
            c2.metric("Saturation",   f"{stats['saturation']:.2f}")
            c3.metric("Pixel Std",    f"{stats['std']:.1f}")

            st.markdown("**Dominant hue:**  " + stats["dominant_hue"].capitalize())
            hue_df_data = {k: round(v * 100, 1) for k, v in stats["hue_pcts"].items()}
            st.caption("Hue distribution (%): " +
                       "  |  ".join(f"{k}: {v:.1f}%" for k, v in hue_df_data.items()))

            st.subheader("Classification Results")
            top = labels[0]
            st.success(f"**Top prediction:** {top[0]}  ({top[1]:.1%} confidence)")
            for label, conf in labels[1:3]:
                st.write(f"• {label} — {conf:.1%}")
        else:
            st.error("The uploaded file could not be decoded as a PNG or JPEG image.")

        st.subheader("Hugging Face Inference API (Optional)")
        st.caption("Provides real deep-learning classification using ViT (Vision Transformer).")
        api_key = st.text_input("HF API Token (hf_...)", type="password")
        if api_key and st.button("🤖 Classify with ViT"):
            with st.spinner("Calling Hugging Face API..."):
                results = hf_classify(img_bytes, api_key)
            if results:
                st.subheader("ViT Classification")
                for r in results[:5]:
                    st.write(f"• **{r['label']}** — {r['score']:.2%}")
            else:
                st.error("API call failed. Check your token and try again.")
    else:
        st.info("Upload a PNG or JPEG image to begin.")

with tab2:
    st.markdown("""
    ### How Rule-Based Classification Works
    1. **Read pixels** from the uploaded PNG or JPEG image.
    2. **Compute statistics**: average RGB, brightness, saturation, variance, hue distribution.
    3. **Apply rules**:
       - High green → Nature/Outdoors
       - High blue + bright → Sky/Water
       - Red-dominant, medium brightness → Portrait
       - Very dark (low brightness) → Night scene
       - Very bright + low saturation → Document/Text
       - High saturation + high variance → Abstract/Colorful
    4. **Normalize** rule scores to confidence values.

    ### HF Inference API
    - Uses Google's ViT-base-patch16-224 model trained on ImageNet-21k.
    - Requires a free Hugging Face account and API token.
    - Classifies into 1000 ImageNet categories with real deep-learning confidence.

    ### Limitations
    - Rule-based results are heuristic and approximate.
    - Deep-learning API requires internet access and a valid token.
    - The optional HF result leaves your computer only after you provide a token and click its button.
    """)
