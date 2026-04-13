"""OCR Tool — Streamlit ML demo.

Extract text from images using Tesseract OCR (pytesseract) if available,
with a graceful fallback to pixel-based brightness detection for demonstration.

Usage:
    streamlit run main.py
"""

import io
import re
import struct
import zlib

import streamlit as st

st.set_page_config(page_title="OCR Tool", layout="wide")
st.title("🔍 OCR — Optical Character Recognition")
st.caption("Extract text from images. Uses Tesseract (pytesseract) when installed.")


# ── Tesseract wrapper ─────────────────────────────────────────────────────────

def try_tesseract(image_bytes: bytes, lang: str = "eng") -> str | None:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img, lang=lang)
    except ImportError:
        return None
    except Exception as e:
        return f"[Tesseract error: {e}]"


def tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ── Pure-Python PNG brightness grid (demo fallback) ──────────────────────────

def read_png_brightness(data: bytes, grid_w: int = 60, grid_h: int = 20) -> list[list[float]] | None:
    """Down-sample PNG to a brightness grid for ASCII visualization."""
    try:
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        pos = 8
        idat = b""
        width = height = 0
        color_type = bit_depth = 0
        while pos < len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            chunk  = data[pos + 4:pos + 8]
            cdata  = data[pos + 8:pos + 8 + length]
            pos   += 12 + length
            if chunk == b"IHDR":
                width, height = struct.unpack(">II", cdata[:8])
                bit_depth     = cdata[8]
                color_type    = cdata[9]
            elif chunk == b"IDAT":
                idat += cdata
            elif chunk == b"IEND":
                break
        if bit_depth != 8 or color_type not in (2, 6):
            return None
        raw      = zlib.decompress(idat)
        channels = 4 if color_type == 6 else 3
        stride   = width * channels + 1

        grid = []
        for gy in range(grid_h):
            row_pix = []
            for gx in range(grid_w):
                px = int(gx / grid_w * width)
                py = int(gy / grid_h * height)
                off = py * stride + 1 + px * channels
                r, g, b = raw[off], raw[off + 1], raw[off + 2]
                row_pix.append((r * 0.299 + g * 0.587 + b * 0.114) / 255)
            grid.append(row_pix)
        return grid
    except Exception:
        return None


def brightness_to_ascii(grid: list[list[float]]) -> str:
    chars = " .,:;i1tfLCG08@#"
    lines = []
    for row in grid:
        line = ""
        for v in row:
            idx   = int(v * (len(chars) - 1))
            line += chars[idx]
        lines.append(line)
    return "\n".join(lines)


# ── Text post-processing ──────────────────────────────────────────────────────

def clean_ocr(text: str) -> str:
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned)


def extract_stats(text: str) -> dict:
    words     = re.findall(r"[a-zA-Z]+", text)
    numbers   = re.findall(r"\d+(?:\.\d+)?", text)
    emails    = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    urls      = re.findall(r"https?://\S+", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return {
        "words":     len(words),
        "chars":     len(text),
        "numbers":   numbers,
        "emails":    emails,
        "urls":      urls,
        "sentences": len(sentences),
    }


# ── Sample images (base64-free synthetic ASCII art as placeholder) ─────────────

SAMPLE_TEXT = {
    "Invoice": """INVOICE #1042
Date: 2024-03-15
Bill To: John Smith
123 Main Street

Item             Qty   Price
Web Design        1   $500.00
Hosting (1 yr)    1   $120.00
Support           5    $50.00

Subtotal: $670.00
Tax (10%):  $67.00
TOTAL:     $737.00

Payment Due: 2024-04-15
Email: billing@example.com""",
    "Article Snippet": """The Rise of AI in Healthcare

Artificial intelligence is transforming medical
diagnosis and patient care. Machine learning
algorithms can now detect cancers from images
with accuracy rivaling specialist physicians.

Key applications include:
- Radiology image analysis
- Drug discovery acceleration
- Personalized treatment plans
- Predictive patient monitoring

Companies investing in health AI exceeded
$14 billion in funding during 2023.""",
    "Product Label": """PRODUCT: UltraBoost Pro X9
Batch: 2024-B112
NET WEIGHT: 500g

Ingredients: Water, Protein Isolate (25g),
Natural Flavors, Stevia, Vitamin B12 (2.4mcg)

Directions: Mix 1 scoop with 250ml water.
Shake well. Consume within 30 minutes.

Store below 25°C. Keep dry.
Manufactured by SupplyCo Inc.
support@supplyco.com
www.supplyco.com""",
}


# ── UI ────────────────────────────────────────────────────────────────────────

tess_ok = tesseract_available()
if tess_ok:
    st.success("Tesseract OCR is installed and ready.")
else:
    st.warning("Tesseract / pytesseract not found. Running in demo mode with sample text.")

tab1, tab2, tab3 = st.tabs(["Extract Text", "Demo Mode", "About"])

with tab1:
    if not tess_ok:
        st.info("Upload is available when Tesseract is installed. Use Demo Mode below.")
    else:
        uploaded = st.file_uploader("Upload image (PNG, JPEG, BMP, TIFF)", type=["png", "jpg", "jpeg", "bmp", "tiff"])
        lang     = st.selectbox("OCR Language", ["eng", "fra", "deu", "spa", "ita", "por"], index=0)
        enhance  = st.checkbox("Clean output (remove blank lines)", value=True)

        if uploaded:
            img_bytes = uploaded.read()
            st.image(img_bytes, caption=uploaded.name, use_container_width=False, width=400)

            if st.button("🔍 Extract Text", type="primary"):
                with st.spinner("Running OCR..."):
                    text = try_tesseract(img_bytes, lang)
                if text is None:
                    st.error("Tesseract is not available.")
                elif text.startswith("[Tesseract error"):
                    st.error(text)
                else:
                    result = clean_ocr(text) if enhance else text
                    st.subheader("Extracted Text")
                    st.text_area("OCR Result", result, height=300)

                    # PNG brightness visualization
                    if uploaded.name.lower().endswith(".png"):
                        grid = read_png_brightness(img_bytes)
                        if grid:
                            with st.expander("ASCII brightness preview"):
                                st.code(brightness_to_ascii(grid), language=None)

                    stats = extract_stats(result)
                    st.subheader("Text Statistics")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Words",      stats["words"])
                    c2.metric("Characters", stats["chars"])
                    c3.metric("Sentences",  stats["sentences"])
                    if stats["emails"]:
                        st.info("Emails found: " + ", ".join(stats["emails"]))
                    if stats["urls"]:
                        st.info("URLs found: " + ", ".join(stats["urls"]))
                    if stats["numbers"]:
                        st.caption("Numbers: " + ", ".join(stats["numbers"][:10]))

with tab2:
    st.subheader("Demo — Sample OCR Output")
    sample_name = st.selectbox("Choose sample", list(SAMPLE_TEXT.keys()))
    raw_sample  = SAMPLE_TEXT[sample_name]
    st.text_area("Simulated OCR Output", raw_sample, height=250)

    stats = extract_stats(raw_sample)
    c1, c2, c3 = st.columns(3)
    c1.metric("Words",      stats["words"])
    c2.metric("Characters", stats["chars"])
    c3.metric("Sentences",  stats["sentences"])
    if stats["numbers"]:
        st.caption("Numbers detected: " + ", ".join(stats["numbers"][:10]))
    if stats["emails"]:
        st.info("Email addresses: " + ", ".join(stats["emails"]))
    if stats["urls"]:
        st.info("URLs: " + ", ".join(stats["urls"]))

with tab3:
    st.markdown("""
    ### Setup Instructions

    **Install Tesseract OCR:**
    - **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
    - **macOS:** `brew install tesseract`
    - **Windows:** Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

    **Install Python bindings:**
    ```
    pip install pytesseract Pillow
    ```

    **Additional languages:**
    ```
    sudo apt install tesseract-ocr-deu tesseract-ocr-fra  # example
    ```

    ### Supported Input Formats
    PNG, JPEG, BMP, TIFF — any format Pillow can read.

    ### Text Post-Processing
    After extraction, the tool:
    - Removes blank lines
    - Detects emails, URLs, and numbers
    - Reports word and character counts

    ### Accuracy Tips
    - Use high-resolution images (≥ 300 DPI).
    - Ensure good contrast between text and background.
    - Avoid skewed or curved text.
    - Use language-specific models for non-English text.
    """)
