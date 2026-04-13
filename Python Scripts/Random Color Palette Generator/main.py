"""Random Color Palette Generator — Streamlit app.

Generate beautiful random color palettes, complementary colors,
analogous schemes, and triadic combinations.  Copy hex codes
and preview swatches.

Usage:
    streamlit run main.py
"""

import colorsys
import random

import streamlit as st

st.set_page_config(page_title="Color Palette Generator", layout="wide")
st.title("🎨 Color Palette Generator")


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def random_hex() -> str:
    return "#{:06X}".format(random.randint(0, 0xFFFFFF))


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def hex_to_hsl(h: str) -> tuple[float, float, float]:
    r, g, b = [x / 255 for x in hex_to_rgb(h)]
    return colorsys.rgb_to_hls(r, g, b)  # returns (H, L, S)


def hsl_to_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return rgb_to_hex(int(r * 255), int(g * 255), int(b * 255))


def complementary(hex_color: str) -> str:
    h, l, s = hex_to_hsl(hex_color)
    return hsl_to_hex((h + 0.5) % 1.0, l, s)


def analogous(hex_color: str, steps: int = 5, spread: float = 0.08) -> list[str]:
    h, l, s = hex_to_hsl(hex_color)
    offset   = -(steps // 2) * spread
    return [hsl_to_hex((h + offset + i * spread) % 1.0, l, s) for i in range(steps)]


def triadic(hex_color: str) -> list[str]:
    h, l, s = hex_to_hsl(hex_color)
    return [hsl_to_hex((h + i / 3) % 1.0, l, s) for i in range(3)]


def split_complementary(hex_color: str) -> list[str]:
    h, l, s = hex_to_hsl(hex_color)
    return [
        hex_color,
        hsl_to_hex((h + 0.5 - 0.083) % 1.0, l, s),
        hsl_to_hex((h + 0.5 + 0.083) % 1.0, l, s),
    ]


def tetradic(hex_color: str) -> list[str]:
    h, l, s = hex_to_hsl(hex_color)
    return [hsl_to_hex((h + i * 0.25) % 1.0, l, s) for i in range(4)]


def monochromatic(hex_color: str, n: int = 5) -> list[str]:
    h, _, s = hex_to_hsl(hex_color)
    return [hsl_to_hex(h, 0.2 + i * 0.12, s) for i in range(n)]


def text_color(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 128 else "#FFFFFF"


def swatch_html(colors: list[str], size: int = 80) -> str:
    parts = []
    for c in colors:
        tc = text_color(c)
        parts.append(
            f'<div style="display:inline-block;width:{size}px;height:{size}px;'
            f'background:{c};border-radius:8px;margin:4px;text-align:center;'
            f'line-height:{size}px;color:{tc};font-size:11px;font-weight:bold">'
            f'{c}</div>'
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

mode = st.sidebar.radio("Generation mode", [
    "Random palette", "Complementary", "Analogous",
    "Triadic", "Split-complementary", "Tetradic", "Monochromatic",
])

n_colors = st.sidebar.slider("Number of colors", 3, 10, 5)
seed_color = st.sidebar.color_picker("Base color", "#3A86FF")

if st.sidebar.button("🎲 Regenerate") or "palette" not in st.session_state:
    st.session_state.palette = [random_hex() for _ in range(n_colors)]

# ---------------------------------------------------------------------------
# Generate palette based on mode
# ---------------------------------------------------------------------------
if mode == "Random palette":
    if st.button("🎲 New Random Palette"):
        st.session_state.palette = [random_hex() for _ in range(n_colors)]
    palette = st.session_state.palette

elif mode == "Complementary":
    palette = [seed_color, complementary(seed_color)]

elif mode == "Analogous":
    palette = analogous(seed_color, n_colors)

elif mode == "Triadic":
    palette = triadic(seed_color)

elif mode == "Split-complementary":
    palette = split_complementary(seed_color)

elif mode == "Tetradic":
    palette = tetradic(seed_color)

else:  # Monochromatic
    palette = monochromatic(seed_color, n_colors)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
st.subheader(f"{mode}  ({len(palette)} colors)")
st.markdown(swatch_html(palette, size=100), unsafe_allow_html=True)

st.divider()
col_data = []
for c in palette:
    r, g, b = hex_to_rgb(c)
    h, l, s = hex_to_hsl(c)
    col_data.append({
        "HEX": c,
        "R": r, "G": g, "B": b,
        "H°": round(h * 360, 1),
        "S%": round(s * 100, 1),
        "L%": round(l * 100, 1),
    })

import pandas as pd
st.dataframe(pd.DataFrame(col_data), use_container_width=True)

# Hex codes for copying
st.subheader("Copy Hex Codes")
st.code(", ".join(palette))

# Individual pickers
st.subheader("Individual Colors")
cols = st.columns(len(palette))
for i, (col_widget, c) in enumerate(zip(cols, palette)):
    col_widget.color_picker(f"Color {i+1}", c, key=f"picker_{i}")
