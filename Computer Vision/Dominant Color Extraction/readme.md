# Dominant Color Extraction

> A Jupyter Notebook project that extracts and visualizes dominant colors from images using K-Means clustering.

## Overview

This project extracts the dominant colors from an image using the K-Means clustering algorithm from scikit-learn. It reads an image with OpenCV, reshapes pixel data, clusters the pixels into groups, and visualizes both the dominant color swatches and a reconstructed image using only those dominant colors.

## Features

- Load and display images using OpenCV and Matplotlib
- Extract dominant colors via K-Means clustering (configurable number of clusters)
- Visualize individual dominant color swatches as solid color blocks
- Reconstruct the image using only the clustered dominant colors
- Display a color distribution bar chart showing the relative proportion of each dominant color
- BGR-to-RGB color space conversion for accurate color representation

## Project Structure

```
Dominant-Color-Extraction/
├── Dominant color Extraction.ipynb   # Main Jupyter Notebook with all code and visualizations
├── requirements.txt                  # Python dependencies
└── readme.md
```

## Requirements

- Python 3.8+
- jupyter 4.7.0
- notebook 6.1.4
- numpy
- opencv-python 4.4.0.46
- matplotlib 3.3.3
- scikit-learn 0.23.2

## Installation

```bash
cd "Dominant-Color-Extraction"
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook "Dominant color Extraction.ipynb"
```

1. Place your target image in an `images/` subdirectory (the notebook expects `images/1-Saint-Basils-Cathedral.jpg` by default)
2. Run the cells sequentially to see each step of the extraction process
3. Adjust the value of `k` (number of clusters) to control how many dominant colors are extracted

## How It Works

1. **Image Loading**: The notebook reads an image using `cv2.imread()` and displays it with Matplotlib.
2. **Pixel Reshaping**: The image's 3D pixel array (height × width × 3 channels) is reshaped into a 2D array of shape `(num_pixels, 3)`.
3. **K-Means Clustering**: `KMeans(n_clusters=k)` from scikit-learn is fitted on the reshaped pixel data. Each cluster center represents a dominant color.
4. **Color Swatch Visualization**: Each cluster center is rendered as a 100×100 solid color block using Matplotlib.
5. **Image Reconstruction**: A new image is created where each pixel is replaced by its nearest cluster center color, producing a posterized version of the original.
6. **Color Bar Chart**: A `plot_colors()` function creates a horizontal bar showing the proportion of each dominant color based on the histogram of cluster label frequencies.
7. **BGR to RGB**: The notebook also demonstrates converting the image from BGR (OpenCV default) to RGB for correct color display, then re-runs the clustering.

## Configuration

- **Number of clusters (`k`)**: Set to `9` by default; change this value to extract more or fewer dominant colors
- **Image path**: Hardcoded as `"images/1-Saint-Basils-Cathedral.jpg"` — update to point to your own image
- **Image reshape dimensions**: The notebook uses `(600, 394, 3)` for reshaping the reconstructed image back; adjust to match your image dimensions

## Limitations

- The image path is hardcoded; you must manually update it for different images
- The reshape dimensions `(600, 394, 3)` are hardcoded for the sample image and will fail for images of different sizes
- K-Means results can vary between runs due to random initialization
- No automated detection of optimal number of clusters
- Large images may be slow to process due to pixel-level iteration in Python loops
- The `dtype="uint"` used in several places may behave differently across NumPy versions (should ideally be `"uint8"`)

## Security Notes

- No external network calls or user authentication involved
- The notebook only reads local image files from disk

## License

Not specified.
