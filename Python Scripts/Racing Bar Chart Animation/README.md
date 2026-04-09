# Racing Bar Chart Animation

> A Jupyter Notebook that creates animated bar chart race visualizations of COVID-19 data across Indian states using matplotlib.

## Overview

This project fetches COVID-19 time-series data from the `api.covid19india.org` API and creates an animated horizontal bar chart race showing state-wise progression of selected metrics (deceased, confirmed, recovered, tested) over time. The animation is rendered using matplotlib's `FuncAnimation` and can be exported as an HTML5 video or MP4 file.

## Features

- Fetches live COVID-19 time-series data from `api.covid19india.org` API
- Supports multiple data types: confirmed, deceased, recovered, tested (configurable)
- Supports `total` and `delta` data views
- Creates animated horizontal bar chart race with smooth interpolation
- Ranks states dynamically at each time step
- Date labels displayed in the chart title
- Customizable chart styling with `nice_axes()` helper function
- Can export animation as HTML5 video or MP4 file

## Project Structure

```
racing_barchart_animation/
├── animated_barchart.ipynb    # Jupyter Notebook with full pipeline
├── requirements.txt           # Python dependencies
├── images/
│   └── deceased.gif           # Sample output animation
└── README.md
```

## Requirements

- Python 3.x
- `jupyterlab==2.2.2`
- `matplotlib==3.3.0`
- `notebook==6.1.1`
- `numpy==1.19.1`
- `pandas==1.1.0`

All dependencies are listed in `requirements.txt`. Note: `requirements.txt` also lists `requests==2.24.0`, but the notebook does not use `requests` — data is fetched via `pd.read_json()` directly.

## Installation

```bash
cd "racing_barchart_animation"
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter:

```bash
jupyter lab
```

2. Open `animated_barchart.ipynb`
3. Run all cells sequentially
4. To change the metric being visualized, modify the `data_selected` variable:

```python
data_selected = 'deceased'  # Options: 'confirmed', 'deceased', 'recovered', 'tested'
```

5. To replace `total` with `delta`, change the data access key in the `obtain_data_for_a_date()` function.
6. To save the animation, uncomment the last code cell:

```python
anim.save('~/Downloads/covid19.mp4')
```

## How It Works

1. **Data Fetching**: Reads JSON time-series data from `https://api.covid19india.org/v4/timeseries.json` into a pandas DataFrame using `pd.read_json()`.
2. **Data Processing**: Extracts the selected metric for each state at each date using `obtain_data_for_a_date()`, building a new DataFrame.
3. **Interpolation**: The data is expanded by a factor of 5 (configurable `steps` parameter), with forward-filled dates and interpolated values for smooth animation.
4. **Ranking**: States are ranked at each interpolated time step using `df.rank(axis=1, method='first')`.
5. **Animation**: `FuncAnimation` iterates through each frame, updating horizontal bar positions and widths. The `init()` function clears the axes, and `update(i)` draws bars for frame `i`.
6. **Rendering**: The animation is converted to HTML5 video using `anim.to_html5_video()` for inline display.

## Configuration

- `data_selected` — the COVID metric to visualize (`'deceased'`, `'confirmed'`, `'recovered'`, `'tested'`)
- `steps=5` in `prepare_data()` — interpolation factor for smoothness
- `interval=100` in `FuncAnimation` — milliseconds between frames
- `figsize=(5, 5)` and `dpi=300` — figure dimensions and resolution
- `colors = plt.cm.Dark2(range(6))` — color map for bars

## Limitations

- The API (`api.covid19india.org`) may no longer be active or may have changed its format
- Uses `fillna(method='ffill')` which is deprecated in newer pandas versions
- Missing data for states is filled with 0 (may skew visualizations)
- Bare `except` clause in `obtain_data_for_a_date()` silently catches all errors
- Only visualizes Indian state-level COVID data
- The `prepare_data()` function references `df3` from outer scope instead of using the `df` parameter

## Security Notes

No security concerns identified.

## License

Not specified.
