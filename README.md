# Machine Learning Projects

A collection of **158 end-to-end machine learning projects** spanning classification, regression, clustering, NLP, computer vision, time series forecasting, and more. Each project lives in its own directory with a Jupyter notebook, dataset (where available), and auto-generated README.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Projects](https://img.shields.io/badge/Projects-158-brightgreen)]()
[![Tests](https://img.shields.io/badge/Tests-1%2C384%20passed-success)]()

---

## Quick Start

```bash
git clone https://github.com/pypi-ahmad/Machine-Learning-Projects.git
cd Machine-Learning-Projects
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Open any project folder and launch its notebook:

```bash
jupyter notebook
```

## Repository Stats

| Metric | Value |
|--------|-------|
| Total projects | 158 |
| Datasets available locally | 127 / 158 |
| Standardized with AutoML | 43 |
| Automated tests | 1,577 (1,384 passed, 193 skipped, 0 failed) |

## Project Categories

| Category | Count | Examples |
|----------|------:|---------|
| **Classification** | 48 | Spam Detection, Fraud Detection, Breast Cancer, Diabetes, Loan Prediction |
| **Regression** | 35 | House Price, Car Price, Flight Fare, Gold Price, IPL Score Prediction |
| **Exploratory Data Analysis** | 37 | Titanic, 911 Calls, Stock Market, Covid-19 Tracking |
| **Clustering** | 10 | Customer Segmentation, Mall Customers, K-Means Imagery |
| **Image Classification** | 8 | Traffic Signs, CIFAR-10, Fashion MNIST, Captcha Recognition |
| **Time Series** | 8 | ARIMA Forecasting, LSTM Power Consumption, Traffic Forecast |
| **NLP** | 9 | Sentiment Analysis, Spam Classifier, Disaster Tweets, Text Summarization |
| **Computer Vision** | 3 | Face Expression, Digit Recognition, Image Captioning |

## Standardized ML Pipeline

43 projects were enhanced with automated benchmarking:

- **[LazyPredict](https://github.com/shankarpandala/lazypredict)** — trains 20+ models in a single call (classification & regression)
- **[PyCaret](https://pycaret.org/)** — full AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> **Note:** PyCaret 3.x requires Python ≤ 3.11.

## Testing

Every project has an automated test validating data loading and model instantiation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests for a specific project (e.g., Project 1)
python -m pytest tests/test_p001_*.py -v
```

## Project Structure

```
Machine-Learning-Projects/
├── Machine Learning Project 1 - Adult Salary Prediction/
│   ├── *.ipynb          # Jupyter notebook(s)
│   ├── *.csv            # Dataset file(s)
│   └── README.md        # Auto-generated project README
├── Machine Learning Project 2 - Bitcoin Price Prediction/
│   └── ...
├── ...                  # 158 project directories
├── data/                # Centralized standardized datasets
├── tests/               # Automated test suite (158 test files)
├── audit_scripts/       # README generator & audit tooling
├── requirements.txt     # All Python dependencies
├── pytest.ini           # Test configuration
├── standardize_ml.py    # ML pipeline standardization script
├── WORKSPACE_OVERVIEW.md # Full project catalog with status table
└── README.md            # This file
```

## Full Project Catalog

See [WORKSPACE_OVERVIEW.md](WORKSPACE_OVERVIEW.md) for the complete table of all 158 projects with:
- ML task type
- Dataset availability status
- Standardization status
- Direct links to each project's README

## Key Libraries

| Library | Purpose |
|---------|---------|
| scikit-learn | Classical ML models & preprocessing |
| TensorFlow / Keras | Deep learning, CNNs, LSTMs |
| PyTorch | Deep learning, transfer learning |
| PyCaret | AutoML pipelines |
| LazyPredict | Quick multi-model benchmarking |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | Visualization |
| NLTK / spaCy | Natural language processing |
| OpenCV | Computer vision |
| statsmodels | Time series analysis |
| XGBoost / LightGBM | Gradient boosting |
| PySpark | Big data processing |

## License

This project is open source and available under the [MIT License](LICENSE).
