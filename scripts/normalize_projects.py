#!/usr/bin/env python3
"""
Project Normalizer — Deep Learning Projects Monorepo
=====================================================
Applies Phase 0 standardization across ALL projects:
- Adds standard setup header to all 2 - Code.py files
- Creates src/ module stubs for each project
- Creates README.md for each project  
- Applies code sanity fixes (duplicate imports, pathlib, logging)

Usage:
    python scripts/normalize_projects.py
"""

import os
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = [
    "Anomaly detection and fraud detection",
    "Associate Rule Learning",
    "Chat bot",
    "GANS",
    "Recommendation Systems",
    "Reinforcement Learning",
    "Speech and Audio processing",
]

# ── Standard Setup Header ───────────────────────────────────
STANDARD_HEADER = '''\
#!/usr/bin/env python3
"""
{project_name}
Category: {category}

Standardized with 2026 best practices:
- Reproducible seed
- CUDA-first device selection
- Pathlib-based paths
- Modular logging
"""

# ── Standard Setup ───────────────────────────────────────────
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# PyTorch CUDA setup (if available)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("Project  : {project_name}")
logger.info("Device   : %s", DEVICE)
logger.info("Data dir : %s", DATA_DIR)

# ── Original Code ────────────────────────────────────────────
'''


def discover_projects():
    """Find all project directories."""
    projects = []
    for category in CATEGORIES:
        cat_path = ROOT / category
        if not cat_path.is_dir():
            continue
        for entry in sorted(cat_path.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                projects.append({
                    "category": category,
                    "name": entry.name,
                    "path": entry,
                })
    return projects


def detect_problem_type(code: str, project_name: str) -> str:
    """Detect the ML problem type from code content."""
    name_lower = project_name.lower()
    code_lower = code.lower()

    if any(k in name_lower for k in ["anomaly", "fraud", "intrusion", "banknote"]):
        return "Classification (Anomaly/Fraud Detection)"
    if any(k in name_lower for k in ["breast cancer", "detection"]):
        return "Classification"
    if any(k in name_lower for k in ["recommendation", "recommend"]):
        return "Recommendation System"
    if any(k in name_lower for k in ["chatbot", "chat bot"]):
        return "NLP (Chatbot)"
    if any(k in name_lower for k in ["gan", "generation", "inpainting", "style transfer", "image-to-image", "text-to-image"]):
        return "Generative (GAN)"
    if any(k in name_lower for k in ["reinforcement", "taxi", "lunar", "gridworld", "cliff", "frozen"]):
        return "Reinforcement Learning"
    if any(k in name_lower for k in ["voice", "audio", "speech", "music", "caption"]):
        return "Speech/Audio Processing"
    if any(k in name_lower for k in ["association", "apriori", "grocery", "retail"]):
        return "Association Rule Learning"
    if any(k in name_lower for k in ["traffic", "prediction", "forecast"]):
        return "Regression (Time Series)"
    if "classif" in code_lower or "accuracy_score" in code_lower:
        return "Classification"
    if "regress" in code_lower or "mse" in code_lower:
        return "Regression"
    return "ML/DL"


def detect_framework(code: str) -> str:
    """Detect ML framework used."""
    frameworks = []
    if "import torch" in code or "from torch" in code:
        frameworks.append("PyTorch")
    if "pycaret" in code.lower():
        frameworks.append("PyCaret")
    if "import timm" in code.lower():
        frameworks.append("timm")
    if "transformers" in code.lower():
        frameworks.append("HuggingFace")
    if "sklearn" in code or "scikit" in code:
        frameworks.append("scikit-learn")
    if "gymnasium" in code.lower() or "import gym" in code.lower():
        frameworks.append("Gymnasium")
    if "mlxtend" in code.lower():
        frameworks.append("mlxtend")
    if "torchaudio" in code.lower():
        frameworks.append("torchaudio")
    if "tweepy" in code.lower():
        frameworks.append("Tweepy")
    return ", ".join(frameworks) if frameworks else "Python stdlib"


def detect_category_type(category: str) -> str:
    """Map category to domain."""
    mapping = {
        "Anomaly detection and fraud detection": "Tabular/CV",
        "Associate Rule Learning": "Tabular",
        "Chat bot": "NLP",
        "GANS": "CV (Generative)",
        "Recommendation Systems": "Tabular/NLP",
        "Reinforcement Learning": "RL",
        "Speech and Audio processing": "Audio/CV",
    }
    return mapping.get(category, "ML/DL")


def get_metrics_for_type(problem_type: str) -> str:
    """Return appropriate metrics string."""
    if "classif" in problem_type.lower() or "anomaly" in problem_type.lower() or "fraud" in problem_type.lower():
        return "accuracy, f1, precision, recall, confusion_matrix, roc_auc"
    if "regress" in problem_type.lower() or "time series" in problem_type.lower():
        return "rmse, mae, r2"
    if "gan" in problem_type.lower() or "generat" in problem_type.lower():
        return "FID, IS (Inception Score)"
    if "reinforcement" in problem_type.lower():
        return "cumulative_reward, episode_length"
    if "recommendation" in problem_type.lower():
        return "precision@k, recall@k, ndcg, map"
    if "association" in problem_type.lower():
        return "support, confidence, lift"
    return "task-specific metrics"


def remove_duplicate_imports(code: str) -> str:
    """Remove duplicate import lines."""
    lines = code.split("\n")
    seen_imports = set()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped in seen_imports:
                continue
            seen_imports.add(stripped)
        result.append(line)
    return "\n".join(result)


def fix_hardcoded_paths(code: str) -> str:
    """Replace common hardcoded path patterns with pathlib-based relative paths."""
    # Replace '/path/to/...' patterns
    code = re.sub(
        r"""['"]\/path\/to\/[^'"]*['"]""",
        'str(DATA_DIR / "placeholder")',
        code,
    )
    return code


def normalize_code_file(project: dict):
    """Normalize a 2 - Code.py file."""
    code_file = project["path"] / "2 - Code.py"
    if not code_file.exists():
        return False

    original = code_file.read_text(encoding="utf-8", errors="ignore")

    # Skip if already normalized
    if "# ── Standard Setup ──" in original:
        return False

    # Skip empty files
    if not original.strip():
        # Write a placeholder
        header = STANDARD_HEADER.format(
            project_name=project["name"],
            category=project["category"],
        )
        code_file.write_text(
            header + "# This file is empty. Implementation pending.\n",
            encoding="utf-8",
        )
        return True

    # Remove duplicate imports
    cleaned = remove_duplicate_imports(original)

    # Fix hardcoded paths
    cleaned = fix_hardcoded_paths(cleaned)

    # Add standard header
    header = STANDARD_HEADER.format(
        project_name=project["name"],
        category=project["category"],
    )

    code_file.write_text(header + cleaned, encoding="utf-8")
    return True


def create_src_modules(project: dict):
    """Create src/ directory with modular stubs."""
    src_dir = project["path"] / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    problem_type = ""
    code_file = project["path"] / "2 - Code.py"
    if code_file.exists():
        code = code_file.read_text(encoding="utf-8", errors="ignore")
        problem_type = detect_problem_type(code, project["name"])

    # __init__.py
    init_file = src_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            f'"""Source modules for: {project["name"]}"""\n',
            encoding="utf-8",
        )

    # data_loader.py
    data_loader = src_dir / "data_loader.py"
    if not data_loader.exists():
        data_loader.write_text(textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            Data Loader for: {project["name"]}
            Handles dataset loading, splitting, and DataLoader creation.
            """

            import logging
            from pathlib import Path

            import numpy as np

            logger = logging.getLogger(__name__)

            DATA_DIR = Path(__file__).resolve().parent.parent / "data"


            def load_data(data_dir: Path = DATA_DIR):
                """Load the dataset from data directory."""
                logger.info("Loading data from %s", data_dir)
                # Implement dataset-specific loading logic here
                raise NotImplementedError("Implement dataset loading for this project")


            def get_splits(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
                """Split data into train/val/test sets."""
                from sklearn.model_selection import train_test_split

                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=(1 - train_ratio), random_state=seed, stratify=y if y is not None else None
                )
                val_fraction = val_ratio / (1 - train_ratio)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=(1 - val_fraction), random_state=seed
                )
                logger.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))
                return X_train, X_val, X_test, y_train, y_val, y_test
        '''), encoding="utf-8")

    # preprocessing.py
    preprocessing = src_dir / "preprocessing.py"
    if not preprocessing.exists():
        preprocessing.write_text(textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            Preprocessing for: {project["name"]}
            Data cleaning, feature engineering, and transformations.
            """

            import logging

            import numpy as np

            logger = logging.getLogger(__name__)


            def preprocess(data):
                """Apply preprocessing pipeline to raw data."""
                logger.info("Preprocessing data...")
                # Implement preprocessing steps here
                return data
        '''), encoding="utf-8")

    # model.py
    model_file = src_dir / "model.py"
    if not model_file.exists():
        model_file.write_text(textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            Model Definition for: {project["name"]}
            Problem Type: {problem_type}
            """

            import logging

            logger = logging.getLogger(__name__)


            def build_model(**kwargs):
                """Build and return the model."""
                logger.info("Building model...")
                # Implement model architecture here
                raise NotImplementedError("Implement model for this project")
        '''), encoding="utf-8")

    # train.py
    train_file = src_dir / "train.py"
    if not train_file.exists():
        train_file.write_text(textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            Training Script for: {project["name"]}
            """

            import logging
            from pathlib import Path

            logger = logging.getLogger(__name__)

            OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


            def train(model, train_data, val_data, config: dict):
                """Train the model."""
                logger.info("Starting training...")
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                # Implement training loop here
                raise NotImplementedError("Implement training for this project")
        '''), encoding="utf-8")

    # evaluate.py
    eval_file = src_dir / "evaluate.py"
    if not eval_file.exists():
        metrics = get_metrics_for_type(problem_type)
        eval_file.write_text(textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            Evaluation Script for: {project["name"]}
            Problem Type: {problem_type}
            Metrics: {metrics}
            """

            import logging

            import numpy as np

            logger = logging.getLogger(__name__)


            def evaluate(model, test_data, test_labels):
                """Evaluate model performance."""
                logger.info("Evaluating model...")
                # Implement evaluation with appropriate metrics:
                # {metrics}
                raise NotImplementedError("Implement evaluation for this project")
        '''), encoding="utf-8")


def create_readme(project: dict):
    """Create README.md for a project."""
    readme_path = project["path"] / "README.md"
    if readme_path.exists():
        return False

    code_file = project["path"] / "2 - Code.py"
    notebooks = list(project["path"].glob("*.ipynb"))
    code = ""
    if code_file.exists():
        code = code_file.read_text(encoding="utf-8", errors="ignore")
    elif notebooks:
        code = "notebook"

    problem_type = detect_problem_type(code, project["name"])
    framework = detect_framework(code)
    domain = detect_category_type(project["category"])
    metrics = get_metrics_for_type(problem_type)
    has_notebook = bool(notebooks)
    notebook_name = notebooks[0].name if notebooks else None

    # Check for download script
    has_download = (project["path"] / "data" / "download_dataset.py").exists()

    content = f"""# {project["name"]}

**Category:** {project["category"]}
**Domain:** {domain}
**Problem Type:** {problem_type}
**Framework:** {framework}

## Overview

This project implements {problem_type.lower()} using {framework}.

## Project Structure

```
{project["name"]}/
├── data/
│   └── download_dataset.py    # Dataset download script
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & splitting
│   ├── preprocessing.py       # Data preprocessing
│   ├── model.py               # Model architecture
│   ├── train.py               # Training loop
│   └── evaluate.py            # Evaluation & metrics
├── outputs/                   # Checkpoints, logs, results
"""
    if has_notebook:
        content += f"├── {notebook_name}\n"
    content += f"""├── 2 - Code.py                # Main implementation script
└── README.md
```

## Quick Start

1. **Download dataset:**
   ```bash
   python data/download_dataset.py
   ```

2. **Run main script:**
   ```bash
   python "2 - Code.py"
   ```
"""
    if has_notebook:
        content += f"""
3. **Or open notebook:**
   ```bash
   jupyter notebook "{notebook_name}"
   ```
"""

    content += f"""
## Evaluation Metrics

- {metrics}

## GPU Support

This project supports CUDA acceleration. Verify with:
```bash
python ../../scripts/check_gpu.py
```

## Configuration

Global defaults from `configs/global_config.yaml` can be overridden locally.
"""

    readme_path.write_text(content, encoding="utf-8")
    return True


def create_outputs_dir(project: dict):
    """Ensure outputs/ directory exists."""
    outputs_dir = project["path"] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = outputs_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


def main():
    projects = discover_projects()
    print(f"\n{'='*60}")
    print(f"  PROJECT NORMALIZER — {len(projects)} projects")
    print(f"{'='*60}\n")

    normalized = 0
    readmes_created = 0
    src_created = 0

    for proj in projects:
        print(f"  Processing: {proj['category']}/{proj['name']}")

        # 1. Normalize code file
        if normalize_code_file(proj):
            normalized += 1
            print(f"    ✓ Normalized 2 - Code.py")

        # 2. Create src/ modules
        src_dir = proj["path"] / "src"
        if not (src_dir / "__init__.py").exists():
            create_src_modules(proj)
            src_created += 1
            print(f"    ✓ Created src/ modules")
        else:
            print(f"    - src/ already exists")

        # 3. Create README.md
        if create_readme(proj):
            readmes_created += 1
            print(f"    ✓ Created README.md")
        else:
            print(f"    - README.md already exists")

        # 4. Create outputs/
        create_outputs_dir(proj)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Code files normalized : {normalized}")
    print(f"  src/ modules created  : {src_created}")
    print(f"  READMEs created       : {readmes_created}")
    print(f"  Total projects        : {len(projects)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
