# Phase 0.2 — Foundation Setup Notes

**Date:** Auto-generated during Phase 0.2 execution  
**Python target:** 3.13.12 (Windows)  
**GPU:** PyTorch cu130 (CUDA 13.0)  
**Constraint:** No TensorFlow anywhere

---

## Files Created

| File / Directory | Purpose |
|---|---|
| `requirements.txt` | Global pip dependencies (no torch — installed separately via cu130 index) |
| `requirements-dev.txt` | Black, isort, ruff, pytest, pre-commit |
| `pyproject.toml` | Ruff / Black / isort tool configs (line-length 100, py313) |
| `.env.example` | Template env vars (HF_TOKEN, GLOBAL_SEED, DEVICE) |
| `README.md` | Workspace overview and quick-start |
| `verify_env.py` | Environment verification script (Python, torch, CUDA, OpenCV, key packages) |
| `scripts/setup.ps1` | Idempotent venv bootstrap (creates .venv, installs torch cu130, reqs, ipykernel) |
| `scripts/patch_notebooks.py` | Patches all 21 notebooks: kernel → nlp-projects, injects bootstrap cell |
| `utils/__init__.py` | Shared utils package |
| `utils/logger.py` | Structured logging helper (`get_logger`) |
| `utils/seed.py` | Reproducibility seed setter (`set_global_seed`) |
| `config/base_config.yaml` | Default hyper-parameters (seed 42, lr 2e-5, batch 16, epochs 5, test 0.2) |
| `data/<project>/{raw,processed}` | 21 project data directories with .gitkeep |

## Files Modified

| File | Change |
|---|---|
| `.gitignore` | Added `.venv/`, `.env`, `/data/`, `/outputs/`, `.ipynb_checkpoints/`, `Thumbs.db` |
| All 21 `code.ipynb` | Kernel metadata → `nlp-projects`, bootstrap cell injected before first code cell |

---

## Known Issues & Warnings

### pycaret
- **Status:** NOT included in `requirements.txt`. Install attempted and **FAILED**.
- **Error:** `pycaret 3.3.2` requires `pandas<2.2.0`. Building pandas 2.1.4 from source on Python 3.13 fails with C API mismatch (`_PyLong_AsByteArray` signature change in CPython 3.13).
- **Workaround:** `requirements-optional-pycaret.txt` created at workspace root (commented out). Use a separate Python 3.11/3.12 venv if pycaret is needed.

### lazypredict
- Included in requirements.txt but may have compatibility issues with Python 3.13.
- Monitor for import errors; remove if it blocks other installs.

### Stories Clustering
- Original `requirements.txt` pins `gensim==3.8.3` and `spacy==2.3.5` — both incompatible with Python 3.13.
- The global `requirements.txt` uses modern unpinned versions instead.
- This project's notebook may need code adjustments in a later phase.

### Automated Image Captioning
- Notebook hardcodes a Google Drive mount path for images.
- Will need path logic fix in a later phase.

### English to French Translation
- Notebook has a path mismatch (`eng-fra.txt` vs expected path).
- Will need path fix in a later phase.

### Fake News Detection
- Likely has train/test data leakage (both True.csv and Fake.csv concatenated then random-split).
- Will need proper data handling in a later phase.

### OpenCV
- Pinned to `>=4.12.0,<4.13` to avoid ABI breakage.

### torch / torchvision
- Deliberately excluded from `requirements.txt` — they must be installed via `--index-url https://download.pytorch.org/whl/cu130` (handled by `scripts/setup.ps1`).

---

## Next Steps (Phase 0.3+)
1. Run `scripts/setup.ps1` to create `.venv` and install all dependencies.
2. Run `python verify_env.py` to confirm environment health.
3. Begin per-project code fixes (paths, imports, deprecated APIs).
4. Add shared data-loading helpers to `utils/`.
5. Run each notebook end-to-end and log failures.
