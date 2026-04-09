"""Tests for NLP Projecct 12.KeywordResearch (no local data files)."""
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def project_dir():
    root = Path(__file__).resolve().parent.parent
    return root / "NLP Projecct 12.KeywordResearch"


@pytest.fixture(scope="module")
def notebook_path(project_dir):
    nbs = list(project_dir.glob("*.ipynb"))
    assert len(nbs) >= 1, "No notebook found"
    return nbs[0]


@pytest.mark.no_local_data
class TestProjectStructure:
    def test_project_dir_exists(self, project_dir):
        assert project_dir.exists()

    def test_notebook_exists(self, notebook_path):
        assert notebook_path.exists()

    def test_notebook_valid_json(self, notebook_path):
        import json
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert "cells" in nb
        assert len(nb["cells"]) > 0

    def test_has_code_cells(self, notebook_path):
        import json
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) > 0


@pytest.mark.no_local_data
class TestPreprocessing:
    def test_basic_text_cleaning(self):
        import re
        sample = "Hello, World! This is a TEST 123."
        cleaned = re.sub(r"[^a-zA-Z\s]", "", sample).lower().strip()
        assert cleaned == "hello world this is a test"

    def test_tokenization(self):
        text = "Natural language processing is amazing"
        tokens = text.lower().split()
        assert tokens == ["natural", "language", "processing", "is", "amazing"]
