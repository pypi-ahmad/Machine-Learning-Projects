"""Shared helpers for building all 100 project notebooks."""
import json, os, textwrap

BASE = os.path.join(os.path.dirname(__file__), "..", "100_Local_AI_Projects")

CATEGORIES = {
    1: "Beginner_Local_LLM_Apps",
    2: "Local_RAG",
    3: "Advanced_RAG_and_Retrieval_Engineering",
    4: "LangGraph_Workflows",
    5: "Local_Tool-Using_Agents",
    6: "CrewAI_Multi-Agent_Systems",
    7: "Local_Eval_and_Observability_Projects",
    8: "Fine-Tuning-Adjacent_Learning_Projects",
    9: "Multimodal_-_OCR_-_Speech_-_VLM",
    10: "Coding_and_Developer_Agents",
}

NB_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.0",
    },
}


def _src(text: str) -> list[str]:
    """Convert a dedented multi-line string to notebook source list."""
    lines = textwrap.dedent(text).strip().split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


def write_nb(group_num: int, folder_name: str, cells: list[dict]):
    """Write a notebook to the correct category folder."""
    cat = CATEGORIES[group_num]
    project_dir = os.path.join(BASE, cat, folder_name)
    os.makedirs(project_dir, exist_ok=True)
    nb = {"cells": cells, "metadata": NB_META, "nbformat": 4, "nbformat_minor": 4}
    path = os.path.join(project_dir, "notebook.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return path
