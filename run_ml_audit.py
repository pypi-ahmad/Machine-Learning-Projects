import os
import json
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_notebooks():
    notebooks = []
    for root, dirs, files in os.walk("."):
        if "venv" in root or ".git" in root or "checkpoint" in root:
            continue
        for file in files:
            if file.endswith(".ipynb"):
                notebooks.append(Path(root) / file)
    return notebooks

def estimate_difficulty(nb_path):
    # Heuristic for difficulty based on path and size
    path_str = str(nb_path).lower()
    if "deep learning" in path_str or "nlp" in path_str or "cv" in path_str:
        return "HARD"
    if "analysis" in path_str or "basic" in path_str:
        return "EASY"
    return "MEDIUM"

def run_notebook(nb_path):
    print(f"--- Processing: {nb_path} ---")
    notebook_name = nb_path.stem
    report_path = REPORTS_DIR / f"{notebook_name}_report.json"
    
    report = {
        "notebook_name": str(nb_path),
        "status": "FAIL",
        "score": 0,
        "difficulty": estimate_difficulty(nb_path),
        "execution": {"ran_successfully": False, "errors": []},
        "performance": {},
        "final_verdict": "FAILED"
    }

    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Set experiment
        mlflow.set_experiment(notebook_name)
        with mlflow.start_run():
            start_time = time.time()
            try:
                ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})
                report["execution"]["ran_successfully"] = True
                report["status"] = "PASS"
            except Exception as e:
                report["execution"]["errors"].append(str(e))
                print(f"Execution failed for {nb_path}: {e}")
            
            end_time = time.time()
            mlflow.log_param("execution_time", end_time - start_time)
            
            # Simple Scoring (placeholder for actual metric extraction)
            score = 0
            if report["execution"]["ran_successfully"]:
                score += 20 # Execution
                score += 20 # Pipeline (assumed complete if ran)
                score += 30 # Performance (placeholder)
                score += 10 # Code quality
                score += 10 # Reproducibility
                score += 10 # MLOps
            
            report["score"] = score
            if score >= 90:
                report["final_verdict"] = "PRODUCTION_READY"
            elif score >= 80:
                report["final_verdict"] = "NEEDS_IMPROVEMENT"
            else:
                report["final_verdict"] = "FAILED"

            mlflow.log_metric("score", score)

    except Exception as e:
        report["execution"]["errors"].append(f"System Error: {str(e)}")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    notebooks = get_notebooks()
    results = []
    for nb in notebooks:
        res = run_notebook(nb)
        results.append(res)
    
    summary = {
        "total": len(results),
        "avg_score": np.mean([r["score"] for r in results]) if results else 0,
        "failed": len([r for r in results if r["status"] == "FAIL"]),
        "ranking": sorted(results, key=lambda x: x["score"])
    }
    
    with open(REPORTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Leaderboard
    leaderboard = [
        {"name": r["notebook_name"], "score": r["score"]}
        for r in sorted(results, key=lambda x: x["score"], reverse=True)
    ]
    with open(REPORTS_DIR / "leaderboard.json", 'w') as f:
        json.dump(leaderboard, f, indent=2)

if __name__ == "__main__":
    main()
