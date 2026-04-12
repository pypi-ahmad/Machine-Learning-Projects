# Notebook Audit Rules

Apply these rules when auditing, fixing, standardizing, optimizing, or validating Jupyter notebooks (`.ipynb`) in this repository.

## Core execution rule

For every notebook in the repository:

1. Discover it
2. Restart kernel
3. Run all cells end-to-end
4. If any error occurs:
   - identify the root cause
   - fix it
   - rerun from scratch
5. Repeat until the notebook runs successfully with zero errors

Do not stop at partial completion.
Do not assume success without execution.
Continue until all notebooks pass all required checks.

## Evaluation rule

Use task-appropriate metrics.
Do not force one generic metric on every notebook.

Also:
- Every notebook must be compared against a simple baseline
- Every notebook must show clear improvement over that baseline
- Every notebook must meet the required quality threshold for its task and difficulty
- If performance is below the required threshold, improve and rerun

## Dataset difficulty rule

Estimate dataset difficulty first:

### EASY
- clean, structured, low noise
- balanced classes
- strong feature-target signal
- simple or moderate size

### MEDIUM
- moderate noise
- some imbalance
- partial feature relevance
- real-world messiness

### HARD
- severe noise or missingness
- strong imbalance
- weak signal
- sparse, high-dimensional, or complex data

## Task-specific thresholds

### 1. Classification
Use:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC when appropriate
- PR-AUC when class imbalance matters
- Confusion matrix

Thresholds by difficulty:
- EASY: primary metric >= 0.90
- MEDIUM: primary metric >= 0.80
- HARD: primary metric >= 0.70

Additional rules:
- For imbalanced classification, do not rely on accuracy alone
- Use PR-AUC / Recall / F1 as the main decision metric when appropriate
- Explicit threshold tuning is required for recall-critical problems

### 2. Regression
Use:
- R²
- RMSE
- MAE
- residual analysis

Thresholds by difficulty:
- EASY: R² >= 0.90
- MEDIUM: R² >= 0.75
- HARD: R² >= 0.60

Additional rule:
- RMSE / MAE must be at least 30–50% better than a naive baseline such as mean prediction

### 3. Clustering / Unsupervised
Use:
- Silhouette score
- cluster separation quality
- cluster interpretability

Thresholds by difficulty:
- EASY: silhouette >= 0.60
- MEDIUM: silhouette >= 0.50
- HARD: silhouette >= 0.40

Additional rule:
- Clusters must be interpretable and clearly described

### 4. Time Series
Use:
- MAPE or sMAPE where appropriate
- R² if appropriate
- visual forecast alignment
- strict leakage-safe temporal splits

Thresholds by difficulty:
- EASY: MAPE <= 10%
- MEDIUM: MAPE <= 20%
- HARD: MAPE <= 30%

Additional rules:
- No data leakage
- Forecast must be visually reasonable against actuals
- Model must beat naive or seasonal naive baseline clearly

### 5. Imbalanced Fraud / Anomaly Detection
Use:
- PR-AUC
- Recall
- threshold tuning
- calibration if relevant

Thresholds:
- Prefer PR-AUC >= 0.90 when realistically achievable
- Or Recall >= 0.90 when recall is the critical objective

Additional rule:
- Accuracy alone is not acceptable for imbalanced problems

### 6. NLP / CV / Deep Learning
If classification-based:
- follow the classification thresholds above

Also require:
- stable training
- no divergence
- no severe overfitting
- train and validation behavior must be reasonable
- loss curves must converge

### 7. Recommendation Systems
Use:
- Precision@K / Recall@K / NDCG or strongest suitable benchmark
- comparison against popularity baseline

Threshold:
- Must clearly beat popularity or naive baseline
- Prefer Precision@K >= 0.90 only when realistic for the dataset/task

### 8. Reinforcement Learning
Use:
- reward curve
- convergence behavior
- comparison against random policy

Threshold:
- Reward must converge or stabilize meaningfully
- Policy must significantly outperform random baseline

## Notebook scoring rule

Each notebook must receive a score from 0 to 100:

1. Execution: 20
   - runs end-to-end without errors

2. Pipeline Completeness: 20
   - data loading
   - preprocessing
   - EDA
   - feature engineering where needed
   - training
   - validation
   - testing
   - evaluation
   - error analysis
   - interpretability where applicable

3. Performance: 30
   - meets or exceeds task-specific threshold

4. Code Quality: 10
   - clean, modular, readable
   - no dead code
   - no hardcoded paths

5. Reproducibility: 10
   - seeds fixed where appropriate
   - deterministic or controlled behavior where practical
   - environment stability

6. MLOps Practices: 10
   - logging
   - artifacts
   - experiment tracking if applicable

Score interpretation:
- 90–100: production-ready
- 80–89: acceptable but needs improvement
- 70–79: weak and must improve
- <70: failed

## Pipeline completeness rule

Each notebook must include, where applicable:
- data loading
- preprocessing
- EDA with meaningful visualizations
- feature engineering
- model training
- validation strategy
- testing
- evaluation metrics
- error analysis
- interpretability
- reproducibility
- experiment tracking (MLflow if applicable)

## Engineering rules

All notebooks must follow these standards:
- no hardcoded paths
- no dead code
- clean modular notebook structure
- no missing dependencies
- reproducible environment
- no Google Colab dependence
- no silent failures
- no hidden assumptions

## Reporting rule

For each notebook, generate a structured report capturing:
- notebook name
- execution status
- difficulty
- primary metrics
- thresholds
- baseline comparison
- score
- major issues
- final verdict

If MLflow is used, log:
- parameters
- metrics
- artifacts
- run metadata

## Failure loop

If any notebook has:
- execution error
- incomplete pipeline
- performance below threshold
- score < 90
- no baseline comparison
- overfitting severe enough to invalidate result

Then:
1. identify the weakness
2. improve data quality / preprocessing / features / model / tuning
3. retrain
4. rerun notebook from scratch
5. reevaluate
6. rescore
7. repeat until standards are met

## Prioritization rule

When multiple notebooks need work:
- rank them by score ascending
- fix lowest-scoring notebooks first
- continue until all notebooks meet the standard

## Hard failure conditions

A notebook is FAILED if any of the following is true:
- execution error exists
- score < 70
- pipeline is incomplete
- performance is below task/difficulty threshold
- no baseline comparison exists
- severe overfitting invalidates the result

## Final completion rule

You are only done when:
- all notebooks run successfully end-to-end
- all notebooks have zero execution errors
- all notebooks meet task-appropriate performance thresholds
- all notebooks beat their baselines clearly
- all notebooks score >= 90
- all pipelines are complete
- code quality is production-grade
- outputs are reproducible and validated

## Absolute rule

If any notebook is below standard:
- continue improving
- do not stop
- do not return partial completion
- do not assume success without rerunning

Your task is complete only when the entire repository is:
- fully executed
- quantitatively validated
- optimized to the required threshold
- production-grade

