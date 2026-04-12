# Repository Execution, Validation, and Quality Rules

Apply these rules whenever the task involves creating, fixing, validating, or auditing repository work.

## Existence-first rule
For every future task:

1. First inspect the repository and determine whether the requested functionality, file, feature, notebook, script, or workflow already exists.
2. If it exists, verify that it:
   - is fully implemented
   - matches the instructions exactly
   - runs successfully
   - has zero errors
3. If all of the above are true, skip reimplementation.
4. If it is missing, create it.
5. If it exists but is incomplete, incorrect, not as instructed, not runnable, or throws errors, then fix it.
6. After any creation or fix, run it and validate it.
7. If any error remains, continue fixing and rerunning in a loop until there are zero errors.
8. Do not stop at partial completion.

## One-by-one file creation rule
If the user asks to create many `.py` or `.ipynb` files:
- never use a generator script
- never use a batch creation script
- never create helper automation just to generate many files
- create each target file directly
- implement each file one by one
- write the actual content in the file itself
- verify each file individually

## Notebook audit rule
When auditing notebooks:
- restart kernel
- run all cells
- if any issue exists, fix it and rerun
- continue until all targeted notebooks pass the requested checks

## Dataset difficulty rule
Estimate dataset difficulty before choosing thresholds:

EASY:
- clean, structured, low noise
- balanced classes
- strong feature-target signal
- simple or moderate size

MEDIUM:
- moderate noise
- some imbalance
- partial feature relevance
- real-world messiness

HARD:
- severe noise or missingness
- strong imbalance
- weak signal
- sparse, high-dimensional, or complex data

## Metric and threshold rule
Use task-appropriate metrics.
Do not force one generic metric on all tasks.

### Classification
Use:
- accuracy
- precision
- recall
- F1
- ROC-AUC when meaningful
- PR-AUC when imbalance matters
- confusion matrix

Thresholds by difficulty:
- EASY: primary metric >= 0.90
- MEDIUM: primary metric >= 0.80
- HARD: primary metric >= 0.70

### Regression
Use:
- R2
- RMSE
- MAE
- residual analysis

Thresholds by difficulty:
- EASY: R2 >= 0.90
- MEDIUM: R2 >= 0.75
- HARD: R2 >= 0.60

Additional rule:
- RMSE / MAE must be at least 30–50% better than a naive baseline where that comparison makes sense

### Clustering
Use:
- silhouette score
- cluster separation quality
- interpretability

Thresholds by difficulty:
- EASY: silhouette >= 0.60
- MEDIUM: silhouette >= 0.50
- HARD: silhouette >= 0.40

### Time series
Use:
- MAE
- RMSE
- MAPE or sMAPE where appropriate
- leakage-safe time splits
- baseline comparison vs naive / seasonal naive

Thresholds by difficulty:
- EASY: MAPE <= 10%
- MEDIUM: MAPE <= 20%
- HARD: MAPE <= 30%

### Imbalanced fraud / anomaly detection
Use:
- PR-AUC
- Recall
- threshold tuning
- calibration if relevant

Rule:
- do not rely on accuracy alone

### NLP / CV / deep learning
If classification-based:
- follow classification thresholds above

Also require:
- stable training
- no severe divergence
- no invalid overfitting
- reasonable train/validation behavior
- loss curves must converge

### Recommendation systems
Use:
- Precision@K / Recall@K / NDCG or strongest suitable benchmark
- comparison against popularity baseline

Threshold:
- Must clearly beat popularity or naive baseline
- Prefer Precision@K >= 0.90 only when realistic for the dataset/task

### Reinforcement learning
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

## Baseline rule
Every model or workflow must be compared against a simple baseline when the task supports it.
Final output must clearly improve over that baseline.

## Improvement loop
If the implementation:
- fails to run
- gives errors
- uses wrong metrics
- performs weakly
- does not meet the required threshold
- does not beat the baseline where relevant

Then:
1. identify the root cause
2. fix preprocessing / data / leakage / logic issues
3. improve features or model choice
4. tune and compare alternatives
5. rerun from scratch
6. reevaluate
7. repeat until the result is strong and error-free

## Engineering rules
- no hardcoded paths
- no dead code
- no missing dependencies
- no hidden assumptions
- no silent failures
- no unrelated edits
- keep implementations reproducible where practical

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
A task is complete only when:
- the requested work exists
- it matches the instructions
- it runs successfully
- it has zero errors
- the correct metrics are used
- the result meets the task-appropriate standard
- baseline comparison is shown where relevant
- no unrelated functionality was broken

## Stricter engineering expectations
- Prefer deterministic or seeded runs where practical.
- Prefer idempotent data-loading and setup steps.
- Prefer explicit error handling over silent fallbacks.
- Prefer scoped, reviewable changes over sweeping rewrites.
- Preserve working behavior unless the requested change requires adjustment.
- Always verify the target path, file type, and project style before editing.

## Review mindset
- Treat every change as if it will be code-reviewed by a senior engineer.
- Avoid unnecessary abstractions.
- Remove duplication only when it is clearly safe.
- Keep naming, structure, and flow consistent with the surrounding project.

## Absolute rule
If any notebook is below standard:
- continue improving
- do not stop
- do not return partial completion
- do not assume success without rerunning

The task is complete only when the entire repository is:
- fully executed
- quantitatively validated
- optimized to the required threshold
- production-grade
