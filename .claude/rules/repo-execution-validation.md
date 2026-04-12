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

## Notebook audit rule
When auditing notebooks:
- restart kernel
- run all cells
- if any issue exists, fix it and rerun
- continue until all targeted notebooks pass the requested checks

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

Additional rule:
- Clusters must be interpretable and clearly described

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

Additional rules:
- No data leakage
- Forecast must be visually reasonable against actuals
- Model must beat naive or seasonal naive baseline clearly

### Imbalanced fraud / anomaly detection
Use:
- PR-AUC
- Recall
- threshold tuning
- calibration if relevant

Thresholds:
- Prefer PR-AUC >= 0.90 when realistically achievable
- Or Recall >= 0.90 when recall is the critical objective

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
- For multi-file tasks, file-by-file direct implementation is mandatory; generator scripts and bulk file automation are forbidden unless explicitly requested by the user.

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
