# Notebook-Only Project Rules

Apply these rules only when the target project is a Jupyter notebook project.

## Hard constraints
- Output must be `.ipynb` only.
- Do not create any `.py` files.
- Do not move logic into helper scripts.
- Do not add Streamlit, Gradio, Flask, or FastAPI.
- Keep everything notebook-first and learning-focused.
- Work only on the target notebook and any strictly necessary local data/artifact folders.
- Do not touch unrelated notebooks or projects.
- Keep the notebook runnable top-to-bottom.

## Notebook purpose
The notebook must teach, not just run.
It should read like a guided lab or learning project.

## Required notebook structure
1. Title
2. Project overview
3. Learning objectives
4. Problem statement
5. Why this project matters
6. Dataset overview
7. Dataset source and license notes
8. Environment setup
9. Imports
10. Configuration / constants
11. Dataset download and loading
12. Data validation checks
13. Data cleaning / preprocessing
14. Exploratory data analysis
15. Task-specific preparation
16. Baseline approach
17. Main workflow / model / method
18. Training or execution
19. Inference / outputs / examples
20. Evaluation
21. Error analysis
22. Interpretation / insights
23. Limitations
24. How to improve this project
25. Production considerations
26. Common mistakes
27. Mini challenge / exercises
28. Final summary / key takeaways

## Writing rules
- Use markdown cells generously.
- Add a markdown explanation before every major code block.
- Explain why each step is done, not just what it does.
- Avoid giant unexplained code cells.
- Keep code easy to study later.
- Use a professional but beginner-friendly tone.

## Dataset rules
- Handle dataset download or loading inside the notebook.
- Prefer public download inside the notebook when practical.
- If Kaggle is used, add a setup section for credentials and a safe fallback explanation.
- Never assume the dataset is already present locally.
- Make loading idempotent.
- Validate:
  - missing files
  - missing columns
  - malformed rows
  - duplicates
  - target leakage risks
- Explain dataset source, target, important columns, and limitations in markdown.

## Evaluation rules
Choose metrics that fit the notebook task.

### Classification
- accuracy
- precision
- recall
- F1
- confusion matrix
- ROC-AUC when meaningful
- PR-AUC when imbalance matters

### Regression
- RMSE
- MAE
- R2
- residual analysis

### Time series
- time-aware split
- MAE
- RMSE
- MAPE or sMAPE where appropriate
- naive / seasonal naive baseline where possible

### Retrieval / RAG
- retrieval quality
- groundedness / source support
- qualitative failure cases

### Generation / summarization
- qualitative comparison
- prompt variation comparison where relevant
- explicit limitations

## Modeling rules
- Start with a baseline when applicable.
- Use LazyPredict only where appropriate: classification and regression.
- Use FLAML where appropriate.
- For time series, do not misuse LazyPredict as a native forecasting framework.
- If using forecasting, explain why the chosen library fits.

## Guardrails
- No hallucinated results.
- No fake benchmark scores.
- No unrelated edits.
- No hidden assumptions about files or credentials.
- Preserve working behavior unless there is a clear reason to improve it safely.

## Final checks
Before finishing:
- verify the output is `.ipynb` only
- verify no `.py` files were created
- verify the notebook runs logically top-to-bottom
- verify all required sections exist
- verify evaluation matches the task
- verify explanations are strong and grounded
