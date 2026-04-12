# Repository Instructions

## General working rules
- Make minimal, safe, non-regressive changes.
- Inspect the target project or file before editing.
- Do not touch unrelated files or projects.
- Do not hallucinate datasets, file paths, columns, APIs, outputs, metrics, or features.
- If something is unclear, inspect first and choose the safest grounded implementation.
- Reuse correct existing work instead of recreating it unnecessarily.

## Scope discipline
- Use notebook rules only for `.ipynb` projects.
- Use single-file Python rules only for one-file `.py` projects.
- Do not mix notebook instructions into `.py` projects.
- Do not mix `.py` app instructions into notebook projects.

## Git discipline
- Never use `git add -A`.
- Stage only relevant files.
- Never use blanket staging for unrelated work.
- Do not commit caches, model weights, temporary files, or unrelated artifacts.
- Use clean, scoped commit messages.

## Dataset preferences
- Use Kaggle as the default source to download datasets.
- All datasets must be downloaded programmatically inside the running code itself (not manually or ahead of time).

## Model and library preferences
- Always use the latest and best models available as of April 2026.
- Use FLAML for AutoML wherever possible.
- Use LazyPredict for quick baseline comparison wherever possible.
- For YOLO tasks, prefer `yolo26m` as the default model.

## Hardware and runtime preferences
- Always use local GPU / CUDA by default wherever possible.
- Prefer local Ollama for LLM inference wherever possible.

## Execution discipline
- Before creating something new, check whether it already exists.
- If it exists and is correct, complete, and runs with zero errors, skip reimplementation.
- If it exists but is incomplete, incorrect, not as instructed, or has errors, fix it.
- After creating or fixing anything, run it and validate it.
- If errors remain, continue fixing and rerunning until there are zero errors.

## Documentation discipline
- After every task execution, update `README.md` and `USAGE.md` at the repository root.
- Keep both files professional, intelligent, and FAANG-grade in quality.
- `README.md` must accurately describe what this repo is, its structure, projects, and how to get started.
- `USAGE.md` must explain how to use the repo: setup, running projects, dependencies, and workflows.
- Only update sections relevant to the work just completed — do not rewrite unrelated parts.
- Keep content grounded in what actually exists — never describe features or projects that do not exist yet.
