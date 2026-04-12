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
- Stage only relevant files.
- Never use blanket staging for unrelated work.
- Do not commit caches, model weights, temporary files, or unrelated artifacts.
- Use clean, scoped commit messages.

## Execution discipline
- Before creating something new, check whether it already exists.
- If it exists and is correct, complete, and runs with zero errors, skip reimplementation.
- If it exists but is incomplete, incorrect, not as instructed, or has errors, fix it.
- After creating or fixing anything, run it and validate it.
- If errors remain, continue fixing and rerunning until there are zero errors.

## Rule files
- `.claude/rules/ipynb-projects.md`
- `.claude/rules/single-file-py-projects.md`
- `.claude/rules/repo-execution-validation.md`
