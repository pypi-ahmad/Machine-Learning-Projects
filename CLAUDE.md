# Repository Instructions

## General rules
- Make minimal, safe, non-regressive changes.
- Inspect the target file or project before editing.
- Do not touch unrelated files or projects.
- Do not hallucinate datasets, file paths, columns, APIs, outputs, metrics, or features.
- If something is unclear, inspect first and choose the safest grounded implementation.

## Scope rules
- Use notebook rules only for `.ipynb` projects.
- Use single-file Python rules only for one-file `.py` projects.
- Do not mix notebook and `.py` project instructions.

## Git discipline
- Stage only relevant files.
- Do not use blanket staging for unrelated work.
- Do not commit caches, junk files, model weights, or unrelated artifacts.

## Rule files
- `.claude/rules/ipynb-projects.md`
- `.claude/rules/single-file-py-projects.md`
