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

## Task workflow
For every task, first inspect the repository and determine whether the requested functionality already exists.

If it already exists, verify that it:
- is fully implemented
- matches the requested instructions exactly
- runs successfully
- has zero errors

If all of the above are true, skip reimplementation.

If it is missing, create it.
If it exists but is incomplete, incorrect, not as instructed, not runnable, or throws errors, then fix it.

After any creation or fix, run it and validate it.
If any error remains, continue fixing and re-running in a loop until there are zero errors.

Do not stop at partial completion.
Do not duplicate correct existing work unnecessarily.
Do not break unrelated working functionality.
Keep changes clean, minimal, consistent with the repo, and non-regressive.

Finish only when:
- the requested work exists
- it matches the instructions
- it runs successfully
- it has zero errors
- no unrelated functionality was broken

## Rule files
- `.claude/rules/ipynb-projects.md`
- `.claude/rules/single-file-py-projects.md`
