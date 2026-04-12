# One-by-One File Creation and Editing Rule

Apply this rule whenever the task involves creating, editing, fixing, or updating multiple files.

## Core rule
- Work on files one by one.
- Do not use bulk generation.
- Do not use scripts, generators, batch writers, scaffolding code, or automation to create many files at once.
- Do not write a helper script whose main purpose is to generate or modify multiple target files.
- Do not programmatically mass-edit files unless the user explicitly asks for that approach.

## Required workflow
For each requested file:

1. Identify the exact target file.
2. Check whether it already exists.
3. If it exists:
   - inspect it
   - verify whether it already satisfies the requested instructions
   - verify whether it runs or works correctly
4. If it is already correct and fully working, skip unnecessary rewrite.
5. If it is missing, create that file directly.
6. If it exists but is incomplete, incorrect, broken, or not as instructed, fix that file directly.
7. After editing or creating that file, validate it individually.
8. Then move to the next file.

## Validation rule
- Validate each file individually after working on it.
- Do not defer validation until the very end if the file can be checked earlier.
- If the file has errors, fix that same file and revalidate before moving on.

## Forbidden approaches
- No generator scripts
- No batch file creation scripts
- No bulk templating scripts
- No mass search-and-replace across many targets without explicit permission
- No writing one script that outputs many final `.py` or `.ipynb` files
- No "temporary automation" used to bypass one-by-one implementation

## Allowed approach
- Directly create the actual target file.
- Directly write the actual final content into that file.
- Directly edit the actual target file when fixing or improving it.
- Repeat file by file until all requested files are done.

## Quality rule
- Each file must be treated as a real deliverable, not as generated boilerplate.
- Keep each file aligned with the specific instructions for that file type and project type.
- Prefer correctness, validation, and stability over speed.

## Completion rule
A multi-file task is complete only when:
- every requested file has been handled individually
- every file is created or fixed directly
- every file has been validated individually
- no bulk-generation method was used
- no unrelated files were modified

