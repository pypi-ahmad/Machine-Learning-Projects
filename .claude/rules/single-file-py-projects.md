# Single-File Python Project Rules

Apply these rules only when the target project is a single-file Python project.

## Hard constraints
- The project must remain a single `.py` file.
- Prefer `main.py` unless the project already has a different single-file entrypoint.
- Do not create notebooks.
- Do not create helper modules, utility modules, packages, or multiple Python files.
- Do not use FastAPI or Flask.
- Streamlit is allowed only when a lightweight UI is clearly useful.
- If Streamlit is not needed, prefer a plain CLI or Tkinter.
- Work only on the target Python file and any strictly necessary local assets.
- Do not touch unrelated projects.

## Project size target
- Keep the project small to medium in scope.
- The entire implementation must stay understandable in one file.
- Avoid ideas that genuinely require a multi-file architecture.

## Preferred project styles
- CLI tools
- Tkinter desktop tools
- mini games
- file utilities
- small dashboards in Streamlit when a UI adds real value
- lightweight demos that are realistic in one file

## Required code organization inside the single file
1. Module docstring or file header
2. Imports
3. Configuration / constants
4. Helper functions
5. Core business logic
6. UI or CLI layer
7. Main entrypoint
8. Optional argument parsing or launch block

## Code quality rules
- Use clear function boundaries even inside one file.
- Avoid giant monolithic functions.
- Use descriptive variable names.
- Keep imports minimal.
- Add comments only where truly helpful.
- Add docstrings for major functions when useful.
- Prefer readability over abstraction.

## Streamlit rule
Use Streamlit only when the project benefits from:
- forms
- charts
- image or file upload
- simple dashboards
- lightweight model demos

If the project is a calculator, parser, utility, mini game, automation script, or local helper tool, prefer CLI or Tkinter first.

## UI rules

### If CLI
- use clear prompts
- validate inputs
- show helpful error messages

### If Tkinter
- keep the interface simple
- avoid overengineering
- keep the UI readable in one file

### If Streamlit
- keep the app simple
- do not turn it into a platform
- keep it runnable from one script only

## Data and file handling
- Avoid hardcoded absolute paths.
- Use relative paths or user input.
- Validate file existence before reading.
- Fail clearly and safely.

## Educational preference
When the project is for learning:
- include a top-of-file explanation in a module docstring
- keep the flow easy to follow
- prefer obvious patterns over clever compactness

## Guardrails
- No unrelated edits.
- No extra `.py` files.
- No notebooks.
- No hidden side effects.
- No fake functionality.
- No unnecessary dependencies.

## Final checks
Before finishing:
- verify there is only one Python file for the project
- verify the file is runnable
- verify no notebook was added
- verify no FastAPI or Flask was added
- verify Streamlit is used only if justified
- verify the code remains readable and maintainable
