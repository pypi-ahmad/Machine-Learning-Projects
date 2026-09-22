# Scientific Calculator

A standard-library command-line calculator for arithmetic, trigonometry,
logarithms, powers, factorials, and mathematical constants.

## Run

```powershell
uv run python main.py
```

Enter an expression at the prompt. Use `help` to display supported operations,
or `quit` to exit.

```text
> sin(30) + log10(100)
= 2.5
```

## Supported expressions

- Operators: `+`, `-`, `*`, `/`, `//`, `%`, and `**`
- Functions: `sin`, `cos`, `tan`, inverse trig functions, logarithms, `sqrt`,
  `cbrt`, `exp`, `abs`, `ceil`, `floor`, `round`, and `factorial`
- Constants: `pi`, `e`, `tau`, and `inf`

Trigonometric function arguments and inverse-trigonometric results use degrees.
Expressions are parsed as restricted arithmetic syntax; attributes, indexing,
assignments, imports, and arbitrary Python code are rejected.

## Dependencies

This project uses only the Python standard library. uv records the Python
requirement and provides a reproducible environment.
