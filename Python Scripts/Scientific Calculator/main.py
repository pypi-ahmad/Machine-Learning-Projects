"""Scientific Calculator — CLI tool.

Supports basic arithmetic, trigonometry, logarithms, powers, factorials,
and constants. Operates in a REPL loop until the user exits.

Usage:
    python main.py
"""

import math
import operator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

HELP_TEXT = """
Scientific Calculator — supported operations
--------------------------------------------
Basic       : +  -  *  /  //  %  ** (e.g. 3 + 4, 2 ** 8)
Functions   : sin, cos, tan, asin, acos, atan   (degrees)
            : log(x), log(x, base), log10, log2
            : sqrt, cbrt, exp, abs, ceil, floor, round
            : factorial(n)
Constants   : pi, e, tau, inf
Type 'help' to show this menu, 'quit' to exit.
"""

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

_SAFE_NAMES: dict = {
    # math functions
    "sin":       lambda x: math.sin(math.radians(x)),
    "cos":       lambda x: math.cos(math.radians(x)),
    "tan":       lambda x: math.tan(math.radians(x)),
    "asin":      lambda x: math.degrees(math.asin(x)),
    "acos":      lambda x: math.degrees(math.acos(x)),
    "atan":      lambda x: math.degrees(math.atan(x)),
    "log":       math.log,
    "log10":     math.log10,
    "log2":      math.log2,
    "sqrt":      math.sqrt,
    "cbrt":      lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "exp":       math.exp,
    "abs":       abs,
    "ceil":      math.ceil,
    "floor":     math.floor,
    "round":     round,
    "factorial": math.factorial,
    # constants
    **CONSTANTS,
    "__builtins__": {},
}


def evaluate(expr: str) -> float:
    """Safely evaluate a mathematical expression string."""
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression.")
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except Exception as exc:
        raise ValueError(f"Cannot evaluate: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("Scientific Calculator  (type 'help' for usage, 'quit' to exit)")
    while True:
        try:
            expr = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not expr:
            continue
        if expr.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if expr.lower() in {"help", "h", "?"}:
            print(HELP_TEXT)
            continue

        try:
            result = evaluate(expr)
            if isinstance(result, int) or (isinstance(result, float) and result.is_integer()):
                print(f"= {int(result)}")
            else:
                print(f"= {result:.10g}")
        except ValueError as err:
            print(f"Error: {err}")


if __name__ == "__main__":
    main()
