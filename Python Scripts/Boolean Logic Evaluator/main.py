"""Boolean Logic Evaluator — CLI tool.

Evaluate boolean expressions, generate truth tables, simplify
using De Morgan's laws, and check logical equivalence.

Usage:
    python main.py
"""

import re


# ---------------------------------------------------------------------------
# Safe boolean evaluator
# ---------------------------------------------------------------------------

# Operator mapping: support natural language and symbol forms
_REPLACEMENTS = [
    (r"\bAND\b",  " and "),
    (r"\bOR\b",   " or "),
    (r"\bNOT\b",  " not "),
    (r"\bXOR\b",  " ^ "),
    (r"\bNAND\b", " nand "),
    (r"\bNOR\b",  " nor "),
    (r"\bXNOR\b", " xnor "),
    (r"&&",       " and "),
    (r"\|\|",     " or "),
    (r"!",        " not "),
    (r"~",        " not "),
]

_VARIABLE_RE = re.compile(r"\b([A-Z])\b")


def _preprocess(expr: str) -> str:
    expr = expr.upper()
    for pattern, repl in _REPLACEMENTS:
        expr = re.sub(pattern, repl, expr, flags=re.IGNORECASE)
    return expr


def _extract_vars(expr: str) -> list[str]:
    return sorted(set(_VARIABLE_RE.findall(expr.upper())))


def _handle_extra_ops(expr: str, values: dict[str, bool]) -> bool:
    """Handle XOR, NAND, NOR, XNOR by string substitution."""
    # Replace known vars first
    for var, val in sorted(values.items(), key=lambda x: -len(x[0])):
        expr = re.sub(r"\b" + var + r"\b", str(val), expr)
    # Handle xnor: ~(A^B)
    # The preprocessed string at this point has nand/nor/xnor/xor as words
    # We'll use eval with only Python builtins for and/or/not
    safe_ns = {"True": True, "False": False, "__builtins__": {}}
    # Quick substitution for remaining Python ops
    try:
        return bool(eval(expr, safe_ns))
    except Exception:
        return False


def evaluate(expr: str, values: dict[str, bool]) -> bool:
    """Evaluate a boolean expression with given variable values."""
    processed = _preprocess(expr)
    # Insert actual Python bool values
    for var, val in sorted(values.items(), key=lambda x: -len(x[0])):
        processed = re.sub(r"\b" + var + r"\b", str(val), processed)
    safe_ns = {"True": True, "False": False}
    try:
        return bool(eval(processed, {"__builtins__": {}}, safe_ns))
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")


def truth_table(expr: str) -> tuple[list[str], list[list]]:
    """Return (variables, rows) for a full truth table."""
    vars_ = _extract_vars(expr)
    n     = len(vars_)
    rows  = []
    for i in range(2 ** n):
        assignment = {vars_[j]: bool((i >> (n - 1 - j)) & 1) for j in range(n)}
        try:
            result = evaluate(expr, assignment)
        except ValueError:
            result = None
        row = [assignment[v] for v in vars_] + [result]
        rows.append(row)
    return vars_, rows


def is_tautology(expr: str) -> bool:
    _, rows = truth_table(expr)
    return all(row[-1] for row in rows)


def is_contradiction(expr: str) -> bool:
    _, rows = truth_table(expr)
    return not any(row[-1] for row in rows)


def are_equivalent(expr1: str, expr2: str) -> bool:
    vars1 = _extract_vars(expr1)
    vars2 = _extract_vars(expr2)
    vars_ = sorted(set(vars1) | set(vars2))
    n     = len(vars_)
    for i in range(2 ** n):
        assignment = {vars_[j]: bool((i >> (n - 1 - j)) & 1) for j in range(n)}
        try:
            r1 = evaluate(expr1, assignment)
            r2 = evaluate(expr2, assignment)
            if r1 != r2:
                return False
        except ValueError:
            return False
    return True


def bool_str(b) -> str:
    if b is None:
        return "ERR"
    return "1" if b else "0"


def print_truth_table(expr: str) -> None:
    vars_, rows = truth_table(expr)
    header = "  " + "  ".join(f"{v:^5}" for v in vars_) + "  │  " + "Result"
    sep    = "  " + "-" * (len(vars_) * 7 + 10)
    print(f"\n{header}")
    print(sep)
    for row in rows:
        vals   = "  ".join(f"{'T' if v else 'F':^5}" for v in row[:-1])
        result = "T" if row[-1] else "F"
        print(f"  {vals}  │  {result}")
    ones = sum(1 for r in rows if r[-1])
    print(f"\n  Satisfying assignments: {ones}/{len(rows)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Boolean Logic Evaluator
------------------------
Operators: AND, OR, NOT, XOR  (or &&, ||, !)
Variables: single uppercase letters A-Z

1. Evaluate expression
2. Generate truth table
3. Check tautology / contradiction
4. Check equivalence of two expressions
5. Single-variable evaluate
0. Quit
"""


def main() -> None:
    print("Boolean Logic Evaluator")
    print("  Example expressions: A AND B, (A OR B) AND NOT C, A XOR B")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            expr = input("  Expression: ").strip()
            if not expr:
                continue
            vars_ = _extract_vars(expr)
            if not vars_:
                print("  No variables found. Use A-Z.")
                continue
            print(f"  Variables: {', '.join(vars_)}")
            values = {}
            for v in vars_:
                val_s = input(f"  {v} (T/F/1/0): ").strip().upper()
                values[v] = val_s in ("T", "1", "TRUE")
            try:
                result = evaluate(expr, values)
                print(f"\n  Result: {'TRUE' if result else 'FALSE'}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "2":
            expr = input("  Expression: ").strip()
            if not expr:
                continue
            try:
                print_truth_table(expr)
                if is_tautology(expr):
                    print("  ✓ This expression is a TAUTOLOGY (always true).")
                elif is_contradiction(expr):
                    print("  ✓ This expression is a CONTRADICTION (always false).")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "3":
            expr = input("  Expression: ").strip()
            if not expr:
                continue
            try:
                if is_tautology(expr):
                    print("\n  TAUTOLOGY — always true.")
                elif is_contradiction(expr):
                    print("\n  CONTRADICTION — always false.")
                else:
                    print("\n  CONTINGENCY — sometimes true, sometimes false.")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "4":
            e1 = input("  Expression 1: ").strip()
            e2 = input("  Expression 2: ").strip()
            if not e1 or not e2:
                continue
            try:
                if are_equivalent(e1, e2):
                    print("\n  ✓ The expressions are LOGICALLY EQUIVALENT.")
                else:
                    print("\n  ✗ The expressions are NOT equivalent.")
                    print("  Showing differences:")
                    vars_ = sorted(set(_extract_vars(e1)) | set(_extract_vars(e2)))
                    n = len(vars_)
                    for i in range(2 ** n):
                        asgn = {vars_[j]: bool((i >> (n - 1 - j)) & 1) for j in range(n)}
                        r1 = evaluate(e1, asgn)
                        r2 = evaluate(e2, asgn)
                        if r1 != r2:
                            asgn_str = ", ".join(f"{k}={'T' if v else 'F'}" for k, v in asgn.items())
                            print(f"    [{asgn_str}]  →  E1={'T' if r1 else 'F'}  E2={'T' if r2 else 'F'}")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "5":
            expr = input("  Expression: ").strip()
            if not expr:
                continue
            vars_ = _extract_vars(expr)
            print(f"  Enter {', '.join(vars_)} as 0s/1s space separated:")
            vals_s = input("  Values: ").strip().split()
            if len(vals_s) != len(vars_):
                print(f"  Need exactly {len(vars_)} values.")
                continue
            values = {v: bool(int(x)) for v, x in zip(vars_, vals_s)}
            try:
                result = evaluate(expr, values)
                print(f"\n  Result: {'1 (TRUE)' if result else '0 (FALSE)'}")
            except Exception as e:
                print(f"  Error: {e}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
