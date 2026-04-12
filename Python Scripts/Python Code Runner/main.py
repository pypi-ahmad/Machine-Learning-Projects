"""Python Code Runner — CLI tool.

Run Python snippets or files interactively with timeout protection,
captured output, and a saved-snippet library.

Usage:
    python main.py
    python main.py script.py
"""

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

SNIPPETS_FILE = Path("code_snippets.json")

SAMPLE_SNIPPETS = {
    "hello": 'print("Hello, World!")',
    "fibonacci": textwrap.dedent("""\
        def fib(n):
            a, b = 0, 1
            for _ in range(n):
                print(a, end=" ")
                a, b = b, a + b
        fib(15)
    """),
    "list-comp": "[x**2 for x in range(1, 11)]",
    "timeit": textwrap.dedent("""\
        import time
        start = time.perf_counter()
        total = sum(range(1_000_000))
        elapsed = time.perf_counter() - start
        print(f"sum = {total:,}  time = {elapsed*1000:.2f} ms")
    """),
}


# ---------------------------------------------------------------------------
# Snippet persistence
# ---------------------------------------------------------------------------

def load_snippets() -> dict[str, str]:
    if SNIPPETS_FILE.exists():
        try:
            return json.loads(SNIPPETS_FILE.read_text())
        except Exception:
            pass
    return dict(SAMPLE_SNIPPETS)


def save_snippets(snippets: dict[str, str]) -> None:
    SNIPPETS_FILE.write_text(json.dumps(snippets, indent=2))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_code(code: str, timeout: int = 10) -> tuple[str, str, float]:
    """Execute code string in subprocess; return (stdout, stderr, elapsed_s)."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.perf_counter() - t0
        return result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        return "", f"Timeout after {timeout}s", time.perf_counter() - t0
    except Exception as e:
        return "", str(e), time.perf_counter() - t0


def run_file(path: Path, timeout: int = 30) -> tuple[str, str, float]:
    """Execute a .py file in subprocess."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.perf_counter() - t0
        return result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        return "", f"Timeout after {timeout}s", time.perf_counter() - t0
    except Exception as e:
        return "", str(e), time.perf_counter() - t0


def print_result(stdout: str, stderr: str, elapsed: float):
    print(f"\n  ─── Output ({'%.3f' % elapsed}s) ───")
    if stdout:
        for line in stdout.splitlines():
            print(f"  {line}")
    else:
        print("  (no output)")
    if stderr:
        print("\n  ─── Errors ───")
        for line in stderr.splitlines():
            print(f"  {line}")


def read_multiline(prompt: str = "  Code (blank line to run, 'cancel' to abort):") -> str | None:
    print(prompt)
    lines = []
    while True:
        line = input("  ... ")
        if line.lower() == "cancel":
            return None
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Python Code Runner
------------------
1. Run a code snippet (type inline)
2. Run a saved snippet
3. Save current code as snippet
4. List saved snippets
5. Run a .py file
6. Delete a snippet
0. Quit
"""


def main():
    # Direct file mode
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.exists():
            print(f"Running {path}…")
            stdout, stderr, elapsed = run_file(path)
            print_result(stdout, stderr, elapsed)
        else:
            print(f"File not found: {path}")
        return

    snippets = load_snippets()
    last_code: str | None = None

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            code = read_multiline()
            if code:
                last_code = code
                print("\n  Running…")
                stdout, stderr, elapsed = run_code(code)
                print_result(stdout, stderr, elapsed)

        elif choice == "2":
            if not snippets:
                print("  No snippets saved.")
                continue
            print("\n  Saved snippets:")
            for name in sorted(snippets):
                print(f"    {name}")
            name = input("  Snippet name: ").strip()
            if name in snippets:
                code = snippets[name]
                last_code = code
                print(f"\n  --- {name} ---\n{textwrap.indent(code, '  ')}")
                stdout, stderr, elapsed = run_code(code)
                print_result(stdout, stderr, elapsed)
            else:
                print("  Snippet not found.")

        elif choice == "3":
            if not last_code:
                print("  No code to save. Run something first.")
                continue
            name = input("  Save as (name): ").strip()
            if name:
                snippets[name] = last_code
                save_snippets(snippets)
                print(f"  Saved as '{name}'.")

        elif choice == "4":
            if not snippets:
                print("  No snippets.")
            else:
                print(f"\n  {len(snippets)} snippet(s):")
                for name, code in sorted(snippets.items()):
                    preview = code.splitlines()[0][:60]
                    print(f"    {name:<20} {preview}")

        elif choice == "5":
            path_str = input("  .py file path: ").strip()
            path = Path(path_str)
            if path.exists():
                stdout, stderr, elapsed = run_file(path)
                print_result(stdout, stderr, elapsed)
            else:
                print(f"  File not found: {path}")

        elif choice == "6":
            print("\n  Snippets:", ", ".join(sorted(snippets)))
            name = input("  Delete snippet: ").strip()
            if name in snippets:
                del snippets[name]
                save_snippets(snippets)
                print(f"  Deleted '{name}'.")
            else:
                print("  Not found.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
