"""Typing Speed Test — CLI tool.

Measures typing speed (WPM), accuracy, and consistency.
Includes multiple difficulty levels and result history.

Usage:
    python main.py
"""

import json
import random
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Text samples
# ---------------------------------------------------------------------------

EASY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "The five boxing wizards jump quickly.",
]

MEDIUM_TEXTS = [
    "Programming is the art of telling a computer what to do in a language it understands. The best programmers write code that humans can read too.",
    "To be or not to be, that is the question. Whether it is nobler in the mind to suffer the slings and arrows of outrageous fortune.",
    "In the beginning was the Word, and the Word was with God. The universe came into being at the dawn of time with a single moment of creation.",
]

HARD_TEXTS = [
    "The asymptotic complexity of an algorithm represents its behavior in the limit as the input size grows. Big-O notation provides an upper bound on the growth rate.",
    "Quantum entanglement is a physical phenomenon that occurs when a pair of particles interact in ways such that the quantum state of each particle cannot be described independently.",
]


def get_text(level: str) -> str:
    if level == "easy":
        return random.choice(EASY_TEXTS)
    elif level == "hard":
        return random.choice(HARD_TEXTS)
    return random.choice(MEDIUM_TEXTS)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_wpm(typed: str, elapsed: float) -> float:
    words = len(typed.split())
    return words / elapsed * 60 if elapsed > 0 else 0


def calculate_accuracy(original: str, typed: str) -> float:
    if not original:
        return 0.0
    correct = sum(1 for a, b in zip(original, typed) if a == b)
    return correct / len(original) * 100


def calculate_net_wpm(raw_wpm: float, errors: int, elapsed: float) -> float:
    """Net WPM = (raw - errors per minute)."""
    error_penalty = errors / elapsed * 60 if elapsed > 0 else 0
    return max(0, raw_wpm - error_penalty)


def count_errors(original: str, typed: str) -> int:
    return sum(1 for a, b in zip(original, typed) if a != b) + abs(len(original) - len(typed))


def highlight_errors(original: str, typed: str) -> str:
    """Return typed text with ANSI color highlighting errors."""
    GREEN = "\033[32m"
    RED   = "\033[31m"
    RESET = "\033[0m"
    result = []
    for i, ch in enumerate(typed):
        if i < len(original) and ch == original[i]:
            result.append(f"{GREEN}{ch}{RESET}")
        else:
            result.append(f"{RED}{ch}{RESET}")
    return "".join(result)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

HISTORY_FILE = Path("typing_history.json")


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    return []


def save_result(result: dict) -> None:
    history = load_history()
    history.append(result)
    HISTORY_FILE.write_text(json.dumps(history[-50:], indent=2))  # keep last 50


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Typing Speed Test
-----------------
1. Quick test (medium)
2. Choose difficulty
3. Custom text
4. View history / best scores
0. Quit
"""


def run_test(text: str) -> dict:
    """Run a single typing test and return results."""
    print(f"\n  {'─'*60}")
    print(f"  {text}")
    print(f"  {'─'*60}")
    print("  Press Enter when ready to start...")
    input()

    print("  Start typing! (press Enter when done)\n")
    start = time.time()
    typed = input("  > ")
    elapsed = time.time() - start

    errors   = count_errors(text, typed)
    raw_wpm  = calculate_wpm(typed, elapsed)
    net_wpm  = calculate_net_wpm(raw_wpm, errors, elapsed)
    accuracy = calculate_accuracy(text, typed)

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wpm":       round(net_wpm, 1),
        "raw_wpm":   round(raw_wpm, 1),
        "accuracy":  round(accuracy, 1),
        "errors":    errors,
        "time_s":    round(elapsed, 2),
        "chars":     len(text),
    }

    print(f"\n  {highlight_errors(text, typed)}\n")
    print(f"  ┌─────────────────────────────┐")
    print(f"  │  WPM (net)  : {net_wpm:>6.1f}        │")
    print(f"  │  WPM (raw)  : {raw_wpm:>6.1f}        │")
    print(f"  │  Accuracy   : {accuracy:>5.1f}%        │")
    print(f"  │  Errors     : {errors:>6}         │")
    print(f"  │  Time       : {elapsed:>5.2f}s        │")
    print(f"  └─────────────────────────────┘")

    return result


def main() -> None:
    print("Typing Speed Test")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            text = get_text("medium")
            result = run_test(text)
            save_result(result)

        elif choice == "2":
            print("  Difficulty: (e)asy / (m)edium / (h)ard")
            level = input("  Level: ").strip().lower()
            if level.startswith("e"):
                level = "easy"
            elif level.startswith("h"):
                level = "hard"
            else:
                level = "medium"
            text = get_text(level)
            result = run_test(text)
            save_result(result)

        elif choice == "3":
            text = input("  Enter your text: ").strip()
            if not text:
                continue
            result = run_test(text)
            save_result(result)

        elif choice == "4":
            history = load_history()
            if not history:
                print("  No history yet.")
                continue
            print(f"\n  Last {min(10, len(history))} results:")
            print(f"  {'Date':>20}  {'WPM':>6}  {'Acc%':>6}  {'Errors':>6}")
            print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}")
            for r in history[-10:]:
                print(f"  {r['timestamp']:>20}  {r['wpm']:>6.1f}  {r['accuracy']:>5.1f}%  {r['errors']:>6}")
            best = max(history, key=lambda x: x["wpm"])
            print(f"\n  🏆 Best: {best['wpm']} WPM on {best['timestamp']}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
