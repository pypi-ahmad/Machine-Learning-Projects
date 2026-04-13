"""Word Scramble — CLI game.

Unscramble random words. Timed mode, hints, difficulty levels.

Usage:
    python main.py
    python main.py --difficulty hard --rounds 10
"""

import argparse
import random
import time


WORD_LISTS = {
    "easy": [
        "cat","dog","sun","run","big","hat","cup","fly","red","bus",
        "map","ant","egg","ice","oak","pit","gem","axe","web","tin",
    ],
    "medium": [
        "python","bridge","flower","castle","planet","mirror","island",
        "frozen","button","garden","silver","purple","orange","bottle",
        "finger","hunter","basket","winter","market","jungle",
    ],
    "hard": [
        "psychology","knowledge","equivalent","mysterious","phenomenon",
        "laboratory","complicated","performance","environment","magnificent",
        "comfortable","electricity","independent","communication","achievement",
        "temperature","engineering","celebration","approximately","circumstances",
    ],
}
ALL_WORDS = [w for ws in WORD_LISTS.values() for w in ws]


def scramble(word: str) -> str:
    """Shuffle word letters, ensuring result differs from original."""
    letters = list(word)
    for _ in range(20):
        random.shuffle(letters)
        if "".join(letters) != word:
            return "".join(letters)
    return "".join(reversed(word))


def get_hint(word: str, revealed: int) -> str:
    """Show first `revealed` letters, rest as underscores."""
    return word[:revealed] + "_" * (len(word) - revealed)


def play_round(word: str, time_limit: int, round_num: int, total: int) -> dict:
    scrambled  = scramble(word)
    hints_used = 0
    start      = time.time()
    max_score  = 10

    print(f"\n  Round {round_num}/{total}  ─  {'─'*30}")
    print(f"  Scrambled: {scrambled.upper()}")
    if time_limit:
        print(f"  Time limit: {time_limit}s  |  Hints cost 2 pts each\n")

    while True:
        elapsed = time.time() - start
        if time_limit and elapsed >= time_limit:
            print(f"\n  ⏰ Time's up! The word was: {word.upper()}")
            return {"correct": False, "hints": hints_used, "time": elapsed, "score": 0}

        try:
            inp = input("  Your answer (or 'hint' / 'skip'): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return {"correct": False, "hints": hints_used,
                    "time": time.time() - start, "score": 0}

        if inp == word.lower():
            elapsed = time.time() - start
            score   = max(1, max_score - hints_used * 2)
            if time_limit:
                time_bonus = max(0, int((time_limit - elapsed) / time_limit * 3))
                score = min(max_score, score + time_bonus)
            print(f"  ✅ Correct! +{score} pts  ({elapsed:.1f}s)")
            return {"correct": True, "hints": hints_used, "time": elapsed, "score": score}
        elif inp == "hint":
            hints_used  = min(hints_used + 1, len(word) - 1)
            print(f"  Hint: {get_hint(word, hints_used + 1).upper()}")
        elif inp == "skip":
            print(f"  Skipped. The word was: {word.upper()}")
            return {"correct": False, "hints": hints_used,
                    "time": time.time() - start, "score": 0}
        else:
            print(f"  ❌ Wrong — try again! ({len(word)} letters)")


def play(difficulty: str = "medium", n_rounds: int = 5, time_limit: int = 30) -> None:
    pool     = WORD_LISTS.get(difficulty, WORD_LISTS["medium"])
    words    = random.sample(pool, min(n_rounds, len(pool)))
    total_score = 0
    correct_count = 0

    print(f"\n=== Word Scramble ===  [{difficulty.upper()}]")
    print(f"  Rounds: {n_rounds}  |  Time limit: {time_limit}s per word\n")

    for i, word in enumerate(words, 1):
        result = play_round(word, time_limit, i, n_rounds)
        total_score   += result["score"]
        correct_count += result["correct"]

    print("\n  ── Final Results ──")
    print(f"  Correct:     {correct_count}/{n_rounds}")
    print(f"  Total score: {total_score}")
    accuracy = correct_count / n_rounds * 100
    print(f"  Accuracy:    {accuracy:.0f}%")
    if accuracy == 100:
        print("  🏆 Perfect round!")
    elif accuracy >= 70:
        print("  👍 Well done!")
    else:
        print("  📚 Keep practicing!")


def main():
    parser = argparse.ArgumentParser(description="Word Scramble Game")
    parser.add_argument("--difficulty", choices=["easy","medium","hard"], default="medium")
    parser.add_argument("--rounds",     type=int, default=5, help="Number of rounds")
    parser.add_argument("--time",       type=int, default=30, help="Seconds per word (0=no limit)")
    args = parser.parse_args()
    play(args.difficulty, args.rounds, args.time)


if __name__ == "__main__":
    main()
