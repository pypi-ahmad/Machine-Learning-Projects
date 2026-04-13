"""Memory Card Game — CLI game.

Flip pairs of cards to find matches. ASCII board, move counter, timer.
Difficulty controls grid size.

Usage:
    python main.py
    python main.py --size 4      # 4×4 grid (8 pairs)
    python main.py --size 6      # 6×6 grid (18 pairs)
"""

import argparse
import random
import time


SYMBOLS = [
    "🍎","🍊","🍋","🍇","🍓","🍑","🍒","🥝",
    "🌸","🌻","🌺","🌼","🦋","🐬","🦁","🐸",
    "⭐","🎯","🎸","🎪","🎨","🎭","🏆","🎲",
    "🌙","☀️","🌊","❄️","🔥","⚡","🌈","💎",
]


def make_board(size: int) -> list[str]:
    n_pairs   = (size * size) // 2
    sym_pool  = SYMBOLS[:n_pairs]
    cards     = sym_pool * 2
    random.shuffle(cards)
    return cards


def display(cards: list[str], revealed: set[int], size: int) -> None:
    print()
    # Column headers
    header = "     " + "  ".join(f"{c+1:^3}" for c in range(size))
    print(header)
    print("     " + "─" * (size * 5))
    for row in range(size):
        row_parts = []
        for col in range(size):
            idx = row * size + col
            if idx in revealed:
                row_parts.append(f" {cards[idx]} ")
            else:
                row_parts.append(" ❓ ")
        print(f"  {row+1:2} │" + " ".join(row_parts))
    print()


def parse_position(inp: str, size: int):
    """Parse 'row col' input to board index."""
    parts = inp.strip().split()
    if len(parts) != 2:
        raise ValueError("Enter row and column separated by space.")
    r, c = int(parts[0]) - 1, int(parts[1]) - 1
    if not (0 <= r < size and 0 <= c < size):
        raise ValueError(f"Row and column must be 1–{size}.")
    return r * size + c


def play(size: int = 4) -> None:
    cards   = make_board(size)
    n       = size * size
    revealed: set[int] = set()
    matched:  set[int] = set()
    moves   = 0
    start   = time.time()
    n_pairs = n // 2

    print(f"\n=== Memory Card Game ===  [{size}×{size}, {n_pairs} pairs]")
    print("  Find all matching pairs!")
    print("  Enter row and column (e.g. '2 3') to flip a card.\n")

    while len(matched) < n:
        display(cards, revealed | matched, size)
        # Pick first card
        while True:
            try:
                inp1 = input("  Flip card 1 (row col): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Game aborted."); return
            if inp1.lower() in ("q","quit"): print("  Quit."); return
            try:
                idx1 = parse_position(inp1, size)
                if idx1 in matched:
                    print("  Already matched!"); continue
                break
            except ValueError as e:
                print(f"  {e}")

        # Temporarily show first card
        revealed_now = revealed | matched | {idx1}
        display(cards, revealed_now, size)

        # Pick second card
        while True:
            try:
                inp2 = input("  Flip card 2 (row col): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Game aborted."); return
            if inp2.lower() in ("q","quit"): print("  Quit."); return
            try:
                idx2 = parse_position(inp2, size)
                if idx2 == idx1:
                    print("  Choose a different card."); continue
                if idx2 in matched:
                    print("  Already matched!"); continue
                break
            except ValueError as e:
                print(f"  {e}")

        moves += 1
        # Show both
        revealed_now = revealed | matched | {idx1, idx2}
        display(cards, revealed_now, size)

        if cards[idx1] == cards[idx2]:
            print(f"  ✅ Match! {cards[idx1]}")
            matched.add(idx1)
            matched.add(idx2)
        else:
            print(f"  ❌ No match ({cards[idx1]} ≠ {cards[idx2]})")
            time.sleep(1.0)

        print(f"  Pairs found: {len(matched)//2}/{n_pairs}  |  Moves: {moves}\n")

    elapsed = time.time() - start
    print(f"\n  🎉 You found all {n_pairs} pairs!")
    print(f"  Moves:   {moves}")
    print(f"  Time:    {elapsed:.1f}s")
    ideal  = n_pairs
    rating = "⭐⭐⭐" if moves <= ideal * 1.5 else ("⭐⭐" if moves <= ideal * 2.5 else "⭐")
    print(f"  Rating:  {rating}")


def main():
    parser = argparse.ArgumentParser(description="Memory Card Game")
    parser.add_argument("--size", type=int, choices=[2,4,6], default=4,
                        help="Grid size (2,4,6 — must be even, default 4)")
    args = parser.parse_args()
    while True:
        play(args.size)
        again = input("\n  Play again? (y/n): ").strip().lower()
        if again != "y": break
    print("  Thanks for playing!")


if __name__ == "__main__":
    main()
