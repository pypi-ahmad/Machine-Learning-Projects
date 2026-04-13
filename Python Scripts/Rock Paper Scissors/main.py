"""Rock Paper Scissors — CLI game.

Play against the computer with score tracking,
win/loss streaks, and optional best-of-N mode.

Usage:
    python main.py
    python main.py --rounds 5
"""

import argparse
import random
import sys

CHOICES   = ["rock", "paper", "scissors"]
EMOJI     = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
BEATS     = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
SHORTCUTS = {"r": "rock", "p": "paper", "s": "scissors"}


def get_winner(player: str, cpu: str) -> str:
    if player == cpu:
        return "draw"
    return "player" if BEATS[player] == cpu else "cpu"


def display_result(player: str, cpu: str, winner: str) -> None:
    p_e = EMOJI[player]
    c_e = EMOJI[cpu]
    print(f"\n  You: {p_e} {player.upper():<10}  CPU: {c_e} {cpu.upper():<10}", end="")
    if winner == "draw":
        print("→ DRAW!")
    elif winner == "player":
        print(f"→ {BEATS[player].upper()} beats {cpu.upper()} — YOU WIN! 🎉")
    else:
        print(f"→ {BEATS[cpu].upper()} beats {player.upper()} — CPU WINS! 🤖")


def play(best_of: int = None) -> None:
    scores      = {"player": 0, "cpu": 0, "draw": 0}
    win_streak  = 0
    lose_streak = 0
    max_ws      = 0
    game_num    = 0
    target      = (best_of + 1) // 2 if best_of else None

    print("=== Rock Paper Scissors ===")
    if best_of:
        print(f"Best of {best_of} — first to {target} wins!\n")
    print("Enter: r=rock  p=paper  s=scissors  q=quit\n")

    while True:
        if target and (scores["player"] >= target or scores["cpu"] >= target):
            break

        game_num += 1
        prompt = f"  Round {game_num}" + (f"/{best_of}" if best_of else "") + " > "
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice in ("q", "quit", "exit"):
            break
        if choice in SHORTCUTS:
            choice = SHORTCUTS[choice]
        if choice not in CHOICES:
            print("  Invalid. Use r / p / s")
            game_num -= 1
            continue

        cpu    = random.choice(CHOICES)
        winner = get_winner(choice, cpu)
        display_result(choice, cpu, winner)
        scores[winner] += 1

        if winner == "player":
            win_streak  += 1
            lose_streak  = 0
            max_ws       = max(max_ws, win_streak)
            if win_streak >= 3:
                print(f"  🔥 Win streak: {win_streak}!")
        elif winner == "cpu":
            lose_streak += 1
            win_streak   = 0
            if lose_streak >= 3:
                print(f"  😬 Lose streak: {lose_streak}…")
        else:
            win_streak = lose_streak = 0

        print(f"  Score — You: {scores['player']}  CPU: {scores['cpu']}  Draws: {scores['draw']}\n")

    # Final report
    total = scores["player"] + scores["cpu"] + scores["draw"]
    print("\n  ── Final Results ──")
    print(f"  Rounds played:  {total}")
    print(f"  Your wins:      {scores['player']}")
    print(f"  CPU wins:       {scores['cpu']}")
    print(f"  Draws:          {scores['draw']}")
    if max_ws > 1:
        print(f"  Best win streak: {max_ws}")
    if total:
        win_pct = scores["player"] / total * 100
        print(f"  Your win rate:  {win_pct:.1f}%")
    if best_of:
        if scores["player"] >= target:
            print("  🏆 YOU WIN THE SERIES!")
        elif scores["cpu"] >= target:
            print("  🤖 CPU WINS THE SERIES!")
        else:
            print("  Series ended early.")


def main():
    parser = argparse.ArgumentParser(description="Rock Paper Scissors")
    parser.add_argument("--rounds", type=int, default=None, metavar="N",
                        help="Best-of-N mode (e.g. --rounds 5)")
    args = parser.parse_args()
    play(best_of=args.rounds)


if __name__ == "__main__":
    main()
