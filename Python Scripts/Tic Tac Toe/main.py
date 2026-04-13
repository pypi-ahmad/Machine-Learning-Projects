"""Tic Tac Toe — CLI game.

Play against another human or an unbeatable AI (minimax).
Supports 3×3 classic board.

Usage:
    python main.py
    python main.py --ai       # play vs AI
    python main.py --ai --first cpu
"""

import argparse
import sys


EMPTY = "."
X, O  = "X", "O"

WINS = [
    (0,1,2),(3,4,5),(6,7,8),   # rows
    (0,3,6),(1,4,7),(2,5,8),   # cols
    (0,4,8),(2,4,6),           # diags
]


def empty_board():
    return [EMPTY] * 9


def print_board(board):
    labels = "123456789"
    print()
    for row in range(3):
        cells = []
        for col in range(3):
            i = row * 3 + col
            c = board[i]
            cells.append(c if c != EMPTY else labels[i])
        print("  " + " │ ".join(cells))
        if row < 2:
            print("  ──┼───┼──")
    print()


def check_winner(board):
    for a, b, c in WINS:
        if board[a] == board[b] == board[c] != EMPTY:
            return board[a]
    return None


def is_full(board):
    return EMPTY not in board


def minimax(board, is_max: bool, alpha: float, beta: float) -> int:
    winner = check_winner(board)
    if winner == O: return  10
    if winner == X: return -10
    if is_full(board): return 0

    if is_max:
        best = -100
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = O
                best = max(best, minimax(board, False, alpha, beta))
                board[i] = EMPTY
                alpha = max(alpha, best)
                if beta <= alpha: break
        return best
    else:
        best = 100
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = X
                best = min(best, minimax(board, True, alpha, beta))
                board[i] = EMPTY
                beta = min(beta, best)
                if beta <= alpha: break
        return best


def ai_move(board) -> int:
    best_val, best_move = -100, -1
    for i in range(9):
        if board[i] == EMPTY:
            board[i] = O
            val = minimax(board, False, -100, 100)
            board[i] = EMPTY
            if val > best_val:
                best_val, best_move = val, i
    return best_move


def get_human_move(board, player: str) -> int:
    while True:
        try:
            inp = input(f"  Player {player}, enter position (1-9): ").strip()
            n   = int(inp) - 1
            if 0 <= n <= 8 and board[n] == EMPTY:
                return n
            print("  Invalid move.")
        except (ValueError, KeyboardInterrupt):
            print("  Invalid input.")


def play_game(vs_ai: bool, human_first: bool) -> None:
    board = empty_board()
    if vs_ai:
        human  = X if human_first else O
        ai_sym = O if human_first else X
        print(f"  You are {human}.  AI is {ai_sym}.")
    else:
        human = ai_sym = None   # unused

    current = X    # X always goes first

    while True:
        print_board(board)
        w = check_winner(board)
        if w:
            print(f"  {'You win! 🎉' if vs_ai and w == human else ('AI wins! 🤖' if vs_ai else f'Player {w} wins!')}")
            break
        if is_full(board):
            print("  It's a draw! 🤝")
            break

        if vs_ai and current == ai_sym:
            print("  AI is thinking…")
            idx = ai_move(board)
            board[idx] = current
        else:
            idx = get_human_move(board, current)
            board[idx] = current

        current = O if current == X else X

    print_board(board)


def main():
    parser = argparse.ArgumentParser(description="Tic Tac Toe")
    parser.add_argument("--ai",    action="store_true", help="Play vs AI")
    parser.add_argument("--first", choices=["human","cpu"], default="human",
                        help="Who goes first (default: human)")
    args = parser.parse_args()

    print("=== Tic Tac Toe ===")
    wins = {"human": 0, "cpu": 0, "draw": 0}

    while True:
        play_game(vs_ai=args.ai, human_first=(args.first == "human"))
        again = input("  Play again? (y/n): ").strip().lower()
        if again != "y":
            break

    print("  Thanks for playing!")


if __name__ == "__main__":
    main()
