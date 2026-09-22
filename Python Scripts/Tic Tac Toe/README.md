# Tic Tac Toe

A terminal 3×3 Tic Tac Toe game for two players or one player against a minimax AI.

## Run it

```powershell
uv sync
uv run python main.py
```

Enter a board position from `1` to `9` when prompted. The numbers match the empty-cell labels shown on the board.

## Play against the AI

```powershell
uv run python main.py --ai
uv run python main.py --ai --first cpu
```

The AI uses minimax with alpha-beta pruning and plays as `O`. With `--first cpu`, the AI makes the first move; otherwise the human starts as `X`.

`tic_tac_toe.py` remains as a legacy example. Run `main.py` for the maintained game.

## Dependencies

- Python 3.14+
- No third-party packages
