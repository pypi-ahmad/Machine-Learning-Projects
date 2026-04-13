"""Snake Game — Tkinter GUI game.

Classic Snake: eat food to grow, avoid walls and yourself.
Arrow keys to steer, P to pause, R to restart.

Usage:
    python main.py
"""

import tkinter as tk
import random
import sys

CELL   = 20
COLS   = 30
ROWS   = 25
WIDTH  = COLS * CELL
HEIGHT = ROWS * CELL
DELAY  = 120     # ms per tick


class SnakeGame:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Snake")
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#1a1a2e")
        self.canvas.pack()

        self.score_var = tk.StringVar(value="Score: 0  |  High: 0")
        tk.Label(root, textvariable=self.score_var, font=("Courier", 13, "bold"),
                 bg="#16213e", fg="#e2e2e2").pack(fill=tk.X)

        root.configure(bg="#16213e")
        root.bind("<Left>",  lambda e: self.set_dir(-1, 0))
        root.bind("<Right>", lambda e: self.set_dir( 1, 0))
        root.bind("<Up>",    lambda e: self.set_dir( 0,-1))
        root.bind("<Down>",  lambda e: self.set_dir( 0, 1))
        root.bind("<p>",     lambda e: self.toggle_pause())
        root.bind("<r>",     lambda e: self.reset())

        self.high_score = 0
        self.reset()

    def reset(self):
        cx, cy   = COLS // 2, ROWS // 2
        self.snake    = [(cx, cy), (cx-1, cy), (cx-2, cy)]
        self.direction = (1, 0)
        self.next_dir  = (1, 0)
        self.score     = 0
        self.paused    = False
        self.game_over = False
        self.place_food()
        self.update_score()
        self.draw()
        self.root.after(DELAY, self.tick)

    def place_food(self):
        occupied = set(self.snake)
        free = [(c, r) for c in range(COLS) for r in range(ROWS) if (c, r) not in occupied]
        self.food = random.choice(free) if free else (0, 0)

    def set_dir(self, dx, dy):
        # Prevent reversing
        if (dx, dy) != (-self.direction[0], -self.direction[1]):
            self.next_dir = (dx, dy)

    def toggle_pause(self):
        if self.game_over: return
        self.paused = not self.paused
        if not self.paused:
            self.root.after(DELAY, self.tick)
        else:
            self.draw()

    def tick(self):
        if self.paused or self.game_over: return
        self.direction = self.next_dir
        hx, hy = self.snake[0]
        dx, dy = self.direction
        nx, ny = hx + dx, hy + dy

        if not (0 <= nx < COLS and 0 <= ny < ROWS) or (nx, ny) in self.snake:
            self.game_over = True
            self.high_score = max(self.high_score, self.score)
            self.draw()
            return

        self.snake.insert(0, (nx, ny))
        if (nx, ny) == self.food:
            self.score += 10
            self.update_score()
            self.place_food()
        else:
            self.snake.pop()

        self.draw()
        self.root.after(DELAY, self.tick)

    def draw(self):
        c = self.canvas
        c.delete("all")

        # Grid lines (subtle)
        for col in range(COLS):
            c.create_line(col*CELL, 0, col*CELL, HEIGHT, fill="#1e1e3a", width=1)
        for row in range(ROWS):
            c.create_line(0, row*CELL, WIDTH, row*CELL, fill="#1e1e3a", width=1)

        # Food
        fx, fy = self.food
        c.create_oval(fx*CELL+2, fy*CELL+2, fx*CELL+CELL-2, fy*CELL+CELL-2,
                      fill="#ff6b6b", outline="#ff4444", width=2)

        # Snake
        for i, (sx, sy) in enumerate(self.snake):
            color = "#4ade80" if i == 0 else ("#22c55e" if i < 3 else "#16a34a")
            c.create_rectangle(sx*CELL+1, sy*CELL+1, sx*CELL+CELL-1, sy*CELL+CELL-1,
                                fill=color, outline="#15803d")

        # Overlays
        if self.paused:
            c.create_rectangle(0, 0, WIDTH, HEIGHT, fill="", stipple="gray50")
            c.create_text(WIDTH//2, HEIGHT//2, text="PAUSED\nPress P to resume",
                          fill="white", font=("Courier", 22, "bold"), justify=tk.CENTER)
        if self.game_over:
            c.create_rectangle(WIDTH//4, HEIGHT//3, 3*WIDTH//4, 2*HEIGHT//3,
                                fill="#1a1a2e", outline="#ff4444", width=3)
            c.create_text(WIDTH//2, HEIGHT//2 - 15,
                          text=f"GAME OVER\nScore: {self.score}",
                          fill="#ff6b6b", font=("Courier", 20, "bold"), justify=tk.CENTER)
            c.create_text(WIDTH//2, HEIGHT//2 + 30,
                          text="Press R to restart",
                          fill="#a0a0ff", font=("Courier", 13))

    def update_score(self):
        self.score_var.set(f"Score: {self.score}  |  High: {self.high_score}  |  P=pause  R=restart")


def main():
    root = tk.Tk()
    SnakeGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
