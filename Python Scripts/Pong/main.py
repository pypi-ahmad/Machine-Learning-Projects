"""Pong — Tkinter GUI game.

Classic two-player Pong. Player 1: W/S keys. Player 2: Up/Down arrows.
First to 10 wins. Press Space to start/pause, R to restart.

Usage:
    python main.py
    python main.py --speed 6    # ball speed multiplier
    python main.py --vs-ai      # play against CPU
"""

import argparse
import math
import random
import tkinter as tk


WIDTH, HEIGHT = 800, 500
PAD_W, PAD_H  = 12, 80
BALL_R        = 8
PAD_SPEED     = 6
WIN_SCORE     = 10
BG_COLOR      = "#111111"
FG_COLOR      = "#ffffff"
BALL_COLOR    = "#ffe066"
P1_COLOR      = "#66aaff"
P2_COLOR      = "#ff7766"


class Pong:
    def __init__(self, root: tk.Tk, ball_speed: float = 5.0, vs_ai: bool = False):
        self.root      = root
        self.ball_spd  = ball_speed
        self.vs_ai     = vs_ai
        root.title("Pong")
        root.resizable(False, False)
        root.configure(bg=BG_COLOR)

        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=BG_COLOR,
                                highlightthickness=0)
        self.canvas.pack()

        self.info_var = tk.StringVar()
        tk.Label(root, textvariable=self.info_var,
                 font=("Courier", 11), bg=BG_COLOR, fg="#aaaaaa").pack()

        # Key state
        self.keys: set[str] = set()
        root.bind("<KeyPress>",   lambda e: self.keys.add(e.keysym))
        root.bind("<KeyRelease>", lambda e: self.keys.discard(e.keysym))
        root.bind("<space>",      lambda e: self.toggle_pause())
        root.bind("<r>",          lambda e: self.reset())

        self.reset()

    def reset(self):
        self.score   = [0, 0]
        self.paused  = True
        self.over    = False
        # Paddle positions (top-y)
        self.py      = [HEIGHT//2 - PAD_H//2, HEIGHT//2 - PAD_H//2]
        self.launch_ball()
        self.draw()
        self.info_var.set("Space = start  |  W/S = P1  |  ↑↓ = P2  |  R = restart")

    def launch_ball(self):
        angle  = random.uniform(-math.pi/4, math.pi/4)
        if random.random() < 0.5: angle += math.pi
        self.bx = WIDTH / 2
        self.by = HEIGHT / 2
        self.vx = self.ball_spd * math.cos(angle)
        self.vy = self.ball_spd * math.sin(angle)

    def toggle_pause(self):
        if self.over: return
        self.paused = not self.paused
        if not self.paused:
            self.root.after(16, self.tick)

    def tick(self):
        if self.paused or self.over: return
        self.move_paddles()
        self.move_ball()
        self.draw()
        self.root.after(16, self.tick)

    def move_paddles(self):
        # Player 1 (left)
        if "w" in self.keys or "W" in self.keys:
            self.py[0] = max(0, self.py[0] - PAD_SPEED)
        if "s" in self.keys or "S" in self.keys:
            self.py[0] = min(HEIGHT - PAD_H, self.py[0] + PAD_SPEED)

        # Player 2 / AI (right)
        if self.vs_ai:
            mid = self.py[1] + PAD_H / 2
            if self.bx > WIDTH / 2:    # only track when ball coming right
                if self.by < mid - 5:
                    self.py[1] = max(0, self.py[1] - PAD_SPEED * 0.9)
                elif self.by > mid + 5:
                    self.py[1] = min(HEIGHT - PAD_H, self.py[1] + PAD_SPEED * 0.9)
        else:
            if "Up" in self.keys:
                self.py[1] = max(0, self.py[1] - PAD_SPEED)
            if "Down" in self.keys:
                self.py[1] = min(HEIGHT - PAD_H, self.py[1] + PAD_SPEED)

    def move_ball(self):
        self.bx += self.vx
        self.by += self.vy

        # Top / bottom bounce
        if self.by - BALL_R <= 0:
            self.by = BALL_R; self.vy = abs(self.vy)
        if self.by + BALL_R >= HEIGHT:
            self.by = HEIGHT - BALL_R; self.vy = -abs(self.vy)

        # Paddle collision — left
        if (self.bx - BALL_R <= PAD_W + 10 and
                self.py[0] <= self.by <= self.py[0] + PAD_H and self.vx < 0):
            self.vx = abs(self.vx) * 1.03
            offset  = (self.by - (self.py[0] + PAD_H/2)) / (PAD_H/2)
            self.vy = offset * self.ball_spd * 1.2
            self.bx = PAD_W + 10 + BALL_R

        # Paddle collision — right
        if (self.bx + BALL_R >= WIDTH - PAD_W - 10 and
                self.py[1] <= self.by <= self.py[1] + PAD_H and self.vx > 0):
            self.vx = -abs(self.vx) * 1.03
            offset  = (self.by - (self.py[1] + PAD_H/2)) / (PAD_H/2)
            self.vy = offset * self.ball_spd * 1.2
            self.bx = WIDTH - PAD_W - 10 - BALL_R

        # Cap speed
        spd = math.hypot(self.vx, self.vy)
        if spd > self.ball_spd * 3:
            self.vx = self.vx / spd * self.ball_spd * 3
            self.vy = self.vy / spd * self.ball_spd * 3

        # Score
        if self.bx < 0:
            self.score[1] += 1
            self.check_win(1)
        elif self.bx > WIDTH:
            self.score[0] += 1
            self.check_win(0)

    def check_win(self, p: int):
        if self.score[p] >= WIN_SCORE:
            self.over = True
        else:
            self.launch_ball()
            self.paused = True

    def draw(self):
        c = self.canvas
        c.delete("all")

        # Center line
        for y in range(0, HEIGHT, 20):
            c.create_line(WIDTH//2, y, WIDTH//2, y+10, fill="#333333", width=2)

        # Scores
        c.create_text(WIDTH//4, 40, text=str(self.score[0]),
                      fill=P1_COLOR, font=("Courier", 36, "bold"))
        c.create_text(3*WIDTH//4, 40, text=str(self.score[1]),
                      fill=P2_COLOR, font=("Courier", 36, "bold"))

        # Paddles
        px = 10
        c.create_rectangle(px, self.py[0], px+PAD_W, self.py[0]+PAD_H,
                            fill=P1_COLOR, outline="")
        c.create_rectangle(WIDTH-px-PAD_W, self.py[1],
                            WIDTH-px, self.py[1]+PAD_H,
                            fill=P2_COLOR, outline="")

        # Ball
        if not self.over:
            c.create_oval(self.bx-BALL_R, self.by-BALL_R,
                          self.bx+BALL_R, self.by+BALL_R,
                          fill=BALL_COLOR, outline="")

        # Pause / game over overlay
        if self.paused and not self.over:
            c.create_text(WIDTH//2, HEIGHT//2, text="SPACE to serve",
                          fill="white", font=("Courier", 18))
        if self.over:
            winner = "Player 1" if self.score[0] >= WIN_SCORE else \
                     ("CPU" if self.vs_ai else "Player 2")
            c.create_text(WIDTH//2, HEIGHT//2 - 20,
                          text=f"{winner} WINS!", fill=BALL_COLOR,
                          font=("Courier", 30, "bold"))
            c.create_text(WIDTH//2, HEIGHT//2 + 30,
                          text="Press R to play again",
                          fill="white", font=("Courier", 14))


def main():
    parser = argparse.ArgumentParser(description="Pong")
    parser.add_argument("--speed",  type=float, default=5.0, help="Ball speed (default 5)")
    parser.add_argument("--vs-ai",  action="store_true",     help="Play against CPU")
    args = parser.parse_args()

    root = tk.Tk()
    Pong(root, ball_speed=args.speed, vs_ai=args.vs_ai)
    root.mainloop()


if __name__ == "__main__":
    main()
