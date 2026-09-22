"""A small turtle-graphics fidget spinner controlled with the spacebar."""

import turtle


ARM_LENGTH = 100
DOT_SIZE = 120
FRAME_DELAY_MS = 20
FLICK_MOMENTUM = 10
MAX_MOMENTUM = 720
state = {"turn": 0}


def spinner() -> None:
    """Draw the spinner at its current angle."""
    turtle.clear()
    turtle.right(state["turn"] / 10)
    turtle.forward(ARM_LENGTH)
    turtle.dot(DOT_SIZE, "red")
    turtle.back(ARM_LENGTH)
    turtle.right(120)
    turtle.forward(ARM_LENGTH)
    turtle.dot(DOT_SIZE, "green")
    turtle.back(ARM_LENGTH)
    turtle.right(120)
    turtle.forward(ARM_LENGTH)
    turtle.dot(DOT_SIZE, "blue")
    turtle.back(ARM_LENGTH)
    turtle.right(120)
    turtle.update()


def animate() -> None:
    """Apply friction, redraw, and schedule the next frame."""
    if state["turn"] > 0:
        state["turn"] -= 1
    spinner()
    turtle.ontimer(animate, FRAME_DELAY_MS)


def flick() -> None:
    """Add bounded momentum when the player presses the spacebar."""
    state["turn"] = min(state["turn"] + FLICK_MOMENTUM, MAX_MOMENTUM)


def main() -> None:
    """Configure the turtle window and start the animation loop."""
    screen = turtle.Screen()
    screen.setup(420, 420, 370, 0)
    screen.title("Fidget Spinner")
    turtle.hideturtle()
    turtle.tracer(False)
    turtle.width(20)
    screen.onkey(flick, "space")
    screen.listen()
    animate()
    screen.mainloop()


if __name__ == "__main__":
    main()
