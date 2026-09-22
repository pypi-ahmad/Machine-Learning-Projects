"""Run a Pygame Snake game.

Usage:
    uv run python game.py
"""

import random

import pygame

YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 400
SNAKE_BLOCK = 10
SNAKE_SPEED = 15


def draw_score(display: pygame.Surface, font: pygame.font.Font, score: int) -> None:
    value = font.render(f"Your Score: {score}", True, YELLOW)
    display.blit(value, (0, 0))


def draw_snake(display: pygame.Surface, snake: list[list[float]]) -> None:
    for x, y in snake:
        pygame.draw.rect(display, BLACK, (x, y, SNAKE_BLOCK, SNAKE_BLOCK))


def draw_message(display: pygame.Surface, font: pygame.font.Font, message: str) -> None:
    text = font.render(message, True, RED)
    display.blit(text, (DISPLAY_WIDTH / 6, DISPLAY_HEIGHT / 3))


def game_loop(
    display: pygame.Surface,
    clock: pygame.time.Clock,
    message_font: pygame.font.Font,
    score_font: pygame.font.Font,
) -> bool:
    """Play one game and return True when the player requests a restart."""
    game_over = False
    game_close = False
    x1 = DISPLAY_WIDTH / 2
    y1 = DISPLAY_HEIGHT / 2
    x1_change = 0
    y1_change = 0
    snake = []
    snake_length = 1
    food_x = round(random.randrange(0, DISPLAY_WIDTH - SNAKE_BLOCK) / SNAKE_BLOCK) * SNAKE_BLOCK
    food_y = round(random.randrange(0, DISPLAY_HEIGHT - SNAKE_BLOCK) / SNAKE_BLOCK) * SNAKE_BLOCK

    while not game_over:
        while game_close:
            display.fill(BLUE)
            draw_message(display, message_font, "You Lost! Press C to play again or Q to quit")
            draw_score(display, score_font, snake_length - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change, y1_change = -SNAKE_BLOCK, 0
                elif event.key == pygame.K_RIGHT:
                    x1_change, y1_change = SNAKE_BLOCK, 0
                elif event.key == pygame.K_UP:
                    x1_change, y1_change = 0, -SNAKE_BLOCK
                elif event.key == pygame.K_DOWN:
                    x1_change, y1_change = 0, SNAKE_BLOCK

        if x1 >= DISPLAY_WIDTH or x1 < 0 or y1 >= DISPLAY_HEIGHT or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        display.fill(BLUE)
        pygame.draw.rect(display, GREEN, (food_x, food_y, SNAKE_BLOCK, SNAKE_BLOCK))
        snake_head = [x1, y1]
        snake.append(snake_head)
        if len(snake) > snake_length:
            del snake[0]

        if snake_head in snake[:-1]:
            game_close = True

        draw_snake(display, snake)
        draw_score(display, score_font, snake_length - 1)
        pygame.display.update()

        if x1 == food_x and y1 == food_y:
            food_x = round(random.randrange(0, DISPLAY_WIDTH - SNAKE_BLOCK) / SNAKE_BLOCK) * SNAKE_BLOCK
            food_y = round(random.randrange(0, DISPLAY_HEIGHT - SNAKE_BLOCK) / SNAKE_BLOCK) * SNAKE_BLOCK
            snake_length += 1

        clock.tick(SNAKE_SPEED)

    return False


def main() -> None:
    pygame.init()
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption("Snake Game in Python")
    clock = pygame.time.Clock()
    message_font = pygame.font.SysFont("bahnschrift", 25)
    score_font = pygame.font.SysFont("comicsansms", 35)

    try:
        restart = True
        while restart:
            restart = game_loop(display, clock, message_font, score_font)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
