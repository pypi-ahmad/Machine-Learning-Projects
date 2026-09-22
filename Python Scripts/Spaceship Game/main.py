"""Run a local two-player spaceship battle game."""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame


WIDTH, HEIGHT = 900, 500
SPACESHIP_SIZE = (50, 40)
FPS = 60
VELOCITY = 5
BULLET_VELOCITY = 7
MAX_BULLETS = 3
STARTING_HEALTH = 10
ASSETS = Path(__file__).with_name("Assets")


def load_assets() -> tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Rect]:
    """Load and orient game sprites using paths relative to this file."""
    yellow = pygame.image.load(ASSETS / "Yellow_Spaceship.png")
    red = pygame.image.load(ASSETS / "Red_Spaceship.png")
    space = pygame.image.load(ASSETS / "space.jpg")
    yellow = pygame.transform.rotate(pygame.transform.scale(yellow, SPACESHIP_SIZE), 90)
    red = pygame.transform.rotate(pygame.transform.scale(red, SPACESHIP_SIZE), -90)
    space = pygame.transform.scale(space, (WIDTH, HEIGHT))
    return yellow, red, space, pygame.Rect(WIDTH // 2 - 5, 0, 10, HEIGHT)


def move_yellow(keys: pygame.key.ScancodeWrapper, ship: pygame.Rect, border: pygame.Rect) -> None:
    """Move yellow ship while keeping it on the left side of the border."""
    if keys[pygame.K_a] and ship.left - VELOCITY > 0:
        ship.x -= VELOCITY
    if keys[pygame.K_d] and ship.right + VELOCITY < border.left:
        ship.x += VELOCITY
    if keys[pygame.K_w] and ship.top - VELOCITY > 0:
        ship.y -= VELOCITY
    if keys[pygame.K_s] and ship.bottom + VELOCITY < HEIGHT:
        ship.y += VELOCITY


def move_red(keys: pygame.key.ScancodeWrapper, ship: pygame.Rect, border: pygame.Rect) -> None:
    """Move red ship while keeping it on the right side of the border."""
    if keys[pygame.K_LEFT] and ship.left - VELOCITY > border.right:
        ship.x -= VELOCITY
    if keys[pygame.K_RIGHT] and ship.right + VELOCITY < WIDTH:
        ship.x += VELOCITY
    if keys[pygame.K_UP] and ship.top - VELOCITY > 0:
        ship.y -= VELOCITY
    if keys[pygame.K_DOWN] and ship.bottom + VELOCITY < HEIGHT:
        ship.y += VELOCITY


def move_bullets(
    yellow_bullets: list[pygame.Rect], red_bullets: list[pygame.Rect], yellow: pygame.Rect, red: pygame.Rect
) -> tuple[int, int]:
    """Advance bullets and return hits against red and yellow ships."""
    red_hits = 0
    yellow_hits = 0
    for bullet in yellow_bullets[:]:
        bullet.x += BULLET_VELOCITY
        if red.colliderect(bullet):
            yellow_bullets.remove(bullet)
            red_hits += 1
        elif bullet.left > WIDTH:
            yellow_bullets.remove(bullet)
    for bullet in red_bullets[:]:
        bullet.x -= BULLET_VELOCITY
        if yellow.colliderect(bullet):
            red_bullets.remove(bullet)
            yellow_hits += 1
        elif bullet.right < 0:
            red_bullets.remove(bullet)
    return red_hits, yellow_hits


def draw_game(
    window: pygame.Surface,
    assets: tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Rect],
    yellow: pygame.Rect,
    red: pygame.Rect,
    yellow_bullets: list[pygame.Rect],
    red_bullets: list[pygame.Rect],
    yellow_health: int,
    red_health: int,
    health_font: pygame.font.Font,
) -> None:
    """Render one game frame."""
    yellow_sprite, red_sprite, space, border = assets
    window.blit(space, (0, 0))
    pygame.draw.rect(window, "white", border)
    window.blit(yellow_sprite, yellow.topleft)
    window.blit(red_sprite, red.topleft)
    for bullet in yellow_bullets:
        pygame.draw.rect(window, "yellow", bullet)
    for bullet in red_bullets:
        pygame.draw.rect(window, "red", bullet)
    window.blit(health_font.render(f"Health: {yellow_health}", True, "white"), (10, 10))
    red_label = health_font.render(f"Health: {red_health}", True, "white")
    window.blit(red_label, (WIDTH - red_label.get_width() - 10, 10))
    pygame.display.flip()


def show_winner(window: pygame.Surface, text: str) -> None:
    """Display a winner for five seconds before the next round."""
    font = pygame.font.SysFont("comicsans", 80)
    label = font.render(text, True, "white")
    window.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - label.get_height() // 2))
    pygame.display.flip()
    pygame.time.delay(5000)


def play_round(window: pygame.Surface, clock: pygame.time.Clock, assets: tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Rect]) -> bool:
    """Play one round and return whether another round should start."""
    border = assets[3]
    yellow = pygame.Rect(100, HEIGHT // 2, *SPACESHIP_SIZE)
    red = pygame.Rect(700, HEIGHT // 2, *SPACESHIP_SIZE)
    yellow_bullets: list[pygame.Rect] = []
    red_bullets: list[pygame.Rect] = []
    yellow_health = STARTING_HEALTH
    red_health = STARTING_HEALTH
    health_font = pygame.font.SysFont("comicsans", 40)
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LCTRL and len(yellow_bullets) < MAX_BULLETS:
                yellow_bullets.append(pygame.Rect(yellow.right, yellow.centery - 2, 10, 5))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RCTRL and len(red_bullets) < MAX_BULLETS:
                red_bullets.append(pygame.Rect(red.left - 10, red.centery - 2, 10, 5))
        move_yellow(pygame.key.get_pressed(), yellow, border)
        move_red(pygame.key.get_pressed(), red, border)
        red_hits, yellow_hits = move_bullets(yellow_bullets, red_bullets, yellow, red)
        red_health -= red_hits
        yellow_health -= yellow_hits
        draw_game(window, assets, yellow, red, yellow_bullets, red_bullets, yellow_health, red_health, health_font)
        if red_health <= 0 or yellow_health <= 0:
            show_winner(window, "Yellow wins!" if red_health <= 0 else "Red wins!")
            running = False
    return True


def main() -> None:
    """Initialize Pygame and run rounds until the window is closed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="load assets, then exit")
    args = parser.parse_args()
    pygame.init()
    try:
        window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Spaceship War Game")
        assets = load_assets()
        if args.smoke:
            print("Assets loaded successfully.")
            return
        clock = pygame.time.Clock()
        while play_round(window, clock, assets):
            pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
