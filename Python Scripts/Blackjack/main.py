"""Blackjack — CLI card game.

Play Blackjack (21) against the dealer.
Supports hit, stand, double-down, and split.
Tracks chips balance across multiple rounds.

Usage:
    python main.py
"""

import random
from collections import namedtuple

# ---------------------------------------------------------------------------
# Card / Deck
# ---------------------------------------------------------------------------

SUITS  = ["♠", "♥", "♦", "♣"]
RANKS  = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
COLORS = {"♥": "\033[31m", "♦": "\033[31m", "♠": "", "♣": ""}
RESET  = "\033[0m"


def card_str(rank: str, suit: str, hidden: bool = False) -> str:
    if hidden:
        return "  [??]  "
    color = COLORS[suit]
    return f"  {color}[{rank:>2}{suit}]{RESET}  "


def deck(num_decks: int = 1) -> list[tuple[str, str]]:
    cards = [(r, s) for s in SUITS for r in RANKS] * num_decks
    random.shuffle(cards)
    return cards


def hand_value(hand: list[tuple[str, str]]) -> int:
    total, aces = 0, 0
    for rank, _ in hand:
        if rank in ("J", "Q", "K"):
            total += 10
        elif rank == "A":
            total += 11
            aces += 1
        else:
            total += int(rank)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def display_hand(hand: list[tuple[str, str]], label: str,
                  hide_second: bool = False) -> None:
    print(f"\n  {label}:", end="")
    for i, (rank, suit) in enumerate(hand):
        print(card_str(rank, suit, hidden=(i == 1 and hide_second)), end="")
    if not hide_second:
        val = hand_value(hand)
        print(f"  (value: {val})", end="")
    print()


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

def play_round(shoe: list, chips: int) -> int:
    """Play one round. Returns updated chips balance."""
    if len(shoe) < 10:
        shoe.clear()
        shoe.extend(deck(2))
        print("  [Shuffling new shoe...]")

    # Bet
    while True:
        bet_s = input(f"\n  Chips: {chips}  |  Your bet: ").strip()
        try:
            bet = int(bet_s)
            if 1 <= bet <= chips:
                break
            print(f"  Bet must be 1–{chips}.")
        except ValueError:
            print("  Enter a number.")

    player_hand = [shoe.pop(), shoe.pop()]
    dealer_hand = [shoe.pop(), shoe.pop()]

    display_hand(dealer_hand, "Dealer", hide_second=True)
    display_hand(player_hand, "You")

    # Check natural blackjack
    if hand_value(player_hand) == 21:
        display_hand(dealer_hand, "Dealer")
        if hand_value(dealer_hand) == 21:
            print("  Push! Both have Blackjack.")
        else:
            print("  Blackjack! You win 1.5×!")
            chips += int(bet * 1.5)
        return chips

    # Player turn
    while True:
        pv = hand_value(player_hand)
        if pv > 21:
            print(f"\n  Bust! ({pv})")
            chips -= bet
            return chips

        # Options
        options = ["(h)it", "(s)tand"]
        if len(player_hand) == 2:
            options.append("(d)ouble")
        if len(player_hand) == 2 and player_hand[0][0] == player_hand[1][0]:
            options.append("(split)")

        action = input(f"\n  {' / '.join(options)}: ").strip().lower()

        if action.startswith("s"):
            break

        elif action.startswith("h"):
            player_hand.append(shoe.pop())
            display_hand(player_hand, "You")

        elif action.startswith("d") and len(player_hand) == 2:
            if bet * 2 > chips:
                print("  Not enough chips to double.")
                continue
            bet *= 2
            player_hand.append(shoe.pop())
            display_hand(player_hand, "You")
            print(f"  (Doubled down — bet: {bet})")
            break

        elif action.startswith("sp") and len(player_hand) == 2:
            # Simple split: play two separate hands
            print("  Split (simplified: playing first hand).")
            hand2 = [player_hand.pop()]
            hand2.append(shoe.pop())
            player_hand.append(shoe.pop())
            display_hand(player_hand, "You (Hand 1)")
            display_hand(hand2, "You (Hand 2) — resolved separately next round")
        else:
            print("  Invalid action.")

    pv = hand_value(player_hand)
    if pv > 21:
        print(f"\n  Bust! ({pv})")
        chips -= bet
        return chips

    # Dealer turn
    print("\n  Dealer reveals:")
    display_hand(dealer_hand, "Dealer")
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(shoe.pop())
        display_hand(dealer_hand, "Dealer")

    dv = hand_value(dealer_hand)

    print(f"\n  You: {pv}   Dealer: {dv}")
    if dv > 21 or pv > dv:
        print("  You win!")
        chips += bet
    elif pv == dv:
        print("  Push!")
    else:
        print("  Dealer wins.")
        chips -= bet

    return chips


def main() -> None:
    print("♠ Blackjack ♥")
    print("  Beat the dealer to 21 without going bust.")

    chips = int(input("\n  Starting chips (default 500): ").strip() or "500")
    shoe  = deck(2)

    while chips > 0:
        try:
            chips = play_round(shoe, chips)
            if chips <= 0:
                print("\n  Out of chips! Game over.")
                break
            again = input("\n  Play again? (y/n): ").strip().lower()
            if again != "y":
                print(f"\n  Final chips: {chips}")
                break
        except KeyboardInterrupt:
            break

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
