"""Morse Code Translator — CLI tool.

Encode text to Morse code and decode Morse code back to text.
Supports standard ITU Morse, visual/audio display, and batch conversion.

Usage:
    python main.py
"""

import time


# ---------------------------------------------------------------------------
# Morse code tables
# ---------------------------------------------------------------------------

MORSE: dict[str, str] = {
    "A": ".-",   "B": "-...", "C": "-.-.", "D": "-..",  "E": ".",
    "F": "..-.", "G": "--.",  "H": "....", "I": "..",   "J": ".---",
    "K": "-.-",  "L": ".-..", "M": "--",   "N": "-.",   "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.",  "S": "...",  "T": "-",
    "U": "..-",  "V": "...-", "W": ".--",  "X": "-..-", "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.",
    "!": "-.-.--", "/": "-..-.",  "(": "-.--.",  ")": "-.--.-",
    "&": ".-...",  ":": "---...", ";": "-.-.-.",  "=": "-...-",
    "+": ".-.-.",  "-": "-....-", "_": "..--.-",  '"': ".-..-.",
    "$": "...-..-","@": ".--.-.", " ": "/",
}

REVERSE: dict[str, str] = {v: k for k, v in MORSE.items()}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def encode(text: str) -> str:
    """Encode plain text to Morse code."""
    result = []
    for ch in text.upper():
        if ch in MORSE:
            result.append(MORSE[ch])
        elif ch == " ":
            result.append("/")
        else:
            result.append("?")  # unknown character
    return " ".join(result)


def decode(morse: str) -> str:
    """Decode Morse code to plain text.

    Words are separated by ' / ' or '  ' (double space).
    Letters are separated by single spaces.
    """
    # Normalize: replace ' / ' with a special marker
    morse = morse.strip().replace("  ", " / ")
    words = morse.split(" / ")
    result = []
    for word in words:
        letters = word.strip().split()
        decoded_word = ""
        for code in letters:
            decoded_word += REVERSE.get(code, "?")
        result.append(decoded_word)
    return " ".join(result)


def visual_display(morse: str, dot_char: str = "·", dash_char: str = "—") -> str:
    """Replace . and - with visual characters."""
    return morse.replace(".", dot_char).replace("-", dash_char)


def morse_to_sound_pattern(morse: str) -> list[tuple[str, float]]:
    """Return (symbol, duration_ms) list for audio playback simulation."""
    DIT  = 60   # ms
    DAH  = 180
    ILS  = 60   # inter-letter space
    IWS  = 180  # inter-word space
    ICS  = 20   # intra-character space (between dit/dah)

    pattern = []
    for token in morse.split():
        if token == "/":
            pattern.append(("space", IWS))
        else:
            for i, symbol in enumerate(token):
                if symbol == ".":
                    pattern.append(("dit",  DIT))
                elif symbol == "-":
                    pattern.append(("dah",  DAH))
                if i < len(token) - 1:
                    pattern.append(("pause", ICS))
            pattern.append(("pause", ILS))
    return pattern


def play_visual(morse: str) -> None:
    """Print a real-time visual representation of the morse code."""
    print("  Playing: ", end="", flush=True)
    for token in morse.split():
        if token == "/":
            print("   ", end="", flush=True)
            time.sleep(0.18)
        else:
            for ch in token:
                if ch == ".":
                    print("●", end="", flush=True)
                    time.sleep(0.06)
                elif ch == "-":
                    print("━━", end="", flush=True)
                    time.sleep(0.18)
                time.sleep(0.02)
            print(" ", end="", flush=True)
            time.sleep(0.06)
    print()


def wpm_timing(morse: str, wpm: int = 5) -> float:
    """Estimate time to transmit in seconds at given WPM (PARIS standard)."""
    # 1 word = 50 units; PARIS has 50 units
    unit_ms = 1200 / wpm
    total   = 0
    for ch in morse:
        if ch == ".":
            total += 1
        elif ch == "-":
            total += 3
        elif ch == " ":
            total += 1  # approximate
    return total * unit_ms / 1000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Morse Code Translator
---------------------
1. Encode text → Morse
2. Decode Morse → text
3. Encode with visual display
4. Batch encode (multiple lines)
5. Encode and play visually
0. Quit
"""


def main() -> None:
    print("Morse Code Translator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            text  = input("  Text: ")
            morse = encode(text)
            print(f"\n  Morse: {morse}")
            secs  = wpm_timing(morse)
            print(f"  (est. {secs:.1f}s at 5 WPM)")

        elif choice == "2":
            morse = input("  Morse (letters sep by space, words by /): ")
            text  = decode(morse)
            print(f"\n  Text: {text}")

        elif choice == "3":
            text  = input("  Text: ")
            morse = encode(text)
            vis   = visual_display(morse)
            print(f"\n  Morse   : {morse}")
            print(f"  Visual  : {vis}")

        elif choice == "4":
            print("  Enter lines (blank to stop):")
            while True:
                line = input("  > ")
                if not line:
                    break
                morse = encode(line)
                print(f"    {morse}")

        elif choice == "5":
            text  = input("  Text: ")
            morse = encode(text)
            print(f"\n  Morse: {morse}")
            play_visual(morse)

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
