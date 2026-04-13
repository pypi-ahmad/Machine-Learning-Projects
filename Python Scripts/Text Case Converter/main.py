"""Text Case Converter — CLI tool.

Converts text between various case styles:
  upper, lower, title, sentence, camelCase, PascalCase,
  snake_case, kebab-case, SCREAMING_SNAKE_CASE, dot.case, alternating.

Usage:
    python main.py
"""

import re


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def to_upper(text: str) -> str:
    return text.upper()


def to_lower(text: str) -> str:
    return text.lower()


def to_title(text: str) -> str:
    return text.title()


def to_sentence(text: str) -> str:
    """Capitalize only the first letter of each sentence."""
    sentences = re.split(r"([.!?]\s*)", text)
    result = ""
    for i, part in enumerate(sentences):
        if i % 2 == 0 and part:
            result += part[0].upper() + part[1:].lower()
        else:
            result += part
    return result


def _split_words(text: str) -> list[str]:
    """Split text by spaces, underscores, hyphens, and camelCase boundaries."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)   # camelCase split
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"[-_.\s]+", " ", text)
    return [w for w in text.split() if w]


def to_camel(text: str) -> str:
    words = _split_words(text)
    if not words:
        return text
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def to_pascal(text: str) -> str:
    return "".join(w.capitalize() for w in _split_words(text))


def to_snake(text: str) -> str:
    return "_".join(w.lower() for w in _split_words(text))


def to_kebab(text: str) -> str:
    return "-".join(w.lower() for w in _split_words(text))


def to_screaming_snake(text: str) -> str:
    return "_".join(w.upper() for w in _split_words(text))


def to_dot(text: str) -> str:
    return ".".join(w.lower() for w in _split_words(text))


def to_alternating(text: str) -> str:
    result = []
    upper = True
    for ch in text:
        if ch.isalpha():
            result.append(ch.upper() if upper else ch.lower())
            upper = not upper
        else:
            result.append(ch)
    return "".join(result)


def to_reverse(text: str) -> str:
    return text[::-1]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONVERTERS: dict[str, tuple[callable, str]] = {
    "1":  (to_upper,          "UPPER CASE"),
    "2":  (to_lower,          "lower case"),
    "3":  (to_title,          "Title Case"),
    "4":  (to_sentence,       "Sentence case"),
    "5":  (to_camel,          "camelCase"),
    "6":  (to_pascal,         "PascalCase"),
    "7":  (to_snake,          "snake_case"),
    "8":  (to_kebab,          "kebab-case"),
    "9":  (to_screaming_snake, "SCREAMING_SNAKE_CASE"),
    "10": (to_dot,            "dot.case"),
    "11": (to_alternating,    "aLtErNaTiNg CaSe"),
    "12": (to_reverse,        "esreveR"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_menu() -> None:
    print("\nConvert to:")
    for key, (_, label) in CONVERTERS.items():
        print(f"  {key:>3}. {label}")
    print("   A. All formats")
    print("   0. Quit")


def main() -> None:
    print("Text Case Converter")
    print("=" * 40)

    while True:
        print_menu()
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        if choice not in CONVERTERS and choice.upper() != "A":
            print("  Invalid choice.")
            continue

        text = input("  Enter text: ")
        if not text:
            print("  No text entered.")
            continue

        if choice.upper() == "A":
            print()
            for _, (fn, label) in CONVERTERS.items():
                print(f"  {label:<28}: {fn(text)}")
        else:
            fn, label = CONVERTERS[choice]
            result = fn(text)
            print(f"\n  {label}: {result}")


if __name__ == "__main__":
    main()
