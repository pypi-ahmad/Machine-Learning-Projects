"""Lorem Ipsum Generator — CLI tool.

Generates classic lorem ipsum placeholder text.
Supports generating by paragraphs, sentences, or word count.
Also supports custom word sets as an alternative.

Usage:
    python main.py
    python main.py --words 100
    python main.py --paragraphs 3
"""

import random
import sys
import textwrap

# ---------------------------------------------------------------------------
# Lorem ipsum corpus
# ---------------------------------------------------------------------------

LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute "
    "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia "
    "deserunt mollit anim id est laborum sed ut perspiciatis unde omnis iste natus "
    "error sit voluptatem accusantium doloremque laudantium totam rem aperiam eaque "
    "ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt "
    "explicabo nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut "
    "fugit sed quia consequuntur magni dolores eos qui ratione voluptatem sequi "
    "nesciunt neque porro quisquam est qui dolorem ipsum quia dolor sit amet "
    "consectetur adipisci velit sed quia non numquam eius modi tempora incidunt"
).split()

CLASSIC_OPENING = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def _make_sentence(word_pool: list[str], min_words: int = 5, max_words: int = 15) -> str:
    count = random.randint(min_words, max_words)
    words = random.choices(word_pool, k=count)
    sentence = " ".join(words)
    return sentence[0].upper() + sentence[1:] + "."


def _make_paragraph(word_pool: list[str], sentences: int | None = None) -> str:
    if sentences is None:
        sentences = random.randint(3, 7)
    return " ".join(_make_sentence(word_pool) for _ in range(sentences))


def generate_words(n: int, word_pool: list[str] | None = None, start_classic: bool = True) -> str:
    pool = word_pool or LOREM_WORDS
    words = random.choices(pool, k=max(0, n - 8))
    if start_classic:
        base = LOREM_WORDS[:8]
        all_words = base + words
    else:
        all_words = words[:n]
    return " ".join(all_words[:n])


def generate_sentences(n: int, word_pool: list[str] | None = None, start_classic: bool = True) -> str:
    pool = word_pool or LOREM_WORDS
    sentences = [_make_sentence(pool) for _ in range(n)]
    if start_classic:
        sentences[0] = CLASSIC_OPENING
    return " ".join(sentences)


def generate_paragraphs(n: int, word_pool: list[str] | None = None, start_classic: bool = True) -> str:
    pool = word_pool or LOREM_WORDS
    paragraphs = [_make_paragraph(pool) for _ in range(n)]
    if start_classic:
        paragraphs[0] = CLASSIC_OPENING + " " + _make_paragraph(pool, 4)
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------

def parse_args() -> dict | None:
    args = sys.argv[1:]
    if not args:
        return None
    result = {}
    i = 0
    while i < len(args):
        if args[i] in ("--words", "-w") and i + 1 < len(args):
            result["words"] = int(args[i + 1])
            i += 2
        elif args[i] in ("--paragraphs", "-p") and i + 1 < len(args):
            result["paragraphs"] = int(args[i + 1])
            i += 2
        elif args[i] in ("--sentences", "-s") and i + 1 < len(args):
            result["sentences"] = int(args[i + 1])
            i += 2
        else:
            i += 1
    return result if result else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Lorem Ipsum Generator
---------------------
1. Generate paragraphs
2. Generate sentences
3. Generate word count
4. Generate with custom word set
0. Quit
"""


def get_int(prompt: str, default: int) -> int:
    val = input(prompt).strip()
    return int(val) if val.isdigit() else default


def main() -> None:
    # CLI argument mode
    cli_args = parse_args()
    if cli_args:
        if "words" in cli_args:
            print(generate_words(cli_args["words"]))
        elif "sentences" in cli_args:
            print(generate_sentences(cli_args["sentences"]))
        elif "paragraphs" in cli_args:
            print(generate_paragraphs(cli_args["paragraphs"]))
        return

    random.seed()
    print("Lorem Ipsum Generator")

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            n = get_int("  Number of paragraphs (default 3): ", 3)
            text = generate_paragraphs(n)
            print(f"\n{textwrap.fill(text, width=72)}\n")

        elif choice == "2":
            n = get_int("  Number of sentences (default 5): ", 5)
            text = generate_sentences(n)
            print(f"\n{textwrap.fill(text, width=72)}\n")

        elif choice == "3":
            n = get_int("  Number of words (default 50): ", 50)
            text = generate_words(n)
            print(f"\n{textwrap.fill(text, width=72)}\n")

        elif choice == "4":
            print("  Enter your custom words (space separated):")
            raw = input("  > ").strip()
            custom_words = raw.split() if raw else []
            if len(custom_words) < 5:
                print("  Need at least 5 custom words.")
                continue
            n = get_int("  Number of paragraphs (default 2): ", 2)
            text = generate_paragraphs(n, word_pool=custom_words, start_classic=False)
            print(f"\n{textwrap.fill(text, width=72)}\n")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
