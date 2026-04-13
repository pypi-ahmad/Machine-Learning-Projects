"""Text Summarizer — CLI tool.

Extractive text summarizer using sentence scoring.
Ranks sentences by TF-IDF-like word frequency.
No external dependencies required.

Usage:
    python main.py
    python main.py article.txt
    python main.py article.txt --sentences 3
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


STOPWORDS = set("""a an the and or but in on at to for of with by from is are
was were be been being have has had do does did will would could should may
might must can shall that this these those it its i me my we our you your
he she him her his they them their what which who when where how not if as
so than too very just also about after before between into through during
same each other more some such no nor only over own under again further
then once all any both few most other out there down up off while although
because since while though whether both either neither nor not only but also
however therefore moreover furthermore consequently accordingly""".split())


def tokenise(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())


def sentence_split(text: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.split()) > 3]


def summarise(text: str, n_sentences: int = 3) -> tuple[str, dict]:
    sentences = sentence_split(text)
    if len(sentences) <= n_sentences:
        return text, {}

    # Word frequency (excluding stopwords)
    words = [w for w in tokenise(text) if w not in STOPWORDS]
    freq  = Counter(words)
    max_f = freq.most_common(1)[0][1] if freq else 1

    # Score each sentence
    scores: dict[int, float] = {}
    for i, sent in enumerate(sentences):
        sent_words = [w for w in tokenise(sent) if w not in STOPWORDS]
        if not sent_words:
            scores[i] = 0.0
            continue
        score = sum(freq[w] / max_f for w in sent_words) / len(sent_words)
        # Bonus for sentences near the start
        position_bonus = 1.0 / (i + 1) * 0.2
        scores[i] = score + position_bonus

    top_indices = sorted(sorted(scores, key=scores.get, reverse=True)[:n_sentences])
    summary = " ".join(sentences[i] for i in top_indices)
    return summary, scores


def stats(text: str) -> dict:
    words     = tokenise(text)
    sentences = sentence_split(text)
    paras     = [p for p in text.split("\n\n") if p.strip()]
    return {
        "chars":     len(text),
        "words":     len(text.split()),
        "sentences": len(sentences),
        "paragraphs":len(paras),
        "avg_words_per_sent": len(words) / max(len(sentences), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Text Summarizer")
    parser.add_argument("file",       nargs="?", help="Text file to summarise")
    parser.add_argument("--sentences","-n", type=int, default=3)
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        text = path.read_text(errors="replace")
        summary, scores = summarise(text, args.sentences)
        print(f"\n{'─'*60}")
        print(f"SUMMARY ({args.sentences} sentences):")
        print('─'*60)
        print(summary)
        s = stats(text)
        print(f"\nOriginal: {s['words']} words, {s['sentences']} sentences")
        print(f"Summary:  {len(summary.split())} words")
        compression = (1 - len(summary.split()) / max(s['words'], 1)) * 100
        print(f"Compression: {compression:.0f}%")
        return

    print("Text Summarizer")
    print("──────────────────────────────")
    print("Options:  [t] type/paste text  [f] from file  [q] quit\n")

    while True:
        choice = input("> ").strip().lower()

        if choice in ("q", "quit"):
            print("Bye!")
            break

        elif choice in ("t", ""):
            n = int(input("  Sentences in summary [3]: ").strip() or "3")
            print("  Paste/type text. Enter a blank line when done:")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if not text:
                print("  No text entered.")
                continue
            summary, _ = summarise(text, n)
            s = stats(text)
            print(f"\n  {'─'*50}")
            print(f"  SUMMARY ({n} sentences):")
            print(f"  {'─'*50}")
            for sent in sentence_split(summary):
                print(f"  {sent}")
            print(f"\n  Original: {s['words']} words | Summary: {len(summary.split())} words")

        elif choice == "f":
            path_str = input("  File path: ").strip()
            path = Path(path_str)
            if not path.exists():
                print(f"  File not found: {path}")
                continue
            n = int(input("  Sentences in summary [3]: ").strip() or "3")
            text = path.read_text(errors="replace")
            summary, _ = summarise(text, n)
            print(f"\n  SUMMARY:\n  {summary}")

        else:
            print("  Commands: t=type text  f=file  q=quit")


if __name__ == "__main__":
    main()
