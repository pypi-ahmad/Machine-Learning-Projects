"""Keyword Extraction Tool — CLI tool.

Extract important keywords and phrases from articles or text using
TF-IDF scoring combined with a simple RAKE-inspired phrase extraction.
No heavy NLP dependencies required — uses only the standard library
plus basic text statistics.

Usage:
    python main.py
    python main.py article.txt
    python main.py article.txt --top 10
"""

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = set(
    "a an the and or but in on at to for of with by from is are was were be "
    "been being have has had do does did will would could should may might "
    "must can shall that this these those it its i me my we our you your he "
    "she him her his they them their what which who when where how not if as "
    "so than too very just also about after before between into through "
    "during same each other more some such no nor only over own under again "
    "further then once all any both few most out there down up off while "
    "although because since though whether however therefore moreover "
    "furthermore consequently accordingly".split()
)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Return lowercased alpha tokens of 2+ chars."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


def sentence_split(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def extract_candidate_phrases(text: str) -> list[list[str]]:
    """Split text into candidate phrases by stopwords and punctuation (RAKE-style)."""
    # Split on stopwords and punctuation to get candidate phrases
    pattern = r"[.!?,;:\-\(\)\[\]{}\"\'\n\t]"
    chunks = re.split(pattern, text.lower())
    phrases = []
    for chunk in chunks:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", chunk)
        # Remove leading/trailing stopwords
        while words and words[0] in STOPWORDS:
            words.pop(0)
        while words and words[-1] in STOPWORDS:
            words.pop()
        if 1 <= len(words) <= 4:
            phrases.append(words)
    return phrases


# ---------------------------------------------------------------------------
# TF-IDF keyword extraction (single document)
# ---------------------------------------------------------------------------

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total tokens."""
    counter = Counter(tokens)
    total = len(tokens)
    return {w: c / total for w, c in counter.items()}


def compute_tfidf_single(text: str) -> list[tuple[str, float]]:
    """Approximate TF-IDF for a single document using sentence-level IDF."""
    sentences = sentence_split(text)
    if not sentences:
        return []

    # Treat each sentence as a mini-document for IDF
    n_docs = len(sentences)
    all_tokens = tokenize(text)
    content_tokens = [w for w in all_tokens if w not in STOPWORDS]
    tf = compute_tf(content_tokens)

    # Document frequency: in how many sentences does each word appear?
    df = defaultdict(int)
    for sent in sentences:
        seen = set(tokenize(sent))
        for w in seen:
            if w not in STOPWORDS:
                df[w] += 1

    # TF-IDF scores
    scores = {}
    for word, tf_val in tf.items():
        idf = math.log((n_docs + 1) / (df.get(word, 0) + 1)) + 1
        scores[word] = tf_val * idf

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked


# ---------------------------------------------------------------------------
# RAKE-style phrase scoring
# ---------------------------------------------------------------------------

def rake_extract(text: str) -> list[tuple[str, float]]:
    """Extract keyword phrases using a RAKE-inspired algorithm."""
    phrases = extract_candidate_phrases(text)
    if not phrases:
        return []

    # Word scores: degree(word) / frequency(word)
    word_freq = Counter()
    word_degree = Counter()
    for phrase in phrases:
        degree = len(phrase) - 1
        for word in phrase:
            if word not in STOPWORDS:
                word_freq[word] += 1
                word_degree[word] += degree

    word_score = {}
    for word in word_freq:
        word_score[word] = (word_degree[word] + word_freq[word]) / word_freq[word]

    # Phrase scores: sum of word scores
    phrase_scores = {}
    for phrase in phrases:
        key = " ".join(phrase)
        if key not in phrase_scores:
            score = sum(word_score.get(w, 0) for w in phrase if w not in STOPWORDS)
            phrase_scores[key] = score

    ranked = sorted(phrase_scores.items(), key=lambda x: -x[1])
    return ranked


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

def extract_keywords(text: str, top_n: int = 15) -> dict:
    """Extract keywords using both TF-IDF and RAKE methods."""
    tfidf_keywords = compute_tfidf_single(text)
    rake_keywords = rake_extract(text)

    # Simple frequency-based keywords for comparison
    tokens = [w for w in tokenize(text) if w not in STOPWORDS]
    freq_keywords = Counter(tokens).most_common(top_n)

    return {
        "tfidf": tfidf_keywords[:top_n],
        "rake": rake_keywords[:top_n],
        "frequency": freq_keywords,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def bar(value: float, max_val: float, width: int = 20) -> str:
    """Simple text bar chart."""
    if max_val == 0:
        return ""
    filled = int((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)


def display_results(results: dict, top_n: int = 15) -> None:
    """Pretty-print extraction results."""
    print(f"\n{'═' * 60}")
    print("  KEYWORD EXTRACTION RESULTS")
    print(f"{'═' * 60}")

    # TF-IDF Keywords
    print(f"\n  ── TF-IDF Keywords (single words) ──")
    if results["tfidf"]:
        max_score = results["tfidf"][0][1]
        for word, score in results["tfidf"][:top_n]:
            print(f"    {word:<25s} {score:>6.4f}  {bar(score, max_score)}")

    # RAKE Phrases
    print(f"\n  ── RAKE Phrases (multi-word) ──")
    if results["rake"]:
        max_score = results["rake"][0][1]
        for phrase, score in results["rake"][:top_n]:
            print(f"    {phrase:<35s} {score:>6.2f}  {bar(score, max_score)}")

    # Frequency
    print(f"\n  ── Most Frequent Words ──")
    if results["frequency"]:
        max_count = results["frequency"][0][1]
        for word, count in results["frequency"][:top_n]:
            print(f"    {word:<25s} {count:>4d}  {bar(count, max_count)}")

    print()


# ---------------------------------------------------------------------------
# Text stats
# ---------------------------------------------------------------------------

def text_stats(text: str) -> dict:
    """Basic text statistics."""
    words = text.split()
    sentences = sentence_split(text)
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "unique_words": len(set(tokenize(text))),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_TEXT = """
Artificial intelligence (AI) is transforming industries across the globe.
Machine learning, a subset of AI, enables computers to learn from data
without being explicitly programmed. Deep learning, which uses neural
networks with many layers, has achieved remarkable results in image
recognition, natural language processing, and speech recognition.

Companies are investing heavily in AI research and development. Google,
Microsoft, and OpenAI are leading the charge in developing large language
models. These models can generate human-like text, translate languages,
and even write computer code.

However, AI also raises important ethical concerns. Issues like bias in
training data, job displacement, and privacy violations need to be
addressed. Responsible AI development requires transparency, fairness,
and accountability.
"""


def main():
    parser = argparse.ArgumentParser(description="Keyword Extraction Tool")
    parser.add_argument("file", nargs="?", help="Text file to extract keywords from")
    parser.add_argument("--top", "-n", type=int, default=15, help="Number of keywords")
    args = parser.parse_args()

    # File argument mode
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")
        s = text_stats(text)
        print(f"\nFile: {path.name} | {s['words']} words, {s['sentences']} sentences")
        results = extract_keywords(text, args.top)
        display_results(results, args.top)
        return

    # Interactive mode
    print("Keyword Extraction Tool")
    print("═" * 40)
    print("Options:  [t] type/paste text  [f] from file  [d] demo  [q] quit\n")

    while True:
        choice = input("> ").strip().lower()

        if choice in ("q", "quit"):
            print("Bye!")
            break

        elif choice == "d":
            print("  Running demo text...")
            results = extract_keywords(DEMO_TEXT, 15)
            display_results(results, 15)

        elif choice in ("t", ""):
            n = int(input("  Top N keywords [15]: ").strip() or "15")
            print("  Paste/type text. Enter a blank line twice to finish:")
            lines = []
            blank_count = 0
            while True:
                line = input()
                if line == "":
                    blank_count += 1
                    if blank_count >= 2:
                        break
                else:
                    blank_count = 0
                lines.append(line)
            text = "\n".join(lines).strip()
            if not text:
                print("  No text entered.")
                continue
            results = extract_keywords(text, n)
            display_results(results, n)

        elif choice == "f":
            path_str = input("  File path: ").strip()
            path = Path(path_str.strip('"'))
            if not path.exists():
                print(f"  File not found: {path}")
                continue
            n = int(input("  Top N keywords [15]: ").strip() or "15")
            text = path.read_text(encoding="utf-8", errors="replace")
            results = extract_keywords(text, n)
            display_results(results, n)

        else:
            print("  Commands: t=type text  f=file  d=demo  q=quit")


if __name__ == "__main__":
    main()

