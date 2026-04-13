"""Trivia Game — CLI game.

Multiple-choice trivia with categories, score tracking,
timed questions, and difficulty levels.

Usage:
    python main.py
    python main.py --category science --difficulty hard
"""

import argparse
import random
import time


QUESTIONS = {
    "science": {
        "easy": [
            {"q": "What is the chemical symbol for water?",
             "a": ["H₂O","CO₂","NaCl","O₂"], "correct": 0},
            {"q": "How many planets are in the Solar System?",
             "a": ["7","8","9","10"], "correct": 1},
            {"q": "What gas do plants absorb from the atmosphere?",
             "a": ["Oxygen","Nitrogen","Carbon Dioxide","Hydrogen"], "correct": 2},
            {"q": "What is the speed of light (approx)?",
             "a": ["300,000 km/s","150,000 km/s","3,000 km/s","30,000 km/s"], "correct": 0},
        ],
        "medium": [
            {"q": "What is the atomic number of carbon?",
             "a": ["6","8","12","14"], "correct": 0},
            {"q": "Which particle has a negative charge?",
             "a": ["Proton","Neutron","Electron","Quark"], "correct": 2},
            {"q": "What is the powerhouse of the cell?",
             "a": ["Nucleus","Ribosome","Mitochondria","Golgi"], "correct": 2},
            {"q": "What is Newton's 2nd law?",
             "a": ["F=ma","E=mc²","PV=nRT","v=u+at"], "correct": 0},
        ],
        "hard": [
            {"q": "What is the Heisenberg Uncertainty Principle about?",
             "a": ["Momentum and position","Energy and mass","Time and space","Charge and spin"],
             "correct": 0},
            {"q": "Which element has the highest electronegativity?",
             "a": ["Oxygen","Nitrogen","Fluorine","Chlorine"], "correct": 2},
            {"q": "What is the half-life of Carbon-14?",
             "a": ["1,000 yr","5,730 yr","10,000 yr","50,000 yr"], "correct": 1},
        ],
    },
    "history": {
        "easy": [
            {"q": "Who was the first US president?",
             "a": ["Lincoln","Jefferson","Washington","Adams"], "correct": 2},
            {"q": "In which year did World War II end?",
             "a": ["1943","1944","1945","1946"], "correct": 2},
            {"q": "Which ancient wonder was in Egypt?",
             "a": ["Colossus of Rhodes","Great Pyramid","Hanging Gardens","Temple of Artemis"],
             "correct": 1},
        ],
        "medium": [
            {"q": "Who wrote the Magna Carta (forced its creation)?",
             "a": ["King John","Henry VIII","Richard I","Edward I"], "correct": 0},
            {"q": "In which year did the Berlin Wall fall?",
             "a": ["1987","1988","1989","1990"], "correct": 2},
            {"q": "Which empire was ruled by Genghis Khan?",
             "a": ["Ottoman","Mongol","Roman","Byzantine"], "correct": 1},
        ],
        "hard": [
            {"q": "The Battle of Hastings was in which year?",
             "a": ["1066","1099","1215","1415"], "correct": 0},
            {"q": "Who was the last Tsar of Russia?",
             "a": ["Alexander III","Nicholas I","Nicholas II","Alexander II"], "correct": 2},
        ],
    },
    "technology": {
        "easy": [
            {"q": "What does CPU stand for?",
             "a": ["Central Processing Unit","Core Power Unit","Central Program Utility","Computer Processing Unit"],
             "correct": 0},
            {"q": "What does HTML stand for?",
             "a": ["HyperText Markup Language","High-Tech Machine Language","Home Tool Markup Language","Hyperlink Text Method Language"],
             "correct": 0},
            {"q": "Who co-founded Apple with Steve Jobs?",
             "a": ["Bill Gates","Steve Wozniak","Larry Page","Linus Torvalds"], "correct": 1},
        ],
        "medium": [
            {"q": "What is the binary representation of 10?",
             "a": ["0b1000","0b1010","0b1100","0b0110"], "correct": 1},
            {"q": "What does DNS stand for?",
             "a": ["Domain Name System","Digital Network Service","Data Node Server","Dynamic Network Solution"],
             "correct": 0},
            {"q": "Which sorting algorithm has O(n log n) average?",
             "a": ["Bubble Sort","Selection Sort","Quicksort","Insertion Sort"], "correct": 2},
        ],
        "hard": [
            {"q": "What is the time complexity of binary search?",
             "a": ["O(n)","O(log n)","O(n²)","O(1)"], "correct": 1},
            {"q": "Which protocol operates at OSI Layer 4?",
             "a": ["IP","TCP","HTTP","ARP"], "correct": 1},
        ],
    },
}

SCORE_TABLE = {"easy": 10, "medium": 20, "hard": 30}
TIME_LIMIT  = {"easy": 30, "medium": 20, "hard": 15}


def ask_question(q: dict, num: int, total: int, difficulty: str, timed: bool) -> bool:
    print(f"\n  Q{num}/{total}: {q['q']}")
    options = q["a"]
    for i, opt in enumerate(options):
        print(f"    {i+1}. {opt}")

    limit = TIME_LIMIT[difficulty] if timed else None
    if limit:
        print(f"  ⏱  {limit}s")

    start = time.time()
    while True:
        elapsed = time.time() - start
        if limit and elapsed >= limit:
            print(f"  ⏰ Time's up! Answer was: {options[q['correct']]}")
            return False
        try:
            inp = input("  Your answer (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if inp.isdigit() and 1 <= int(inp) <= 4:
            idx = int(inp) - 1
            if idx == q["correct"]:
                elapsed = time.time() - start
                print(f"  ✅ Correct! ({elapsed:.1f}s)")
                return True
            else:
                print(f"  ❌ Wrong! Correct: {options[q['correct']]}")
                return False
        print("  Enter 1, 2, 3, or 4.")


def play(category: str, difficulty: str, n_questions: int, timed: bool) -> None:
    pool = QUESTIONS.get(category, {}).get(difficulty, [])
    if not pool:
        print(f"  No questions for {category}/{difficulty}.")
        return

    questions  = random.sample(pool, min(n_questions, len(pool)))
    score      = 0
    correct    = 0
    pts        = SCORE_TABLE[difficulty]

    print(f"\n=== Trivia Game ===  [{category.title()} / {difficulty.upper()}]")
    print(f"  Questions: {len(questions)}  |  Points per correct: {pts}"
          + (f"  |  Timed" if timed else "") + "\n")

    for i, q in enumerate(questions, 1):
        if ask_question(q, i, len(questions), difficulty, timed):
            score   += pts
            correct += 1

    pct = correct / len(questions) * 100
    print(f"\n  ── Results ──")
    print(f"  Correct:     {correct}/{len(questions)}")
    print(f"  Score:       {score}")
    print(f"  Accuracy:    {pct:.0f}%")
    if pct == 100: print("  🏆 Perfect!")
    elif pct >= 70: print("  👍 Well done!")
    else: print("  📚 Keep studying!")


def main():
    cats   = list(QUESTIONS.keys())
    diffs  = ["easy","medium","hard"]
    parser = argparse.ArgumentParser(description="Trivia Game")
    parser.add_argument("--category",   choices=cats,  default=None)
    parser.add_argument("--difficulty", choices=diffs, default="medium")
    parser.add_argument("--questions",  type=int, default=5)
    parser.add_argument("--timed",      action="store_true")
    args   = parser.parse_args()

    if not args.category:
        print(f"Available categories: {', '.join(cats)}")
        cat = input("Choose category: ").strip().lower()
        if cat not in cats:
            cat = random.choice(cats)
            print(f"  Using: {cat}")
        args.category = cat

    play(args.category, args.difficulty, args.questions, args.timed)


if __name__ == "__main__":
    main()
