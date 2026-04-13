"""Random Animal Image — CLI tool.

Fetch random animal images and facts from public APIs.
Supports dogs, cats, foxes, and more.

Usage:
    python main.py
    python main.py --animal dog
    python main.py --animal cat --fact
    python main.py --animal fox --count 3
"""

import argparse
import json
import sys
import urllib.request


APIS = {
    "dog":  {
        "url":   "https://dog.ceo/api/breeds/image/random",
        "parse": lambda d: d["message"],
        "breed_list": "https://dog.ceo/api/breeds/list/all",
    },
    "cat":  {
        "url":   "https://api.thecatapi.com/v1/images/search",
        "parse": lambda d: d[0]["url"],
    },
    "fox":  {
        "url":   "https://randomfox.ca/floof/",
        "parse": lambda d: d["image"],
    },
    "duck": {
        "url":   "https://random-d.uk/api/random",
        "parse": lambda d: d["url"],
    },
}

CAT_FACTS_URL = "https://catfact.ninja/fact"
DOG_FACTS = [
    "Dogs have a sense of smell that is 10,000–100,000 times more acute than humans.",
    "A dog's nose print is unique, much like a human's fingerprint.",
    "Dogs can understand around 165 words and gestures.",
    "The Basenji is the only dog that cannot bark.",
    "Dogs can see some colors — mainly blue and yellow.",
    "A dog's heart beats between 60 and 140 times per minute.",
    "Puppies are born blind, deaf, and toothless.",
    "Dogs sweat through their paws.",
]
FOX_FACTS = [
    "Foxes use the Earth's magnetic field to hunt prey under snow.",
    "A group of foxes is called a 'skulk' or 'earth'.",
    "Foxes have vertically slit pupils like cats.",
    "Foxes communicate with over 40 different sounds.",
    "The fennec fox has the largest ears relative to body size of any canid.",
]
import random as rnd


def fetch(url: str) -> dict | list:
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def get_image_url(animal: str) -> str:
    cfg = APIS.get(animal.lower())
    if not cfg:
        raise ValueError(f"Unsupported animal. Choose from: {', '.join(APIS)}")
    data = fetch(cfg["url"])
    return cfg["parse"](data)


def get_fact(animal: str) -> str:
    if animal == "cat":
        data = fetch(CAT_FACTS_URL)
        return data.get("fact","")
    elif animal == "dog":
        return rnd.choice(DOG_FACTS)
    elif animal == "fox":
        return rnd.choice(FOX_FACTS)
    return ""


def get_dog_breeds() -> list[str]:
    data = fetch(APIS["dog"]["breed_list"])
    return sorted(data.get("message", {}).keys())


def display_animal(animal: str, with_fact: bool = False) -> None:
    url = get_image_url(animal)
    emoji = {"dog":"🐶","cat":"🐱","fox":"🦊","duck":"🦆"}.get(animal,"🐾")
    print(f"\n  {emoji}  Random {animal.title()} Image:")
    print(f"  🔗  {url}")

    if with_fact:
        fact = get_fact(animal)
        if fact:
            print(f"\n  📚  Fun fact: {fact}")
    print()


def interactive():
    print("=== Random Animal Image ===")
    print(f"Animals: {', '.join(APIS.keys())}")
    print("Commands: <animal> | breeds | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "breeds":
            try:
                breeds = get_dog_breeds()
                print(f"  Dog breeds ({len(breeds)}):")
                cols = 4
                for i in range(0, len(breeds), cols):
                    row = breeds[i:i+cols]
                    print("  " + "  ".join(f"{b:<20}" for b in row))
            except ValueError as e: print(f"  Error: {e}")
        elif cmd in APIS:
            try:
                display_animal(cmd, with_fact=True)
            except ValueError as e: print(f"  Error: {e}")
        else:
            print(f"  Choose from: {', '.join(APIS.keys())} | breeds | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Random Animal Image Fetcher")
    parser.add_argument("--animal",  choices=list(APIS.keys()), default=None)
    parser.add_argument("--fact",    action="store_true", help="Include a fun fact")
    parser.add_argument("--count",   type=int, default=1, help="Number of images")
    parser.add_argument("--breeds",  action="store_true", help="List dog breeds")
    args = parser.parse_args()

    try:
        if args.breeds:
            breeds = get_dog_breeds()
            print(f"Dog breeds ({len(breeds)}): {', '.join(breeds)}")
        elif args.animal:
            for _ in range(args.count):
                display_animal(args.animal, with_fact=args.fact)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
