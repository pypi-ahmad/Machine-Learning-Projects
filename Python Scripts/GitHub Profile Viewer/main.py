"""GitHub Profile Viewer CLI tool.

Fetch and display a GitHub user's profile, repos, and activity
using the authenticated GitHub CLI API.

Usage:
    python main.py
    python main.py --user torvalds
    python main.py --user torvalds --repos
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime


USERNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")


def gh_get(path: str) -> dict | list:
    """Fetch a GitHub API endpoint through authenticated gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "api", path], capture_output=True, text=True, timeout=15, check=False
        )
    except FileNotFoundError as error:
        raise ValueError("GitHub CLI (gh) is not installed or on PATH.") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("GitHub request timed out.") from error
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "GitHub request failed.")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("GitHub CLI returned invalid JSON.") from error


def validate_username(user: str) -> str:
    if not USERNAME.fullmatch(user):
        raise ValueError("Enter a valid GitHub username.")
    return user


def fmt_date(s: str) -> str:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y")
    except Exception:
        return s or "-"


def fmt_num(n) -> str:
    if n is None: return "-"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)


def display_profile(user: str) -> None:
    user = validate_username(user)
    data = gh_get(f"/users/{user}")

    print(f"\n{'=' * 50}")
    print(f"  {data.get('name') or data['login']}  (@{data['login']})")
    if data.get("bio"):
        print(f"  {data['bio']}")
    print(f"{'-' * 50}")
    print(f"  Location:  {data.get('location') or '-'}")
    print(f"  Company:   {data.get('company') or '-'}")
    print(f"  Website:   {data.get('blog') or '-'}")
    print(f"  Email:     {data.get('email') or '-'}")
    print(f"  Twitter:   {data.get('twitter_username') or '-'}")
    print(f"{'-' * 50}")
    print(f"  Repos:       {fmt_num(data.get('public_repos'))}")
    print(f"  Followers:   {fmt_num(data.get('followers'))}")
    print(f"  Following:   {fmt_num(data.get('following'))}")
    print(f"  Gists:       {fmt_num(data.get('public_gists'))}")
    print(f"  Joined:      {fmt_date(data.get('created_at',''))}")
    print(f"  Updated:     {fmt_date(data.get('updated_at',''))}")
    print(f"{'=' * 50}\n")


def display_repos(user: str, top: int = 10) -> None:
    user = validate_username(user)
    repos = gh_get(f"/users/{user}/repos?sort=stargazers_count&per_page=30")
    repos = sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)[:top]

    print(f"\n  Top {len(repos)} repositories for @{user}:")
    print(f"  {'-' * 60}")
    for r in repos:
        lang  = r.get("language") or "-"
        stars = fmt_num(r.get("stargazers_count", 0))
        forks = fmt_num(r.get("forks_count", 0))
        desc  = (r.get("description") or "")[:50]
        print(f"  stars={stars:>5}  forks={forks:>5}  [{lang:<12}]  {r['name']}")
        if desc: print(f"            {desc}")
    print()


def interactive() -> None:
    print("=== GitHub Profile Viewer ===")
    while True:
        user = input("  GitHub username (or 'quit'): ").strip()
        if user.lower() in ("quit", "q", ""): break
        try:
            display_profile(user)
            show = input("  Show repositories? (y/n): ").strip().lower()
            if show == "y":
                display_repos(user)
        except ValueError as e:
            print(f"  Error: {e}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub Profile Viewer")
    parser.add_argument("--user",  metavar="USER", help="GitHub username")
    parser.add_argument("--repos", action="store_true", help="Show top repositories")
    parser.add_argument("--top",   type=int, default=10, help="Number of repos to show")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")

    if args.user:
        try:
            display_profile(args.user)
            if args.repos:
                display_repos(args.user, args.top)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
