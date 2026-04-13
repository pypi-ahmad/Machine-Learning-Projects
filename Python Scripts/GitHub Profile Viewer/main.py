"""GitHub Profile Viewer — CLI tool.

Fetch and display a GitHub user's profile, repos, and activity
using the public GitHub API (no auth required for basic info).

Usage:
    python main.py
    python main.py --user torvalds
    python main.py --user torvalds --repos
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime


API = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github.v3+json",
           "User-Agent": "github-profile-viewer-cli"}


def gh_get(path: str) -> dict | list:
    url = f"{API}{path}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"Not found: {path}")
        if e.code == 403:
            raise ValueError("Rate limit exceeded. Wait a minute or add a GitHub token.")
        raise ValueError(f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def fmt_date(s: str) -> str:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y")
    except Exception:
        return s or "—"


def fmt_num(n) -> str:
    if n is None: return "—"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)


def display_profile(user: str) -> None:
    data = gh_get(f"/users/{user}")

    print(f"\n{'═'*50}")
    print(f"  👤  {data.get('name') or data['login']}  (@{data['login']})")
    if data.get("bio"):
        print(f"  📝  {data['bio']}")
    print(f"{'─'*50}")
    print(f"  🌍  {data.get('location') or '—'}")
    print(f"  🏢  {data.get('company') or '—'}")
    print(f"  🔗  {data.get('blog') or '—'}")
    print(f"  📧  {data.get('email') or '—'}")
    print(f"  🐦  {data.get('twitter_username') or '—'}")
    print(f"{'─'*50}")
    print(f"  📦  Repos:       {fmt_num(data.get('public_repos'))}")
    print(f"  👥  Followers:   {fmt_num(data.get('followers'))}")
    print(f"  👣  Following:   {fmt_num(data.get('following'))}")
    print(f"  ⭐  Gists:       {fmt_num(data.get('public_gists'))}")
    print(f"  📅  Joined:      {fmt_date(data.get('created_at',''))}")
    print(f"  🔄  Updated:     {fmt_date(data.get('updated_at',''))}")
    print(f"{'═'*50}\n")


def display_repos(user: str, top: int = 10) -> None:
    repos = gh_get(f"/users/{user}/repos?sort=stargazers_count&per_page=30")
    repos = sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)[:top]

    print(f"\n  Top {len(repos)} repositories for @{user}:")
    print(f"  {'─'*60}")
    for r in repos:
        lang  = r.get("language") or "—"
        stars = fmt_num(r.get("stargazers_count", 0))
        forks = fmt_num(r.get("forks_count", 0))
        desc  = (r.get("description") or "")[:50]
        print(f"  ⭐{stars:>5}  🍴{forks:>5}  [{lang:<12}]  {r['name']}")
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


def main():
    parser = argparse.ArgumentParser(description="GitHub Profile Viewer")
    parser.add_argument("--user",  metavar="USER", help="GitHub username")
    parser.add_argument("--repos", action="store_true", help="Show top repositories")
    parser.add_argument("--top",   type=int, default=10, help="Number of repos to show")
    args = parser.parse_args()

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
