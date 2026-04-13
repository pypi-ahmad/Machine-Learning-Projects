"""GitHub Repo Viewer — CLI tool.

View details about a GitHub repository: stats, recent commits,
open issues, contributors, and language breakdown.

Usage:
    python main.py
    python main.py --repo torvalds/linux
    python main.py --repo python/cpython --issues --commits
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime


API     = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github.v3+json",
           "User-Agent": "github-repo-viewer-cli"}


def gh_get(path: str) -> dict | list:
    url = f"{API}{path}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404: raise ValueError(f"Not found: {path}")
        if e.code == 403: raise ValueError("Rate limit exceeded.")
        raise ValueError(f"HTTP {e.code}")
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def fmt_date(s: str) -> str:
    try: return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y")
    except: return s or "—"


def fmt_num(n) -> str:
    if n is None: return "—"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)


def display_repo(owner: str, repo: str) -> None:
    r = gh_get(f"/repos/{owner}/{repo}")
    print(f"\n{'═'*56}")
    print(f"  📦  {r['full_name']}")
    if r.get("description"):
        print(f"  📝  {r['description']}")
    if r.get("homepage"):
        print(f"  🔗  {r['homepage']}")
    print(f"{'─'*56}")
    print(f"  ⭐  Stars:       {fmt_num(r.get('stargazers_count'))}")
    print(f"  🍴  Forks:       {fmt_num(r.get('forks_count'))}")
    print(f"  👁  Watchers:    {fmt_num(r.get('watchers_count'))}")
    print(f"  🐛  Open Issues: {fmt_num(r.get('open_issues_count'))}")
    print(f"  📏  Size:        {r.get('size',0)/1024:.1f} MB")
    print(f"  💬  Language:    {r.get('language') or '—'}")
    print(f"  🏷  License:     {(r.get('license') or {}).get('name','—')}")
    print(f"  🌿  Default br:  {r.get('default_branch','—')}")
    print(f"  📅  Created:     {fmt_date(r.get('created_at',''))}")
    print(f"  🔄  Pushed:      {fmt_date(r.get('pushed_at',''))}")
    topics = r.get("topics", [])
    if topics:
        print(f"  🏷  Topics:      {', '.join(topics[:8])}")
    print(f"{'═'*56}\n")


def display_commits(owner: str, repo: str, n: int = 10) -> None:
    commits = gh_get(f"/repos/{owner}/{repo}/commits?per_page={n}")
    print(f"  Recent {len(commits)} commits:")
    print(f"  {'─'*52}")
    for c in commits:
        sha  = c["sha"][:7]
        msg  = c["commit"]["message"].splitlines()[0][:50]
        author = c["commit"]["author"]["name"][:18]
        date   = fmt_date(c["commit"]["author"]["date"])
        print(f"  {sha}  {date}  {author:<18}  {msg}")
    print()


def display_issues(owner: str, repo: str, n: int = 10) -> None:
    issues = gh_get(f"/repos/{owner}/{repo}/issues?state=open&per_page={n}")
    prs    = [i for i in issues if "pull_request" in i]
    real   = [i for i in issues if "pull_request" not in i]
    print(f"  Open issues ({len(real)} issues, {len(prs)} PRs shown):")
    print(f"  {'─'*52}")
    for i in real[:n]:
        labels = ",".join(l["name"] for l in i.get("labels", [])[:3])
        label_str = f" [{labels}]" if labels else ""
        print(f"  #{i['number']:<5}  {i['title'][:50]}{label_str}")
    print()


def display_contributors(owner: str, repo: str, n: int = 10) -> None:
    contribs = gh_get(f"/repos/{owner}/{repo}/contributors?per_page={n}")
    print(f"  Top contributors:")
    print(f"  {'─'*40}")
    for c in contribs[:n]:
        print(f"  {c['login']:<22}  {c['contributions']:>6} commits")
    print()


def display_languages(owner: str, repo: str) -> None:
    langs = gh_get(f"/repos/{owner}/{repo}/languages")
    if not langs: return
    total = sum(langs.values())
    print(f"  Languages:")
    for lang, bytes_ in sorted(langs.items(), key=lambda x: -x[1]):
        pct = bytes_ / total * 100
        bar = "█" * int(pct / 3)
        print(f"  {lang:<20}  {pct:5.1f}%  {bar}")
    print()


def interactive() -> None:
    print("=== GitHub Repo Viewer ===")
    print("Enter as owner/repo (e.g. python/cpython)\n")
    while True:
        inp = input("  Repo (or 'quit'): ").strip()
        if inp.lower() in ("quit", "q", ""): break
        parts = inp.split("/")
        if len(parts) != 2:
            print("  Format: owner/repo"); continue
        owner, repo = parts
        try:
            display_repo(owner, repo)
            display_languages(owner, repo)
            if input("  Show recent commits? (y/n): ").strip().lower() == "y":
                display_commits(owner, repo)
            if input("  Show open issues?   (y/n): ").strip().lower() == "y":
                display_issues(owner, repo)
            if input("  Show contributors?  (y/n): ").strip().lower() == "y":
                display_contributors(owner, repo)
        except ValueError as e:
            print(f"  Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="GitHub Repo Viewer")
    parser.add_argument("--repo",         metavar="OWNER/REPO")
    parser.add_argument("--commits",      action="store_true")
    parser.add_argument("--issues",       action="store_true")
    parser.add_argument("--contributors", action="store_true")
    parser.add_argument("--n",            type=int, default=10)
    args = parser.parse_args()

    if args.repo:
        parts = args.repo.split("/")
        if len(parts) != 2:
            print("Error: use owner/repo format", file=sys.stderr); sys.exit(1)
        owner, repo = parts
        try:
            display_repo(owner, repo)
            display_languages(owner, repo)
            if args.commits:      display_commits(owner, repo, args.n)
            if args.issues:       display_issues(owner, repo, args.n)
            if args.contributors: display_contributors(owner, repo, args.n)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr); sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
