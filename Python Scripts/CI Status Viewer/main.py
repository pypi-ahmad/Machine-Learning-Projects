"""CI Status Viewer — CLI developer tool.

View CI/CD pipeline status for GitHub Actions, GitLab CI,
and generic CI systems directly from the terminal.

Usage:
    python main.py
    python main.py --github owner/repo
    python main.py --github owner/repo --token YOUR_TOKEN
    python main.py --watch owner/repo --interval 30
    python main.py --local   (parse .github/workflows/)
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m",
        "magenta": "\033[95m", "blue": "\033[94m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _api_get(url: str, token: str = None, accept: str = None) -> dict | list | None:
    headers = {"User-Agent": "CI-Status-Viewer/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        code = e.code
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if code == 403:
            raise RuntimeError("HTTP 403 — rate limited or bad token")
        if code == 404:
            raise RuntimeError("HTTP 404 — repository not found")
        raise RuntimeError(f"HTTP {code}: {body[:120]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


# ── Status symbols ─────────────────────────────────────────────────────────────

def status_icon(status: str, conclusion: str = None) -> tuple[str, str]:
    """Return (icon, color) for a run status."""
    s = (status or "").lower()
    co = (conclusion or "").lower()

    if s == "completed":
        if co == "success":          return "✓", "green"
        if co == "failure":          return "✗", "red"
        if co == "cancelled":        return "○", "yellow"
        if co in ("skipped",):       return "—", "dim"
        if co == "timed_out":        return "⏱", "yellow"
        if co == "action_required":  return "!", "yellow"
        return "?", "dim"
    if s in ("in_progress", "running"):
        return "⟳", "cyan"
    if s in ("queued", "waiting", "pending"):
        return "·", "yellow"
    return "?", "dim"


def relative_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        s = int(delta.total_seconds())
        if s < 60:    return f"{s}s ago"
        if s < 3600:  return f"{s//60}m ago"
        if s < 86400: return f"{s//3600}h ago"
        return f"{s//86400}d ago"
    except Exception:
        return iso[:10] if iso else ""


def duration_str(started: str, ended: str) -> str:
    try:
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        e = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        diff = int((e - s).total_seconds())
        if diff < 60:    return f"{diff}s"
        if diff < 3600:  return f"{diff//60}m {diff%60}s"
        return f"{diff//3600}h {(diff%3600)//60}m"
    except Exception:
        return ""


# ── GitHub Actions ─────────────────────────────────────────────────────────────

BASE = "https://api.github.com"
GH_ACCEPT = "application/vnd.github+json"


def gh_workflow_runs(repo: str, token: str = None,
                     branch: str = None, limit: int = 10) -> list[dict]:
    url = f"{BASE}/repos/{repo}/actions/runs?per_page={limit}"
    if branch:
        url += f"&branch={urllib.parse.quote(branch)}"
    data = _api_get(url, token=token, accept=GH_ACCEPT)
    return data.get("workflow_runs", [])


def gh_workflows(repo: str, token: str = None) -> list[dict]:
    url  = f"{BASE}/repos/{repo}/actions/workflows"
    data = _api_get(url, token=token, accept=GH_ACCEPT)
    return data.get("workflows", [])


def gh_jobs(repo: str, run_id: int, token: str = None) -> list[dict]:
    url  = f"{BASE}/repos/{repo}/actions/runs/{run_id}/jobs"
    data = _api_get(url, token=token, accept=GH_ACCEPT)
    return data.get("jobs", [])


def print_github_runs(runs: list[dict], repo: str, show_jobs: bool = False,
                      token: str = None):
    if not runs:
        print(c("  No workflow runs found.", "yellow"))
        return

    print(c(f"\n  GitHub Actions — {repo}\n", "bold"))
    print(f"  {'#':>8}  {'Workflow':28}  {'Branch':20}  {'Status':10}  {'When':12}  Duration")
    print(c("  " + "─" * 90, "dim"))

    for run in runs:
        icon, col  = status_icon(run.get("status"), run.get("conclusion"))
        num        = run.get("run_number", "?")
        name       = run.get("name", "?")[:27]
        branch     = run.get("head_branch", "?")[:19]
        when       = relative_time(run.get("updated_at", ""))
        dur        = duration_str(run.get("run_started_at",""), run.get("updated_at",""))
        status_s   = (run.get("conclusion") or run.get("status") or "?")[:9]

        print(f"  {c(str(num),'dim'):>14}  {name:28}  {c(branch,'cyan'):30}  "
              f"{c(icon+' '+status_s, col):18}  {when:12}  {dur}")

        if show_jobs:
            try:
                jobs = gh_jobs(repo, run["id"], token)
                for job in jobs[:6]:
                    ji, jc = status_icon(job.get("status"), job.get("conclusion"))
                    jname  = job.get("name","")[:40]
                    jdur   = duration_str(job.get("started_at",""), job.get("completed_at",""))
                    print(f"       {c(ji,jc)}  {c(jname,'dim'):42} {jdur}")
            except RuntimeError as e:
                print(c(f"    (jobs: {e})", "dim"))


def print_github_summary(repo: str, token: str = None):
    """Show workflow status summary table."""
    try:
        workflows = gh_workflows(repo, token)
    except RuntimeError as e:
        print(c(f"  Error: {e}", "red"))
        return

    print(c(f"\n  Workflow Summary — {repo}\n", "bold"))
    for wf in workflows:
        state_col = "green" if wf.get("state") == "active" else "dim"
        print(f"  {c(wf.get('name','?'), 'cyan'):35}  "
              f"{c(wf.get('state','?'), state_col):10}  "
              f"id={wf.get('id')}")


# ── Local workflow parser ──────────────────────────────────────────────────────

def parse_local_workflows(directory: str = "."):
    """Find and summarise .github/workflows/*.yml files."""
    wf_dir = os.path.join(directory, ".github", "workflows")
    if not os.path.isdir(wf_dir):
        print(c(f"  No .github/workflows/ found in {directory}", "yellow"))
        return

    files = [f for f in os.listdir(wf_dir) if f.endswith((".yml", ".yaml"))]
    if not files:
        print(c("  No workflow files found.", "yellow"))
        return

    print(c(f"\n  Local Workflows — {wf_dir}\n", "bold"))
    for fname in sorted(files):
        path = os.path.join(wf_dir, fname)
        try:
            with open(path) as f:
                content = f.read()
        except OSError:
            continue

        # Extract workflow name (best-effort, no YAML parser)
        name_m    = re.search(r"^name:\s*(.+)", content, re.MULTILINE)
        triggers  = re.findall(r"on:\s*\[([^\]]+)\]|on:\s*\n((?:\s+\w+[:\n])+)", content)
        jobs      = re.findall(r"^\s{2}([a-zA-Z_][\w-]*):\s*$", content, re.MULTILINE)
        steps_cnt = len(re.findall(r"^\s*-\s+(?:name:|uses:|run:)", content, re.MULTILINE))

        name = name_m.group(1).strip() if name_m else fname
        print(f"  {c(fname,'cyan'):35}  {c(name,'bold')}")
        print(f"    Jobs: {c(', '.join(jobs[:6]),'dim')}  |  Steps: ~{steps_cnt}")
        print()


# ── Watch mode ─────────────────────────────────────────────────────────────────

def watch(repo: str, token: str, interval: int, branch: str):
    print(c(f"\n  Watching {repo} (every {interval}s, Ctrl+C to stop)\n", "dim"))
    last_run_id = None
    try:
        while True:
            try:
                runs = gh_workflow_runs(repo, token, branch=branch, limit=5)
                if runs:
                    run = runs[0]
                    rid = run.get("id")
                    icon, col = status_icon(run.get("status"), run.get("conclusion"))
                    ts   = time.strftime("%H:%M:%S")
                    name = run.get("name","")[:30]
                    status_s = (run.get("conclusion") or run.get("status") or "?")

                    line = (f"  {c(ts,'dim')}  #{run.get('run_number')}  "
                            f"{name:32}  {c(icon+' '+status_s, col)}")

                    if rid != last_run_id:
                        print(line + c("  ← new", "cyan"))
                        last_run_id = rid
                    else:
                        print(line)
            except RuntimeError as e:
                print(c(f"  [{time.strftime('%H:%M:%S')}] Error: {e}", "red"))

            time.sleep(interval)
    except KeyboardInterrupt:
        print(c("\n  Stopped.", "dim"))


def interactive_mode():
    print(c("CI Status Viewer\n", "bold"))
    print("Commands: github <owner/repo>, local [dir], watch <owner/repo>, quit\n")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        print(c("  GITHUB_TOKEN detected — will use for authenticated requests.\n", "dim"))

    while True:
        try:
            line = input(c("ci> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower() if parts else ""

        if cmd == "github" and len(parts) > 1:
            repo   = parts[1]
            branch = parts[2] if len(parts) > 2 else None
            try:
                runs = gh_workflow_runs(repo, token, branch=branch, limit=10)
                print_github_runs(runs, repo, token=token)
            except RuntimeError as e:
                print(c(f"  Error: {e}", "red"))

        elif cmd == "local":
            d = parts[1] if len(parts) > 1 else "."
            parse_local_workflows(d)

        elif cmd == "watch" and len(parts) > 1:
            repo = parts[1]
            try:
                interval = int(parts[2]) if len(parts) > 2 else 30
            except ValueError:
                interval = 30
            watch(repo, token, interval, branch=None)

        elif cmd in ("quit", "exit", "q"):
            break
        else:
            print(c("  Unknown command.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="CI pipeline status viewer")
    parser.add_argument("--github",    metavar="OWNER/REPO", help="GitHub repository")
    parser.add_argument("--token",     metavar="TOKEN",      help="GitHub personal access token")
    parser.add_argument("--branch",    metavar="BRANCH",     help="Filter by branch")
    parser.add_argument("--limit",     type=int, default=10, help="Number of runs to show")
    parser.add_argument("--jobs",      action="store_true",  help="Show job details per run")
    parser.add_argument("--summary",   action="store_true",  help="Show workflow summary")
    parser.add_argument("--local",     metavar="DIR", nargs="?", const=".",
                        help="Parse local .github/workflows/ files")
    parser.add_argument("--watch",     action="store_true",  help="Watch for new runs")
    parser.add_argument("--interval",  type=int, default=30, help="Watch poll interval (seconds)")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if args.local is not None:
        parse_local_workflows(args.local)
        return

    if args.github:
        if args.watch:
            watch(args.github, token, args.interval, args.branch)
            return
        if args.summary:
            print_github_summary(args.github, token)
            return
        try:
            runs = gh_workflow_runs(args.github, token,
                                    branch=args.branch, limit=args.limit)
            print_github_runs(runs, args.github, show_jobs=args.jobs, token=token)
        except RuntimeError as e:
            print(c(f"Error: {e}", "red"))
            sys.exit(1)
        return

    interactive_mode()


if __name__ == "__main__":
    main()
