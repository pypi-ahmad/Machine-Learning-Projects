"""Git Helper CLI — Command-line tool.

Shortcuts for common Git workflows:
  status, log summary, branch management, stash helpers,
  undo last commit, clean untracked files, and more.

Usage:
    python main.py
    python main.py status
    python main.py log
"""

import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], capture: bool = True) -> tuple[int, str, str]:
    """Run a command; return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=capture, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git(*args: str, capture: bool = True) -> tuple[int, str, str]:
    return run(["git", *args], capture=capture)


def is_git_repo() -> bool:
    code, _, _ = git("rev-parse", "--is-inside-work-tree")
    return code == 0


def print_separator(char: str = "─", width: int = 50):
    print(char * width)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status():
    """Show short status + branch info."""
    code, out, err = git("status", "-sb")
    if code != 0:
        print("Not a git repository (or git not found).")
        return
    print("\nGit Status")
    print_separator()
    print(out or "(clean — nothing to commit)")


def cmd_log(n: int = 15):
    """Pretty one-line log."""
    fmt = "--pretty=format:%C(yellow)%h%Creset  %C(green)%ad%Creset  %s  %C(blue)[%an]%Creset"
    code, out, _ = git("log", f"-{n}", fmt, "--date=short")
    if code != 0:
        print("No commits yet or not a git repo.")
        return
    print(f"\nLast {n} Commits")
    print_separator()
    print(out)


def cmd_branches():
    """List branches with current marker."""
    _, out, _ = git("branch", "-a", "--sort=-committerdate")
    print("\nBranches")
    print_separator()
    print(out or "(no branches)")


def cmd_diff():
    """Show staged + unstaged diff summary."""
    _, staged, _  = git("diff", "--stat", "--cached")
    _, unstaged, _ = git("diff", "--stat")
    print("\nDiff Summary")
    print_separator()
    print("Staged:")
    print(staged or "  (nothing staged)")
    print("\nUnstaged:")
    print(unstaged or "  (nothing unstaged)")


def cmd_stash_list():
    """Show stash entries."""
    _, out, _ = git("stash", "list")
    print("\nStash List")
    print_separator()
    print(out or "(stash is empty)")


def cmd_quick_commit():
    """Stage all changes and commit with a message."""
    cmd_status()
    msg = input("\nCommit message (blank to cancel): ").strip()
    if not msg:
        print("Cancelled.")
        return
    _, o1, _ = git("add", "-A")
    code, o2, err = git("commit", "-m", msg)
    if code == 0:
        print(f"  Committed: {msg}")
    else:
        print(f"  Failed: {err}")


def cmd_undo_last():
    """Soft-undo the last commit (keeps changes staged)."""
    _, head, _ = git("log", "-1", "--pretty=%H %s")
    confirm = input(f"Undo last commit?\n  {head}\n  (changes will be kept staged) [y/N]: ").strip().lower()
    if confirm == "y":
        code, out, err = git("reset", "--soft", "HEAD~1")
        print(out or err or ("Undone." if code == 0 else "Failed."))


def cmd_push():
    """Push current branch to origin."""
    _, branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    confirm = input(f"Push '{branch}' to origin? [y/N]: ").strip().lower()
    if confirm == "y":
        code, out, err = run(["git", "push", "-u", "origin", branch], capture=False)


def cmd_pull():
    """Pull latest changes."""
    code, out, err = run(["git", "pull", "--rebase"], capture=False)


def cmd_new_branch():
    """Create and checkout a new branch."""
    name = input("New branch name: ").strip()
    if not name:
        return
    code, out, err = git("checkout", "-b", name)
    print(out or err)


def cmd_delete_branch():
    """Delete a local branch."""
    _, out, _ = git("branch")
    print(out)
    name = input("Branch to delete (! for force): ").strip()
    if not name:
        return
    flag = "-D" if name.startswith("!") else "-d"
    name = name.lstrip("!")
    code, out, err = git("branch", flag, name)
    print(out or err)


def cmd_repo_info():
    """Show remote URLs, top-level directory, and HEAD."""
    _, remote, _  = git("remote", "-v")
    _, toplevel, _ = git("rev-parse", "--show-toplevel")
    _, head, _    = git("log", "-1", "--pretty=%H %s")
    print("\nRepository Info")
    print_separator()
    print(f"Root:    {toplevel}")
    print(f"HEAD:    {head}")
    print(f"Remote:\n{remote or '  (no remote)'}")


def cmd_search_commits():
    """Search commits by message keyword."""
    keyword = input("Search keyword: ").strip()
    if not keyword:
        return
    _, out, _ = git("log", "--all", "--oneline", f"--grep={keyword}")
    print(out or "No matching commits.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMMANDS: dict[str, tuple] = {
    "1":  ("status",          cmd_status,        "Show git status"),
    "2":  ("log",             cmd_log,            "Show commit log"),
    "3":  ("diff",            cmd_diff,           "Show diff summary"),
    "4":  ("branches",        cmd_branches,       "List branches"),
    "5":  ("new-branch",      cmd_new_branch,     "Create new branch"),
    "6":  ("delete-branch",   cmd_delete_branch,  "Delete a branch"),
    "7":  ("quick-commit",    cmd_quick_commit,   "Stage all & commit"),
    "8":  ("push",            cmd_push,           "Push current branch"),
    "9":  ("pull",            cmd_pull,           "Pull (rebase)"),
    "10": ("undo",            cmd_undo_last,      "Undo last commit (soft)"),
    "11": ("stash-list",      cmd_stash_list,     "List stash entries"),
    "12": ("info",            cmd_repo_info,      "Repository info"),
    "13": ("search",          cmd_search_commits, "Search commits"),
    "0":  ("quit",            None,               "Quit"),
}

SHORTCUTS = {v[0]: v[1] for v in COMMANDS.values() if v[1]}


def print_menu():
    print("\nGit Helper")
    print_separator()
    for key, (name, _, desc) in COMMANDS.items():
        if key == "0":
            continue
        print(f"  {key:>2}. {name:<18} {desc}")
    print("   0. quit")
    print_separator()


def main():
    # Direct command mode
    if len(sys.argv) > 1:
        cmd_name = sys.argv[1].lower()
        fn = SHORTCUTS.get(cmd_name)
        if fn:
            fn()
        else:
            print(f"Unknown command: {cmd_name}")
            print("Available: " + ", ".join(SHORTCUTS))
        return

    while True:
        print_menu()
        choice = input("Choice: ").strip()
        if choice == "0":
            print("Bye!")
            break
        entry = COMMANDS.get(choice)
        if entry and entry[1]:
            entry[1]()
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
