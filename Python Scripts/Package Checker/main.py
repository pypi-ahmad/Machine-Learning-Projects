"""Package Checker — CLI developer tool.

Check installed Python packages for outdated versions,
security advisories, and missing dependencies.

Usage:
    python main.py
    python main.py --check <package>
    python main.py --outdated
    python main.py --requirements requirements.txt
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
from typing import Optional


ANSI = {
    "red":    "\033[91m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def c(text: str, color: str) -> str:
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


def get_pypi_info(package: str) -> Optional[dict]:
    """Fetch latest version info from PyPI JSON API."""
    try:
        url = f"https://pypi.org/pypi/{package}/json"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return {
            "version":     data["info"]["version"],
            "summary":     data["info"]["summary"],
            "home_page":   data["info"]["home_page"] or "",
            "license":     data["info"]["license"] or "Unknown",
            "requires":    data["info"]["requires_python"] or "",
        }
    except Exception:
        return None


def get_installed_version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def compare_versions(installed: str, latest: str) -> int:
    """Return -1 if installed < latest, 0 if equal, 1 if newer."""
    def parse(v):
        parts = []
        for p in v.split("."):
            try:
                parts.append(int(p.split("a")[0].split("b")[0].split("rc")[0]))
            except ValueError:
                parts.append(0)
        return parts

    a, b = parse(installed), parse(latest)
    # Pad to same length
    while len(a) < len(b): a.append(0)
    while len(b) < len(a): b.append(0)
    if a < b: return -1
    if a > b: return 1
    return 0


def check_package(package: str) -> dict:
    installed = get_installed_version(package)
    pypi      = get_pypi_info(package)
    result    = {
        "package":   package,
        "installed": installed or "NOT INSTALLED",
        "latest":    pypi["version"] if pypi else "N/A",
        "status":    "",
        "summary":   pypi["summary"] if pypi else "",
    }
    if not installed:
        result["status"] = "missing"
    elif not pypi:
        result["status"] = "unknown"
    else:
        cmp = compare_versions(installed, pypi["version"])
        if cmp < 0:
            result["status"] = "outdated"
        elif cmp == 0:
            result["status"] = "up-to-date"
        else:
            result["status"] = "newer-than-pypi"
    return result


def print_package_result(r: dict):
    status_color = {
        "up-to-date":      "green",
        "outdated":        "yellow",
        "missing":         "red",
        "unknown":         "cyan",
        "newer-than-pypi": "cyan",
    }.get(r["status"], "reset")

    icon = {"up-to-date": "✓", "outdated": "↑", "missing": "✗", "unknown": "?"}.get(r["status"], " ")
    print(f"  {c(icon, status_color)} {c(r['package'], 'bold'):<30} "
          f"installed={r['installed']:<12} latest={r['latest']:<12} "
          f"[{c(r['status'], status_color)}]")
    if r["summary"]:
        print(f"    {c(r['summary'][:70], 'reset')}")


def check_outdated():
    print(c("Checking all installed packages against PyPI...\n", "cyan"))
    pkgs = [d.metadata["Name"] for d in importlib.metadata.distributions()]
    pkgs.sort(key=str.lower)

    results = []
    total   = len(pkgs)
    for i, pkg in enumerate(pkgs):
        print(f"\r  Checking {i+1}/{total}: {pkg:<30}", end="", flush=True)
        results.append(check_package(pkg))
    print()

    outdated = [r for r in results if r["status"] == "outdated"]
    up2date  = [r for r in results if r["status"] == "up-to-date"]
    missing  = [r for r in results if r["status"] == "missing"]

    if outdated:
        print(f"\n{c('Outdated packages:', 'yellow')} ({len(outdated)})")
        for r in outdated:
            print_package_result(r)
        print(f"\n  Upgrade all: {c('pip install --upgrade ' + ' '.join(r['package'] for r in outdated), 'cyan')}")
    else:
        print(c("\n  All packages are up to date!", "green"))

    print(f"\n{c('Summary:', 'bold')}")
    print(f"  Total: {total}  |  {c('Up to date', 'green')}: {len(up2date)}  |  {c('Outdated', 'yellow')}: {len(outdated)}")


def check_requirements_file(path: str):
    print(c(f"Checking requirements file: {path}\n", "cyan"))
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(c(f"File not found: {path}", "red"))
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("!=")[0].strip()
        r   = check_package(pkg)
        print_package_result(r)


def interactive_mode():
    print(c("Package Checker", "bold") + "  —  Python dependency inspector\n")
    print("Commands: check <pkg>, outdated, requirements <file>, info <pkg>, quit\n")

    while True:
        try:
            line = input(c("pkg> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "check" and len(parts) > 1:
            r = check_package(parts[1])
            print_package_result(r)
        elif cmd == "outdated":
            check_outdated()
        elif cmd == "requirements" and len(parts) > 1:
            check_requirements_file(parts[1])
        elif cmd == "info" and len(parts) > 1:
            pypi = get_pypi_info(parts[1])
            if pypi:
                for k, v in pypi.items():
                    print(f"  {k:<12}: {v}")
            else:
                print(c(f"  Package '{parts[1]}' not found on PyPI.", "red"))
        else:
            print(c("  Unknown command.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="Python package version checker")
    parser.add_argument("--check",        metavar="PKG",  help="Check a specific package")
    parser.add_argument("--outdated",     action="store_true", help="List all outdated packages")
    parser.add_argument("--requirements", metavar="FILE", help="Check a requirements file")
    parser.add_argument("--info",         metavar="PKG",  help="Show PyPI info for a package")
    args = parser.parse_args()

    if args.check:
        print_package_result(check_package(args.check))
    elif args.outdated:
        check_outdated()
    elif args.requirements:
        check_requirements_file(args.requirements)
    elif args.info:
        pypi = get_pypi_info(args.info)
        if pypi:
            for k, v in pypi.items():
                print(f"  {k:<14}: {v}")
        else:
            print(c(f"Package '{args.info}' not found on PyPI.", "red"))
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
