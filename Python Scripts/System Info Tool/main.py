"""System Info Tool — CLI tool.

Displays OS, CPU, memory, disk, network, and Python environment
information using only the standard library.

Usage:
    python main.py
"""

import os
import platform
import socket
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Info collectors
# ---------------------------------------------------------------------------

def os_info() -> dict:
    return {
        "System":      platform.system(),
        "Node name":   platform.node(),
        "Release":     platform.release(),
        "Version":     platform.version(),
        "Machine":     platform.machine(),
        "Processor":   platform.processor() or "N/A",
        "Architecture":platform.architecture()[0],
    }


def python_info() -> dict:
    return {
        "Python version":   sys.version.split()[0],
        "Implementation":   platform.python_implementation(),
        "Executable":       sys.executable,
        "Prefix":           sys.prefix,
        "Platform tag":     platform.platform(),
    }


def disk_info() -> list[dict]:
    import shutil
    drives = []
    if platform.system() == "Windows":
        import string
        import ctypes
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                mount = f"{letter}:\\"
                try:
                    total, used, free = shutil.disk_usage(mount)
                    drives.append({"mount": mount, "total": total,
                                   "used": used, "free": free})
                except (PermissionError, OSError):
                    pass
            bitmask >>= 1
    else:
        for line in _run_cmd("df -h").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 6:
                drives.append({
                    "device": parts[0], "size": parts[1],
                    "used": parts[2], "avail": parts[3],
                    "use%": parts[4], "mount": parts[5],
                })
    return drives


def network_info() -> dict:
    info = {}
    try:
        info["Hostname"] = socket.gethostname()
        info["Local IP"] = socket.gethostbyname(socket.gethostname())
    except Exception:
        info["Hostname"] = "N/A"
        info["Local IP"] = "N/A"
    return info


def env_vars(filter_key: str = "") -> list[tuple[str, str]]:
    items = sorted(os.environ.items())
    if filter_key:
        fk = filter_key.lower()
        items = [(k, v) for k, v in items if fk in k.lower()]
    return items


def _run_cmd(cmd: str) -> str:
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout
    except Exception:
        return ""


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_section(title: str, data: dict) -> None:
    print(f"\n  {'─' * 40}")
    print(f"  {title}")
    print(f"  {'─' * 40}")
    for k, v in data.items():
        print(f"  {k:<22}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
System Info Tool
----------------
1. OS & hardware info
2. Python environment
3. Disk usage
4. Network info
5. Environment variables
6. Full report
0. Quit
"""


def main() -> None:
    print("System Info Tool")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            print_section("Operating System & Hardware", os_info())
            # Try to get CPU count
            cpu_count = os.cpu_count()
            print(f"  {'CPU logical cores':<22}: {cpu_count or 'N/A'}")

        elif choice == "2":
            print_section("Python Environment", python_info())
            print(f"\n  sys.path ({len(sys.path)} entries):")
            for p in sys.path[:8]:
                print(f"    {p}")
            if len(sys.path) > 8:
                print(f"    ... {len(sys.path) - 8} more")

        elif choice == "3":
            import shutil
            print(f"\n  {'─' * 40}")
            print("  Disk Usage")
            print(f"  {'─' * 40}")
            try:
                drives = disk_info()
                if drives and "total" in drives[0]:
                    print(f"\n  {'Drive':<8} {'Total':>10} {'Used':>10} {'Free':>10} {'Use%':>6}")
                    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
                    for d in drives:
                        pct = d["used"] / d["total"] * 100 if d["total"] else 0
                        print(f"  {d['mount']:<8} {human_size(d['total']):>10}"
                              f" {human_size(d['used']):>10}"
                              f" {human_size(d['free']):>10} {pct:>5.1f}%")
                else:
                    for d in drives:
                        print(f"  {d}")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "4":
            print_section("Network", network_info())

        elif choice == "5":
            filt = input("  Filter key (blank = all): ").strip()
            items = env_vars(filt)
            print(f"\n  {len(items)} environment variable(s):")
            for k, v in items[:40]:
                print(f"  {k:<35} = {v[:60]}")
            if len(items) > 40:
                print(f"  ... {len(items) - 40} more")

        elif choice == "6":
            print_section("OS & Hardware",    os_info())
            print_section("Python",           python_info())
            print_section("Network",          network_info())
            # Quick disk
            import shutil
            cwd_usage = shutil.disk_usage(Path.cwd())
            print(f"\n  {'─' * 40}")
            print("  Current Drive Usage")
            print(f"  {'─' * 40}")
            print(f"  Total : {human_size(cwd_usage.total)}")
            print(f"  Used  : {human_size(cwd_usage.used)}")
            print(f"  Free  : {human_size(cwd_usage.free)}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
