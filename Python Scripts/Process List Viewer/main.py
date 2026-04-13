"""Process List Viewer — CLI tool.

Lists running processes, filters by name/PID/user, shows CPU and
memory usage, and terminates processes by PID.
Uses psutil if available, falls back to platform-native commands.

Usage:
    python main.py
"""

import subprocess
import sys


# ---------------------------------------------------------------------------
# Backend: psutil (preferred) or platform fallback
# ---------------------------------------------------------------------------

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _get_processes_psutil() -> list[dict]:
    procs = []
    for p in psutil.process_iter(
        ["pid", "name", "username", "status", "cpu_percent", "memory_info", "create_time"]
    ):
        try:
            info = p.info
            mem  = info["memory_info"].rss if info["memory_info"] else 0
            procs.append({
                "pid":    info["pid"],
                "name":   info["name"] or "",
                "user":   info["username"] or "",
                "status": info["status"] or "",
                "cpu":    info["cpu_percent"] or 0.0,
                "mem":    mem,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs


def _get_processes_fallback() -> list[dict]:
    procs = []
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True
        ).stdout
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 5:
                try:
                    mem = int(parts[4].replace(",", "").replace(" K", "")) * 1024
                except ValueError:
                    mem = 0
                try:
                    pid = int(parts[1])
                except ValueError:
                    pid = 0
                procs.append({"pid": pid, "name": parts[0],
                               "user": "", "status": "", "cpu": 0.0, "mem": mem})
    else:
        out = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        ).stdout
        for line in out.splitlines()[1:]:
            parts = line.split(None, 10)
            if len(parts) >= 11:
                try:
                    procs.append({
                        "pid":    int(parts[1]),
                        "name":   parts[10].split("/")[-1][:30],
                        "user":   parts[0],
                        "status": parts[7],
                        "cpu":    float(parts[2]),
                        "mem":    int(float(parts[3]) * 10),
                    })
                except (ValueError, IndexError):
                    pass
    return procs


def get_processes() -> list[dict]:
    if HAS_PSUTIL:
        return _get_processes_psutil()
    return _get_processes_fallback()


def kill_process(pid: int) -> tuple[bool, str]:
    if HAS_PSUTIL:
        try:
            p = psutil.Process(pid)
            p.terminate()
            return True, f"Sent SIGTERM to PID {pid} ({p.name()})"
        except psutil.NoSuchProcess:
            return False, f"PID {pid} not found."
        except psutil.AccessDenied:
            return False, f"Access denied for PID {pid}."
    else:
        if sys.platform == "win32":
            r = subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                capture_output=True, text=True)
        else:
            r = subprocess.run(["kill", str(pid)], capture_output=True, text=True)
        if r.returncode == 0:
            return True, f"Killed PID {pid}"
        return False, r.stderr.strip()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def human_mem(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f}K"
    if n < 1024 ** 3:
        return f"{n/1024**2:.1f}M"
    return f"{n/1024**3:.1f}G"


def print_procs(procs: list[dict], limit: int = 30) -> None:
    print(f"\n  {'PID':>7}  {'CPU%':>6}  {'MEM':>8}  {'Status':<10}  {'User':<15}  Name")
    print(f"  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*15}  {'-'*30}")
    for p in procs[:limit]:
        print(f"  {p['pid']:>7}  {p['cpu']:>5.1f}%  {human_mem(p['mem']):>8}"
              f"  {p['status']:<10}  {(p['user'] or '')[:15]:<15}  {p['name'][:40]}")
    if len(procs) > limit:
        print(f"\n  ... {len(procs) - limit} more processes")
    print(f"\n  Total: {len(procs)} processes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Process List Viewer
-------------------
1. List all processes
2. Filter by name
3. Filter by user
4. Top CPU consumers
5. Top memory consumers
6. Kill process by PID
0. Quit
"""


def main() -> None:
    if not HAS_PSUTIL:
        print("  Note: psutil not installed — using system commands (limited info).")
    print("Process List Viewer")

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            procs = get_processes()
            n_s = input("  Processes to show (default 30): ").strip()
            n = int(n_s) if n_s.isdigit() else 30
            print_procs(sorted(procs, key=lambda p: p["pid"]), n)

        elif choice == "2":
            procs = get_processes()
            kw = input("  Name contains: ").strip().lower()
            filtered = [p for p in procs if kw in p["name"].lower()]
            print_procs(filtered)

        elif choice == "3":
            procs = get_processes()
            user = input("  Username: ").strip().lower()
            filtered = [p for p in procs if user in (p["user"] or "").lower()]
            print_procs(filtered)

        elif choice == "4":
            procs = get_processes()
            n_s = input("  How many? (default 10): ").strip()
            n = int(n_s) if n_s.isdigit() else 10
            top = sorted(procs, key=lambda p: p["cpu"], reverse=True)
            print_procs(top, n)

        elif choice == "5":
            procs = get_processes()
            n_s = input("  How many? (default 10): ").strip()
            n = int(n_s) if n_s.isdigit() else 10
            top = sorted(procs, key=lambda p: p["mem"], reverse=True)
            print_procs(top, n)

        elif choice == "6":
            pid_s = input("  PID to kill: ").strip()
            if not pid_s.isdigit():
                print("  Invalid PID.")
                continue
            pid = int(pid_s)
            confirm = input(f"  Kill PID {pid}? (y/n): ").strip().lower()
            if confirm == "y":
                ok, msg = kill_process(pid)
                icon = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
                print(f"  {icon} {msg}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
