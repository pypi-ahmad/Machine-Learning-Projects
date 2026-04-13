"""Log File Analyzer — CLI tool.

Parses log files (Apache/Nginx common format, Python logging,
or generic line-based logs).  Supports filtering by level/keyword,
date range, error counting, and frequency analysis.

Usage:
    python main.py
    python main.py app.log
"""

import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# Python logging: 2024-01-15 12:34:56,789 - name - LEVEL - message
PYTHON_LOG_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>[\d:,]+)"
    r"\s+-\s+(?P<logger>\S+)\s+-\s+(?P<level>[A-Z]+)\s+-\s+(?P<message>.*)"
)

# Apache/Nginx combined: 127.0.0.1 - - [01/Jan/2024:12:00:00 +0000] "GET / HTTP/1.1" 200 512
APACHE_LOG_RE = re.compile(
    r"(?P<ip>[\d.]+)\s+\S+\s+\S+\s+\[(?P<datetime>[^\]]+)\]\s+"
    r'"(?P<method>\S+)\s+(?P<path>\S+)\s+\S+"\s+(?P<status>\d+)\s+(?P<size>\d+|-)'
)

# Generic: anything with a log level keyword
GENERIC_LEVEL_RE = re.compile(
    r"\b(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b",
    re.IGNORECASE
)

LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "WARN": 2,
               "ERROR": 3, "CRITICAL": 4, "FATAL": 4}


def detect_format(lines: list[str]) -> str:
    for line in lines[:20]:
        if PYTHON_LOG_RE.match(line):
            return "python"
        if APACHE_LOG_RE.match(line):
            return "apache"
    return "generic"


def parse_lines(lines: list[str], fmt: str) -> list[dict]:
    records = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        if fmt == "python":
            m = PYTHON_LOG_RE.match(line)
            if m:
                records.append({
                    "date": m.group("date"),
                    "level": m.group("level").upper(),
                    "logger": m.group("logger"),
                    "message": m.group("message"),
                    "raw": line,
                })
                continue
        elif fmt == "apache":
            m = APACHE_LOG_RE.match(line)
            if m:
                records.append({
                    "ip": m.group("ip"),
                    "datetime": m.group("datetime"),
                    "method": m.group("method"),
                    "path": m.group("path"),
                    "status": m.group("status"),
                    "size": m.group("size"),
                    "level": "INFO" if m.group("status").startswith("2") else "ERROR",
                    "raw": line,
                })
                continue
        # Generic fallback
        m = GENERIC_LEVEL_RE.search(line)
        level = m.group("level").upper() if m else "INFO"
        if level == "WARN":
            level = "WARNING"
        records.append({"level": level, "message": line, "raw": line})
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def level_counts(records: list[dict]) -> Counter:
    return Counter(r.get("level", "UNKNOWN") for r in records)


def top_errors(records: list[dict], n: int = 10) -> list[tuple[str, int]]:
    msgs = [r.get("message", r.get("raw", ""))[:120]
            for r in records if r.get("level") in ("ERROR", "CRITICAL", "FATAL")]
    return Counter(msgs).most_common(n)


def filter_by_level(records: list[dict], min_level: str) -> list[dict]:
    threshold = LEVEL_ORDER.get(min_level.upper(), 0)
    return [r for r in records if LEVEL_ORDER.get(r.get("level", ""), 0) >= threshold]


def filter_by_keyword(records: list[dict], keyword: str) -> list[dict]:
    kw = keyword.lower()
    return [r for r in records if kw in r.get("raw", "").lower()]


def ip_frequency(records: list[dict]) -> Counter:
    return Counter(r["ip"] for r in records if "ip" in r)


def status_frequency(records: list[dict]) -> Counter:
    return Counter(r["status"] for r in records if "status" in r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Log File Analyzer
-----------------
1. Summary (level counts)
2. Show lines by level (filter)
3. Search by keyword
4. Top error messages
5. IP / status frequency (Apache/Nginx)
6. Load different file
0. Quit
"""

LEVEL_COLORS = {
    "DEBUG":    "\033[37m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "WARN":     "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
    "FATAL":    "\033[35m",
}
RESET = "\033[0m"


def colorize(record: dict) -> str:
    level = record.get("level", "")
    color = LEVEL_COLORS.get(level, "")
    return f"  {color}{record['raw'][:120]}{RESET}"


def load_file(path: Path) -> tuple[list[dict], str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    fmt   = detect_format(lines)
    recs  = parse_lines(lines, fmt)
    return recs, fmt


def main() -> None:
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    records: list[dict] = []
    fmt = "generic"

    def load(path: Path) -> bool:
        nonlocal records, fmt
        if not path.exists():
            print(f"  File not found: {path}")
            return False
        records, fmt = load_file(path)
        print(f"  Loaded {len(records)} lines. Detected format: {fmt}")
        return True

    if path_arg:
        load(Path(path_arg))
    else:
        path_str = input("  Log file path: ").strip().strip('"')
        if path_str:
            load(Path(path_str))

    print("Log File Analyzer")
    work = list(records)

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            if not records:
                print("  No file loaded.")
                continue
            counts = level_counts(work)
            total  = sum(counts.values())
            print(f"\n  Total lines: {total}")
            for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                cnt = counts.get(level, 0)
                bar = "█" * (cnt * 30 // total) if total else ""
                color = LEVEL_COLORS.get(level, "")
                print(f"  {color}{level:<10}{RESET} {cnt:>6}  {bar}")

        elif choice == "2":
            if not records:
                print("  No file loaded.")
                continue
            print("  Levels: DEBUG / INFO / WARNING / ERROR / CRITICAL")
            min_lev = input("  Minimum level (default WARNING): ").strip().upper() or "WARNING"
            filtered = filter_by_level(work, min_lev)
            n_str = input("  Lines to show (default 20): ").strip()
            n = int(n_str) if n_str.isdigit() else 20
            print(f"\n  {len(filtered)} line(s) at {min_lev}+:")
            for r in filtered[:n]:
                print(colorize(r))
            if len(filtered) > n:
                print(f"  ... {len(filtered) - n} more")

        elif choice == "3":
            if not records:
                print("  No file loaded.")
                continue
            kw = input("  Keyword: ").strip()
            if not kw:
                continue
            filtered = filter_by_keyword(work, kw)
            n_str = input("  Lines to show (default 20): ").strip()
            n = int(n_str) if n_str.isdigit() else 20
            print(f"\n  {len(filtered)} match(es) for '{kw}':")
            for r in filtered[:n]:
                print(colorize(r))
            if len(filtered) > n:
                print(f"  ... {len(filtered) - n} more")

        elif choice == "4":
            if not records:
                print("  No file loaded.")
                continue
            errors = top_errors(work)
            if not errors:
                print("  No errors found.")
            else:
                print(f"\n  Top {len(errors)} error message(s):")
                for i, (msg, cnt) in enumerate(errors, 1):
                    print(f"  {i:>3}. [{cnt}x] {msg[:100]}")

        elif choice == "5":
            if not records:
                print("  No file loaded.")
                continue
            ip_freq = ip_frequency(work)
            stat_freq = status_frequency(work)
            if ip_freq:
                print("\n  Top IPs:")
                for ip, cnt in ip_freq.most_common(10):
                    print(f"    {ip:<20} {cnt:>6}")
            if stat_freq:
                print("\n  HTTP Status Codes:")
                for code, cnt in sorted(stat_freq.items()):
                    print(f"    {code}  {cnt:>6}")
            if not ip_freq and not stat_freq:
                print("  No Apache/Nginx records found.")

        elif choice == "6":
            path_str = input("  Log file path: ").strip().strip('"')
            if path_str and load(Path(path_str)):
                work = list(records)

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
