"""Log Parser Dashboard — CLI developer tool.

Parse, filter, analyse, and summarise log files.
Supports common formats: Apache/Nginx access logs,
Python logging, JSON logs, and plain text with timestamps.

Usage:
    python main.py access.log
    python main.py app.log --format python --level ERROR
    python main.py access.log --stats --top 10
    python main.py app.log --tail --follow
    python main.py --watch app.log --level WARN
"""

import argparse
import collections
import json
import os
import re
import sys
import time
from datetime import datetime

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m",
        "magenta": "\033[95m", "blue": "\033[94m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Log formats ────────────────────────────────────────────────────────────────

APACHE_RE = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<method>\S+)?\s*(?P<path>\S+)?\s*(?P<proto>[^"]+)?"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)'
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<agent>[^"]*)")?'
)

PYTHON_RE = re.compile(
    r'(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]?\d*)\s+'
    r'(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s+'
    r'(?P<logger>\S+)?\s*:?\s*(?P<message>.*)'
)

JSON_RE = re.compile(r'^\s*\{.*\}\s*$')

SYSLOG_RE = re.compile(
    r'(?P<time>\w{3}\s+\d+\s+\d+:\d+:\d+)\s+'
    r'(?P<host>\S+)\s+(?P<process>[^:]+):\s+(?P<message>.*)'
)

GENERIC_TS_RE = re.compile(
    r'(?P<time>\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}:\d{2})'
    r'(?:,\d+)?'
    r'\s+(?P<rest>.*)'
)

LEVEL_COLORS = {
    "DEBUG":    "dim",
    "INFO":     "green",
    "WARNING":  "yellow",
    "WARN":     "yellow",
    "ERROR":    "red",
    "CRITICAL": "magenta",
}

LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"]


def level_rank(level: str) -> int:
    return LEVEL_ORDER.index(level.upper()) if level.upper() in LEVEL_ORDER else 0


def detect_format(line: str) -> str:
    if JSON_RE.match(line):
        return "json"
    if APACHE_RE.match(line):
        return "apache"
    if PYTHON_RE.match(line):
        return "python"
    if SYSLOG_RE.match(line):
        return "syslog"
    if GENERIC_TS_RE.match(line):
        return "generic"
    return "plain"


def parse_line(line: str, fmt: str = "auto") -> dict:
    line = line.rstrip("\n")
    if fmt == "auto":
        fmt = detect_format(line)

    entry = {"raw": line, "format": fmt, "level": None, "time": None, "message": line}

    if fmt == "json":
        try:
            data = json.loads(line)
            entry["level"]   = str(data.get("level", data.get("severity", ""))).upper() or None
            entry["time"]    = str(data.get("time", data.get("timestamp", data.get("ts", ""))))
            entry["message"] = str(data.get("message", data.get("msg", line)))
            entry["data"]    = data
        except json.JSONDecodeError:
            pass

    elif fmt == "apache":
        m = APACHE_RE.match(line)
        if m:
            d = m.groupdict()
            status = int(d.get("status", 0))
            entry["level"]   = "ERROR" if status >= 500 else ("WARNING" if status >= 400 else "INFO")
            entry["time"]    = d.get("time", "")
            entry["ip"]      = d.get("ip", "")
            entry["method"]  = d.get("method", "")
            entry["path"]    = d.get("path", "")
            entry["status"]  = status
            entry["size"]    = d.get("size", "")
            entry["message"] = f'{d.get("method","")} {d.get("path","")} → {status}'

    elif fmt == "python":
        m = PYTHON_RE.match(line)
        if m:
            d = m.groupdict()
            entry["level"]   = d["level"].upper() if d["level"] else None
            entry["time"]    = d["time"]
            entry["logger"]  = d.get("logger", "")
            entry["message"] = d.get("message", "")

    elif fmt == "syslog":
        m = SYSLOG_RE.match(line)
        if m:
            d = m.groupdict()
            entry["time"]    = d["time"]
            entry["host"]    = d["host"]
            entry["process"] = d["process"]
            entry["message"] = d["message"]

    elif fmt == "generic":
        m = GENERIC_TS_RE.match(line)
        if m:
            entry["time"]    = m.group("time")
            entry["message"] = m.group("rest")

    return entry


# ── Filtering ──────────────────────────────────────────────────────────────────

def passes_filter(entry: dict, level_min: str = None, grep: str = None,
                  grep_re: re.Pattern = None) -> bool:
    if level_min and entry.get("level"):
        if level_rank(entry["level"]) < level_rank(level_min):
            return False
    if grep and grep.lower() not in entry["raw"].lower():
        return False
    if grep_re and not grep_re.search(entry["raw"]):
        return False
    return True


# ── Display ────────────────────────────────────────────────────────────────────

def format_entry(entry: dict, show_color: bool = True) -> str:
    level = entry.get("level") or ""
    ts    = entry.get("time",    "")[:19]
    msg   = entry.get("message", entry["raw"])

    if not show_color:
        return f"{ts:20} {level:8} {msg}"

    col   = LEVEL_COLORS.get(level.upper(), "reset") if level else "reset"
    ts_s  = c(ts[:19], "dim") if ts else ""
    lv_s  = c(f"{level:8}", col) if level else " " * 8
    msg_s = c(msg, col) if level in ("ERROR", "CRITICAL") else msg

    return f"  {ts_s}  {lv_s}  {msg_s}"


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(entries: list[dict], fmt: str) -> dict:
    stats: dict = {
        "total":         len(entries),
        "by_level":      collections.Counter(),
        "errors":        [],
        "by_hour":       collections.Counter(),
        "by_status":     collections.Counter(),
        "top_paths":     collections.Counter(),
        "top_ips":       collections.Counter(),
    }
    for e in entries:
        if e.get("level"):
            stats["by_level"][e["level"]] += 1
        if e.get("level") in ("ERROR", "CRITICAL"):
            stats["errors"].append(e["message"][:120])
        if e.get("time"):
            m = re.search(r"(\d{2}):\d{2}:\d{2}", e["time"])
            if m:
                stats["by_hour"][int(m.group(1))] += 1
        if e.get("status"):
            stats["by_status"][str(e["status"])] += 1
        if e.get("path"):
            stats["top_paths"][e["path"]] += 1
        if e.get("ip"):
            stats["top_ips"][e["ip"]] += 1
    return stats


def print_stats(stats: dict, top_n: int = 10):
    print(c("\n  ─── Log Statistics ───────────────────────────", "dim"))
    print(f"  Total lines   : {c(str(stats['total']), 'bold')}")

    if stats["by_level"]:
        print(c("\n  By Level:", "bold"))
        for lvl, cnt in sorted(stats["by_level"].items(), key=lambda x: -x[1]):
            col = LEVEL_COLORS.get(lvl, "reset")
            bar = "█" * min(cnt * 20 // max(stats["by_level"].values(), default=1), 20)
            print(f"    {c(lvl,'dim'):10} {c(bar,col):25} {c(str(cnt),'bold')}")

    if stats["by_status"]:
        print(c("\n  HTTP Status Codes:", "bold"))
        for code, cnt in sorted(stats["by_status"].items(), key=lambda x: -x[1])[:top_n]:
            sc = int(code)
            col = "green" if sc < 300 else ("yellow" if sc < 400 else "red")
            print(f"    {c(code, col):8} {cnt}")

    if stats["top_paths"]:
        print(c(f"\n  Top {top_n} Paths:", "bold"))
        for path, cnt in stats["top_paths"].most_common(top_n):
            print(f"    {cnt:>6}  {path}")

    if stats["top_ips"]:
        print(c(f"\n  Top {top_n} IPs:", "bold"))
        for ip, cnt in stats["top_ips"].most_common(top_n):
            print(f"    {cnt:>6}  {ip}")

    if stats["by_hour"]:
        print(c("\n  Activity by Hour:", "bold"))
        mx = max(stats["by_hour"].values(), default=1)
        for h in range(24):
            cnt = stats["by_hour"].get(h, 0)
            bar = "▇" * int(cnt * 30 / mx) if cnt else ""
            print(f"    {h:02d}h  {c(bar,'cyan'):35} {cnt}")

    if stats["errors"]:
        print(c(f"\n  Recent Errors (last 5):", "bold"))
        for msg in stats["errors"][-5:]:
            print(c(f"    ✗ {msg}", "red"))


# ── Follow / watch ─────────────────────────────────────────────────────────────

def follow_file(path: str, fmt: str, level_min: str, grep: str):
    """Tail a file and print new lines as they appear."""
    grep_re = re.compile(grep, re.IGNORECASE) if grep else None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)   # seek to end
            print(c(f"  Following {path} (Ctrl+C to stop)...\n", "dim"))
            while True:
                line = f.readline()
                if line:
                    entry = parse_line(line, fmt)
                    if passes_filter(entry, level_min, grep, grep_re):
                        print(format_entry(entry))
                else:
                    time.sleep(0.2)
    except (FileNotFoundError, PermissionError) as e:
        print(c(f"  {e}", "red"))
    except KeyboardInterrupt:
        print(c("\n  Stopped.", "dim"))


# ── Main processing ────────────────────────────────────────────────────────────

def process_file(path: str, fmt: str = "auto", level_min: str = None,
                 grep: str = None, stats_only: bool = False, top_n: int = 10,
                 tail: int = 0, head: int = 0):
    grep_re = re.compile(grep, re.IGNORECASE) if grep else None

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
    except (FileNotFoundError, PermissionError) as e:
        print(c(f"  Error: {e}", "red"))
        sys.exit(1)

    entries = [parse_line(line, fmt) for line in raw_lines]
    filtered = [e for e in entries if passes_filter(e, level_min, grep, grep_re)]

    print(c(f"\n  File     : {path}", "bold"))
    print(f"  Lines    : {len(raw_lines):,}  →  filtered: {len(filtered):,}")
    if fmt != "auto":
        print(f"  Format   : {fmt}")

    if stats_only:
        print_stats(compute_stats(filtered, fmt), top_n)
        return

    if tail:
        display = filtered[-tail:]
        print(c(f"\n  Last {tail} matching lines:", "bold"))
    elif head:
        display = filtered[:head]
        print(c(f"\n  First {head} matching lines:", "bold"))
    else:
        display = filtered
        print(c(f"\n  {len(display)} matching lines:", "bold"))

    for entry in display:
        print(format_entry(entry))

    print_stats(compute_stats(filtered, fmt), top_n)


def interactive_mode():
    print(c("Log Parser Dashboard\n", "bold"))
    print("Commands: parse <file>, stats <file>, tail <file> [n], follow <file>, quit\n")

    while True:
        try:
            line = input(c("log> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower() if parts else ""

        if cmd in ("parse", "stats", "tail", "follow") and len(parts) > 1:
            fpath = parts[1]
            if not os.path.exists(fpath):
                print(c(f"  File not found: {fpath}", "red"))
                continue
            if cmd == "follow":
                follow_file(fpath, "auto", None, None)
            elif cmd == "tail":
                n = int(parts[2]) if len(parts) > 2 else 20
                process_file(fpath, tail=n)
            elif cmd == "stats":
                process_file(fpath, stats_only=True)
            else:
                level = input(c("  Min level (DEBUG/INFO/WARN/ERROR, Enter=all): ",
                                "cyan")).strip().upper() or None
                grep  = input(c("  Filter text (Enter=all): ", "cyan")).strip() or None
                process_file(fpath, level_min=level, grep=grep)
        elif cmd in ("quit", "exit", "q"):
            break
        else:
            print(c("  Unknown command.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="Parse and analyse log files")
    parser.add_argument("file",             nargs="?",     help="Log file path")
    parser.add_argument("--format",  "-f",  default="auto",
                        choices=["auto","apache","python","json","syslog","generic","plain"])
    parser.add_argument("--level",   "-l",  metavar="LEVEL", help="Minimum log level")
    parser.add_argument("--grep",    "-g",  metavar="TEXT",  help="Filter text/regex")
    parser.add_argument("--stats",          action="store_true", help="Show statistics only")
    parser.add_argument("--top",            type=int, default=10, help="Top N in stats")
    parser.add_argument("--tail",           type=int, default=0,  metavar="N",
                        help="Show last N matching lines")
    parser.add_argument("--head",           type=int, default=0,  metavar="N",
                        help="Show first N matching lines")
    parser.add_argument("--follow",         action="store_true",  help="Follow file (like tail -f)")
    args = parser.parse_args()

    if args.file:
        if args.follow:
            follow_file(args.file, args.format, args.level, args.grep)
        else:
            process_file(
                path=args.file,
                fmt=args.format,
                level_min=args.level,
                grep=args.grep,
                stats_only=args.stats,
                top_n=args.top,
                tail=args.tail,
                head=args.head,
            )
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
