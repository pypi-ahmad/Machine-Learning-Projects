"""Cron Helper — CLI developer tool.

Parse, validate, describe, and generate cron expressions.
Show next N run times for any cron schedule.

Usage:
    python main.py
    python main.py "*/5 * * * *"
    python main.py "0 9 * * 1-5" --next 10
    python main.py --generate
    python main.py --validate "0 25 * * *"
"""

import argparse
import re
from datetime import datetime, timedelta

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m",
        "magenta": "\033[95m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Field definitions ──────────────────────────────────────────────────────────

FIELDS = [
    {"name": "minute",     "range": (0, 59),  "names": None},
    {"name": "hour",       "range": (0, 23),  "names": None},
    {"name": "day",        "range": (1, 31),  "names": None},
    {"name": "month",      "range": (1, 12),  "names":
        ["", "JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]},
    {"name": "weekday",    "range": (0, 6),   "names":
        ["SUN","MON","TUE","WED","THU","FRI","SAT"]},
]

MONTH_NAMES   = {m: i for i, m in enumerate(
    ["", "JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]) if m}
WEEKDAY_NAMES = {d: i for i, d in enumerate(["SUN","MON","TUE","WED","THU","FRI","SAT"])}

PRESETS = {
    "@yearly":   "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly":  "0 0 1 * *",
    "@weekly":   "0 0 * * 0",
    "@daily":    "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly":   "0 * * * *",
}


# ── Parser ─────────────────────────────────────────────────────────────────────

def expand_names(token: str, names: list | None) -> str:
    """Replace month/weekday names with numbers."""
    if not names:
        return token
    for name, num in (MONTH_NAMES if len(names) == 13 else WEEKDAY_NAMES).items():
        token = re.sub(r"\b" + name + r"\b", str(num), token, flags=re.IGNORECASE)
    return token


def parse_field(token: str, lo: int, hi: int, names: list | None = None) -> list[int] | str:
    """Return sorted list of matching values or error string."""
    token = expand_names(token.strip(), names)

    if token == "*":
        return list(range(lo, hi + 1))

    values = set()
    for part in token.split(","):
        # Step: */n or a-b/n
        m = re.match(r"^(\*|\d+(?:-\d+)?)/(\d+)$", part)
        if m:
            start_s, step_s = m.group(1), m.group(2)
            step = int(step_s)
            if step == 0:
                return "step cannot be 0"
            if start_s == "*":
                start, end = lo, hi
            else:
                ab = start_s.split("-")
                start = int(ab[0])
                end   = int(ab[1]) if len(ab) == 2 else hi
            values.update(range(start, end + 1, step))
            continue
        # Range: a-b
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                return f"invalid range {a}-{b}"
            values.update(range(a, b + 1))
            continue
        # Single value
        if re.match(r"^\d+$", part):
            values.add(int(part))
            continue
        # L (last)
        if part.upper() == "L":
            values.add(hi)
            continue
        return f"unrecognized token '{part}'"

    out_of_range = [v for v in values if not (lo <= v <= hi)]
    if out_of_range:
        return f"value(s) {out_of_range} out of range {lo}-{hi}"

    return sorted(values)


def parse_cron(expr: str) -> tuple[list[list[int]] | None, list[str]]:
    """Parse cron expression into 5 field value lists. Returns (fields, errors)."""
    expr = expr.strip()
    if expr in PRESETS:
        expr = PRESETS[expr]

    parts = expr.split()
    if len(parts) != 5:
        return None, [f"expected 5 fields, got {len(parts)}"]

    errors = []
    fields = []
    for i, (part, field) in enumerate(zip(parts, FIELDS)):
        result = parse_field(part, field["range"][0], field["range"][1], field["names"])
        if isinstance(result, str):
            errors.append(f"field '{field['name']}': {result}")
            fields.append([])
        else:
            fields.append(result)

    return fields, errors


# ── Human description ──────────────────────────────────────────────────────────

def describe_field(values: list[int], field: dict, original: str = "") -> str:
    lo, hi = field["range"]
    if values == list(range(lo, hi + 1)):
        return "every " + field["name"]
    if len(values) == 1:
        v = values[0]
        if field["name"] == "month" and field["names"]:
            return field["names"][v]
        if field["name"] == "weekday" and field["names"]:
            return field["names"][v % len(field["names"])]
        return str(v)
    if len(values) > 8:
        step = values[1] - values[0] if len(values) > 1 else 1
        if all(values[i] - values[i-1] == step for i in range(1, len(values))):
            return f"every {step} {field['name']}s"
    return ", ".join(str(v) for v in values[:6]) + ("…" if len(values) > 6 else "")


def describe(expr: str) -> str:
    expr = expr.strip()
    if expr in PRESETS:
        expr = PRESETS[expr]

    fields, errors = parse_cron(expr)
    if errors:
        return "invalid: " + "; ".join(errors)

    mins, hrs, days, months, wdays = fields

    minute  = describe_field(mins,   FIELDS[0])
    hour    = describe_field(hrs,    FIELDS[1])
    day     = describe_field(days,   FIELDS[2])
    month   = describe_field(months, FIELDS[3])
    weekday = describe_field(wdays,  FIELDS[4])

    parts = []
    if minute   != "every minute":  parts.append(f"at minute {minute}")
    if hour     != "every hour":    parts.append(f"hour {hour}")
    if day      != "every day":     parts.append(f"on day {day}")
    if month    != "every month":   parts.append(f"in {month}")
    if weekday  != "every weekday": parts.append(f"on {weekday}")

    if not parts:
        return "every minute"
    return "Runs " + ", ".join(parts)


# ── Next run calculator ────────────────────────────────────────────────────────

def next_runs(expr: str, n: int = 5, start: datetime = None) -> list[datetime] | str:
    fields, errors = parse_cron(expr)
    if errors:
        return "invalid: " + "; ".join(errors)

    mins, hrs, days, months, wdays = fields
    results = []
    dt = (start or datetime.now()).replace(second=0, microsecond=0) + timedelta(minutes=1)
    max_iter = 60 * 24 * 366 * 4   # ~4 years of minutes

    for _ in range(max_iter):
        if (dt.month in months and dt.day in days and
                dt.weekday() in [w % 7 for w in wdays] and   # Python: Mon=0, cron: Sun=0
                dt.hour in hrs and dt.minute in mins):
            # Convert cron weekday to Python weekday (cron Sun=0/7 → Python Sun=6)
            cron_wd = (dt.weekday() + 1) % 7   # Python Mon=0 → cron Mon=1
            if cron_wd in wdays:
                results.append(dt)
                if len(results) == n:
                    break
        dt += timedelta(minutes=1)

    return results


def _next_runs_corrected(expr: str, n: int = 5, start: datetime = None) -> list[datetime] | str:
    """Corrected next-run calculator with proper weekday mapping."""
    fields, errors = parse_cron(expr)
    if errors:
        return "invalid: " + "; ".join(errors)

    mins, hrs, days, months, wdays = fields
    # Convert cron weekday (0=Sun) to Python weekday (0=Mon)
    py_wdays = set((w - 1) % 7 for w in wdays)

    results = []
    dt = (start or datetime.now()).replace(second=0, microsecond=0) + timedelta(minutes=1)
    max_iter = 60 * 24 * 366 * 4

    for _ in range(max_iter):
        if (dt.month  in months  and
            dt.day    in days    and
            dt.weekday() in py_wdays and
            dt.hour   in hrs     and
            dt.minute in mins):
            results.append(dt)
            if len(results) == n:
                break
        dt += timedelta(minutes=1)

    return results


# ── Generator wizard ───────────────────────────────────────────────────────────

def generate_wizard() -> str:
    print(c("\n  Cron Expression Generator\n", "bold"))

    presets = [
        ("Every minute",           "* * * * *"),
        ("Every hour",             "0 * * * *"),
        ("Every day at midnight",  "0 0 * * *"),
        ("Every Monday 9 AM",      "0 9 * * 1"),
        ("Every weekday 9 AM",     "0 9 * * 1-5"),
        ("Every 15 minutes",       "*/15 * * * *"),
        ("First of every month",   "0 0 1 * *"),
        ("Custom",                 None),
    ]

    print("  Presets:")
    for i, (name, _) in enumerate(presets, 1):
        print(f"    {c(str(i),'cyan')} {name}")
    choice = input(c("\n  Choose preset [8]: ", "cyan")).strip() or "8"

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(presets) - 1:
            return presets[idx][1]
    except ValueError:
        pass

    print("\n  Build custom expression:")
    fields_input = []
    for field in FIELDS:
        lo, hi = field["range"]
        val = input(c(f"  {field['name']:10} ({lo}-{hi}, * for all): ", "cyan")).strip() or "*"
        fields_input.append(val)

    return " ".join(fields_input)


# ── Main ───────────────────────────────────────────────────────────────────────

def print_expression_info(expr: str, n: int):
    expr = expr.strip()
    canonical = PRESETS.get(expr, expr)

    _, errors = parse_cron(canonical)

    print(c(f"\n  Expression : {expr}", "bold"))
    if canonical != expr:
        print(c(f"  Expanded   : {canonical}", "dim"))

    if errors:
        print(c(f"  Errors     : {'; '.join(errors)}", "red"))
        return

    print(c(f"  Valid      : ✓", "green"))
    print(f"  Description: {describe(canonical)}")

    runs = _next_runs_corrected(canonical, n=n)
    if isinstance(runs, str):
        print(c(f"  Next runs  : {runs}", "red"))
    elif not runs:
        print(c("  Next runs  : (none found in next 4 years)", "yellow"))
    else:
        print(c(f"\n  Next {n} runs:", "bold"))
        for i, dt in enumerate(runs, 1):
            delta = dt - datetime.now()
            h, m  = divmod(int(delta.total_seconds() / 60), 60)
            d, h  = divmod(h, 24)
            eta   = (f"{d}d " if d else "") + (f"{h}h " if h else "") + f"{m}m"
            print(f"    {c(str(i),'dim')}  {c(dt.strftime('%Y-%m-%d %H:%M'),'cyan')}  "
                  f"({c('in '+eta,'dim')})")


def interactive_mode():
    print(c("Cron Helper\n", "bold"))
    print("Commands: describe <expr>, next <expr> [n], validate <expr>, generate, quit")
    print(f"Presets:  {', '.join(PRESETS.keys())}\n")

    while True:
        try:
            line = input(c("cron> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split(None, 1)
        cmd   = parts[0].lower() if parts else ""
        rest  = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "describe":
            print(f"  {describe(rest)}")

        elif cmd == "next":
            sub = rest.rsplit(None, 1)
            try:
                n   = int(sub[-1])
                expr = sub[0]
            except (ValueError, IndexError):
                expr = rest
                n    = 5
            print_expression_info(expr, n)

        elif cmd == "validate":
            _, errors = parse_cron(rest)
            if errors:
                for e in errors:
                    print(c(f"  ✗ {e}", "red"))
            else:
                print(c("  ✓ Valid cron expression.", "green"))
                print(f"  {describe(rest)}")

        elif cmd == "generate":
            expr = generate_wizard()
            print(c(f"\n  Generated: {expr}", "green"))
            print_expression_info(expr, 5)

        elif cmd in ("quit", "exit", "q"):
            break
        else:
            # Treat whole line as expression
            if line:
                print_expression_info(line, 5)
            else:
                print(c("  Unknown command.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="Cron expression helper")
    parser.add_argument("expression",    nargs="?",     help="Cron expression to analyse")
    parser.add_argument("--next", "-n",  type=int, default=5, metavar="N",
                        help="Show next N run times (default: 5)")
    parser.add_argument("--validate",    action="store_true",  help="Validate only")
    parser.add_argument("--generate",    action="store_true",  help="Launch wizard")
    parser.add_argument("--describe",    action="store_true",  help="Print human description")
    args = parser.parse_args()

    if args.generate:
        expr = generate_wizard()
        print(c(f"\n  Generated: {expr}", "green"))
        print_expression_info(expr, args.next)
        return

    if args.expression:
        expr = args.expression
        if args.validate:
            _, errors = parse_cron(expr)
            if errors:
                for e in errors:
                    print(c(f"✗ {e}", "red"))
                import sys; sys.exit(1)
            else:
                print(c("✓ Valid", "green"))
                print(describe(expr))
        elif args.describe:
            print(describe(expr))
        else:
            print_expression_info(expr, args.next)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
