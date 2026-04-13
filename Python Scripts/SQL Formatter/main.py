"""SQL Formatter — CLI developer tool.

Format, beautify, and analyze SQL queries. Supports
pretty-printing, keyword case normalization, and basic linting.

Usage:
    python main.py
    python main.py --file query.sql
    python main.py --inline "SELECT * FROM users WHERE id=1"
    python main.py --file query.sql --output formatted.sql
"""

import argparse
import os
import re
import sys

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── SQL keyword lists ─────────────────────────────────────────────────────────

KEYWORDS_MAJOR = {
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
    "OUTER JOIN", "FULL JOIN", "CROSS JOIN", "ON", "GROUP BY", "ORDER BY",
    "HAVING", "LIMIT", "OFFSET", "UNION", "UNION ALL", "INTERSECT", "EXCEPT",
    "INSERT INTO", "VALUES", "UPDATE", "SET", "DELETE FROM", "CREATE TABLE",
    "ALTER TABLE", "DROP TABLE", "CREATE INDEX", "WITH", "AS",
}

KEYWORDS_MINOR = {
    "AND", "OR", "NOT", "IN", "NOT IN", "BETWEEN", "LIKE", "ILIKE",
    "IS NULL", "IS NOT NULL", "EXISTS", "DISTINCT", "ALL", "ANY", "SOME",
    "ASC", "DESC", "NULLS FIRST", "NULLS LAST", "CASE", "WHEN", "THEN",
    "ELSE", "END", "OVER", "PARTITION BY", "ROWS", "RANGE",
    "COUNT", "SUM", "AVG", "MAX", "MIN", "COALESCE", "NULLIF",
    "CAST", "CONVERT", "DATE", "NOW", "CURRENT_DATE", "CURRENT_TIMESTAMP",
}

ALL_KEYWORDS = KEYWORDS_MAJOR | KEYWORDS_MINOR

# Keywords that should start on a new line
NEWLINE_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
    "OUTER JOIN", "FULL JOIN", "CROSS JOIN", "ON", "GROUP BY", "ORDER BY",
    "HAVING", "LIMIT", "OFFSET", "UNION", "UNION ALL", "INTERSECT", "EXCEPT",
    "INSERT INTO", "VALUES", "UPDATE", "SET", "DELETE FROM", "WITH",
}

INDENT_AFTER = {"SELECT", "WHERE", "FROM", "GROUP BY", "ORDER BY", "HAVING"}


def normalize_whitespace(sql: str) -> str:
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*=\s*", " = ", sql)
    sql = re.sub(r"\s*\(\s*", " (", sql)
    sql = re.sub(r"\s*\)\s*", ") ", sql)
    return sql.strip()


def uppercase_keywords(sql: str) -> str:
    """Uppercase all SQL keywords."""
    def replace_keyword(m):
        word = m.group(0)
        if word.upper() in ALL_KEYWORDS:
            return word.upper()
        return word
    return re.sub(r"\b[a-zA-Z_]+\b", replace_keyword, sql)


def lowercase_identifiers(sql: str) -> str:
    """Lowercase non-keyword identifiers (best-effort)."""
    def maybe_lower(m):
        word = m.group(0)
        if word.upper() in ALL_KEYWORDS:
            return word.upper()
        return word.lower()
    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", maybe_lower, sql)


def format_sql(sql: str, indent: int = 4, keyword_case: str = "upper") -> str:
    sql = normalize_whitespace(sql)
    if keyword_case == "upper":
        sql = uppercase_keywords(sql)
    elif keyword_case == "lower":
        sql = sql.lower()
        for kw in sorted(ALL_KEYWORDS, key=len, reverse=True):
            sql = re.sub(r"\b" + kw.lower() + r"\b", kw.lower(), sql, flags=re.IGNORECASE)

    pad  = " " * indent
    lines = []
    current = sql

    # Split on major newline keywords
    for kw in sorted(NEWLINE_KEYWORDS, key=len, reverse=True):
        pattern = r"(?i)\b" + re.escape(kw) + r"\b"
        current = re.sub(pattern, f"\n{kw}", current)

    raw_lines = [l.strip() for l in current.splitlines()]
    current_indent = 0

    for line in raw_lines:
        if not line:
            continue
        kw_up = line.split()[0].upper() if line.split() else ""

        # Determine indentation
        if kw_up in NEWLINE_KEYWORDS:
            current_indent = 0

        prefix = pad * current_indent
        lines.append(f"{prefix}{line}")

        # Items after SELECT / WHERE / etc get +1 indent
        # (handle comma-separated lists)
        if kw_up in INDENT_AFTER:
            current_indent = 1
        elif kw_up in ("FROM", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER"):
            current_indent = 0

    result = "\n".join(lines)

    # Align commas in SELECT clause
    result = re.sub(r",\s*(?=\n)", ",", result)
    return result.strip() + "\n"


def lint_sql(sql: str) -> list[str]:
    issues = []
    sql_upper = sql.upper()
    if "SELECT *" in sql_upper:
        issues.append("⚠ Avoid SELECT * — specify columns explicitly.")
    if re.search(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", sql_upper):
        issues.append("⚠ Destructive operation detected (DROP).")
    if re.search(r"\bDELETE\b.*(?!\bWHERE\b)", sql_upper) and "WHERE" not in sql_upper:
        issues.append("⚠ DELETE without WHERE — will delete all rows!")
    if re.search(r"\bUPDATE\b.*(?!\bWHERE\b)", sql_upper) and "WHERE" not in sql_upper:
        issues.append("⚠ UPDATE without WHERE — will update all rows!")
    if sql_upper.count("(") != sql_upper.count(")"):
        issues.append("⚠ Unbalanced parentheses.")
    if re.search(r"'\s*OR\s*'\d+'\s*=\s*'\d+", sql_upper, re.IGNORECASE):
        issues.append("⚠ Possible SQL injection pattern detected.")
    if not issues:
        issues.append("✓ No obvious issues found.")
    return issues


def analyze_sql(sql: str) -> dict:
    sql_upper = sql.upper()
    tables = re.findall(r"\bFROM\s+(\w+)|JOIN\s+(\w+)", sql_upper)
    tables = list({t or j for t, j in tables if (t or j)})
    cols   = re.findall(r"SELECT\s+(.*?)\s+FROM", sql_upper, re.DOTALL)
    col_str = cols[0] if cols else ""
    return {
        "tables":     tables,
        "has_where":  "WHERE" in sql_upper,
        "has_group":  "GROUP BY" in sql_upper,
        "has_order":  "ORDER BY" in sql_upper,
        "has_join":   "JOIN" in sql_upper,
        "col_list":   col_str[:80] if col_str else "",
    }


def interactive_mode():
    print(c("SQL Formatter\n", "bold"))
    print("Commands: format, lint, analyze, file <path>, quit")
    print("Paste SQL then type END on a new line.\n")

    while True:
        try:
            line = input(c("sql> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower()

        if cmd == "file" and len(parts) > 1:
            try:
                with open(parts[1]) as f:
                    sql = f.read()
            except FileNotFoundError:
                print(c(f"  File not found: {parts[1]}", "red"))
                continue
        elif cmd in ("format", "lint", "analyze"):
            print("Paste SQL (type END to finish):")
            lines = []
            while True:
                try:
                    l = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if l.strip() == "END":
                    break
                lines.append(l)
            sql = "\n".join(lines)
            if not sql.strip():
                continue
        else:
            # Treat as direct SQL
            sql = line

        # Apply command
        if cmd in ("format", "file"):
            kc = input(c("  Keyword case (upper/lower/keep) [upper]: ", "cyan")).strip() or "upper"
            formatted = format_sql(sql, keyword_case=kc)
            print(c("\n─── Formatted SQL ───────────────────────", "dim"))
            print(formatted)
        elif cmd == "lint":
            issues = lint_sql(sql)
            for issue in issues:
                col = "red" if "⚠" in issue else "green"
                print(c(f"  {issue}", col))
        elif cmd == "analyze":
            info = analyze_sql(sql)
            for k, v in info.items():
                print(f"  {c(k,'cyan')}: {v}")
        else:
            # Just format
            print(c("\n─── Formatted SQL ───────────────────────", "dim"))
            print(format_sql(sql))


def main():
    parser = argparse.ArgumentParser(description="SQL formatter and linter")
    parser.add_argument("--file",    metavar="FILE",  help="Input SQL file")
    parser.add_argument("--inline",  metavar="SQL",   help="Inline SQL string")
    parser.add_argument("--output",  metavar="FILE",  help="Output file")
    parser.add_argument("--case",    default="upper", choices=["upper","lower","keep"])
    parser.add_argument("--lint",    action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    sql = None
    if args.file:
        with open(args.file) as f:
            sql = f.read()
    elif args.inline:
        sql = args.inline

    if sql:
        if args.lint:
            for issue in lint_sql(sql):
                print(issue)
        if args.analyze:
            for k, v in analyze_sql(sql).items():
                print(f"{k}: {v}")
        if not args.lint and not args.analyze:
            result = format_sql(sql, keyword_case=args.case)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(result)
                print(c(f"✓ Written to {args.output}", "green"))
            else:
                print(result)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
