"""ENV Validator — CLI developer tool.

Validate .env files against a schema, check for missing
required variables, detect type mismatches, and compare
environments.

Usage:
    python main.py
    python main.py .env
    python main.py .env --schema .env.schema
    python main.py .env.production --compare .env.staging
    python main.py --check-all
"""

import argparse
import os
import re
import sys

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m",
        "magenta": "\033[95m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_env(path: str) -> dict[str, str]:
    """Parse a .env file into a key→value dict."""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Remove surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            result[key] = val
    return result


def parse_schema(path: str) -> dict[str, dict]:
    """
    Parse a .env.schema file.
    Format per line:
      KEY=<type>[,required][,min=N][,max=N][,pattern=REGEX][,default=VAL]
    Types: string, int, float, bool, url, email, port
    """
    schema = {}
    if not os.path.exists(path):
        return schema
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, rest = line.partition("=")
            key  = key.strip()
            spec = {}
            for part in rest.split(","):
                part = part.strip()
                if "=" in part:
                    k2, _, v2 = part.partition("=")
                    spec[k2.strip()] = v2.strip()
                else:
                    spec[part] = True
            schema[key] = spec
    return schema


# ── Validators ─────────────────────────────────────────────────────────────────

URL_RE   = re.compile(r"^https?://[^\s]+$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_type(val: str, typ: str) -> str | None:
    """Return error string or None."""
    if typ == "int":
        try:
            int(val)
        except ValueError:
            return f"expected int, got '{val}'"
    elif typ == "float":
        try:
            float(val)
        except ValueError:
            return f"expected float, got '{val}'"
    elif typ == "bool":
        if val.lower() not in ("true", "false", "1", "0", "yes", "no"):
            return f"expected bool (true/false/1/0/yes/no), got '{val}'"
    elif typ == "url":
        if not URL_RE.match(val):
            return f"expected URL (http/https), got '{val}'"
    elif typ == "email":
        if not EMAIL_RE.match(val):
            return f"expected email address, got '{val}'"
    elif typ == "port":
        try:
            p = int(val)
            if not 1 <= p <= 65535:
                raise ValueError
        except ValueError:
            return f"expected port (1-65535), got '{val}'"
    return None


def validate_constraints(val: str, spec: dict) -> list[str]:
    errors = []
    if "min" in spec:
        try:
            if float(val) < float(spec["min"]):
                errors.append(f"value {val} < min {spec['min']}")
        except (ValueError, TypeError):
            if len(val) < int(spec["min"]):
                errors.append(f"length {len(val)} < min {spec['min']}")
    if "max" in spec:
        try:
            if float(val) > float(spec["max"]):
                errors.append(f"value {val} > max {spec['max']}")
        except (ValueError, TypeError):
            if len(val) > int(spec["max"]):
                errors.append(f"length {len(val)} > max {spec['max']}")
    if "pattern" in spec:
        try:
            if not re.search(spec["pattern"], val):
                errors.append(f"does not match pattern '{spec['pattern']}'")
        except re.error:
            errors.append(f"invalid pattern in schema '{spec['pattern']}'")
    return errors


def check_sensitive(key: str, val: str) -> str | None:
    """Warn if a sensitive key appears to have a weak/default value."""
    sensitive_kw = ("password", "secret", "token", "key", "api_key", "passwd")
    key_lower = key.lower()
    if not any(kw in key_lower for kw in sensitive_kw):
        return None
    if val in ("", "password", "secret", "changeme", "example", "test",
               "your_api_key", "your-secret-key", "xxx", "TODO"):
        return "looks like a placeholder or weak value"
    if len(val) < 8:
        return f"short secret (only {len(val)} chars)"
    return None


# ── Core validation ────────────────────────────────────────────────────────────

class Issue:
    def __init__(self, key, severity, message):
        self.key      = key
        self.severity = severity   # "error" | "warning" | "info"
        self.message  = message


def validate(env: dict, schema: dict) -> list[Issue]:
    issues = []

    # Schema-based checks
    for key, spec in schema.items():
        required = spec.get("required") is True or spec.get("required") == "true"
        if key not in env:
            if required:
                issues.append(Issue(key, "error", "required variable is missing"))
            elif "default" not in spec:
                issues.append(Issue(key, "warning", "optional variable not set"))
            continue

        val = env[key]

        # Type check
        typ = spec.get("string", None)
        for t in ("int", "float", "bool", "url", "email", "port", "string"):
            if t in spec:
                err = validate_type(val, t)
                if err:
                    issues.append(Issue(key, "error", err))
                break

        # Constraints
        for err in validate_constraints(val, spec):
            issues.append(Issue(key, "error", err))

    # Sensitivity checks on all env vars
    for key, val in env.items():
        warn = check_sensitive(key, val)
        if warn:
            issues.append(Issue(key, "warning", f"sensitive key: {warn}"))

    # Detect empty values
    for key, val in env.items():
        if val == "" and key not in schema:
            issues.append(Issue(key, "warning", "value is empty"))

    # Detect duplicate-ish keys (case insensitive)
    keys_lower = {}
    for key in env:
        lower = key.lower()
        if lower in keys_lower:
            issues.append(Issue(key, "warning",
                f"similar key to '{keys_lower[lower]}' (case difference)"))
        keys_lower[lower] = key

    return issues


def compare_envs(env_a: dict, path_a: str, env_b: dict, path_b: str):
    """Show keys present in one but not the other."""
    only_a = set(env_a) - set(env_b)
    only_b = set(env_b) - set(env_a)
    common = set(env_a) & set(env_b)
    differ = {k for k in common if env_a[k] != env_b[k]}

    print(c(f"\n  Comparing {path_a}  ↔  {path_b}\n", "bold"))
    print(f"  {c('Common keys', 'dim')} : {len(common)}")
    print(f"  {c('Differing values', 'dim')} : {len(differ)}")

    if only_a:
        print(c(f"\n  Only in {path_a}:", "yellow"))
        for k in sorted(only_a):
            print(f"    {c(k,'cyan')}")
    if only_b:
        print(c(f"\n  Only in {path_b}:", "yellow"))
        for k in sorted(only_b):
            print(f"    {c(k,'cyan')}")
    if differ:
        print(c("\n  Different values:", "yellow"))
        for k in sorted(differ):
            av = env_a[k][:40] + ("…" if len(env_a[k]) > 40 else "")
            bv = env_b[k][:40] + ("…" if len(env_b[k]) > 40 else "")
            print(f"    {c(k,'cyan'):32} {path_a}: {av}")
            print(f"    {'':32} {path_b}: {bv}")
    if not only_a and not only_b and not differ:
        print(c("\n  ✓ Environments are identical.", "green"))


def report(path: str, issues: list[Issue], env: dict, schema: dict):
    errors   = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos    = [i for i in issues if i.severity == "info"]

    print(c(f"\n  Validation: {path}\n", "bold"))
    print(f"  Variables : {len(env)}")
    print(f"  Schema    : {len(schema)} rules")
    print(f"  Errors    : {c(str(len(errors)), 'red' if errors else 'green')}")
    print(f"  Warnings  : {c(str(len(warnings)), 'yellow' if warnings else 'dim')}")

    if issues:
        print()
        for i in sorted(issues, key=lambda x: ("error","warning","info").index(x.severity)):
            if i.severity == "error":
                sym = c("✗", "red");   col = "red"
            elif i.severity == "warning":
                sym = c("⚠", "yellow"); col = "yellow"
            else:
                sym = c("ℹ", "cyan");   col = "cyan"
            print(f"  {sym}  {c(i.key, col)}: {i.message}")
    else:
        print(c("\n  ✓ All checks passed.", "green"))


def interactive_mode():
    print(c("ENV Validator\n", "bold"))
    print("Commands: validate <file>, compare <a> <b>, schema <file>, quit\n")

    while True:
        try:
            line = input(c("env> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower() if parts else ""

        if cmd == "validate" and len(parts) > 1:
            env_path    = parts[1]
            schema_path = parts[2] if len(parts) > 2 else env_path + ".schema"
            env    = parse_env(env_path)
            schema = parse_schema(schema_path)
            if not env and not os.path.exists(env_path):
                print(c(f"  File not found: {env_path}", "red"))
                continue
            issues = validate(env, schema)
            report(env_path, issues, env, schema)

        elif cmd == "compare" and len(parts) > 2:
            env_a = parse_env(parts[1])
            env_b = parse_env(parts[2])
            compare_envs(env_a, parts[1], env_b, parts[2])

        elif cmd == "schema" and len(parts) > 1:
            schema = parse_schema(parts[1])
            if not schema and not os.path.exists(parts[1]):
                print(c(f"  File not found: {parts[1]}", "red"))
                continue
            print(c(f"\n  Schema: {parts[1]}\n", "bold"))
            for k, spec in schema.items():
                flags = ", ".join(f"{k2}={v2}" if v2 is not True else k2
                                  for k2, v2 in spec.items())
                print(f"  {c(k,'cyan')}: {flags or '(no constraints)'}")

        elif cmd in ("quit", "exit", "q"):
            break
        else:
            print(c("  Unknown command.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="Validate .env files")
    parser.add_argument("env_file",         nargs="?",       help=".env file path")
    parser.add_argument("--schema", "-s",   metavar="FILE",  help="Schema file path")
    parser.add_argument("--compare", "-c",  metavar="FILE",  help="Compare with another .env file")
    parser.add_argument("--check-all",      action="store_true", dest="check_all",
                        help="Find and validate all .env* files in current directory")
    args = parser.parse_args()

    if args.check_all:
        files = [f for f in os.listdir(".") if f.startswith(".env") and os.path.isfile(f)]
        if not files:
            print(c("No .env files found in current directory.", "yellow"))
            return
        for env_file in sorted(files):
            schema_path = env_file + ".schema"
            env    = parse_env(env_file)
            schema = parse_schema(schema_path)
            issues = validate(env, schema)
            report(env_file, issues, env, schema)
        return

    if args.env_file:
        env    = parse_env(args.env_file)
        if not env and not os.path.exists(args.env_file):
            print(c(f"File not found: {args.env_file}", "red"))
            sys.exit(1)

        if args.compare:
            env_b = parse_env(args.compare)
            compare_envs(env, args.env_file, env_b, args.compare)
            return

        schema_path = args.schema or (args.env_file + ".schema")
        schema      = parse_schema(schema_path)
        issues      = validate(env, schema)
        report(args.env_file, issues, env, schema)

        if any(i.severity == "error" for i in issues):
            sys.exit(1)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
