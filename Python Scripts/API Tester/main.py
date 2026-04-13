"""API Tester — CLI developer tool.

Test REST API endpoints from the command line with support for
all HTTP methods, custom headers, JSON/form body, and response formatting.

Usage:
    python main.py
    python main.py GET https://api.example.com/users
    python main.py POST https://api.example.com/users -d '{"name":"Alice"}'
    python main.py --collection my_collection.json
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "blue": "\033[94m",
        "magenta": "\033[95m", "dim": "\033[2m", "reset": "\033[0m"}

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_history.json")


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


def status_color(code: int) -> str:
    if code < 300:   return "green"
    if code < 400:   return "yellow"
    if code < 500:   return "red"
    return "magenta"


def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list[dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-200:], f, indent=2)   # keep last 200


def make_request(method: str, url: str, headers: dict = None,
                 body: str = None, timeout: int = 10) -> dict:
    headers = headers or {}
    data    = body.encode("utf-8") if body else None

    if data and "Content-Type" not in headers:
        try:
            json.loads(body)
            headers["Content-Type"] = "application/json"
        except (ValueError, TypeError):
            headers["Content-Type"] = "application/x-www-form-urlencoded"

    req    = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    start  = time.time()
    result = {
        "method":    method.upper(), "url": url, "timestamp": datetime.now().isoformat(),
        "status":    None, "headers": {}, "body": "", "elapsed_ms": 0, "error": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result["status"]  = resp.status
            result["headers"] = dict(resp.headers)
            raw               = resp.read()
            result["body"]    = raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        result["status"]  = e.code
        result["headers"] = dict(e.headers)
        try:
            result["body"] = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        result["error"] = str(e)
    except urllib.error.URLError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)

    result["elapsed_ms"] = round((time.time() - start) * 1000, 1)
    return result


def print_response(result: dict, verbose: bool = False, json_only: bool = False):
    if result["error"] and not result["status"]:
        print(c(f"\n  ✗ Error: {result['error']}", "red"))
        return

    sc   = result["status"] or 0
    col  = status_color(sc)
    print(f"\n  {c(str(sc), col)} {c(f'{result[\"elapsed_ms\"]} ms', 'dim')}")

    if verbose:
        print(c("  Response Headers:", "dim"))
        for k, v in result["headers"].items():
            print(f"    {c(k,'cyan')}: {v}")

    body = result["body"]
    if not body:
        print(c("  (empty body)", "dim"))
        return

    try:
        parsed = json.loads(body)
        formatted = json.dumps(parsed, indent=2)
        print(c("\n  Body (JSON):", "bold"))
        for line in formatted.splitlines()[:100]:
            # Simple JSON colorizing
            line = line.replace('":', '":\033[0m')
            print(f"  {line}")
        if len(formatted.splitlines()) > 100:
            print(c(f"  ... ({len(formatted.splitlines())-100} more lines)", "dim"))
    except (ValueError, json.JSONDecodeError):
        if not json_only:
            print(c("\n  Body:", "bold"))
            for line in body.splitlines()[:50]:
                print(f"  {line}")
            if len(body.splitlines()) > 50:
                print(c(f"  ... ({len(body.splitlines())-50} more lines)", "dim"))


def parse_headers(header_list: list[str]) -> dict:
    headers = {}
    for h in (header_list or []):
        if ":" in h:
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers


def interactive_mode():
    print(c("API Tester", "bold") + "  —  test REST APIs from the CLI\n")
    print("Commands: GET/POST/PUT/DELETE/PATCH <url>, history, clear, quit")
    print("Examples: GET https://api.github.com/users/torvalds")
    print("          POST https://httpbin.org/post {\"key\":\"value\"}\n")

    history = load_history()

    while True:
        try:
            line = input(c("api> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd   = parts[0].upper()

        if cmd in ("QUIT", "EXIT", "Q"):
            break
        elif cmd == "HISTORY":
            n = min(int(parts[1]) if len(parts) > 1 else 10, len(history))
            for r in history[-n:]:
                print(f"  {c(r['method'], 'cyan')} {r['url']} → "
                      f"{c(str(r.get('status','?')), status_color(r.get('status',0)))} "
                      f"{r.get('elapsed_ms','?')}ms")
        elif cmd == "CLEAR":
            history = []
            save_history(history)
            print(c("  History cleared.", "green"))
        elif cmd in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
            if len(parts) < 2:
                print(c("  Provide a URL.", "yellow"))
                continue
            url    = parts[1]
            body   = " ".join(parts[2:]) if len(parts) > 2 else None
            result = make_request(cmd, url, body=body)
            print_response(result, verbose=False)
            history.append(result)
            save_history(history)
        else:
            # Try treating the whole line as a URL with GET
            if line.startswith("http"):
                result = make_request("GET", line)
                print_response(result)
                history.append(result)
                save_history(history)
            else:
                print(c("  Unknown command.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="CLI REST API tester")
    parser.add_argument("method",   nargs="?", help="HTTP method (GET, POST, etc.)")
    parser.add_argument("url",      nargs="?", help="Request URL")
    parser.add_argument("-d", "--data",    metavar="BODY",    help="Request body (JSON or form)")
    parser.add_argument("-H", "--header",  metavar="HEADER",  action="append",
                        help="Request header (e.g. 'Authorization: Bearer TOKEN')")
    parser.add_argument("-t", "--timeout", type=int,          default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.method and args.url:
        headers = parse_headers(args.header)
        result  = make_request(args.method, args.url, headers=headers,
                               body=args.data, timeout=args.timeout)
        print_response(result, verbose=args.verbose)
        history = load_history()
        history.append(result)
        save_history(history)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
