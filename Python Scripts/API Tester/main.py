"""API Tester - CLI developer tool.

Test REST API endpoints from the command line with support for
all HTTP methods, custom headers, JSON/form body, and response formatting.

Usage:
    python main.py
    python main.py GET https://api.example.com/users
    python main.py POST https://api.example.com/users -d '{"name":"Alice"}'
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "blue": "\033[94m",
        "magenta": "\033[95m", "dim": "\033[2m", "reset": "\033[0m"}

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_history.json")


def c(text: str, color: str) -> str:
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


def status_color(code: int) -> str:
    if code < 300:   return "green"
    if code < 400:   return "yellow"
    if code < 500:   return "red"
    return "magenta"


def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as history_file:
                history = json.load(history_file)
        except (OSError, json.JSONDecodeError):
            return []
        return history if isinstance(history, list) else []
    return []


def history_entry(result: dict) -> dict:
    """Return the non-sensitive request summary saved in local history."""
    return {
        key: result[key]
        for key in ("method", "url", "timestamp", "status", "elapsed_ms", "error")
    }


def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as history_file:
        json.dump(history[-200:], history_file, indent=2)


def add_to_history(result: dict) -> None:
    """Store a request summary without response headers or body content."""
    history = load_history()
    history.append(history_entry(result))
    save_history(history)


def make_request(method: str, url: str, headers: dict | None = None,
                 body: str | None = None, timeout: float = 10) -> dict:
    """Send an HTTP request and return a serializable response summary."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("URL must include an http:// or https:// scheme and host.")
    if timeout <= 0:
        raise ValueError("Timeout must be greater than zero.")
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
        "method":    method.upper(), "url": url, "timestamp": datetime.now(timezone.utc).isoformat(),
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
    except OSError as e:
        result["error"] = str(e)

    result["elapsed_ms"] = round((time.time() - start) * 1000, 1)
    return result


def print_response(result: dict, verbose: bool = False, json_only: bool = False) -> None:
    if result["error"] and not result["status"]:
        print(c(f"\n  Error: {result['error']}", "red"))
        return

    sc   = result["status"] or 0
    col  = status_color(sc)
    elapsed = f"{result['elapsed_ms']} ms"
    print(f"\n  {c(str(sc), col)} {c(elapsed, 'dim')}")

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


def parse_headers(header_list: list[str] | None) -> dict[str, str]:
    headers = {}
    for h in (header_list or []):
        if ":" not in h:
            raise ValueError(f"Invalid header: {h!r}. Use 'Name: value'.")
        key, value = h.split(":", 1)
        if not key.strip():
            raise ValueError(f"Invalid header: {h!r}. Header name is required.")
        headers[key.strip()] = value.strip()
    return headers


def interactive_mode() -> None:
    print(c("API Tester", "bold") + " - test REST APIs from the CLI\n")
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
                print(f"  {c(r['method'], 'cyan')} {r['url']} -> "
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
            try:
                result = make_request(cmd, url, body=body)
            except ValueError as error:
                print(c(f"  Error: {error}", "red"))
                continue
            print_response(result, verbose=False)
            history.append(result)
            save_history([history_entry(item) for item in history])
        else:
            # Try treating the whole line as a URL with GET
            if line.startswith("http"):
                try:
                    result = make_request("GET", line)
                except ValueError as error:
                    print(c(f"  Error: {error}", "red"))
                    continue
                print_response(result)
                history.append(result)
                save_history([history_entry(item) for item in history])
            else:
                print(c("  Unknown command.", "yellow"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI REST API tester")
    parser.add_argument("method",   nargs="?", help="HTTP method (GET, POST, etc.)")
    parser.add_argument("url",      nargs="?", help="Request URL")
    parser.add_argument("-d", "--data",    metavar="BODY",    help="Request body (JSON or form)")
    parser.add_argument("-H", "--header",  metavar="HEADER",  action="append",
                        help="Request header (e.g. 'Authorization: Bearer TOKEN')")
    parser.add_argument("-t", "--timeout", type=float,        default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-history", action="store_true", help="do not save a request summary")
    args = parser.parse_args()

    if args.method and args.url:
        try:
            headers = parse_headers(args.header)
            result = make_request(args.method, args.url, headers=headers,
                                  body=args.data, timeout=args.timeout)
        except ValueError as error:
            raise SystemExit(f"Error: {error}") from error
        print_response(result, verbose=args.verbose)
        if not args.no_history:
            add_to_history(result)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
