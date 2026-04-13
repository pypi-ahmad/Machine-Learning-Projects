"""URL Parser — CLI tool.

Parses URLs into components, builds URLs from parts, encodes/decodes
query strings, and extracts information such as domain, TLD, and path
segments.

Usage:
    python main.py
"""

import json
from urllib.parse import (
    ParseResult,
    parse_qs,
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def parse_url(url: str) -> dict:
    """Break a URL into labelled components."""
    parsed: ParseResult = urlparse(url)
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    query_list   = parse_qsl(parsed.query, keep_blank_values=True)
    path_segments = [s for s in parsed.path.split("/") if s]

    # Extract TLD (simple heuristic)
    host = parsed.hostname or ""
    parts = host.split(".")
    tld    = parts[-1]   if len(parts) >= 1 else ""
    domain = parts[-2]   if len(parts) >= 2 else host
    sub    = ".".join(parts[:-2]) if len(parts) >= 3 else ""

    return {
        "original":       url,
        "scheme":         parsed.scheme,
        "netloc":         parsed.netloc,
        "host":           host,
        "subdomain":      sub,
        "domain":         domain,
        "tld":            tld,
        "port":           parsed.port,
        "path":           parsed.path,
        "path_segments":  path_segments,
        "query_string":   parsed.query,
        "query_params":   query_params,
        "query_list":     query_list,
        "fragment":       parsed.fragment,
        "username":       parsed.username,
        "password":       parsed.password,
    }


def build_url(scheme: str, host: str, path: str = "",
              params: dict | None = None, fragment: str = "") -> str:
    """Construct a URL from components."""
    query = urlencode(params or {}, doseq=True)
    return urlunparse((scheme, host, path, "", query, fragment))


def encode_query(params: dict) -> str:
    return urlencode(params, doseq=True)


def decode_query(query_string: str) -> dict:
    return parse_qs(query_string, keep_blank_values=True)


def url_encode(text: str, safe: str = "") -> str:
    return quote(text, safe=safe)


def url_decode(text: str) -> str:
    return unquote(text)


def resolve_url(base: str, relative: str) -> str:
    return urljoin(base, relative)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_parsed(info: dict) -> None:
    WIDTH = 20
    print()
    for key, val in info.items():
        if key in ("query_params", "query_list"):
            continue  # shown separately
        if val is None or val == "" or val == []:
            continue
        label = key.replace("_", " ").title()
        if isinstance(val, list):
            print(f"  {label:<{WIDTH}}: {' / '.join(str(v) for v in val)}")
        else:
            print(f"  {label:<{WIDTH}}: {val}")

    if info["query_list"]:
        print(f"\n  {'Query Parameters':<{WIDTH}}:")
        for k, v in info["query_list"]:
            print(f"    {k} = {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
URL Parser
----------
1. Parse URL
2. Build URL from parts
3. Encode query string
4. Decode query string
5. URL-encode / decode text
6. Resolve relative URL
0. Quit
"""


def main() -> None:
    print("URL Parser")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            url = input("  URL: ").strip()
            if not url:
                continue
            if "://" not in url:
                url = "https://" + url
            info = parse_url(url)
            print_parsed(info)

        elif choice == "2":
            scheme   = input("  Scheme (default https): ").strip() or "https"
            host     = input("  Host (e.g. example.com): ").strip()
            path     = input("  Path (e.g. /api/v1/users): ").strip()
            qs_str   = input("  Query params as key=val&... (or blank): ").strip()
            fragment = input("  Fragment (or blank): ").strip()
            params   = dict(parse_qsl(qs_str)) if qs_str else {}
            result   = build_url(scheme, host, path, params, fragment)
            print(f"\n  URL: {result}")

        elif choice == "3":
            print("  Enter key=value pairs (one per line, blank to finish):")
            params: dict[str, list[str]] = {}
            while True:
                line = input("  > ").strip()
                if not line:
                    break
                if "=" in line:
                    k, _, v = line.partition("=")
                    params.setdefault(k.strip(), []).append(v.strip())
            encoded = encode_query(params)
            print(f"\n  Encoded: {encoded}")

        elif choice == "4":
            qs = input("  Query string: ").strip()
            decoded = decode_query(qs)
            print("\n  Decoded parameters:")
            for k, vals in decoded.items():
                for v in vals:
                    print(f"    {k} = {v}")

        elif choice == "5":
            sub = input("  (e)ncode or (d)ecode? ").strip().lower()
            text = input("  Text: ")
            if sub.startswith("e"):
                safe = input("  Safe characters (default none): ").strip()
                print(f"\n  Encoded: {url_encode(text, safe)}")
            else:
                print(f"\n  Decoded: {url_decode(text)}")

        elif choice == "6":
            base     = input("  Base URL: ").strip()
            relative = input("  Relative URL: ").strip()
            print(f"\n  Resolved: {resolve_url(base, relative)}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
