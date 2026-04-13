"""Local HTTP Server — CLI developer tool.

Serve a local directory over HTTP with directory listing,
CORS headers, custom port, and basic authentication support.

Usage:
    python main.py
    python main.py --port 3000
    python main.py --dir ./dist
    python main.py --port 8080 --cors --no-cache
    python main.py --auth user:pass
"""

import argparse
import base64
import html
import io
import json
import mimetypes
import os
import sys
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


def human_size(n: int) -> str:
    if n < 1024:     return f"{n} B"
    if n < 1024**2:  return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"


# ── Request handler ────────────────────────────────────────────────────────────

class DevServerHandler(BaseHTTPRequestHandler):
    root_dir   = "."
    enable_cors = False
    no_cache   = False
    auth_cred  = None   # base64-encoded "user:pass"
    request_log: list = []

    def _check_auth(self) -> bool:
        if not self.auth_cred:
            return True
        ah = self.headers.get("Authorization", "")
        if not ah.startswith("Basic "):
            return False
        return ah[6:].strip() == self.auth_cred

    def _send_auth_challenge(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Local Dev Server"')
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Unauthorized")

    def _common_headers(self, content_type: str, size: int = None):
        self.send_header("Content-Type", content_type)
        if size is not None:
            self.send_header("Content-Length", str(size))
        if self.enable_cors:
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
        if self.no_cache:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")

    def do_OPTIONS(self):
        self.send_response(204)
        if self.enable_cors:
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if not self._check_auth():
            self._send_auth_challenge()
            return
        self._serve()

    def do_HEAD(self):
        if not self._check_auth():
            self._send_auth_challenge()
            return
        self._serve(head_only=True)

    def _serve(self, head_only: bool = False):
        path = urllib.parse.unquote(self.path.split("?")[0])
        fs_path = os.path.normpath(os.path.join(self.root_dir, path.lstrip("/")))

        # Security: prevent path traversal
        if not fs_path.startswith(os.path.normpath(self.root_dir)):
            self._send_error(403, "Forbidden")
            return

        if os.path.isdir(fs_path):
            # Check for index file
            for idx in ("index.html", "index.htm"):
                idx_path = os.path.join(fs_path, idx)
                if os.path.exists(idx_path):
                    fs_path = idx_path
                    break
            else:
                self._serve_listing(fs_path, path, head_only)
                return

        if not os.path.isfile(fs_path):
            self._send_error(404, "Not Found")
            return

        mime, _ = mimetypes.guess_type(fs_path)
        mime     = mime or "application/octet-stream"

        try:
            with open(fs_path, "rb") as f:
                data = f.read()
        except OSError:
            self._send_error(500, "Internal Server Error")
            return

        self.send_response(200)
        self._common_headers(mime, len(data))
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def _serve_listing(self, fs_path: str, url_path: str, head_only: bool):
        try:
            entries = sorted(os.scandir(fs_path), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            self._send_error(403, "Forbidden")
            return

        rows = []
        if url_path != "/":
            rows.append('<tr><td><a href="../">../</a></td><td>—</td><td>—</td></tr>')
        for entry in entries:
            name  = entry.name + ("/" if entry.is_dir() else "")
            href  = urllib.parse.quote(entry.name) + ("/" if entry.is_dir() else "")
            try:
                stat = entry.stat()
                size = human_size(stat.st_size) if entry.is_file() else "—"
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                size = mtime = "—"
            rows.append(f'<tr><td><a href="{href}">{html.escape(name)}</a></td>'
                        f'<td>{mtime}</td><td>{size}</td></tr>')

        body = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Index of {html.escape(url_path)}</title>
<style>
body{{font-family:monospace;padding:1rem;background:#111;color:#ccc}}
a{{color:#7ec8e3;text-decoration:none}}
a:hover{{text-decoration:underline}}
table{{border-collapse:collapse;width:100%}}
th,td{{padding:.3rem .8rem;text-align:left}}
th{{border-bottom:1px solid #444;color:#888}}
tr:hover{{background:#1a1a2e}}
</style></head>
<body>
<h2>Index of {html.escape(url_path)}</h2>
<table><thead><tr><th>Name</th><th>Modified</th><th>Size</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<hr><p style="color:#555">Local HTTP Server</p>
</body></html>"""
        data = body.encode("utf-8")
        self.send_response(200)
        self._common_headers("text/html; charset=utf-8", len(data))
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def _send_error(self, code: int, msg: str):
        body = f"<h1>{code} {msg}</h1>".encode("utf-8")
        self.send_response(code)
        self._common_headers("text/html", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        code   = args[1] if len(args) > 1 else "???"
        method = self.command or "?"
        path   = self.path or "/"
        ts     = time.strftime("%H:%M:%S")

        # Color by status
        try:
            sc = int(code)
            if sc < 300:    col = "green"
            elif sc < 400:  col = "yellow"
            elif sc < 500:  col = "red"
            else:           col = "magenta"
        except ValueError:
            col = "dim"

        entry = {"time": ts, "method": method, "path": path, "status": code}
        DevServerHandler.request_log.append(entry)

        print(f"  {c(ts,'dim')}  {c(method,'cyan'):12}  {c(code, col)}  {path}")


# ── Stats endpoint ─────────────────────────────────────────────────────────────

class StatsHandler(DevServerHandler):
    def _serve(self, head_only: bool = False):
        if self.path == "/__stats__":
            data = json.dumps({
                "requests": len(DevServerHandler.request_log),
                "log": DevServerHandler.request_log[-20:],
            }, indent=2).encode("utf-8")
            self.send_response(200)
            self._common_headers("application/json", len(data))
            self.end_headers()
            if not head_only:
                self.wfile.write(data)
        else:
            super()._serve(head_only)


# ── Server launcher ────────────────────────────────────────────────────────────

def run_server(directory: str, port: int, cors: bool, no_cache: bool,
               auth: str = None, stats: bool = False):
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        print(c(f"Directory not found: {directory}", "red"))
        sys.exit(1)

    handler = StatsHandler if stats else DevServerHandler
    handler.root_dir    = directory
    handler.enable_cors = cors
    handler.no_cache    = no_cache

    if auth:
        cred = base64.b64encode(auth.encode()).decode()
        handler.auth_cred = cred
    else:
        handler.auth_cred = None

    server = HTTPServer(("", port), handler)

    print(c("\n  Local HTTP Server\n", "bold"))
    print(f"  Serving  : {c(directory, 'cyan')}")
    print(f"  URL      : {c(f'http://localhost:{port}', 'green')}")
    print(f"  CORS     : {'enabled' if cors else 'disabled'}")
    print(f"  Cache    : {'disabled' if no_cache else 'enabled'}")
    if auth:
        user = auth.split(":")[0]
        print(f"  Auth     : {c(f'Basic ({user})', 'yellow')}")
    if stats:
        print(f"  Stats    : http://localhost:{port}/__stats__")
    print(c("\n  Press Ctrl+C to stop.\n", "dim"))
    print(f"  {'Time':8}  {'Method':12}  {'Status'}  Path")
    print(c("  " + "─" * 60, "dim"))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(c("\n\n  Server stopped.", "dim"))


def interactive_mode():
    print(c("Local HTTP Server\n", "bold"))
    directory = input(c("  Directory to serve [.]: ", "cyan")).strip() or "."
    port_str  = input(c("  Port [8000]: ", "cyan")).strip() or "8000"
    cors_in   = input(c("  Enable CORS? [y/N]: ", "cyan")).strip().lower()
    cache_in  = input(c("  Disable cache? [y/N]: ", "cyan")).strip().lower()
    auth_in   = input(c("  Basic auth (user:pass, or Enter to skip): ", "cyan")).strip()

    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    run_server(
        directory=directory,
        port=port,
        cors=cors_in == "y",
        no_cache=cache_in == "y",
        auth=auth_in or None,
    )


def main():
    parser = argparse.ArgumentParser(description="Local HTTP development server")
    parser.add_argument("dir",         nargs="?", default=".",    help="Directory to serve")
    parser.add_argument("--port","-p", type=int,  default=8000,   help="Port number")
    parser.add_argument("--cors",      action="store_true",       help="Enable CORS headers")
    parser.add_argument("--no-cache",  action="store_true", dest="no_cache",
                        help="Disable caching")
    parser.add_argument("--auth",      metavar="USER:PASS",       help="Enable basic auth")
    parser.add_argument("--stats",     action="store_true",       help="Enable /__stats__ endpoint")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        interactive_mode()
    else:
        run_server(
            directory=args.dir,
            port=args.port,
            cors=args.cors,
            no_cache=args.no_cache,
            auth=args.auth,
            stats=args.stats,
        )


if __name__ == "__main__":
    main()
