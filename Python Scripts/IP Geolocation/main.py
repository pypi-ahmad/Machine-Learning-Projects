"""IP Geolocation — CLI tool.

Look up geographic location, ISP, timezone, and coordinates for
an IP address using the ip-api.com free JSON API.
Falls back to ipinfo.io if primary is unavailable.

Usage:
    python main.py
    python main.py 8.8.8.8
"""

import json
import socket
import sys
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def lookup_ip_api(ip: str) -> dict | None:
    """http://ip-api.com/json/{ip} (free, no key needed, 45 req/min)."""
    url = f"http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,region,regionName,city,zip,lat,lon,timezone,isp,org,as,query"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "success":
                return data
    except Exception:
        pass
    return None


def lookup_ipinfo(ip: str) -> dict | None:
    """https://ipinfo.io/{ip}/json (free, no key, 50k/month)."""
    url = f"https://ipinfo.io/{ip}/json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        pass
    return None


def get_my_ip() -> str:
    """Fetch public IP."""
    try:
        with urllib.request.urlopen("https://api.ipify.org?format=json", timeout=5) as r:
            return json.loads(r.read())["ip"]
    except Exception:
        return "8.8.8.8"  # fallback to Google DNS for demo


def hostname(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except socket.herror:
        return "N/A"


def resolve_host(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return ""


def format_result(data: dict, source: str = "ip-api") -> None:
    print()
    if source == "ip-api":
        fields = [
            ("IP",          data.get("query")),
            ("Country",     f"{data.get('country')} ({data.get('countryCode')})"),
            ("Region",      data.get("regionName")),
            ("City",        data.get("city")),
            ("ZIP",         data.get("zip")),
            ("Coordinates", f"{data.get('lat')}, {data.get('lon')}"),
            ("Timezone",    data.get("timezone")),
            ("ISP",         data.get("isp")),
            ("Org",         data.get("org")),
            ("AS",          data.get("as")),
        ]
    else:
        loc = data.get("loc", ",").split(",")
        fields = [
            ("IP",          data.get("ip")),
            ("Hostname",    data.get("hostname")),
            ("City",        data.get("city")),
            ("Region",      data.get("region")),
            ("Country",     data.get("country")),
            ("Coordinates", data.get("loc")),
            ("Org",         data.get("org")),
            ("Timezone",    data.get("timezone")),
        ]

    for label, val in fields:
        if val:
            print(f"  {label:<14}: {val}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
IP Geolocation
--------------
1. Look up an IP address
2. Look up my public IP
3. Resolve hostname to IP
4. Batch lookup (multiple IPs)
0. Quit
"""


def do_lookup(ip: str) -> None:
    print(f"\n  Looking up {ip}...", end=" ", flush=True)
    data = lookup_ip_api(ip)
    if data:
        print("OK (ip-api.com)")
        format_result(data, "ip-api")
    else:
        data = lookup_ipinfo(ip)
        if data:
            print("OK (ipinfo.io)")
            format_result(data, "ipinfo")
        else:
            print("Failed — check internet connection.")


def main() -> None:
    if len(sys.argv) > 1:
        do_lookup(sys.argv[1])
        return

    print("IP Geolocation")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            ip_or_host = input("  IP or hostname: ").strip()
            if not ip_or_host:
                continue
            # Resolve hostname if needed
            if not ip_or_host.replace(".", "").isdigit():
                resolved = resolve_host(ip_or_host)
                if resolved:
                    print(f"  Resolved: {ip_or_host} → {resolved}")
                    ip_or_host = resolved
                else:
                    print(f"  Could not resolve: {ip_or_host}")
                    continue
            do_lookup(ip_or_host)

        elif choice == "2":
            my_ip = get_my_ip()
            do_lookup(my_ip)

        elif choice == "3":
            host = input("  Hostname: ").strip()
            resolved = resolve_host(host)
            if resolved:
                print(f"\n  {host} → {resolved}")
                print(f"  Reverse: {hostname(resolved)}")
            else:
                print(f"  Could not resolve: {host}")

        elif choice == "4":
            print("  Enter IPs one per line (blank to finish):")
            ips = []
            while True:
                ip = input("  > ").strip()
                if not ip:
                    break
                ips.append(ip)
            for ip in ips:
                do_lookup(ip)
                print()

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
