"""Resolve a hostname to an IPv4 address."""

import argparse
import socket


def resolve(hostname: str) -> str:
    return socket.gethostbyname(hostname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a hostname to IPv4.")
    parser.add_argument("hostname", nargs="?", help="Hostname to resolve")
    args = parser.parse_args()
    hostname = args.hostname or input("Hostname: ").strip()
    if not hostname:
        parser.error("hostname is required")
    try:
        print(f"Hostname: {hostname}")
        print(f"IP: {resolve(hostname)}")
    except socket.gaierror:
        print("Invalid hostname.")


if __name__ == "__main__":
    main()

