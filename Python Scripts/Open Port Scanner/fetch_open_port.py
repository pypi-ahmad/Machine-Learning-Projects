"""Check a bounded TCP port range on one authorized host."""

import argparse
import socket
import time


def parse_port_range(value: str) -> tuple[int, int]:
    """Parse an inclusive START:END port range."""
    try:
        start, end = map(int, value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("port range must use START:END") from error
    if not 1 <= start <= end <= 65535:
        raise argparse.ArgumentTypeError("ports must be between 1 and 65535")
    return start, end


def scan(address: str, start: int, end: int, timeout: float) -> list[int]:
    """Return TCP ports that accept one connection attempt."""
    open_ports = []
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            if connection.connect_ex((address, port)) == 0:
                open_ports.append(port)
    return open_ports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan one TCP port range on a host you are authorized to test."
    )
    parser.add_argument("host", help="Hostname or IPv4 address to scan")
    parser.add_argument("--ports", type=parse_port_range, default="50:500", help="Inclusive START:END range")
    parser.add_argument("--timeout", type=float, default=0.5, help="Per-port timeout in seconds")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    start, end = args.ports
    try:
        address = socket.gethostbyname(args.host)
        print(f"Scanning {address} on TCP ports {start}-{end}.")
        started = time.perf_counter()
        open_ports = scan(address, start, end, args.timeout)
    except socket.gaierror as error:
        parser.exit(1, f"Unable to resolve host: {error}\n")

    for port in open_ports:
        print(f"Port {port}: OPEN")
    print(f"Completed in {time.perf_counter() - started:.2f} seconds.")


if __name__ == "__main__":
    main()
