"""Ping Monitor — CLI developer tool.

Continuously ping one or more hosts, display response times,
track availability, and alert on failures.

Usage:
    python main.py google.com
    python main.py google.com 8.8.8.8 1.1.1.1
    python main.py google.com --interval 2 --count 20
    python main.py google.com --alert --threshold 200
"""

import argparse
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m",
        "magenta": "\033[95m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Ping implementation ────────────────────────────────────────────────────────

def _ping_tcp(host: str, port: int = 80, timeout: float = 2.0) -> float | None:
    """TCP connect-based ping (fallback when ICMP not available)."""
    try:
        start = time.time()
        sock  = socket.create_connection((host, port), timeout=timeout)
        elapsed = (time.time() - start) * 1000
        sock.close()
        return round(elapsed, 2)
    except (socket.timeout, socket.error, OSError):
        return None


def _ping_icmp_subprocess(host: str, timeout: float = 2.0) -> float | None:
    """Use system ping command to measure ICMP latency."""
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), host]
    else:
        cmd = ["ping", "-c", "1", "-W", str(int(timeout)), host]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 2
        )
        output = result.stdout + result.stderr

        # Parse RTT from output
        m = re.search(r"[Tt]ime[=<](\d+(?:\.\d+)?)\s*ms", output)
        if m:
            return float(m.group(1))
        # Windows "time<1ms" case
        m = re.search(r"time<1ms", output)
        if m:
            return 0.5
        # No rtt found but command exited 0 → host reachable but no time info
        if result.returncode == 0:
            return 0.0
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def ping_host(host: str, timeout: float = 2.0) -> float | None:
    """Return RTT in ms, or None on timeout/failure."""
    rtt = _ping_icmp_subprocess(host, timeout)
    if rtt is None:
        # Fall back to TCP
        rtt = _ping_tcp(host, port=80, timeout=timeout)
    return rtt


def resolve_host(host: str) -> str | None:
    """Resolve hostname to IP."""
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return None


# ── Bar display ────────────────────────────────────────────────────────────────

def rtt_bar(rtt: float, max_rtt: float = 500) -> str:
    """ASCII bar proportional to RTT."""
    if rtt is None:
        return c("✗", "red")
    width  = 20
    filled = int(min(rtt / max_rtt, 1.0) * width)
    bar    = "█" * filled + "░" * (width - filled)
    if rtt < 50:     col = "green"
    elif rtt < 150:  col = "cyan"
    elif rtt < 300:  col = "yellow"
    else:            col = "red"
    return c(bar, col) + f" {rtt:.1f} ms"


# ── Host monitor ───────────────────────────────────────────────────────────────

class HostStats:
    def __init__(self, host: str):
        self.host     = host
        self.sent     = 0
        self.received = 0
        self.rtts: list[float] = []
        self.last_rtt: float | None = None
        self.last_err: str | None   = None

    @property
    def packet_loss(self) -> float:
        return (1 - self.received / self.sent) * 100 if self.sent else 0.0

    @property
    def avg_rtt(self) -> float | None:
        return statistics.mean(self.rtts) if self.rtts else None

    @property
    def min_rtt(self) -> float | None:
        return min(self.rtts) if self.rtts else None

    @property
    def max_rtt(self) -> float | None:
        return max(self.rtts) if self.rtts else None

    @property
    def jitter(self) -> float | None:
        if len(self.rtts) < 2:
            return None
        diffs = [abs(self.rtts[i] - self.rtts[i-1]) for i in range(1, len(self.rtts))]
        return statistics.mean(diffs)


def print_summary(stats_list: list[HostStats]):
    print(c("\n─── Summary ─────────────────────────────────", "dim"))
    print(f"  {'Host':24} {'Sent':>6} {'Recv':>6} {'Loss%':>7} "
          f"{'Min ms':>8} {'Avg ms':>8} {'Max ms':>8} {'Jitter':>8}")
    print(c("  " + "─" * 75, "dim"))
    for s in stats_list:
        avg = f"{s.avg_rtt:.1f}" if s.avg_rtt is not None else "—"
        mn  = f"{s.min_rtt:.1f}" if s.min_rtt is not None else "—"
        mx  = f"{s.max_rtt:.1f}" if s.max_rtt is not None else "—"
        jit = f"{s.jitter:.1f}"  if s.jitter is not None  else "—"
        loss_col = "red" if s.packet_loss > 10 else ("yellow" if s.packet_loss > 0 else "green")
        print(f"  {c(s.host,'cyan'):34} {s.sent:>6} {s.received:>6} "
              f"{c(f'{s.packet_loss:.1f}', loss_col):>15} "
              f"{mn:>8} {avg:>8} {mx:>8} {jit:>8}")


def monitor(hosts: list[str], interval: float, count: int,
            timeout: float, threshold: float, alert: bool):
    stats_list = [HostStats(h) for h in hosts]

    # Resolve IPs
    print(c("\n  Ping Monitor\n", "bold"))
    for s in stats_list:
        ip = resolve_host(s.host)
        ip_str = f" ({ip})" if ip else " (unresolved)"
        print(f"  {c(s.host,'cyan')}{c(ip_str,'dim')}")
    print()

    seq = 0
    try:
        while count == 0 or seq < count:
            seq += 1
            ts = datetime.now().strftime("%H:%M:%S")
            print(c(f"  [{ts}] seq={seq}", "dim"))

            for s in stats_list:
                s.sent += 1
                rtt = ping_host(s.host, timeout=timeout)
                s.last_rtt = rtt

                if rtt is not None:
                    s.received += 1
                    s.rtts.append(rtt)
                    bar = rtt_bar(rtt)
                    print(f"    {c(s.host,'cyan'):28} {bar}")

                    if alert and threshold and rtt > threshold:
                        print(c(f"    ⚠ ALERT: {s.host} latency {rtt:.1f} ms > threshold {threshold} ms", "red"))
                else:
                    print(f"    {c(s.host,'cyan'):28} {c('✗  TIMEOUT', 'red')}")
                    if alert:
                        print(c(f"    ⚠ ALERT: {s.host} is unreachable!", "red"))

            if count == 0 or seq < count:
                time.sleep(interval)

    except KeyboardInterrupt:
        pass

    print_summary(stats_list)


def interactive_mode():
    print(c("Ping Monitor\n", "bold"))
    hosts_in  = input(c("  Hosts (space-separated): ", "cyan")).strip()
    interval  = input(c("  Interval seconds [1]: ", "cyan")).strip() or "1"
    count_in  = input(c("  Packet count (0=infinite) [20]: ", "cyan")).strip() or "20"
    threshold = input(c("  Alert threshold ms (0=off) [200]: ", "cyan")).strip() or "200"

    hosts = hosts_in.split()
    if not hosts:
        print(c("  No hosts specified.", "red"))
        return

    monitor(
        hosts=hosts,
        interval=float(interval),
        count=int(count_in),
        timeout=2.0,
        threshold=float(threshold),
        alert=float(threshold) > 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Continuous host ping monitor")
    parser.add_argument("hosts",       nargs="*",      help="Host(s) to ping")
    parser.add_argument("--interval",  type=float, default=1.0, help="Seconds between pings")
    parser.add_argument("--count","-n",type=int,   default=0,   help="Number of pings (0=infinite)")
    parser.add_argument("--timeout",   type=float, default=2.0, help="Ping timeout in seconds")
    parser.add_argument("--threshold", type=float, default=0,   help="Alert threshold in ms")
    parser.add_argument("--alert",     action="store_true",     help="Print alerts on failure/high latency")
    args = parser.parse_args()

    if args.hosts:
        monitor(
            hosts=args.hosts,
            interval=args.interval,
            count=args.count,
            timeout=args.timeout,
            threshold=args.threshold,
            alert=args.alert or args.threshold > 0,
        )
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
