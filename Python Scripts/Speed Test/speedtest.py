"""Measure connection speed through the speedtest-cli package API.

Usage:
    uv run python speedtest.py --run [--json]
"""

import argparse
import json

import speedtest


def measure() -> dict[str, float | str]:
    """Run a live download and upload bandwidth measurement."""
    tester = speedtest.Speedtest(secure=True)
    tester.get_best_server()
    download_mbps = tester.download() / 1_000_000
    upload_mbps = tester.upload() / 1_000_000
    results = tester.results.dict()
    return {
        "download_mbps": round(download_mbps, 2),
        "upload_mbps": round(upload_mbps, 2),
        "ping_ms": round(results["ping"], 2),
        "server": results["server"]["name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure internet download, upload, and ping.")
    parser.add_argument("--run", action="store_true", help="Run a live bandwidth measurement")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required because a speed test uses network bandwidth")

    try:
        results = measure()
    except Exception as error:
        raise SystemExit(f"Speed test failed: {error}") from error

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("Speed test results")
    print(f"Download: {results['download_mbps']} Mbps")
    print(f"Upload:   {results['upload_mbps']} Mbps")
    print(f"Ping:     {results['ping_ms']} ms")
    print(f"Server:   {results['server']}")


if __name__ == "__main__":
    main()
