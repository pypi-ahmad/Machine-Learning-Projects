# Speed Test

A command-line internet speed test using the `speedtest-cli` Python package.
It measures download bandwidth, upload bandwidth, ping, and the selected test
server.

## Run

```powershell
uv sync
uv run python speedtest.py --run
```

Use JSON output when another program needs the result:

```powershell
uv run python speedtest.py --run --json
```

`--run` is required. A live test contacts Speedtest infrastructure and consumes
network bandwidth, so the script does nothing merely by being imported or
invoked without explicit consent.

## Limitations

- Results vary with network load, Wi-Fi quality, VPNs, server selection, and
  competing traffic.
- The result measures the connection at that moment; it is not a service-level
  guarantee.
- The test needs internet access and may disclose connection metadata to the
  selected speed-test service.

## Dependencies

uv manages `speedtest-cli` in `pyproject.toml`; the resolved version is
recorded in `uv.lock`.
