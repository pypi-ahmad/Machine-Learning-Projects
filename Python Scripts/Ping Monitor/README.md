# Ping Monitor

Monitor one or more hosts with ICMP ping and a TCP fallback, then report latency, packet loss, and jitter.

## Setup

```powershell
uv sync --no-config
```

## Run

Use a bounded count for a finite monitoring session:

```powershell
uv run --no-config python main.py example.com --count 20 --interval 2
```

Monitor several hosts and print alerts for failures or latency above a threshold:

```powershell
uv run --no-config python main.py example.com 1.1.1.1 --count 10 --threshold 200 --alert
```

Run without host arguments for the interactive prompts. `--count 0` keeps monitoring until interrupted with `Ctrl+C`.

## Behavior and limits

- Uses the operating system `ping` command first, then tries a TCP connection to port 80 when ICMP does not return a result.
- Requires network access to the supplied hosts and may be affected by firewalls, ICMP policies, DNS, or port-80 availability.
- Runs locally and does not write files or send data to third-party services beyond the requested network probes.
