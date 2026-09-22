# Internet Speed Test

This documentation-only project describes a one-time command-line internet speed check. It has no Python source or project environment to migrate.

## Run

```powershell
uvx speedtest-cli
```

`uvx` installs and runs `speedtest-cli` in an isolated temporary environment, so it does not add a global package. The test contacts Speedtest.net infrastructure and transfers data to measure download speed, upload speed, and ping.

No bandwidth test was run while updating this documentation.
