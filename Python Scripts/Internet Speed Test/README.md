# Internet Speed Test

> A guide for testing internet speed from the command line using the `speedtest-cli` package.

## Overview

This project contains instructions (no Python source code) for using the `speedtest-cli` CLI tool to measure download speed, upload speed, and ping from the terminal.

## Features

- Measure download speed
- Measure upload speed
- Measure ping latency

## Project Structure

```
Internet-Speed-Test/
├── Internet-Speed-Test.md   # Instructions document
└── README.md
```

## Requirements

- Python 3.x
- `speedtest-cli`

## Installation

```bash
pip install speedtest-cli
```

## Usage

```bash
speedtest-cli
```

This will output your download speed, upload speed, and ping.

## How It Works

The `speedtest-cli` package connects to a nearby Speedtest.net server and measures your network bandwidth by downloading and uploading test data.

## Configuration

None.

## Limitations

- **No Python source code** is included in this project — it is solely a set of CLI instructions.
- The `Internet-Speed-Test.md` file duplicates what this README covers.

## Security Notes

No security concerns identified.

## License

Not specified.
