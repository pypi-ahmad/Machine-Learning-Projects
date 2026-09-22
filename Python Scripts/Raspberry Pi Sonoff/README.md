# Raspberry Pi Sonoff

> A Flask web interface for controlling a Sonoff-style relay connected to a Raspberry Pi GPIO pin.

## Overview

This project runs a Flask server on a Raspberry Pi with an HTML interface for turning a relay on or off through GPIO pin 2. The page displays the current status and provides ON/OFF buttons.

This project must run on the wired Raspberry Pi. Dependency installation can be resolved on Windows, but GPIO control cannot be verified without the target hardware.

## Features

- Flask web server running on port 8000, accessible from any device on the local network
- ON/OFF control via dedicated URL endpoints (`/on`, `/off`)
- Real-time relay status display on the home page (ON or OFF)
- HTML interface with ON and OFF buttons
- Displays a GPIO pin layout reference image from gpiozero documentation

## Project Structure

```
Raspberry-Pi-Sonoff/
├── Main.py              # Flask application
├── templates/
│   └── home.html        # Web interface template
├── img/
│   ├── rpi.png          # Raspberry Pi image (for docs)
│   └── relay.png        # Relay board image (for docs)
├── LICENSE              # MIT License
└── README.md
```

## Requirements

- Python 3.x
- Raspberry Pi (any model)
- Relay board connected to GPIO pin 2
- `flask`
- `gpiozero` (typically pre-installed on Raspberry Pi OS)

## Installation

```powershell
uv sync --no-config
```

## Usage

Run on the Raspberry Pi:

```powershell
uv run --no-config python Main.py
```

The server starts at `http://0.0.0.0:8000/`. Access it from any browser on the same network using the Pi's IP address:

```
http://<raspberry-pi-ip>:8000/
```

### Endpoints

| Route  | Action                           |
|--------|----------------------------------|
| `/`    | Home page with status and buttons|
| `/on`  | Turns the LED/relay ON           |
| `/off` | Turns the LED/relay OFF          |

## How it works

1. **GPIO Setup**: Uses `gpiozero.LED(2)` to control GPIO pin 2 as a digital output.
2. **Home Route (`/`)**: Checks `led.value` — if 1, status is "ON"; otherwise "OFF". Renders `home.html` with the status passed as a template variable.
3. **ON Route (`/on`)**: Calls `led.on()` and returns "LED on" as plain text.
4. **OFF Route (`/off`)**: Calls `led.off()` and returns "LED off" as plain text.
5. **HTML Template**: Displays the relay status using Jinja2 `{{status}}` and provides hyperlinked buttons pointing to `/on` and `/off`.

## Configuration

- `LED(2)` — GPIO pin number (line 4 of `Main.py`); change to match your wiring
- `app.run(host='0.0.0.0', port=8000)` — host and port; `0.0.0.0` makes it accessible on the network
- The HTML template references an external GPIO pin layout image from `gpiozero.readthedocs.io`

## Limitations

- No authentication — anyone on the network can control the relay
- `/on` and `/off` routes return plain text instead of redirecting back to the home page
- No error handling for GPIO failures
- The `sleep` import from the `time` module is unused
- No HTTPS support
- Hardcoded to a single relay on GPIO pin 2

## Security Notes

- **No authentication or access control** — the server is accessible to anyone on the local network. Consider adding authentication for production use.
- The server binds to `0.0.0.0`, making it accessible on all network interfaces.
- Flask debug mode is not enabled, but no CSRF protection is implemented.

## License

MIT License (Copyright (c) 2020 Arbaz Khan)
