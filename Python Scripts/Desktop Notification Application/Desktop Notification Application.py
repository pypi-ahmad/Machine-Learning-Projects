"""Send configurable desktop reminders at a fixed interval."""

import argparse
import time

from plyer import notification

DEFAULT_INTERVAL_SECONDS = 3600
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_TITLE = "Break reminder"
DEFAULT_MESSAGE = "Take a break. You have been working for an hour."


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer for an argparse option."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Return the reminder configuration supplied at the command line."""
    parser = argparse.ArgumentParser(description="Send periodic desktop break reminders.")
    parser.add_argument("--interval", type=positive_integer, default=DEFAULT_INTERVAL_SECONDS,
                        help="seconds between notifications (default: 3600)")
    parser.add_argument("--timeout", type=positive_integer, default=DEFAULT_TIMEOUT_SECONDS,
                        help="notification display time in seconds (default: 10)")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="notification title")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="notification message")
    parser.add_argument("--once", action="store_true", help="send one notification and exit")
    return parser.parse_args(arguments)


def send_notification(title: str, message: str, timeout: int) -> None:
    """Send one operating-system notification through plyer."""
    notification.notify(title=title, message=message, timeout=timeout)


def main(arguments: list[str] | None = None) -> int:
    """Run the reminder until interrupted, or once when requested."""
    settings = parse_arguments(arguments)
    try:
        while True:
            send_notification(settings.title, settings.message, settings.timeout)
            if settings.once:
                return 0
            time.sleep(settings.interval)
    except KeyboardInterrupt:
        print("Reminder stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
