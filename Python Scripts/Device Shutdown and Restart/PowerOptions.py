"""Preview or explicitly execute a local shutdown or restart command."""

import argparse
import platform
import subprocess


def build_command(action: str, system_name: str | None = None) -> tuple[str, ...]:
    """Return the platform command for the requested power action."""
    system_name = system_name or platform.system()
    if system_name == "Windows":
        return ("shutdown", "/s" if action == "shutdown" else "/r", "/t", "0")
    if system_name in {"Linux", "Darwin"}:
        return ("shutdown", "-h" if action == "shutdown" else "-r", "now")
    raise ValueError(f"Unsupported operating system: {system_name}")


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse an optional power action and explicit execution flag."""
    parser = argparse.ArgumentParser(description="Preview or execute a local power action.")
    parser.add_argument("action", choices=("shutdown", "restart"), nargs="?")
    parser.add_argument("--execute", action="store_true",
                        help="execute the displayed command instead of previewing it")
    return parser.parse_args(arguments)


def prompt_for_action() -> str | None:
    """Read a legacy one-letter action choice from the terminal."""
    choice = input("Use 'r' for restart or 's' for shutdown: ").strip().lower()
    if choice == "r":
        return "restart"
    if choice == "s":
        return "shutdown"
    print("Choose 'r' for restart or 's' for shutdown.")
    return None


def main(arguments: list[str] | None = None) -> int:
    """Show the command by default and run it only with --execute."""
    settings = parse_arguments(arguments)
    action = settings.action or prompt_for_action()
    if action is None:
        return 1

    try:
        command = build_command(action)
    except ValueError as error:
        print(error)
        return 1

    print(f"{action.title()} command: {' '.join(command)}")
    if not settings.execute:
        print("Preview only. Add --execute to run this command.")
        return 0

    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not {action}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
