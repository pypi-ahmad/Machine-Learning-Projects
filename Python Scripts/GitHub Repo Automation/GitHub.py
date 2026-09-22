"""Preview or create a private GitHub repository and local folder.

Usage:
    uv run --no-config python GitHub.py NAME --directory PARENT
    uv run --no-config python GitHub.py NAME --directory PARENT --apply --confirm CREATE
"""

import argparse
import re
import subprocess
from pathlib import Path


REPOSITORY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def validate_name(name: str) -> str:
    if not REPOSITORY_NAME.fullmatch(name):
        raise ValueError("Repository names may contain letters, numbers, periods, underscores, and hyphens.")
    return name


def create_repository(name: str, visibility: str, description: str | None) -> None:
    command = ["gh", "repo", "create", name, f"--{visibility}"]
    if description:
        command.extend(["--description", description])
    result = subprocess.run(command, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("GitHub repository creation failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GitHub repository and matching local folder.")
    parser.add_argument("name", help="New repository name")
    parser.add_argument("--directory", type=Path, required=True, help="Existing parent directory for the new folder")
    parser.add_argument("--visibility", choices=("private", "public"), default="private")
    parser.add_argument("--description", help="Optional GitHub repository description")
    parser.add_argument("--apply", action="store_true", help="Create the remote repository and local folder")
    parser.add_argument("--confirm", help="Type CREATE to authorize creation")
    args = parser.parse_args()

    try:
        name = validate_name(args.name)
    except ValueError as error:
        parser.error(str(error))

    parent = args.directory.resolve()
    destination = parent / name
    if not parent.is_dir():
        parser.error(f"directory is not an existing folder: {parent}")
    if destination.exists():
        parser.error(f"destination already exists: {destination}")

    print(f"GitHub repository: {name} ({args.visibility})")
    print(f"Local folder:      {destination}")
    if not args.apply:
        print("Preview only. Add --apply --confirm CREATE to create both resources.")
        return
    if args.confirm != "CREATE":
        parser.error("--confirm CREATE is required with --apply")

    create_repository(name, args.visibility, args.description)
    try:
        destination.mkdir()
    except OSError as error:
        raise RuntimeError(f"Remote repository exists, but local folder creation failed: {error}") from error
    print("Created remote repository and local folder.")


if __name__ == "__main__":
    main()
