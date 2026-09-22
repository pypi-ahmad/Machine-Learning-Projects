"""Preview or explicitly retrieve public Instagram profile metadata and posts."""

from __future__ import annotations

import argparse
from pathlib import Path

import instaloader


CONFIRMATION = "DOWNLOAD_POSTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview or retrieve Instagram profile data.")
    parser.add_argument("profile", help="Instagram profile username.")
    parser.add_argument("--metadata", action="store_true", help="Retrieve public profile metadata.")
    parser.add_argument("--download-posts", action="store_true", help="Download posts into a new directory.")
    parser.add_argument("--output", type=Path, help="New post-download directory.")
    parser.add_argument("--max-posts", type=int, default=1, help="Maximum posts to download (default: 1).")
    parser.add_argument("--confirm", help=f"Required for downloads: {CONFIRMATION}")
    return parser.parse_args()


def validate_profile(profile: str) -> str:
    profile = profile.strip().lstrip("@")
    if not profile.replace("_", "").replace(".", "").isalnum():
        raise ValueError("Profile names may contain letters, numbers, underscores, and periods only.")
    return profile


def load_profile(loader: instaloader.Instaloader, username: str) -> instaloader.Profile:
    return instaloader.Profile.from_username(loader.context, username)


def print_metadata(profile: instaloader.Profile) -> None:
    print(f"Username: {profile.username}")
    print(f"User ID: {profile.userid}")
    print(f"Posts: {profile.mediacount}")
    print(f"Followers: {profile.followers}")
    print(f"Followees: {profile.followees}")
    print(f"Bio: {profile.biography}")
    print(f"External URL: {profile.external_url or 'None'}")


def download_posts(loader: instaloader.Instaloader, profile: instaloader.Profile, output: Path, limit: int) -> int:
    output.mkdir(parents=True)
    count = 0
    for post in profile.get_posts():
        loader.download_post(post, target=str(output / f"post-{count + 1:03}"))
        count += 1
        if count >= limit:
            break
    return count


def main() -> None:
    args = parse_args()
    try:
        username = validate_profile(args.profile)
        if args.max_posts <= 0:
            raise ValueError("--max-posts must be greater than zero.")
    except ValueError as error:
        raise SystemExit(f"Invalid input: {error}") from error

    if not args.metadata and not args.download_posts:
        print(f"Preview only for @{username}. No Instagram request or download will occur.")
        print("Use --metadata to request public profile details.")
        print(f"Use --download-posts --confirm {CONFIRMATION} to download posts.")
        return
    if args.download_posts and args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing to download. Use --confirm {CONFIRMATION}.")

    output = args.output or Path("downloads") / username
    if args.download_posts and output.exists():
        raise SystemExit(f"Output already exists: {output}. Choose a new --output path.")

    loader = instaloader.Instaloader()
    try:
        profile = load_profile(loader, username)
        if args.metadata:
            print_metadata(profile)
        if args.download_posts:
            print(f"Downloaded {download_posts(loader, profile, output, args.max_posts)} post(s) to {output}.")
    except instaloader.exceptions.InstaloaderException as error:
        raise SystemExit(f"Instagram request failed: {error}") from error


if __name__ == "__main__":
    main()
