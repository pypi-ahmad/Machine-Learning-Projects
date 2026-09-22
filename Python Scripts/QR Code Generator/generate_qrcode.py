"""Generate a PNG QR code from supplied text or a URL."""

import argparse
from pathlib import Path

import qrcode


def create_qr(data: str, output: Path) -> None:
    """Write a QR code PNG using the project's original visual settings."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=15,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr.make_image(fill_color="red", back_color="white").save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PNG QR code.")
    parser.add_argument("data", help="Text or URL to encode")
    parser.add_argument("--output", type=Path, default=Path("qrcode.png"))
    args = parser.parse_args()
    create_qr(args.data, args.output)
    print(f"Saved QR code to {args.output}")


if __name__ == "__main__":
    main()
