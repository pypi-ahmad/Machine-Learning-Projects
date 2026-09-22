"""Convert one decimal integer to binary or one binary integer to decimal."""


def decimal_to_binary(value: str) -> str:
    """Convert a signed base-10 integer string to binary."""
    return format(int(value), "b")


def binary_to_decimal(value: str) -> int:
    """Convert a signed binary integer string to decimal."""
    text = value.strip()
    digits = text[1:] if text.startswith(("-", "+")) else text
    if not digits or not set(digits) <= {"0", "1"}:
        raise ValueError("not a binary integer")
    return int(text, 2)


def main() -> int:
    """Run one interactive conversion."""
    menu = input("Choose an option:\n1. Decimal to binary\n2. Binary to decimal\nOption: ")
    if menu == "1":
        try:
            print(f"Binary: {decimal_to_binary(input('Decimal: '))}")
        except ValueError:
            print("Enter a valid decimal integer.")
            return 1
    elif menu == "2":
        try:
            print(f"Decimal: {binary_to_decimal(input('Binary: '))}")
        except ValueError:
            print("Enter a valid binary integer using only 0 and 1.")
            return 1
    else:
        print("Choose option 1 or 2.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
