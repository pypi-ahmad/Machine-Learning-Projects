"""Convert integers through nonillion into English words."""

import argparse


ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
MAXIMUM = 10**33 - 1


def under_thousand(number: int) -> str:
    """Convert an integer from zero through 999."""
    if number < 20:
        return ONES[number]
    if number < 100:
        tens, remainder = divmod(number, 10)
        return TENS[tens] if remainder == 0 else f"{TENS[tens]} {ONES[remainder]}"
    hundreds, remainder = divmod(number, 100)
    prefix = f"{ONES[hundreds]} hundred"
    return prefix if remainder == 0 else f"{prefix} and {under_thousand(remainder)}"


def positive_to_words(number: int) -> str:
    """Convert a positive integer through nonillion without decimal rounding."""
    if number < 1_000:
        return under_thousand(number)
    if number < 1_000_000:
        divisor, scale = 1_000, "thousand"
    elif number < 1_000_000_000:
        divisor, scale = 1_000_000, "million"
    elif number < 1_000_000_000_000:
        divisor, scale = 1_000_000_000, "billion"
    elif number < 1_000_000_000_000_000:
        divisor, scale = 1_000_000_000_000, "trillion"
    elif number < 1_000_000_000_000_000_000:
        divisor, scale = 1_000_000_000_000_000, "quadrillion"
    elif number < 1_000_000_000_000_000_000_000:
        divisor, scale = 1_000_000_000_000_000_000, "quintillion"
    elif number < 1_000_000_000_000_000_000_000_000:
        divisor, scale = 1_000_000_000_000_000_000_000, "sextillion"
    elif number < 1_000_000_000_000_000_000_000_000_000:
        divisor, scale = 1_000_000_000_000_000_000_000_000, "septillion"
    elif number < 1_000_000_000_000_000_000_000_000_000_000:
        divisor, scale = 1_000_000_000_000_000_000_000_000_000, "octillion"
    else:
        divisor, scale = 1_000_000_000_000_000_000_000_000_000_000, "nonillion"

    quotient, remainder = divmod(number, divisor)
    prefix = f"{positive_to_words(quotient)} {scale}"
    return prefix if remainder == 0 else f"{prefix} {positive_to_words(remainder)}"


def number_to_words(number: int) -> str:
    """Convert a signed integer through nonillion into title-cased English."""
    if not -MAXIMUM <= number <= MAXIMUM:
        raise ValueError("Number must be between -999999999999999999999999999999999 and 999999999999999999999999999999999.")
    if number < 0:
        return f"Negative {positive_to_words(abs(number))}".title()
    return positive_to_words(number).title()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an integer to English words.")
    parser.add_argument("number", nargs="?", type=int, help="Integer to convert")
    args = parser.parse_args()

    if args.number is not None:
        print(number_to_words(args.number))
        return

    while True:
        answer = input("Enter an integer or 'exit': ").strip()
        if answer.lower() == "exit":
            return
        try:
            print(number_to_words(int(answer)))
        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
