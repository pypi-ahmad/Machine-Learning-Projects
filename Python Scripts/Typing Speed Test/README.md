# Typing Speed Test

> A terminal-based typing speed and accuracy test using a fixed sample phrase.

## Overview

A command-line typing speed test that displays a predefined English sentence and measures how fast and accurately the user can retype it. Calculates words per minute (WPM) and accuracy percentage based on word-level matching.

## Features

- Measures typing speed in words per minute (WPM)
- Calculates accuracy by comparing typed words against the reference phrase
- Displays total words typed and time taken in seconds
- Continuous retry loop — play as many rounds as desired
- Bordered display box for the typing prompt

## Project Structure

```
TYPING_SPEED_TEST/
├── Typing Speed Test.py
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `time` from the standard library)

## Installation

```bash
cd "TYPING_SPEED_TEST"
```

No package installation required.

## Usage

```bash
python "Typing Speed Test.py"
```

1. The reference phrase is displayed: `Python is an interpreted, high-level programming language`
2. Type the phrase as fast and accurately as possible, then press Enter.
3. Results are shown: total words, time, accuracy percentage, and WPM.
4. Press Enter on an empty line to exit, or type anything to retry.

### Example Output

```
-+--+--+--+--+--+--+--+--+--+-

Enter the phrase as fast as possible and with accuracy

Python is an interpreted, high-level programming language

Total words      : 8
Time used        : 5.23 seconds
Your accuracy    : 87.5 %
Speed is         : 91.78 words per minute
```

## How It Works

1. A fixed string is set: `"Python is an interpreted, high-level programming language"` (7 words).
2. A timer starts when the prompt appears (before user input).
3. After the user presses Enter, the timer stops.
4. **Accuracy** is calculated as the ratio of common unique words (set intersection) between the input and reference to the total word count in the reference.
5. **WPM** is calculated as `(words typed / time in seconds) * 60`.
6. The retry prompt uses Python's truthy evaluation: empty input (falsy) exits, any input (truthy) continues.

## Configuration

- The reference phrase is hardcoded in the script as `string = "Python is an interpreted, high-level programming language"`.

## Limitations

- Only one hardcoded test phrase — no variety or randomization.
- Accuracy uses set intersection, so duplicate words and word order are ignored (typing "Python Python Python" would count "Python" as matched).
- Timer starts when the prompt is displayed, not when the user starts typing.
- Accuracy calculation multiplies by 100 after rounding to 3 decimal places, which can give unexpected results (e.g., `0.875` rounds to `0.875`, then `* 100 = 87.5`).
- No handling of punctuation differences (e.g., missing commas affect word matching since `split()` keeps punctuation attached).

## License

Not specified.
