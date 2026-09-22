# Age calculator

`main.py` calculates a person's completed age, time-lived statistics, and next
birthday from a supplied birth date.

## Install and run

```powershell
cd "Python Scripts/Age Calculator"
uv sync
uv run python main.py 1990-05-15
```

The project uses only the Python standard library.

## Accepted date formats

- `YYYY-MM-DD` such as `1990-05-15`
- `DD/MM/YYYY` such as `15/05/1990`
- `DD-MM-YYYY` such as `15-05-1990`
- `MM/DD/YYYY` such as `05/15/1990`

The output includes completed years, months and days, time-lived totals, the
weekday of birth, and the next birthday.

For February 29 birthdays, the calculator treats February 28 as the birthday
in years that do not have a February 29.

Future dates and invalid calendar dates are rejected with a clear error. Run
`uv run python main.py --help` for the command reference.
