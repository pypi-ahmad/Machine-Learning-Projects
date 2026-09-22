# Age calculator GUI

`age_calc_gui.py` is a small Tkinter desktop application that calculates a
person's completed years from their name and birth date.

## Install and run

```powershell
cd "Python Scripts/Age Calculator GUI"
uv sync
uv run python age_calc_gui.py
```

The project has no third-party dependencies. Tkinter is included with standard
Windows Python installations.

## Use

Enter a name, year, month, and day, then select **Calculate age**. The app
displays whole years and accounts for whether the birthday has occurred in the
current calendar year.

Invalid calendar dates, empty names, and future birth dates are reported in the
window instead of crashing the application.
