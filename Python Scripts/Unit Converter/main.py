"""Unit Converter — CLI tool.

Converts between common units across length, mass, temperature,
volume, speed, time, area, and digital storage.

Usage:
    python main.py
"""

# ---------------------------------------------------------------------------
# Configuration — conversion factors relative to a base unit
# ---------------------------------------------------------------------------

# Each category maps unit_name → factor_to_base
# Temperature is handled separately.

CONVERSIONS: dict[str, dict[str, float]] = {
    "length": {
        "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
        "inch": 0.0254, "ft": 0.3048, "yard": 0.9144, "mile": 1609.344,
        "nautical_mile": 1852.0,
    },
    "mass": {
        "mg": 1e-6, "g": 0.001, "kg": 1.0, "tonne": 1000.0,
        "lb": 0.453592, "oz": 0.028350,
    },
    "volume": {
        "ml": 0.001, "l": 1.0, "m3": 1000.0,
        "cup": 0.236588, "pint": 0.473176, "quart": 0.946353,
        "gallon": 3.785411, "fl_oz": 0.029574,
    },
    "speed": {
        "m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704,
        "knot": 0.514444, "ft/s": 0.3048,
    },
    "time": {
        "ms": 0.001, "s": 1.0, "min": 60.0, "hr": 3600.0,
        "day": 86400.0, "week": 604800.0, "year": 31_557_600.0,
    },
    "area": {
        "mm2": 1e-6, "cm2": 1e-4, "m2": 1.0, "km2": 1e6,
        "inch2": 6.4516e-4, "ft2": 0.092903, "acre": 4046.856,
        "hectare": 10000.0,
    },
    "digital": {
        "bit": 1.0, "byte": 8.0, "kb": 8e3, "mb": 8e6,
        "gb": 8e9, "tb": 8e12,
    },
    "pressure": {
        "pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": 1e5,
        "atm": 101325.0, "psi": 6894.757,
    },
}


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

def convert_standard(value: float, from_unit: str, to_unit: str, table: dict) -> float:
    if from_unit not in table:
        raise ValueError(f"Unknown unit: {from_unit}")
    if to_unit not in table:
        raise ValueError(f"Unknown unit: {to_unit}")
    base = value * table[from_unit]
    return base / table[to_unit]


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between C, F, K."""
    fu, tu = from_unit.lower(), to_unit.lower()
    # to Celsius first
    if fu == "c":
        celsius = value
    elif fu == "f":
        celsius = (value - 32) * 5 / 9
    elif fu == "k":
        celsius = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    # from Celsius to target
    if tu == "c":
        return celsius
    elif tu == "f":
        return celsius * 9 / 5 + 32
    elif tu == "k":
        return celsius + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def convert(value: float, from_unit: str, to_unit: str) -> tuple[float, str]:
    fu = from_unit.lower()
    tu = to_unit.lower()

    # Temperature?
    if fu in {"c", "f", "k"} or tu in {"c", "f", "k"}:
        result = convert_temperature(value, fu, tu)
        return result, "temperature"

    # Search all tables
    for category, table in CONVERSIONS.items():
        if fu in table and tu in table:
            result = convert_standard(value, fu, tu, table)
            return result, category

    raise ValueError(
        f"Cannot convert '{from_unit}' to '{to_unit}'. "
        "Make sure both units belong to the same category."
    )


def list_units() -> None:
    print("\nAvailable units by category:")
    for cat, table in CONVERSIONS.items():
        units = ", ".join(sorted(table.keys()))
        print(f"  {cat:<12}: {units}")
    print("  temperature : c, f, k\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("Unit Converter  (type 'list' for units, 'quit' to exit)")
    print("Format:  <value> <from_unit> <to_unit>   e.g.  100 km mile")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if raw.lower() in {"list", "units", "help"}:
            list_units()
            continue

        parts = raw.split()
        if len(parts) != 3:
            print("Usage: <value> <from_unit> <to_unit>  e.g.  100 km mile")
            continue

        value_str, from_unit, to_unit = parts
        try:
            value = float(value_str)
        except ValueError:
            print(f"Invalid number: {value_str}")
            continue

        try:
            result, category = convert(value, from_unit, to_unit)
            print(f"  {value:g} {from_unit}  =  {result:.6g} {to_unit}  [{category}]")
        except ValueError as err:
            print(f"Error: {err}")


if __name__ == "__main__":
    main()
