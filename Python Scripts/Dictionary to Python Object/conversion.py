"""Convert nested dictionaries into objects with attribute access."""


class obj:
    """Wrap a dictionary whose keys are safe Python attribute names."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if not isinstance(key, str) or not key.isidentifier() or key.startswith("_"):
                raise ValueError(f"Invalid attribute key: {key!r}")
            setattr(self, key, obj(value) if isinstance(value, dict) else value)


def main() -> None:
    """Run the original nested-dictionary demonstration."""
    data = {"a": 5, "b": 7, "c": {"d": 8}}
    converted = obj(data)
    print(f"a={converted.a}, c.d={converted.c.d}")


if __name__ == "__main__":
    main()
