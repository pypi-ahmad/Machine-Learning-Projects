# Convert Dictionary to Python Object

## Overview

A utility script that converts a Python dictionary into a Python object, allowing attribute-style access to dictionary keys instead of bracket notation.

**Type:** Utility

## Features

- Converts dictionaries to object instances using `setattr`
- Supports nested dictionaries (recursively converts inner dicts to objects)
- Allows attribute-style access (e.g., `ob.a` instead of `data['a']`)

## Dependencies

- Python 3.x (no external dependencies)

## How It Works

1. A class `obj` is defined that inherits from `object`.
2. The `__init__` method iterates over all key-value pairs in the provided dictionary.
3. For each pair, it calls `setattr` to set the key as an attribute on the object.
4. If a value is itself a dictionary, it recursively wraps it in another `obj` instance, enabling nested attribute access.

## Project Structure

```
convert_dictionary_to_python_object/
├── conversion.py    # Main script with the obj class and demo usage
└── README.md
```

## Setup & Installation

No installation required. Only a working Python 3 interpreter is needed.

## How to Run

```bash
python conversion.py
```

Or import and use in your own code:

```python
from conversion import obj

data = {'a': 5, 'b': 7, 'c': {'d': 8}}
ob = obj(data)
print(ob.a)    # 5
print(ob.c.d)  # 8
```

## Testing

No formal test suite present.

## Limitations

- The script includes a hardcoded example dictionary (`{'a':5,'b':7,'c':{'d':8}}`); modify as needed for your data.
- Does not handle lists of dictionaries — only top-level and nested dictionaries are converted.
- No validation or error handling for non-string dictionary keys.

