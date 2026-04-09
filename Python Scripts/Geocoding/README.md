# Geocoding

> A Python script that converts a street address into geographic coordinates (latitude and longitude) using the LocationIQ geocoding API.

## Overview

This script takes an address as input, sends it to the LocationIQ geocoding API, and returns the corresponding latitude and longitude coordinates.

## Features

- Converts any address to latitude/longitude coordinates (geocoding)
- Uses the LocationIQ REST API (`us1.locationiq.com`)
- Interactive command-line address input
- Displays both latitude and longitude from the API response

## Project Structure

```
Geocoding/
├── geocoding.py       # Main geocoding script
└── requirements.txt   # Python dependencies
```

## Requirements

- Python 3.x
- `requests==2.24.0`
- A LocationIQ API token (free tier available)

Full dependency list in `requirements.txt`:
```
certifi==2020.6.20
chardet==3.0.4
idna==2.10
requests==2.24.0
urllib3==1.25.10
```

## Installation

```bash
cd "Geocoding"
pip install -r requirements.txt
```

## Usage

1. Create a free account at [LocationIQ](https://locationiq.com/) and get your private token.
2. Edit `geocoding.py` and replace `"Your_private_token"` with your actual token.
3. Run:

```bash
python geocoding.py
```

Example interaction:

```
Input the address: 1600 Amphitheatre Parkway, Mountain View, CA
The latitude of the given address is: 37.4224764
The longitude of the given address is: -122.0842499
Thanks for using this script
```

## How It Works

1. Sets the base URL to `https://us1.locationiq.com/v1/search.php`.
2. Prompts the user for an address.
3. Constructs request parameters: API key, query address, and `format=json`.
4. Sends a GET request with the parameters using `requests.get()`.
5. Parses the JSON response and extracts `lat` and `lon` from the first result.
6. Prints the latitude and longitude.

## Configuration

| Variable        | Location         | Description                           |
|-----------------|------------------|---------------------------------------|
| `private_token` | `geocoding.py`   | Your LocationIQ API private token     |

## Limitations

- No error handling for network failures, invalid API tokens, or unresolvable addresses
- Always takes the first result from the API response — may not be the most accurate for ambiguous addresses
- The API token is hardcoded as a placeholder string
- No rate limiting consideration (LocationIQ free tier has request limits)
- The dependency versions in `requirements.txt` are outdated

## Security Notes

- The API token is stored as a **plaintext placeholder** in the source code — replace with your own token and consider using environment variables
- Do not share or commit your private LocationIQ token to version control

## License

Not specified.
