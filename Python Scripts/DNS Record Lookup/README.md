# DNS Record Fetcher

> A CLI script that fetches DNS A and MX records for a given website using `dnspython`.

## Overview

This script prompts the user for a website domain name, then queries its DNS A record (IP address) and MX records (mail exchange servers), storing the results in a dictionary and printing them to the console.

## Features

- Fetches the DNS **A record** (IP address) for a domain
- Fetches all DNS **MX records** (mail servers) for a domain
- Displays results in a key-value format

## Project Structure

```
Dns_record/
├── dns_record.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `dnspython==2.0.0`

## Installation

```bash
cd Dns_record
pip install -r requirements.txt
```

## Usage

```bash
python dns_record.py
```

When prompted, enter a domain name (e.g., `google.com`):

```
Enter the name of the website: google.com
```

Sample output:

```
A_Record_IP = 142.250.80.46
('MX_Record', 1) = 10 smtp.google.com.
```

## How It Works

1. Uses `dns.resolver.resolve()` to query the **A record** and stores the first IP address in a dictionary under the key `A_Record_IP`.
2. Queries the **MX records** and stores each mail server in the dictionary with a tuple key `('MX_Record', index)`.
3. Iterates over the dictionary and prints all records.

## Configuration

No configuration files. The website domain is provided interactively at runtime.

## Limitations

- Only stores the **last** A record IP (overwrites if multiple A records exist).
- Uses a tuple `('MX_Record', i)` as a dictionary key, which produces non-standard output formatting.
- No error handling for invalid domains or DNS resolution failures.
- No command-line argument support — input is interactive only.

## License

Not specified.
