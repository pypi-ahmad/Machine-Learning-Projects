"""Fetch DNS A and MX records for a domain."""

import argparse

import dns.exception
import dns.resolver


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch DNS A and MX records.')
    parser.add_argument('domain', nargs='?', help='Domain to query')
    args = parser.parse_args()
    domain = args.domain or input('Enter the name of the website: ').strip()

    try:
        a_records = dns.resolver.resolve(domain, 'A')
        mx_records = dns.resolver.resolve(domain, 'MX')
    except dns.exception.DNSException as error:
        print(f'DNS lookup failed: {error}')
        return

    print('A records:')
    print(*a_records, sep='\n')
    print('MX records:')
    print(*mx_records, sep='\n')


if __name__ == '__main__':
    main()

