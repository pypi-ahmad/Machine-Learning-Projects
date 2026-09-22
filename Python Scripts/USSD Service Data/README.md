# USSD Service Data

A terminal-only simulation of a USSD banking menu. It demonstrates menu navigation for account opening, balance checks, transfers, and bank selection; it does not connect to a bank, process payments, store accounts, or verify identities.

## Run it

```powershell
uv sync
uv run python ussdtim.py
```

The simulator begins with a short welcome delay and accepts `*919#` as its demonstration USSD code.

## Important limits

- This is an educational simulation, not a banking product or USSD client.
- Do not enter real names, PINs, account numbers, phone numbers, BVNs, or other personal or financial information.
- All prompts and displayed values are local placeholders; no network request or real transfer occurs.

## Dependencies

- Python 3.14+
- No third-party packages
