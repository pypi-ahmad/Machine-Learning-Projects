# Instagram Follow and Message Bot

This Selenium script previews Instagram follow and direct-message requests by default. It does not open Chrome, ask for credentials, or interact with Instagram unless you use the explicit apply command.

## Preview

```powershell
uv sync --no-config
uv run --no-config python instabot.py example_account another_account
```

The preview validates usernames locally and prints the actions that would be requested.

## Apply actions

Only use this with accounts you are authorized to contact and after reviewing Instagram's current rules. The command requires the exact confirmation phrase:

```powershell
uv run --no-config python instabot.py example_account `
  --apply --confirm FOLLOW_AND_MESSAGE
```

Chrome opens only after that confirmation. The script prompts for an Instagram username, a hidden password, and one message to send to every listed account. It does not save credentials.

## Limitations

- Instagram can change its interface or block automated actions, so selectors may stop working.
- Following or messaging accounts may have account-policy consequences; use the script only with clear permission.
- The actual apply action requires Chrome and internet access. It was not exercised during local verification.
