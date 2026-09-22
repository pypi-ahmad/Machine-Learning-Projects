# Instagram Follow Tracker

Compare the following and follower lists for one Instagram account. The default command is a local preview; it does not open Chrome, request a password, or contact Instagram.

## Preview

```powershell
uv sync --no-config
uv run --no-config python main.py example_account
```

## Run the read-only check

Use `--run` only for an account you are authorized to inspect:

```powershell
uv run --no-config python main.py example_account --run
```

The command opens Chrome, asks for the password without echoing it, reads Instagram's follower and following lists, then prints accounts that do not follow back. It does not follow, unfollow, message, or modify account data.

Instagram can change its interface or restrict automated access, so this browser-based check may stop working or be subject to account-policy consequences. The live browser flow was not exercised during local verification.
