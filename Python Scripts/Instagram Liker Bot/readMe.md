# Instagram Liker Bot

Preview a bounded Instagram-like request before any browser activity. The default command does not open Chrome, log in, or like posts.

## Preview

```powershell
uv sync --no-config
uv run --no-config python Instagram_Liker_Bot.py example_account --max-likes 1
```

## Apply

Use only with permission and after reviewing Instagram's rules. The action requires a finite limit and exact confirmation:

```powershell
uv run --no-config python Instagram_Liker_Bot.py example_account `
  --max-likes 1 --apply --confirm LIKE_POSTS
```

Credentials are prompted only after confirmation and are not saved. Instagram may change its interface or restrict automation; the live action was not run during local verification.
