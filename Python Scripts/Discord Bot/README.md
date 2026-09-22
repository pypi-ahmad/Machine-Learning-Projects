# Discord Bot

A Discord moderation bot with welcome messages and nickname commands. It uses the prefix `!` and performs actions only in servers, not direct messages.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- A Discord bot application and token

## Configuration

Set the token only in the process environment; do not add it to source code or commit it to the repository.

```powershell
$env:DISCORD_BOT_TOKEN = "your-bot-token"
```

In the Discord Developer Portal, enable **Message Content Intent** for prefix commands and **Server Members Intent** for the member-join welcome event. Discord.py requires Message Content Intent both in code and in the portal for commands to receive message text. [discord.py command documentation](https://discordpy.readthedocs.io/en/stable/ext/commands/commands.html)

If you configured the variable after launching your coding host, restart that host before running the bot.

## Run

From this directory:

```powershell
uv sync
uv run python main.py
```

## Commands

- `!ban @member` — ban a member; requires `ban_members`.
- `!unban user` — unban a user; requires `ban_members`.
- `!kick @member` — kick a member; requires `kick_members`.
- `!random_nick` or `!rnick` — set your nickname to an option from `NICKS`.
- `!change_nick @member new name` or `!change_name` — change a nickname; requires `manage_nicknames`.

Set `WELCOME_CHANNEL` and `NICKS` in `main.py` to suit the server. The bot reports a missing welcome channel instead of trying to send through `None`.

## Security and moderation notes

- Give the bot only the Discord permissions it needs. Its role must be positioned above roles it will moderate.
- The bot is capable of banning, kicking, and changing nicknames. Test it in a non-production server first.
- Token values are never logged by this project.

## Project files

```text
main.py         # Discord bot entry point
pyproject.toml  # uv dependency definition
uv.lock         # Resolved dependency versions
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
