# Discord Bot

## Overview

A Discord bot built with the `discord.py` library that provides moderation commands (ban, unban, kick), a welcome message system for new members, and nickname management commands.

**Type:** Bot

## Features

- **Welcome messages:** Automatically sends a welcome message in a configurable channel when a new member joins the server
- **Ban command (`!ban`):** Bans a specified user (requires `ban_members` permission)
- **Unban command (`!unban`):** Unbans a specified user (requires `ban_members` permission)
- **Kick command (`!kick`):** Kicks a specified user (requires `kick_members` permission)
- **Random nickname (`!random_nick` or `!rnick`):** Sets the invoking user's nickname to a random choice from a predefined list
- **Change nickname (`!change_nick` or `!change_name`):** Changes another user's nickname (requires `manage_nicknames` permission)
- Permission-based command access using `@commands.has_permissions` decorators
- Configurable command prefix (default: `!`)

## Dependencies

- `discord.py` — Discord API wrapper
- `random` (Python standard library)

## How It Works

1. The bot is initialized with `commands.Bot(command_prefix="!")`.
2. On startup, `on_ready` prints "bot started" to the console.
3. When a new member joins, `on_member_join` finds the channel named in `WELCOME_CHANNEL` and sends a welcome message mentioning the new member.
4. Moderation commands (`ban`, `unban`, `kick`) use Discord.py's permission checks to ensure the invoking user has the required server permissions.
5. `random_nick` picks a random nickname from the `NICKS` list and applies it to the command invoker.
6. `change_nick` accepts a target user and a new nickname string, applying it to the target.
7. The bot is started with `bot.run(TOKEN)` where `TOKEN` must be set to a valid Discord bot token.

## Project Structure

```
Discord-Bot/
├── main.py     # Main bot script
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed.
2. Install dependencies:
   ```bash
   pip install discord.py
   ```
3. Create a Discord bot application at the [Discord Developer Portal](https://discord.com/developers/applications) and obtain a bot token.
4. Open `main.py` and set the `TOKEN` variable to your bot token.
5. Optionally configure `WELCOME_CHANNEL` and `NICKS` at the top of the file.

## How to Run

```bash
python main.py
```

## Configuration

Configuration is done by editing variables at the top of `main.py`:

| Variable          | Description                                        | Default       |
|-------------------|----------------------------------------------------|---------------|
| `TOKEN`           | Discord bot token (required)                       | `""` (empty)  |
| `WELCOME_CHANNEL` | Name of the channel for welcome messages           | `"welcome"`   |
| `NICKS`           | List of nicknames for the random nickname command   | `["example1", "example2", "example3"]` |
| `command_prefix`  | The prefix for bot commands                        | `"!"`         |

## Testing

No formal test suite present.

## Limitations

- The bot token is stored as a plaintext string in the source code.
- The `NICKS` list is hardcoded and cannot be modified at runtime.
- No error handling for missing welcome channel (will raise an `AttributeError` if the channel doesn't exist).
- The `!kick` command type-hints `discord.User` instead of `discord.Member`, which may cause issues in certain contexts.
- No logging beyond the initial "bot started" print statement.
- No help text customization (uses default discord.py help command).

## Security Notes

- **Bot Token:** The `TOKEN` variable is stored in plaintext in the source code. For production use, store the token in an environment variable or a `.env` file and load it securely (e.g., using `python-dotenv`).
- The bot requires elevated Discord permissions (ban, kick, manage nicknames). Ensure the bot's role is configured with least-privilege access appropriate for your server.
