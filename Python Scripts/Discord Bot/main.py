"""Discord moderation bot with welcome and nickname commands."""

import os
import random

import discord
from discord.ext import commands

TOKEN_VARIABLE = "DISCORD_BOT_TOKEN"
WELCOME_CHANNEL = "welcome"
NICKS = ["example1", "example2", "example3"]

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
)


@bot.event
async def on_ready() -> None:
    """Report the authenticated bot identity after a successful login."""
    print(f"Bot started as {bot.user}.")


@bot.event
async def on_member_join(member: discord.Member) -> None:
    """Send a welcome message only when the configured channel exists."""
    welcome_channel = discord.utils.get(member.guild.text_channels, name=WELCOME_CHANNEL)
    if welcome_channel is None:
        print(f"Welcome channel '{WELCOME_CHANNEL}' was not found in {member.guild.name}.")
        return
    await welcome_channel.send(
        f"Welcome {member.mention}; please read the server rules."
    )


@bot.command()
@commands.guild_only()
@commands.has_permissions(ban_members=True)
async def ban(ctx: commands.Context, user: discord.Member) -> None:
    """Ban a server member."""
    await user.ban(reason=f"Requested by {ctx.author}")
    await ctx.send(f"Banned {user}.")


@bot.command()
@commands.guild_only()
@commands.has_permissions(ban_members=True)
async def unban(ctx: commands.Context, user: discord.User) -> None:
    """Unban a Discord user from the current server."""
    await ctx.guild.unban(user, reason=f"Requested by {ctx.author}")
    await ctx.send(f"Unbanned {user}.")


@bot.command()
@commands.guild_only()
@commands.has_permissions(kick_members=True)
async def kick(ctx: commands.Context, user: discord.Member) -> None:
    """Kick a server member."""
    await user.kick(reason=f"Requested by {ctx.author}")
    await ctx.send(f"Kicked {user}.")


@bot.command(aliases=["rnick"])
@commands.guild_only()
async def random_nick(ctx: commands.Context) -> None:
    """Set the command author's nickname to a configured random option."""
    new_nick = random.choice(NICKS)
    await ctx.author.edit(nick=new_nick, reason="Random nickname command")
    await ctx.send(f"Your new nickname is {new_nick}.")


@bot.command(aliases=["change_name"])
@commands.guild_only()
@commands.has_permissions(manage_nicknames=True)
async def change_nick(ctx: commands.Context, user: discord.Member, *, new_nick: str) -> None:
    """Change a server member's nickname."""
    await user.edit(nick=new_nick, reason=f"Requested by {ctx.author}")
    await ctx.send(f"Changed the nickname of {user.mention} to `{new_nick}`.")


def start_bot(token: str | None) -> int:
    """Start the gateway client when a token was supplied."""
    if not token:
        print(f"Set {TOKEN_VARIABLE} before starting the bot.")
        return 1
    bot.run(token)
    return 0


def main() -> int:
    """Read the bot token from the runtime environment."""
    return start_bot(os.environ.get(TOKEN_VARIABLE))


if __name__ == "__main__":
    raise SystemExit(main())
