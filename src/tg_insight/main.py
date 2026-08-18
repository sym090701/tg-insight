from __future__ import annotations

import argparse
import asyncio
import logging
import os

from . import __version__
from .config import ConfigError, Settings
from .service import TelegramInsightService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tg-insight")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    auth = subcommands.add_parser("auth", help="Authorize the Telegram user session")
    auth.add_argument("--phone", default=os.environ.get("TG_PHONE"))
    subcommands.add_parser("run", help="Run archive, bot and scheduler")
    subcommands.add_parser("check-config", help="Validate environment configuration")
    return parser


async def _run(args: argparse.Namespace, settings: Settings) -> None:
    service = TelegramInsightService(settings)
    if args.command == "auth":
        await service.auth(args.phone)
    elif args.command == "run":
        await service.run()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    if args.command == "check-config":
        print("Configuration is valid.")
        return
    asyncio.run(_run(args, settings))


if __name__ == "__main__":
    main()
