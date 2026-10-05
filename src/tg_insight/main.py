from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from . import __version__
from .config import ConfigError, Settings
from .health import is_healthy
from .service import TelegramInsightService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tg-insight")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    auth = subcommands.add_parser("auth", help="Authorize the Telegram user session")
    auth.add_argument("--phone", default=os.environ.get("TG_PHONE"))
    subcommands.add_parser("run", help="Run archive, bot and scheduler")
    subcommands.add_parser("check-config", help="Validate environment configuration")
    subcommands.add_parser("healthcheck", help="Check whether the running service is ready")
    return parser


async def _run(args: argparse.Namespace, settings: Settings) -> None:
    service = TelegramInsightService(settings)
    if args.command == "auth":
        await service.auth(args.phone)
    elif args.command == "run":
        await service.run()


def main() -> None:
    # Keep app diagnostics useful without enabling verbose protocol dumps from
    # dependencies such as Telethon or httpx. Message bodies and credentials
    # are never logged by the application.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s.%(msecs)03dZ %(levelname)s pid=%(process)d %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.Formatter.converter = time.gmtime
    for name in ("telethon", "httpx", "httpcore", "openai", "aiohttp"):
        logging.getLogger(name).setLevel(logging.WARNING)
    args = build_parser().parse_args()
    if args.command == "healthcheck":
        raise SystemExit(0 if is_healthy() else 1)
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
