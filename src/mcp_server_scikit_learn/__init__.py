import asyncio

from . import prompts, server, tools, utils


def main():
    asyncio.run(server.main())


__all__ = ["main", "server", "utils", "tools", "prompts"]
