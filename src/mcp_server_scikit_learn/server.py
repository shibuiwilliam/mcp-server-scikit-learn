from typing import Any

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Prompt

from .configurations import Settings
from .configure_logger import make_logger
from .prompts.handlers import list_prompts as handler_list_prompts
from .tools.isolation_forest_tool import handle_isolation_forest_tool, isolation_forest_tool
from .tools.random_forest_classifier_tool import handle_random_forest_classifier_tool, random_forest_classifier_tool
from .tools.tools import MCPServerScikitLearnTools

settings = Settings()
logger = make_logger(__name__)
server = Server(settings.APP_NAME)


@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts."""
    return await handler_list_prompts()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [isolation_forest_tool, random_forest_classifier_tool]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    try:
        match name:
            case MCPServerScikitLearnTools.ISOLATION_FOREST.value:
                return await handle_isolation_forest_tool(arguments)
            case MCPServerScikitLearnTools.RANDOM_FOREST_CLASSIFIER.value:
                return await handle_random_forest_classifier_tool(arguments)
            case _:
                return [types.TextContent(type="text", text=f"Error: Unknown tool {name}")]
    except Exception as e:
        logger.error(f"Tool error: {str(e)}")
        return [types.TextContent(type="text", text=f"Error {name}: {e}")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=settings.APP_NAME,
                server_version=settings.APP_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(resources_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )
