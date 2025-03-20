"""Main server module for the scikit-learn MCP server."""

from typing import Any, Callable, Coroutine

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .configurations import Settings
from .configure_logger import make_logger
from .tools.decision_tree_classifier_tool import (
    decision_tree_classifier_tool,
    handle_decision_tree_classifier_tool,
)
from .tools.decision_tree_regressor_tool import (
    decision_tree_regressor_tool,
    handle_decision_tree_regressor_tool,
)
from .tools.isolation_forest_tool import handle_isolation_forest_tool, isolation_forest_tool
from .tools.k_means_tool import handle_k_means_tool, k_means_tool
from .tools.k_neighbors_classifier_tool import (
    handle_k_neighbors_classifier_tool,
    k_neighbors_classifier_tool,
)
from .tools.k_neighbors_regressor_tool import (
    handle_k_neighbors_regressor_tool,
    k_neighbors_regressor_tool,
)
from .tools.nearest_neighbors_tool import (
    handle_nearest_neighbors_tool,
    nearest_neighbors_tool,
)
from .tools.one_class_svm_tool import handle_one_class_svm_tool, one_class_svm_tool
from .tools.random_forest_classifier_tool import (
    handle_random_forest_classifier_tool,
    random_forest_classifier_tool,
)
from .tools.random_forest_regressor_tool import (
    handle_random_forest_regressor_tool,
    random_forest_regressor_tool,
)
from .tools.tools import MCPServerScikitLearnTools

logger = make_logger(__name__)

settings = Settings()
server = Server(settings.APP_NAME)


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List available prompts for the scikit-learn MCP server.

    Returns:
        List of available prompts.
    """
    return []


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools for the scikit-learn MCP server.

    Returns:
        List of available tools.
    """
    return [
        random_forest_classifier_tool,
        random_forest_regressor_tool,
        isolation_forest_tool,
        decision_tree_classifier_tool,
        decision_tree_regressor_tool,
        k_neighbors_classifier_tool,
        k_neighbors_regressor_tool,
        nearest_neighbors_tool,
        k_means_tool,
        one_class_svm_tool,
    ]


@server.call_tool()
async def call_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Call a specific tool with the given arguments.

    Args:
        tool_name: Name of the tool to call.
        arguments: Arguments to pass to the tool.

    Returns:
        List of content items returned by the tool.

    Raises:
        ValueError: If the tool name is not recognized.
    """
    tool_handlers: dict[
        str,
        Callable[
            [dict[str, Any]], Coroutine[Any, Any, list[types.TextContent | types.ImageContent | types.EmbeddedResource]]
        ],
    ] = {
        MCPServerScikitLearnTools.RANDOM_FOREST_CLASSIFIER.value: handle_random_forest_classifier_tool,
        MCPServerScikitLearnTools.RANDOM_FOREST_REGRESSOR.value: handle_random_forest_regressor_tool,
        MCPServerScikitLearnTools.ISOLATION_FOREST.value: handle_isolation_forest_tool,
        MCPServerScikitLearnTools.DECISION_TREE_CLASSIFIER.value: handle_decision_tree_classifier_tool,
        MCPServerScikitLearnTools.DECISION_TREE_REGRESSOR.value: handle_decision_tree_regressor_tool,
        MCPServerScikitLearnTools.K_NEIGHBORS_CLASSIFIER.value: handle_k_neighbors_classifier_tool,
        MCPServerScikitLearnTools.K_NEIGHBORS_REGRESSOR.value: handle_k_neighbors_regressor_tool,
        MCPServerScikitLearnTools.NEAREST_NEIGHBORS.value: handle_nearest_neighbors_tool,
        MCPServerScikitLearnTools.K_MEANS.value: handle_k_means_tool,
        MCPServerScikitLearnTools.ONE_CLASS_SVM.value: handle_one_class_svm_tool,
    }

    if tool_name not in tool_handlers:
        raise ValueError(f"Tool {tool_name} not found")

    return await tool_handlers[tool_name](arguments)


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
