from mcp.types import Prompt

from ..configure_logger import make_logger
from .prompts import PROMPTS

logger = make_logger(__name__)


async def list_prompts() -> list[Prompt]:
    return list(PROMPTS.values())
