"""测试用 MCP 服务（stdio）。"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """回显文本。"""
    return f"echo: {text}"


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def wipe() -> str:
    """危险操作（测试写操作标记）。"""
    return "wiped"


if __name__ == "__main__":
    mcp.run()
