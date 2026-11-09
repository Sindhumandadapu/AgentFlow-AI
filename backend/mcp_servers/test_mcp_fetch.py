
import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():

    server_path = Path(__file__).with_name("search_server.py")

    client = MultiServerMCPClient({
        "duckduckgo": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_path)]
        }
    })

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print("-", tool.name)

    fetch_tool = next(
        tool for tool in tools
        if tool.name == "fetch_page"
    )

    print("\nFetching webpage...")

    result = await fetch_tool.ainvoke({
        "url": "https://www.python.org/"
    })

    print("\nFetched Webpage Content:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
