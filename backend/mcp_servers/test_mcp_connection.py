
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

    # Discover tools exposed by the MCP server
    tools = await client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # Find our web search tool
    search_tool = next(
        tool for tool in tools
        if tool.name == "search_web"
    )

    # Execute a real search
    print("\nExecuting web search...")

    results = await search_tool.ainvoke({
        "query": "AI companies hiring freshers in India 2026",
        "max_results": 3
    })

    print("\nSearch Results:")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
