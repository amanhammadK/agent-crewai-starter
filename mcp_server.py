"""MCP server exposing the CrewAI crew as a tool via the Python MCP SDK."""
import os
import asyncio
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

server = Server("agent-crewai-starter")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_crew",
            description="Run the research+write crew on a topic and return a report",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic for the crew to research and report on",
                    }
                },
                "required": ["topic"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "run_crew":
        raise ValueError(f"Unknown tool: {name}")
    # Imported lazily so the server starts even without crewai installed
    from main import run

    topic = arguments.get("topic", "autonomous AI agents")
    report = run(topic)
    return [types.TextContent(type="text", text=report)]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="agent-crewai-starter",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
