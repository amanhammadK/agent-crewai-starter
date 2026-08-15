# agent-crewai-starter

A working [CrewAI](https://crewai.com) multi-agent starter, wrapped as a Model Context Protocol server. Two agents — a **Research Analyst** and a **Technical Writer** — run sequentially to produce an executive report on any topic.

This is a real starter: `main.py` actually builds and kicks off a CrewAI `Crew` against an OpenAI-compatible model, and `mcp_server.py` exposes that crew as an MCP tool over stdio.

## Why this exists

Most "agent starters" print a placeholder. This one runs a genuine two-agent pipeline and is wired to MCP so it can be invoked by any MCP client (Claude Desktop, custom orchestrators).

## Install

```bash
pip install -r requirements.txt
```

## Configure

```env
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

## Run the crew directly

```bash
python main.py "the future of autonomous AI agents"
```

## Run as an MCP server (stdio)

```bash
python mcp_server.py
```

## Customize

- Edit `build_crew()` in `main.py` to add agents, tools, or tasks.
- Swap the `web_search` tool for a real search provider (SerpAPI, Tavily) in production.
