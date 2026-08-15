import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { spawn } from "child_process";

const runningTasks = new Map();
let taskCounter = 0;

function runPythonScript(scriptPath, args = []) {
    return new Promise((resolve, reject) => {
        const proc = spawn("python3", [scriptPath, ...args], {
            cwd: process.cwd(),
            env: { ...process.env },
            stdio: ["pipe", "pipe", "pipe"],
        });

        let stdout = "";
        let stderr = "";
        proc.stdout.on("data", (d) => (stdout += d.toString()));
        proc.stderr.on("data", (d) => (stderr += d.toString()));
        proc.on("close", (code) => {
            if (code === 0) resolve(stdout.trim());
            else reject(new Error(stderr || `Python exited with code ${code}`));
        });
        proc.on("error", reject);
    });
}

export class AgentCrewaiStarterServer {
    constructor() {
        this.server = new McpServer({
            name: "agent-crewai-starter",
            version: "0.1.0",
        });
        this.setupTools();
    }

    setupTools() {
        this.server.tool(
            "run_research",
            "Trigger a CrewAI research+write crew task on a given topic",
            { topic: z.string().min(1, "Topic is required") },
            async (args) => {
                const taskId = `crew-${++taskCounter}`;
                runningTasks.set(taskId, {
                    status: "running",
                    topic: args.topic,
                    startedAt: new Date().toISOString(),
                });

                try {
                    const report = await runPythonScript("main.py", [
                        args.topic,
                    ]);
                    runningTasks.set(taskId, {
                        status: "completed",
                        topic: args.topic,
                        startedAt: runningTasks.get(taskId).startedAt,
                        completedAt: new Date().toISOString(),
                        result: report,
                    });
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify(
                                    {
                                        task_id: taskId,
                                        status: "completed",
                                        topic: args.topic,
                                        report,
                                    },
                                    null,
                                    2,
                                ),
                            },
                        ],
                    };
                } catch (error) {
                    runningTasks.set(taskId, {
                        status: "failed",
                        topic: args.topic,
                        error: error.message,
                    });
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify(
                                    {
                                        task_id: taskId,
                                        status: "failed",
                                        error: error.message,
                                    },
                                    null,
                                    2,
                                ),
                            },
                        ],
                    };
                }
            },
        );

        this.server.tool(
            "get_status",
            "Check if a crew task is running, completed, or failed",
            { task_id: z.string().min(1, "Task ID is required") },
            async (args) => {
                const task = runningTasks.get(args.task_id);
                if (!task) {
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify(
                                    {
                                        error: `No task found with ID: ${args.task_id}`,
                                    },
                                    null,
                                    2,
                                ),
                            },
                        ],
                    };
                }
                return {
                    content: [
                        {
                            type: "text",
                            text: JSON.stringify(task, null, 2),
                        },
                    ],
                };
            },
        );

        this.server.tool(
            "list_agents",
            "List available crew agents and their roles",
            {},
            async () => {
                const agents = [
                    {
                        name: "Senior Research Analyst",
                        role: "Research",
                        description:
                            "Meticulous analyst who verifies facts and cites sources. Synthesizes complex topics into clear briefing notes.",
                        tools: ["web_search"],
                    },
                    {
                        name: "Technical Writer",
                        role: "Writing",
                        description:
                            "Writes for busy executives. Leads with conclusions, uses bullets, and avoids jargon.",
                        tools: [],
                    },
                ];
                return {
                    content: [
                        {
                            type: "text",
                            text: JSON.stringify({ agents }, null, 2),
                        },
                    ],
                };
            },
        );
    }

    async run() {
        const { StdioServerTransport } = await import(
            "@modelcontextprotocol/sdk/server/stdio.js"
        );
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.log("Agent-CrewAI MCP Server running on stdio");
    }
}
