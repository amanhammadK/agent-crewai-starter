import { AgentCrewaiStarterServer } from "./mcpServer.js";

const server = new AgentCrewaiStarterServer();
server.run().catch((error) => {
    console.error("Server error:", error);
    process.exit(1);
});
