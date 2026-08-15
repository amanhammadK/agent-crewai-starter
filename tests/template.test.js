import { AgentCrewaiStarterServer } from "../src/mcpServer.js";

describe("AgentCrewaiStarterServer", () => {
    let server;

    beforeEach(() => {
        server = new AgentCrewaiStarterServer();
    });

    test("should initialize server", () => {
        expect(server).toBeDefined();
        expect(server.server).toBeDefined();
    });
});
