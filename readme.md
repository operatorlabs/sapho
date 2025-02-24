# Sapho - Deep Research Agent for Crypto

<div align="center">
  <img 
    src="https://cc-client-assets.nyc3.cdn.digitaloceanspaces.com/photo/categoryonegames/file/cd68275759e2483b84a5450356f7b499/large/juiceofsaphomentatdrug-250x363.jpg" 
    alt="Sapho Juice from Dune" 
    width="200px" 
    style="border-radius: 8px; margin: 20px 0 10px 0;"
  />
  <p style="font-style: italic; margin-bottom: 20px;">
    Sapho Juice - The mystical substance used by Mentats to enhance their cognitive abilities
  </p>
</div>

## Overview

[Sapho](https://github.com/operator-labs/sapho) is a research agent built by [Operator Labs](https://operator.io). It is designed to buff up the core reasoner loop for any AI agent, and is designed to interact well with extensible plugins like [Grok](https://x.ai). [mentat.fun](https://mentat.fun) uses Sapho as its core research agent. 

Thanks to [dzhng](https://github.com/dzhng) for the original [deep-research](https://github.com/dzhng/deep-research) repo, which inspired this project. 

### How Does It Work?

Sapho.io operates through a recursive process where each step involves:
- An LLM deciding which tool to use (e.g., web search, price API) based on the current query and previous findings
- Updating a "scratchpad" with new learnings, like "Regulation X passed in January 2023"
- Either generating a new subquery to dig deeper or stopping if the answer is complete

It explores one path at a time (breadth = 1) and can stop early if it has enough data, or continue until it hits a maximum depth, then synthesize a final answer from all learnings.

The architecture of Sapho.io is characterized by an improvisational, depth-focused process, meaning it explores one path deeply before moving to another. Each step involves a single tool call, dynamically decided by an LLM that examines a scratchpad's updated "learnings" to decide which tool to use. Recursion continues until the system reaches a predefined maximum depth (max_depth), at which point it iterates over the scratchpad to synthesize a final result.

### Codebase

```
sapho/
├── .env                   # Environment variables (not committed to Git)
├── .env.example           # Example env file (for documentation)
├── src/                 # Python package for Sapho AI agent
│   ├── __init__.py
│   ├── tools.py           # Implements external tool calls
│   ├── agent.py           # Implements agent orchestration
├── baml_src/              # BAML files defining reasoning logic
│   ├── main.baml          # Defines AI agent’s reasoning logic
│   ├── tools.baml         # Defines available tools
├── baml_client/           # Auto-generated Python client (via BAML CLI)
│   ├── types/             # Generated Pydantic models for BAML types
│   ├── client.py          # Interface for calling BAML functions
├── requirements.txt       # Python dependencies
├── main.py                # Entry point to run the agent
└── README.md              # Documentation
```

### Process

1. Initialization
- Begins with an initial query and maximum depth setting
- Sets up a scratchpad to track progress and store learnings
- Prepares necessary tools and APIs for data gathering

2. Research Function
- Makes recursive calls based on depth and current findings
- Decides which tools to use at each step
- Updates the scratchpad with new information
- Generates subqueries when needed

3. Tool Search and Selection
- Dynamically discovers relevant tools based on query context
- Optimizes memory by loading only necessary tools
- Ranks tools by relevance to current research path
- Supports extensible plugin ecosystem

4. Tool Execution
- Integrates with various data sources (SERP, price APIs, etc.)
- Processes raw data into structured learnings
- Maintains modularity for easy tool addition

5. Scratchpad Management
- Maintains raw text format for maximum flexibility and human readability
- Allows natural language updates and annotations
- Preserves context and nuance 
- Enables easy manual review and editing when needed
- Supports both structured and unstructured information

6. Early Stopping Mechanism
- Evaluates answer completeness
- Can terminate before maximum depth
- Optimizes for efficiency

### Example Workflow

Consider a query about regulation impacts on Bitcoin prices:

Step 1 (Depth 0):
- Initial Query: "Compare the impact of Regulation X and Y on Bitcoin prices"
- Tool: Web search for Regulation X
- Learning: "Regulation X passed in January 2023"

Step 2 (Depth 1):
- Subquery: "Get Regulation Y details"
- Tool: Web search for Regulation Y
- Learning: "Regulation Y passed in February 2023"

Step 3 (Depth 2):
- Subquery: "Get price data"
- Tool: Price API
- Learning: "X caused 5% drop, Y caused 3% drop"

Final Step:
- Synthesizes all learnings
- Produces comprehensive comparison
- Returns structured response

This architecture enables Sapho to handle complex queries with precision while maintaining flexibility and efficiency in its research process.

## Deployment

Sapho is deployable in two ways:

### 1. As a Model Context Protocol (MCP) Server

You can run Sapho as an MCP server to integrate with Claude Desktop and other MCP clients. This allows direct interaction with the research agent for users who want to use Sapho as part of their workflow. 

For installation instructions and documentation, visit:
https://opentools.com/registry/sapho

### 2. As a standalone FastAPI server

You can run Sapho as a standalone server, which can be useful for development and testing purposes. Sapho is deployable as a standalone Docker container, which wraps a FastAPI server around the agent. This service is able to handle multiple concurrent requests, using FastAPI's task-based concurrency model.

## Support

- Join our [Discord community](https://discord.gg/FQagDmCkvC)
- Check the #support channel for common issues
- Ask in #general for specific questions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

