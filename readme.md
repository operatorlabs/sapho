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

## What is Sapho.io?

Sapho.io is a specialized search engine designed to handle complex cryptocurrency queries, like comparing the impact of regulations on Bitcoin prices. It uses advanced AI to break down questions into manageable steps, ensuring thorough and accurate answers.

### How Does It Work?

Sapho.io operates through a recursive process where each step involves:
- An LLM deciding which tool to use (e.g., web search, price API) based on the current query and previous findings
- Updating a "scratchpad" with new learnings, like "Regulation X passed in January 2023"
- Either generating a new subquery to dig deeper or stopping if the answer is complete

It explores one path at a time (breadth = 1) and can stop early if it has enough data, or continue until it hits a maximum depth, then synthesize a final answer from all learnings.

### System Architecture

The architecture of Sapho.io is characterized by an improvisational, depth-focused process with a breadth of 1, meaning it explores one path deeply before moving to another. Each step involves a single tool call, dynamically decided by an LLM that examines a scratchpad's updated "learnings" to produce a result. Recursion continues until the system reaches a predefined maximum depth (max_depth), at which point it iterates over the scratchpad to synthesize a final result.

### Key Components

1. Initialization
- Begins with an initial query and maximum depth setting
- Sets up a scratchpad to track progress and store learnings
- Prepares necessary tools and APIs for data gathering

2. Research Function
- Makes recursive calls based on depth and current findings
- Decides which tools to use at each step
- Updates the scratchpad with new information
- Generates subqueries when needed

3. Tool Execution
- Integrates with various data sources (SERP, price APIs, etc.)
- Processes raw data into structured learnings
- Maintains modularity for easy tool addition

4. Scratchpad Management
- Tracks query progress and depth
- Stores learnings hierarchically
- Enables backtracking and synthesis

5. Early Stopping Mechanism
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

This architecture enables Sapho.io to handle complex queries with precision while maintaining flexibility and efficiency in its research process.

## Support

- Join our [Discord community](https://discord.gg/FQagDmCkvC)
- Check the #support channel for common issues
- Ask in #general for specific questions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

