"""
Main entry point for running Sapho.
Can be run as an API server or directly as a research agent.
"""
import uvicorn
import asyncio
import argparse
from api.server import app
from core.agent import ResearchAgent, ResearchRequest, ResearchProgress

async def run_research(query: str, depth: int = 2, breadth: int = 4):
    """Run research directly with progress tracking."""
    agent = ResearchAgent()
    
    def progress_callback(progress: ResearchProgress):
        """Print progress updates."""
        print("\n" + "=" * 80)
        print(f"Current Query: {progress.current_query}")
        print(f"Progress: {progress.completed_queries}/{progress.total_queries} queries completed")
        print(f"Depth: Level {progress.current_depth}/{progress.total_depth}")
        print(f"Breadth: {progress.current_breadth}/{progress.total_breadth} parallel paths")
        print("=" * 80)
    
    request = ResearchRequest(
        query=query,
        depth=depth,
        breadth=breadth
    )
    
    print(f"\nStarting research with depth={depth}, breadth={breadth}")
    result = await agent.research(request, on_progress=progress_callback)
    
    print("\nResearch Results:")
    print("=" * 80)
    print("\nKey Learnings:")
    for i, learning in enumerate(result.learnings, 1):
        print(f"{i}. {learning}")
    
    print("\nPlugins Used:")
    for plugin in result.plugins_used:
        print(f"- {plugin}")
    
    print("\nResearch Paths:")
    def print_path(path, indent=0):
        print("  " * indent + f"Query: {path['query']}")
        if 'learnings' in path:
            for learning in path['learnings']:
                print("  " * (indent + 1) + f"→ {learning}")
        if 'sub_queries' in path:
            for sub in path['sub_queries']:
                print_path(sub, indent + 2)
    
    for path in result.research_paths:
        print_path(path)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sapho Research Agent")
    parser.add_argument("--mode", choices=["api", "research"], default="api",
                      help="Run as API server or direct research")
    parser.add_argument("--query", help="Research query when in research mode")
    parser.add_argument("--depth", type=int, default=2, help="Research depth")
    parser.add_argument("--breadth", type=int, default=4, help="Research breadth")
    args = parser.parse_args()
    
    if args.mode == "api":
        uvicorn.run(
            "api.server:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    else:
        if not args.query:
            print("Error: --query is required in research mode")
            return
        asyncio.run(run_research(args.query, args.depth, args.breadth))

if __name__ == "__main__":
    main() 