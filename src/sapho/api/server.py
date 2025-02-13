"""
FastAPI application exposing the Sapho research functionality.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from sapho.core.agent import ResearchAgent, ResearchRequest

load_dotenv()

app = FastAPI(
    title="Sapho Research API",
    description="Deep Research Agent for Crypto",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize research agent
api_key = os.getenv("PERPLEXITY_API_KEY")
if not api_key:
    raise ValueError("PERPLEXITY_API_KEY environment variable is required")

# Initialize with Grok plugin
agent = ResearchAgent(api_key=api_key)

@app.post("/research")
async def research(request: ResearchRequest):
    """Execute research based on query and parameters."""
    try:
        result = await agent.research(request)
        return {
            "learnings": result.learnings,
            "research_paths": result.research_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 