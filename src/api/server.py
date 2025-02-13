"""
FastAPI application exposing the Sapho research functionality.

The server provides endpoints for:
1. Executing deep research queries
2. Health monitoring
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from core.agent import ResearchAgent, ResearchRequest
from __init__ import __version__

# API Models
class APIResearchRequest(BaseModel):
    """Research request parameters."""
    query: str = Field(..., description="The research topic or question")
    depth: int = Field(default=2, ge=1, le=5, description="How many levels deep to explore")
    breadth: int = Field(default=4, ge=1, le=8, description="How many parallel queries at each level")

class APIResearchResponse(BaseModel):
    """Research results."""
    learnings: List[str] = Field(..., description="Key findings from the research")
    plugins_used: List[str] = Field(..., description="Plugins that contributed to the research")
    research_paths: List[Dict[str, Any]] = Field(..., description="Detailed research exploration paths")

# Initialize FastAPI app
app = FastAPI(
    title="Sapho Research API",
    description="Deep Research Agent for Crypto",
    version=__version__
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
agent = ResearchAgent()

@app.post("/research", response_model=APIResearchResponse)
async def research(request: APIResearchRequest) -> APIResearchResponse:
    """Execute deep research on a topic."""
    try:
        # Convert API request to internal request
        internal_request = ResearchRequest(
            query=request.query,
            depth=request.depth,
            breadth=request.breadth
        )
        
        # Execute research
        result = await agent.research(internal_request)
        
        # Return results
        return APIResearchResponse(
            learnings=result.learnings,
            plugins_used=result.plugins_used,
            research_paths=result.research_paths
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    ) 