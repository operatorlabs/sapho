from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Sapho.io",
    description="Open Deep Research Agent for Crypto",
    version="0.1.0"
)

class ResearchRequest(BaseModel):
    query: str
    breadth: Optional[int] = 4
    depth: Optional[int] = 2
    plugins: Optional[List[str]] = []

class ResearchResponse(BaseModel):
    report: str
    sources: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to Sapho.io Research Agent"}

@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    # Validate API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Perplexity API key not configured")
    
    # TODO: Implement research logic
    # This is a placeholder response
    return ResearchResponse(
        report="Research in progress...",
        sources=[]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 