"""Grok plugin for advanced reasoning and analysis."""
import os
import aiohttp
from typing import Dict, Any, Type, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from .base import Plugin, PluginQuery, PluginResponse

class GrokQuery(PluginQuery):
    """Query format for Grok."""
    prompt: str = Field(description="The prompt to send to Grok")
    system_prompt: str = Field(
        default="You are an advanced AI assistant focused on research and analysis.",
        description="System prompt to guide Grok's behavior"
    )
    temperature: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Temperature for response generation"
    )

class GrokResponse(PluginResponse):
    """Response format from Grok."""
    response: str = Field(description="Grok's response")
    analysis: List[str] = Field(
        default_factory=list,
        description="Key points from Grok's analysis"
    )

class GrokPlugin(Plugin):
    """Plugin for using Grok's advanced reasoning capabilities."""
    
    def __init__(self):
        load_dotenv()  # Load environment variables
        self.api_key = os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("GROK_API_KEY environment variable is required")
        self.base_url = "https://api.x.ai/v1"
    
    @property
    def name(self) -> str:
        return "grok"
    
    @property
    def query_schema(self) -> Type[PluginQuery]:
        return GrokQuery
    
    @property
    def response_schema(self) -> Type[PluginResponse]:
        return GrokResponse
    
    async def query(self, query: str) -> Dict[str, Any]:
        """Query Grok API."""
        params = GrokQuery.model_validate_json(query)
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": params.system_prompt
                    },
                    {
                        "role": "user",
                        "content": params.prompt
                    }
                ],
                "model": "grok-2-latest",
                "stream": False,
                "temperature": params.temperature
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    raise ValueError(f"Grok API error: {await response.text()}")
                
                result = await response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Extract key points from response
                analysis = [
                    point.strip()
                    for point in response_text.split("\n")
                    if point.strip() and not point.strip().startswith("#")
                ]
                
                return GrokResponse(
                    response=response_text,
                    analysis=analysis
                ).model_dump() 