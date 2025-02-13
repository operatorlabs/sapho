import os
from typing import Any, Dict, Optional
import httpx
from . import BasePlugin

class DexScreenerPlugin(BasePlugin):
    """Plugin for interacting with DexScreener API"""
    
    def __init__(self):
        self.api_key = os.getenv("DEXSCREENER_API_KEY")
        self.base_url = "https://api.dexscreener.com/latest"
    
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using DexScreener API
        
        Args:
            query: The search query
            context: Optional context information
            
        Returns:
            Dict containing the API response
        """
        if not self.validate_config():
            return {"error": "DexScreener API key not configured"}
            
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            try:
                response = await client.get(
                    f"{self.base_url}/dex/search",
                    params={"query": query},
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {"error": f"DexScreener API error: {str(e)}"}
    
    def validate_config(self) -> bool:
        """Check if the API key is configured"""
        return bool(self.api_key) 