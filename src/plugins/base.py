from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from pydantic import BaseModel, ConfigDict

class PluginQuery(BaseModel):
    """Base schema for plugin queries."""
    model_config = ConfigDict(extra="forbid")

class PluginResponse(BaseModel):
    """Base schema for plugin responses."""
    model_config = ConfigDict(extra="forbid")

class Plugin(ABC):
    """Base plugin interface."""
    
    @classmethod
    @abstractmethod
    def plugin_name(cls) -> str:
        """Plugin identifier."""
        pass
    
    @property
    @abstractmethod
    def query_schema(self) -> Type[PluginQuery]:
        """Expected query format."""
        pass
    
    @property
    @abstractmethod
    def response_schema(self) -> Type[PluginResponse]:
        """Response data format."""
        pass
    
    @abstractmethod
    async def query(self, query: str) -> Dict[str, Any]:
        """Execute plugin query."""
        pass 