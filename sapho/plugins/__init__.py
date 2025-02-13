"""
Sapho.io plugins package
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BasePlugin(ABC):
    """Base class for all Sapho plugins"""
    
    @abstractmethod
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query and return results"""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that the plugin has all required configuration"""
        pass 