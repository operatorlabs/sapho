"""
Plugin system initialization and management.

All plugins in this directory are loaded by default.
"""
import os
import importlib
import inspect
from typing import Dict, Type, Optional
from .base import Plugin

def load_plugins() -> Dict[str, Type[Plugin]]:
    """Load all plugins from the plugins directory."""
    plugins = {}
    
    # Get the current directory
    plugin_dir = os.path.dirname(__file__)
    
    # Iterate through all python files in the directory
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            # Import the module
            module_name = filename[:-3]  # Remove .py extension
            module = importlib.import_module(f'.{module_name}', package='plugins')
            
            # Find plugin classes in the module
            for _, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin):
                    try:
                        # Get plugin name without instantiating
                        plugins[obj.plugin_name()] = obj
                    except Exception:
                        # Skip plugins that can't be loaded
                        continue
    
    return plugins

def get_plugin(name: str) -> Optional[Type[Plugin]]:
    """Get a plugin by name."""
    return load_plugins().get(name)

# Export the plugin loader
__all__ = ['load_plugins', 'get_plugin'] 