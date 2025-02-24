import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Retrieve configuration values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
DEXSCREENER_API_KEY = os.getenv("DEXSCREENER_API_KEY")
DEFAULT_BLOCKCHAIN = os.getenv("DEFAULT_BLOCKCHAIN", "ethereum")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Ensure required variables are set
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment")