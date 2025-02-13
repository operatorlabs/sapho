"""
Main entry point for running the Sapho API server.
"""
import uvicorn
from sapho.api.server import app

def main():
    """Run the API server."""
    uvicorn.run(
        "sapho.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 