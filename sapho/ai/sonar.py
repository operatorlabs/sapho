from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

class SonarReasoner:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or PERPLEXITY_API_KEY
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url="https://api.perplexity.ai"
        )

    def reason(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None
    ):
        """
        Execute a reasoning step using Sonar Reasoning Pro
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt or (
                    "You are an artificial intelligence assistant and you need to "
                    "engage in a helpful, detailed, polite conversation with a user."
                ),
            }
        ]
        
        # Add any context messages if provided
        if context:
            messages.extend(context)
            
        # Add the current query
        messages.append({
            "role": "user",
            "content": query,
        })

        response = self.client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=messages,
        )
        
        return response

def main():
    """Example usage when run directly"""
    reasoner = SonarReasoner()
    response = reasoner.reason(
        query="How many stars are in the universe?",
    )
    print(response)

if __name__ == "__main__":
    main()

