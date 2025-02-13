"""
Sonar Reasoning Pro integration module.

Includes both the Sonar Reasoning Pro and the Sonar Pro models, which are used for different purposes.
"""
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Union
import json
from pydantic import BaseModel, Field

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 0
    frequency_penalty: float = 1
    response_format: Optional[Dict[str, Any]] = None

class ResponseFormat(BaseModel):
    """Response format specification."""
    type: str
    json_schema: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    """Request parameters for chat completion."""
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(default=0.2, ge=0, lt=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=2048)
    stream: bool = False
    max_tokens: Optional[int] = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=1, gt=0)
    response_format: Optional[ResponseFormat] = None

async def format_structured_output(
    query: str,
    schema: Dict[str, Any],
    api_key: Optional[str] = None
) -> str:
    """
    Use Sonar Pro to format a query according to a schema.
    
    Args:
        query: The query to format
        schema: JSON schema for the expected structure
        api_key: Optional API key for Perplexity
        
    Returns:
        JSON string matching the provided schema
    """
    client = OpenAI(
        api_key=api_key or PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai"
    )
    
    request = ChatCompletionRequest(
        model="sonar-pro",
        messages=[ChatMessage(role="user", content=query)],
        response_format=ResponseFormat(
            type="json_schema",
            json_schema=schema
        )
    )
    
    response = await client.chat.completions.create(
        **request.model_dump(exclude_none=True)
    )
    
    return response.choices[0].message.content

def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI client configured for Perplexity"""
    return OpenAI(
        api_key=api_key or PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai"
    )

def reason(
    query: str,
    system_prompt: Optional[str] = None,
    context: Optional[List[Dict[str, str]]] = None,
    functions: Optional[List[Dict]] = None,
    function_call: Optional[Dict] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Execute a reasoning step using Sonar Reasoning Pro
    
    Args:
        query: The user's query
        system_prompt: Optional system prompt to guide the assistant
        context: Optional list of previous conversation messages
        functions: Optional list of function definitions
        function_call: Optional specification of which function to call
        api_key: Optional API key for Perplexity
    """
    client = create_client(api_key)
    
    messages = [
        {
            "role": "system",
            "content": system_prompt or (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        }
    ]
    
    if context:
        messages.extend(context)
        
    messages.append({
        "role": "user",
        "content": query,
    })

    completion_args = {
        "model": "sonar-reasoning-pro",
        "messages": messages,
    }

    if functions:
        completion_args["functions"] = functions
    if function_call:
        completion_args["function_call"] = function_call

    return client.chat.completions.create(**completion_args)

class SonarReasoner:
    """Perplexity Sonar API client."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )

    async def _complete(self, request: ChatRequest) -> Dict[str, Any]:
        """Make a chat completion request."""
        response = await self.client.chat.completions.create(
            **request.model_dump(exclude_none=True)
        )
        return response.choices[0].message.content

    async def analyze(self, query: str, messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze a query."""
        request = ChatRequest(
            model="sonar-pro",
            messages=[
                ChatMessage(role="user", content=query),
                *(ChatMessage(**msg) for msg in (messages or []))
            ],
            response_format={"type": "json_object"}
        )
        return await self._complete(request)

    async def synthesize(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Synthesize findings."""
        request = ChatRequest(
            model="sonar-pro",
            messages=[ChatMessage(**msg) for msg in messages],
            response_format={"type": "json_object"}
        )
        return await self._complete(request)

def main():
    """Example usage when run directly"""
    response = reason(
        query="How many stars are in the universe?",
    )
    print(response)

if __name__ == "__main__":
    main()

