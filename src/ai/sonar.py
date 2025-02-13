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
import asyncio
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    type: str = "text"

class JSONSchemaFormat(BaseModel):
    """JSON Schema response format."""
    type: str = "json_schema"
    json_schema: Dict[str, Any]

class ChatCompletionRequest(BaseModel):
    """Request parameters for chat completion."""
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(default=0.2, ge=0, lt=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: Optional[int] = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=1, gt=0)
    stream: bool = Field(default=False)
    response_format: Optional[Union[ResponseFormat, JSONSchemaFormat]] = None

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
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )
        self.logger = logging.getLogger(__name__)

    async def format_structured_output(
        self,
        query: str,
        schema: Dict[str, Any],
    ) -> str:
        """
        Use Sonar Pro to format a query according to a schema.
        """
        try:
            self.logger.info("Making format_structured_output request:")
            self.logger.info(f"Model: sonar-pro")
            self.logger.info(f"Query length: {len(query)} chars")
            self.logger.info(f"Schema: {json.dumps(schema, indent=2)}")
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="sonar-pro",
                messages=[{
                    "role": "user", 
                    "content": f"Format your response as JSON. Here is the required schema: {json.dumps(schema)}\n\nQuery: {query}"
                }],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": schema
                    }
                }
            )
            
            self.logger.info("Received response:")
            self.logger.info(f"Response status: success")
            self.logger.info(f"Model used: {response.model}")
            self.logger.info(f"Response length: {len(response.choices[0].message.content)} chars")
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error("Error in format_structured_output:")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def reason(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Dict] = None,
    ) -> dict:
        """Execute a reasoning step using Sonar Reasoning Pro."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt or (
                        "You are an artificial intelligence assistant and you need to "
                        "engage in a helpful, detailed, polite conversation with a user."
                    ),
                },
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

            self.logger.info("Making reason request:")
            self.logger.info(f"Model: sonar-reasoning-pro")
            self.logger.info(f"Query length: {len(query)} chars")
            self.logger.info(f"System prompt length: {len(system_prompt) if system_prompt else 0} chars")
            self.logger.info(f"Context messages: {len(context) if context else 0}")
            self.logger.info(f"Functions: {len(functions) if functions else 0}")
            self.logger.info(f"Function call: {json.dumps(function_call) if function_call else None}")

            # OpenAI's create() is synchronous, wrap in asyncio.to_thread
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **completion_args
            )
            
            self.logger.info("Received response:")
            self.logger.info(f"Response status: success")
            self.logger.info(f"Model used: {response.model}")
            self.logger.info(f"Response length: {len(response.choices[0].message.content)} chars")
            
            return response

        except Exception as e:
            self.logger.error("Error in reason:")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def _complete(self, request: ChatRequest) -> str:
        """Make a chat completion request."""
        try:
            self.logger.info("Making _complete request:")
            self.logger.info(f"Request: {json.dumps(request.model_dump(), indent=2)}")
            
            response = await self.client.chat.completions.create(
                **request.model_dump(exclude_none=True)
            )
            
            self.logger.info("Received response:")
            self.logger.info(f"Response status: success")
            self.logger.info(f"Model used: {response.model}")
            self.logger.info(f"Response length: {len(response.choices[0].message.content)} chars")
            
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error("Error in _complete:")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

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

