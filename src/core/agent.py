"""Research agent using Perplexity Sonar API."""
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from ai.sonar import SonarReasoner, ChatMessage, ChatRequest
from plugins import load_plugins
from plugins.base import Plugin
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchProgress(BaseModel):
    """Research progress tracking."""
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: Optional[str] = None
    total_queries: int = 0
    completed_queries: int = 0
    current_action: Optional[str] = None
    current_state: Optional[Dict[str, Any]] = None

class ResearchState(BaseModel):
    """Research state - acts as a notepad during research."""
    learnings: List[str] = Field(default_factory=list)
    plugins_used: List[str] = Field(default_factory=list)
    query_history: List[Dict[str, Any]] = Field(default_factory=list)
    context_notes: List[str] = Field(default_factory=list)
    research_paths: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_traces: List[Dict[str, Any]] = Field(default_factory=list)

    def add_reasoning_trace(self, query: str, response: str, result: Any, trace_type: str):
        """Add a reasoning trace with timestamp."""
        self.reasoning_traces.append({
            "timestamp": datetime.now().isoformat(),
            "type": trace_type,
            "query": query,
            "response": response,
            "result": result
        })

    def summarize(self) -> Dict[str, Any]:
        """Get a summary of current research state."""
        return {
            "total_learnings": len(self.learnings),
            "plugins_used": self.plugins_used,
            "queries_processed": len(self.query_history),
            "context_notes": len(self.context_notes),
            "research_paths": len(self.research_paths),
            "recent_learnings": self.learnings[-3:] if self.learnings else [],
            "reasoning_traces": len(self.reasoning_traces)
        }

    def generate_research_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the research process."""
        return {
            "overview": {
                "total_queries": len(self.query_history),
                "total_learnings": len(self.learnings),
                "plugins_used": self.plugins_used,
                "total_reasoning_steps": len(self.reasoning_traces)
            },
            "key_findings": self.learnings,
            "research_path": [{
                "query": entry["query"],
                "learnings": entry.get("learnings", []),
                "plugins": entry.get("plugins_used", [])
            } for entry in self.query_history],
            "reasoning_process": [{
                "step": i + 1,
                "type": trace["type"],
                "query": trace["query"],
                "result": trace["result"]
            } for i, trace in enumerate(self.reasoning_traces)],
            "context_notes": self.context_notes
        }

class ResearchResult(BaseModel):
    """Final research results."""
    learnings: List[str]
    plugins_used: List[str]
    research_paths: List[Dict[str, Any]]
    summary: Dict[str, Any]

class ResearchRequest(BaseModel):
    """Research request."""
    query: str
    depth: int = 2
    breadth: int = 4

class ResearchAgent:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResearchAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ResearchAgent._initialized:
            """Initialize agent with all available plugins."""
            self.reasoner = SonarReasoner()  # Sonar handles its own API key
            self.plugins = {}  # All available plugins
            self.logger = logging.getLogger(__name__)
            
            # Load and initialize all plugins upfront
            available_plugins = load_plugins()
            for name, plugin_class in available_plugins.items():
                try:
                    plugin = plugin_class()  # Each plugin handles its own API key
                    self.plugins[name] = plugin
                    self.logger.info(f"Initialized plugin: {name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize plugin {name}: {e}")
            ResearchAgent._initialized = True

    async def _should_use_plugins(self, query: str) -> bool:
        """Evaluate if any plugins should be used for this query."""
        analysis = await self.reasoner.reason(
            query=f"""Given this research query: {query}

Available plugins:
{chr(10).join(f'- {name}: {plugin.query_schema.model_json_schema()}' for name, plugin in self.plugins.items())}

The core reasoning model already has access to general internet information, EXCEPT for:
- Closed/authenticated networks (e.g. Twitter, Reddit)
- Sites that block scrapers
- Real-time API data that requires authentication
- Specialized databases requiring API keys

Should we use plugins to access such restricted data sources for this query, or would the core reasoner's general internet access be sufficient? Consider:
1. Does the query require data from closed networks or authenticated APIs?
2. Is real-time or very recent data essential?
3. Can this be answered well using publicly accessible internet information?

Return PLUGINS if we need access to restricted data sources, or REASONING if general internet access is sufficient.""",
            system_prompt="You are a research assistant with general internet access. Evaluate if restricted data sources would be valuable for this query."
        )
        
        return "PLUGINS" in analysis.choices[0].message.content.upper()

    async def _evaluate_plugin_relevance(
        self,
        query: str,
        plugin_name: str,
        plugin: Plugin
    ) -> bool:
        """Evaluate if a specific plugin is relevant for the current query."""
        analysis = await self.reasoner.reason(
            query=f"""Given this research query: {query}

Plugin capability:
{plugin_name}: {plugin.query_schema.model_json_schema()}

The core reasoning model already has access to general internet information. This plugin provides access to restricted data sources.

Would this specific plugin's access to restricted data be useful? Consider:
1. Would the data from this plugin significantly improve the answer beyond what's publicly available?
2. Is the specialized data type from this plugin essential for this question?

Return a clear YES or NO.""",
            system_prompt="You are a research assistant with general internet access. Evaluate if this specific restricted data source would be helpful."
        )
        
        return "YES" in analysis.choices[0].message.content.upper()

    async def _process_long_query(
        self,
        query: str,
        schema: Dict[str, Any],
        max_chars: int = 1500,
        progress: Optional[ResearchProgress] = None,
        state: Optional[ResearchState] = None
    ) -> str:
        """Process a query by requesting a direct, structured response."""
        try:
            # Log the attempt
            self.logger.info(f"Processing query ({len(query)} chars)")
            self.logger.info("Query content:")
            self.logger.info("=" * 40)
            self.logger.info(query)
            self.logger.info("=" * 40)
            
            # Make a single, direct request with clear constraints
            response = await self.reasoner.reason(
                query=f"""Analyze this and provide a structured response:
{query}

IMPORTANT: Your response must:
1. Be under {max_chars} characters
2. Match this schema exactly: {json.dumps(schema, indent=2)}
3. Focus on key points only""",
                system_prompt="You are a research assistant. Provide concise, structured responses that fit the specified schema."
            )
            
            response_content = response.choices[0].message.content
            self.logger.info(f"Got response ({len(response_content)} chars)")
            
            # Try to format it
            try:
                return await self.reasoner.format_structured_output(
                    query=response_content,
                    schema=schema
                )
            except Exception as e:
                self.logger.error(f"Failed to format response: {str(e)}")
                self.logger.error("Response content:")
                self.logger.error(response_content)
                raise
                
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Original query ({len(query)} chars):")
            self.logger.error(query[:1000] + "..." if len(query) > 1000 else query)
            raise

    async def _generate_queries(
        self,
        query: str,
        num_queries: int,
        state: ResearchState,
        progress: Optional[ResearchProgress] = None
    ) -> List[Dict[str, str]]:
        """Generate follow-up queries using state context."""
        if progress:
            self._report_action(
                progress,
                "Generating follow-up queries",
                state
            )
            
        # Build context from recent history and notes
        context = []
        if state.query_history:
            recent_history = state.query_history[-3:]  # Last 3 queries
            context.append("Recent research path:")
            for entry in recent_history:
                context.append(f"Query: {entry['query']}")
                context.append(f"Findings: {', '.join(entry['learnings'])}")
        
        if state.context_notes:
            context.append("\nResearch context:")
            context.extend(state.context_notes[-3:])  # Last 3 notes

        # Use sonar-reasoning-pro for generating research directions
        response = await self.reasoner.reason(
            query=f"Given this query: {query}\n\nContext:\n{chr(10).join(context)}",
            system_prompt="""You are a research assistant. Given a query and context:
1. Analyze what specific information we need
2. Generate targeted sub-queries to gather this information
3. Return queries that will help us explore different aspects of the topic"""
        )

        # Format the response into structured queries
        schema = {
            "type": "object",
            "required": ["queries"],
            "additionalProperties": False,
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["query", "goal"],
                        "additionalProperties": False,
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The research sub-query to investigate"
                            },
                            "goal": {
                                "type": "string",
                                "description": "The specific goal or purpose of this sub-query"
                            }
                        }
                    }
                }
            }
        }
        
        # Handle long responses appropriately
        if len(response.choices[0].message.content) > 2000:
            structured = await self._process_long_query(
                response.choices[0].message.content,
                schema
            )
        else:
            structured = await self.reasoner.format_structured_output(
                response.choices[0].message.content,
                schema
            )
        
        queries = json.loads(structured).get("queries", [])
        return queries[:num_queries]

    async def _process_query(
        self,
        query: str,
        state: ResearchState,
        progress: Optional[ResearchProgress] = None,
        num_learnings: int = 3,
        num_follow_up: int = 3
    ) -> Dict[str, Any]:
        """Process a single query using state context."""
        if progress:
            self._report_action(
                progress,
                f"Processing query: {query}",
                state
            )
        
        # First decide if we should use plugins at all
        if not await self._should_use_plugins(query):
            # Use pure reasoning if plugins aren't needed
            synthesis_response = await self.reasoner.reason(
                query=query,
                system_prompt="""You are a research assistant. Given the query:
1. Analyze the question thoroughly
2. Extract key insights
3. Identify areas that need further investigation"""
            )
            
            synthesis_str = await self.reasoner.format_structured_output(
                synthesis_response.choices[0].message.content,
                schema={
                    "type": "object",
                    "required": ["learnings", "next"],
                    "additionalProperties": False,
                    "properties": {
                        "learnings": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "A key learning or insight from the research"
                            }
                        },
                        "next": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "A follow-up direction or area to investigate"
                            }
                        }
                    }
                }
            )
            synthesis = json.loads(synthesis_str)
            
            return {
                "learnings": synthesis.get("learnings", []),
                "next": synthesis.get("next", []),
                "plugins": []
            }
        
        # If plugins are needed, evaluate which ones are relevant
        relevant_plugins = {}
        for name, plugin in self.plugins.items():
            try:
                if await self._evaluate_plugin_relevance(query, name, plugin):
                    relevant_plugins[name] = plugin
                    print(f"Using plugin {name} for query: {query}")
            except Exception as e:
                print(f"Failed to evaluate plugin {name}: {e}")
        
        # Generate structured queries for relevant plugins
        analysis = await self.reasoner.reason(
            query=f"""Given this research query: {query}

Available plugins and their capabilities:
{chr(10).join(f'- {name}: {plugin.query_schema.model_json_schema()}' for name, plugin in relevant_plugins.items())}

What specific data should we request from each plugin to answer this query?""",
            system_prompt="You are a research assistant. Analyze the query and determine what data we need from each available plugin."
        )
        
        # Format the analysis into structured plugin queries
        plugin_queries_str = await self.reasoner.format_structured_output(
            analysis.choices[0].message.content,
            schema={
                "type": "object",
                "required": ["plugin_queries"],
                "additionalProperties": False,
                "properties": {
                    "plugin_queries": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "required": ["prompt"],
                            "additionalProperties": False,
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The specific question or query to analyze"
                                }
                            }
                        }
                    }
                }
            }
        )
        plugin_queries = json.loads(plugin_queries_str).get("plugin_queries", {})

        # Get plugin data
        plugin_results = {}
        for name, plugin in relevant_plugins.items():
            try:
                if name in plugin_queries:
                    # For Grok plugin, ensure proper query structure
                    query_params = {"prompt": plugin_queries[name]["prompt"]}
                    
                    # Validate against plugin schema
                    query_json = json.dumps(
                        plugin.query_schema(**query_params).model_dump()
                    )
                    
                    plugin_results[name] = await plugin.query(query_json)
                    if name not in state.plugins_used:
                        state.plugins_used.append(name)
            except Exception as e:
                print(f"Plugin {name} failed: {e}")
                continue

        # Use reasoning to analyze findings
        synthesis_response = await self.reasoner.reason(
            query=json.dumps({
                "query": query,
                "plugin_data": plugin_results,
                "current_context": state.context_notes[-3:] if state.context_notes else []
            }),
            system_prompt="""You are a research assistant. Given the query and plugin data:
1. Analyze the information we've gathered
2. Extract key learnings
3. Identify areas that need further investigation
4. Add any important context notes"""
        )
        
        # Format into structured synthesis
        synthesis_str = await self.reasoner.format_structured_output(
            synthesis_response.choices[0].message.content,
            schema={
                "type": "object",
                "required": ["learnings", "next", "context_notes"],
                "additionalProperties": False,
                "properties": {
                    "learnings": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A key learning or insight from the research"
                        }
                    },
                    "next": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A follow-up direction or area to investigate"
                        }
                    },
                    "context_notes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Additional context or notes about the research"
                        }
                    }
                }
            }
        )
        synthesis = json.loads(synthesis_str)

        # Update state
        new_learnings = synthesis.get("learnings", [])
        state.learnings.extend(new_learnings)
        
        state.query_history.append({
            "query": query,
            "learnings": new_learnings,
            "plugins_used": list(plugin_results.keys())
        })

        if "context_notes" in synthesis:
            state.context_notes.extend(synthesis.get("context_notes", []))

        return {
            "learnings": new_learnings,
            "next": synthesis.get("next", []),
            "plugins": list(plugin_results.keys())
        }

    async def research(
        self,
        request: ResearchRequest,
        on_progress: Optional[callable] = None
    ) -> ResearchResult:
        """Execute research."""
        progress = ResearchProgress(
            current_depth=request.depth,
            total_depth=request.depth,
            current_breadth=request.breadth,
            total_breadth=request.breadth,
            total_queries=0,
            completed_queries=0
        )

        state = ResearchState()
        
        self._report_action(
            progress,
            f"Starting research on: {request.query}",
            state
        )

        def report_progress(**kwargs):
            for k, v in kwargs.items():
                setattr(progress, k, v)
            if on_progress:
                on_progress(progress)

        async def deep_research(
            query: str,
            depth: int,
            breadth: int,
            path_id: str = "root"
        ) -> None:
            self._report_action(
                progress,
                f"Starting depth {depth} research: {query}",
                state
            )
            
            # Generate queries
            queries = await self._generate_queries(
                query=query,
                num_queries=breadth,
                state=state
            )
            
            report_progress(
                total_queries=len(queries),
                current_query=queries[0]["query"] if queries else None
            )

            # Track this research path
            current_path = {
                "id": path_id,
                "query": query,
                "depth": depth,
                "sub_queries": []
            }
            state.research_paths.append(current_path)

            for i, q in enumerate(queries):
                try:
                    # Process query
                    result = await self._process_query(
                        query=q["query"],
                        state=state,
                        num_follow_up=max(1, breadth // 2)
                    )
                    
                    sub_path_id = f"{path_id}_{i}"
                    current_path["sub_queries"].append({
                        "id": sub_path_id,
                        "query": q["query"],
                        "learnings": result["learnings"]
                    })

                    # Recurse if needed
                    if depth > 0:
                        report_progress(
                            current_depth=depth - 1,
                            current_breadth=max(1, breadth // 2),
                            completed_queries=progress.completed_queries + 1,
                            current_query=q["query"]
                        )

                        next_query = f"""
                        Previous goal: {q.get('goal', '')}
                        Follow-up directions: {json.dumps(result['next'])}
                        """.strip()

                        await deep_research(
                            query=next_query,
                            depth=depth - 1,
                            breadth=max(1, breadth // 2),
                            path_id=sub_path_id
                        )
                    else:
                        report_progress(
                            current_depth=0,
                            completed_queries=progress.completed_queries + 1,
                            current_query=q["query"]
                        )
                except Exception as e:
                    print(f"Error processing query: {q['query']}: {e}")

        # Execute research
        await deep_research(
            query=request.query,
            depth=request.depth,
            breadth=request.breadth
        )

        # Generate comprehensive summary before returning
        summary = state.generate_research_summary()
        
        # Return final results including summary
        return ResearchResult(
            learnings=state.learnings,
            plugins_used=state.plugins_used,
            research_paths=state.research_paths,
            summary=summary
        )

    def _report_action(self, progress: ResearchProgress, action: str, state: ResearchState):
        """Report current action and state."""
        progress.current_action = action
        progress.current_state = state.summarize()
        
        # Report current action and status
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"STATUS UPDATE - {action}")
        self.logger.info("=" * 80)
        
        # Current task status
        self.logger.info("CURRENT TASK:")
        self.logger.info(f"→ Action: {action}")
        if progress.current_query:
            self.logger.info(f"→ Query: {progress.current_query}")
        
        # Progress metrics
        self.logger.info("\nPROGRESS:")
        self.logger.info(f"→ Depth: {progress.current_depth}/{progress.total_depth}")
        self.logger.info(f"→ Breadth: {progress.current_breadth}/{progress.total_breadth}")
        self.logger.info(f"→ Queries: {progress.completed_queries}/{progress.total_queries}")
        
        # Active state
        self.logger.info("\nACTIVE STATE:")
        self.logger.info(f"→ Total Findings: {len(state.learnings)}")
        self.logger.info(f"→ Active Plugins: {', '.join(state.plugins_used) or 'None'}")
        
        # Most recent activity
        if state.query_history:
            self.logger.info("\nLAST 3 QUERIES:")
            for entry in state.query_history[-3:]:
                self.logger.info(f"Query: {entry['query']}")
                if entry.get('learnings'):
                    for learning in entry['learnings']:
                        self.logger.info(f"  • {learning}")
                if entry.get('plugins_used'):
                    self.logger.info(f"  Using: {', '.join(entry['plugins_used'])}")
                self.logger.info("-" * 40)
        
        # Latest findings
        if state.learnings:
            self.logger.info("\nLATEST FINDINGS:")
            for learning in state.learnings[-3:]:
                self.logger.info(f"• {learning}")
        
        self.logger.info("=" * 80)
