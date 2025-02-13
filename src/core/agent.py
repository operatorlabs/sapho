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

# Research Agent Configuration
RESEARCH_CONFIG = {
    # Context and token limits
    "MAX_CONTEXT_TOKENS": 200000,  # Maximum tokens for context in reasoning
    "MAX_RESPONSE_TOKENS": 5000,   # Target length for final responses
    "MAX_SUMMARY_TOKENS": 2000,    # Maximum tokens for summaries
    
    # History tracking
    "MAX_HISTORY_ITEMS": 5,        # Number of recent items to track in history
    "MAX_CONTEXT_NOTES": 3,        # Number of context notes to maintain
    
    # Research parameters
    "DEFAULT_DEPTH": 2,            # Default research depth
    "DEFAULT_BREADTH": 4,          # Default research breadth
    "MAX_FOLLOW_UP": 3,            # Maximum follow-up queries per branch
    "MAX_LEARNINGS_PER_QUERY": 3,  # Maximum learnings to extract per query
    
    # Formatting
    "MAX_CHARS_PER_CHUNK": 1500,   # Maximum characters per chunk for processing
    "MAX_SUMMARY_ITERATIONS": 3,    # Maximum iterations for recursive summarization
}

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
            
            if state:
                state.add_reasoning_trace(
                    query=query,
                    response=response_content,
                    result={"length": len(response_content), "schema": schema},
                    trace_type="structured_response"
                )
            
            # Try to format it
            try:
                formatted = await self.reasoner.format_structured_output(
                    query=response_content,
                    schema=schema
                )
                if state:
                    state.add_reasoning_trace(
                        query=response_content,
                        response=formatted,
                        result={"success": True},
                        trace_type="format_output"
                    )
                return formatted
            except Exception as e:
                self.logger.error(f"Failed to format response: {str(e)}")
                self.logger.error("Response content:")
                self.logger.error(response_content)
                if state:
                    state.add_reasoning_trace(
                        query=response_content,
                        response=str(e),
                        result={"success": False, "error": str(e)},
                        trace_type="format_error"
                    )
                raise
                
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Original query ({len(query)} chars):")
            self.logger.error(query[:1000] + "..." if len(query) > 1000 else query)
            if state:
                state.add_reasoning_trace(
                    query=query,
                    response=str(e),
                    result={"success": False, "error": str(e)},
                    trace_type="process_error"
                )
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
        num_learnings: int = RESEARCH_CONFIG["MAX_LEARNINGS_PER_QUERY"],
        num_follow_up: int = RESEARCH_CONFIG["MAX_FOLLOW_UP"]
    ) -> Dict[str, Any]:
        """Process a single query using state context."""
        try:
            if progress:
                self._report_action(
                    progress,
                    f"Processing query: {query}",
                    state
                )
            
            # First decide if we should use plugins at all
            plugin_analysis = await self._should_use_plugins(query)
            state.add_reasoning_trace(
                query=query,
                response=str(plugin_analysis),
                result="PLUGINS" if plugin_analysis else "REASONING",
                trace_type="plugin_decision"
            )
            
            if not plugin_analysis:
                # Use pure reasoning if plugins aren't needed
                try:
                    # First get initial analysis
                    initial_response = await self.reasoner.reason(
                        query=query,
                        system_prompt=f"""You are a research assistant. Given the query:
1. Analyze the question thoroughly
2. Extract key insights and findings
3. BE CONCISE - keep your response under {RESEARCH_CONFIG["MAX_CHARS_PER_CHUNK"]} characters
4. You can use up to {RESEARCH_CONFIG["MAX_CONTEXT_TOKENS"]} tokens of context
5. Focus on direct, factual answers"""
                    )
                    
                    # Then structure the findings
                    synthesis_response = await self.reasoner.reason(
                        query=f"""Based on this analysis, extract key learnings and next steps:

{initial_response.choices[0].message.content}

Format your response to include:
1. Key learnings/findings (3-5 clear points)
2. Potential follow-up directions""",
                        system_prompt="Extract and structure the key findings and next steps."
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
                    
                    # Update state with new learnings
                    new_learnings = synthesis.get("learnings", [])[:num_learnings]
                    if new_learnings:  # Only extend if we actually found learnings
                        state.learnings.extend(new_learnings)
                        
                        # Add to query history
                        state.query_history.append({
                            "query": query,
                            "learnings": new_learnings,
                            "plugins_used": []
                        })
                        
                        state.add_reasoning_trace(
                            query=query,
                            response=synthesis_response.choices[0].message.content,
                            result=synthesis,
                            trace_type="pure_reasoning"
                        )
                    
                    return {
                        "learnings": new_learnings,
                        "next": synthesis.get("next", [])[:num_follow_up],
                        "plugins": []
                    }
                except Exception as e:
                    self.logger.error(f"Pure reasoning failed: {str(e)}")
                    # Return empty results on error
                    return {"learnings": [], "next": [], "plugins": []}
            
            # If plugins are needed, evaluate which ones are relevant
            relevant_plugins = {}
            for name, plugin in self.plugins.items():
                try:
                    relevance = await self._evaluate_plugin_relevance(query, name, plugin)
                    state.add_reasoning_trace(
                        query=f"Evaluate {name} plugin relevance for: {query}",
                        response=str(relevance),
                        result={"plugin": name, "relevant": relevance},
                        trace_type="plugin_evaluation"
                    )
                    if relevance:
                        relevant_plugins[name] = plugin
                        self.logger.info(f"Using plugin {name} for query: {query}")
                except Exception as e:
                    self.logger.error(f"Failed to evaluate plugin {name}: {e}")
                    continue
            
            if not relevant_plugins:
                self.logger.warning("No relevant plugins found, falling back to pure reasoning")
                return {"learnings": [], "next": [], "plugins": []}
            
            # Generate structured queries for relevant plugins
            try:
                analysis = await self.reasoner.reason(
                    query=f"""Given this research query: {query}

Available plugins and their capabilities:
{chr(10).join(f'- {name}: {plugin.query_schema.model_json_schema()}' for name, plugin in relevant_plugins.items())}

What specific data should we request from each plugin to answer this query?
BE CONCISE - keep your response under 1000 characters.""",
                    system_prompt="You are a research assistant. Analyze the query and determine what data we need from each available plugin."
                )
                
                state.add_reasoning_trace(
                    query=query,
                    response=analysis.choices[0].message.content,
                    result={"plugins": list(relevant_plugins.keys())},
                    trace_type="plugin_query_generation"
                )
                
                # Format the analysis into structured plugin queries
                plugin_queries_str = await self._process_long_query(
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
                    },
                    max_chars=1500,
                    state=state
                )
                plugin_queries = json.loads(plugin_queries_str).get("plugin_queries", {})
            except Exception as e:
                self.logger.error(f"Failed to generate plugin queries: {e}")
                return {"learnings": [], "next": [], "plugins": []}

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
                    self.logger.error(f"Plugin {name} failed: {e}")
                    continue

            if not plugin_results:
                self.logger.warning("No plugin results obtained")
                return {"learnings": [], "next": [], "plugins": []}

            # Use reasoning to analyze findings
            try:
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
4. Add any important context notes
5. BE CONCISE - keep your response under 1000 characters"""
                )
                
                # Format into structured synthesis
                synthesis_str = await self._process_long_query(
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
                    },
                    max_chars=1500,
                    state=state
                )
                synthesis = json.loads(synthesis_str)

                # Update state
                new_learnings = synthesis.get("learnings", [])[:num_learnings]
                state.learnings.extend(new_learnings)
                
                state.query_history.append({
                    "query": query,
                    "learnings": new_learnings,
                    "plugins_used": list(plugin_results.keys())
                })

                if "context_notes" in synthesis:
                    state.context_notes.extend(synthesis.get("context_notes", [])[:3])

                state.add_reasoning_trace(
                    query=json.dumps({
                        "query": query,
                        "plugin_data": plugin_results,
                        "current_context": state.context_notes[-3:] if state.context_notes else []
                    }),
                    response=synthesis_response.choices[0].message.content,
                    result=synthesis,
                    trace_type="reasoning_analysis"
                )

                return {
                    "learnings": new_learnings,
                    "next": synthesis.get("next", [])[:num_follow_up],
                    "plugins": list(plugin_results.keys())
                }
            except Exception as e:
                self.logger.error(f"Failed to synthesize results: {e}")
                return {
                    "learnings": [],
                    "next": [],
                    "plugins": list(plugin_results.keys())
                }
                
        except Exception as e:
            self.logger.error(f"Error in _process_query: {e}")
            return {"learnings": [], "next": [], "plugins": []}

    async def _recursive_summarize(
        self,
        content: str,
        max_chars: int = 2000,
        max_iterations: int = 3,
        iteration: int = 0
    ) -> str:
        """Recursively summarize content until it fits within max_chars."""
        if len(content) <= max_chars:
            return content
            
        if iteration >= max_iterations:
            self.logger.warning(f"Hit max iterations ({max_iterations}), truncating to {max_chars} chars")
            return content[:max_chars-100] + "\n[Content truncated due to length]"
            
        # Split into chunks and summarize each
        chunks = [content[i:i + max_chars] for i in range(0, len(content), max_chars)]
        summaries = []
        
        for chunk in chunks:
            response = await self.reasoner.reason(
                query=f"Summarize this content VERY concisely, focusing only on the most important points:\n\n{chunk}",
                system_prompt="You are a research assistant. Create an extremely concise summary that captures only the most essential information."
            )
            summaries.append(response.choices[0].message.content)
        
        # Combine summaries
        combined = "\n\n".join(summaries)
        
        # If still too long, recurse with iteration counter
        if len(combined) > max_chars:
            return await self._recursive_summarize(combined, max_chars, max_iterations, iteration + 1)
        
        return combined

    async def _create_final_report(self, state: ResearchState) -> str:
        """Create a final research report by analyzing and summarizing all findings."""
        # Get all available information
        summary = {
            "original_query": state.research_paths[0]["query"] if state.research_paths else None,
            "key_findings": state.learnings[-RESEARCH_CONFIG["MAX_HISTORY_ITEMS"]:],  # Last N key findings
            "latest_query": state.query_history[-1]["query"] if state.query_history else None,
            "latest_findings": state.query_history[-1].get("learnings", []) if state.query_history else [],
            "plugins_used": state.plugins_used,
            "total_learnings": len(state.learnings),
            "total_context": len(state.context_notes),
            "citations": [],  # Add this to store citations
            "reasoning_traces": [
                trace for trace in state.reasoning_traces 
                if trace["type"] in ["pure_reasoning", "reasoning_analysis", "plugin_response"]  # Include plugin responses
            ][-3:]  # Last 3 reasoning traces
        }
        
        # Extract citations from reasoning traces
        for trace in state.reasoning_traces:
            if "result" in trace and isinstance(trace["result"], dict):
                if "citations" in trace["result"]:
                    summary["citations"].extend(trace["result"]["citations"])
                elif "sources" in trace["result"]:
                    summary["citations"].extend(trace["result"]["sources"])
        
        # Create final report with strict length limit and citations
        prompt = f"""Create a final research report answering this query: {summary["original_query"]}

Available Sources:
{chr(10).join(f'[{i+1}] {citation}' for i, citation in enumerate(summary["citations"]))}

Latest Understanding:
{chr(10).join(f"- {finding}" for finding in summary["latest_findings"]) if summary["latest_findings"] else "No direct findings available."}

Key Findings Throughout Research ({summary["total_learnings"]} total):
{chr(10).join(f"- {finding}" for finding in summary["key_findings"]) if summary["key_findings"] else "Using analysis from research process."}

Context Notes: {summary["total_context"]} notes collected
Tools Used: {', '.join(summary["plugins_used"]) if summary["plugins_used"] else "Pure reasoning"}

IMPORTANT: Your response must:
1. Be under {RESEARCH_CONFIG["MAX_RESPONSE_TOKENS"]} tokens
2. Focus on directly answering the original query
3. Incorporate all available information
4. Use clear formatting for readability
5. Provide a clear conclusion
6. Reference sources using [n] citation format
7. If no direct findings available, synthesize from the analysis process"""

        try:
            response = await self.reasoner.reason(
                query=prompt,
                system_prompt=f"""You are a research assistant. Create a final report that:
1. Directly answers the main query
2. Uses all available information to provide insights
3. Uses clear formatting with headers and bullet points
4. Keeps the response under {RESEARCH_CONFIG["MAX_RESPONSE_TOKENS"]} tokens
5. Provides clear conclusions based on the research
6. Can use up to {RESEARCH_CONFIG["MAX_CONTEXT_TOKENS"]} tokens of context
7. If no direct findings available, synthesize insights from the analysis process"""
            )
            
            report = response.choices[0].message.content.strip()
            
            # Enforce length limit if needed
            if len(report) > RESEARCH_CONFIG["MAX_SUMMARY_TOKENS"]:
                report = report[:RESEARCH_CONFIG["MAX_SUMMARY_TOKENS"]-100] + "\n\n[Report truncated to fit length limits]"
                
            # Add metadata footer
            report += f"\n\nResearch Statistics:\n"
            report += f"- Total Findings: {summary['total_learnings']}\n"
            report += f"- Context Notes: {summary['total_context']}\n"
            report += f"- Tools Used: {', '.join(summary['plugins_used']) if summary['plugins_used'] else 'Pure reasoning'}"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating final report: {str(e)}")
            return f"Error creating final report: {str(e)}"

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

            # Process each query and its sub-queries
            for i, q in enumerate(queries):
                try:
                    # Process current query
                    result = await self._process_query(
                        query=q["query"],
                        state=state,
                        progress=progress,
                        num_follow_up=max(1, breadth // 2)
                    )
                    
                    sub_path_id = f"{path_id}_{i}"
                    sub_query_result = {
                        "id": sub_path_id,
                        "query": q["query"],
                        "learnings": result["learnings"],
                        "plugins_used": result["plugins"],
                        "sub_queries": []
                    }
                    current_path["sub_queries"].append(sub_query_result)

                    # Add to state's query history if not already there
                    if not any(entry["query"] == q["query"] for entry in state.query_history):
                        state.query_history.append({
                            "query": q["query"],
                            "learnings": result["learnings"],
                            "plugins_used": result["plugins"]
                        })

                    # Recurse if needed and there are follow-up directions
                    if depth > 0 and result["next"]:
                        report_progress(
                            current_depth=depth - 1,
                            current_breadth=max(1, breadth // 2),
                            completed_queries=progress.completed_queries + 1,
                            current_query=q["query"]
                        )

                        # Process each follow-up direction
                        for j, next_direction in enumerate(result["next"]):
                            next_query = f"""
                            Previous query: {q["query"]}
                            Previous findings: {json.dumps(result["learnings"])}
                            Follow-up direction: {next_direction}
                            """.strip()

                            await deep_research(
                                query=next_query,
                                depth=depth - 1,
                                breadth=max(1, breadth // 2),
                                path_id=f"{sub_path_id}_{j}"
                            )
                    else:
                        report_progress(
                            current_depth=0,
                            completed_queries=progress.completed_queries + 1,
                            current_query=q["query"]
                        )
                except Exception as e:
                    self.logger.error(f"Error processing query: {q['query']}: {e}")
                    # Still add failed query to history for tracking
                    state.query_history.append({
                        "query": q["query"],
                        "learnings": [],
                        "plugins_used": [],
                        "error": str(e)
                    })

        # Execute research
        await deep_research(
            query=request.query,
            depth=request.depth,
            breadth=request.breadth
        )

        # Generate final report
        self.logger.info("\nGenerating final research report...")
        try:
            final_report = await self._create_final_report(state)
            self.logger.info("\nFinal Report:")
            self.logger.info("=" * 80)
            self.logger.info(final_report)
            self.logger.info("=" * 80)
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
            final_report = "Error generating final report"
        
        # Generate comprehensive summary
        summary = state.generate_research_summary()
        summary["final_report"] = final_report
        
        # Return final results including summary and report
        result = ResearchResult(
            learnings=state.learnings,
            plugins_used=state.plugins_used,
            research_paths=state.research_paths,
            summary=summary
        )
        
        # Log the final results
        self.logger.info("\nResearch Complete!")
        self.logger.info(f"Total Learnings: {len(result.learnings)}")
        self.logger.info(f"Plugins Used: {', '.join(result.plugins_used) if result.plugins_used else 'None'}")
        self.logger.info(f"Research Paths: {len(result.research_paths)}")
        
        return result

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
