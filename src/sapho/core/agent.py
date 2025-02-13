"""Research agent using Perplexity Sonar API."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sapho.ai.sonar import SonarReasoner, ChatMessage, ChatRequest
from sapho.plugins import load_plugins
import json

class ResearchProgress(BaseModel):
    """Research progress tracking."""
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: Optional[str] = None
    total_queries: int = 0
    completed_queries: int = 0

class ResearchState(BaseModel):
    """Research state - acts as a notepad during research."""
    learnings: List[str] = Field(default_factory=list)
    plugins_used: List[str] = Field(default_factory=list)
    query_history: List[Dict[str, Any]] = Field(default_factory=list)
    context_notes: List[str] = Field(default_factory=list)
    research_paths: List[Dict[str, Any]] = Field(default_factory=list)

class ResearchResult(BaseModel):
    """Final research results."""
    learnings: List[str]
    plugins_used: List[str]
    research_paths: List[Dict[str, Any]]

class ResearchRequest(BaseModel):
    """Research request."""
    query: str
    depth: int = 2
    breadth: int = 4

class ResearchAgent:
    def __init__(self, api_key: str):
        """Initialize agent with all available plugins.
        
        Args:
            api_key: API key for Sonar
        """
        self.reasoner = SonarReasoner(api_key)
        self.plugins = {}  # All available plugins
        
        # Load and initialize all plugins upfront
        available_plugins = load_plugins()
        for name, plugin_class in available_plugins.items():
            try:
                self.plugins[name] = plugin_class()
                print(f"Initialized plugin: {name}")
            except Exception as e:
                print(f"Failed to initialize plugin {name}: {e}")

    async def _evaluate_plugin_relevance(
        self,
        query: str,
        plugin_name: str,
        plugin: Plugin
    ) -> bool:
        """Evaluate if a plugin is relevant for the current query."""
        analysis = await self.reasoner.reason(
            query=f"""Given this research query: {query}

Plugin capability:
{plugin_name}: {plugin.query_schema.model_json_schema()}

Should this plugin be used to help answer the query? Consider:
1. Is the plugin's data relevant to the query?
2. Would the plugin's capabilities help advance the research?
3. Is the data type provided by the plugin useful for this specific question?

Return a clear YES or NO.""",
            system_prompt="You are a research assistant. Evaluate if the plugin would be helpful for the current query."
        )
        
        return "YES" in analysis.choices[0].message.content.upper()

    async def _generate_queries(
        self,
        query: str,
        num_queries: int,
        state: ResearchState
    ) -> List[Dict[str, str]]:
        """Generate follow-up queries using state context."""
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
        
        # Format the response into structured queries using sonar-pro
        structured = await self.reasoner.format_structured_output(
            response.choices[0].message.content,
            schema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "goal": {"type": "string"}
                            },
                            "required": ["query", "goal"]
                        }
                    }
                }
            }
        )
        
        queries = json.loads(structured).get("queries", [])
        return queries[:num_queries]

    async def _process_query(
        self,
        query: str,
        state: ResearchState,
        num_learnings: int = 3,
        num_follow_up: int = 3
    ) -> Dict[str, Any]:
        """Process a single query using state context."""
        # First evaluate which plugins would be useful for this specific query
        relevant_plugins = {}
        for name, plugin in self.plugins.items():
            try:
                if await self._evaluate_plugin_relevance(query, name, plugin):
                    relevant_plugins[name] = plugin
                    print(f"Using plugin {name} for query: {query}")
            except Exception as e:
                print(f"Failed to evaluate plugin {name}: {e}")
        
        if not relevant_plugins:
            # Use pure reasoning if no plugins are relevant
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
                    "properties": {
                        "learnings": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "next": {
                            "type": "array",
                            "items": {"type": "string"}
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
        
        # For relevant plugins, proceed with structured queries
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
                "additionalProperties": {
                    "type": "object",
                    "description": "Plugin parameters"
                }
            }
        )
        plugin_queries = json.loads(plugin_queries_str)

        # Get plugin data
        plugin_results = {}
        for name, plugin in relevant_plugins.items():
            try:
                if name in plugin_queries:
                    # Validate against plugin schema
                    query_json = json.dumps(
                        plugin.query_schema(**plugin_queries[name]).model_dump()
                    )
                    
                    plugin_results[name] = await plugin.query(query_json)
                    if name not in state.plugins_used:
                        state.plugins_used.append(name)
            except Exception as e:
                print(f"Plugin {name} failed: {e}")

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
                "properties": {
                    "learnings": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "next": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "context_notes": {
                        "type": "array",
                        "items": {"type": "string"}
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

        # Return final results from state
        return ResearchResult(
            learnings=state.learnings,
            plugins_used=state.plugins_used,
            research_paths=state.research_paths
        )
