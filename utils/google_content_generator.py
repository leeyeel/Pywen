"""Google Gemini API client wrapper with tool integration."""

import json
import traceback
import uuid
from typing import Dict, Any, List, Optional, Union, AsyncGenerator

try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None

from .base_content_generator import BaseLLMClient, ContentGenerator
from .config import Config, GenerateContentConfig, ModelParameters
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from tools.base import Tool, ToolCall, ToolResult


class GoogleContentGenerator(ContentGenerator):
    """Google Gemini content generator."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google GenAI library not available. Install with: pip install google-genai")
        
        self.client = genai.Client(api_key=config.api_key)
        self.message_history: List[types.Content] = []
        self.system_instruction: Optional[str] = None
    
    async def generate_content(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> LLMResponse:
        """Generate content from messages."""
        
        # Parse messages to Google format
        current_chat_contents, system_instruction = self.parse_messages(messages)
        
        # Set up generation config
        generation_config = types.GenerateContentConfig(
            temperature=config.temperature if config else self.config.model_params.temperature,
            top_p=config.top_p if config else self.config.model_params.top_p,
            top_k=config.top_k if config else self.config.model_params.top_k,
            max_output_tokens=config.max_output_tokens if config else self.config.model_params.max_tokens,
            system_instruction=system_instruction,
        )
        
        # Add tools if provided
        if tools:
            tool_schemas = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    ]
                )
                for tool in tools
            ]
            generation_config.tools = tool_schemas
        
        # Make API call
        response = self.client.models.generate_content(
            model=self.config.model_params.model,
            contents=current_chat_contents,
            config=generation_config,
        )
        
        # Parse response
        return self._parse_response(response)
    
    async def generate_content_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate content stream from messages."""
        
        # Parse messages to Google format
        current_chat_contents, system_instruction = self.parse_messages(messages)
        
        # Set up generation config
        generation_config = types.GenerateContentConfig(
            temperature=config.temperature if config else self.config.model_params.temperature,
            top_p=config.top_p if config else self.config.model_params.top_p,
            top_k=config.top_k if config else self.config.model_params.top_k,
            max_output_tokens=config.max_output_tokens if config else self.config.model_params.max_tokens,
            system_instruction=system_instruction,
        )
        
        # Add tools if provided
        if tools:
            tool_schemas = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    ]
                )
                for tool in tools
            ]
            generation_config.tools = tool_schemas
        
        # Make streaming API call
        stream = self.client.models.generate_content_stream(
            model=self.config.model_params.model,
            contents=current_chat_contents,
            config=generation_config,
        )
        
        # Yield streaming responses
        accumulated_content = ""
        for chunk in stream:
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            accumulated_content += part.text
                            yield LLMResponse(
                                content=accumulated_content,
                                model=self.config.model_params.model,
                                finish_reason="partial",
                                usage=self._extract_usage(chunk),
                                tool_calls=[]
                            )
    
    async def generate_response_with_google_search(
        self,
        messages: List[LLMMessage],
        search_tools: List[Dict[str, Any]]
    ) -> LLMResponse:
        """Generate response using Google Search tools."""
        
        # Parse messages to Google format
        current_chat_contents, system_instruction = self.parse_messages(messages)
        
        # Set up generation config with Google Search
        generation_config = types.GenerateContentConfig(
            temperature=self.config.model_params.temperature,
            top_p=self.config.model_params.top_p,
            top_k=self.config.model_params.top_k,
            max_output_tokens=self.config.model_params.max_tokens,
            system_instruction=system_instruction,
            tools=[types.Tool(google_search={})]  # Enable Google Search
        )
        
        # Make API call
        response = self.client.models.generate_content(
            model=self.config.model_params.model,
            contents=current_chat_contents,
            config=generation_config,
        )
        
        # Parse response with grounding metadata
        llm_response = self._parse_response(response)
        
        # Extract grounding metadata
        if response.candidates:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, 'grounding_metadata', None)
            llm_response.grounding_metadata = grounding_metadata
        
        return llm_response
    
    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in messages."""
        current_chat_contents, _ = self.parse_messages(messages)
        
        try:
            response = self.client.models.count_tokens(
                model=self.config.model_params.model,
                contents=current_chat_contents
            )
            return response.total_tokens
        except Exception:
            # Fallback estimation
            total_chars = sum(len(msg.content or "") for msg in messages)
            return total_chars // 4  # Rough estimation
    
    async def embed_content(self, content: str) -> List[float]:
        """Generate embeddings for content."""
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",  # Default embedding model
                content=content
            )
            return response.embedding
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return []
    
    def parse_messages(self, messages: List[LLMMessage]) -> tuple[List[types.Content], Optional[str]]:
        """Parse the messages to Gemini format, separating system instructions."""
        gemini_messages: List[types.Content] = []
        system_instruction: Optional[str] = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue
            elif hasattr(msg, 'tool_result') and msg.tool_result:
                gemini_messages.append(
                    types.Content(
                        role="tool",
                        parts=[self.parse_tool_call_result(msg.tool_result)],
                    )
                )
            elif hasattr(msg, 'tool_call') and msg.tool_call:
                gemini_messages.append(
                    types.Content(role="model", parts=[self.parse_tool_call(msg.tool_call)])
                )
            else:
                role = "user" if msg.role == "user" else "model"
                gemini_messages.append(
                    types.Content(role=role, parts=[types.Part(text=msg.content or "")])
                )
        
        return gemini_messages, system_instruction
    
    def parse_tool_call(self, tool_call: ToolCall) -> types.Part:
        """Parse a ToolCall into a Gemini FunctionCall Part for history."""
        return types.Part.from_function_call(name=tool_call.name, args=tool_call.arguments)
    
    def parse_tool_call_result(self, tool_result: ToolResult) -> types.Part:
        """Parse a ToolResult into a Gemini FunctionResponse Part for history."""
        result_content = {}
        
        if tool_result.result is not None:
            if isinstance(tool_result.result, (str, int, float, bool, list, dict)):
                try:
                    json.dumps(tool_result.result)
                    result_content["result"] = tool_result.result
                except (TypeError, OverflowError) as e:
                    tb = traceback.format_exc()
                    serialization_error = f"JSON serialization failed for tool result: {e}\n{tb}"
                    if tool_result.error:
                        result_content["error"] = f"{tool_result.error}\n\n{serialization_error}"
                    else:
                        result_content["error"] = serialization_error
                    result_content["result"] = str(tool_result.result)
            else:
                result_content["result"] = str(tool_result.result)
        
        if tool_result.error and "error" not in result_content:
            result_content["error"] = tool_result.error
        
        if not result_content:
            result_content["status"] = "Tool executed successfully but returned no output."
        
        if not hasattr(tool_result, "name") or not tool_result.name:
            raise AttributeError(
                "ToolResult must have a 'name' attribute matching the function that was called."
            )
        
        return types.Part.from_function_response(name=tool_result.name, response=result_content)
    
    def _parse_response(self, response) -> LLMResponse:
        """Parse Google API response to LLMResponse."""
        content = ""
        tool_calls: List[ToolCall] = []
        
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        content += part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        tool_calls.append(
                            ToolCall(
                                call_id=str(uuid.uuid4()),
                                name=part.function_call.name,
                                arguments=dict(part.function_call.args)
                                if part.function_call.args
                                else {},
                            )
                        )
        
        return LLMResponse(
            content=content,
            model=self.config.model_params.model,
            finish_reason="stop",
            usage=self._extract_usage(response),
            tool_calls=tool_calls
        )
    
    def _extract_usage(self, response) -> Optional[LLMUsage]:
        """Extract usage information from response."""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            return LLMUsage(
                input_tokens=getattr(usage, 'prompt_token_count', 0),
                output_tokens=getattr(usage, 'candidates_token_count', 0),
                total_tokens=getattr(usage, 'total_token_count', 0),
                cached_tokens=getattr(usage, 'cached_content_token_count', None),
            )
        return None


class GoogleClient(BaseLLMClient):
    """Google Gemini client wrapper."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.content_generator = GoogleContentGenerator(config)
    
    async def generate_response(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate response using Google Gemini."""
        
        # Handle Google Search tools specially
        if extra_params and "tools" in extra_params:
            google_search_tools = extra_params["tools"]
            return await self.content_generator.generate_response_with_google_search(
                messages=messages,
                search_tools=google_search_tools
            )
        
        if stream:
            return self.content_generator.generate_content_stream(
                messages=messages,
                tools=tools
            )
        else:
            return await self.content_generator.generate_content(
                messages=messages,
                tools=tools
            )
    
    async def generate_with_retry(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        max_retries: int = 3
    ) -> LLMResponse:
        """Generate response with retry logic."""
        import asyncio
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.generate_response(
                    messages=messages,
                    tools=tools,
                    stream=False
                )
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    break
        
        raise last_exception
    
    def supports_tool_calling(self) -> bool:
        """Check if the current model supports tool calling."""
        tool_capable_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]
        return any(model_name in self.config.model_params.model for model_name in tool_capable_models)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": "google",
            "model": self.config.model_params.model,
            "supports_tools": self.supports_tool_calling(),
            "supports_streaming": True,
            "supports_search": True
        }