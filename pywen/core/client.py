"""LLM client factory using the new architecture."""

import sys
from pathlib import Path
from typing import List, Optional, Union, AsyncGenerator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pywen.config.config import Config, ModelConfig, ModelProvider
from pywen.utils.llm_config import Config as UtilsConfig, AuthType, ModelParameters
from pywen.utils.llm_client import LLMClient as UtilsLLMClient
from pywen.utils.llm_basics import LLMMessage, LLMResponse
from pywen.tools.base import Tool


class LLMClient:
    """Factory and wrapper for LLM clients."""
    
    def __init__(self, config: Union[Config, ModelConfig]):
        # Handle both Config and ModelConfig inputs
        if isinstance(config, ModelConfig):
            self.model_config = config
        else:
            self.model_config = config.model_config
        
        # Convert to utils config format
        self.utils_config = self._convert_config(self.model_config)
        self.client = UtilsLLMClient(self.utils_config)
    
    def _convert_config(self, model_config: ModelConfig) -> UtilsConfig:
        """Convert ModelConfig to utils config format."""
        
        # Determine auth type based on provider
        if model_config.provider == ModelProvider.QWEN:
            auth_type = AuthType.API_KEY
        elif model_config.provider == ModelProvider.OPENAI:
            auth_type = AuthType.OPENAI
        else:
            auth_type = AuthType.API_KEY
        
        # Create model parameters
        model_params = ModelParameters(
            model=model_config.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            base_url=model_config.base_url
        )
        
        return UtilsConfig(
            auth_type=auth_type,
            api_key=model_config.api_key,
            model_params=model_params
        )
    
    async def generate_response(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate response using the underlying client."""
        
        # Convert tools to the format expected by utils client
        utils_tools = None
        if tools:
            utils_tools = []
            for tool in tools:
                utils_tool = type('UtilsTool', (), {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters
                })()
                utils_tools.append(utils_tool)
        
        return await self.client.generate_response(
            messages=messages,
            tools=utils_tools,
            stream=stream
        )
    
    async def generate_with_retry(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        max_retries: int = 3
    ) -> LLMResponse:
        """Generate response with retry logic."""
        return await self.client.generate_with_retry(
            messages=messages,
            tools=tools,
            max_retries=max_retries
        )
    
    def get_statistics(self):
        """Get client statistics."""
        return self.client.get_statistics()
    
    @staticmethod
    def create(model_config: ModelConfig) -> "LLMClient":
        """Create LLM client based on model configuration."""
        return LLMClient(model_config)




