"""Enhanced logging system for Qwen Python Agent."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class Logger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str = "pywen_agent", level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.logger.info(message, extra=extra or {})
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.logger.debug(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.logger.error(message, extra=extra or {})
    
    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any):
        """Log tool call with structured data."""
        self.info(f"Tool call: {tool_name}", extra={
            "tool_name": tool_name,
            "arguments": arguments,
            "result_type": type(result).__name__,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_llm_interaction(self, prompt: str, response: str, tokens_used: int):
        """Log LLM interaction."""
        self.info(f"LLM interaction - tokens: {tokens_used}", extra={
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_used": tokens_used,
            "timestamp": datetime.now().isoformat()
        })