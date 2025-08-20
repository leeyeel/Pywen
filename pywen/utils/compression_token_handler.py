"""
Compression Token Handler - Python version of Kode's approach
Handles token counting for compressed conversations intelligently
"""

from typing import List, Optional
from pywen.utils.llm_basics import LLMMessage, LLMUsage, LLMResponse


def create_compressed_conversation(
    summary_response: LLMResponse,
    recovered_files: List[dict] = None
) -> List[LLMMessage]:
    """
    Create compressed conversation following Kode's approach.
    
    Args:
        summary_response: LLM response containing the conversation summary
        recovered_files: Optional list of recovered development files
        
    Returns:
        List of compressed messages with proper token handling
    """
    # Reset usage info following Kode's strategy
    if summary_response.usage:
        # Keep the actual output tokens from the summary generation
        # Reset input tokens since context has changed
        summary_response.usage = LLMUsage(
            input_tokens=0,
            output_tokens=summary_response.usage.output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            total_tokens=summary_response.usage.output_tokens
        )
    
    # Create the compressed message list
    compressed_messages = [
        LLMMessage(
            role="user",
            content="Context automatically compressed due to token limit. Essential information preserved."
        ),
        LLMMessage(
            role="assistant",
            content=summary_response.content,
            usage=summary_response.usage
        )
    ]
    
    # Add recovered files if any (following Kode's file recovery pattern)
    if recovered_files:
        for file_info in recovered_files:
            file_path = file_info.get('path', 'unknown')
            file_content = file_info.get('content', '')
            file_tokens = file_info.get('tokens', 0)
            is_truncated = file_info.get('truncated', False)
            
            # Add line numbers like Kode does
            content_with_lines = add_line_numbers(file_content)
            
            recovery_message = LLMMessage(
                role="user",
                content=(
                    f"**Recovered File: {file_path}**\n\n"
                    f"```\n{content_with_lines}\n```\n\n"
                    f"*Automatically recovered ({file_tokens} tokens)"
                    f"{' [truncated]' if is_truncated else ''}*"
                )
            )
            compressed_messages.append(recovery_message)
    
    return compressed_messages


def add_line_numbers(content: str, start_line: int = 1) -> str:
    """
    Add line numbers to content, similar to Kode's addLineNumbers.
    
    Args:
        content: File content to add line numbers to
        start_line: Starting line number
        
    Returns:
        Content with line numbers added
    """
    if not content:
        return content
    
    lines = content.split('\n')
    numbered_lines = []
    
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:4d} | {line}")
    
    return '\n'.join(numbered_lines)


def estimate_compressed_tokens(compressed_messages: List[LLMMessage]) -> int:
    """
    Estimate tokens for compressed conversation.
    Uses Kode's approach: rely on API usage when available, estimate otherwise.
    
    Args:
        compressed_messages: List of compressed messages
        
    Returns:
        Estimated token count
    """
    # Try to get from the most recent assistant message with usage
    for message in reversed(compressed_messages):
        if (
            message.role == "assistant" and 
            hasattr(message, 'usage') and 
            message.usage and
            message.usage.output_tokens > 0
        ):
            # Use the actual API token count
            return (
                message.usage.input_tokens +
                message.usage.output_tokens +
                getattr(message.usage, 'cache_creation_input_tokens', 0) +
                getattr(message.usage, 'cache_read_input_tokens', 0)
            )
    
    # Fallback: estimate from content length
    total_chars = sum(len(msg.content or '') for msg in compressed_messages)
    
    # Conservative estimate for compressed content
    # Compressed summaries tend to be dense, so higher token ratio
    return max(1, total_chars // 2)


def handle_post_compression_tokens(
    compressed_messages: List[LLMMessage],
    original_token_count: int
) -> dict:
    """
    Handle token counting after compression, following Kode's philosophy.
    
    Args:
        compressed_messages: The compressed conversation
        original_token_count: Token count before compression
        
    Returns:
        Dictionary with token information
    """
    # Estimate current tokens
    current_tokens = estimate_compressed_tokens(compressed_messages)
    
    # Calculate compression ratio
    compression_ratio = current_tokens / max(original_token_count, 1)
    tokens_saved = original_token_count - current_tokens
    
    return {
        'original_tokens': original_token_count,
        'compressed_tokens': current_tokens,
        'tokens_saved': tokens_saved,
        'compression_ratio': compression_ratio,
        'compression_percentage': f"{(1 - compression_ratio) * 100:.1f}%",
        'method': 'api_usage_based' if any(
            hasattr(msg, 'usage') and msg.usage for msg in compressed_messages
        ) else 'estimated'
    }


class CompressionTokenManager:
    """
    Manages token counting for compressed conversations.
    Follows Kode's lightweight and intelligent approach.
    """
    
    def __init__(self):
        self.compression_history = []
    
    def compress_and_track(
        self,
        original_messages: List[LLMMessage],
        summary_response: LLMResponse,
        recovered_files: List[dict] = None
    ) -> tuple[List[LLMMessage], dict]:
        """
        Compress conversation and track token changes.
        
        Returns:
            Tuple of (compressed_messages, token_info)
        """
        # Calculate original token count
        original_tokens = self._count_tokens_from_messages(original_messages)
        
        # Create compressed conversation
        compressed_messages = create_compressed_conversation(
            summary_response, 
            recovered_files
        )
        
        # Handle token tracking
        token_info = handle_post_compression_tokens(
            compressed_messages,
            original_tokens
        )
        
        # Track compression history
        self.compression_history.append({
            'timestamp': __import__('time').time(),
            'original_message_count': len(original_messages),
            'compressed_message_count': len(compressed_messages),
            **token_info
        })
        
        return compressed_messages, token_info
    
    def _count_tokens_from_messages(self, messages: List[LLMMessage]) -> int:
        """Count tokens from messages using Kode's approach."""
        # Import here to avoid circular imports
        from pywen.utils.tokens import count_tokens
        return count_tokens(messages)
    
    def get_compression_stats(self) -> dict:
        """Get overall compression statistics."""
        if not self.compression_history:
            return {'total_compressions': 0}
        
        total_compressions = len(self.compression_history)
        total_tokens_saved = sum(h['tokens_saved'] for h in self.compression_history)
        avg_compression_ratio = sum(h['compression_ratio'] for h in self.compression_history) / total_compressions
        
        return {
            'total_compressions': total_compressions,
            'total_tokens_saved': total_tokens_saved,
            'average_compression_ratio': avg_compression_ratio,
            'average_compression_percentage': f"{(1 - avg_compression_ratio) * 100:.1f}%",
            'latest_compression': self.compression_history[-1] if self.compression_history else None
        }


# Example usage:
"""
# Replace heavy tokenizer approach:
# tokenizer.encode(compressed_text)  # Heavy, inaccurate

# Use Kode's approach:
manager = CompressionTokenManager()
compressed_messages, token_info = manager.compress_and_track(
    original_messages, 
    summary_response
)

print(f"Compression saved {token_info['tokens_saved']} tokens "
      f"({token_info['compression_percentage']})")
"""
