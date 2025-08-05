from typing import Dict, Any, List, AsyncGenerator
from agents.base_agent import BaseAgent
from utils.llm_basics import LLMMessage
from agents.research.research_prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions
)

class ResearchAgent(BaseAgent):
    """Research agent specialized for multi-step research tasks."""
    
    def __init__(self, config, cli_console=None):
        super().__init__(config, cli_console)
        
        # Research state
        self.research_state = {
            "topic": "",
            "queries": [],
            "summaries": [],
            "current_step": "query_generation",
            "iteration": 0
        }
    
    def _build_system_prompt(self) -> str:
        """构建研究专用的系统提示"""
        current_date = get_current_date()
        
        return f"""You are an expert research assistant conducting comprehensive research. Today's date is {current_date}.

You have access to these tools:
- web_search: Search the web for information
- web_fetch: Fetch and read content from specific URLs  
- write_file: Save research findings and reports
- read_file: Read previously saved research files

Follow the research process step by step and use the appropriate prompts for each stage."""

    def _get_query_writer_prompt(self, topic: str, number_queries: int = 3) -> str:
        """生成查询生成提示"""
        return query_writer_instructions.format(
            current_date=get_current_date(),
            research_topic=topic,
            number_queries=number_queries
        )

    def _get_web_searcher_prompt(self, topic: str) -> str:
        """生成网络搜索提示"""
        return web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=topic
        )

    def _get_reflection_prompt(self, topic: str) -> str:
        """生成反思提示"""
        summaries = "\n".join(self.research_state.get("summaries", []))
        
        return reflection_instructions.format(
            research_topic=topic,
            summaries=summaries if summaries else "No summaries available yet. This indicates we need to conduct initial research."
        )

    def _get_answer_prompt(self, topic: str) -> str:
        """生成最终答案提示"""
        summaries = "\n".join(self.research_state.get("summaries", []))
        
        return answer_instructions.format(
            current_date=get_current_date(),
            research_topic=topic,
            summaries=summaries
        )

    async def run(self, user_message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run research agent with multi-step research workflow."""
        try:
            all_summaries = []
            # 处理用户输入，生成多个搜索请求

            # 处理搜索请求

            # 整合搜索结果

            # 反思搜索结果

            # 提交最终答案
            
        except Exception as e:
            yield {"type": "error", "data": {"error": str(e)}}

    
    async def _process_tool_calls(self, tool_calls):
        """处理工具调用"""
        for tool_call in tool_calls:
            yield {"type": "tool_call", "data": {"tool_name": tool_call.function.name}}
            
            # 执行工具
            result = await self.tool_executor.execute_tool_call(tool_call)
            
            yield {"type": "tool_result", "data": {"result": result}}
            
            # 添加工具结果到对话历史
            self.conversation_history.append(LLMMessage(
                role= "tool",
                content= str(result),
                tool_call_id= tool_call.id
            ))

    def _is_research_sufficient(self) -> bool:
        """检查研究是否充分，基于最后一次reflection的结果"""
        # 检查对话历史中最后一次assistant的回复
        for message in reversed(self.conversation_history):
            if message.get("role") == "assistant" and message.get("content"):
                content = message["content"]
                try:
                    # 尝试解析JSON格式的reflection结果
                    import re
                    import json
                    
                    # 查找JSON代码块
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        reflection_data = json.loads(json_match.group(1))
                        is_sufficient = reflection_data.get("is_sufficient", False)
                        
                        # 如果研究充分，更新研究状态
                        if is_sufficient:
                            self.research_state["current_step"] = "sufficient"
                        else:
                            # 如果不充分，保存follow_up_queries用于下一轮
                            follow_up_queries = reflection_data.get("follow_up_queries", [])
                            if follow_up_queries:
                                self.research_state["queries"].extend(follow_up_queries)
                        
                        return is_sufficient
                    
                    # 如果没有找到JSON格式，尝试直接解析
                    if '"is_sufficient"' in content:
                        if '"is_sufficient": true' in content or '"is_sufficient":true' in content:
                            return True
                        elif '"is_sufficient": false' in content or '"is_sufficient":false' in content:
                            return False
                            
                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    self.cli_console.print(f"[yellow]Warning: Failed to parse reflection result: {e}[/yellow]")
                    continue
        
        # 如果无法解析reflection结果，默认继续研究
        return False

    def get_enabled_tools(self) -> List[str]:
        """Return list of enabled tool names for ResearchAgent."""
        return [
            'web_search',
            'web_fetch', 
            'write_file',
            'read_file'
        ]
