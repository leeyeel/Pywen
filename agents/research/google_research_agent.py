from typing import Dict, Any, List, AsyncGenerator, Optional
from agents.base_agent import BaseAgent
from utils.llm_basics import LLMMessage, LLMResponse
from utils.tool_basics import ToolResult
from agents.research.research_prompts import (
    get_current_date,
    query_writer_instructions,
    web_search_executor_instructions,
    web_fetch_executor_instructions,
    summary_generator_instructions,
    reflection_instructions,
    answer_instructions
)
import json
import re
def _extract_json(content: str) -> str:

    # 使用正则表达式提取 ```json``` 代码块中的内容
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return content

class GeminiResearchDemo(BaseAgent):
    """Research agent specialized for multi-step research tasks."""
    
    def __init__(self, config, cli_console=None):
        super().__init__(config, cli_console)
        
        self.type = "GeminiResearchDemo"
        # Research state
        self.research_state = {
            "topic": "",
            "queries": [],
            "search_results": [],
            "summaries": [],
            "current_step": "query_generation",
            "iteration": 0
        }
        self.available_tools = self.tool_registry.list_tools()
    
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

    def _get_web_search_executor_prompt(self, queries: List[str]) -> str:
        """生成网络搜索提示"""
        # 将查询列表格式化为多行文本
        queries_text = "\n".join([f"- {query}" for query in queries])
        return web_search_executor_instructions.format(
            current_date=get_current_date(),
            research_topic=queries_text
        )

    def _get_web_fetch_executor_prompt(self, queries: List[str], web_search_results: List[ToolResult]) -> str:
        """生成网络抓取提示"""
        queries_text = "\n".join([f"- {query}" for query in queries])
        return web_fetch_executor_instructions.format(
            current_date=get_current_date(),
            web_search_results="\n".join([result for result in web_search_results]),
            research_topic=queries_text
        )
    def _get_summary_generator_prompt(self, topic: str, search_results: Optional[List[Any]] = None) -> str:
        """生成总结生成提示"""
        
        return summary_generator_instructions.format(
            research_topic=topic,
            search_results=search_results
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

    # TODO:
    async def run(self, user_message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run research agent with multi-step research workflow."""
        
        # 初始化研究状态
        self.research_state["topic"] = user_message
        self.research_state["iteration"] = 0
        yield {"type": "user_message", "data": {"message": user_message}}
        # Start trajectory recording
        self.trajectory_recorder.start_recording(
            task=user_message,
            provider=self.config.model_config.provider.value,
            model=self.config.model_config.model,
            max_steps=None
        )
        self.conversation_history.append(LLMMessage(role="user",content=user_message))
        # 1. 首次生成查询
        async for queries in self.generate_queries(user_message):
            # yield query事件给CLI消费
            yield queries

        # 2. 搜索
        async for search in self.web_search():
            # yield summary事件给CLI消费
            yield search

        # Google LangGraph 循环逻辑
        while True:
            try:
                
                # 3. 反思搜索结果
                yield {"type": "step", "data": {"step": "reflecting"}}
                reflection_prompt = self._get_reflection_prompt(self.research_state["summaries"])
                reflection_response = await self.llm_client.generate_response([LLMMessage(role="user", content=reflection_prompt)])
                # ```json输出格式
                # {{
                #     "is_sufficient": true, // or false
                #     "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
                #     "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
                # }}
                # ```

                # 添加reflection结果到对话历史
                self.conversation_history.append(LLMMessage(
                    role="assistant", 
                    content=reflection_response.content
                ))

                # 提取反思生成响应中的JSON数据
                reflection_data = json.loads(reflection_response.content)
                is_sufficient = reflection_data.get("is_sufficient", False)
                # 4.检查研究是否充分，如果充分就退出循环
                if is_sufficient:
                    break
                else:
                    ###继续处理follow_up_queries的搜索内容
                    follow_up_queries = reflection_data.get("follow_up_queries", [])
                    search_prompt = self._get_web_searcher_prompt(follow_up_queries)
                    search_response = await self.llm_client.generate_response([LLMMessage(role="user", content=search_prompt)])
                    
                    if search_response.tool_calls:
                        async for result in self._process_tool_calls(search_response.tool_calls):
                            yield result


            except Exception as e:
                yield {"type": "error", "data": {"error": str(e)}}
                break
        
        # 5. 提交最终答案
        yield {"type": "step", "data": {"step": "generating_final_answer"}}
        final_prompt = self._get_answer_prompt(user_message)
        final_response = await self.llm_client.generate_response([LLMMessage(role="user", content=final_prompt)])
        
        yield {"type": "final_answer", "data": {"content": final_response.content}}

    async def generate_queries(self,user_message: str):
        query_prompt = self._get_query_writer_prompt(user_message)
        query_response = await self.llm_client.generate_response([LLMMessage(role="user", content=query_prompt)],stream=False)

        json_content = _extract_json(query_response.content)
        try:
            query_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            yield {"type": "error", "data": {"error": f"Failed to parse reflection response JSON: {str(e)}. Content: {query_response.content}"}}
            return
        queries = query_data.get("query", [])

        yield {"type": "query", "data":{"queries": queries}}

        # 记录第一步的交互轨迹
        self.research_state["queries"] = queries
        self.conversation_history.append(LLMMessage(role="assistant",content=json_content))
        
        self.trajectory_recorder.record_llm_interaction(
            messages= self.conversation_history,
            response=LLMResponse(
                content=json_content,
                model=self.config.model_config.model,
                usage=query_response.usage,
                tool_calls=None,
            ),
            provider=self.config.model_config.provider.value,
            model="query_generator",
            current_task=user_message,
        )
    async def web_search(self):
        # 主要工作逻辑，先搜索找到相关的URLs，然后根据URL进行精读，最终生成总结

        # 1. 搜索URLs
        search_prompt = self._get_web_search_executor_prompt(self.research_state['queries'])
        search_response = await self.llm_client.generate_response(
            messages=[LLMMessage(role="user", content=search_prompt)],
            tools=self.availabe_tools
            )
        print(search_response)
        search_results = []
        if search_response.content:
            yield {"type": "search", "data": {"content": search_response.content}}
        if search_response.tool_calls:
            async for result in self._process_tool_calls(search_response.tool_calls):
                yield result
                search_results.append(result.metadata)

        # 2. 获取网页内容
        fetch_prompt = self._get_web_fetch_executor_prompt(self.research_state['queries'], search_results)
        fetch_response = await self.llm_client.generate_response([LLMMessage(role="user", content=fetch_prompt)])

        fetch_results = {}
        if fetch_response.content:
            yield {"type": "fetch", "data": {"content": fetch_response.content}}
        if fetch_response.tool_calls:
            async for result in self._process_tool_calls(fetch_response.tool_calls):
                yield result
                fetch_results[result.url] = result.result
        
        for query_result in search_results:
            for search_result in query_result:
                if search_result['url'] in fetch_results:
                    search_result['web_content'] = fetch_results[search_result['url']]
        # 3. 生成总结
        summary_prompt = self._get_summary_generator_prompt(self.research_state['queries'], search_results)
        summary_response = await self.llm_client.generate_response([LLMMessage(role="user", content=summary_prompt)])
        yield {"type": "summary", "data": {"content": summary_response.content}}
        self.research_state["summaries"] = summary_response.content
    async def _process_tool_calls(self, tool_calls):
        """处理工具调用"""
        # 执行工具
        results = await self.tool_executor.execute_tools(tool_calls) #ToolResult
        self.research_state["summaries"].extend(results)
        async for result in results:
            yield {"type": "tool_result", "data": {"result": result}}
            # 添加工具结果到对话历史
            self.conversation_history.append(LLMMessage(
                role= "tool",
                content= str(result),
                tool_call_id= result.tool_call_id
            )) 

    def get_enabled_tools(self) -> List[str]:
        """Return list of enabled tool names for ResearchAgent."""
        return [
            'web_search',
            'web_fetch', 
            'write_file',
            'read_file'
        ]
    
