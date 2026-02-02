"""LLM client for LangChain Cerebras with tool calling support."""

import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_cerebras import ChatCerebras

from modules.config import ConfigEnv

logger = logging.getLogger(__name__)


def serialize_for_json(obj: Any) -> Any:
    """Recursively serialize datetime objects and other non-JSON types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, "isoformat"):  # Handle other date/time types
        return obj.isoformat()
    return obj


class LLMClient:
    """Client for interacting with Cerebras via LangChain."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        region_name: Optional[str] = None,  # Kept for backward compatibility, unused
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: Cerebras API key. If not provided, reads from CEREBRAS_API_KEY env var.
            model_name: Name of the Cerebras model. If not provided, reads from CEREBRAS_MODEL_ID env var.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens in response.
            region_name: Unused. Present for backward compatibility.
        """
        self.api_key = api_key or ConfigEnv.CEREBRAS_API_KEY
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY must be set in environment variables or provided")
        self.model_name = model_name or ConfigEnv.CEREBRAS_MODEL_ID
        if not self.model_name:
            raise ValueError("CEREBRAS_MODEL_ID must be set in environment variables or provided")
        self.temperature = temperature
        self.max_tokens = max_tokens or ConfigEnv.CEREBRAS_MAX_TOKENS

        self.model = ChatCerebras(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _build_messages(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        skip_prompt_append: bool = False,
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = []

        if conversation_history:
            for item in conversation_history:
                role = item.get("role")
                content = item.get("content") or item.get("text", "")
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))

        # Only append prompt if not already the last message and not skipped
        if not skip_prompt_append:
            # Check if the prompt is already the last message
            if messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == prompt:
                pass  # Skip, prompt already in conversation_history
            else:
                messages.append(HumanMessage(content=prompt))
        return messages

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        return ""
    
    async def generate_with_tools(
        self,
        prompt: str,
        tool_schemas: List[Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response with tool calling support using LangChain.
        
        Args:
            prompt: The user prompt.
            tool_schemas: List of available tool schemas.
            conversation_history: Optional conversation history (not used in initial call).
            
        Returns:
            Dictionary containing the response, tool calls, and interaction ID.
        """
        try:
            tools = tool_schemas or []
            logger.info(f"[LLMClient] Number of tools available: {len(tools)}")
            for t in tools:
                logger.info(f"[LLMClient] Tool: {t.name} - {t.description[:50]}...")
            
            llm = self.model.bind_tools(tools) if tools else self.model

            messages = self._build_messages(prompt, conversation_history)
            logger.info(f"[LLMClient] Messages to LLM: {[type(m).__name__ for m in messages]}")
            for m in messages:
                logger.info(f"[LLMClient] {type(m).__name__}: {str(m.content)[:100]}...")
            
            response = await llm.ainvoke(messages)
            logger.info(f"[LLMClient] Response tool_calls: {response.tool_calls}")
            logger.info(f"[LLMClient] Response content: {str(response.content)[:200]}")

            tool_calls = []
            for tool_call in response.tool_calls or []:
                tool_calls.append({
                    "name": tool_call.get("name"),
                    "arguments": tool_call.get("args", {}),
                    "call_id": tool_call.get("id"),
                })

            return {
                "text": response.content or "",
                "tool_calls": tool_calls,
                "messages": [*messages, response],
                "interaction_id": getattr(response, "id", None),
            }

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise
    
    async def generate_final_response(
        self,
        prompt: str,
        tool_results: List[Dict[str, Any]],
        interaction_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        messages: Optional[List[BaseMessage]] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """
        Generate final response after tool execution using LangChain.
        
        Args:
            prompt: Original user prompt.
            tool_results: Results from tool executions.
            interaction_id: ID from the previous interaction.
            conversation_history: Optional conversation history (not used with interaction_id).
            
        Returns:
            Final text response.
        """
        try:
            base_messages = messages or self._build_messages(prompt, conversation_history)
            tool_messages: List[ToolMessage] = []

            for tool_result in tool_results:
                tool_messages.append(
                    ToolMessage(
                        name=tool_result.get("tool_name", ""),
                        content=json.dumps(serialize_for_json(tool_result.get("result", {}))),
                        tool_call_id=tool_result.get("call_id") or "",
                    )
                )

            llm = self.model.bind_tools(tools) if tools else self.model
            response = await llm.ainvoke([*base_messages, *tool_messages])
            return self._normalize_content(response.content)

        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            raise

    async def stream_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        messages: Optional[List[BaseMessage]] = None,
        tools: Optional[List[Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response text.

        Args:
            prompt: The user prompt.
            conversation_history: Optional conversation history.
            messages: Optional pre-built messages.
            tools: Optional tools to bind for model context.
            tool_results: Optional tool results to include as ToolMessages.

        Yields:
            Text chunks as they are generated.
        """
        try:
            base_messages = messages or self._build_messages(prompt, conversation_history)
            tool_messages: List[ToolMessage] = []

            if tool_results:
                for tool_result in tool_results:
                    tool_messages.append(
                        ToolMessage(
                            name=tool_result.get("tool_name", ""),
                            content=json.dumps(serialize_for_json(tool_result.get("result", {}))),
                            tool_call_id=tool_result.get("call_id") or "",
                        )
                    )

            llm = self.model.bind_tools(tools) if tools else self.model

            async for chunk in llm.astream([*base_messages, *tool_messages]):
                normalized = self._normalize_content(chunk.content)
                if normalized:
                    yield normalized

        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            raise
