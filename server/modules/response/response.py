"""Main response pipeline orchestrator."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator, Callable, Awaitable, List
import uuid
from datetime import datetime, timezone
from pathlib import Path

from db.connection import get_db
from modules.config import ConfigEnv
from .intent_detector import IntentDetector
from .llm_client import LLMClient
from .tool_registry import get_registry
from langchain_core.messages import SystemMessage

from .prompts import build_system_prompt, build_router_system_prompt, parse_intent_from_router_response
from .intent_tools_mapping import derive_intent_from_tool_calls

logger = logging.getLogger(__name__)


class ResponsePipeline:
    """Main pipeline for processing text through intent detection, LLM, and tool calls."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize the response pipeline.
        
        Args:
            api_key: Google Gemini API key.
            model_name: Name of the Bedrock model. If not provided, reads from BEDROCK_MODEL_ID env var.
            temperature: Sampling temperature for LLM.
        """
        _model_name = model_name or ConfigEnv.BEDROCK_MODEL_ID
        if not _model_name:
            raise ValueError("BEDROCK_MODEL_ID must be set in environment variables or provided")
        self.intent_detector = IntentDetector(api_key=api_key)
        self.llm_client = LLMClient(
            api_key=api_key,
            model_name=_model_name,
            temperature=temperature,
        )
        self.tool_registry = get_registry()
    
    async def process_text(
        self,
        text: str,
        conversation_history: list = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_twilio_call: bool = False,
        on_before_tool_execution: Optional[Callable[[List[Dict[str, Any]]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Process text through the complete pipeline.

        Pipeline flow (2 AI calls max):
        1. LLM Processing with broad tool set (intent + tool decision in one call)
        2. Tool Execution (if needed), optionally after on_before_tool_execution (e.g. play tool-wait audio)
        3. LLM final response (only if tools were called)
        4. Text Output

        Args:
            text: Input text to process.
            conversation_history: Previous conversation for context.
            session_id: Optional; when set, conversation and intent_log are persisted.
            user_id: Optional; used for conversation.user_id when persisting.
            is_twilio_call: Whether this is a Twilio phone call (no GPS available).
            on_before_tool_execution: Optional async callback(tool_calls) invoked before running tools (e.g. to play tool-wait audio).

        Returns:
            Dictionary containing the final response and metadata.
        """
        try:
            logger.info(f"Processing text: {text[:100]}...")
            logger.info(f"Context: user_id={user_id}, is_twilio_call={is_twilio_call}")

            # Build context from conversation history
            if conversation_history:
                history_text = "\n".join(
                    [f"{msg['role'].upper()}: {msg['text']}" for msg in conversation_history]
                )
                logger.info(f"Added conversation context ({len(conversation_history)} turns)")

            # Step 1: Router LLM (intent + tool decision in one call; use router prompt)
            logger.info("Step 1: Router LLM with tools (intent + tool_calls)")
            langchain_tools = self.tool_registry.get_langchain_tools(intent="general")
            logger.info(f"Loaded {len(langchain_tools)} tools for general intent")

            router_prompt = build_router_system_prompt(
                user_id=user_id,
                tools=langchain_tools,
                is_twilio_call=is_twilio_call,
            )
            tool_conversation = [{"role": "system", "content": router_prompt}]
            tool_conversation.extend(conversation_history or [])

            llm_response = await self.llm_client.generate_with_tools(
                prompt=text,
                tool_schemas=langchain_tools,
                conversation_history=tool_conversation,
                use_fast_model=True,
            )

            # Parse intent from router response text; fallback to tool names if not found
            tool_names = [tc.get("name") for tc in llm_response.get("tool_calls", [])]
            router_text = (llm_response.get("text") or "").strip()
            parsed_intent = parse_intent_from_router_response(router_text, fallback_intent="general")
            if parsed_intent == "general" and tool_names:
                parsed_intent = derive_intent_from_tool_calls(tool_names)
            intent_result = {
                "intent": parsed_intent,
                "confidence": 1.0,
                "reasoning": "from router prompt" if router_text else "fallback from tool names",
            }
            logger.info(f"Intent for logging: {parsed_intent}")

            # Persist conversation and intent_log in background (don't block pipeline)
            if session_id:
                async def _persist_conversation_and_intent():
                    try:
                        db = get_db()
                        now = datetime.now(timezone.utc)
                        await db.conversations.update_one(
                            {"session_id": session_id},
                            {
                                "$setOnInsert": {
                                    "session_id": session_id,
                                    "user_id": user_id or "unknown",
                                    "language": "en",
                                    "start_time": now,
                                    "end_time": None,
                                    "outcome": None,
                                }
                            },
                            upsert=True,
                        )
                        await db.intent_logs.insert_one({
                            "intent_id": str(uuid.uuid4()),
                            "session_id": session_id,
                            "intent_name": intent_result.get("intent", "general"),
                            "confidence": float(intent_result.get("confidence", 0)),
                        })
                    except Exception as e:
                        logger.warning(f"Failed to persist conversation/intent_log: {e}")
                asyncio.create_task(_persist_conversation_and_intent())

            # Step 2: Tool Execution (if LLM requested tools) — optionally run on_before_tool_execution, then run in parallel
            tool_results = []
            if llm_response.get("tool_calls"):
                tool_calls = llm_response["tool_calls"]
                logger.info(f"Step 2: Executing {len(tool_calls)} tool(s) in parallel")
                if on_before_tool_execution:
                    await on_before_tool_execution(tool_calls)

                async def _run_one_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})
                    call_id = tool_call.get("call_id")
                    try:
                        result = await self.tool_registry.execute_tool(
                            name=tool_name,
                            arguments=tool_args,
                        )
                        logger.info(f"Tool {tool_name} executed successfully")
                        return {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "call_id": call_id,
                            "result": result,
                        }
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        return {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "call_id": call_id,
                            "result": {"status": "error", "error": str(e)},
                        }

                tool_results = await asyncio.gather(*[_run_one_tool(tc) for tc in tool_calls])
            
            # Step 3: Generate final LLM response (only if tools were called); use main system prompt
            logger.info("Step 3: Generating final response")
            if tool_results:
                # Use main (full) system prompt for final response; replace router prompt in messages
                main_system = SystemMessage(
                    content=build_system_prompt(
                        user_id=user_id,
                        tools=langchain_tools,
                        is_twilio_call=is_twilio_call,
                    )
                )
                router_messages = llm_response.get("messages") or []
                base_for_final = [main_system] + list(router_messages[1:])
                final_response = await self.llm_client.generate_final_response(
                    prompt=text,
                    tool_results=tool_results,
                    interaction_id=llm_response.get("interaction_id"),
                    messages=base_for_final,
                    tools=langchain_tools,
                )
            else:
                # If no tools were called, use the initial response
                final_response = llm_response.get("text", "")
            
            # Step 4: Return text output
            logger.info("Step 4: Pipeline complete")
            
            return {
                "response": final_response,
                "intent": intent_result,
                "tool_calls": [tc.get("name") for tc in llm_response.get("tool_calls", [])],
                "tool_results": tool_results,
                "metadata": {
                    "model": self.llm_client.model_name,
                    "temperature": self.llm_client.temperature,
                },
            }
            
        except Exception as e:
            logger.error(f"Error in response pipeline: {e}")
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "error": str(e),
                "intent": {"intent": "general", "confidence": 0.0},
                "tool_calls": [],
                "tool_results": [],
            }

    async def process_text_streaming(
        self,
        text: str,
        conversation_history: list = None,
        user_id: Optional[str] = None,
        is_twilio_call: bool = False,
        on_before_tool_execution: Optional[Callable[[List[Dict[str, Any]]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Process text and stream the final LLM response (2 AI calls max).

        Args:
            text: Input text to process.
            conversation_history: Previous conversation for context.
            user_id: Optional; used for tool calls requiring user identification.
            is_twilio_call: Whether this is a Twilio phone call (no GPS available).
            on_before_tool_execution: Optional async callback(tool_calls) invoked before running tools (e.g. play tool-wait audio).

        Returns:
            Dictionary containing metadata and an async generator under "stream".
        """
        try:
            logger.info(f"Processing text (streaming): {text[:100]}...")
            logger.info(f"[Pipeline] user_id received: {user_id}, is_twilio_call: {is_twilio_call}")

            # Step 1: Router LLM (intent + tool decision in one call; use router prompt + fast model)
            logger.info("Step 1: Router LLM with tools (intent + tool_calls)")
            langchain_tools = self.tool_registry.get_langchain_tools(intent="general")
            logger.info(f"Loaded {len(langchain_tools)} tools for general intent")

            router_prompt = build_router_system_prompt(
                user_id=user_id,
                tools=langchain_tools,
                is_twilio_call=is_twilio_call,
            )
            logger.info(f"[Pipeline] Router prompt (first 500 chars): {router_prompt[:500]}...")
            tool_conversation = [{"role": "system", "content": router_prompt}]
            tool_conversation.extend(conversation_history or [])

            llm_response = await self.llm_client.generate_with_tools(
                prompt=text,
                tool_schemas=langchain_tools,
                conversation_history=tool_conversation,
                use_fast_model=True,
            )

            # Parse intent from router response text; fallback to tool names for logging
            tool_names = [tc.get("name") for tc in llm_response.get("tool_calls", [])]
            router_text = (llm_response.get("text") or "").strip()
            parsed_intent = parse_intent_from_router_response(router_text, fallback_intent="general")
            if parsed_intent == "general" and tool_names:
                parsed_intent = derive_intent_from_tool_calls(tool_names)
            intent_result = {
                "intent": parsed_intent,
                "confidence": 1.0,
                "reasoning": "from router prompt" if router_text else "fallback from tool names",
            }
            logger.info(f"Intent for logging: {parsed_intent}")

            # Step 2: Tool Execution (if LLM requested tools) — optionally on_before_tool_execution, then run in parallel
            tool_results = []
            if llm_response.get("tool_calls"):
                logger.info(f"Step 2: Executing {len(llm_response['tool_calls'])} tool(s) in parallel")
                tool_calls = llm_response["tool_calls"]
                if on_before_tool_execution:
                    await on_before_tool_execution(tool_calls)

                async def _run_one_tool_stream(tool_call: Dict[str, Any]) -> Dict[str, Any]:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})
                    call_id = tool_call.get("call_id")
                    try:
                        result = await self.tool_registry.execute_tool(
                            name=tool_name,
                            arguments=tool_args,
                        )
                        logger.info(f"Tool {tool_name} executed successfully")
                        return {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "call_id": call_id,
                            "result": result,
                        }
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        return {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "call_id": call_id,
                            "result": {"status": "error", "error": str(e)},
                        }

                tool_results = await asyncio.gather(*[_run_one_tool_stream(tc) for tc in tool_calls])

            # Step 3: Stream final LLM response; use main system prompt when tools were called
            logger.info("Step 3: Streaming final response")
            if tool_results:
                main_system = SystemMessage(
                    content=build_system_prompt(
                        user_id=user_id,
                        tools=langchain_tools,
                        is_twilio_call=is_twilio_call,
                    )
                )
                router_messages = llm_response.get("messages") or []
                base_for_final = [main_system] + list(router_messages[1:])
                stream = self.llm_client.stream_response(
                    prompt=text,
                    tool_results=tool_results,
                    messages=base_for_final,
                    tools=langchain_tools,
                )
            else:
                stream = self.llm_client.stream_response(
                    prompt=text,
                    conversation_history=conversation_history,
                    tools=None,
                )

            return {
                "stream": stream,
                "intent": intent_result,
                "tool_calls": [tc.get("name") for tc in llm_response.get("tool_calls", [])],
                "tool_results": tool_results,
                "metadata": {
                    "model": self.llm_client.model_name,
                    "temperature": self.llm_client.temperature,
                },
            }

        except Exception as e:
            logger.error(f"Error in streaming response pipeline: {e}")
            return {
                "stream": None,
                "response": f"I encountered an error processing your request: {str(e)}",
                "error": str(e),
                "intent": {"intent": "general", "confidence": 0.0},
                "tool_calls": [],
                "tool_results": [],
            }
    
    def load_test_prompts(self, file_path: Optional[str] = None) -> list:
        """
        Load test prompts from JSON file.
        
        Args:
            file_path: Path to test prompts file. Defaults to data/test-prompts.json.
            
        Returns:
            List of test prompts.
        """
        if file_path is None:
            # Default to data/test-prompts.json relative to this file
            current_dir = Path(__file__).parent
            file_path = current_dir / "data" / "test-prompts.json"
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "prompts" in data:
                    return data["prompts"]
                elif isinstance(data, dict) and "prompt" in data:
                    return [data["prompt"]]
                else:
                    return [str(data)]
        except FileNotFoundError:
            logger.warning(f"Test prompts file not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing test prompts JSON: {e}")
            return []


# Global pipeline instance
_pipeline: Optional[ResponsePipeline] = None


def get_pipeline(api_key: Optional[str] = None) -> ResponsePipeline:
    """
    Get the global response pipeline instance.
    
    Args:
        api_key: Optional API key to initialize pipeline.
        
    Returns:
        The global ResponsePipeline instance.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = ResponsePipeline(api_key=api_key)
    return _pipeline
