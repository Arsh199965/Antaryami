"""Main response pipeline orchestrator - simplified single LLM call approach."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from modules.config import ConfigEnv
from .llm_client import LLMClient
from .tool_registry import get_registry
from .prompts import build_system_prompt
from .prompts_fast import build_fast_system_prompt, get_filler_response

logger = logging.getLogger(__name__)


class ResponsePipeline:
    """
    Simplified pipeline for processing text through LLM with tools.
    
    Flow:
    1. Build system prompt with all tools
    2. Single LLM call (LLM decides which tools to use)
    3. Execute tools if requested
    4. Generate/stream final response
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize the response pipeline.
        
        Args:
            api_key: Cerebras API key.
            model_name: Name of the model. If not provided, reads from CEREBRAS_MODEL_ID env var.
            temperature: Sampling temperature for LLM.
        """
        _model_name = model_name or ConfigEnv.CEREBRAS_MODEL_ID
        if not _model_name:
            raise ValueError("CEREBRAS_MODEL_ID must be set in environment variables or provided")
        
        self.llm_client = LLMClient(
            api_key=api_key,
            model_name=_model_name,
            temperature=temperature,
        )
        self.tool_registry = get_registry()
        logger.info(f"✓ ResponsePipeline initialized with model: {_model_name}")
    
    async def process_text(
        self,
        text: str,
        conversation_history: list = None,
        user_id: Optional[str] = None,
        is_twilio_call: bool = False,
    ) -> Dict[str, Any]:
        """
        Process text through the pipeline (non-streaming).

        Args:
            text: Input text to process.
            conversation_history: Previous conversation for context.
            user_id: Optional; used for tool calls requiring user identification.
            is_twilio_call: Whether this is a Twilio phone call (no GPS available).

        Returns:
            Dictionary containing the final response and metadata.
        """
        try:
            logger.info(f"Processing text: {text[:100]}...")
            logger.info(f"Context: user_id={user_id}, is_twilio_call={is_twilio_call}")

            # Get all tools - let LLM decide which to use
            langchain_tools = self.tool_registry.get_langchain_tools()
            logger.info(f"Loaded {len(langchain_tools)} tools")

            # Build system prompt with user context and tool instructions
            system_prompt = build_system_prompt(
                user_id=user_id, 
                tools=langchain_tools,
                is_twilio_call=is_twilio_call,
            )
            tool_conversation = [{"role": "system", "content": system_prompt}]
            tool_conversation.extend(conversation_history or [])

            # Single LLM call - let LLM understand intent and decide on tools
            logger.info("Calling LLM with tools...")
            llm_response = await self.llm_client.generate_with_tools(
                prompt=text,
                tool_schemas=langchain_tools,
                conversation_history=tool_conversation,
            )
            logger.info(f"LLM response received. Tool calls: {len(llm_response.get('tool_calls', []))}")
            
            # Execute tools if LLM requested any
            tool_results = []
            if llm_response.get("tool_calls"):
                logger.info(f"Executing {len(llm_response['tool_calls'])} tool(s)...")
                tool_calls = llm_response["tool_calls"]

                async def _run_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
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

                tool_results = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])
            
            # Generate final response
            if tool_results:
                final_response = await self.llm_client.generate_final_response(
                    prompt=text,
                    tool_results=tool_results,
                    messages=llm_response.get("messages"),
                    tools=langchain_tools,
                )
            else:
                final_response = llm_response.get("text", "")
            
            logger.info("Pipeline complete")
            
            return {
                "response": final_response,
                "tool_calls": [tc.get("name") for tc in llm_response.get("tool_calls", [])],
                "tool_results": tool_results,
                "metadata": {
                    "model": self.llm_client.model_name,
                    "temperature": self.llm_client.temperature,
                },
            }
            
        except Exception as e:
            logger.error(f"Error in response pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "error": str(e),
                "tool_calls": [],
                "tool_results": [],
            }

    async def process_text_streaming(
        self,
        text: str,
        conversation_history: list = None,
        user_id: Optional[str] = None,
        is_twilio_call: bool = False,
        fast_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Process text and stream the final LLM response.

        Args:
            text: Input text to process.
            conversation_history: Previous conversation for context.
            user_id: Optional; used for tool calls requiring user identification.
            is_twilio_call: Whether this is a Twilio phone call.
            fast_mode: Use optimized shorter prompts for faster inference (default: True).

        Returns:
            Dictionary containing metadata and an async generator under "stream".
        """
        try:
            logger.info(f"Processing text (streaming): {text[:100]}...")
            logger.info(f"Context: user_id={user_id}, is_twilio_call={is_twilio_call}, fast_mode={fast_mode}")

            # Get all tools - let LLM decide which to use
            langchain_tools = self.tool_registry.get_langchain_tools()
            logger.info(f"Loaded {len(langchain_tools)} tools")

            # Build system prompt - use fast mode for lower latency
            if fast_mode:
                system_prompt = build_fast_system_prompt(
                    user_id=user_id,
                    tools=langchain_tools,
                    is_twilio_call=is_twilio_call,
                )
            else:
                system_prompt = build_system_prompt(
                    user_id=user_id, 
                    tools=langchain_tools,
                    is_twilio_call=is_twilio_call,
                )
            logger.info(f"System prompt built ({len(system_prompt)} chars, fast_mode={fast_mode})")
            
            # Limit conversation history to reduce prompt size (keep last 6 turns max)
            limited_history = (conversation_history or [])[-6:]
            tool_conversation = [{"role": "system", "content": system_prompt}]
            tool_conversation.extend(limited_history)

            # Single LLM call - let LLM understand intent and decide on tools
            logger.info("Calling LLM with tools...")
            try:
                llm_response = await self.llm_client.generate_with_tools(
                    prompt=text,
                    tool_schemas=langchain_tools,
                    conversation_history=tool_conversation,
                )
                logger.info(f"LLM response received. Tool calls: {len(llm_response.get('tool_calls', []))}")
            except Exception as llm_err:
                logger.error(f"LLM call failed: {llm_err}")
                import traceback
                traceback.print_exc()
                raise

            # Execute tools if LLM requested any
            tool_results = []
            if llm_response.get("tool_calls"):
                logger.info(f"Executing {len(llm_response['tool_calls'])} tool(s)...")
                tool_calls = llm_response["tool_calls"]

                async def _run_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
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

                tool_results = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])

            # Stream final response
            logger.info("Streaming final response...")
            if tool_results:
                # Stream response with tool results
                stream = self.llm_client.stream_response(
                    prompt=text,
                    tool_results=tool_results,
                    messages=llm_response.get("messages"),
                    tools=langchain_tools,
                )
            else:
                # No tools called - yield the pre-generated response word-by-word
                already_generated_text = llm_response.get("text", "")
                logger.info(f"No tools called, streaming pre-generated response ({len(already_generated_text)} chars)")
                
                async def yield_existing_response():
                    """Yield the already-generated response as a stream"""
                    if already_generated_text:
                        words = already_generated_text.split(' ')
                        for i, word in enumerate(words):
                            if i > 0:
                                yield ' ' + word
                            else:
                                yield word
                    
                stream = yield_existing_response()

            # Include filler for caller to play while waiting (if tools were used)
            filler = get_filler_response("hi") if tool_results else None

            return {
                "stream": stream,
                "tool_calls": [tc.get("name") for tc in llm_response.get("tool_calls", [])],
                "tool_results": tool_results,
                "filler": filler,  # Caller can play this while tools executed
                "metadata": {
                    "model": self.llm_client.model_name,
                    "temperature": self.llm_client.temperature,
                },
            }

        except Exception as e:
            logger.error(f"Error in streaming response pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stream": None,
                "response": f"I encountered an error processing your request: {str(e)}",
                "error": str(e),
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
