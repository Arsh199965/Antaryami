"""
LLM Service - Language Model Processing
Handles LLM interactions for processing transcripts using the response pipeline
"""
import logging
from typing import Callable, Awaitable, List, Dict, Any, Optional

from modules.config import ConfigEnv
from modules.response.response import ResponsePipeline

logger = logging.getLogger(__name__)


# =========================
# LLM Service
# =========================
class LLMService:
    """Process transcripts with LLM using the response pipeline"""

    def __init__(self):
        """Initialize LLM service with response pipeline"""
        self.api_key = ConfigEnv.BEDROCK_API_KEY
        # Bedrock uses AWS credentials; keep api_key for backward compatibility only

        self.pipeline = ResponsePipeline(
            api_key=self.api_key,
            model_name=ConfigEnv.BEDROCK_MODEL_ID,
            temperature=ConfigEnv.BEDROCK_TEMPERATURE,
        )
        logger.info("✓ LLM Service initialized with response pipeline")
    
    async def process(
        self,
        transcript: str,
        conversation_history: list = None,
        session_id: str = None,
        user_id: str = None,
        is_twilio_call: bool = False,
        on_before_tool_execution: Optional[Callable[[List[Dict[str, Any]]], Awaitable[None]]] = None,
    ) -> dict:
        """
        Process transcript through the complete pipeline (2 AI calls max when tools used).

        Pipeline flow:
        1. LLM with broad tools (intent + tool decision in one call)
        2. Tool Execution (if needed), optionally after on_before_tool_execution (e.g. play tool-wait audio)
        3. LLM final response (only if tools were called)

        Args:
            transcript: The transcribed text to process
            conversation_history: Previous conversation turns for context
            session_id: Optional; when set, conversation and intent_log are persisted
            user_id: Optional; used for conversation.user_id when persisting
            is_twilio_call: Whether this is a Twilio phone call (no GPS available)
            on_before_tool_execution: Optional async callback(tool_calls) invoked before running tools (e.g. play tool-wait audio)

        Returns:
            Dictionary containing:
            - response: Final text response
            - intent: Derived intent (from tool calls) for logging
            - tool_calls: List of tools that were called
            - tool_results: Results from tool execution
        """
        logger.info(f"Processing transcript: {transcript[:100]}...")
        logger.info(f"[LLMService] user_id: {user_id}, is_twilio_call: {is_twilio_call}")

        try:
            result = await self.pipeline.process_text(
                transcript,
                conversation_history=conversation_history,
                session_id=session_id,
                user_id=user_id,
                is_twilio_call=is_twilio_call,
                on_before_tool_execution=on_before_tool_execution,
            )
            
            logger.info(f"LLM processing complete. Response: {result.get('response', '')[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "intent": {},
                "tool_calls": [],
                "tool_results": [],
                "error": str(e)
            }

    async def process_stream(
        self,
        transcript: str,
        conversation_history: list = None,
        user_id: str | None = None,
        is_twilio_call: bool = False,
        on_before_tool_execution: Optional[Callable[[List[Dict[str, Any]]], Awaitable[None]]] = None,
    ) -> dict:
        """
        Process transcript and stream the LLM response (2 AI calls max when tools used).

        Args:
            transcript: The transcribed text to process
            conversation_history: Previous conversation turns for context
            user_id: Optional; used for tool calls requiring user identification
            is_twilio_call: Whether this is a Twilio phone call (no GPS available)
            on_before_tool_execution: Optional async callback(tool_calls) invoked before running tools (e.g. play tool-wait audio)

        Returns:
            Dictionary containing metadata and an async generator under "stream".
        """
        logger.info(f"Processing transcript (streaming): {transcript[:100]}...")
        logger.info(f"[LLMService] user_id received: {user_id}, is_twilio_call: {is_twilio_call}")

        try:
            result = await self.pipeline.process_text_streaming(
                transcript,
                conversation_history,
                user_id=user_id,
                is_twilio_call=is_twilio_call,
                on_before_tool_execution=on_before_tool_execution,
            )
            return result
        except Exception as e:
            logger.error(f"Error in LLM streaming: {e}")
            return {
                "stream": None,
                "response": f"I encountered an error processing your request: {str(e)}",
                "intent": {},
                "tool_calls": [],
                "tool_results": [],
                "error": str(e)
            }
