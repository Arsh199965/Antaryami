"""
LLM Service - Language Model Processing
Handles LLM interactions for processing transcripts using the response pipeline.
Simplified: No intent detection - LLM decides which tools to use directly.
"""
import logging

from modules.config import ConfigEnv
from modules.response.response import ResponsePipeline

logger = logging.getLogger(__name__)


class LLMService:
    """Process transcripts with LLM using the response pipeline"""

    def __init__(self):
        """Initialize LLM service with response pipeline"""
        self.api_key = ConfigEnv.CEREBRAS_API_KEY

        self.pipeline = ResponsePipeline(
            api_key=self.api_key,
            model_name=ConfigEnv.CEREBRAS_MODEL_ID,
            temperature=ConfigEnv.CEREBRAS_TEMPERATURE,
        )
        logger.info("✓ LLM Service initialized with Cerebras response pipeline")
    
    async def process(
        self,
        transcript: str,
        conversation_history: list = None,
        user_id: str = None,
        is_twilio_call: bool = False,
    ) -> dict:
        """
        Process transcript through the pipeline.

        Flow:
        1. Single LLM call with all tools available
        2. LLM decides which tools to use (if any)
        3. Tool execution
        4. Final response generation

        Args:
            transcript: The transcribed text to process
            conversation_history: Previous conversation turns for context
            user_id: Optional; used for tool calls requiring user identification
            is_twilio_call: Whether this is a Twilio phone call (no GPS available)

        Returns:
            Dictionary containing:
            - response: Final text response
            - tool_calls: List of tools that were called
            - tool_results: Results from tool execution
        """
        logger.info(f"Processing transcript: {transcript[:100]}...")
        logger.info(f"Context: user_id={user_id}, is_twilio_call={is_twilio_call}")

        try:
            result = await self.pipeline.process_text(
                transcript,
                conversation_history=conversation_history,
                user_id=user_id,
                is_twilio_call=is_twilio_call,
            )
            
            logger.info(f"LLM processing complete. Response: {result.get('response', '')[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
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
    ) -> dict:
        """
        Process transcript and stream the LLM response.

        Args:
            transcript: The transcribed text to process
            conversation_history: Previous conversation turns for context
            user_id: Optional; used for tool calls requiring user identification
            is_twilio_call: Whether this is a Twilio phone call (no GPS available)

        Returns:
            Dictionary containing metadata and an async generator under "stream".
        """
        logger.info(f"Processing transcript (streaming): {transcript[:100]}...")
        logger.info(f"Context: user_id={user_id}, is_twilio_call={is_twilio_call}")

        try:
            result = await self.pipeline.process_text_streaming(
                transcript,
                conversation_history,
                user_id=user_id,
                is_twilio_call=is_twilio_call,
            )
            return result
        except Exception as e:
            logger.error(f"Error in LLM streaming: {e}")
            return {
                "stream": None,
                "response": f"I encountered an error processing your request: {str(e)}",
                "tool_calls": [],
                "tool_results": [],
                "error": str(e)
            }
