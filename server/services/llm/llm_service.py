"""
LLM Service - Language Model Processing
Handles LLM interactions for processing transcripts using the response pipeline
"""
import logging

from modules.config import ConfigEnv
from modules.response.response import ResponsePipeline
from modules.response.intent_detector import IntentDetector

logger = logging.getLogger(__name__)

# Intents that typically require tool calls and may cause delay
TOOL_HEAVY_INTENTS = {
    "station_query",      # getNearestStation - DB query + distance calc
    "battery_query",      # getBatteryInfo - DB query
    "user_query",         # getUserInfo - DB query
    "subscription_query", # getSubscriptionInfo - DB query
    "swap_attempt_query", # getLastSwapAttempt - DB query
    "service_center_query", # getLastServiceCenterVisit - DB query
    "problem_report",     # reportBatteryIssue - DB write
    "location_query",     # getCurrentLocation - DB query
    "service_request",    # Multiple tools possible
}


# =========================
# LLM Service
# =========================
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
        self.intent_detector = IntentDetector(api_key=self.api_key)
        logger.info("✓ LLM Service initialized with Cerebras response pipeline")
    
    async def detect_intent(self, transcript: str) -> dict:
        """
        Detect intent from transcript without full LLM processing.
        
        This is useful for early detection to play wait audio before
        the full LLM processing starts.
        
        Args:
            transcript: The transcribed text to analyze
            
        Returns:
            Dictionary containing:
            - intent: The detected intent category
            - confidence: Confidence score
            - reasoning: Explanation
            - requires_tools: Whether this intent typically requires tool calls
        """
        try:
            intent_result = await self.intent_detector.detect_intent(transcript)
            detected_intent = intent_result.get("intent", "general")
            
            # Add flag indicating if this intent typically requires tools
            intent_result["requires_tools"] = detected_intent in TOOL_HEAVY_INTENTS
            
            logger.info(f"Intent detected: {detected_intent} (requires_tools: {intent_result['requires_tools']})")
            return intent_result
            
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return {
                "intent": "general",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "requires_tools": False,
            }
    
    async def process(
        self,
        transcript: str,
        conversation_history: list = None,
        session_id: str = None,
        user_id: str = None,
        is_twilio_call: bool = False,
    ) -> dict:
        """
        Process transcript through the complete pipeline.

        Pipeline flow:
        1. Intent Detection
        2. LLM Processing (with tool definitions and conversation context)
        3. Tool Execution (if needed)
        4. LLM Response Generation

        Args:
            transcript: The transcribed text to process
            conversation_history: Previous conversation turns for context
            session_id: Optional; when set, conversation and intent_log are persisted
            user_id: Optional; used for conversation.user_id when persisting
            is_twilio_call: Whether this is a Twilio phone call (no GPS available)

        Returns:
            Dictionary containing:
            - response: Final text response
            - intent: Detected intent information
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
        pre_detected_intent: dict | None = None,
    ) -> dict:
        """
        Process transcript and stream the LLM response.

        Args:
            transcript: The transcribed text to process
            conversation_history: Previous conversation turns for context
            user_id: Optional; used for tool calls requiring user identification
            is_twilio_call: Whether this is a Twilio phone call (no GPS available)
            pre_detected_intent: Optional; pre-detected intent to skip redundant detection

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
                pre_detected_intent=pre_detected_intent,
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
