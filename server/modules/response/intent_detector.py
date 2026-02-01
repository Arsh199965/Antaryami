"""
Fast intent + difficulty detection optimized for Claude Haiku 3 (AWS Bedrock)
"""

import json
import logging
import re
from typing import Dict, Any, Optional

from langchain_aws import ChatBedrockConverse
from modules.config import ConfigEnv

logger = logging.getLogger(__name__)


class IntentDetector:
    INTENT_CATEGORIES = [
        "user_query",
        "service_request",
        "problem_report",
        "location_query",
        "station_query",
        "battery_query",
        "subscription_query",
        "service_center_query",
        "swap_attempt_query",
        "human_handoff",
        "general",
    ]

    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    HEURISTICS = {
        "human_handoff": r"\b(agent|human|customer care|real person|insaan|baat|बात)\b",
        "station_query": r"\b(nearest|nearby|station|swap|स्टेशन|स्वैप)\b",
        "problem_report": r"\b(not working|problem|issue|hot|slow|discharge|nahi|समस्या|issue)\b",
        "battery_query": r"\b(battery status|battery health|battery level|battery|बैटरी)\b",
        "subscription_query": r"\b(subscription|plan|valid|expire|price|सदस्यता|प्लान)\b",
    }

    def __init__(self, region_name: Optional[str] = None):
        self.model_name = ConfigEnv.BEDROCK_MODEL_ID_INTENT
        if not self.model_name:
            raise ValueError("BEDROCK_MODEL_ID not set")

        self.region_name = region_name or ConfigEnv.get_bedrock_region()

        self.model = ChatBedrockConverse(
            model=self.model_name,
            temperature=0,
            max_tokens=160,
            region_name=self.region_name,
        )

    # -------------------------------
    # Difficulty Heuristic
    # -------------------------------
    def _difficulty_heuristic(self, text: str) -> str:
        words = text.split()
        length = len(words)

        if length <= 6:
            return "easy"
        if length <= 15:
            return "medium"
        return "hard"

    # -------------------------------
    # Intent Heuristic
    # -------------------------------
    def _heuristic_detect(self, text: str) -> Optional[Dict[str, Any]]:
        t = text.lower()
        for intent, pattern in self.HEURISTICS.items():
            if re.search(pattern, t):
                return {
                    "intent": intent,
                    "difficulty": self._difficulty_heuristic(text),
                    "confidence": 0.85,
                    "reasoning": "Matched heuristic rule"
                }
        return None

    # -------------------------------
    # LLM Prompt
    # -------------------------------
    def _prompt(self, text: str) -> str:
        valid_intents = ", ".join(self.INTENT_CATEGORIES)
        return f"""Classify the user's intent and sentence difficulty.

Text: "{text}"

IMPORTANT: The intent MUST be one of these exact values only:
{valid_intents}

Difficulty:
- easy: short, direct, single intent
- medium: longer or slightly ambiguous  
- hard: long, vague, or multiple intents

Return ONLY valid JSON (no markdown, no extra text):
{{
  "intent": "one_of_the_valid_intents",
  "difficulty": "easy",
  "confidence": 0.9
}}"""

    # -------------------------------
    # Main API
    # -------------------------------
    async def detect_intent(self, text: str) -> Dict[str, Any]:
        """Detect intent from user text with fallback mechanisms"""
        # Try heuristic first (fast path)
        heuristic = self._heuristic_detect(text)
        if heuristic:
            logger.debug(f"Intent detected via heuristic: {heuristic['intent']}, difficulty: {heuristic.get('difficulty')}")
            return heuristic
        
        # Fall back to LLM with extensive error handling
        try:
            response = await self.model.ainvoke(self._prompt(text))
            raw = response.content

            # Handle list responses from LLM
            if isinstance(raw, list):
                raw = " ".join(
                    c.get("text", str(c)) if isinstance(c, dict) else str(c)
                    for c in raw
                )

            raw = raw.strip()
            
            # Remove markdown code blocks if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            # Parse JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {raw[:100]}")
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        raise ValueError(f"Could not extract valid JSON from: {raw[:100]}")
                else:
                    raise ValueError(f"No JSON found in response: {raw[:100]}")

            # Validate intent is in allowed categories
            intent = data.get("intent", "").lower().strip()
            if intent not in self.INTENT_CATEGORIES:
                logger.warning(f"LLM returned invalid intent '{intent}'. Valid options: {self.INTENT_CATEGORIES}")
                # Try to map common variations to valid intents
                intent = self._map_intent_to_valid_category(intent)
                if not intent:
                    raise ValueError(f"Invalid intent: {data.get('intent')}")

            # Validate difficulty
            difficulty = data.get("difficulty", "medium").lower().strip()
            if difficulty not in self.DIFFICULTY_LEVELS:
                logger.debug(f"Invalid difficulty '{difficulty}', using heuristic")
                difficulty = self._difficulty_heuristic(text)

            # Parse confidence safely
            try:
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
            except (ValueError, TypeError):
                confidence = 0.5

            logger.debug(f"Intent detected via LLM: {intent} (confidence: {confidence})")
            
            return {
                "intent": intent,
                "difficulty": difficulty,
                "confidence": confidence,
                "reasoning": "LLM classification"
            }

        except Exception as e:
            difficulty = self._difficulty_heuristic(text)
            logger.error(f"Intent detection failed: {e}, fallback difficulty: {difficulty}", exc_info=True)
            return {
                "intent": "general",
                "difficulty": difficulty,
                "confidence": 0.0,
                "reasoning": f"Fallback due to error: {str(e)}"
            }

    def _map_intent_to_valid_category(self, detected_intent: str) -> Optional[str]:
        """Map common intent variations to valid categories"""
        detected_intent = detected_intent.lower().strip()
        
        # If already valid, return as-is
        if detected_intent in self.INTENT_CATEGORIES:
            return detected_intent
        
        # Map common variations
        intent_mappings = {
            "request": "service_request",
            "problem": "problem_report",
            "location": "location_query",
            "service": "service_request",
            "help": "service_request",
            "assist": "service_request",
            "question": "user_query",
            "query": "user_query",
            "agent": "human_handoff",
            "handoff": "human_handoff",
            "station": "station_query",
            "battery": "battery_query",
            "subscription": "subscription_query",
        }
        
        # Try exact match in mappings
        if detected_intent in intent_mappings:
            return intent_mappings[detected_intent]
        
        # Try partial match
        for key, value in intent_mappings.items():
            if key in detected_intent or detected_intent in key:
                logger.debug(f"Mapped intent '{detected_intent}' to '{value}'")
                return value
        
        logger.warning(f"Could not map intent '{detected_intent}' to valid category, defaulting to 'general'")
        return "general"
