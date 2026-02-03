"""Fast/optimized system prompts for low-latency LLM responses.

These prompts are shorter and more concise to reduce token count and inference time.
"""

import random
from typing import Optional, List
from langchain_core.tools import StructuredTool


# Filler responses to play while tools are executing
FILLER_RESPONSES = {
    "hi": [
        "एक सेकंड।",
        "बस एक मिनट।",
        "देखता हूं।",
        "चेक करता हूं।",
        "रुकिए।",
    ],
    "en": [
        "One moment.",
        "Let me check.",
        "Just a second.",
        "Checking now.",
        "Hold on.",
    ],
}


def get_filler_response(language: str = "hi") -> str:
    """
    Get a random filler response for the given language.
    
    Args:
        language: Language code ("hi" for Hindi, "en" for English).
        
    Returns:
        A short filler phrase.
    """
    responses = FILLER_RESPONSES.get(language, FILLER_RESPONSES["hi"])
    return random.choice(responses)


def build_fast_system_prompt(
    user_id: Optional[str] = None,
    tools: Optional[List[StructuredTool]] = None,
    is_twilio_call: bool = False,
) -> str:
    """
    Build an optimized, shorter system prompt for faster inference.
    
    This prompt contains essential instructions only, reducing token count
    for faster LLM response times.
    
    Args:
        user_id: The user's ID for tool calls.
        tools: List of available tools.
        is_twilio_call: Whether this is a Twilio phone call.
        
    Returns:
        Concise system prompt string.
    """
    parts = []
    
    # Minimal identity
    parts.append(
        "You are a battery swap service assistant. "
        "Keep responses SHORT (2-3 sentences). "
        "This is VOICE - write like natural speech, no markdown/lists/special chars."
    )
    
    # User context (minimal)
    if user_id:
        parts.append(f"\nUser ID: {user_id}. Use this for all tool calls.")
    
    # Twilio context (minimal)
    if is_twilio_call:
        parts.append(
            "\nPhone call - NO GPS. Ask user location first, then use geocodeAddress."
        )
        if not user_id:
            parts.append("Unknown caller - ask for registered phone/ID for account queries.")
    
    # Tools (compact format)
    if tools:
        parts.append("\n\nTOOLS:")
        for tool in tools:
            # Just name and brief description
            parts.append(f"- {tool.name}: {tool.description[:100]}")
        
        parts.append(
            "\n\nTool tips:"
            "\n- getUserInfo: for profile/account"
            "\n- getNearestStation: handles location internally, just call it"
            "\n- reportBatteryIssue: for complaints (heating, draining, not charging)"
            "\n- getBatteryInfo: only for status queries"
            "\n- geocodeAddress: convert address to coords for phone callers"
        )
    
    # Response rules (minimal)
    parts.append(
        "\n\nRESPONSE RULES:"
        "\n- Match user language (Hindi→Hindi devanagari, English→English)"
        "\n- No **, *, #, bullets, numbered lists"
        "\n- Numbers: small as words (पांच सौ), large as digits"
        "\n- For complaints: confirm registered, then safety advice"
    )
    
    return "\n".join(parts)
