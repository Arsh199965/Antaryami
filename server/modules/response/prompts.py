"""System prompts for the LLM including tool instructions."""

import re
from typing import Optional, List
from langchain_core.tools import StructuredTool

from .intent_detector import IntentDetector

# Intent categories for router prompt (same as intent_detector for consistency)
ROUTER_INTENT_CATEGORIES = IntentDetector.INTENT_CATEGORIES


def build_system_prompt(
    user_id: Optional[str] = None,
    tools: Optional[List[StructuredTool]] = None,
    is_twilio_call: bool = False,
) -> str:
    """
    Build a system prompt with user context and tool instructions.
    
    Args:
        user_id: The user's ID for tool calls requiring user identification.
        tools: List of available tools to describe to the LLM.
        is_twilio_call: Whether this is a phone call via Twilio (no GPS available).
        
    Returns:
        Complete system prompt string.
    """
    prompt_parts = []
    
    # Base system identity
    prompt_parts.append(
        "You are a helpful customer service assistant for a battery swap service company. "
        "You help users with their account, swap history, service issues, and general inquiries."
    )
    
    # User context
    if user_id:
        prompt_parts.append(
            f"\n\nCURRENT USER CONTEXT:\n"
            f"- User ID: {user_id}\n"
            f"- Use this user_id when calling any tool that requires a userId parameter."
        )
    
    # Twilio-specific context
    if is_twilio_call:
        prompt_parts.append(
            "\n\nIMPORTANT - PHONE CALL CONTEXT:"
            "\n- This user is calling via phone (Twilio). GPS/browser location is NOT available."
            "\n- DO NOT use getCurrentLocation tool - it will not work for phone calls."
            "\n- If you need the user's location (e.g., to find nearest station):"
            "\n  1. ASK the user: 'Aap abhi kahan hain?' or 'Can you tell me your current location or nearest landmark?'"
            "\n  2. Once they provide a location/address, use geocodeAddress tool to convert it to coordinates"
            "\n  3. Then use getNearestStation with the latitude and longitude parameters"
            "\n- Be patient and conversational - phone users may take time to respond."
        )
        
        if not user_id:
            prompt_parts.append(
                "\n- UNKNOWN CALLER: This phone number is not registered in our system."
                "\n- You can still help with general queries, station information (if they provide location), etc."
                "\n- For account-specific queries, politely ask them to provide their registered phone number or user ID."
            )
    
    # Tool instructions
    if tools:
        prompt_parts.append("\n\nAVAILABLE TOOLS:")
        prompt_parts.append(
            "You have access to the following tools. Use them to retrieve information when needed:"
        )
        
        for tool in tools:
            prompt_parts.append(f"\n- {tool.name}: {tool.description}")
        
        prompt_parts.append(
            "\n\nTOOL USAGE GUIDELINES:"
            "\n1. When the user asks about their name, profile, account, or personal information, use the getUserInfo tool."
            "\n2. When the user asks about their location or where they are, use the getCurrentLocation tool."
            "\n3. When the user asks about a recent swap or swap history, use the getLastSwapAttempt tool."
            "\n4. When the user asks about service center visits, use the getLastServiceCenterVisit tool."
            "\n5. When the user reports a general problem, use the getProblemContext tool to analyze it."
            "\n6. IMPORTANT - For nearest station queries:"
            "\n   - Use getNearestStation DIRECTLY - it fetches user location internally"
            "\n   - Do NOT call getCurrentLocation first, then getNearestStation"
            "\n   - Just call getNearestStation(userId='...') and it will handle location"
            "\n7. If the user specifically asks for stations with available batteries, call getNearestStation with requireAvailableBatteries=true."
            "\n8. IMPORTANT - Battery tools distinction:"
            "\n   - getBatteryInfo: Use ONLY when user ASKS/QUERIES about battery (e.g., 'what is my battery health?', 'battery status?')"
            "\n   - reportBatteryIssue: Use when user COMPLAINS/REPORTS a problem (e.g., 'battery garam ho rahi hai', 'battery heating', 'battery not charging', 'drains fast')"
            "\n   - If user says their battery IS having a problem, use reportBatteryIssue, NOT getBatteryInfo."
            "\n9. Always use the provided user_id when calling tools that require userId."
            "\n10. For reportBatteryIssue, pass the user's exact complaint as issueDescription."
            "\n11. If a tool returns an error or 'not found', inform the user politely."
            "\n12. For critical battery issues (overheating, swelling, leakage), emphasize safety and urgency in your response."
            "\n13. When you need to know if a similar situation has happened before, what worked or failed, or what company policy says, use getCallInsights with a short situation_summary (and optionally issue_type like penalty_dispute or battery_swap). Use the returned similar scenarios, response patterns, and policy snippets to inform your response."
        )
        
        # Geocoding instructions for Twilio calls
        if is_twilio_call:
            prompt_parts.append(
                "\n\nLOCATION HANDLING FOR PHONE CALLS:"
                "\n13. For phone callers, NEVER assume you have their location. Always ask first."
                "\n14. When user provides a location verbally (e.g., 'Andheri Station', 'Mumbai Central'):"
                "\n    a. Use geocodeAddress tool with the address they provided"
                "\n    b. Use the returned latitude/longitude with getNearestStation"
                "\n    c. Example: geocodeAddress(address='Andheri Station', city='Mumbai') -> getNearestStation(userId='...', latitude=..., longitude=...)"
                "\n15. If geocoding fails, ask for a more specific location or nearby landmark."
            )
    
    # Response guidelines
    prompt_parts.append(
        "\n\nRESPONSE GUIDELINES:"
        "\n- Be concise and helpful."
        "\n- If you need to use a tool to answer, use it before responding."
        "\n- After receiving tool results, incorporate the information naturally into your response."
        "\n- Don't mention that you're using tools; just provide the information."
        "\n- IMPORTANT: When a complaint/issue is reported using reportBatteryIssue tool:"
        "\n  1. FIRST confirm that the complaint has been registered/recorded"
        "\n  2. THEN provide any safety advice or next steps"
        "\n  3. Example: 'Your complaint has been registered. [Then safety advice if applicable]'"
    )
    
    return "\n".join(prompt_parts)


# Default max tokens for router step (intent + tool decision only)
ROUTER_MAX_TOKENS_GUIDELINE = 256


def build_router_system_prompt(
    user_id: Optional[str] = None,
    tools: Optional[List[StructuredTool]] = None,
    is_twilio_call: bool = False,
    max_tokens_guideline: int = ROUTER_MAX_TOKENS_GUIDELINE,
) -> str:
    """
    Build the system prompt for the first step (router): classify query type (intent)
    and decide which tools to call. Used with a fast model; keep response short.

    Args:
        user_id: The user's ID for tool calls requiring user identification.
        tools: List of available tools (same as general intent).
        is_twilio_call: Whether this is a phone call via Twilio (no GPS available).
        max_tokens_guideline: Suggested max tokens for this step.

    Returns:
        Router system prompt string.
    """
    categories_str = ", ".join(ROUTER_INTENT_CATEGORIES)
    prompt_parts = [
        "You are the routing step for a battery swap customer service assistant.",
        "Your job is to (1) classify the user's query type (intent) and (2) call any tools needed to answer it.",
        f"Use at most {max_tokens_guideline} tokens for any text in this step; only classify intent and issue tool calls. Do not generate the full answer here.",
        "",
        "INTENT OUTPUT (required):",
        "You must classify the query type. Start your text response with exactly: Intent: <category>",
        f"where category is one of: {categories_str}",
        "Examples: Intent: station_query , Intent: battery_query , Intent: general",
        "",
        "MULTIPLE TOOLS:",
        "You may call multiple tools in one turn when the user query requires it (e.g. getUserInfo and getNearestStation). Call all needed tools in this single response.",
    ]
    if user_id:
        prompt_parts.append(
            f"\nCURRENT USER CONTEXT:\n- User ID: {user_id}\n- Use this user_id when calling any tool that requires a userId parameter."
        )
    if is_twilio_call:
        prompt_parts.append(
            "\nPHONE CALL CONTEXT: This user is on a phone call (Twilio). GPS is NOT available. Do NOT use getCurrentLocation. For location, use geocodeAddress after the user provides an address/landmark."
        )
    if tools:
        prompt_parts.append("\nAVAILABLE TOOLS:")
        prompt_parts.append("You have access to the following tools. Call the ones needed for the user's query:")
        for tool in tools:
            prompt_parts.append(f"- {tool.name}: {tool.description}")
        prompt_parts.append(
            "\nCall the appropriate tool(s) now. You may call multiple tools in one response when needed."
        )
    return "\n".join(prompt_parts)


# Regex to parse "Intent: <category>" from router response (case-insensitive, flexible whitespace)
_INTENT_LINE_RE = re.compile(r"^\s*Intent\s*:\s*(\w+)\s*", re.IGNORECASE | re.MULTILINE)


def parse_intent_from_router_response(
    text: str,
    fallback_intent: str = "general",
) -> str:
    """
    Parse intent from the first step (router) LLM text response.
    Expects a line like "Intent: station_query" at the start.

    Args:
        text: The text content from the router LLM response (may be empty if only tool_calls returned).
        fallback_intent: Intent to return if no valid "Intent: <category>" line is found.

    Returns:
        Intent category string (e.g. station_query, general).
    """
    if not text or not text.strip():
        return fallback_intent
    match = _INTENT_LINE_RE.search(text.strip())
    if not match:
        return fallback_intent
    raw = match.group(1).strip().lower()
    for c in ROUTER_INTENT_CATEGORIES:
        if c.lower() == raw:
            return c
    return fallback_intent


def build_tool_result_prompt(tool_name: str, result: dict) -> str:
    """
    Build a prompt explaining a tool result to help the LLM form a response.
    
    Args:
        tool_name: Name of the tool that was executed.
        result: The result dictionary from the tool.
        
    Returns:
        Prompt string describing the tool result.
    """
    status = result.get("status", "unknown")
    data = result.get("data", {})
    
    if status == "success":
        return f"Tool {tool_name} returned successfully with data: {data}"
    if status == "ok":
        return f"Tool {tool_name} returned successfully with data: {data}"
    elif status == "not_found":
        return f"Tool {tool_name} could not find the requested information: {data.get('message', 'Not found')}"
    elif status == "not_implemented":
        return f"Tool {tool_name} is not yet available: {data.get('message', 'Not implemented')}"
    else:
        return f"Tool {tool_name} encountered an error: {data.get('error', 'Unknown error')}"
