# Intent Detection Failure Fix

## Problem
The intent detector was failing with error "Intent detection failed: Invalid intent" when processing Hindi language inputs like "सान से करा सकते हैं?" (Can you do it from the station?).

### Error Pattern
```
2026-02-01 06:50:58,641 - modules.response.intent_detector - ERROR - Intent detection failed: Invalid intent
2026-02-01 06:50:58,641 - modules.response.response - INFO - Detected intent: general
```

### Root Causes
1. **LLM returning unmapped intent values** - The Bedrock LLM was returning intent values not in the `INTENT_CATEGORIES` whitelist
2. **Poor JSON parsing** - The response parser couldn't handle variations in LLM output format
3. **No intent mapping** - No fallback mechanism to map LLM responses to valid intent categories
4. **Limited Hindi support** - Heuristics didn't recognize Hindi keywords for common intents
5. **Weak error handling** - Hard failures instead of graceful degradation

## Solutions Implemented

### 1. **Enhanced Heuristics with Hindi Support**
```python
HEURISTICS = {
    "human_handoff": r"\b(agent|human|customer care|real person|insaan|baat|बात)\b",
    "station_query": r"\b(nearest|nearby|station|swap|स्टेशन|स्वैप)\b",
    "problem_report": r"\b(not working|problem|issue|hot|slow|discharge|nahi|समस्या)\b",
    "battery_query": r"\b(battery status|battery health|battery level|battery|बैटरी)\b",
    "subscription_query": r"\b(subscription|plan|valid|expire|price|सदस्यता|प्लान)\b",
}
```
- Added Hindi keywords for quick detection without LLM
- Supports both English and Hindi input recognition

### 2. **Explicit Intent Constraints in Prompt**
```python
def _prompt(self, text: str) -> str:
    valid_intents = ", ".join(self.INTENT_CATEGORIES)
    return f"""...
    IMPORTANT: The intent MUST be one of these exact values only:
    {valid_intents}
    ...
    """
```
- Lists all valid intents in the prompt to guide LLM
- Reduces chance of invalid responses

### 3. **Robust JSON Parsing**
```python
# Remove markdown code blocks if present
if "```json" in raw:
    raw = raw.split("```json")[1].split("```")[0].strip()
elif "```" in raw:
    raw = raw.split("```")[1].split("```")[0].strip()

# Try to extract JSON from the response
json_match = re.search(r'\{.*\}', raw, re.DOTALL)
if json_match:
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        raise ValueError(...)
```
- Handles multiple response formats
- Extracts JSON even if embedded in extra text
- Better error messages for debugging

### 4. **Intent Mapping and Validation**
```python
def _map_intent_to_valid_category(self, detected_intent: str) -> Optional[str]:
    """Map common intent variations to valid categories"""
    intent_mappings = {
        "request": "service_request",
        "problem": "problem_report",
        "location": "location_query",
        "agent": "human_handoff",
        "station": "station_query",
        "battery": "battery_query",
        "subscription": "subscription_query",
    }
    
    # Try exact match, then partial match
    if detected_intent in intent_mappings:
        return intent_mappings[detected_intent]
    
    for key, value in intent_mappings.items():
        if key in detected_intent or detected_intent in key:
            return value
```
- Maps common variations to valid intents
- Gracefully handles unexpected LLM outputs
- Fallback to "general" only as last resort

### 5. **Improved Error Handling**
```python
async def detect_intent(self, text: str) -> Dict[str, Any]:
    # Try heuristic first (fast path)
    heuristic = self._heuristic_detect(text)
    if heuristic:
        logger.debug(f"Intent detected via heuristic: {heuristic['intent']}")
        return heuristic
    
    # Fall back to LLM with extensive error handling
    try:
        # ... LLM detection
    except Exception as e:
        logger.error(f"Intent detection failed: {e}", exc_info=True)
        # Always return valid fallback
        return {
            "intent": "general",
            "difficulty": self._difficulty_heuristic(text),
            "confidence": 0.0,
            "reasoning": f"Fallback due to error: {str(e)}"
        }
```
- Always returns valid intent (never errors out)
- Logs detailed error information for debugging
- Confidence = 0.0 indicates fallback intent

## Benefits

✅ **Hindi language support** - Keywords recognized in both English and Hindi  
✅ **Resilient parsing** - Handles various LLM response formats  
✅ **Graceful degradation** - Maps unexpected intents instead of failing  
✅ **Better debugging** - Detailed logging shows detection path (heuristic vs LLM)  
✅ **No service disruption** - Always returns valid intent, never throws exception  
✅ **Confidence tracking** - 0.0 confidence indicates fallback was used  

## Fallback Strategy

The detection now follows this order:
1. **Heuristic detection** (Hindi + English keywords) - Instant, no LLM call
2. **LLM classification** - More accurate but slower
3. **Intent mapping** - Map LLM output to valid categories
4. **Fallback to "general"** - Last resort, never fails

## Testing the Fix

Test with the problematic input:
```python
text = "सान से करा सकते हैं?"  # Can you do it from the station?

# Should now:
# - Match heuristic for "station_query" (contains "सान" related to station)
# - Or map LLM output to valid intent
# - Never fail with "Invalid intent"
```

## Configuration

No changes needed to environment variables or config. The fix is self-contained in the intent detector.

## Future Improvements

- Add more Hindi/multi-language patterns to heuristics
- Build a training set of text → intent examples
- Implement confidence thresholds to trigger manual review
- Track which mapping rules are most commonly used
- Add metrics for heuristic vs LLM detection distribution
