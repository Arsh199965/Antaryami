"""
Twilio Voice Router - Bi-directional Media Streams
Handles incoming Twilio voice calls and establishes WebSocket streams
for real-time audio processing with STT, LLM, and TTS pipeline
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
import asyncio
import json
import base64
import audioop
import logging
import numpy as np
import uuid
from datetime import datetime
from typing import Literal, cast, Optional

from modules.config import ConfigEnv
from services.stt import STTService, VADService
from services.llm import LLMService
from services.tts import TTSService
from services.call_analytics import CallAnalyticsService
from db.connection import get_db
from services.user_lookup import lookup_user_by_phone, get_user_id_from_phone
from services.greeting_audio import get_greeting_mulaw_8k_chunks
from services.wait_audio import get_wait_mulaw_8k_chunks
from routers.agent import get_handoff_manager

TTSLanguage = Literal["hi", "en", "auto"]

# =========================
# Router Setup
# =========================
router = APIRouter(prefix="/twilio", tags=["twilio"])
logger = logging.getLogger(__name__)

def normalize_tts_language(lang: str) -> TTSLanguage:
    if lang in ("hi", "en"):
        return cast(TTSLanguage, lang)
    return "auto"

# =========================
# Twilio Configuration
# =========================
TWILIO_ACCOUNT_SID = ConfigEnv.TWILIO_ACCOUNT_SID or ""
TWILIO_AUTH_TOKEN = ConfigEnv.TWILIO_AUTH_TOKEN or ""
TWILIO_WEBSOCKET_URL = ConfigEnv.TWILIO_WEBSOCKET_URL or "wss://your-domain.com/twilio/media"

# =========================
# Constants
# =========================
# Twilio sends mulaw audio at 8kHz
TWILIO_SAMPLE_RATE = 8000
TWILIO_ENCODING = "audio/x-mulaw"

# Convert to 16kHz PCM16 for Soniox
TARGET_SAMPLE_RATE = 16000

# VAD settings
SILENCE_LIMIT_CHUNKS = 15  # ~0.5s of silence
VAD_MIN_SAMPLES = 512
VAD_MIN_BYTES = VAD_MIN_SAMPLES * 2  # PCM16


# =========================
# TwiML Endpoint
# =========================
@router.api_route("/voice", methods=["GET", "POST"])
async def voice_webhook(request: Request):
    """
    Twilio Voice Webhook - Returns TwiML to start media stream
    Called when an incoming call arrives
    Supports both GET and POST methods
    """
    # Get parameters from query string (GET) or form data (POST)
    if request.method == "GET":
        params = request.query_params
    else:
        params = await request.form()
    
    call_sid = params.get("CallSid", "Unknown")
    from_number = params.get("From", "Unknown")
    
    logger.info(f"📞 Incoming call from {from_number}, CallSid: {call_sid}")
    
    # Use WebSocket URL from environment or default
    ws_url = TWILIO_WEBSOCKET_URL
    
    # Generate TwiML to connect bidirectional media stream (greeting plays over WebSocket)
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="callSid" value="{call_sid}"/>
            <Parameter name="from" value="{from_number}"/>
        </Stream>
    </Connect>
</Response>'''
    
    return Response(content=twiml, media_type="application/xml")


# =========================
# Media Stream WebSocket
# =========================
@router.websocket("/media")
async def media_stream(ws: WebSocket):
    """
    Twilio Media Stream WebSocket endpoint for bi-directional audio
    
    Receives from Twilio:
    - connected: Connection established
    - start: Stream metadata (callSid, streamSid, from number, etc.)
    - media: Audio chunks (mulaw, 8kHz, base64 encoded)
    - stop: Stream ended
    
    Sends to Twilio:
    - media: Audio chunks (mulaw, 8kHz, base64 encoded) for TTS playback
    - mark: Markers to track playback completion
    """
    await ws.accept()
    logger.info("✅ Twilio WebSocket connected")
    
    # State tracking
    stream_sid = None
    call_sid = None
    call_id = str(uuid.uuid4())  # Unique identifier for this call
    call_start_time = datetime.utcnow()
    from_number = None  # Caller's phone number
    user_id = None  # Will be looked up from phone number
    is_twilio_call = True  # Flag to indicate this is a phone call (no GPS)
    conversation_history = []  # Store conversation: [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}]
    tts_task = None
    detected_language = "en"  # Track detected language from STT
    processing_lock = asyncio.Lock()  # Prevent concurrent LLM calls
    
    # Handoff state
    handoff_manager = get_handoff_manager()
    handoff_session_id = None  # Track if user is in handoff queue or active call
    is_agent_connected = False  # Track if agent has joined the call
    
    # Utterance accumulation state
    current_utterance_parts = []  # Accumulate transcripts from same utterance
    utterance_timer_task = None  # Timer to finalize utterance after pause
    UTTERANCE_TIMEOUT = 1.0  # Seconds to wait before finalizing utterance
    
    # Get event loop for thread-safe task creation
    loop = asyncio.get_event_loop()
    
    async def check_agent_connected():
        """Check if an agent has connected to this user's call."""
        nonlocal is_agent_connected, handoff_session_id, tts_task
        if handoff_session_id and handoff_session_id in handoff_manager.active_calls:
            if not is_agent_connected:
                # Agent just connected - cancel any ongoing TTS immediately
                is_agent_connected = True
                logger.info(f"🎧 Agent connected to session {handoff_session_id}")
                
                # Cancel TTS task if running
                if tts_task and not tts_task.done():
                    logger.info("🛑 Cancelling TTS - agent has joined")
                    tts_task.cancel()
                    try:
                        await tts_task
                    except asyncio.CancelledError:
                        pass
                    tts_task = None
                
                # Clear Twilio audio buffer
                if stream_sid:
                    await ws.send_json({
                        "event": "clear",
                        "streamSid": stream_sid
                    })
                    logger.info(f"🧹 Cleared Twilio audio buffer for stream {stream_sid}")
            return True
        is_agent_connected = False
        return False

    async def send_greeting_twilio(websocket: WebSocket, sid: str) -> None:
        """Stream greeting WAV as Twilio media (mulaw 8kHz) then send mark."""
        try:
            chunks = get_greeting_mulaw_8k_chunks()
            for chunk in chunks:
                payload = base64.b64encode(chunk).decode("utf-8")
                await websocket.send_json({
                    "event": "media",
                    "streamSid": sid,
                    "media": {"payload": payload},
                })
            await websocket.send_json({
                "event": "mark",
                "streamSid": sid,
                "mark": {"name": "greeting_complete"},
            })
            logger.info("✅ Greeting audio sent to Twilio")
        except Exception as e:
            logger.warning("Failed to send greeting to Twilio: %s", e)
    
    # Initialize services
    vad_service = VADService()
    llm_service = LLMService()
    tts_service = TTSService()
    analytics_service = CallAnalyticsService()
    
    async def save_call_transcript():
        """Save complete call transcript with AI-generated insights to database."""
        nonlocal conversation_history, call_id, call_sid, call_start_time, detected_language
        
        try:
            if not conversation_history:
                logger.warning("⚠️  No conversation to save")
                return
            
            call_end_time = datetime.utcnow()
            duration_seconds = int((call_end_time - call_start_time).total_seconds())
            
            logger.info(f"📊 Analyzing call transcript ({len(conversation_history)} messages)...")
            
            # Generate AI insights
            analysis = await analytics_service.analyze_call(conversation_history)
            
            logger.info(f"✅ Analysis complete:")
            logger.info(f"   Summary: {analysis['summary'][:100]}...")
            logger.info(f"   Satisfaction: {analysis['satisfaction_score']}/5 - {analysis['satisfaction_reasoning']}")
            
            # Prepare call transcript document
            call_data = {
                "call_id": call_id,
                "user_id": None,  # Twilio calls don't have user_id by default
                "start_time": call_start_time,
                "end_time": call_end_time,
                "duration_seconds": duration_seconds,
                "messages": [
                    {
                        "role": msg["role"],
                        "text": msg["text"],
                        "timestamp": call_start_time  # Approximate timestamp
                    }
                    for msg in conversation_history
                ],
                "detected_language": detected_language,
                "summary": analysis["summary"],
                "satisfaction_score": analysis["satisfaction_score"],
                "satisfaction_reasoning": analysis["satisfaction_reasoning"],
                "call_source": "twilio",
                "twilio_call_sid": call_sid
            }
            
            # Save to database
            db = get_db()
            await db.call_transcripts.insert_one(call_data)
            
            logger.info(f"✅ Call transcript saved: {call_id} (Twilio SID: {call_sid})")
        
        except Exception as e:
            logger.error(f"❌ Error saving call transcript: {e}", exc_info=True)
    
    async def finalize_utterance():
        """Finalize accumulated utterance and trigger LLM processing"""
        nonlocal current_utterance_parts, tts_task, is_agent_connected
        
        if not current_utterance_parts:
            return
        
        # Combine all parts into one user message
        full_text = " ".join(current_utterance_parts).strip()
        current_utterance_parts.clear()
        
        if not full_text:
            return
        
        logger.info(f"✅ Utterance finalized: {full_text}")
        
        # Check if agent is connected - if so, just relay transcript, don't process with LLM
        if await check_agent_connected() and handoff_session_id:
            logger.info(f"🎧 Agent connected - skipping LLM, relaying transcript")
            # Send transcript to agent for display
            await handoff_manager.relay_message_to_agent(handoff_session_id, {
                "type": "user_transcript",
                "text": full_text,
                "language": detected_language,
            })
            # Add to conversation history for record keeping
            conversation_history.append({
                "role": "user",
                "text": full_text
            })
            return
        
        # Cancel any ongoing TTS if user spoke
        if tts_task and not tts_task.done():
            logger.info("🛑 Interrupting previous response - New user utterance")
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
        
        # Add to conversation history as user message
        conversation_history.append({
            "role": "user",
            "text": full_text
        })
        
        # Trigger LLM processing
        tts_task = asyncio.create_task(process_and_respond())
    
    # STT service with callbacks
    def on_partial_transcript(text: str, language: str):
        """Handle streaming partial transcripts"""
        nonlocal detected_language, current_utterance_parts
        
        # Log language changes
        if detected_language != language:
            logger.info(f"🌐 Language changed: {detected_language} → {language}")
        
        # Store detected language
        detected_language = language
        logger.info(f"📝 Partial ({language}): {text}")
    
    def on_transcript(text: str, language: str):
        nonlocal detected_language, current_utterance_parts, utterance_timer_task
        
        # Log language changes
        if detected_language != language:
            logger.info(f"🌐 Language changed: {detected_language} → {language}")
        
        # Store detected language
        detected_language = language
        
        # Accumulate transcript part
        logger.info(f"📝 Final Transcript ({language}): {text}")
        current_utterance_parts.append(text)
        
        # Cancel existing timer if any
        if utterance_timer_task and not utterance_timer_task.done():
            utterance_timer_task.cancel()
        
        # Start new timer to finalize utterance after pause
        def schedule_timer():
            nonlocal utterance_timer_task
            
            async def timer():
                await asyncio.sleep(UTTERANCE_TIMEOUT)
                await finalize_utterance()
            
            utterance_timer_task = asyncio.create_task(timer())
        
        loop.call_soon_threadsafe(schedule_timer)
    
    def on_error(error: str):
        logger.error(f"❌ STT Error: {error}")
    
    stt_service = STTService(
        on_transcript=on_transcript,
        on_partial_transcript=on_partial_transcript,
        on_error=on_error
    )
    
    # Connect to Soniox
    await asyncio.to_thread(stt_service.connect)
    logger.info("🎙️ Connected to Soniox")

    async def process_and_respond():
        """Process transcript with LLM and stream TTS response to Twilio"""
        nonlocal tts_task, stream_sid, conversation_history, user_id, is_twilio_call, handoff_session_id
        
        # Use lock to prevent concurrent processing
        async with processing_lock:
            # Get the last user message from history
            if not conversation_history or conversation_history[-1]["role"] != "user":
                logger.warning("⚠️  No user message to process")
                return
            
            full_transcript = conversation_history[-1]["text"]
            logger.info(f"📄 Processing: {full_transcript}")
            logger.info(f"📱 User ID: {user_id}, Is Twilio Call: {is_twilio_call}")
            
            # Step 1: Detect intent FIRST (fast operation)
            logger.info("🔍 Detecting intent...")
            intent_result = await llm_service.detect_intent(full_transcript)
            detected_intent = intent_result.get("intent", "general")
            requires_tools = intent_result.get("requires_tools", False)
            logger.info(f"🎯 Intent: {detected_intent} (requires_tools: {requires_tools})")
            
            # Step 2: Play wait audio if intent requires tools (causes delay)
            if requires_tools and stream_sid:
                logger.info("⏳ Playing wait audio - tool processing expected...")
                try:
                    wait_chunks = get_wait_mulaw_8k_chunks()
                    for payload_bytes in wait_chunks:
                        payload = base64.b64encode(payload_bytes).decode('utf-8')
                        await ws.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload}
                        })
                    logger.info("✅ Wait audio sent to Twilio")
                except Exception as wait_err:
                    logger.warning(f"⚠️ Failed to play wait audio: {wait_err}")
            
            # Step 3: Process with LLM - pass user_id and is_twilio_call context
            llm_result = await llm_service.process(
                full_transcript, 
                conversation_history,
                user_id=user_id,
                is_twilio_call=True,  # This is a Twilio phone call - no GPS available
            )
            response_text = llm_result.get('response', '')
            
            logger.info(f"💬 LLM Response: {response_text}")
            
            # Check if requestHumanAgent tool was called
            tool_results = llm_result.get("tool_results", [])
            for result in tool_results:
                if isinstance(result, dict) and result.get("tool_name") == "requestHumanAgent":
                    tool_output = result.get("result", {})
                    # The tool returns {"status": "ok", "action": "handoff_requested", "data": {...}}
                    if isinstance(tool_output, dict) and tool_output.get("action") == "handoff_requested":
                        # Add user to handoff queue
                        handoff_data = tool_output.get("data", {})
                        reason = handoff_data.get("reason", "User requested human agent")
                        logger.info(f"📞 Handoff requested: {reason}")
                        
                        if user_id and not handoff_session_id:
                            handoff_session_id = await handoff_manager.request_handoff(
                                user_id=user_id,
                                user_ws=ws,
                                reason=reason,
                                conversation_history=conversation_history.copy(),
                                is_twilio=True,
                                stream_sid=stream_sid
                            )
                            logger.info(f"✅ User {user_id} added to handoff queue (session: {handoff_session_id})")
            
            # Add to conversation history
            conversation_history.append({"role": "assistant", "text": response_text})
            
            # Stream TTS Response back to Twilio
            if response_text and response_text.strip() and stream_sid:
                logger.info(f"🔊 Streaming TTS to Twilio...")
                
                try:
                    # Generate TTS audio chunks (PCM float32 44100Hz)
                    tts_language = normalize_tts_language(detected_language)
                    logger.info(f"🔊 TTS using language: {tts_language} (detected: {detected_language})")
                    async for audio_chunk in tts_service.stream_tts(
                        text=response_text,
                        language=tts_language
                    ):
                        current_task = asyncio.current_task()
                        if current_task and current_task.cancelled():
                            logger.warning("⚠️ TTS interrupted")
                            return
                        
                        # Convert audio_chunk to the right format
                        # TTS returns bytes (PCM float32 44100Hz)
                        if isinstance(audio_chunk, bytes):
                            # Already bytes, convert to numpy array
                            audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
                        else:
                            # Already numpy array
                            audio_array = audio_chunk
                        
                        # Convert Float32 to PCM16
                        audio_int16 = (audio_array * 32767).astype(np.int16)
                        pcm_bytes = audio_int16.tobytes()
                        
                        # 2. Resample from 44100Hz to 8000Hz
                        resampled, _ = audioop.ratecv(
                            pcm_bytes, 
                            2,  # 2 bytes per sample (int16)
                            1,  # mono
                            44100,  # from rate
                            8000,  # to rate
                            None
                        )
                        
                        # 3. Convert to mulaw
                        mulaw_audio = audioop.lin2ulaw(resampled, 2)
                        
                        # 4. Encode to base64
                        payload = base64.b64encode(mulaw_audio).decode('utf-8')
                        
                        # 5. Send to Twilio
                        await ws.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": payload
                            }
                        })
                    
                    # Send mark to know when playback is done
                    await ws.send_json({
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {
                            "name": "end_of_response"
                        }
                    })
                    
                    logger.info(f"✅ TTS streaming complete\n")
                
                except asyncio.CancelledError:
                    logger.warning("⚠️ TTS cancelled")
                    raise


    try:
        while True:
            message = await ws.receive_text()
            data = json.loads(message)
            
            event_type = data.get("event")
            
            # Handle connection established
            if event_type == "connected":
                logger.info(f"🔗 Twilio connected: {data}")
            
            # Handle stream start - get metadata
            elif event_type == "start":
                stream_sid = data.get("streamSid")
                start_data = data.get("start", {})
                call_sid = start_data.get("callSid")
                
                # Get caller's phone number from custom parameters
                custom_params = start_data.get("customParameters", {})
                from_number = custom_params.get("from", "Unknown")
                
                logger.info(f"🎬 Stream started - StreamSid: {stream_sid}, CallSid: {call_sid}, From: {from_number}")
                
                # Look up user by phone number
                if from_number and from_number != "Unknown":
                    user_id = await get_user_id_from_phone(from_number)
                    if user_id:
                        logger.info(f"✅ Identified caller as user: {user_id}")
                    else:
                        logger.info(f"⚠️ Unknown caller, phone: {from_number}")

                # Play greeting audio (mulaw 8kHz) without blocking receive loop
                if stream_sid:
                    asyncio.create_task(send_greeting_twilio(ws, stream_sid))
            
            # Handle incoming audio media
            elif event_type == "media":
                media = data.get("media", {})
                payload = media.get("payload")
                
                if payload:
                    # Decode base64 mulaw audio
                    mulaw_audio = base64.b64decode(payload)
                    
                    # Convert mulaw to linear PCM16
                    pcm_audio = audioop.ulaw2lin(mulaw_audio, 2)
                    
                    # Resample from 8kHz to 16kHz for Soniox
                    resampled, _ = audioop.ratecv(
                        pcm_audio,
                        2,  # 2 bytes per sample
                        1,  # mono
                        8000,  # from rate
                        16000,  # to rate
                        None
                    )
                    
                    # Check if agent is connected and relay audio
                    agent_connected = await check_agent_connected()
                    if agent_connected and handoff_session_id:
                        # Relay audio to agent (will be converted to Float32 44100Hz)
                        await handoff_manager.relay_audio_to_agent(handoff_session_id, resampled)
                        # Skip STT when agent is connected for real-time performance
                        logger.debug("🎧 Agent connected - bypassing STT for real-time audio")
                    else:
                        # Only stream to Soniox when no agent is connected
                        try:
                            await asyncio.to_thread(stt_service.stream, resampled)
                        except Exception as e:
                            logger.error(f"⚠️ STT streaming error: {e}")
            
            # Handle stream stop
            elif event_type == "stop":
                logger.info(f"🛑 Stream stopped: {data}")
                break
            
            # Handle mark (playback confirmation)
            elif event_type == "mark":
                mark_name = data.get("mark", {}).get("name")
                logger.info(f"✅ Mark received: {mark_name}")
    
    except WebSocketDisconnect:
        logger.info("✋ Twilio disconnected")
        if tts_task and not tts_task.done():
            tts_task.cancel()
        # Clean up handoff if user was in queue or call
        if handoff_session_id:
            await handoff_manager.cancel_handoff(handoff_session_id)
            await handoff_manager.end_call(handoff_session_id, ended_by="user_disconnect")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
        if tts_task and not tts_task.done():
            tts_task.cancel()
        # Clean up handoff on error
        if handoff_session_id:
            await handoff_manager.cancel_handoff(handoff_session_id)
            await handoff_manager.end_call(handoff_session_id, ended_by="error")
    finally:
        # Save call transcript with analytics
        await save_call_transcript()
        # Cleanup
        try:
            await asyncio.to_thread(stt_service.disconnect)
            logger.info("🔌 Disconnected from Soniox")
        except Exception as e:
            logger.warning(f"⚠️ Error disconnecting: {e}")


# =========================
# Status Callback
# =========================
@router.post("/status")
async def stream_status(request: Request):
    """
    Receives status callbacks from Twilio about stream state
    """
    form_data = await request.form()
    logger.info(f"📊 Stream status: {dict(form_data)}")
    return {"status": "ok"}
