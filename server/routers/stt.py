from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import uuid
import logging
from typing import Literal, Optional, Any
from datetime import datetime
import jwt

from services.stt import STTService, VADService
from services.llm import LLMService
from services.tts import TTSService
from services.call_analytics import CallAnalyticsService
from modules.config import ConfigEnv
from modules.response.tool_registry import get_registry
from db.connection import get_db
from routers.agent import get_handoff_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stt", tags=["stt"])

# VAD: Silero requires >= 512 samples (~32 ms at 16 kHz)
SAMPLE_RATE = 16000
VAD_MIN_SAMPLES = 512
BYTES_PER_SAMPLE = 2  # PCM16
VAD_MIN_BYTES = VAD_MIN_SAMPLES * BYTES_PER_SAMPLE
SILENCE_LIMIT_CHUNKS = 15  # ~0.5s of silence to trigger LLM
UTTERANCE_TIMEOUT = 2.5  # seconds to wait before finalizing utterance (increased for natural pauses)
MIN_UTTERANCE_LENGTH = 3  # minimum words required to send to LLM

TTSLanguage = Literal["hi", "en", "auto"]

def normalize_tts_language(lang: str) -> TTSLanguage:
    return lang if lang in ("hi", "en") else "auto"

def serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj


@router.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()

    call_id = str(uuid.uuid4())
    call_start = datetime.utcnow()
    user_id: Optional[str] = None

    # ===== Auth once, early =====
    token = ws.query_params.get("token")
    if token and ConfigEnv.AUTH_JWT_SECRET:
        try:
            user_id = jwt.decode(
                token,
                ConfigEnv.AUTH_JWT_SECRET,
                algorithms=["HS256"]
            ).get("sub")
        except Exception:
            pass

    if not user_id:
        await ws.send_json({"type": "error", "message": "unauthorized"})
        await ws.close()
        return

    # ===== Fetch user info =====
    async def send_user_info():
        """Fetch user info via tool call on connection"""
        try:
            registry = get_registry()
            result = await registry.execute_tool("getUserInfo", {"userId": user_id})
            await ws.send_json({
                "type": "user_info",
                "user_id": user_id,
                "result": serialize(result),  # Serialize datetime objects
            })
        except Exception as exc:
            logger.warning(f"Failed to fetch user info: {exc}")

    await send_user_info()

    # ===== Services =====
    vad = VADService()
    llm = LLMService()
    tts = TTSService()
    analytics = CallAnalyticsService()
    handoff = get_handoff_manager()

    conversation = []
    detected_language = "en"

    audio_q: asyncio.Queue[bytes] = asyncio.Queue()
    utterance_buffer: list[str] = []
    vad_buffer = bytearray()
    
    # VAD state tracking
    silence_counter = 0
    is_speech_active = False
    utterance_timer_task: Optional[asyncio.Task] = None
    is_processing_utterance = False  # Prevent duplicate processing

    tts_task: Optional[asyncio.Task] = None
    llm_lock = asyncio.Lock()
    running = True
    
    # Get event loop for scheduling tasks from callback thread
    loop = asyncio.get_event_loop()

    # ===== Audio ingest worker (no per-chunk threads) =====
    async def audio_worker():
        while running:
            chunk = await audio_q.get()
            try:
                stt.stream(chunk)
            except Exception as e:
                logger.warning("STT error: %s", e)

    audio_worker_task = asyncio.create_task(audio_worker())

    # ===== STT callbacks =====
    def on_partial(text: str, lang: str):
        nonlocal detected_language
        detected_language = lang

        async def send():
            await ws.send_json({
                "type": "partial_transcript",
                "text": text,
                "language": lang
            })

        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(send())
        )

    def on_final(text: str, lang: str):
        nonlocal detected_language, utterance_timer_task, is_processing_utterance
        
        # Skip if already processing or text is empty
        if is_processing_utterance or not text.strip():
            return
            
        detected_language = lang
        utterance_buffer.append(text)
        
        # Cancel existing timer
        if utterance_timer_task and not utterance_timer_task.done():
            utterance_timer_task.cancel()
        
        # Schedule new timer to finalize utterance after pause
        async def timer():
            await asyncio.sleep(UTTERANCE_TIMEOUT)
            if utterance_buffer and not is_processing_utterance:
                full_text = " ".join(utterance_buffer).strip()
                word_count = len(full_text.split())
                
                # Only process if we have minimum words
                if word_count >= MIN_UTTERANCE_LENGTH:
                    utterance_buffer.clear()
                    await handle_utterance(full_text)
                else:
                    # Clear buffer if too short (user might still be speaking)
                    logger.debug(f"Skipping short utterance: {word_count} words < {MIN_UTTERANCE_LENGTH}")
                    utterance_buffer.clear()
        
        # Use call_soon_threadsafe since callback runs in STT thread
        def schedule_timer():
            nonlocal utterance_timer_task
            utterance_timer_task = asyncio.create_task(timer())
        
        loop.call_soon_threadsafe(schedule_timer)

    # Initialize STT service with callbacks
    stt = STTService(
        on_transcript=on_final,
        on_partial_transcript=on_partial
    )

    await asyncio.to_thread(stt.connect)

    # ===== Call transcript analytics =====
    async def save_call_transcript():
        """Save call with AI-generated insights"""
        if not conversation:
            logger.info("No conversation to save")
            return
        
        try:
            call_end = datetime.utcnow()
            duration = int((call_end - call_start).total_seconds())
            
            logger.info(f"Analyzing call transcript ({len(conversation)} messages)...")
            
            # Generate AI insights
            analysis = await analytics.analyze_call(conversation)
            
            logger.info(f"Analysis complete - Satisfaction: {analysis['satisfaction_score']}/5")
            
            # Save to database
            db = get_db()
            await db.call_transcripts.insert_one({
                "call_id": call_id,
                "user_id": user_id,
                "start_time": call_start,
                "end_time": call_end,
                "duration_seconds": duration,
                "messages": [
                    {
                        "role": msg["role"],
                        "text": msg["text"],
                        "timestamp": call_start  # Approximate timestamp
                    }
                    for msg in conversation
                ],
                "detected_language": detected_language,
                "summary": analysis["summary"],
                "satisfaction_score": analysis["satisfaction_score"],
                "satisfaction_reasoning": analysis["satisfaction_reasoning"],
                "call_source": "web",
                "twilio_call_sid": None
            })
            
            logger.info(f"Call transcript saved: {call_id}")
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")

    # ===== LLM + TTS =====
    async def handle_utterance(text: str):
        nonlocal tts_task, is_processing_utterance
        
        # Prevent duplicate processing
        if is_processing_utterance:
            logger.debug("Skipping duplicate utterance processing")
            return
            
        # Final check for minimum length
        word_count = len(text.split())
        if word_count < MIN_UTTERANCE_LENGTH:
            logger.debug(f"Skipping short utterance: {word_count} words < {MIN_UTTERANCE_LENGTH}")
            return

        is_processing_utterance = True
        
        try:
            async with llm_lock:
                conversation.append({"role": "user", "text": text})
                logger.info(f"Processing utterance: {text[:50]}... ({word_count} words)")

                # Cancel TTS only if it's actively streaming (not just queued)
                if tts_task and not tts_task.done():
                    try:
                        tts_task.cancel()
                        await asyncio.wait_for(tts_task, timeout=0.1)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

                result = await llm.process_stream(
                    text,
                    conversation,
                    user_id=user_id
                )

            stream = result.get("stream")
            response_text = ""

            async def tts_stream():
                nonlocal response_text
                started = False
                
                try:
                    if not stream:
                        logger.warning("No stream in LLM result")
                        return
                    
                    async for chunk in stream:
                        if not running:  # Check if connection is still alive
                            break
                            
                        response_text += chunk
                        try:
                            await ws.send_json({"type": "response_stream", "text": chunk})
                        except (WebSocketDisconnect, Exception) as e:
                            logger.warning(f"Failed to send response stream: {e}")
                            break

                        if tts.enabled:
                            if not started:
                                started = True
                                try:
                                    await ws.send_json({"type": "audio_start"})
                                except (WebSocketDisconnect, Exception) as e:
                                    logger.warning(f"Failed to send audio_start: {e}")
                                    break

                            try:
                                async for audio in tts.stream_tts_chunk(
                                    transcript=chunk,
                                    context_id=call_id,
                                    continue_flag=True,
                                    language=normalize_tts_language(detected_language),
                                ):
                                    if not running:
                                        break
                                    await ws.send_bytes(audio)
                            except (WebSocketDisconnect, Exception) as e:
                                logger.warning(f"TTS streaming error: {e}")
                                # Continue with text even if TTS fails
                                break

                    if started:
                        try:
                            await ws.send_json({"type": "audio_end"})
                        except (WebSocketDisconnect, Exception):
                            pass

                    conversation.append({
                        "role": "assistant",
                        "text": response_text
                    })
                    # Send final llm_response so client can show completed state and history
                    try:
                        await ws.send_json({
                            "type": "llm_response",
                            "transcript": text,
                            "response": response_text,
                            "conversation_history": conversation[-10:]  # Last 10 messages
                        })
                    except (WebSocketDisconnect, Exception) as e:
                        logger.warning(f"Failed to send final response: {e}")
                except Exception as e:
                    logger.error(f"Error in TTS stream: {e}", exc_info=True)
            
            tts_task = asyncio.create_task(tts_stream())
        except Exception as e:
            logger.error(f"Error processing utterance: {e}", exc_info=True)
        finally:
            # Reset processing flag to allow next utterance
            is_processing_utterance = False

    # ===== Main loop =====
    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg:
                audio = msg["bytes"]
                audio_q.put_nowait(audio)

                # VAD: buffer until we have enough samples (Silero requires >= 512)
                # Note: VAD is used for speech detection only, not for utterance finalization
                # Utterance finalization is handled by the timer in on_final callback
                vad_buffer.extend(audio)
                while len(vad_buffer) >= VAD_MIN_BYTES:
                    chunk = bytes(vad_buffer[:VAD_MIN_BYTES])
                    del vad_buffer[:VAD_MIN_BYTES]
                    try:
                        is_speech = vad.is_speech(chunk)
                        
                        if is_speech:
                            is_speech_active = True
                            silence_counter = 0
                        else:
                            if is_speech_active:
                                silence_counter += 1
                                
                                # Reset speech state after sustained silence
                                # (but don't trigger LLM - that's done by on_final timer)
                                if silence_counter >= SILENCE_LIMIT_CHUNKS:
                                    is_speech_active = False
                                    silence_counter = 0
                    except ValueError as e:
                        logger.warning(f"VAD error: {e}")
                        continue

            elif "text" in msg:
                if msg["text"] == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    # finally:
    #     running = False
    #     audio_worker_task.cancel()
    #     if tts_task:
    #         tts_task.cancel()

    #     # Save transcript async, don’t block disconnect
    #     if conversation:
    #         asyncio.create_task(
    #             analytics.analyze_call(conversation)
    #         )

    #     await asyncio.to_thread(stt.disconnect)

    finally:
        running = False
        
        # Cancel audio worker
        if audio_worker_task and not audio_worker_task.done():
            audio_worker_task.cancel()
            try:
                await audio_worker_task
            except asyncio.CancelledError:
                pass
        
        # Cancel TTS task
        if tts_task and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
        
        # Cancel utterance timer
        if utterance_timer_task and not utterance_timer_task.done():
            utterance_timer_task.cancel()
            try:
                await utterance_timer_task
            except asyncio.CancelledError:
                pass
        
        # Close TTS service (critical for Cartesia WebSocket)
        try:
            await tts.close()
        except Exception as e:
            logger.warning(f"TTS cleanup error: {e}")
        
        # Disconnect STT
        try:
            await asyncio.to_thread(stt.disconnect)
        except Exception as e:
            logger.warning(f"STT disconnect error: {e}")
        
        # Save transcript async (don't block disconnect)
        if conversation:
            asyncio.create_task(save_call_transcript())

