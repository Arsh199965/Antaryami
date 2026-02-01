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

    # ===== Services =====
    vad = VADService()
    stt = STTService()
    llm = LLMService()
    tts = TTSService()
    analytics = CallAnalyticsService()
    handoff = get_handoff_manager()

    conversation = []
    detected_language = "en"

    audio_q: asyncio.Queue[bytes] = asyncio.Queue()
    utterance_buffer: list[str] = []

    tts_task: Optional[asyncio.Task] = None
    llm_lock = asyncio.Lock()
    running = True

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
                "type": "partial",
                "text": text,
                "language": lang
            })

        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: asyncio.create_task(send())
        )

    def on_final(text: str, lang: str):
        nonlocal detected_language
        detected_language = lang
        utterance_buffer.append(text)

    stt.on_partial_transcript = on_partial
    stt.on_transcript = on_final

    await asyncio.to_thread(stt.connect)

    # ===== LLM + TTS =====
    async def handle_utterance(text: str):
        nonlocal tts_task

        async with llm_lock:
            conversation.append({"role": "user", "text": text})

            # Cancel TTS instantly
            if tts_task and not tts_task.done():
                tts_task.cancel()

            result = await llm.process_stream(
                text,
                conversation,
                user_id=user_id
            )

            stream = result.get("stream")
            response_text = ""

            async def tts_stream():
                started = False
                async for chunk in stream:
                    response_text_nonlocal = chunk
                    await ws.send_json({"type": "response_chunk", "text": chunk})

                    if tts.enabled:
                        if not started:
                            started = True
                            await ws.send_json({"type": "audio_start"})

                        async for audio in tts.stream_tts_chunk(
                            transcript=chunk,
                            context_id=call_id,
                            continue_flag=True,
                            language=normalize_tts_language(detected_language),
                        ):
                            await ws.send_bytes(audio)

                if started:
                    await ws.send_json({"type": "audio_end"})

                conversation.append({
                    "role": "assistant",
                    "text": response_text
                })

            tts_task = asyncio.create_task(tts_stream())

    # ===== Main loop =====
    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg:
                audio = msg["bytes"]
                audio_q.put_nowait(audio)

                # VAD-driven utterance end
                # if vad.is_silence(audio) and utterance_buffer:
                #     full = " ".join(utterance_buffer).strip()
                #     utterance_buffer.clear()
                #     asyncio.create_task(handle_utterance(full))
                if not vad.is_speech(audio) and utterance_buffer:
                    full = " ".join(utterance_buffer).strip()
                    utterance_buffer.clear()
                    asyncio.create_task(handle_utterance(full))



            elif "text" in msg:
                if msg["text"] == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
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

        if tts_task and not tts_task.done():
            tts_task.cancel()

        try:
            await tts_service.close()  # THIS IS CRITICAL
        except Exception:
            pass

        await asyncio.to_thread(stt.disconnect)

