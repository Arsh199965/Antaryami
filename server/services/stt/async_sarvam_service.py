"""
STT Service - Async Sarvam AI Streaming Integration
Handles continuous audio streaming to Sarvam AI WebSocket API using native async.

This replaces the sync websocket + threading approach for better performance:
- No thread pool overhead for each audio chunk
- Native async/await throughout
- Better integration with asyncio event loop
"""
import json
import base64
import asyncio
import logging
from typing import Callable, Optional, Awaitable, Any, Union
import websockets

from modules.config import ConfigEnv

logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================
SAMPLE_RATE = 16000
SARVAM_WEBSOCKET_URL = "wss://api.sarvam.ai/speech-to-text/ws"
SARVAM_API_KEY = ConfigEnv.SARVAM_API_KEY or ""


class AsyncSTTService:
    """
    Async STT Service using native async WebSocket.
    
    Benefits over sync version:
    - No thread context switching overhead
    - Native async callbacks
    - Better backpressure handling
    - Cleaner integration with async frameworks
    """
    
    def __init__(
        self,
        on_transcript: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_partial_transcript: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_error: Optional[Callable[[str], Awaitable[None]]] = None,
        on_speech_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_speech_end: Optional[Callable[[], Awaitable[None]]] = None
    ):
        """
        Initialize the async STT service.
        
        Args:
            on_transcript: Async callback for final transcripts (text, language)
            on_partial_transcript: Async callback for partial transcripts (text, language)
            on_error: Async callback for errors
            on_speech_start: Async callback when speech is detected
            on_speech_end: Async callback when speech ends
        """
        self.ws: Any = None
        self.on_transcript_callback = on_transcript
        self.on_partial_transcript_callback = on_partial_transcript
        self.on_error_callback = on_error
        self.on_speech_start_callback = on_speech_start
        self.on_speech_end_callback = on_speech_end
        
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False
        self._connected = asyncio.Event()
        
        self.current_language = "en"
        self.language_code = "hi-IN"  # Default for multilingual/Hinglish
        
    async def connect(self, language_code: str = "hi-IN"):
        """
        Connect to Sarvam AI WebSocket API asynchronously.
        
        Args:
            language_code: Language code (e.g., "en-IN", "hi-IN")
        """
        try:
            self.language_code = language_code
            self.current_language = language_code.split("-")[0] if "-" in language_code else language_code
            
            # Build WebSocket URL with query parameters
            ws_url = (
                f"{SARVAM_WEBSOCKET_URL}"
                f"?language-code={language_code}"
                f"&model=saaras:v3"
                f"&mode=codemix"
                f"&sample_rate={SAMPLE_RATE}"
                f"&input_audio_codec=pcm_s16le"
                f"&high_vad_sensitivity=true"
                f"&vad_signals=true"
            )
            
            # Connect with API key in headers
            self.ws = await websockets.connect(
                ws_url,
                additional_headers={
                    "Api-Subscription-Key": SARVAM_API_KEY
                },
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            
            self._running = True
            self._connected.set()
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_messages())
            
            logger.info(f"✓ Connected to Sarvam AI WebSocket API (async, language: {language_code})")
            
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam AI: {e}")
            if self.on_error_callback:
                await self.on_error_callback(str(e))
            raise
    
    async def _receive_messages(self):
        """Receive and process messages from Sarvam AI WebSocket."""
        try:
            while self._running and self.ws:
                try:
                    message = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=30.0  # Keepalive timeout
                    )
                    
                    if not message:
                        continue
                    
                    # Parse JSON response
                    response = json.loads(message)
                    
                    # Debug log
                    logger.debug(f"Sarvam AI received: {response}")
                    
                    await self._process_message(response)
                    
                except asyncio.TimeoutError:
                    # No message received, but connection is still alive
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"Sarvam AI WebSocket connection closed: {e}")
                    break
                except Exception as e:
                    if self._running:
                        logger.error(f"Error receiving message: {e}")
                        if self.on_error_callback:
                            await self.on_error_callback(str(e))
                    break
                    
        except Exception as e:
            logger.error(f"Receive task error: {e}")
            if self.on_error_callback:
                await self.on_error_callback(str(e))
        finally:
            self._running = False
            self._connected.clear()
    
    async def _process_message(self, response: dict):
        """Process a single message from Sarvam AI."""
        msg_type = response.get("type")
        
        if msg_type == "error":
            error_data = response.get("data", {})
            error_msg = f"{error_data.get('code', 'Unknown')}: {error_data.get('message', 'Unknown error')}"
            logger.error(f"Sarvam AI error: {error_msg}")
            if self.on_error_callback:
                await self.on_error_callback(error_msg)
            return
        
        # Handle VAD events
        if msg_type == "events":
            event_data = response.get("data", {})
            signal_type = event_data.get("signal_type", "")
            
            if signal_type == "START_SPEECH":
                logger.info("🎤 Sarvam: Speech detected")
                if self.on_speech_start_callback:
                    await self.on_speech_start_callback()
            elif signal_type == "END_SPEECH":
                logger.info("🔇 Sarvam: Speech ended")
                if self.on_speech_end_callback:
                    await self.on_speech_end_callback()
            return
        
        if msg_type == "speech_start":
            logger.info("🎤 Sarvam: Speech detected (legacy)")
            if self.on_speech_start_callback:
                await self.on_speech_start_callback()
            return
        
        if msg_type == "speech_end":
            logger.info("🔇 Sarvam: Speech ended (legacy)")
            if self.on_speech_end_callback:
                await self.on_speech_end_callback()
            return
        
        # Handle transcription data
        if msg_type == "data":
            data = response.get("data", {})
            text = data.get("transcript", "")
            is_final = data.get("is_final", True)
            
            if text:
                logger.info(f"📝 Sarvam transcript (final={is_final}): {text}")
                if is_final:
                    if self.on_transcript_callback:
                        await self.on_transcript_callback(text.strip(), self.current_language)
                else:
                    if self.on_partial_transcript_callback:
                        await self.on_partial_transcript_callback(text.strip(), self.current_language)
            return
        
        if msg_type == "transcript":
            text = response.get("text", "")
            is_final = response.get("is_final", False)
            
            if text:
                logger.info(f"📝 Sarvam transcript alt (final={is_final}): {text}")
                if is_final:
                    if self.on_transcript_callback:
                        await self.on_transcript_callback(text.strip(), self.current_language)
                else:
                    if self.on_partial_transcript_callback:
                        await self.on_partial_transcript_callback(text.strip(), self.current_language)
            return
        
        # Legacy format
        if "transcript" in response:
            text = response.get("transcript", "")
            is_final = response.get("is_final", False)
            
            if text:
                logger.info(f"📝 Sarvam legacy transcript (final={is_final}): {text}")
                if is_final:
                    if self.on_transcript_callback:
                        await self.on_transcript_callback(text.strip(), self.current_language)
                else:
                    if self.on_partial_transcript_callback:
                        await self.on_partial_transcript_callback(text.strip(), self.current_language)
            return
        
        # Log unhandled
        logger.warning(f"Sarvam AI unhandled message: {response}")
    
    async def stream(self, audio_bytes: bytes):
        """
        Stream audio chunk to Sarvam AI asynchronously.
        
        Args:
            audio_bytes: Raw PCM audio bytes (16-bit, mono, 16kHz)
        """
        if not self.ws or not self._running:
            return
        
        try:
            # Sarvam AI expects base64 encoded audio
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            message = {
                "audio": {
                    "data": audio_base64,
                    "encoding": "audio/wav",
                    "sample_rate": SAMPLE_RATE
                }
            }
            
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            if self.on_error_callback:
                await self.on_error_callback(str(e))
    
    async def flush(self):
        """Send flush signal to finalize transcription."""
        if not self.ws or not self._running:
            return
        
        try:
            flush_message = {"flush": True}
            await self.ws.send(json.dumps(flush_message))
            logger.debug("Sent flush signal to Sarvam AI")
        except Exception as e:
            logger.error(f"Error sending flush signal: {e}")
    
    async def disconnect(self):
        """Disconnect from Sarvam AI."""
        self._running = False
        
        if self.ws:
            try:
                await self.flush()
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")
            finally:
                self.ws = None
        
        # Cancel receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        self._connected.clear()
        logger.info("Disconnected from Sarvam AI (async)")
    
    def is_connected(self) -> bool:
        """Check if connected to Sarvam AI."""
        return self._running and self.ws is not None
    
    def set_language(self, language_code: str):
        """
        Update the language code.
        Note: This requires reconnecting to take effect.
        """
        self.language_code = language_code
        self.current_language = language_code.split("-")[0]
        logger.info(f"Language set to: {language_code}")
