"""
STT Service - Sarvam AI Streaming Integration
Handles continuous audio streaming to Sarvam AI WebSocket API
"""
import json
import base64
import asyncio
from typing import Callable, Optional
import websockets
from websockets.sync.client import connect, ClientConnection
import threading
import logging

from modules.config import ConfigEnv

logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================
SAMPLE_RATE = 16000
SARVAM_WEBSOCKET_URL = "wss://api.sarvam.ai/speech-to-text/ws"
SARVAM_API_KEY = ConfigEnv.SARVAM_API_KEY or ""


# =========================
# STT Service
# =========================
class STTService:
    """Manages continuous streaming to Sarvam AI WebSocket API"""
    
    def __init__(
        self,
        on_transcript: Optional[Callable[[str, str], None]] = None,
        on_partial_transcript: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None
    ):
        self.ws: Optional[ClientConnection] = None
        self.on_transcript_callback = on_transcript
        self.on_partial_transcript_callback = on_partial_transcript
        self.on_error_callback = on_error
        self.on_speech_start_callback = on_speech_start
        self.on_speech_end_callback = on_speech_end
        self.receive_thread: Optional[threading.Thread] = None
        self.running = False
        self.current_language = "en"  # Track current stable language
        self.language_code = "en-IN"  # Default language code for Sarvam
        
    def connect(self, language_code: str = "hi-IN"):
        """Connect to Sarvam AI WebSocket API
        
        Args:
            language_code: Language code (e.g., "en-IN", "hi-IN")
                          Note: For multilingual/Hinglish, use "hi-IN" - it handles code-mixing well
        """
        try:
            self.language_code = language_code
            self.current_language = language_code.split("-")[0] if "-" in language_code else language_code
            
            # Build WebSocket URL with query parameters
            # Using saaras:v3 model with codemix mode for best multilingual support
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
            self.ws = connect(
                ws_url,
                additional_headers={
                    "Api-Subscription-Key": SARVAM_API_KEY
                }
            )
            
            logger.info(f"✓ Connected to Sarvam AI WebSocket API (language: {language_code})")
            
            # Start thread to receive messages
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.receive_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam AI: {e}")
            if self.on_error_callback:
                self.on_error_callback(str(e))
            raise
    
    def _receive_messages(self):
        """Receive and process messages from Sarvam AI WebSocket (runs in thread)"""
        try:
            while self.running and self.ws:
                try:
                    message = self.ws.recv()
                    if not message:
                        continue
                    
                    # Parse JSON response
                    response = json.loads(message)
                    
                    # Debug: Log all received messages
                    logger.info(f"Sarvam AI received: {response}")
                    
                    # Handle different message types from Sarvam AI
                    msg_type = response.get("type")
                    
                    if msg_type == "error":
                        # Handle error messages
                        error_data = response.get("data", {})
                        error_msg = f"{error_data.get('code', 'Unknown')}: {error_data.get('message', 'Unknown error')}"
                        logger.error(f"Sarvam AI error: {error_msg}")
                        if self.on_error_callback:
                            self.on_error_callback(error_msg)
                        continue
                    
                    # Handle VAD events from saaras:v3 model (type: "events")
                    if msg_type == "events":
                        event_data = response.get("data", {})
                        signal_type = event_data.get("signal_type", "")
                        
                        if signal_type == "START_SPEECH":
                            logger.info("🎤 Sarvam: Speech detected (events)")
                            if self.on_speech_start_callback:
                                self.on_speech_start_callback()
                        elif signal_type == "END_SPEECH":
                            logger.info("🔇 Sarvam: Speech ended (events)")
                            if self.on_speech_end_callback:
                                self.on_speech_end_callback()
                        continue
                    
                    if msg_type == "speech_start":
                        # Voice activity detected (legacy format)
                        logger.info("🎤 Sarvam: Speech detected")
                        if self.on_speech_start_callback:
                            self.on_speech_start_callback()
                        continue
                    
                    if msg_type == "speech_end":
                        # Voice activity stopped (legacy format)
                        logger.info("🔇 Sarvam: Speech ended")
                        if self.on_speech_end_callback:
                            self.on_speech_end_callback()
                        continue
                    
                    # Handle "data" type response (main transcription response)
                    if msg_type == "data":
                        data = response.get("data", {})
                        text = data.get("transcript", "")
                        is_final = data.get("is_final", True)  # Default to final for data type
                        logger.info(f"📝 Sarvam transcript (final={is_final}): {text}")
                        
                        if text:
                            if is_final:
                                if self.on_transcript_callback:
                                    self.on_transcript_callback(text.strip(), self.current_language)
                            else:
                                if self.on_partial_transcript_callback:
                                    self.on_partial_transcript_callback(text.strip(), self.current_language)
                        continue
                    
                    if msg_type == "transcript":
                        # Transcription result (alternative format)
                        text = response.get("text", "")
                        is_final = response.get("is_final", False)
                        logger.info(f"📝 Sarvam transcript alt (final={is_final}): {text}")
                        
                        if text:
                            if is_final:
                                # Final transcript
                                if self.on_transcript_callback:
                                    self.on_transcript_callback(text.strip(), self.current_language)
                            else:
                                # Partial transcript
                                if self.on_partial_transcript_callback:
                                    self.on_partial_transcript_callback(text.strip(), self.current_language)
                        continue
                    
                    # Handle legacy response format (direct transcript without type)
                    if "transcript" in response:
                        text = response.get("transcript", "")
                        is_final = response.get("is_final", False)
                        logger.info(f"📝 Sarvam legacy transcript (final={is_final}): {text}")
                        
                        if text:
                            if is_final:
                                if self.on_transcript_callback:
                                    self.on_transcript_callback(text.strip(), self.current_language)
                            else:
                                if self.on_partial_transcript_callback:
                                    self.on_partial_transcript_callback(text.strip(), self.current_language)
                        continue
                    
                    # Log any unhandled message types
                    logger.warning(f"Sarvam AI unhandled message: {response}")
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Sarvam AI WebSocket connection closed")
                    break
                except Exception as e:
                    if self.running:
                        logger.error(f"Error receiving message: {e}")
                        if self.on_error_callback:
                            self.on_error_callback(str(e))
                    break
                    
        except Exception as e:
            logger.error(f"Receive thread error: {e}")
            if self.on_error_callback:
                self.on_error_callback(str(e))
    
    def stream(self, audio_bytes: bytes):
        """Stream audio chunk to Sarvam AI
        
        Args:
            audio_bytes: Raw PCM audio bytes (16-bit, mono, 16kHz)
        """
        if not self.ws or not self.running:
            return
        
        try:
            # Sarvam AI expects base64 encoded audio with encoding and sample_rate
            # Format per API docs: {"audio": {"data": "...", "encoding": "...", "sample_rate": ...}}
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            message = {
                "audio": {
                    "data": audio_base64,
                    "encoding": "audio/wav",
                    "sample_rate": SAMPLE_RATE
                }
            }
            
            self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            if self.on_error_callback:
                self.on_error_callback(str(e))
    
    def flush(self):
        """Send flush signal to finalize transcription"""
        if not self.ws or not self.running:
            return
        
        try:
            flush_message = {"flush": True}
            self.ws.send(json.dumps(flush_message))
            logger.debug("Sent flush signal to Sarvam AI")
        except Exception as e:
            logger.error(f"Error sending flush signal: {e}")
    
    def disconnect(self):
        """Disconnect from Sarvam AI"""
        self.running = False
        
        if self.ws:
            try:
                # Send flush signal before closing
                self.flush()
                self.ws.close()
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")
            finally:
                self.ws = None
        
        # Wait for receive thread to finish
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        
        logger.info("Disconnected from Sarvam AI")
    
    def set_language(self, language_code: str):
        """Update the language code
        
        Note: This requires reconnecting to take effect.
        
        Args:
            language_code: Language code (e.g., "en-IN", "hi-IN")
        """
        self.language_code = language_code
        self.current_language = language_code.split("-")[0]
        logger.info(f"Language set to: {language_code}")
