"""
STT Service - Soniox Streaming Integration
Handles continuous audio streaming to Soniox WebSocket API
"""
import json
import asyncio
from typing import Callable, Optional
from websockets.sync.client import connect, ClientConnection
import threading
import logging
import time
from websockets.exceptions import ConnectionClosed

from modules.config import ConfigEnv

logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================
SAMPLE_RATE = 16000
SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_API_KEY = ConfigEnv.SONIOX_API_KEY or ""
WEBSOCKET_TIMEOUT = 30  # seconds for individual operations
KEEPALIVE_INTERVAL = 5  # seconds between keepalive checks
CONNECTION_RETRY_DELAY = 1  # initial retry delay in seconds
MAX_RETRY_DELAY = 10  # maximum retry delay in seconds
MAX_CONSECUTIVE_ERRORS = 5  # max errors before reconnect attempt


# =========================
# STT Service
# =========================
class STTService:
    """Manages continuous streaming to Soniox WebSocket API"""
    
    def __init__(
        self,
        on_transcript: Optional[Callable[[str, str], None]] = None,
        on_partial_transcript: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        self.ws: Optional[ClientConnection] = None
        self.on_transcript_callback = on_transcript
        self.on_partial_transcript_callback = on_partial_transcript
        self.on_error_callback = on_error
        self.receive_thread: Optional[threading.Thread] = None
        self.running = False
        self.current_language = "en"  # Track current stable language
        self.language_token_count = {"en": 0, "hi": 0}  # Count tokens per language
        self.last_message_time = time.time()  # Track last message received
        self.consecutive_errors = 0
        self.retry_delay = CONNECTION_RETRY_DELAY
        self.connection_lock = threading.Lock()
        self.is_reconnecting = False
        
    def connect(self):
        """Connect to Soniox WebSocket API with timeout handling"""
        try:
            with self.connection_lock:
                # Connect to Soniox WebSocket with timeout
                try:
                    self.ws = connect(
                        SONIOX_WEBSOCKET_URL,
                        close_timeout=WEBSOCKET_TIMEOUT,
                        max_size=None  # Allow larger messages
                    )
                except TimeoutError as e:
                    logger.error(f"Connection timeout to Soniox: {e}")
                    raise
                
                # Send initial configuration
                config = {
                    "api_key": SONIOX_API_KEY,
                    "model": "stt-rt-v3",
                    "audio_format": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                    "num_channels": 1,
                    "language_hints": ["en", "hi"],  # Support English and Hindi
                    "enable_language_identification": True,  # Identify which language is spoken
                    "enable_endpoint_detection": False,  # We handle this with VAD
                }
                
                try:
                    self.ws.send(json.dumps(config))
                except (TimeoutError, ConnectionClosed) as e:
                    logger.error(f"Failed to send config to Soniox: {e}")
                    self._cleanup_connection()
                    raise
                
                logger.info("✓ Connected to Soniox WebSocket API")
                self.last_message_time = time.time()
                self.consecutive_errors = 0
                self.retry_delay = CONNECTION_RETRY_DELAY
                self.is_reconnecting = False
                
                # Start thread to receive messages
                self.running = True
                self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
                self.receive_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to connect to Soniox: {e}")
            self._cleanup_connection()
            if self.on_error_callback:
                self.on_error_callback(f"Connection failed: {str(e)}")
            raise
    
    def _cleanup_connection(self):
        """Clean up WebSocket connection safely"""
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None
    
    def _receive_messages(self):
        """Receive and process messages from Soniox WebSocket (runs in thread)"""
        try:
            while self.running and self.ws:
                try:
                    # Set a timeout for recv to allow checking connection health
                    try:
                        message = self.ws.recv(timeout=WEBSOCKET_TIMEOUT)
                    except TimeoutError:
                        # Check if this is a genuine timeout or keepalive issue
                        time_since_last = time.time() - self.last_message_time
                        if time_since_last > WEBSOCKET_TIMEOUT * 2:
                            logger.warning(f"WebSocket timeout - no messages for {time_since_last:.1f}s")
                            raise ConnectionError("WebSocket timeout - no keepalive response")
                        continue
                    
                    if not message:
                        continue
                    
                    self.last_message_time = time.time()
                    
                    # Parse JSON response
                    try:
                        response = json.loads(message)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Soniox response: {e}")
                        continue
                    
                    # Handle errors
                    if response.get("error_code"):
                        error_msg = f"{response.get('error_code')}: {response.get('error_message', 'Unknown error')}"
                        logger.error(f"Soniox error: {error_msg}")
                        self.consecutive_errors += 1
                        if self.on_error_callback:
                            self.on_error_callback(error_msg)
                        
                        # Trigger reconnection after too many errors
                        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logger.warning("Too many consecutive errors, triggering reconnection")
                            break
                        continue
                    
                    # Reset error counter on successful message
                    self.consecutive_errors = 0
                    
                    # Process tokens
                    tokens = response.get("tokens", [])
                    if tokens:
                        # Separate final and non-final tokens
                        final_text = ""
                        partial_text = ""
                        
                        # Count language occurrences in this batch
                        batch_languages = {"en": 0, "hi": 0}
                        
                        for token in tokens:
                            text = token.get("text", "")
                            token_lang = token.get("language", self.current_language)
                            
                            # Count this token's language
                            if token_lang in batch_languages:
                                batch_languages[token_lang] += 1
                            
                            if token.get("is_final"):
                                final_text += text
                            else:
                                partial_text += text
                        
                        # Determine dominant language in this batch
                        if batch_languages["en"] > 0 or batch_languages["hi"] > 0:
                            # Update language token counts
                            self.language_token_count["en"] += batch_languages["en"]
                            self.language_token_count["hi"] += batch_languages["hi"]
                            
                            # Only switch language if we have significant evidence
                            # Require at least 3 tokens in the new language
                            total_tokens = self.language_token_count["en"] + self.language_token_count["hi"]
                            if total_tokens >= 3:
                                # Switch if the new language has >60% of recent tokens
                                if self.language_token_count["hi"] > self.language_token_count["en"] * 1.5:
                                    if self.current_language != "hi":
                                        logger.info(f"Language switched: {self.current_language} → hi (tokens: en={self.language_token_count['en']}, hi={self.language_token_count['hi']})")
                                        self.current_language = "hi"
                                        # Reset counters after switch
                                        self.language_token_count = {"en": 0, "hi": 0}
                                elif self.language_token_count["en"] > self.language_token_count["hi"] * 1.5:
                                    if self.current_language != "en":
                                        logger.info(f"Language switched: {self.current_language} → en (tokens: en={self.language_token_count['en']}, hi={self.language_token_count['hi']})")
                                        self.current_language = "en"
                                        # Reset counters after switch
                                        self.language_token_count = {"en": 0, "hi": 0}
                        
                        detected_language = self.current_language
                        
                        # Send partial transcripts (non-final tokens) with stable language
                        if partial_text and self.on_partial_transcript_callback:
                            try:
                                self.on_partial_transcript_callback(partial_text.strip(), detected_language)
                            except Exception as e:
                                logger.error(f"Error in partial transcript callback: {e}")
                        
                        # Send final transcripts with stable language
                        if final_text and self.on_transcript_callback:
                            try:
                                self.on_transcript_callback(final_text.strip(), detected_language)
                            except Exception as e:
                                logger.error(f"Error in transcript callback: {e}")
                    
                    # Check if session finished
                    if response.get("finished"):
                        logger.info("Soniox session finished")
                        break
                        
                except (ConnectionClosed, ConnectionError) as e:
                    if self.running:
                        logger.error(f"WebSocket connection closed: {e}")
                        self.consecutive_errors += 1
                        if self.on_error_callback:
                            self.on_error_callback(f"Connection lost: {str(e)}")
                    break
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Error receiving message: {e}")
                        self.consecutive_errors += 1
                        if self.on_error_callback:
                            self.on_error_callback(f"Receive error: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"Receive thread error: {e}")
            if self.on_error_callback:
                self.on_error_callback(str(e))
        finally:
            self._cleanup_connection()
    
    def stream(self, audio_bytes: bytes):
        """Stream audio chunk to Soniox with error handling"""
        if not self.ws or not self.running:
            return
        
        try:
            # Send raw audio bytes with timeout
            self.ws.send(audio_bytes)
            self.last_message_time = time.time()
        except (ConnectionClosed, TimeoutError, ConnectionError) as e:
            logger.error(f"Error streaming audio: {e}")
            self.consecutive_errors += 1
            if self.on_error_callback:
                self.on_error_callback(f"Stream error: {str(e)}")
            # Connection is likely broken, mark for reconnection
            self._cleanup_connection()
        except Exception as e:
            logger.error(f"Unexpected error streaming audio: {e}")
            if self.on_error_callback:
                self.on_error_callback(f"Stream error: {str(e)}")
    
    def disconnect(self):
        """Disconnect from Soniox cleanly"""
        self.running = False
        
        if self.ws:
            try:
                with self.connection_lock:
                    # Send empty string to signal end of audio
                    try:
                        self.ws.send("")
                    except Exception as e:
                        logger.debug(f"Error sending disconnect signal: {e}")
                    
                    # Close connection gracefully
                    try:
                        self.ws.close(code=1000, reason="Client disconnect")
                    except Exception as e:
                        logger.debug(f"Error sending close frame: {e}")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._cleanup_connection()
        
        # Wait for receive thread to finish
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
            if self.receive_thread.is_alive():
                logger.warning("Receive thread did not terminate gracefully")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.ws is not None and self.running and not self.is_reconnecting
