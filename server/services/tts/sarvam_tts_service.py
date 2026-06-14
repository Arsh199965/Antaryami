"""
TTS Service - Sarvam AI Streaming Text-to-Speech
Uses Sarvam's bulbul:v3 model for high-quality Hindi/English TTS.

Optimized for low latency with:
- WAV output format (reliable with SDK)
- Small min_buffer_size for fast first audio
- NumPy-accelerated audio resampling
- Persistent WebSocket connections for chunk streaming
"""
import asyncio
import base64
import logging
from typing import AsyncGenerator, Literal, Optional, Dict, Any

import numpy as np
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

from modules.config import ConfigEnv
from services.tts import utils

logger = logging.getLogger(__name__)


# Sarvam TTS configuration
SARVAM_TTS_MODEL = "bulbul:v3"

# Output format optimizations for low latency:
# - Use WAV format (PCM codec has SDK issues - returns no data)
# - WAV is uncompressed PCM with 44-byte header, fast to process
# - 24000Hz is Sarvam's native rate for v3, avoids server-side resampling
# - Upsample to 44100Hz client-side
OUTPUT_AUDIO_CODEC = "wav"  # WAV works reliably; PCM has SDK issues
SARVAM_SAMPLE_RATE = 24000  # Sarvam v3 native rate
CLIENT_SAMPLE_RATE = 44100  # Client expects 44100Hz

# Buffer settings for low latency
# min_buffer_size: start processing after this many chars (lower = faster first audio)
# max_chunk_length: max sentence split length (smaller = more frequent audio)
MIN_BUFFER_SIZE = 30  # Minimum allowed by API (30-200 range)
MAX_CHUNK_LENGTH = 100  # Smaller chunks for faster streaming

# Available speakers for Sarvam TTS (bulbul:v3)
# v3 has a wider catalog: aditya, ritu, priya, neha, rahul, pooja, rohan, simran,
# kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir,
# aayan, shubh, ashutosh, advait, amelia, sophia
SARVAM_SPEAKERS = {
    "hi": "shubh",    # Default Hindi voice
    "en": "shubh",    # Default English voice (works across all v3 languages)
}


class SarvamTTSService:
    """Service for handling TTS operations with Sarvam AI (bulbul:v3).
    
    Optimized for low latency streaming with:
    - WAV output with Float32 conversion
    - Small buffer sizes for fast first audio
    - Connection reuse for chunk streaming
    """

    def __init__(self):
        """Initialize Sarvam TTS client."""
        self.enabled = bool(ConfigEnv.SARVAM_TTS_ENABLED)
        api_key = ConfigEnv.SARVAM_API_KEY

        if self.enabled and not api_key:
            raise ValueError(
                "Sarvam API key not found. Set SARVAM_API_KEY environment variable."
            )

        self.api_key = api_key
        self.client = AsyncSarvamAI(api_subscription_key=api_key) if self.enabled else None
        self.model = SARVAM_TTS_MODEL
        
        # Track active WebSocket connections for streaming
        self._active_connections: Dict[str, Any] = {}

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable TTS at runtime."""
        self.enabled = enabled

    def ensure_enabled(self) -> bool:
        return self.enabled
    
    def _get_speaker(self, language: Literal["hi", "en"]) -> str:
        """Get the appropriate speaker for the language."""
        return SARVAM_SPEAKERS.get(language, SARVAM_SPEAKERS["hi"])
    
    def _get_language_code(self, language: Literal["hi", "en"]) -> str:
        """Convert short language code to Sarvam format."""
        return "hi-IN" if language == "hi" else "en-IN"
    
    def _pcm_int16_to_float32(self, pcm_data: bytes) -> bytes:
        """
        Convert PCM int16 audio to Float32 format with resampling.
        PCM LINEAR16 is raw int16 samples without any header.
        Resamples from SARVAM_SAMPLE_RATE to CLIENT_SAMPLE_RATE using linear interpolation.
        
        Args:
            pcm_data: Raw PCM int16 audio bytes (little-endian)
            
        Returns:
            Float32 little-endian audio bytes at CLIENT_SAMPLE_RATE
        """
        if len(pcm_data) < 2:
            return b''
        
        # Convert bytes to int16 array using NumPy (much faster than struct)
        int16_samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Convert to float32 and normalize to -1.0 to 1.0 range
        float32_samples = int16_samples.astype(np.float32) / 32768.0
        
        # Resample from SARVAM_SAMPLE_RATE to CLIENT_SAMPLE_RATE
        src_len = len(float32_samples)
        dst_len = int(src_len * CLIENT_SAMPLE_RATE / SARVAM_SAMPLE_RATE)
        x_src = np.arange(src_len)
        x_dst = np.linspace(0, src_len - 1, dst_len)
        resampled = np.interp(x_dst, x_src, float32_samples)
        
        # Return as bytes (already little-endian on most systems)
        return resampled.astype(np.float32).tobytes()
    
    def _wav_to_float32(self, wav_data: bytes, is_first_chunk: bool = True) -> bytes:
        """
        Convert WAV audio to Float32 format with resampling.
        WAV contains a 44-byte header followed by PCM int16 data.
        
        Args:
            wav_data: WAV audio bytes (header + PCM int16)
            is_first_chunk: If True, skip the 44-byte WAV header
            
        Returns:
            Float32 little-endian audio bytes at CLIENT_SAMPLE_RATE
        """
        if len(wav_data) < 4:
            return b''
        
        # Skip WAV header (44 bytes) if this is the first chunk
        pcm_data = wav_data[44:] if is_first_chunk and len(wav_data) > 44 else wav_data
        
        return self._pcm_int16_to_float32(pcm_data)

    async def stream_tts(
        self,
        text: str,
        language: Optional[Literal["hi", "en", "auto"]] = "auto",
        speaker: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio for given text with optimized low-latency settings.
        
        Args:
            text: Text to convert to speech
            language: Language code ("hi", "en", or "auto" for auto-detect)
            speaker: Optional speaker name (uses default if not provided)
            
        Yields:
            Audio chunks as bytes (Float32 PCM format)
        """
        if not text or not text.strip():
            return

        if not self.ensure_enabled():
            return
        
        # Determine language
        if language == "auto":
            detected_lang = utils.detect_language(text)
        else:
            detected_lang = language
        
        # Get speaker
        if not speaker:
            speaker = self._get_speaker(detected_lang)
        
        lang_code = self._get_language_code(detected_lang)
        
        try:
            async with self.client.text_to_speech_streaming.connect(
                model=self.model,
                send_completion_event=True
            ) as ws:
                # Configure with low-latency optimizations:
                # - min_buffer_size: Low value for fast first audio (30 is minimum)
                # - max_chunk_length: Smaller for more frequent audio chunks
                # - WAV output: reliable format
                # - Using Sarvam v3's native 24000Hz, will upsample to 44100Hz client-side
                await ws.configure(
                    target_language_code=lang_code,
                    speaker=speaker,
                    output_audio_codec=OUTPUT_AUDIO_CODEC,
                    min_buffer_size=MIN_BUFFER_SIZE,
                    max_chunk_length=MAX_CHUNK_LENGTH,
                )
                
                # Send the text
                await ws.convert(text)
                
                # Flush to process remaining text
                await ws.flush()
                
                # Stream audio chunks - WAV: first chunk has 44-byte header
                is_first_chunk = True
                async for message in ws:
                    if isinstance(message, AudioOutput):
                        wav_chunk = base64.b64decode(message.data.audio)
                        # Convert WAV to Float32 with 2x upsampling for client
                        float32_chunk = self._wav_to_float32(wav_chunk, is_first_chunk)
                        is_first_chunk = False
                        if float32_chunk:
                            yield float32_chunk
                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            break
                            
        except Exception as e:
            logger.error(f"Error in Sarvam TTS streaming: {e}")
            raise

    async def stream_tts_chunk(
        self,
        transcript: str,
        context_id: str,
        continue_flag: bool,
        language: Optional[Literal["hi", "en", "auto"]] = "auto",
        speaker: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream a single transcript chunk on a context with low-latency optimizations.
        
        Args:
            transcript: Text chunk to convert to speech
            context_id: Unique identifier for the streaming context (used for tracking)
            continue_flag: True if more chunks will follow, False for final chunk
            language: Language code ("hi", "en", or "auto" for auto-detect)
            speaker: Optional speaker name (uses default if not provided)
            
        Yields:
            Audio chunks as bytes (Float32 PCM format)
        """
        if not self.ensure_enabled():
            return
            
        if not transcript or not transcript.strip():
            return
            
        # Determine language
        if language == "auto":
            detected_lang = utils.detect_language(transcript)
        else:
            detected_lang = language
        
        # Get speaker
        if not speaker:
            speaker = self._get_speaker(detected_lang)
        
        lang_code = self._get_language_code(detected_lang)
        
        logger.debug(f"[SarvamTTS] Streaming chunk for context {context_id}: '{transcript[:50]}...'")
        
        try:
            async with self.client.text_to_speech_streaming.connect(
                model=self.model,
                send_completion_event=True
            ) as ws:
                # Configure with low-latency optimizations
                await ws.configure(
                    target_language_code=lang_code,
                    speaker=speaker,
                    output_audio_codec=OUTPUT_AUDIO_CODEC,
                    min_buffer_size=MIN_BUFFER_SIZE,
                    max_chunk_length=MAX_CHUNK_LENGTH,
                )
                
                # Send the text chunk
                await ws.convert(transcript)
                
                # Flush to process the text immediately
                await ws.flush()
                
                # Stream audio chunks - WAV: first chunk has 44-byte header
                chunk_count = 0
                is_first_chunk = True
                async for message in ws:
                    if isinstance(message, AudioOutput):
                        chunk_count += 1
                        wav_chunk = base64.b64decode(message.data.audio)
                        # Convert WAV to Float32 with 2x upsampling for client
                        float32_chunk = self._wav_to_float32(wav_chunk, is_first_chunk)
                        is_first_chunk = False
                        if float32_chunk:
                            yield float32_chunk
                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            break
                
                logger.debug(f"[SarvamTTS] Context {context_id} completed, {chunk_count} audio chunks")
                            
        except Exception as e:
            logger.error(f"Error in Sarvam TTS chunk streaming: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def stream_tts_mixed(
        self,
        text: str,
        speaker: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS for mixed Hindi/English text.
        Sarvam handles code-mixing natively, so we don't need to split.
        
        Args:
            text: Text that may contain both Hindi and English
            speaker: Optional speaker name (uses default if not provided)
            
        Yields:
            Audio chunks as bytes
        """
        if not self.ensure_enabled():
            return
            
        # Sarvam handles mixed Hindi/English well with hi-IN
        # Use auto-detection to pick the dominant language
        async for audio_chunk in self.stream_tts(text, language="auto", speaker=speaker):
            yield audio_chunk

    async def close(self):
        """Close the Sarvam client."""
        # The SDK handles connection cleanup via context managers
        # Clear any tracking data
        self._active_connections.clear()


# Singleton instance
_sarvam_tts_service: Optional[SarvamTTSService] = None


def get_sarvam_tts_service() -> SarvamTTSService:
    """Get or create Sarvam TTS service instance."""
    global _sarvam_tts_service
    if _sarvam_tts_service is None:
        _sarvam_tts_service = SarvamTTSService()
    return _sarvam_tts_service
