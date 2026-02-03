"""
Sentence-Level LLM to TTS Streaming Pipeline

This module implements an optimized pipeline that:
1. Streams LLM response in real-time
2. Buffers text until complete sentences are detected
3. Immediately sends each sentence to TTS
4. Overlaps TTS generation with LLM generation

This reduces perceived latency by ~500ms+ compared to waiting for full LLM response.
"""
import asyncio
import re
import logging
from typing import AsyncGenerator, Callable, Awaitable, Optional, Literal, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sentence boundary patterns
# Matches: period, exclamation, question mark followed by space or end
# Also handles Hindi sentence endings (।)
SENTENCE_END_PATTERN = re.compile(r'[.!?।]\s*(?=[A-Z\u0900-\u097F]|$)')

# Minimum characters before we consider sending to TTS
MIN_SENTENCE_LENGTH = 10

# Maximum wait time for more text before flushing
FLUSH_TIMEOUT = 0.3  # seconds


@dataclass
class SentenceBuffer:
    """Buffers streaming text and emits complete sentences."""
    
    buffer: str = ""
    sentences_emitted: int = 0
    
    def add_text(self, text: str) -> List[str]:
        """
        Add text to buffer and return any complete sentences.
        
        Args:
            text: New text chunk from LLM
            
        Returns:
            List of complete sentences ready for TTS
        """
        self.buffer += text
        
        sentences = []
        
        # Find all sentence boundaries
        while True:
            match = SENTENCE_END_PATTERN.search(self.buffer)
            if match:
                # Extract the sentence including the punctuation
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                
                # Only emit if sentence is long enough
                if len(sentence) >= MIN_SENTENCE_LENGTH:
                    sentences.append(sentence)
                    self.sentences_emitted += 1
                    self.buffer = self.buffer[end_pos:].lstrip()
                else:
                    # Too short, wait for more text
                    break
            else:
                break
        
        return sentences
    
    def flush(self) -> Optional[str]:
        """
        Flush any remaining text as a final sentence.
        
        Returns:
            Remaining text if any, None otherwise
        """
        if self.buffer.strip() and len(self.buffer.strip()) >= MIN_SENTENCE_LENGTH // 2:
            sentence = self.buffer.strip()
            self.buffer = ""
            self.sentences_emitted += 1
            return sentence
        return None
    
    def reset(self):
        """Reset the buffer."""
        self.buffer = ""
        self.sentences_emitted = 0


@dataclass
class StreamingMetrics:
    """Track streaming performance metrics."""
    
    first_token_time: Optional[float] = None
    first_sentence_time: Optional[float] = None
    first_audio_time: Optional[float] = None
    total_sentences: int = 0
    total_audio_chunks: int = 0
    
    def record_first_token(self, elapsed: float):
        if self.first_token_time is None:
            self.first_token_time = elapsed
            logger.info(f"⚡ First LLM token: {elapsed*1000:.0f}ms")
    
    def record_first_sentence(self, elapsed: float):
        if self.first_sentence_time is None:
            self.first_sentence_time = elapsed
            logger.info(f"⚡ First sentence ready: {elapsed*1000:.0f}ms")
    
    def record_first_audio(self, elapsed: float):
        if self.first_audio_time is None:
            self.first_audio_time = elapsed
            logger.info(f"⚡ First audio chunk: {elapsed*1000:.0f}ms")


async def stream_llm_to_tts(
    llm_stream: AsyncGenerator[str, None],
    tts_generator: Callable[[str, str], AsyncGenerator[bytes, None]],
    language: Literal["hi", "en", "auto"] = "auto",
    on_sentence: Optional[Callable[[str], Awaitable[None]]] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Stream LLM output to TTS with sentence-level batching.
    
    This function:
    1. Receives streaming text from LLM
    2. Buffers until complete sentences are detected
    3. Immediately sends sentences to TTS
    4. Yields audio as soon as it's available
    
    Args:
        llm_stream: Async generator yielding LLM text chunks
        tts_generator: Function that takes (text, language) and returns audio stream
        language: Target language for TTS
        on_sentence: Optional callback when a sentence is ready (for logging/display)
    
    Yields:
        Audio bytes from TTS
    """
    import time
    start_time = time.monotonic()
    metrics = StreamingMetrics()
    
    buffer = SentenceBuffer()
    pending_tts_tasks: List[asyncio.Task] = []
    audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    
    async def process_sentence(sentence: str, lang: str):
        """Process a single sentence through TTS and queue audio."""
        try:
            async for audio_chunk in tts_generator(sentence, lang):
                await audio_queue.put(audio_chunk)
                metrics.total_audio_chunks += 1
                if metrics.first_audio_time is None:
                    metrics.record_first_audio(time.monotonic() - start_time)
        except Exception as e:
            logger.error(f"TTS error for sentence: {e}")
    
    async def llm_consumer():
        """Consume LLM stream and buffer sentences."""
        nonlocal language
        
        try:
            async for chunk in llm_stream:
                if metrics.first_token_time is None:
                    metrics.record_first_token(time.monotonic() - start_time)
                
                # Add chunk to buffer
                sentences = buffer.add_text(chunk)
                
                # Process any complete sentences
                for sentence in sentences:
                    metrics.total_sentences += 1
                    if metrics.first_sentence_time is None:
                        metrics.record_first_sentence(time.monotonic() - start_time)
                    
                    logger.debug(f"Sentence {metrics.total_sentences}: {sentence[:50]}...")
                    
                    if on_sentence:
                        await on_sentence(sentence)
                    
                    # Start TTS for this sentence immediately
                    task = asyncio.create_task(process_sentence(sentence, language))
                    pending_tts_tasks.append(task)
            
            # Flush any remaining text
            remaining = buffer.flush()
            if remaining:
                metrics.total_sentences += 1
                logger.debug(f"Final sentence: {remaining[:50]}...")
                
                if on_sentence:
                    await on_sentence(remaining)
                
                task = asyncio.create_task(process_sentence(remaining, language))
                pending_tts_tasks.append(task)
            
            # Wait for all TTS tasks to complete
            if pending_tts_tasks:
                await asyncio.gather(*pending_tts_tasks, return_exceptions=True)
            
            # Signal end of audio
            await audio_queue.put(None)
            
        except Exception as e:
            logger.error(f"LLM consumer error: {e}")
            await audio_queue.put(None)
    
    # Start LLM consumer in background
    consumer_task = asyncio.create_task(llm_consumer())
    
    try:
        # Yield audio chunks as they become available
        while True:
            audio = await audio_queue.get()
            if audio is None:
                break
            yield audio
    finally:
        # Ensure consumer task is done
        if not consumer_task.done():
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        
        # Log final metrics
        total_time = time.monotonic() - start_time
        logger.info(
            f"📊 Streaming complete: {metrics.total_sentences} sentences, "
            f"{metrics.total_audio_chunks} audio chunks, {total_time*1000:.0f}ms total"
        )


class SentenceLevelPipeline:
    """
    High-level pipeline for sentence-level LLM to TTS streaming.
    
    Usage:
        pipeline = SentenceLevelPipeline(llm_service, tts_service)
        async for audio in pipeline.process(transcript, conversation_history, user_id):
            # Send audio to client
    """
    
    def __init__(self, llm_service, tts_service):
        """
        Initialize the pipeline.
        
        Args:
            llm_service: LLM service with process_stream method
            tts_service: TTS service with stream_tts method
        """
        self.llm_service = llm_service
        self.tts_service = tts_service
    
    async def process(
        self,
        transcript: str,
        conversation_history: Optional[list] = None,
        user_id: Optional[str] = None,
        is_twilio_call: bool = False,
        language: Literal["hi", "en", "auto"] = "auto",
        on_llm_complete: Optional[Callable[[str, list], Awaitable[None]]] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Process transcript through LLM and stream audio sentence-by-sentence.
        
        Args:
            transcript: User's transcript to process
            conversation_history: Previous conversation turns
            user_id: User ID for tool calls
            is_twilio_call: Whether this is a Twilio call
            language: Target language
            on_llm_complete: Callback when full LLM response is available
            
        Yields:
            Audio chunks as bytes
        """
        full_response_parts: List[str] = []
        tool_calls = []
        tool_results = []
        
        # Get streaming response from LLM
        llm_result = await self.llm_service.process_stream(
            transcript,
            conversation_history=conversation_history,
            user_id=user_id,
            is_twilio_call=is_twilio_call,
        )
        
        if llm_result.get("error"):
            logger.error(f"LLM error: {llm_result.get('error')}")
            return
        
        # Store tool information
        tool_calls = llm_result.get("tool_calls", [])
        tool_results = llm_result.get("tool_results", [])
        
        llm_stream = llm_result.get("stream")
        if not llm_stream:
            logger.warning("No LLM stream available")
            return
        
        # Wrap stream to capture full response
        async def wrapped_stream():
            async for chunk in llm_stream:
                full_response_parts.append(chunk)
                yield chunk
        
        # Create TTS generator function
        async def tts_gen(text: str, lang: str):
            async for audio in self.tts_service.stream_tts(text=text, language=lang):
                yield audio
        
        # Stream through the pipeline
        async for audio in stream_llm_to_tts(
            llm_stream=wrapped_stream(),
            tts_generator=tts_gen,
            language=language,
        ):
            yield audio
        
        # Call completion callback with full response
        if on_llm_complete:
            full_response = "".join(full_response_parts)
            await on_llm_complete(full_response, tool_results)
