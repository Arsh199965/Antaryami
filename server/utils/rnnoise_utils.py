"""
Real-time RNNoise denoiser for 16 kHz PCM16 audio (STT pipeline).
Uses pyrnnoise at 48 kHz with 16k↔48k resampling.
"""
import numpy as np
from pyrnnoise.rnnoise import (
    SAMPLE_RATE as RNNOISE_SAMPLE_RATE,
    FRAME_SIZE as RNNOISE_FRAME_SIZE,
    create,
    destroy,
    process_mono_frame,
)

# 10 ms at 16 kHz = 160 samples = 320 bytes; at 48 kHz = 480 samples
FRAME_SAMPLES_16K = 160
FRAME_BYTES_16K = FRAME_SAMPLES_16K * 2  # PCM16
FRAME_SAMPLES_48K = 480
assert FRAME_SAMPLES_48K == RNNOISE_FRAME_SIZE and RNNOISE_SAMPLE_RATE == 48000


class RealtimeRNNoiseDenoiser:
    """
    Stateful real-time denoiser: accepts PCM16 at 16 kHz, returns denoised PCM16 at 16 kHz.
    One instance per connection (e.g. per call).
    """

    def __init__(self) -> None:
        self._state = create()
        self._buffer = bytearray()

    def __del__(self) -> None:
        if getattr(self, "_state", None) is not None:
            try:
                destroy(self._state)
            except Exception:
                pass

    def _upsample_16k_to_48k(self, samples_16k: np.ndarray) -> np.ndarray:
        """Repeat each sample 3x: 160 -> 480."""
        return np.repeat(samples_16k, 3).astype(np.int16)

    def _downsample_48k_to_16k(self, samples_48k: np.ndarray) -> np.ndarray:
        """Take every 3rd sample: 480 -> 160."""
        return samples_48k[::3].astype(np.int16)

    def _process_one_frame(self, frame_16k_bytes: bytes) -> bytes:
        """Process one 160-sample (320-byte) frame: resample -> RNNoise -> resample -> bytes."""
        samples_16k = np.frombuffer(frame_16k_bytes, dtype=np.int16)
        samples_48k = self._upsample_16k_to_48k(samples_16k)
        denoised_48k, _ = process_mono_frame(self._state, samples_48k)
        denoised_16k = self._downsample_48k_to_16k(denoised_48k)
        return denoised_16k.tobytes()

    def process(self, audio_bytes: bytes) -> bytes:
        """
        Append audio (PCM16 16 kHz), process full 10 ms frames, return denoised bytes.
        May return empty if not enough data buffered.
        """
        self._buffer.extend(audio_bytes)
        out = bytearray()
        while len(self._buffer) >= FRAME_BYTES_16K:
            chunk = bytes(self._buffer[:FRAME_BYTES_16K])
            del self._buffer[:FRAME_BYTES_16K]
            out.extend(self._process_one_frame(chunk))
        return bytes(out)

    def flush(self) -> bytes:
        """
        Pad buffer to full frame(s) with zeros, process, return remaining denoised bytes.
        Call on disconnect to send last buffered audio.
        """
        if not self._buffer:
            return b""
        out = bytearray()
        # Pad to multiple of FRAME_BYTES_16K
        remainder = len(self._buffer) % FRAME_BYTES_16K
        if remainder:
            self._buffer.extend(b"\x00" * (FRAME_BYTES_16K - remainder))
        while len(self._buffer) >= FRAME_BYTES_16K:
            chunk = bytes(self._buffer[:FRAME_BYTES_16K])
            del self._buffer[:FRAME_BYTES_16K]
            out.extend(self._process_one_frame(chunk))
        self._buffer.clear()
        return bytes(out)


if __name__ == "__main__":
    # Quick sanity check
    denoiser = RealtimeRNNoiseDenoiser()
    raw = np.random.randint(-32768, 32767, 480, dtype=np.int16).tobytes()  # 3 frames
    out = denoiser.process(raw)
    assert len(out) == 960, "3 frames in -> 3 frames out"
    out2 = denoiser.flush()
    assert len(out2) == 0, "Buffer empty after process"
    # Test flush with partial frame
    denoiser2 = RealtimeRNNoiseDenoiser()
    denoiser2.process(np.zeros(100, dtype=np.int16).tobytes())
    out3 = denoiser2.flush()
    assert len(out3) == 320, "Flush pads and outputs one frame"
    print("OK: RealtimeRNNoiseDenoiser process + flush")
