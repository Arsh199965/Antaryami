"""TTS Services"""
from .sarvam_tts_service import SarvamTTSService, get_sarvam_tts_service

# Default TTS service is now Sarvam
TTSService = SarvamTTSService
get_tts_service = get_sarvam_tts_service

__all__ = ["TTSService", "SarvamTTSService", "get_tts_service", "get_sarvam_tts_service"]
