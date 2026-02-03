"""TTS Services"""
from .sarvam_tts_service import SarvamTTSService, get_sarvam_tts_service
from .connection_pool import TTSConnectionPool, PooledTTSService, get_tts_pool, get_pooled_tts_service

# Default TTS service is now Sarvam
TTSService = SarvamTTSService
get_tts_service = get_sarvam_tts_service

__all__ = [
    "TTSService", 
    "SarvamTTSService", 
    "get_tts_service", 
    "get_sarvam_tts_service",
    "TTSConnectionPool",
    "PooledTTSService",
    "get_tts_pool",
    "get_pooled_tts_service",
]
