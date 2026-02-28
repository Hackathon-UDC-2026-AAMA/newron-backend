import os
import tempfile

from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("STT_MODEL", "base")
MODEL_DEVICE = os.getenv("STT_DEVICE", "cpu").strip().lower()
MODEL_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")

_MODEL = WhisperModel(MODEL_NAME, device=MODEL_DEVICE, compute_type=MODEL_COMPUTE_TYPE)


def transcribe_audio(file_bytes: bytes, suffix: str = ".m4a") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(file_bytes)
        temp_path = temp_audio.name

    try:
        segments, _ = _MODEL.transcribe(temp_path, beam_size=5)
        text = " ".join(segment.text for segment in segments)
        return text.strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
