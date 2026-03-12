from __future__ import annotations

import base64
import logging
import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Protocol

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised when dependency is missing at runtime
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-pro-preview-tts"
DEFAULT_SAMPLE_RATE_HZ = 24000
DEFAULT_SAMPLE_WIDTH_BYTES = 2
DEFAULT_CHANNEL_COUNT = 1


class TTSService(Protocol):
    def synthesize_line(
        self,
        text: str,
        output_mp3: Path,
        voice_id: str | None = None,
        model_id: str = DEFAULT_GEMINI_TTS_MODEL,
        voice_settings: dict[str, Any] | None = None,
    ) -> None: ...

    def get_duration_seconds(self, audio_file: Path) -> float: ...

    def concat_mp3(self, inputs: list[Path], output_file: Path) -> None: ...

    def create_silence_mp3(self, duration_seconds: float, output_file: Path) -> None: ...


class GeminiTTSService:
    def __init__(self, api_key: str, voice_id: str, ffprobe_bin: str, ffmpeg_bin: str):
        self._api_key = api_key.strip()
        self._voice_id = voice_id
        self._ffprobe_bin = ffprobe_bin
        self._ffmpeg_bin = ffmpeg_bin
        self._client = (
            genai.Client(api_key=self._api_key)
            if self._api_key and genai is not None
            else None
        )

    def synthesize_line(
        self,
        text: str,
        output_mp3: Path,
        voice_id: str | None = None,
        model_id: str = DEFAULT_GEMINI_TTS_MODEL,
        voice_settings: dict[str, Any] | None = None,
    ) -> None:
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for manuscript processing")
        if self._client is None or genai_types is None:
            raise RuntimeError(
                "google-genai is required for Gemini TTS. Run `uv sync` to install dependencies."
            )

        voice_name = (voice_id or self._voice_id).strip()
        if not voice_name:
            raise ValueError("A Gemini TTS voice name is required for manuscript processing")

        output_mp3.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Calling Gemini TTS with voice_name=%s model_id=%s",
            voice_name,
            model_id,
        )
        response = self._client.models.generate_content(
            model=model_id or DEFAULT_GEMINI_TTS_MODEL,
            contents=self._build_tts_prompt(text=text, voice_settings=voice_settings),
            config=genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )

        inline_audio = self._first_google_audio(response)
        if inline_audio is None:
            raise ValueError("Gemini TTS did not return audio data")

        sample_rate_hz = self._sample_rate_hz(getattr(inline_audio, "mime_type", None))
        tmp_wav = output_mp3.with_suffix(".wav")
        self._write_pcm_wav(
            wav_path=tmp_wav,
            pcm_bytes=self._blob_bytes(inline_audio),
            sample_rate_hz=sample_rate_hz,
        )
        try:
            self._convert_wav_to_mp3(tmp_wav, output_mp3)
        finally:
            tmp_wav.unlink(missing_ok=True)

    def get_duration_seconds(self, audio_file: Path) -> float:
        cmd = [
            self._ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return max(0.0, float(result.stdout.strip() or 0.0))

    def concat_mp3(self, inputs: list[Path], output_file: Path) -> None:
        if not inputs:
            raise ValueError("Cannot concatenate zero audio files")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        concat_file = output_file.parent / "concat.txt"
        concat_lines = [f"file '{path.resolve()}'" for path in inputs]
        concat_file.write_text("\n".join(concat_lines), encoding="utf-8")

        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_file),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def create_silence_mp3(self, duration_seconds: float, output_file: Path) -> None:
        if duration_seconds <= 0:
            raise ValueError("Silence duration must be positive")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            f"{duration_seconds:.3f}",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_file),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _build_tts_prompt(self, text: str, voice_settings: dict[str, Any] | None) -> str:
        instructions = None
        if voice_settings:
            raw = voice_settings.get("instructions")
            if isinstance(raw, str) and raw.strip():
                instructions = raw.strip()

        if not instructions:
            return text

        return "\n".join(
            [
                instructions,
                "",
                "Read the following script exactly as written:",
                text,
            ]
        )

    def _first_google_audio(self, response: Any) -> Any | None:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None and getattr(inline_data, "data", None):
                    return inline_data
        return None

    def _blob_bytes(self, blob: Any) -> bytes:
        data = getattr(blob, "data", b"")
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return base64.b64decode(data)
        raise ValueError("Unsupported Gemini audio payload")

    def _sample_rate_hz(self, mime_type: str | None) -> int:
        if not mime_type:
            return DEFAULT_SAMPLE_RATE_HZ
        match = re.search(r"rate=(\d+)", mime_type)
        if match:
            return int(match.group(1))
        return DEFAULT_SAMPLE_RATE_HZ

    def _write_pcm_wav(self, wav_path: Path, pcm_bytes: bytes, sample_rate_hz: int) -> None:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(DEFAULT_CHANNEL_COUNT)
            handle.setsampwidth(DEFAULT_SAMPLE_WIDTH_BYTES)
            handle.setframerate(sample_rate_hz)
            handle.writeframes(pcm_bytes)

    def _convert_wav_to_mp3(self, input_wav: Path, output_mp3: Path) -> None:
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-i",
            str(input_wav),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_mp3),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
