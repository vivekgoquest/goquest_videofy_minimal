from pathlib import Path
from types import SimpleNamespace

import api.tts_service as tts_module


class _Factory:
    def __call__(self, **kwargs):
        return kwargs


class FakeModelsAPI:
    def __init__(self):
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                inline_data=SimpleNamespace(
                                    data=b"\x00\x00\x01\x01",
                                    mime_type="audio/L16;rate=24000",
                                )
                            )
                        ]
                    )
                )
            ]
        )


class FakeGeminiClient:
    last_instance = None

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = FakeModelsAPI()
        FakeGeminiClient.last_instance = self


def test_tts_service_calls_gemini_with_voice_name(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tts_module, "genai", SimpleNamespace(Client=FakeGeminiClient))
    monkeypatch.setattr(
        tts_module,
        "genai_types",
        SimpleNamespace(
            GenerateContentConfig=_Factory(),
            SpeechConfig=_Factory(),
            VoiceConfig=_Factory(),
            PrebuiltVoiceConfig=_Factory(),
        ),
    )

    service = tts_module.GeminiTTSService(
        api_key="test-api-key",
        voice_id="fallback-voice",
        ffprobe_bin="ffprobe",
        ffmpeg_bin="ffmpeg",
    )

    def fake_convert(input_wav: Path, output_mp3: Path) -> None:
        assert input_wav.exists()
        output_mp3.write_bytes(b"abc")

    monkeypatch.setattr(service, "_convert_wav_to_mp3", fake_convert)

    out = tmp_path / "line.mp3"
    service.synthesize_line(
        text="Hei verden",
        output_mp3=out,
        voice_id="Kore",
        model_id="gemini-2.5-pro-preview-tts",
        voice_settings={"instructions": "Use a calm narrative tone."},
    )

    client = FakeGeminiClient.last_instance
    assert client is not None
    assert len(client.models.calls) == 1

    payload = client.models.calls[0]
    assert payload["model"] == "gemini-2.5-pro-preview-tts"
    assert "Read the following script exactly as written" in payload["contents"]
    assert "Hei verden" in payload["contents"]
    assert (
        payload["config"]["speech_config"]["voice_config"]["prebuilt_voice_config"]["voice_name"]
        == "Kore"
    )
    assert payload["config"]["response_modalities"] == ["AUDIO"]
    assert out.read_bytes() == b"abc"
