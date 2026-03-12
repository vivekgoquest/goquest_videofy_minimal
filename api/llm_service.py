from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised when dependency is missing at runtime
    genai = None
    genai_types = None

from .schemas import SummarizationResult

logger = logging.getLogger(__name__)

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def _guess_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


class LLMService:
    def __init__(self, api_key: str, model: str, google_api_key: str = ""):
        self._openai_api_key = api_key
        self._openai_model = model
        self._google_api_key = google_api_key
        self._openai_client = OpenAI(api_key=api_key) if api_key else None
        self._google_client = (
            genai.Client(api_key=google_api_key)
            if google_api_key and genai is not None
            else None
        )

    def summarize_into_lines(
        self,
        text: str,
        title: str,
        system_prompt: str,
        model_override: str | None = None,
        provider: str = "openai",
    ) -> list[str]:
        payload = {"title": title, "text": text}
        model_name = model_override or self._default_model_for_provider(provider)
        logger.info(
            "[llm] Requesting script summary with provider '%s' model '%s' (title=%r, text_chars=%d)",
            provider,
            model_name,
            title,
            len(text),
        )
        parsed = self.parse_structured_payload(
            provider=provider,
            model=model_name,
            system_prompt=system_prompt,
            payload=payload,
            response_model=SummarizationResult,
            temperature=0.4,
            max_output_tokens=400,
            missing_key_message=(
                f"{self._provider_env_name(provider)} is required to summarize article text when "
                "script_lines are not provided"
            ),
        )

        lines = [line.strip() for line in parsed.lines if line and line.strip()]
        if not lines:
            raise ValueError(f"{provider.capitalize()} summarization produced no usable script lines")
        logger.info("[llm] Script summary completed (provider=%s, lines=%d)", provider, len(lines))
        return lines

    def parse_structured_payload(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        payload: dict[str, Any] | list[Any] | str,
        response_model: type[StructuredModel],
        temperature: float,
        max_output_tokens: int,
        missing_key_message: str,
    ) -> StructuredModel:
        if provider == "openai":
            return self._parse_openai_payload(
                model=model,
                system_prompt=system_prompt,
                payload=payload,
                response_model=response_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                missing_key_message=missing_key_message,
            )
        if provider == "gemini":
            return self._parse_gemini_payload(
                model=model,
                system_prompt=system_prompt,
                payload=payload,
                response_model=response_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                missing_key_message=missing_key_message,
            )
        raise ValueError(f"Unsupported LLM provider '{provider}'")

    def parse_structured_image(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        payload: dict[str, Any] | str,
        image_path: Path,
        response_model: type[StructuredModel],
        temperature: float,
        max_output_tokens: int,
        missing_key_message: str,
    ) -> StructuredModel:
        if provider == "openai":
            return self._parse_openai_image(
                model=model,
                system_prompt=system_prompt,
                payload=payload,
                image_path=image_path,
                response_model=response_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                missing_key_message=missing_key_message,
            )
        if provider == "gemini":
            return self._parse_gemini_image(
                model=model,
                system_prompt=system_prompt,
                payload=payload,
                image_path=image_path,
                response_model=response_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                missing_key_message=missing_key_message,
            )
        raise ValueError(f"Unsupported LLM provider '{provider}'")

    def _parse_openai_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        payload: dict[str, Any] | list[Any] | str,
        response_model: type[StructuredModel],
        temperature: float,
        max_output_tokens: int,
        missing_key_message: str,
    ) -> StructuredModel:
        if not self._openai_api_key or self._openai_client is None:
            raise ValueError(missing_key_message)

        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        response = self._openai_client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            text_format=response_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI did not return a parseable response")
        return parsed

    def _parse_gemini_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        payload: dict[str, Any] | list[Any] | str,
        response_model: type[StructuredModel],
        temperature: float,
        max_output_tokens: int,
        missing_key_message: str,
    ) -> StructuredModel:
        if not self._google_api_key or self._google_client is None or genai_types is None:
            raise ValueError(missing_key_message)

        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        response = self._google_client.models.generate_content(
            model=model,
            contents=content,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return self._coerce_gemini_parsed(response=response, response_model=response_model)

    def _parse_openai_image(
        self,
        *,
        model: str,
        system_prompt: str,
        payload: dict[str, Any] | str,
        image_path: Path,
        response_model: type[StructuredModel],
        temperature: float,
        max_output_tokens: int,
        missing_key_message: str,
    ) -> StructuredModel:
        if not self._openai_api_key or self._openai_client is None:
            raise ValueError(missing_key_message)

        from .asset_analysis import _to_data_url  # local import to avoid circular import at module load

        text_payload = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        response = self._openai_client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text_payload},
                        {"type": "input_image", "image_url": _to_data_url(image_path)},
                    ],
                },
            ],
            text_format=response_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI did not return a parseable image response")
        return parsed

    def _parse_gemini_image(
        self,
        *,
        model: str,
        system_prompt: str,
        payload: dict[str, Any] | str,
        image_path: Path,
        response_model: type[StructuredModel],
        temperature: float,
        max_output_tokens: int,
        missing_key_message: str,
    ) -> StructuredModel:
        if not self._google_api_key or self._google_client is None or genai_types is None:
            raise ValueError(missing_key_message)

        text_payload = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        response = self._google_client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_text(text=text_payload),
                genai_types.Part.from_bytes(
                    data=image_path.read_bytes(),
                    mime_type=_guess_mime_type(image_path),
                ),
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return self._coerce_gemini_parsed(response=response, response_model=response_model)

    def _coerce_gemini_parsed(
        self,
        *,
        response: Any,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, response_model):
                return parsed
            if isinstance(parsed, dict):
                return response_model.model_validate(parsed)
            return response_model.model_validate(parsed)

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return response_model.model_validate_json(text)
        raise ValueError("Gemini did not return a parseable response")

    def _provider_env_name(self, provider: str) -> str:
        if provider == "openai":
            return "OPENAI_API_KEY"
        if provider == "gemini":
            return "GEMINI_API_KEY or GOOGLE_API_KEY"
        return provider

    def _default_model_for_provider(self, provider: str) -> str:
        if provider == "gemini":
            return "gemini-2.5-flash"
        return self._openai_model
