from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import GenerationManifest


class ConfigResolverError(Exception):
    pass


@dataclass
class ResolvedOpenAIImageGenerationConfig:
    model: str
    size: str
    quality: str
    background: str


@dataclass
class ResolvedNanobananaImageGenerationConfig:
    model: str
    aspect_ratio: str
    thinking_budget: str | None


@dataclass
class ResolvedImageGenerationConfig:
    enabled: bool
    provider: str
    prompt_builder_model: str
    variants: int
    prefer_generated: bool
    brief_prompt: str
    openai_prompt_builder: str
    nanobanana_prompt_builder: str
    openai: ResolvedOpenAIImageGenerationConfig
    nanobanana: ResolvedNanobananaImageGenerationConfig


@dataclass
class ResolvedLLMNodeConfig:
    provider: str
    model: str


@dataclass
class ResolvedLLMConfig:
    default_provider: str
    script_generation: ResolvedLLMNodeConfig
    image_description: ResolvedLLMNodeConfig
    asset_placement: ResolvedLLMNodeConfig
    image_prompt_builder: ResolvedLLMNodeConfig


@dataclass
class ResolvedConfig:
    manuscript_model: str
    media_model: str
    script_prompt: str
    placement_prompt: str
    describe_images_prompt: str
    llm: ResolvedLLMConfig
    image_generation: ResolvedImageGenerationConfig
    tts_provider: str
    voice_id: str
    tts_model_id: str
    voice_settings: dict[str, Any]
    segment_pause_seconds: float
    player: dict[str, Any]
    export_defaults: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigResolverError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigResolverError(f"Invalid config JSON object in {path}")
    return data


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_text_llm_provider(value: Any) -> str:
    normalized = str(value or "openai").strip().lower()
    if normalized in {"openai", "gemini"}:
        return normalized
    return "openai"


def _pick_object(source: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _pick_value(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return default


class ConfigResolver:
    def __init__(self, config_root: Path):
        self.config_root = config_root

    def _resolve_brand_path(self, brand_id: str) -> Path:
        direct = self.config_root / f"{brand_id}.json"
        if direct.exists():
            return direct

        fallback_paths = sorted(path for path in self.config_root.glob("*.json") if path.is_file())
        if not fallback_paths:
            raise ConfigResolverError(f"Missing config file: {direct}")

        preferred = self.config_root / "default.json"
        if preferred.exists():
            return preferred
        return fallback_paths[0]

    def resolve(
        self,
        manifest: GenerationManifest,
        llm_override: dict[str, Any] | None = None,
        image_generation_override: dict[str, Any] | None = None,
    ) -> ResolvedConfig:
        brand = _read_json(self._resolve_brand_path(manifest.brandId))
        options = brand.get("options", {}) if isinstance(brand.get("options"), dict) else {}
        player = brand.get("player", {}) if isinstance(brand.get("player"), dict) else {}
        prompts = brand.get("prompts", {}) if isinstance(brand.get("prompts"), dict) else {}
        openai_cfg = brand.get("openai", {}) if isinstance(brand.get("openai"), dict) else {}
        gemini_cfg = brand.get("gemini", {}) if isinstance(brand.get("gemini"), dict) else {}
        llm_cfg_raw = brand.get("llm", {}) if isinstance(brand.get("llm"), dict) else {}
        llm_cfg = _merge_dict(llm_cfg_raw, llm_override) if llm_override else llm_cfg_raw
        audio_cfg = brand.get("audio", {}) if isinstance(brand.get("audio"), dict) else {}
        image_generation_cfg_raw = (
            brand.get("imageGeneration", {})
            if isinstance(brand.get("imageGeneration"), dict)
            else {}
        )
        image_generation_cfg = (
            _merge_dict(image_generation_cfg_raw, image_generation_override)
            if image_generation_override
            else image_generation_cfg_raw
        )
        people_cfg = brand.get("people", {}) if isinstance(brand.get("people"), dict) else {}
        default_person = (
            people_cfg.get("default", {})
            if isinstance(people_cfg.get("default"), dict)
            else {}
        )
        export_defaults = (
            brand.get("exportDefaults", {}) if isinstance(brand.get("exportDefaults"), dict) else {}
        )

        pause = manifest.options.segmentPauseSeconds
        if pause is None:
            pause = float(options.get("segmentPauseSeconds", 0.4))

        tts_provider = str(audio_cfg.get("tts", "google")).strip().lower() or "google"
        if tts_provider != "google":
            raise ConfigResolverError(
                "Only audio.tts='google' is supported. ElevenLabs has been removed from this repo."
            )

        voice_id = default_person.get("voice")
        if not isinstance(voice_id, str) or not voice_id:
            raise ConfigResolverError(
                f"Brand '{manifest.brandId}' must define people.default.voice"
            )

        voice_settings: dict[str, Any] = {}
        for key in ("stability", "similarity_boost", "style"):
            value = default_person.get(key)
            if isinstance(value, (int, float)):
                voice_settings[key] = value
        use_speaker_boost = default_person.get("use_speaker_boost")
        if isinstance(use_speaker_boost, bool):
            voice_settings["use_speaker_boost"] = use_speaker_boost

        instructions = default_person.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            voice_settings["instructions"] = instructions.strip()

        tts_model_id = default_person.get("model_id")
        if not isinstance(tts_model_id, str) or not tts_model_id:
            tts_model_id = default_person.get("modelId")
        if not isinstance(tts_model_id, str) or not tts_model_id:
            tts_model_id = "gemini-2.5-pro-preview-tts"

        openai_manuscript_model = str(openai_cfg.get("manuscriptModel", "gpt-4o-mini"))
        openai_media_model = str(openai_cfg.get("mediaModel", openai_manuscript_model))
        gemini_manuscript_model = str(gemini_cfg.get("manuscriptModel", "gemini-2.5-flash"))
        gemini_media_model = str(gemini_cfg.get("mediaModel", gemini_manuscript_model))
        legacy_prompt_builder_model = str(
            _pick_value(
                image_generation_cfg,
                "prompt_builder_model",
                "promptBuilderModel",
                default="",
            )
        )
        gemini_prompt_builder_model = str(
            gemini_cfg.get("promptBuilderModel", legacy_prompt_builder_model or "gemini-2.5-flash")
        )
        openai_prompt_builder_model = legacy_prompt_builder_model or "gpt-5-mini"

        llm_nodes_cfg = llm_cfg.get("nodes", {}) if isinstance(llm_cfg.get("nodes"), dict) else {}
        default_llm_provider = _normalize_text_llm_provider(
            llm_cfg.get("defaultProvider", llm_cfg.get("default_provider", "openai"))
        )

        def resolve_text_node(
            camel_key: str,
            snake_key: str,
            *,
            openai_default_model: str,
            gemini_default_model: str,
        ) -> ResolvedLLMNodeConfig:
            node_cfg = _pick_object(llm_nodes_cfg, camel_key, snake_key)
            provider = _normalize_text_llm_provider(node_cfg.get("provider", default_llm_provider))
            raw_model = node_cfg.get("model")
            if isinstance(raw_model, str) and raw_model.strip():
                model = raw_model.strip()
            else:
                model = openai_default_model if provider == "openai" else gemini_default_model
            return ResolvedLLMNodeConfig(provider=provider, model=model)

        script_generation_llm = resolve_text_node(
            "scriptGeneration",
            "script_generation",
            openai_default_model=openai_manuscript_model,
            gemini_default_model=gemini_manuscript_model,
        )
        image_description_llm = resolve_text_node(
            "imageDescription",
            "image_description",
            openai_default_model=openai_media_model,
            gemini_default_model=gemini_media_model,
        )
        asset_placement_llm = resolve_text_node(
            "assetPlacement",
            "asset_placement",
            openai_default_model=openai_media_model,
            gemini_default_model=gemini_media_model,
        )
        image_prompt_builder_llm = resolve_text_node(
            "imagePromptBuilder",
            "image_prompt_builder",
            openai_default_model=openai_prompt_builder_model,
            gemini_default_model=gemini_prompt_builder_model,
        )

        openai_image_cfg = (
            image_generation_cfg.get("openai", {})
            if isinstance(image_generation_cfg.get("openai"), dict)
            else {}
        )
        nanobanana_image_cfg = (
            image_generation_cfg.get("nanobanana", {})
            if isinstance(image_generation_cfg.get("nanobanana"), dict)
            else {}
        )
        image_generation_prompts = (
            image_generation_cfg.get("prompts", {})
            if isinstance(image_generation_cfg.get("prompts"), dict)
            else {}
        )

        return ResolvedConfig(
            manuscript_model=script_generation_llm.model,
            media_model=image_description_llm.model,
            script_prompt=str(
                prompts.get(
                    "scriptPrompt",
                    "Return JSON with key lines and 3-4 short factual lines suitable for voice-over.",
                )
            ),
            placement_prompt=str(prompts.get("placementPrompt", "")),
            describe_images_prompt=str(prompts.get("describeImagesPrompt", "")),
            llm=ResolvedLLMConfig(
                default_provider=default_llm_provider,
                script_generation=script_generation_llm,
                image_description=image_description_llm,
                asset_placement=asset_placement_llm,
                image_prompt_builder=image_prompt_builder_llm,
            ),
            image_generation=ResolvedImageGenerationConfig(
                enabled=bool(_pick_value(image_generation_cfg, "enabled", default=False)),
                provider=str(_pick_value(image_generation_cfg, "provider", default="openai")),
                prompt_builder_model=image_prompt_builder_llm.model,
                variants=int(_pick_value(image_generation_cfg, "variants", default=1) or 1),
                prefer_generated=bool(
                    _pick_value(
                        image_generation_cfg,
                        "prefer_generated",
                        "preferGenerated",
                        default=True,
                    )
                ),
                brief_prompt=str(
                    _pick_value(
                        image_generation_prompts,
                        "brief_prompt",
                        "briefPrompt",
                        default="",
                    )
                ),
                openai_prompt_builder=str(
                    _pick_value(
                        image_generation_prompts,
                        "openai_prompt_builder",
                        "openaiPromptBuilder",
                        default="",
                    )
                ),
                nanobanana_prompt_builder=str(
                    _pick_value(
                        image_generation_prompts,
                        "nanobanana_prompt_builder",
                        "nanobananaPromptBuilder",
                        default="",
                    )
                ),
                openai=ResolvedOpenAIImageGenerationConfig(
                    model=str(openai_image_cfg.get("model", openai_prompt_builder_model)),
                    size=str(openai_image_cfg.get("size", "1024x1536")),
                    quality=str(openai_image_cfg.get("quality", "high")),
                    background=str(openai_image_cfg.get("background", "opaque")),
                ),
                nanobanana=ResolvedNanobananaImageGenerationConfig(
                    model=str(
                        nanobanana_image_cfg.get(
                            "model", "gemini-2.5-flash-image-preview"
                        )
                    ),
                    aspect_ratio=str(
                        _pick_value(
                            nanobanana_image_cfg,
                            "aspect_ratio",
                            "aspectRatio",
                            default="9:16",
                        )
                    ),
                    thinking_budget=(
                        str(
                            _pick_value(
                                nanobanana_image_cfg,
                                "thinking_budget",
                                "thinkingBudget",
                            )
                        )
                        if _pick_value(
                            nanobanana_image_cfg,
                            "thinking_budget",
                            "thinkingBudget",
                        )
                        is not None
                        else None
                    ),
                ),
            ),
            tts_provider=tts_provider,
            voice_id=voice_id,
            tts_model_id=tts_model_id,
            voice_settings=voice_settings,
            segment_pause_seconds=float(pause),
            player=player,
            export_defaults=export_defaults,
        )
