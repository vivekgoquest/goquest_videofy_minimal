from pathlib import Path

import pytest

from api.config_resolver import ConfigResolver, ConfigResolverError
from api.schemas import GenerationManifest


def test_brand_people_default_overrides_google_tts_defaults(tmp_path: Path):
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        '{"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4o"},"audio":{"tts":"google"},"options":{"segmentPauseSeconds":0.4},"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts","instructions":"Warm, precise explainer delivery.","stability":1,"similarity_boost":1,"style":0,"use_speaker_boost":true}},"prompts":{"scriptPrompt":"brand-prompt"}}',
        encoding="utf-8",
    )

    resolver = ConfigResolver(config_root)
    manifest = GenerationManifest(projectId="demo")

    resolved = resolver.resolve(manifest)

    assert resolved.tts_provider == "google"
    assert resolved.voice_id == "Kore"
    assert resolved.voice_settings["stability"] == 1
    assert resolved.voice_settings["similarity_boost"] == 1
    assert resolved.voice_settings["style"] == 0
    assert resolved.voice_settings["use_speaker_boost"] is True
    assert resolved.voice_settings["instructions"] == "Warm, precise explainer delivery."
    assert resolved.tts_model_id == "gemini-2.5-pro-preview-tts"
    assert resolved.script_prompt == "brand-prompt"
    assert resolved.manuscript_model == "gpt-4o-mini"
    assert resolved.media_model == "gpt-4o"


def test_brand_must_define_voice(tmp_path: Path):
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        '{"audio":{"tts":"google"},"people":{"default":{}}}',
        encoding="utf-8",
    )

    resolver = ConfigResolver(config_root)
    manifest = GenerationManifest(projectId="demo")

    with pytest.raises(ConfigResolverError, match="people.default.voice"):
        resolver.resolve(manifest)


def test_brand_rejects_legacy_elevenlabs_provider(tmp_path: Path):
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        '{"audio":{"tts":"elevenlabs"},"people":{"default":{"voice":"Kore"}}}',
        encoding="utf-8",
    )

    resolver = ConfigResolver(config_root)
    manifest = GenerationManifest(projectId="demo")

    with pytest.raises(ConfigResolverError, match="audio.tts='google'"):
        resolver.resolve(manifest)


def test_brand_can_default_to_gemini_with_per_node_overrides(tmp_path: Path):
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        (
            '{'
            '"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4.1-mini"},'
            '"gemini":{"manuscriptModel":"gemini-2.5-flash","mediaModel":"gemini-2.5-pro","promptBuilderModel":"gemini-2.5-flash"},'
            '"llm":{"defaultProvider":"gemini","nodes":{"assetPlacement":{"provider":"openai","model":"gpt-4.1-mini"}}},'
            '"audio":{"tts":"google"},'
            '"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts"}}'
            '}'
        ),
        encoding="utf-8",
    )

    resolver = ConfigResolver(config_root)
    manifest = GenerationManifest(projectId="demo")

    resolved = resolver.resolve(manifest)

    assert resolved.llm.default_provider == "gemini"
    assert resolved.llm.script_generation.provider == "gemini"
    assert resolved.llm.script_generation.model == "gemini-2.5-flash"
    assert resolved.llm.image_description.provider == "gemini"
    assert resolved.llm.image_description.model == "gemini-2.5-pro"
    assert resolved.llm.asset_placement.provider == "openai"
    assert resolved.llm.asset_placement.model == "gpt-4.1-mini"
    assert resolved.llm.image_prompt_builder.provider == "gemini"
    assert resolved.llm.image_prompt_builder.model == "gemini-2.5-flash"


def test_resolve_applies_runtime_llm_override(tmp_path: Path):
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        (
            '{'
            '"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4.1-mini"},'
            '"gemini":{"manuscriptModel":"gemini-2.5-flash","mediaModel":"gemini-2.5-pro","promptBuilderModel":"gemini-2.5-flash"},'
            '"audio":{"tts":"google"},'
            '"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts"}}'
            '}'
        ),
        encoding="utf-8",
    )

    resolver = ConfigResolver(config_root)
    manifest = GenerationManifest(projectId="demo")

    resolved = resolver.resolve(
        manifest,
        llm_override={
            "default_provider": "gemini",
            "nodes": {
                "script_generation": {"provider": "gemini", "model": "gemini-2.5-pro"},
                "asset_placement": {"provider": "openai", "model": "gpt-4.1"},
            },
        },
    )

    assert resolved.llm.default_provider == "gemini"
    assert resolved.llm.script_generation.provider == "gemini"
    assert resolved.llm.script_generation.model == "gemini-2.5-pro"
    assert resolved.llm.image_description.provider == "gemini"
    assert resolved.llm.asset_placement.provider == "openai"
    assert resolved.llm.asset_placement.model == "gpt-4.1"


def test_resolve_applies_snake_case_image_generation_override(tmp_path: Path):
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        (
            "{"
            '"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4.1-mini"},'
            '"gemini":{"manuscriptModel":"gemini-2.5-flash","mediaModel":"gemini-2.5-pro","promptBuilderModel":"gemini-2.5-flash"},'
            '"imageGeneration":{"enabled":false,"provider":"openai","variants":1,"preferGenerated":true,'
            '"prompts":{"briefPrompt":"brand brief","openaiPromptBuilder":"brand-openai","nanobananaPromptBuilder":"brand-nano"},'
            '"openai":{"model":"gpt-5-mini","size":"1024x1536","quality":"high","background":"opaque"},'
            '"nanobanana":{"model":"gemini-2.5-flash-image-preview","aspectRatio":"9:16","thinkingBudget":"low"}},'
            '"audio":{"tts":"google"},'
            '"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts"}}'
            "}"
        ),
        encoding="utf-8",
    )

    resolver = ConfigResolver(config_root)
    manifest = GenerationManifest(projectId="demo")

    resolved = resolver.resolve(
        manifest,
        image_generation_override={
            "enabled": True,
            "provider": "nanobanana",
            "variants": 3,
            "prefer_generated": False,
            "prompts": {
                "brief_prompt": "override brief",
                "openai_prompt_builder": "override-openai",
                "nanobanana_prompt_builder": "override-nano",
            },
            "openai": {"background": "transparent"},
            "nanobanana": {
                "model": "gemini-2.5-pro-image-preview",
                "aspect_ratio": "16:9",
                "thinking_budget": "high",
            },
        },
    )

    assert resolved.image_generation.enabled is True
    assert resolved.image_generation.provider == "nanobanana"
    assert resolved.image_generation.variants == 3
    assert resolved.image_generation.prefer_generated is False
    assert resolved.image_generation.brief_prompt == "override brief"
    assert resolved.image_generation.openai_prompt_builder == "override-openai"
    assert resolved.image_generation.nanobanana_prompt_builder == "override-nano"
    assert resolved.image_generation.openai.background == "transparent"
    assert resolved.image_generation.nanobanana.model == "gemini-2.5-pro-image-preview"
    assert resolved.image_generation.nanobanana.aspect_ratio == "16:9"
    assert resolved.image_generation.nanobanana.thinking_budget == "high"
