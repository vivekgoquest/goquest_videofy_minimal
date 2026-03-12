import base64
from pathlib import Path
from types import SimpleNamespace

from api.config_resolver import (
    ResolvedConfig,
    ResolvedImageGenerationConfig,
    ResolvedLLMConfig,
    ResolvedLLMNodeConfig,
    ResolvedNanobananaImageGenerationConfig,
    ResolvedOpenAIImageGenerationConfig,
)
from api.image_generation_service import (
    ImageGenerationService,
    ImagePromptContext,
    NanoBananaImagePromptSpec,
    OpenAIImagePromptSpec,
    format_nanobanana_image_prompt,
    format_openai_image_prompt,
)
from api.project_store import ProjectStore
from api.settings import Settings


def _resolved_config() -> ResolvedConfig:
    return ResolvedConfig(
        manuscript_model="gpt-4o-mini",
        media_model="gpt-4o",
        script_prompt="script",
        placement_prompt="placement",
        describe_images_prompt="describe",
        llm=ResolvedLLMConfig(
            default_provider="openai",
            script_generation=ResolvedLLMNodeConfig(provider="openai", model="gpt-4o-mini"),
            image_description=ResolvedLLMNodeConfig(provider="openai", model="gpt-4o"),
            asset_placement=ResolvedLLMNodeConfig(provider="openai", model="gpt-4o"),
            image_prompt_builder=ResolvedLLMNodeConfig(
                provider="openai",
                model="gpt-5-mini",
            ),
        ),
        image_generation=ResolvedImageGenerationConfig(
            enabled=True,
            provider="openai",
            prompt_builder_model="gpt-5-mini",
            variants=1,
            prefer_generated=True,
            brief_prompt="brief",
            openai_prompt_builder="openai builder",
            nanobanana_prompt_builder="nano builder",
            openai=ResolvedOpenAIImageGenerationConfig(
                model="gpt-5-mini",
                size="1024x1536",
                quality="high",
                background="transparent",
            ),
            nanobanana=ResolvedNanobananaImageGenerationConfig(
                model="gemini-2.5-flash-image-preview",
                aspect_ratio="16:9",
                thinking_budget="high",
            ),
        ),
        tts_provider="google",
        voice_id="Kore",
        tts_model_id="gemini-2.5-pro-preview-tts",
        voice_settings={},
        segment_pause_seconds=0.4,
        player={},
        export_defaults={},
    )


def _context() -> ImagePromptContext:
    return ImagePromptContext(
        project_id="demo",
        article_title="Demo",
        article_byline="Tester",
        article_text="Body",
        script_lines=["Line 1"],
        line_index=0,
        line_text="Line 1",
        previous_line=None,
        next_line=None,
        brief_prompt="brief",
    )


def _service(tmp_path: Path) -> tuple[ImageGenerationService, ProjectStore]:
    projects_root = tmp_path / "projects"
    store = ProjectStore(projects_root)
    store.ensure_layout("demo")
    settings = Settings(
        projects_root=projects_root,
        config_root=tmp_path / "brands",
        app_base_url="http://localhost:8001",
        openai_api_key="test-openai",
        google_api_key="test-gemini",
    )
    return ImageGenerationService(settings=settings, store=store), store


def test_format_openai_image_prompt_uses_xml_structure():
    prompt = format_openai_image_prompt(
        "Create one still image for a narrative explainer video segment.",
        OpenAIImagePromptSpec(
            scene="A tired father studies a foreclosure notice at a kitchen table.",
            setting="A middle-class home at night with visible everyday wear.",
            composition="Close framing on the notice, hands, and worried face with layered depth.",
            lighting="Single warm practical bulb over the table with cool shadow falloff behind.",
            color_palette="Amber-brown dominant tones with a muted crimson accent on the notice.",
            style="Painterly editorial illustration with tactile materials and cinematic contrast.",
            negative_constraints="No text overlays. No watermarks.",
        ),
    )

    assert prompt.startswith("<render_prompt>")
    assert "<scene>A tired father studies a foreclosure notice at a kitchen table.</scene>" in prompt
    assert "<negative_constraints>No text overlays. No watermarks.</negative_constraints>" in prompt


def test_format_nanobanana_image_prompt_uses_brief_structure():
    prompt = format_nanobanana_image_prompt(
        "Create a single still image for a narrative explainer video segment.",
        NanoBananaImagePromptSpec(
            subject="A father holding a foreclosure notice",
            environment="A modest kitchen late at night",
            composition="Tight framing with hands and notice in the foreground",
            lighting="Single tungsten bulb with deep blue shadows",
            palette="Amber browns with a restrained crimson accent",
            style="Editorial illustration with cinematic realism",
            constraints="No text overlays, no watermarks",
        ),
    )

    assert prompt.startswith("Create a single still image for a narrative explainer video segment.")
    assert "Subject: A father holding a foreclosure notice." in prompt
    assert "Constraints: No text overlays, no watermarks." in prompt


def test_openai_generation_forwards_background_setting(tmp_path: Path):
    service, _store = _service(tmp_path)
    recorded_calls: list[dict] = []
    service._llm = SimpleNamespace(
        parse_structured_payload=lambda **_: OpenAIImagePromptSpec(
            scene="Scene",
            setting="Setting",
            composition="Composition",
            lighting="Lighting",
            color_palette="Palette",
            style="Style",
            negative_constraints="No text",
        )
    )
    service._openai_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: recorded_calls.append(kwargs)
            or SimpleNamespace(
                id="resp_1",
                output=[
                    SimpleNamespace(
                        type="image_generation_call",
                        result=base64.b64encode(b"png-bytes").decode("ascii"),
                        revised_prompt="revised",
                    )
                ],
            )
        )
    )

    service._generate_openai_assets(
        project_id="demo",
        context=_context(),
        resolved_config=_resolved_config(),
        image_config=_resolved_config().image_generation,
    )

    assert recorded_calls[0]["tools"][0]["background"] == "transparent"


def test_nanobanana_generation_forwards_aspect_ratio_setting(tmp_path: Path):
    service, _store = _service(tmp_path)
    recorded_calls: list[dict] = []
    resolved_config = _resolved_config()
    resolved_config.llm.image_prompt_builder = ResolvedLLMNodeConfig(
        provider="gemini",
        model="gemini-2.5-flash",
    )
    service._llm = SimpleNamespace(
        parse_structured_payload=lambda **_: NanoBananaImagePromptSpec(
            subject="Subject",
            environment="Environment",
            composition="Composition",
            lighting="Lighting",
            palette="Palette",
            style="Style",
            constraints="No text",
        )
    )
    service._google_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_images=lambda **kwargs: recorded_calls.append(kwargs)
            or SimpleNamespace(
                generated_images=[
                    SimpleNamespace(
                        image=SimpleNamespace(
                            image_bytes=b"png-bytes",
                            mime_type="image/png",
                        )
                    )
                ]
            )
        )
    )

    service._generate_nanobanana_assets(
        project_id="demo",
        context=_context(),
        resolved_config=resolved_config,
        image_config=resolved_config.image_generation,
    )

    assert recorded_calls[0]["config"].aspect_ratio == "16:9"
