from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised when dependency is missing at runtime
    genai = None
    genai_types = None

from .asset_analysis import AnalysisInputAsset
from .config_resolver import ResolvedConfig, ResolvedImageGenerationConfig
from .llm_service import LLMService
from .project_store import ProjectStore
from .schemas import ArticleInput
from .settings import Settings

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _thinking_budget_value(level: str) -> int:
    mapping = {
        "none": 0,
        "low": 256,
        "medium": 1024,
        "high": 2048,
    }
    return mapping.get(level, 256)


@dataclass
class ImagePromptContext:
    project_id: str
    article_title: str
    article_byline: str
    article_text: str
    script_lines: list[str]
    line_index: int
    line_text: str
    previous_line: str | None
    next_line: str | None
    brief_prompt: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "article_title": self.article_title,
            "article_byline": self.article_byline,
            "article_text": self.article_text,
            "script_lines": self.script_lines,
            "line_number": self.line_index + 1,
            "line_text": self.line_text,
            "previous_line": self.previous_line,
            "next_line": self.next_line,
            "brief_prompt": self.brief_prompt,
        }


@dataclass
class GeneratedImageAsset:
    asset_id: str
    line_index: int
    rel_path: str
    metadata_rel_path: str
    url: str
    byline: str | None


class OpenAIImagePromptSpec(BaseModel):
    scene: str
    setting: str
    composition: str
    lighting: str
    color_palette: str
    style: str
    negative_constraints: str
    caption: str | None = None


class NanoBananaImagePromptSpec(BaseModel):
    subject: str
    environment: str
    composition: str
    lighting: str
    palette: str
    style: str
    constraints: str
    caption: str | None = None


class ImageGenerationService:
    def __init__(
        self,
        settings: Settings,
        store: ProjectStore,
    ):
        self._settings = settings
        self._store = store
        self._llm = LLMService(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            google_api_key=settings.gemini_api_key,
        )
        self._openai_client = (
            OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        )
        self._google_client = (
            genai.Client(api_key=settings.nanobanana_api_key)
            if settings.nanobanana_api_key and genai is not None
            else None
        )

    def generate_for_script_lines(
        self,
        project_id: str,
        article: ArticleInput,
        script_lines: list[str],
        resolved_config: ResolvedConfig,
    ) -> list[GeneratedImageAsset]:
        image_config = resolved_config.image_generation
        if not image_config.enabled:
            return []

        generated_assets: list[GeneratedImageAsset] = []
        for line_index, line_text in enumerate(script_lines):
            context = ImagePromptContext(
                project_id=project_id,
                article_title=article.title,
                article_byline=article.byline,
                article_text=article.text,
                script_lines=script_lines,
                line_index=line_index,
                line_text=line_text,
                previous_line=script_lines[line_index - 1] if line_index > 0 else None,
                next_line=script_lines[line_index + 1]
                if line_index + 1 < len(script_lines)
                else None,
                brief_prompt=image_config.brief_prompt,
            )
            generated_assets.extend(
                self._generate_for_context(
                    project_id=project_id,
                    context=context,
                    resolved_config=resolved_config,
                    image_config=image_config,
                )
            )

        self._store.save_json(
            project_id,
            "working/image-generation/run.json",
            {
                "version": 1,
                "createdAt": _now_iso(),
                "provider": image_config.provider,
                "lineCount": len(script_lines),
                "generatedCount": len(generated_assets),
                "assets": [
                    {
                        "assetId": asset.asset_id,
                        "lineNumber": asset.line_index + 1,
                        "path": asset.rel_path,
                        "metadataPath": asset.metadata_rel_path,
                        "url": asset.url,
                    }
                    for asset in generated_assets
                ],
            },
        )
        return generated_assets

    def _generate_for_context(
        self,
        project_id: str,
        context: ImagePromptContext,
        resolved_config: ResolvedConfig,
        image_config: ResolvedImageGenerationConfig,
    ) -> list[GeneratedImageAsset]:
        provider = image_config.provider
        if provider == "openai":
            return self._generate_openai_assets(
                project_id,
                context,
                resolved_config,
                image_config,
            )
        if provider == "nanobanana":
            return self._generate_nanobanana_assets(
                project_id,
                context,
                resolved_config,
                image_config,
            )
        raise ValueError(f"Unsupported image generation provider '{provider}'")

    def _generate_openai_assets(
        self,
        project_id: str,
        context: ImagePromptContext,
        resolved_config: ResolvedConfig,
        image_config: ResolvedImageGenerationConfig,
    ) -> list[GeneratedImageAsset]:
        if self._openai_client is None:
            raise ValueError(
                "OPENAI_API_KEY is required to generate AI images with provider 'openai'"
            )

        prompt_builder_provider = resolved_config.llm.image_prompt_builder.provider
        prompt_builder_model = resolved_config.llm.image_prompt_builder.model
        prompt_spec = self._llm.parse_structured_payload(
            provider=prompt_builder_provider,
            model=prompt_builder_model,
            system_prompt=image_config.openai_prompt_builder,
            payload=context.as_payload(),
            response_model=OpenAIImagePromptSpec,
            temperature=0.5,
            max_output_tokens=700,
            missing_key_message=(
                f"{'OPENAI_API_KEY' if prompt_builder_provider == 'openai' else 'GEMINI_API_KEY or GOOGLE_API_KEY'} "
                "is required to build OpenAI image prompts with the selected provider"
            ),
        )

        final_prompt = self._render_openai_prompt(context=context, spec=prompt_spec)
        generated_assets: list[GeneratedImageAsset] = []
        for variant_index in range(image_config.variants):
            response = self._openai_client.responses.create(
                model=image_config.openai.model,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": final_prompt}],
                    }
                ],
                tools=[
                    {
                        "type": "image_generation",
                        "size": image_config.openai.size,
                        "quality": image_config.openai.quality,
                        "background": image_config.openai.background,
                    }
                ],
            )

            image_bytes: bytes | None = None
            revised_prompt: str | None = None
            assistant_text: str | None = None
            output_types: list[str] = []
            for output in response.output:
                output_types.append(output.type)
                if output.type == "image_generation_call" and getattr(output, "result", None):
                    image_bytes = base64.b64decode(output.result)
                    revised_prompt = getattr(output, "revised_prompt", None)
                if output.type == "message" and assistant_text is None:
                    content = getattr(output, "content", None) or []
                    for item in content:
                        text = getattr(item, "text", None)
                        if text:
                            assistant_text = text
                            break

            if image_bytes is None:
                raise ValueError(
                    f"OpenAI image generation returned no image for line {context.line_index + 1}"
                )

            rel_path = (
                f"working/generated-images/seg-{context.line_index + 1:03}-"
                f"v{variant_index + 1:02}.png"
            )
            metadata_rel_path = (
                f"working/generated-prompts/seg-{context.line_index + 1:03}-"
                f"v{variant_index + 1:02}.json"
            )
            self._store.save_bytes(project_id, rel_path, image_bytes)
            self._store.save_json(
                project_id,
                metadata_rel_path,
                {
                    "provider": "openai",
                    "createdAt": _now_iso(),
                    "lineNumber": context.line_index + 1,
                    "lineText": context.line_text,
                    "promptBuilderProvider": prompt_builder_provider,
                    "promptBuilderModel": prompt_builder_model,
                    "generationModel": image_config.openai.model,
                    "finalPrompt": final_prompt,
                    "revisedPrompt": revised_prompt,
                    "assistantText": assistant_text,
                    "promptSpec": prompt_spec.model_dump(mode="json", exclude_none=True),
                    "responseId": getattr(response, "id", None),
                    "outputTypes": output_types,
                },
            )
            generated_assets.append(
                GeneratedImageAsset(
                    asset_id=f"gen-{context.line_index + 1:03}-{variant_index + 1:02}",
                    line_index=context.line_index,
                    rel_path=rel_path,
                    metadata_rel_path=metadata_rel_path,
                    url=self._asset_url(project_id, rel_path),
                    byline="AI generated via OpenAI",
                )
            )

        return generated_assets

    def _generate_nanobanana_assets(
        self,
        project_id: str,
        context: ImagePromptContext,
        resolved_config: ResolvedConfig,
        image_config: ResolvedImageGenerationConfig,
    ) -> list[GeneratedImageAsset]:
        if self._google_client is None or genai_types is None:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required to generate AI images with provider 'nanobanana'"
            )

        prompt_builder_provider = resolved_config.llm.image_prompt_builder.provider
        prompt_builder_model = resolved_config.llm.image_prompt_builder.model
        prompt_spec = self._llm.parse_structured_payload(
            provider=prompt_builder_provider,
            model=prompt_builder_model,
            system_prompt=image_config.nanobanana_prompt_builder,
            payload=context.as_payload(),
            response_model=NanoBananaImagePromptSpec,
            temperature=0.5,
            max_output_tokens=700,
            missing_key_message=(
                f"{'OPENAI_API_KEY' if prompt_builder_provider == 'openai' else 'GEMINI_API_KEY or GOOGLE_API_KEY'} "
                "is required to build Nano Banana prompts with the selected provider"
            ),
        )
        final_prompt = self._render_nanobanana_prompt(context=context, spec=prompt_spec)

        generated_assets: list[GeneratedImageAsset] = []
        for variant_index in range(image_config.variants):
            response = self._google_client.models.generate_images(
                model=image_config.nanobanana.model,
                prompt=final_prompt,
                config=genai_types.GenerateImagesConfig(
                    aspect_ratio=image_config.nanobanana.aspect_ratio,
                    number_of_images=1,
                ),
            )

            generated_image = None
            generated_images = getattr(response, "generated_images", None) or []
            if generated_images:
                generated_image = getattr(generated_images[0], "image", None)

            if generated_image is None:
                raise ValueError(
                    f"Nano Banana image generation returned no image for line {context.line_index + 1}"
                )

            mime_type = getattr(generated_image, "mime_type", None) or "image/png"
            extension = "jpg" if mime_type == "image/jpeg" else "png"
            rel_path = (
                f"working/generated-images/seg-{context.line_index + 1:03}-"
                f"v{variant_index + 1:02}.{extension}"
            )
            metadata_rel_path = (
                f"working/generated-prompts/seg-{context.line_index + 1:03}-"
                f"v{variant_index + 1:02}.json"
            )
            self._store.save_bytes(project_id, rel_path, self._blob_bytes(generated_image))
            self._store.save_json(
                project_id,
                metadata_rel_path,
                {
                    "provider": "nanobanana",
                    "createdAt": _now_iso(),
                    "lineNumber": context.line_index + 1,
                    "lineText": context.line_text,
                    "promptBuilderProvider": prompt_builder_provider,
                    "promptBuilderModel": prompt_builder_model,
                    "generationModel": image_config.nanobanana.model,
                    "aspectRatio": image_config.nanobanana.aspect_ratio,
                    "thinkingBudget": image_config.nanobanana.thinking_budget,
                    "finalPrompt": final_prompt,
                    "promptSpec": prompt_spec.model_dump(mode="json", exclude_none=True),
                    "assistantText": None,
                },
            )
            generated_assets.append(
                GeneratedImageAsset(
                    asset_id=f"gen-{context.line_index + 1:03}-{variant_index + 1:02}",
                    line_index=context.line_index,
                    rel_path=rel_path,
                    metadata_rel_path=metadata_rel_path,
                    url=self._asset_url(project_id, rel_path),
                    byline="AI generated via Nano Banana",
                )
            )

        return generated_assets

    def _render_openai_prompt(
        self,
        context: ImagePromptContext,
        spec: OpenAIImagePromptSpec,
    ) -> str:
        return "\n".join(
            [
                "<render_prompt>",
                f"  <task>{context.brief_prompt or 'Create one still image for a vertical narrative explainer video segment.'}</task>",
                f"  <scene>{spec.scene}</scene>",
                f"  <setting>{spec.setting}</setting>",
                f"  <composition>{spec.composition}</composition>",
                "  <lighting>",
                f"    <primary_source>{spec.lighting}</primary_source>",
                "  </lighting>",
                f"  <color_palette>{spec.color_palette}</color_palette>",
                f"  <style>{spec.style}</style>",
                f"  <negative_constraints>{spec.negative_constraints}</negative_constraints>",
                "</render_prompt>",
            ]
        )

    def _render_nanobanana_prompt(
        self,
        context: ImagePromptContext,
        spec: NanoBananaImagePromptSpec,
    ) -> str:
        aspect_ratio = "Use a vertical 9:16 composition." if spec else ""
        return "\n".join(
            [
                context.brief_prompt
                or "Create a single still image for a narrative explainer video segment.",
                aspect_ratio,
                f"Subject: {spec.subject}.",
                f"Environment: {spec.environment}.",
                f"Composition: {spec.composition}.",
                f"Lighting: {spec.lighting}.",
                f"Palette: {spec.palette}.",
                f"Style: {spec.style}.",
                f"Constraints: {spec.constraints}.",
                f"Story beat: line {context.line_index + 1} of {len(context.script_lines)}: {context.line_text}.",
            ]
        ).strip()

    def _asset_url(self, project_id: str, rel_path: str) -> str:
        return f"{self._settings.app_base_url}/projects/{project_id}/files/{rel_path}"

    def _google_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        parts = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    parts.append(part_text.strip())
        return "\n".join(parts).strip()

    def _first_google_image(self, response: Any) -> Any | None:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None and getattr(inline_data, "data", None):
                    return inline_data
        return None

    def _blob_bytes(self, blob: Any) -> bytes:
        image_bytes = getattr(blob, "image_bytes", None)
        if isinstance(image_bytes, bytes):
            return image_bytes
        if isinstance(image_bytes, str):
            return base64.b64decode(image_bytes)
        data = getattr(blob, "data", b"")
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return base64.b64decode(data)
        raise ValueError("Unsupported Nano Banana image payload")


def format_openai_image_prompt(brief_prompt: str, spec: OpenAIImagePromptSpec) -> str:
    task = brief_prompt or "Create one still image for a narrative explainer video segment."
    return "\n".join(
        [
            "<render_prompt>",
            f"  <task>{task}</task>",
            f"  <scene>{spec.scene}</scene>",
            f"  <setting>{spec.setting}</setting>",
            f"  <composition>{spec.composition}</composition>",
            "  <lighting>",
            f"    <primary_source>{spec.lighting}</primary_source>",
            "  </lighting>",
            f"  <color_palette>{spec.color_palette}</color_palette>",
            f"  <style>{spec.style}</style>",
            f"  <negative_constraints>{spec.negative_constraints}</negative_constraints>",
            "</render_prompt>",
        ]
    )


def format_nanobanana_image_prompt(brief_prompt: str, spec: NanoBananaImagePromptSpec) -> str:
    return "\n".join(
        [
            brief_prompt or "Create a single still image for a narrative explainer video segment.",
            "Use a vertical 9:16 composition.",
            f"Subject: {spec.subject}.",
            f"Environment: {spec.environment}.",
            f"Composition: {spec.composition}.",
            f"Lighting: {spec.lighting}.",
            f"Palette: {spec.palette}.",
            f"Style: {spec.style}.",
            f"Constraints: {spec.constraints}.",
        ]
    ).strip()


class AIImageGenerationService(ImageGenerationService):
    def generate_assets(
        self,
        project_id: str,
        article: ArticleInput,
        script_lines: list[str],
        resolved_config: ResolvedConfig,
        app_base_url: str | None = None,
    ) -> list[AnalysisInputAsset]:
        generated_assets = self.generate_for_script_lines(
            project_id=project_id,
            article=article,
            script_lines=script_lines,
            resolved_config=resolved_config,
        )
        return [
            AnalysisInputAsset(
                asset_id=asset.asset_id,
                type="image",
                rel_path=asset.rel_path,
                local_path=self._store.resolve_asset_path(project_id, asset.rel_path),
                url=asset.url,
                byline=asset.byline,
            )
            for asset in generated_assets
        ]
