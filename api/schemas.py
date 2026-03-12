from datetime import datetime, timezone
from typing import Any, Literal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ArticleImageInput(BaseModel):
    path: str
    byline: str | None = None


class ArticleVideoInput(BaseModel):
    path: str
    byline: str | None = None
    start_from: float | None = None
    end_at: float | None = None


class ArticleInput(BaseModel):
    title: str
    byline: str = ""
    pubdate: datetime
    text: str
    script_lines: list[str] | None = None
    images: list[ArticleImageInput] = Field(default_factory=list)
    videos: list[ArticleVideoInput] = Field(default_factory=list)


class GenerationManifestOptions(BaseModel):
    orientationDefault: Literal["vertical", "horizontal"] = "vertical"
    segmentPauseSeconds: float | None = None


class GenerationManifest(BaseModel):
    projectId: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    brandId: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")] = "default"
    promptPack: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")] = "default"
    voicePack: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")] = "default"
    options: GenerationManifestOptions = Field(default_factory=GenerationManifestOptions)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TextLine(BaseModel):
    type: Literal["text"] = "text"
    text: str
    line_id: int
    who: str = "default"
    start: float | None = None
    end: float | None = None
    displayText: str | None = None


class Hotspot(BaseModel):
    x: float
    y: float
    width: float
    height: float
    x_norm: float | None = None
    y_norm: float | None = None
    width_norm: float | None = None
    height_norm: float | None = None


class MediaSize(BaseModel):
    width: int
    height: int


class ImageAssetRef(BaseModel):
    id: str
    size: MediaSize


class VideoStreamUrls(BaseModel):
    hls: str | None = None
    hds: str | None = None
    mp4: str | None = None
    pseudostreaming: list[str] | None = None


class VideoAssetRef(BaseModel):
    id: str
    title: str
    streamUrls: VideoStreamUrls
    assetType: Literal["audio", "video"] | None = None
    displays: int | None = None
    duration: float | None = None


class MediaAssetImage(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["image"] = "image"
    path: str
    url: str
    byline: str | None = None
    imageAsset: ImageAssetRef | None = None
    hotspot: Hotspot | None = None
    description: str | None = None


class MediaAssetVideo(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["video"] = "video"
    path: str
    url: str
    byline: str | None = None
    start_from: float | None = None
    end_at: float | None = None
    changedId: str | None = None
    videoAsset: VideoAssetRef | None = None
    description: str | None = None


class Segment(BaseModel):
    id: int
    mood: str = "neutral"
    type: str = "segment"
    style: str = "bottom"
    cameraMovement: str = "none"
    texts: list[TextLine]
    text: str | None = None
    images: list[MediaAssetImage | MediaAssetVideo] = Field(default_factory=list)
    start: float | None = None
    end: float | None = None


class ManuscriptMeta(BaseModel):
    title: str
    byline: str
    pubdate: datetime
    id: int = 1
    uniqueId: str
    articleUrl: str | None = None
    description: str = ""
    audio: dict[str, Any] = Field(default_factory=dict)


class Manuscript(BaseModel):
    meta: ManuscriptMeta
    segments: list[Segment]
    media: list[MediaAssetImage | MediaAssetVideo] | None = None


class GenerateOpenAIImageOverride(BaseModel):
    model: str | None = None
    size: str | None = None
    quality: str | None = None
    background: str | None = None


class GenerateNanobananaImageOverride(BaseModel):
    model: str | None = None
    aspect_ratio: str | None = None
    thinking_budget: str | None = None


class GenerateImagePromptOverride(BaseModel):
    brief_prompt: str | None = None
    openai_prompt_builder: str | None = None
    nanobanana_prompt_builder: str | None = None


class GenerateLLMNodeOverride(BaseModel):
    provider: Literal["openai", "gemini"] | None = None
    model: str | None = None


class GenerateLLMNodesOverride(BaseModel):
    script_generation: GenerateLLMNodeOverride | None = None
    image_description: GenerateLLMNodeOverride | None = None
    asset_placement: GenerateLLMNodeOverride | None = None
    image_prompt_builder: GenerateLLMNodeOverride | None = None


class GenerateLLMOverride(BaseModel):
    default_provider: Literal["openai", "gemini"] | None = None
    nodes: GenerateLLMNodesOverride | None = None


class GenerateImageOverride(BaseModel):
    enabled: bool | None = None
    provider: Literal["openai", "nanobanana"] | None = None
    prompt_builder_model: str | None = None
    variants: int | None = None
    prefer_generated: bool | None = None
    prompts: GenerateImagePromptOverride | None = None
    openai: GenerateOpenAIImageOverride | None = None
    nanobanana: GenerateNanobananaImageOverride | None = None


class GenerateRequest(BaseModel):
    script_prompt: str | None = None
    llm: GenerateLLMOverride | None = None
    image_generation: GenerateImageOverride | None = None


GenerateImageGenerationOverride = GenerateImageOverride


class ProcessRequest(BaseModel):
    manuscript: Manuscript | None = None


class GenerationResponse(BaseModel):
    project_id: str
    status: str
    manuscript_json: dict | None = None
    processed_json: dict | None = None


class UploadResponse(BaseModel):
    path: str
    url: str


class SummarizationResult(BaseModel):
    lines: list[str]
