import json
import time
from pathlib import Path

from api.asset_analysis import AssetAnalysisResult, AssetAnalysisService
from api.config_resolver import ConfigResolver
from api.image_generation_service import GeneratedImageAsset
from api.llm_service import LLMService
from api.pipeline import PipelineService
from api.project_store import ProjectStore
from api.settings import Settings
from api.tts_service import GeminiTTSService


def write_brand_config(config_root: Path) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        '{"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4o"},"audio":{"tts":"google"},"options":{"segmentPauseSeconds":0.4},"prompts":{"scriptPrompt":"Return JSON with lines."},"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts"}},"player":{"defaultCameraMovements":["pan-left","zoom-out"]}}',
        encoding="utf-8",
    )


def test_generate_manuscript_uses_script_lines_from_article(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    store = ProjectStore(projects_root)
    project_id = "demo"
    store.ensure_layout(project_id)

    image_path = store.project_path(project_id) / "input" / "images" / "a.jpg"
    image_path.write_bytes(b"jpg")

    store.save_json(
        project_id,
        "input/article.json",
        {
            "title": "Demo",
            "byline": "Tester",
            "pubdate": "2026-01-01T00:00:00Z",
            "text": "Long text",
            "script_lines": ["Line 1", "Line 2", "Line 3"],
            "images": [{"path": "images/a.jpg"}],
            "videos": [],
        },
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://localhost:8001",
    )

    pipeline = PipelineService(
        settings=settings,
        store=store,
        llm_service=LLMService(api_key="", model="gpt-4o-mini"),
        tts_service=GeminiTTSService(
            api_key="",
            voice_id="voice",
            ffprobe_bin="ffprobe",
            ffmpeg_bin="ffmpeg",
        ),
        config_resolver=ConfigResolver(settings.config_root_abs),
        asset_analysis_service=AssetAnalysisService(
            store=store,
            openai_api_key="",
            ffmpeg_bin=settings.ffmpeg_bin,
            ffprobe_bin=settings.ffprobe_bin,
        ),
    )

    manuscript = pipeline.generate_manuscript(project_id)

    assert len(manuscript.segments) == 3
    assert manuscript.segments[0].texts[0].text == "Line 1"
    assert manuscript.segments[0].images[0].path == "input/images/a.jpg"
    assert manuscript.segments[0].cameraMovement == "pan-left"
    assert manuscript.segments[1].cameraMovement == "zoom-out"
    assert manuscript.segments[2].cameraMovement == "pan-left"
    assert manuscript.segments[0].images[0].type == "image"
    assert manuscript.segments[0].images[0].imageAsset is not None

    analysis_dir = store.project_path(project_id) / "working" / "analysis"
    catalog = analysis_dir / "assets.catalog.json"
    descriptions = analysis_dir / "descriptions.json"
    placements = analysis_dir / "placements.json"
    hotspots = analysis_dir / "hotspots.json"
    video_scenes = analysis_dir / "video_scenes.json"
    run_file = analysis_dir / "run.json"
    assert catalog.exists()
    assert descriptions.exists()
    assert placements.exists()
    assert hotspots.exists()
    assert video_scenes.exists()
    assert run_file.exists()

    run_payload = json.loads(run_file.read_text(encoding="utf-8"))
    assert run_payload["assetCount"] == 1
    assert run_payload["lineCount"] == 3

    placement_payload = json.loads(placements.read_text(encoding="utf-8"))
    assert len(placement_payload["lineAssetIds"]) == 3


def test_generate_recomputes_analysis_on_each_run(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    store = ProjectStore(projects_root)
    project_id = "rerun"
    store.ensure_layout(project_id)
    image_path = store.project_path(project_id) / "input" / "images" / "a.jpg"
    image_path.write_bytes(b"jpg")

    store.save_json(
        project_id,
        "input/article.json",
        {
            "title": "Rerun Demo",
            "byline": "Tester",
            "pubdate": "2026-01-01T00:00:00Z",
            "text": "Long text",
            "script_lines": ["One", "Two"],
            "images": [{"path": "images/a.jpg"}],
            "videos": [],
        },
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://localhost:8001",
    )
    pipeline = PipelineService(
        settings=settings,
        store=store,
        llm_service=LLMService(api_key="", model="gpt-4o-mini"),
        tts_service=GeminiTTSService(
            api_key="",
            voice_id="voice",
            ffprobe_bin="ffprobe",
            ffmpeg_bin="ffmpeg",
        ),
        config_resolver=ConfigResolver(settings.config_root_abs),
        asset_analysis_service=AssetAnalysisService(
            store=store,
            openai_api_key="",
            ffmpeg_bin=settings.ffmpeg_bin,
            ffprobe_bin=settings.ffprobe_bin,
        ),
    )

    pipeline.generate_manuscript(project_id)
    analysis_dir = store.project_path(project_id) / "working" / "analysis"
    descriptions_before = json.loads(
        (analysis_dir / "descriptions.json").read_text(encoding="utf-8")
    )
    first_created_at = descriptions_before["createdAt"]

    time.sleep(0.01)
    pipeline.generate_manuscript(project_id)

    descriptions_after = json.loads(
        (analysis_dir / "descriptions.json").read_text(encoding="utf-8")
    )
    second_created_at = descriptions_after["createdAt"]
    assert second_created_at != first_created_at

    run_payload = json.loads((analysis_dir / "run.json").read_text(encoding="utf-8"))
    assert "cache" not in run_payload


class _FakeAssetAnalysisService:
    def __init__(self, app_base_url: str):
        self._app_base_url = app_base_url
        self.last_call: dict | None = None

    def analyze(
        self,
        project_id: str,
        script_lines: list[str],
        input_assets: list,
        describe_prompt: str,
        placement_prompt: str,
        describe_provider: str,
        describe_model: str,
        placement_provider: str,
        placement_model: str,
    ) -> AssetAnalysisResult:
        self.last_call = {
            "project_id": project_id,
            "describe_provider": describe_provider,
            "describe_model": describe_model,
            "placement_provider": placement_provider,
            "placement_model": placement_model,
        }
        assert len(script_lines) == 2
        return AssetAnalysisResult(
            assets=[
                {
                    "asset_id": "vid-001",
                    "type": "video",
                    "rel_path": "input/videos/a.mp4",
                    "url": f"{self._app_base_url}/projects/{project_id}/files/input/videos/a.mp4",
                    "description": "Video summary",
                    "videoAsset": {
                        "id": "input/videos/a.mp4",
                        "title": "a.mp4",
                        "duration": 10000,
                        "streamUrls": {
                            "mp4": f"{self._app_base_url}/projects/{project_id}/files/input/videos/a.mp4"
                        },
                    },
                    "videoScenes": [
                        {
                            "scene_id": "vid-001-scene-001",
                            "start_seconds": 1.0,
                            "end_seconds": 3.0,
                            "description": "Scene one",
                        },
                        {
                            "scene_id": "vid-001-scene-002",
                            "start_seconds": 3.0,
                            "end_seconds": 5.0,
                            "description": "Scene two",
                        },
                    ],
                }
            ],
            placement_asset_ids=["vid-001", "vid-001"],
            used_fallback_placement=False,
            hotspot_provider="noop",
            description_model="gpt-4o",
            placement_model="gpt-4o",
        )


class _FakeLLMService:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def summarize_into_lines(
        self,
        *,
        text: str,
        title: str,
        system_prompt: str,
        model_override: str | None = None,
        provider: str = "openai",
    ) -> list[str]:
        self.calls.append(
            {
                "text": text,
                "title": title,
                "system_prompt": system_prompt,
                "model_override": model_override or "",
                "provider": provider,
            }
        )
        return ["Line 1", "Line 2"]


class _FakeImageGenerationService:
    def __init__(self, app_base_url: str) -> None:
        self._app_base_url = app_base_url

    def generate_for_script_lines(
        self,
        project_id: str,
        article,
        script_lines: list[str],
        resolved_config,
    ) -> list[GeneratedImageAsset]:
        generated_assets = [
            GeneratedImageAsset(
                asset_id="gen-001-01",
                line_index=0,
                rel_path="working/generated-images/seg-001-v01.png",
                metadata_rel_path="working/generated-prompts/seg-001-v01.json",
                url=f"{self._app_base_url}/projects/{project_id}/files/working/generated-images/seg-001-v01.png",
                byline="AI generated",
            ),
            GeneratedImageAsset(
                asset_id="gen-001-02",
                line_index=0,
                rel_path="working/generated-images/seg-001-v02.png",
                metadata_rel_path="working/generated-prompts/seg-001-v02.json",
                url=f"{self._app_base_url}/projects/{project_id}/files/working/generated-images/seg-001-v02.png",
                byline="AI generated",
            ),
            GeneratedImageAsset(
                asset_id="gen-002-01",
                line_index=1,
                rel_path="working/generated-images/seg-002-v01.png",
                metadata_rel_path="working/generated-prompts/seg-002-v01.json",
                url=f"{self._app_base_url}/projects/{project_id}/files/working/generated-images/seg-002-v01.png",
                byline="AI generated",
            ),
        ]
        return generated_assets


class _FakeGeneratedAssetAnalysisService:
    def __init__(self, app_base_url: str):
        self._app_base_url = app_base_url

    def analyze(
        self,
        project_id: str,
        script_lines: list[str],
        input_assets: list,
        describe_prompt: str,
        placement_prompt: str,
        describe_provider: str,
        describe_model: str,
        placement_provider: str,
        placement_model: str,
    ) -> AssetAnalysisResult:
        assets = []
        for input_asset in input_assets:
            assets.append(
                {
                    "asset_id": input_asset.asset_id,
                    "type": "image",
                    "rel_path": input_asset.rel_path,
                    "url": input_asset.url,
                    "description": input_asset.asset_id,
                    "imageAsset": {
                        "id": input_asset.rel_path,
                        "size": {"width": 1080, "height": 1920},
                    },
                }
            )

        return AssetAnalysisResult(
            assets=assets,
            placement_asset_ids=[],
            used_fallback_placement=False,
            hotspot_provider="noop",
            description_model="gemini-2.5-flash",
            placement_model="gemini-2.5-flash",
        )


def test_generate_wires_video_scenes_into_segment_trims(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    store = ProjectStore(projects_root)
    project_id = "video-scenes"
    store.ensure_layout(project_id)

    video_path = store.project_path(project_id) / "input" / "videos" / "a.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")

    store.save_json(
        project_id,
        "input/article.json",
        {
            "title": "Video Demo",
            "byline": "Tester",
            "pubdate": "2026-01-01T00:00:00Z",
            "text": "Long text",
            "script_lines": ["Line 1", "Line 2"],
            "images": [],
            "videos": [{"path": "videos/a.mp4"}],
        },
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://localhost:8001",
    )

    pipeline = PipelineService(
        settings=settings,
        store=store,
        llm_service=LLMService(api_key="", model="gpt-4o-mini"),
        tts_service=GeminiTTSService(
            api_key="",
            voice_id="voice",
            ffprobe_bin="ffprobe",
            ffmpeg_bin="ffmpeg",
        ),
        config_resolver=ConfigResolver(settings.config_root_abs),
        asset_analysis_service=_FakeAssetAnalysisService(settings.app_base_url),  # type: ignore[arg-type]
    )

    manuscript = pipeline.generate_manuscript(project_id)
    first_segment_video = manuscript.segments[0].images[0]
    second_segment_video = manuscript.segments[1].images[0]
    assert first_segment_video.type == "video"
    assert second_segment_video.type == "video"
    assert first_segment_video.start_from == 1.0
    assert first_segment_video.end_at == 3.0
    assert second_segment_video.start_from == 3.0
    assert second_segment_video.end_at == 5.0


def test_generate_keeps_all_generated_variants_on_segment_when_preferred(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        (
            "{"
            '"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4o"},'
            '"audio":{"tts":"google"},'
            '"prompts":{"scriptPrompt":"Return JSON with lines."},'
            '"imageGeneration":{"enabled":true,"provider":"openai","variants":2,"preferGenerated":true,'
            '"prompts":{"briefPrompt":"brief","openaiPromptBuilder":"builder","nanobananaPromptBuilder":"nano"}},'
            '"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts"}}'
            "}"
        ),
        encoding="utf-8",
    )

    store = ProjectStore(projects_root)
    project_id = "generated-variants"
    store.ensure_layout(project_id)
    for rel_path in [
        "working/generated-images/seg-001-v01.png",
        "working/generated-images/seg-001-v02.png",
        "working/generated-images/seg-002-v01.png",
    ]:
        output_path = store.project_path(project_id) / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"img")

    store.save_json(
        project_id,
        "input/article.json",
        {
            "title": "Generated Variant Demo",
            "byline": "Tester",
            "pubdate": "2026-01-01T00:00:00Z",
            "text": "Long text",
            "script_lines": ["Line 1", "Line 2"],
            "images": [],
            "videos": [],
        },
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://localhost:8001",
    )

    pipeline = PipelineService(
        settings=settings,
        store=store,
        llm_service=LLMService(api_key="", model="gpt-4o-mini"),
        tts_service=GeminiTTSService(
            api_key="",
            voice_id="voice",
            ffprobe_bin="ffprobe",
            ffmpeg_bin="ffmpeg",
        ),
        config_resolver=ConfigResolver(settings.config_root_abs),
        asset_analysis_service=_FakeGeneratedAssetAnalysisService(settings.app_base_url),  # type: ignore[arg-type]
        image_generation_service=_FakeImageGenerationService(settings.app_base_url),  # type: ignore[arg-type]
    )

    manuscript = pipeline.generate_manuscript(project_id)

    assert [image.path for image in manuscript.segments[0].images] == [
        "working/generated-images/seg-001-v01.png",
        "working/generated-images/seg-001-v02.png",
    ]
    assert [image.path for image in manuscript.segments[1].images] == [
        "working/generated-images/seg-002-v01.png",
    ]


def test_generate_routes_llm_nodes_from_runtime_override(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        (
            '{'
            '"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4.1-mini"},'
            '"gemini":{"manuscriptModel":"gemini-2.5-flash","mediaModel":"gemini-2.5-pro","promptBuilderModel":"gemini-2.5-flash"},'
            '"audio":{"tts":"google"},'
            '"prompts":{"scriptPrompt":"Return JSON with lines.","placementPrompt":"place","describeImagesPrompt":"describe"},'
            '"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts"}}'
            '}'
        ),
        encoding="utf-8",
    )

    store = ProjectStore(projects_root)
    project_id = "llm-routing"
    store.ensure_layout(project_id)
    image_path = store.project_path(project_id) / "input" / "images" / "a.jpg"
    image_path.write_bytes(b"jpg")

    store.save_json(
        project_id,
        "input/article.json",
        {
            "title": "Routing Demo",
            "byline": "Tester",
            "pubdate": "2026-01-01T00:00:00Z",
            "text": "Long text",
            "images": [{"path": "images/a.jpg"}],
            "videos": [],
        },
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://localhost:8001",
    )
    fake_llm = _FakeLLMService()
    fake_analysis = _FakeAssetAnalysisService(settings.app_base_url)
    pipeline = PipelineService(
        settings=settings,
        store=store,
        llm_service=fake_llm,  # type: ignore[arg-type]
        tts_service=GeminiTTSService(
            api_key="",
            voice_id="voice",
            ffprobe_bin="ffprobe",
            ffmpeg_bin="ffmpeg",
        ),
        config_resolver=ConfigResolver(settings.config_root_abs),
        asset_analysis_service=fake_analysis,  # type: ignore[arg-type]
    )

    manuscript = pipeline.generate_manuscript(
        project_id,
        llm_override={
            "default_provider": "gemini",
            "nodes": {
                "script_generation": {"provider": "gemini", "model": "gemini-2.5-pro"},
                "image_description": {"provider": "openai", "model": "gpt-4.1-mini"},
                "asset_placement": {"provider": "gemini", "model": "gemini-2.5-flash"},
            },
        },  # type: ignore[arg-type]
    )

    assert len(manuscript.segments) == 2
    assert fake_llm.calls[0]["provider"] == "gemini"
    assert fake_llm.calls[0]["model_override"] == "gemini-2.5-pro"
    assert fake_analysis.last_call is not None
    assert fake_analysis.last_call["describe_provider"] == "openai"
    assert fake_analysis.last_call["describe_model"] == "gpt-4.1-mini"
    assert fake_analysis.last_call["placement_provider"] == "gemini"
    assert fake_analysis.last_call["placement_model"] == "gemini-2.5-flash"
