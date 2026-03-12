from pathlib import Path

from fastapi.testclient import TestClient

from api.factory import create_app
from api.settings import Settings


def write_brand_config(config_root: Path) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "default.json").write_text(
        '{"openai":{"manuscriptModel":"gpt-4o-mini","mediaModel":"gpt-4o"},"audio":{"tts":"google"},"options":{"segmentPauseSeconds":0.4},"prompts":{"scriptPrompt":"Return JSON with lines."},"people":{"default":{"voice":"Kore","model_id":"gemini-2.5-pro-preview-tts"}}}',
        encoding="utf-8",
    )


def test_api_lists_projects_and_generates(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    project_id = "demo"
    (projects_root / project_id / "input" / "images").mkdir(parents=True)
    (projects_root / project_id / "input" / "images" / "img.jpg").write_bytes(b"x")

    article = {
        "title": "Demo",
        "byline": "Test",
        "pubdate": "2026-01-01T00:00:00Z",
        "text": "Body",
        "script_lines": ["A", "B", "C"],
        "images": [{"path": "images/img.jpg"}],
        "videos": [],
    }

    article_path = projects_root / project_id / "input" / "article.json"
    article_path.write_text(__import__("json").dumps(article), encoding="utf-8")

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://testserver",
        openai_api_key="",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        projects = client.get("/api/projects")
        assert projects.status_code == 200
        assert projects.json()["projects"] == [project_id]

        generated = client.post(f"/api/projects/{project_id}/generate", json={})
        assert generated.status_code == 200
        payload = generated.json()
        assert payload["status"] == "generated"
        assert payload["project_id"] == project_id
        assert payload["manuscript_json"] is not None


def test_api_upload_image_and_reject_invalid_project_id(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    project_id = "demo"
    (projects_root / project_id / "input" / "images").mkdir(parents=True, exist_ok=True)
    (projects_root / project_id / "input" / "article.json").write_text(
        __import__("json").dumps(
            {
                "title": "Demo",
                "byline": "Test",
                "pubdate": "2026-01-01T00:00:00Z",
                "text": "Body",
                "script_lines": ["A"],
                "images": [],
                "videos": [],
            }
        ),
        encoding="utf-8",
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://testserver",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        upload_ok = client.post(
            f"/api/projects/{project_id}/upload-image",
            files={"file": ("test.jpg", b"abc", "image/jpeg")},
        )
        assert upload_ok.status_code == 200
        body = upload_ok.json()
        assert body["path"].startswith("working/uploads/")
        assert body["url"].startswith("http://testserver/projects/demo/files/")

        upload_invalid = client.post(
            "/api/projects/bad%20id/upload-image",
            files={"file": ("test.jpg", b"abc", "image/jpeg")},
        )
        assert upload_invalid.status_code == 400


def test_api_generate_returns_400_when_openai_key_missing(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    project_id = "missing-openai"
    (projects_root / project_id / "input" / "images").mkdir(parents=True)
    (projects_root / project_id / "input" / "images" / "img.jpg").write_bytes(b"x")

    article = {
        "title": "Demo",
        "byline": "Test",
        "pubdate": "2026-01-01T00:00:00Z",
        "text": "Body",
        "images": [{"path": "images/img.jpg"}],
        "videos": [],
    }

    (projects_root / project_id / "input" / "article.json").write_text(
        __import__("json").dumps(article),
        encoding="utf-8",
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://testserver",
        openai_api_key="",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        generated = client.post(f"/api/projects/{project_id}/generate", json={})
        assert generated.status_code == 400
        assert "OPENAI_API_KEY" in generated.json()["detail"]


def test_api_process_returns_400_when_gemini_key_missing(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    project_id = "missing-gemini-tts"
    (projects_root / project_id / "input" / "images").mkdir(parents=True)
    (projects_root / project_id / "input" / "images" / "img.jpg").write_bytes(b"x")

    article = {
        "title": "Demo",
        "byline": "Test",
        "pubdate": "2026-01-01T00:00:00Z",
        "text": "Body",
        "script_lines": ["A", "B"],
        "images": [{"path": "images/img.jpg"}],
        "videos": [],
    }

    (projects_root / project_id / "input" / "article.json").write_text(
        __import__("json").dumps(article),
        encoding="utf-8",
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://testserver",
        openai_api_key="",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        generated = client.post(f"/api/projects/{project_id}/generate", json={})
        assert generated.status_code == 200
        manuscript_json = generated.json()["manuscript_json"]

        processed = client.post(
            f"/api/projects/{project_id}/process",
            json={"manuscript": manuscript_json},
        )
        assert processed.status_code == 400
        assert "GEMINI_API_KEY or GOOGLE_API_KEY" in processed.json()["detail"]


def test_api_generate_returns_400_when_gemini_script_provider_selected_without_key(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    project_id = "missing-gemini-script"
    (projects_root / project_id / "input" / "images").mkdir(parents=True)
    (projects_root / project_id / "input" / "images" / "img.jpg").write_bytes(b"x")

    article = {
        "title": "Demo",
        "byline": "Test",
        "pubdate": "2026-01-01T00:00:00Z",
        "text": "Body",
        "images": [{"path": "images/img.jpg"}],
        "videos": [],
    }

    (projects_root / project_id / "input" / "article.json").write_text(
        __import__("json").dumps(article),
        encoding="utf-8",
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://testserver",
        openai_api_key="",
        google_api_key="",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        generated = client.post(
            f"/api/projects/{project_id}/generate",
            json={"llm": {"default_provider": "gemini"}},
        )
        assert generated.status_code == 400
        assert "GEMINI_API_KEY or GOOGLE_API_KEY" in generated.json()["detail"]


def test_api_generate_returns_400_when_openai_image_provider_selected_without_key(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    project_id = "missing-openai-images"
    (projects_root / project_id / "input").mkdir(parents=True)

    article = {
        "title": "Demo",
        "byline": "Test",
        "pubdate": "2026-01-01T00:00:00Z",
        "text": "Body",
        "script_lines": ["A single story beat"],
        "images": [],
        "videos": [],
    }

    (projects_root / project_id / "input" / "article.json").write_text(
        __import__("json").dumps(article),
        encoding="utf-8",
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://testserver",
        openai_api_key="",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        generated = client.post(
            f"/api/projects/{project_id}/generate",
            json={"image_generation": {"enabled": True, "provider": "openai"}},
        )
        assert generated.status_code == 400
        assert "provider 'openai'" in generated.json()["detail"]


def test_api_generate_returns_400_when_nanobanana_provider_selected_without_key(tmp_path: Path):
    projects_root = tmp_path / "projects"
    config_root = tmp_path / "brands"
    write_brand_config(config_root)

    project_id = "missing-nanobanana"
    (projects_root / project_id / "input").mkdir(parents=True)

    article = {
        "title": "Demo",
        "byline": "Test",
        "pubdate": "2026-01-01T00:00:00Z",
        "text": "Body",
        "script_lines": ["A single story beat"],
        "images": [],
        "videos": [],
    }

    (projects_root / project_id / "input" / "article.json").write_text(
        __import__("json").dumps(article),
        encoding="utf-8",
    )

    settings = Settings(
        projects_root=projects_root,
        config_root=config_root,
        app_base_url="http://testserver",
        openai_api_key="",
        google_api_key="",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        generated = client.post(
            f"/api/projects/{project_id}/generate",
            json={"image_generation": {"enabled": True, "provider": "nanobanana"}},
        )
        assert generated.status_code == 400
        assert "GEMINI_API_KEY or GOOGLE_API_KEY" in generated.json()["detail"]
