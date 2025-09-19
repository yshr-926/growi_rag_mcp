import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_dockerfile() -> str:
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at project root"
    return dockerfile_path.read_text(encoding="utf-8")


def load_compose_service() -> dict:
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at project root"

    with compose_path.open(encoding="utf-8") as handle:
        compose_config = yaml.safe_load(handle)

    assert isinstance(compose_config, dict), "docker-compose.yml must parse to a mapping"
    services = compose_config.get("services")
    assert isinstance(services, dict), "docker-compose.yml must define services"
    assert "growi-rag-mcp" in services, "docker-compose.yml must define growi-rag-mcp service"
    return services["growi-rag-mcp"]


def test_dockerfile_uses_python311_base_image():
    dockerfile = read_dockerfile()
    match = re.search(r"^FROM\s+python:(?P<tag>[^\s]+)", dockerfile, re.MULTILINE)
    assert match, "Dockerfile must start from an official python base image tag"

    version_tag = match.group("tag")
    version_match = re.match(r"(?P<major>\d+)\.(?P<minor>\d+)", version_tag)
    assert version_match, "Python base image tag must contain MAJOR.MINOR version"

    major = int(version_match.group("major"))
    minor = int(version_match.group("minor"))
    assert (major, minor) >= (3, 11), "Docker base image must use Python 3.11 or higher"


def test_dockerfile_configures_uv_package_manager():
    dockerfile = read_dockerfile()
    assert (
        "ghcr.io/astral-sh/uv" in dockerfile
        or "COPY --from=ghcr.io/astral-sh/uv" in dockerfile
    ), "Dockerfile must install uv from official distribution image"
    assert re.search(r"RUN\s+uv\s+sync", dockerfile), "Dockerfile must install deps with uv sync"


def test_dockerfile_packages_dependencies_and_models():
    dockerfile = read_dockerfile()
    expected_fragments = [
        "COPY pyproject.toml uv.lock ./",
        "COPY src/",
        "COPY config.yaml",
        "RUN uv sync",
    ]
    for fragment in expected_fragments:
        assert fragment in dockerfile, f"Dockerfile must include '{fragment}' step"

    assert "COPY models" in dockerfile, "Dockerfile must copy local models for offline execution"


def test_docker_compose_defines_service_with_build():
    service = load_compose_service()
    build_config = service.get("build")
    # Handle both simple string format and detailed build object format
    if isinstance(build_config, str):
        assert build_config == ".", "growi-rag-mcp service must build from current context"
    elif isinstance(build_config, dict):
        assert build_config.get("context") == ".", "growi-rag-mcp service must build from current context"
    else:
        assert False, "growi-rag-mcp service must specify build configuration"

    env = service.get("environment", [])
    flattened_env = env if isinstance(env, list) else list(env.values())
    assert any("GROWI_API_TOKEN" in item for item in flattened_env), (
        "docker-compose service must surface GROWI_API_TOKEN environment variable"
    )


def test_docker_compose_exposes_mcp_port():
    service = load_compose_service()
    ports = service.get("ports") or []
    assert ports, "growi-rag-mcp service must expose ports"
    assert any(str(port).startswith("3000:3000") or str(port).endswith(":3000") for port in ports), (
        "docker-compose service must map container port 3000"
    )


def test_docker_compose_mounts_vector_database_volume():
    service = load_compose_service()
    volumes = service.get("volumes") or []
    assert volumes, "growi-rag-mcp service must configure persistent volumes"
    assert any("chroma_db" in str(volume) for volume in volumes), (
        "docker-compose service must mount chroma_db volume for persistence"
    )