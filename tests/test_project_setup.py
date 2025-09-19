"""
Test module for project setup verification.
These tests verify that the basic project structure is correctly configured.
"""

import os
import subprocess
from pathlib import Path
import pytest
import toml


class TestProjectStructure:
    """Test basic project structure and configuration."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists in project root."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml file should exist"

    def test_pyproject_toml_has_correct_structure(self):
        """Test that pyproject.toml has required project configuration."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        # This will fail initially since pyproject.toml doesn't exist yet
        assert pyproject_path.exists(), "pyproject.toml must exist"

        config = toml.load(pyproject_path)

        # Check project section
        assert "project" in config, "pyproject.toml must have [project] section"
        project = config["project"]

        assert project["name"] == "growi-rag-mcp", "Project name should be growi-rag-mcp"
        assert "version" in project, "Project must have version"
        assert "requires-python" in project, "Project must specify Python version requirement"
        assert project["requires-python"] == ">=3.11", "Project should require Python 3.11+"

    def test_required_dependencies(self):
        """Test that pyproject.toml includes all required dependencies."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        assert pyproject_path.exists(), "pyproject.toml must exist"
        config = toml.load(pyproject_path)

        dependencies = config["project"]["dependencies"]
        required_deps = [
            "mcp",
            "transformers",
            "torch",
            "chromadb",
            "langchain",
            "pydantic",
            "pyyaml",
            "httpx",
            "uvloop"
        ]

        for dep in required_deps:
            # Check if dependency is in the list (allowing for version specs)
            dep_found = any(dep_name.startswith(dep) for dep_name in dependencies)
            assert dep_found, f"Required dependency '{dep}' not found in dependencies"

    def test_python_version_file_exists(self):
        """Test that .python-version file exists and specifies correct version."""
        project_root = Path(__file__).parent.parent
        python_version_path = project_root / ".python-version"

        assert python_version_path.exists(), ".python-version file should exist"

        with open(python_version_path, 'r') as f:
            version = f.read().strip()

        # Should be 3.11 or higher
        assert version.startswith("3.11") or version.startswith("3.12") or version.startswith("3.13"), \
            "Python version should be 3.11 or higher"

    def test_uv_sync_succeeds(self):
        """Test that uv sync command succeeds with current configuration."""
        project_root = Path(__file__).parent.parent

        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(project_root)

        try:
            # This will fail initially since pyproject.toml doesn't exist
            result = subprocess.run(
                ["uv", "sync", "--no-dev"],
                capture_output=True,
                text=True,
                timeout=120
            )

            assert result.returncode == 0, f"uv sync failed: {result.stderr}"
            assert Path("uv.lock").exists(), "uv.lock file should be created after sync"

        finally:
            os.chdir(original_cwd)


class TestDirectoryStructure:
    """Test that required directory structure exists."""

    def test_src_directory_structure(self):
        """Test that src/ directory and all required subdirectories exist."""
        project_root = Path(__file__).parent.parent
        src_path = project_root / "src"

        assert src_path.exists() and src_path.is_dir(), "src/ directory should exist"

        required_dirs = ["growi", "vector", "llm", "mcp_handlers"]
        for dir_name in required_dirs:
            dir_path = src_path / dir_name
            assert dir_path.exists() and dir_path.is_dir(), \
                f"src/{dir_name}/ directory should exist"

    def test_main_module_exists(self):
        """Test that src/main.py exists and can be imported."""
        project_root = Path(__file__).parent.parent
        main_path = project_root / "src" / "main.py"

        assert main_path.exists(), "src/main.py should exist"

    def test_main_help_command(self):
        """Test that main module can show help when executed."""
        project_root = Path(__file__).parent.parent

        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(project_root)

        try:
            # This will fail initially since main.py doesn't exist
            result = subprocess.run(
                ["uv", "run", "python", "-m", "src.main", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )

            assert result.returncode == 0, f"main module help failed: {result.stderr}"
            assert "help" in result.stdout.lower() or "usage" in result.stdout.lower(), \
                "Help output should contain help or usage information"

        finally:
            os.chdir(original_cwd)

    def test_tests_directory_structure(self):
        """Test that tests directory exists."""
        project_root = Path(__file__).parent.parent
        tests_path = project_root / "tests"

        assert tests_path.exists() and tests_path.is_dir(), "tests/ directory should exist"

    def test_config_yaml_template(self):
        """Test that config.yaml template exists."""
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"

        # For now, just check it can be created - actual content validation will come later
        assert config_path.exists(), "config.yaml template should exist"