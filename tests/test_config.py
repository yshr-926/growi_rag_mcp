"""
Test suite for configuration management module.

This module tests the YAML-based configuration system including:
- Loading configuration from config.yaml
- Environment variable overrides
- Configuration validation
- Error handling for missing/invalid config
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

# This import will fail initially since config.py doesn't exist yet (RED phase)
from src.config import ConfigManager, Config


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def test_config_manager_can_be_imported(self):
        """Test that ConfigManager class can be imported and instantiated."""
        # This will fail since ConfigManager doesn't exist yet
        config_manager = ConfigManager()
        assert config_manager is not None

    def test_load_config_from_yaml(self):
        """Test loading configuration from config.yaml file."""
        # This will fail since load_config method doesn't exist yet
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # Verify all required sections exist
        assert hasattr(config, 'server')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'growi')
        assert hasattr(config, 'vector_db')
        assert hasattr(config, 'llm')
        assert hasattr(config, 'mcp')

    def test_growi_configuration_loaded_correctly(self):
        """Test that GROWI configuration section is loaded with correct structure."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # Verify GROWI config structure
        assert hasattr(config.growi, 'base_url')
        assert hasattr(config.growi, 'api_token')
        assert config.growi.base_url == "http://localhost:3000"

    def test_vector_db_configuration_loaded_correctly(self):
        """Test that vector database configuration is loaded correctly."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # Verify vector DB config
        assert config.vector_db.type == "chromadb"
        assert config.vector_db.persist_directory == "./data/chroma"

    def test_llm_configuration_loaded_correctly(self):
        """Test that LLM configuration is loaded correctly."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # Verify LLM config
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4"
        assert config.llm.max_tokens == 4096
        assert config.llm.temperature == 0.2

    def test_mcp_configuration_loaded_correctly(self):
        """Test that MCP configuration is loaded correctly."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # Verify MCP config
        assert config.mcp.name == "growi-rag"
        assert config.mcp.version == "0.1.0"
        assert "Retrieval Augmented Generation" in config.mcp.description


class TestEnvironmentVariableOverrides:
    """Test cases for environment variable overrides."""

    @patch.dict(os.environ, {
        'GROWI_URL': 'https://override-growi.com',
        'GROWI_API_TOKEN': 'override-token-123'
    })
    def test_growi_env_vars_override_yaml(self):
        """Test that GROWI environment variables override YAML values."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # Environment variables should override YAML values
        assert config.growi.base_url == "https://override-growi.com"
        assert config.growi.api_token == "override-token-123"

    @patch.dict(os.environ, {
        'LLM_API_KEY': 'override-api-key',
        'LLM_MODEL': 'gpt-4',
        'LLM_MAX_TOKENS': '8192'
    })
    def test_llm_env_vars_override_yaml(self):
        """Test that LLM environment variables override YAML values."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # Environment variables should override YAML values
        assert config.llm.api_key == "override-api-key"
        assert config.llm.model == "gpt-4"
        assert config.llm.max_tokens == 8192

    @patch.dict(os.environ, {
        'VECTOR_DB_PERSIST_DIR': '/custom/vector/path'
    })
    def test_vector_db_env_vars_override_yaml(self):
        """Test that vector DB environment variables override YAML values."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        assert config.vector_db.persist_directory == "/custom/vector/path"


class TestConfigValidation:
    """Test cases for configuration validation and error handling."""

    def test_missing_config_file_raises_error(self):
        """Test that missing config.yaml file raises appropriate error."""
        config_manager = ConfigManager()

        with pytest.raises(FileNotFoundError):
            config_manager.load_config("nonexistent-config.yaml")

    def test_invalid_yaml_format_raises_error(self):
        """Test that invalid YAML format raises appropriate error."""
        invalid_yaml_content = """
        growi:
          base_url: "https://example.com"
          invalid_yaml: [unclosed bracket
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml_content)
            f.flush()

            config_manager = ConfigManager()
            with pytest.raises(Exception):  # Should be a YAML parsing error
                config_manager.load_config(f.name)

            # Cleanup
            os.unlink(f.name)

    def test_missing_required_growi_section_raises_error(self):
        """Test that missing required GROWI section raises validation error."""
        incomplete_yaml = """
        server:
          host: "localhost"
          port: 3000
        # Missing growi section
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(incomplete_yaml)
            f.flush()

            config_manager = ConfigManager()
            with pytest.raises(Exception):  # Should be a validation error
                config_manager.load_config(f.name)

            # Cleanup
            os.unlink(f.name)

    def test_missing_required_llm_section_raises_error(self):
        """Test that missing required LLM section raises validation error."""
        incomplete_yaml = """
        growi:
          base_url: "https://example.com"
          api_token: "token"
        # Missing llm section
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(incomplete_yaml)
            f.flush()

            config_manager = ConfigManager()
            with pytest.raises(Exception):  # Should be a validation error
                config_manager.load_config(f.name)

            # Cleanup
            os.unlink(f.name)


class TestConfigClass:
    """Test cases for Config data class."""

    def test_config_class_can_be_instantiated(self):
        """Test that Config class can be instantiated."""
        # This will fail since Config class doesn't exist yet
        config = Config()
        assert config is not None

    def test_config_has_required_attributes(self):
        """Test that Config class has all required attributes."""
        config = Config()

        # Verify all required attributes exist
        required_attrs = ['server', 'logging', 'growi', 'vector_db', 'llm', 'mcp']
        for attr in required_attrs:
            assert hasattr(config, attr), f"Config missing required attribute: {attr}"


class TestConfigYamlHierarchicalStructure:
    """Test cases for T024 - hierarchical config.yaml structure requirements."""

    def test_config_yaml_has_required_hierarchical_sections(self):
        """Test that config.yaml has the required hierarchical sections."""
        import yaml
        from pathlib import Path

        config_path = Path("config.yaml")
        assert config_path.exists(), "config.yaml file must exist"

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # T024 requirement: hierarchical structure with growi, mcp, models sections
        required_sections = ['growi', 'mcp', 'models']
        for section in required_sections:
            assert section in config_data, f"config.yaml missing required section: {section}"

        # Verify models section has embedding and summarizer subsections
        assert 'embedding' in config_data['models'], "models section missing 'embedding' subsection"
        assert 'summarizer' in config_data['models'], "models section missing 'summarizer' subsection"

    def test_config_manager_loads_hierarchical_structure(self):
        """Test that ConfigManager can load and parse hierarchical config.yaml."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # T024 requirement: all configuration values are parsed and accessible
        assert hasattr(config, 'growi'), "Config missing growi section"
        assert hasattr(config, 'mcp'), "Config missing mcp section"
        assert hasattr(config, 'models'), "Config missing models section"

        # Verify nested access works
        assert hasattr(config.growi, 'api_url'), "growi section missing api_url"
        assert hasattr(config.growi, 'api_token'), "growi section missing api_token"
        assert hasattr(config.models, 'embedding'), "models section missing embedding"
        assert hasattr(config.models, 'summarizer'), "models section missing summarizer"

    @patch.dict(os.environ, {'GROWI_API_TOKEN': 'env-override-token'})
    def test_growi_api_token_environment_override(self):
        """Test that GROWI_API_TOKEN environment variable overrides yaml token."""
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")

        # T024 requirement: environment variable value overrides yaml token
        assert config.growi.api_token == 'env-override-token', \
            "Environment variable GROWI_API_TOKEN should override yaml value"

    def test_config_missing_file_raises_filenotfound(self):
        """Test that missing config.yaml raises FileNotFoundError."""
        config_manager = ConfigManager()

        with pytest.raises(FileNotFoundError):
            config_manager.load_config("nonexistent-config.yaml")