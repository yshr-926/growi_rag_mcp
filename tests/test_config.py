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
        assert config.growi.base_url == "https://your-growi-instance.com"

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
        assert config.llm.model == "gpt-3.5-turbo"
        assert config.llm.max_tokens == 4096
        assert config.llm.temperature == 0.7

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