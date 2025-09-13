"""
Configuration management module for GROWI RAG MCP server.

This module provides YAML-based configuration with environment variable overrides,
validation, and type safety using Pydantic models.

Example usage:
    >>> config_manager = ConfigManager()
    >>> config = config_manager.load_config("config.yaml")
    >>> print(config.growi.base_url)

Environment variable overrides:
    - GROWI_URL: Overrides growi.base_url
    - GROWI_API_TOKEN: Overrides growi.api_token
    - LLM_API_KEY: Overrides llm.api_key
    - LLM_MODEL: Overrides llm.model
    - LLM_MAX_TOKENS: Overrides llm.max_tokens
    - VECTOR_DB_PERSIST_DIR: Overrides vector_db.persist_directory
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


# Constants for validation and defaults
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3000
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

# Logger instance
logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Server configuration section.

    Controls the basic server settings for the MCP interface including
    host binding, port configuration, and debug mode.

    Attributes:
        host (str): The hostname or IP address to bind the server to.
                   Defaults to 'localhost' for security.
        port (int): The port number for the server to listen on.
                   Should be between 1 and 65535. Defaults to 3000.
        debug (bool): Enable debug mode with verbose logging and error details.
                     Defaults to False for production safety.
    """
    host: str = Field(default=DEFAULT_HOST, description="Server bind address")
    port: int = Field(default=DEFAULT_PORT, description="Server port number", ge=1, le=65535)
    debug: bool = Field(default=False, description="Enable debug mode")


class LoggingConfig(BaseModel):
    """Logging configuration section.

    Controls the logging behavior for the application including log level
    and message formatting.

    Attributes:
        level (str): Log level determining which messages are output.
                    Common values: DEBUG, INFO, WARNING, ERROR, CRITICAL.
                    Defaults to 'INFO'.
        format (str): Python logging format string for message formatting.
                     Defaults to a standard format with timestamp, logger name,
                     level, and message.
    """
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format string"
    )


class GrowiConfig(BaseModel):
    """GROWI connection configuration section.

    Configuration for connecting to a GROWI wiki instance including
    the base URL and API authentication token.

    Attributes:
        base_url (str): The base URL of the GROWI instance (required).
                       Should be a valid HTTP/HTTPS URL.
        api_token (str): API token for authenticating with GROWI.
                        Can be empty initially but required for actual usage.
                        Keep this secure and never log it.
    """
    base_url: str = Field(description="Base URL of the GROWI instance")
    api_token: str = Field(default="", description="API token for GROWI authentication")


class VectorDbConfig(BaseModel):
    """Vector database configuration section.

    Configuration for the vector database used for storing and retrieving
    document embeddings in the RAG system.

    Attributes:
        type (str): Type of vector database to use.
                   Supported types include: chromadb, pinecone, weaviate, faiss.
                   Defaults to 'chromadb'.
        persist_directory (str): Directory path for storing vector database files.
                                Should be a valid writable directory path.
                                Defaults to './data/chroma'.
    """
    type: str = Field(default="chromadb", description="Vector database type")
    persist_directory: str = Field(
        default="./data/chroma",
        description="Directory for vector database persistence"
    )


class LlmConfig(BaseModel):
    """Language model configuration section.

    Configuration for the language model provider used for generating
    responses in the RAG system.

    Attributes:
        provider (str): LLM provider name (openai, anthropic, etc.).
                       Defaults to 'openai'.
        model (str): Specific model name to use with the provider.
                    Defaults to 'gpt-3.5-turbo'.
        api_key (str): API key for authenticating with the LLM provider.
                      Keep this secure and never log it.
        max_tokens (int): Maximum number of tokens per response.
                         Should be positive and reasonable for the model.
                         Defaults to 4096.
        temperature (float): Creativity/randomness factor for responses.
                           Should be between 0.0 and 2.0. Defaults to 0.7.
    """
    provider: str = Field(default="openai", description="LLM provider name")
    model: str = Field(default="gpt-3.5-turbo", description="Model name to use")
    api_key: str = Field(default="", description="API key for the LLM provider")
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum tokens per response",
        gt=0
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Response creativity factor (0.0-2.0)",
        ge=0.0,
        le=2.0
    )


class McpConfig(BaseModel):
    """MCP server configuration section.

    Model Context Protocol specific configuration for server identification
    and metadata.

    Attributes:
        name (str): Server name identifier used in MCP protocol.
                   Should be URL-safe. Defaults to 'growi-rag'.
        version (str): Semantic version string for the server.
                      Should follow semver format. Defaults to '0.1.0'.
        description (str): Human-readable description of the server's purpose.
                          Defaults to a description of RAG for GROWI.
    """
    name: str = Field(default="growi-rag", description="MCP server name identifier")
    version: str = Field(default="0.1.0", description="Server version")
    description: str = Field(
        default="Retrieval Augmented Generation for GROWI wiki",
        description="Server description"
    )


class Config(BaseModel):
    """Main configuration container.

    Root configuration object that contains all configuration sections
    for the GROWI RAG MCP server.

    Attributes:
        server (ServerConfig): Server binding and debug configuration
        logging (LoggingConfig): Logging level and format configuration
        growi (GrowiConfig): GROWI instance connection configuration
        vector_db (VectorDbConfig): Vector database configuration
        llm (LlmConfig): Language model provider configuration
        mcp (McpConfig): MCP protocol server metadata

    Example:
        >>> config = Config(growi={"base_url": "https://wiki.example.com"})
        >>> config.server.port = 8080
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    growi: GrowiConfig = GrowiConfig(base_url="")
    vector_db: VectorDbConfig = Field(default_factory=VectorDbConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)


class ConfigManager:
    """Configuration manager for loading and managing configuration.

    Provides methods for loading YAML configuration files with environment
    variable overrides, validation, and error handling.

    Example:
        >>> manager = ConfigManager()
        >>> config = manager.load_config("config.yaml")
        >>> print(config.growi.base_url)
    """

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def load_config(self, config_path: str) -> Config:
        """
        Load configuration from YAML file with environment variable overrides.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Config: Populated configuration object

        Raises:
            FileNotFoundError: If the config file doesn't exist
            Exception: If YAML parsing fails or validation fails
        """
        self.logger.info(f"Loading configuration from: {config_path}")

        # Check if file exists
        config_file = Path(config_path)
        if not config_file.exists():
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please ensure the file exists and is readable."
            )

        try:
            # Load YAML content
            self.logger.debug(f"Reading YAML content from {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            if yaml_data is None:
                self.logger.warning("YAML file is empty, using default configuration")
                yaml_data = {}

            self.logger.debug(f"Loaded YAML sections: {list(yaml_data.keys())}")

        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML format in {config_path}: {e}")
            raise Exception(
                f"Invalid YAML format in {config_path}: {e}\n"
                f"Please check the YAML syntax and ensure proper indentation."
            )
        except Exception as e:
            self.logger.error(f"Error reading config file {config_path}: {e}")
            raise Exception(
                f"Error reading config file {config_path}: {e}\n"
                f"Please check file permissions and accessibility."
            )

        # Validate required sections exist in original YAML
        if 'growi' not in yaml_data:
            raise Exception("Missing required 'growi' section in configuration")
        if 'llm' not in yaml_data:
            raise Exception("Missing required 'llm' section in configuration")

        # Apply environment variable overrides
        self.logger.debug("Applying environment variable overrides")
        yaml_data = self._apply_env_overrides(yaml_data)

        try:
            # Validate and create config object
            self.logger.debug("Validating configuration with Pydantic models")
            config = Config(**yaml_data)
            self.logger.info("Configuration loaded and validated successfully")
            return config
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise Exception(
                f"Configuration validation failed: {e}\n"
                f"Please check your configuration values and ensure they meet the required format."
            )

    def _apply_env_overrides(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to YAML data."""
        # Ensure vector_db section exists if we need to override it
        if 'vector_db' not in yaml_data:
            yaml_data['vector_db'] = {}

        # GROWI overrides (only if section exists)
        if 'growi' in yaml_data:
            if 'GROWI_URL' in os.environ:
                yaml_data['growi']['base_url'] = os.environ['GROWI_URL']
            if 'GROWI_API_TOKEN' in os.environ:
                yaml_data['growi']['api_token'] = os.environ['GROWI_API_TOKEN']

        # LLM overrides (only if section exists)
        if 'llm' in yaml_data:
            if 'LLM_API_KEY' in os.environ:
                yaml_data['llm']['api_key'] = os.environ['LLM_API_KEY']
            if 'LLM_MODEL' in os.environ:
                yaml_data['llm']['model'] = os.environ['LLM_MODEL']
            if 'LLM_MAX_TOKENS' in os.environ:
                try:
                    yaml_data['llm']['max_tokens'] = int(os.environ['LLM_MAX_TOKENS'])
                except ValueError:
                    pass  # Keep original value if conversion fails

        # Vector DB overrides
        if 'VECTOR_DB_PERSIST_DIR' in os.environ:
            yaml_data['vector_db']['persist_directory'] = os.environ['VECTOR_DB_PERSIST_DIR']

        return yaml_data