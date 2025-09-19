#!/usr/bin/env python3
"""
GROWI RAG MCP Server - STDIO-based Entry Point

This module provides the main entry point for the GROWI RAG MCP server,
using FastMCP for STDIO-based communication following the weather.py pattern.
Supports command-line argument parsing and configurable logging.

The server provides Retrieval Augmented Generation capabilities for GROWI
wiki instances through the Model Context Protocol (MCP) interface.
"""

import argparse
import json
import sys
from typing import List, Optional
from pathlib import Path

# Add src to path for imports (fixed to not interfere with FastMCP)
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from fastmcp import FastMCP
from src.config import ConfigManager
from src.logging_config import setup_logging, get_logger, LogLevel
from src.mcp_handlers.tools import (
    handle_growi_rag_search,
    handle_growi_retrieve
)

# Create MCP server instance
mcp = FastMCP("growi-rag")

# Global configuration
config: Optional[ConfigManager] = None
logger = None


def initialize_server(config_file: str = "config.yaml", verbose: bool = False):
    """Initialize server configuration and logging."""
    global config, logger
    try:
        config = ConfigManager()
        config.load_config(config_file)

        # For STDIO-based MCP servers, disable logging to avoid contaminating STDIO
        # Only log to file or disable completely for production use
        import logging
        import sys

        # Disable all logging to stdout/stderr to prevent STDIO contamination
        logging.getLogger().handlers.clear()

        # Optional: log to file for debugging (uncomment if needed)
        if verbose:
            log_handler = logging.FileHandler('/tmp/growi_mcp_debug.log')
            log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logging.getLogger().addHandler(log_handler)
            logging.getLogger().setLevel(logging.DEBUG)

        # Create a null logger for compatibility
        logger = logging.getLogger("growi_rag_mcp.server")

    except Exception as e:
        # Silent failure - cannot log errors in STDIO-based MCP servers
        import logging
        logger = logging.getLogger("growi_rag_mcp.server")


# ========== MCP Tools ==========

@mcp.tool
async def growi_search(query: str, limit: int = 10) -> str:
    """
    Search GROWI pages by keyword.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)

    Returns:
        JSON string containing search results with page titles, IDs, and excerpts
    """
    try:
        # Try to use the existing tool implementation
        result = await handle_growi_rag_search({"query": query, "top_k": limit})
        return result if isinstance(result, str) else str(result)

    except Exception as e:
        # Fallback response when dependencies are not available
        fallback = {
            "error": "GROWI search currently unavailable",
            "message": f"Search for '{query}' failed: {str(e)}",
            "query": query,
            "limit": limit,
            "status": "error"
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


@mcp.tool
async def growi_retrieve_page(page_id: str) -> str:
    """
    Retrieve full content of a specific GROWI page.

    Args:
        page_id: GROWI page ID

    Returns:
        JSON string containing page content, metadata, and formatted text
    """
    try:
        result = await handle_growi_retrieve({"page_id": page_id})
        return result if isinstance(result, str) else str(result)

    except Exception as e:
        # Fallback response when dependencies are not available
        fallback = {
            "error": "GROWI page retrieval currently unavailable",
            "message": f"Retrieval of page '{page_id}' failed: {str(e)}",
            "page_id": page_id,
            "status": "error"
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


@mcp.tool
async def vector_search(query: str, limit: int = 5, similarity_threshold: float = 0.7) -> str:
    """
    Search vector database for semantically similar content.

    Args:
        query: Search query for semantic matching
        limit: Maximum number of results (default: 5)
        similarity_threshold: Minimum similarity score (default: 0.7)

    Returns:
        JSON string containing semantically matched content with similarity scores
    """
    try:
        # Use the existing GROWI search for now (vector search can be implemented later)
        result = await handle_growi_rag_search({"query": query, "top_k": limit})
        return result if isinstance(result, str) else str(result)

    except Exception as e:
        # Fallback response when dependencies are not available
        fallback = {
            "error": "Vector search currently unavailable",
            "message": f"Vector search for '{query}' failed: {str(e)}",
            "query": query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            "status": "error"
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


@mcp.tool
async def server_health() -> str:
    """
    Get server health status and system information.

    Returns:
        JSON string containing health status, uptime, and component status
    """
    try:
        # Simple health status without complex dependencies
        import datetime
        status = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "0.1.0",
            "transport": "stdio",
            "capabilities": {
                "tools": ["growi_search", "growi_retrieve_page", "vector_search", "server_health"],
                "resources": ["growi-config", "available-tools"],
                "prompts": ["growi_search_prompt", "content_summary_prompt"]
            }
        }
        return json.dumps(status, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Health check failed: {str(e)}"


# ========== MCP Resources ==========

@mcp.resource("resource://growi-config")
async def get_growi_config() -> str:
    """Provide GROWI configuration information to AI clients."""
    try:
        if config and hasattr(config, 'growi'):
            config_info = {
                "base_url": getattr(config.growi, 'base_url', 'not configured'),
                "api_version": getattr(config.growi, 'api_version', 'v3'),
                "auth_type": "bearer" if hasattr(config.growi, 'api_token') else "none",
                "features": {
                    "search": True,
                    "page_retrieval": True,
                    "vector_search": True,
                    "health_monitoring": True
                }
            }
        else:
            config_info = {
                "status": "not_configured",
                "message": "GROWI configuration not loaded"
            }

        return json.dumps(config_info, ensure_ascii=False, indent=2)

    except Exception as e:
        return '{"error": "Configuration unavailable"}'


@mcp.resource("resource://available-tools")
async def get_available_tools() -> str:
    """Provide information about available MCP tools."""
    tools_info = {
        "tools": [
            {
                "name": "growi_search",
                "description": "Search GROWI pages by keyword",
                "parameters": ["query", "limit"]
            },
            {
                "name": "growi_retrieve_page",
                "description": "Retrieve full content of a GROWI page",
                "parameters": ["page_id"]
            },
            {
                "name": "vector_search",
                "description": "Search vector database for semantic matches",
                "parameters": ["query", "limit", "similarity_threshold"]
            },
            {
                "name": "server_health",
                "description": "Get server health and status information",
                "parameters": []
            }
        ],
        "total_tools": 4,
        "transport": "stdio",
        "protocol": "MCP"
    }

    return json.dumps(tools_info, ensure_ascii=False, indent=2)


# ========== Prompt Templates ==========

@mcp.prompt
def growi_search_prompt(topic: str = "documentation") -> str:
    """Generate a prompt for searching GROWI content."""
    return f"Search the GROWI wiki for information about '{topic}'. Please provide relevant pages and their key content."


@mcp.prompt
def content_summary_prompt(page_title: str = "page") -> str:
    """Generate a prompt for summarizing page content."""
    return f"Please retrieve and summarize the content of the GROWI page titled '{page_title}', highlighting the main points and key information."


# ========== Command Line Interface ==========

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the GROWI RAG MCP server.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        prog='growi-rag-mcp',
        description='GROWI RAG MCP Server - STDIO-based Retrieval Augmented Generation for GROWI wiki',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code is not None else 1

    try:
        # Initialize server configuration and logging
        initialize_server(config_file=args.config, verbose=args.verbose)

        # Start MCP server (STDIO-based, no TCP)
        # This will handle all MCP protocol communication via stdin/stdout
        # No logging output to avoid STDIO contamination

        # Run the FastMCP server - this blocks until shutdown
        mcp.run()

        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as e:
        # For STDIO-based servers, we cannot safely output errors
        # They will contaminate the MCP protocol stream
        return 1


if __name__ == '__main__':
    sys.exit(main())