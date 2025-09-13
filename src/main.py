#!/usr/bin/env python3
"""
GROWI RAG MCP Server - Main Entry Point

This module provides the main entry point for the GROWI RAG MCP server,
handling command-line argument parsing and basic application setup.

The server provides Retrieval Augmented Generation capabilities for GROWI
wiki instances through the Model Context Protocol (MCP) interface.
"""

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the GROWI RAG MCP server.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        prog='growi-rag-mcp',
        description='GROWI RAG MCP Server - Retrieval Augmented Generation for GROWI wiki',
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

    # Display application information
    print(f"GROWI RAG MCP Server v0.1.0")
    print(f"Config file: {args.config}")

    if args.verbose:
        print("Verbose mode enabled")
        print(f"Python version: {sys.version}")

    # TODO: Initialize and start the MCP server
    # This will be implemented in future iterations

    return 0


if __name__ == '__main__':
    sys.exit(main())