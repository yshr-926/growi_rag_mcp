# GROWI RAG MCP Server

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/MCP-Protocol-green.svg)](https://modelcontextprotocol.io)

*Retrieval Augmented Generation MCP server for GROWI wiki integration*

</div>

## 🎯 Overview

**growi-rag-mcp** は、[GROWI](https://growi.org) wiki のコンテンツを利用して RAG（Retrieval-Augmented Generation）検索機能を提供する独立した MCP サーバーです。Claude Desktop などの MCP クライアントから呼び出すことができ、GROWI の知識ベースに対してインテリジェントな検索と要約を実行できます。

### 🔑 Key Features

- **🔍 Vector Search**: GROWI ページのベクトル検索機能
- **📝 RAG Summarization**: 検索結果を基にした AI 要約生成
- **🔒 Security First**: 公開ページのみを処理、Bearer token 認証
- **⚡ High Performance**: 10秒以内のレスポンス目標
- **🐳 Docker Ready**: コンテナ化対応済み
- **🔧 MCP Protocol**: 標準的な MCP プロトコルでの通信

## 🏗️ Architecture

```mermaid
graph TD
    A[MCP Client<br/>Claude Desktop] -->|MCP Protocol| B[growi-rag-mcp Server<br/>Port 3000]

    %% Periodic sync for document updates
    B -->|Periodic Sync<br/>12h intervals| C[GROWI Wiki<br/>API v3]
    C -->|Page Updates| B

    %% Data processing pipeline
    B -->|New/Updated Content| E[Embedding Model<br/>plamo-1b]
    E -->|Vector Embeddings| D[ChromaDB<br/>Vector Storage]

    %% Query processing flow
    A -->|Search Query| B
    B -->|Similarity Search| D
    D -->|Relevant Chunks| B
    B -->|RAG Query + Context| F[LLM Model<br/>gpt-oss-20b]
    F -->|Generated Summary| B
    B -->|Search Results / Summary| A
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **uv** package manager
- **GROWI** instance with API access
- **Docker** (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd growi-rag-mcp
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

3. **Configure the server**
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your GROWI settings
   ```

4. **Set environment variables**
   ```bash
   export GROWI_API_URL="https://your-growi.example.com"
   export GROWI_API_TOKEN="your-bearer-token"
   ```

5. **Run the server**
   ```bash
   uv run python src/main.py
   ```

### Docker Deployment

```bash
# Build the image
docker build -t growi-rag-mcp .

# Run the container
docker run -d \
  -p 3000:3000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e GROWI_API_URL="https://your-growi.example.com" \
  -e GROWI_API_TOKEN="your-bearer-token" \
  growi-rag-mcp
```

## ⚙️ Configuration

### config.yaml Structure

```yaml
growi:
  api_url: "https://your-growi.example.com"
  api_token: "${GROWI_API_TOKEN}"
  page_limit: 1000

models:
  embedding_model_path: "./models/plamo-embedding-1b"
  llm_model_path: "./models/gpt-oss-20b"

vector_db:
  persist_directory: "./data/chroma"
  collection_name: "growi_pages"

mcp:
  host: "0.0.0.0"
  port: 3000

logging:
  level: "INFO"
  format: "json"
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROWI_API_URL` | GROWI instance URL | ✅ |
| `GROWI_API_TOKEN` | Bearer token for API access | ✅ |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | ❌ |
| `MCP_PORT` | MCP server port (default: 3000) | ❌ |

## 🛠️ Available MCP Tools

### 1. `growi_retrieve`

ベクトル検索機能で関連するページチャンクを取得します。

**Parameters:**
- `query` (string): 検索クエリ
- `limit` (int, optional): 結果数の上限 (default: 5)
- `similarity_threshold` (float, optional): 類似度の閾値 (default: 0.7)

**Returns:**
```json
{
  "results": [
    {
      "page_id": "60f1b2a3c4d5e6f7a8b9c0d1",
      "title": "API Documentation",
      "path": "/api/overview",
      "content": "このページではAPIの概要について...",
      "similarity_score": 0.89,
      "chunk_id": "chunk_1"
    }
  ]
}
```

### 2. `growi_rag_search`

検索 + AI要約を実行し、包括的な回答を生成します。

**Parameters:**
- `query` (string): 検索クエリ
- `max_chunks` (int, optional): 使用するチャンク数 (default: 10)

**Returns:**
```json
{
  "summary": "APIに関する質問への回答として...",
  "sources": [
    {
      "page_title": "API Documentation",
      "page_path": "/api/overview",
      "relevance": 0.89
    }
  ],
  "processing_time_ms": 3240
}
```

## 🧪 Development

### Project Structure

```
growi-rag-mcp/
├── src/                    # Source code
│   ├── growi/             # GROWI API integration
│   ├── vector/            # Vector search functionality
│   ├── llm/               # LLM integration
│   ├── mcp/               # MCP server implementation
│   ├── config.py          # Configuration management
│   ├── logging_config.py  # Structured logging
│   ├── health.py          # Health checks
│   └── main.py            # Application entry point
├── tests/                 # Test suite
├── docs/                  # Documentation
│   ├── spec.md           # Technical specification
│   └── tasks.md          # Development tasks
├── .claude/               # Claude Code integration
│   ├── commands/         # Custom commands
│   ├── agents/           # Agent configurations
│   └── flow/             # Development workflow
├── config.yaml           # Configuration file
├── pyproject.toml        # Python project configuration
└── Dockerfile            # Container configuration
```

### Development Workflow

このプロジェクトは **Claude Code** と **TDD (Test-Driven Development)** による開発フローを採用しています。

```bash
# Check current development status
/flow-stats

# Start working on the next task
/flow-next

# Run full milestone
/flow-run --milestone M2
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_config.py::TestConfigManager -v
```

### Code Quality

```bash
# Format code
uv run ruff format src tests

# Lint code
uv run ruff check src tests

# Type checking
uv run mypy src
```

## 📊 Monitoring & Health Checks

The server exposes health check endpoints:

- **GET `/health`**: Basic health status, uptime, and version
- **GET `/ready`**: Readiness check (dependencies loaded)

### Health Response Example

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "version": "0.1.0",
  "dependencies": {
    "growi_api": "connected",
    "vector_db": "ready",
    "models": "loaded"
  },
  "last_sync": "2025-01-15T10:30:00Z",
  "indexed_pages": 856
}
```

## 🔐 Security Considerations

- **Public Pages Only**: サーバーは `grant=1` の公開ページのみを処理
- **Bearer Token**: GROWI API への認証にBearer tokenを使用
- **Local Models**: 外部API依存なし、すべてローカルで実行
- **Data Privacy**: センシティブな情報の自動検出と除外機能

## 📈 Performance

- **Response Time**: 検索+要約で10秒以内を目標
- **Throughput**: 同時接続数に応じた自動スケーリング
- **Memory Usage**: 効率的なベクトル検索とモデルロード
- **Storage**: ChromaDBによる永続化ストレージ

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Follow** TDD methodology: Red → Green → Refactor
4. **Test** your changes: `uv run pytest tests/ -v`
5. **Commit** with conventional format: `feat: add amazing feature (T042)`
6. **Submit** a pull request

### Development Milestones

- **M1**: ✅ Project Setup & Infrastructure
- **M2**: 🔄 GROWI API Integration (Current)
- **M3**: 📋 Vector Embedding System
- **M4**: 🔌 MCP Server Implementation
- **M5**: 🤖 RAG Search Functionality
- **M6**: 🚀 Integration & Testing

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GROWI](https://growi.org) - Wiki platform
- [MCP Protocol](https://modelcontextprotocol.io) - Model Context Protocol
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Transformers](https://huggingface.co/transformers) - ML models

---

<div align="center">

**Built with ❤️ using Claude Code and TDD methodology**

[Documentation](docs/) • [Issues](issues/) • [Discussions](discussions/)

</div>