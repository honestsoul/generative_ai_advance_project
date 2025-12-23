# GenAI Project - Advanced Structure

A production-ready, enterprise-grade template for Generative AI projects with RAG capabilities.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (optional, for local services)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genai-project.git
cd genai-project

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,api]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run with Docker

```bash
cd docker
docker-compose up -d
```

### Run Locally

```bash
./scripts/run_local.sh
```

## 📁 Project Structure

```
genai_project/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
│
├── src/genai_project/        # Main package
│   ├── core/                 # Configuration & utilities
│   ├── providers/            # LLM & embedding clients
│   ├── storage/              # Cache & blob storage
│   ├── prompts/              # Prompt management
│   ├── workflows/            # Pipeline orchestration
│   ├── api/                  # FastAPI endpoints
│   ├── cli/                  # CLI entrypoints
│   └── eval/                 # Evaluation framework
│
├── retrieval/                # RAG pipeline
│   ├── chunking.py           # Document chunking
│   ├── index.py              # Index building
│   ├── vectorstore.py        # Vector stores
│   └── rerank.py             # Reranking
│
├── tests/                    # Test suite
│   ├── test_prompts.py
│   ├── test_retrieval.py
│   └── test_workflows.py
│
├── scripts/                  # Utility scripts
│   ├── run_local.sh
│   └── build_index.py
│
├── docker/                   # Docker configuration
│   ├── Dockerfile
│   └── compose.yaml
│
├── docs/                     # Documentation
│   └── architecture/
│
├── .github/workflows/        # CI/CD
│   └── ci.yml
│
├── notebooks/                # Jupyter notebooks
├── examples/                 # Example code
└── artifacts/                # Local outputs
```

## 🔧 Features

### LLM Providers
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude 3.5)
- AWS Bedrock

### RAG Pipeline
- Multiple chunking strategies
- In-memory and pgvector stores
- Cohere and cross-encoder reranking

### Production Ready
- Structured logging with structlog
- OpenTelemetry tracing
- Redis caching
- Docker deployment
- GitHub Actions CI

## 🧪 Development

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/

# Format code
ruff format .
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Brij Kishore Pandey**

---

*Built with the GenAI Project Structure V2.0 (2026)*
