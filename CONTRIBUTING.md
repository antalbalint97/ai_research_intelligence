# Contributing to AI Research Intelligence RAG

Thank you for your interest in contributing to this project. This guide explains how to set up the development environment and submit changes.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ai-research-rag
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Copy the environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Start the database:**
   ```bash
   docker compose up db -d
   ```

## Code Style

This project uses:
- **Black** for code formatting (line-length: 99)
- **isort** for import sorting (profile: black)
- **Ruff** for linting
- Type annotations on all public functions
- Docstrings on all public functions and classes

Run formatting and linting:
```bash
black .
isort .
ruff check .
```

## Running Tests

```bash
pytest tests/ -v
```

## Commit Messages

Use clear, descriptive commit messages:
- `feat: add new topic category for ...`
- `fix: handle missing date field in arXiv loader`
- `docs: update architecture diagram`
- `test: add coverage for chunker edge cases`

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass
4. Submit a PR with a clear description

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full technical architecture.
