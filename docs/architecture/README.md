# Architecture Documentation

This directory contains Architecture Decision Records (ADRs) and system documentation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         GenAI Project                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │    API      │    │    CLI      │    │   Workers   │        │
│  │  (FastAPI)  │    │  (argparse) │    │  (Optional) │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         └──────────────────┼──────────────────┘                │
│                            │                                    │
│  ┌─────────────────────────┴─────────────────────────┐         │
│  │                    Workflows                       │         │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐           │         │
│  │  │ Chains  │  │ Graphs  │  │  Tools  │           │         │
│  │  └────┬────┘  └────┬────┘  └────┬────┘           │         │
│  └───────┼────────────┼────────────┼─────────────────┘         │
│          │            │            │                            │
│  ┌───────┴────────────┴────────────┴─────────────────┐         │
│  │                   Providers                        │         │
│  │  ┌─────────┐  ┌─────────────┐  ┌─────────────┐   │         │
│  │  │   LLM   │  │  Embeddings │  │  Retrieval  │   │         │
│  │  └────┬────┘  └──────┬──────┘  └──────┬──────┘   │         │
│  └───────┼──────────────┼────────────────┼───────────┘         │
│          │              │                │                      │
│  ┌───────┴──────────────┴────────────────┴───────────┐         │
│  │                    Storage                         │         │
│  │  ┌─────────┐  ┌─────────────┐  ┌─────────────┐   │         │
│  │  │  Cache  │  │    Blob     │  │  VectorDB   │   │         │
│  │  │ (Redis) │  │  (S3/GCS)   │  │  (pgvector) │   │         │
│  │  └─────────┘  └─────────────┘  └─────────────┘   │         │
│  └───────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## ADR Template

When making architectural decisions, create a new ADR using this template:

```markdown
# ADR-XXX: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```

## Key Decisions

- **ADR-001**: Use Pydantic Settings for configuration
- **ADR-002**: Structured logging with structlog
- **ADR-003**: Abstract base classes for providers
- **ADR-004**: Jinja2 for prompt templates
- **ADR-005**: pgvector for production vector storage
