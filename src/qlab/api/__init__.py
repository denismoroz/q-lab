"""Read-only HTTP API over the registry (docs/REGISTRY.md).

Run with: ``uv run uvicorn qlab.api:app --reload``.
"""

from qlab.api.app import app, create_app

__all__ = ["app", "create_app"]
