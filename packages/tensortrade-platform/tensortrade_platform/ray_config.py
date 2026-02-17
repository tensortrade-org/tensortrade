"""Shared Ray runtime configuration helpers.

These helpers let training subprocesses, inference, and live/paper trading
resolve Ray startup settings from the same environment variables.
"""

from __future__ import annotations

import os
from typing import Any


def resolve_ray_address() -> str | None:
    """Resolve the Ray cluster address from environment settings."""
    for env_name in ("TENSORTRADE_RAY_ADDRESS", "RAY_ADDRESS"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def build_ray_init_kwargs(*, default_num_cpus: int | None = None) -> dict[str, Any]:
    """Build consistent ``ray.init`` kwargs for all TensorTrade components.

    If a shared address is configured, connect to that cluster. Otherwise,
    fall back to process-local Ray with optional ``num_cpus`` sizing.
    """
    kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "log_to_driver": False,
    }

    address = resolve_ray_address()
    if address:
        kwargs["address"] = address
    elif default_num_cpus is not None:
        kwargs["num_cpus"] = max(int(default_num_cpus), 1)

    namespace = os.getenv("TENSORTRADE_RAY_NAMESPACE", "").strip()
    if namespace:
        kwargs["namespace"] = namespace

    return kwargs


def using_shared_ray_cluster() -> bool:
    """Whether a shared Ray address has been configured."""
    return resolve_ray_address() is not None
