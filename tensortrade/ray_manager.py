"""Centralized Ray lifecycle manager.

Provides ref-counted Ray init/shutdown so that inference, live trading,
and any other in-process consumers can share a single Ray runtime without
stomping on each other.

Training subprocesses manage their own Ray instance (separate process),
so they are unaffected.

Usage::

    from tensortrade.ray_manager import ray_manager

    ray_manager.acquire("inference")   # starts Ray if not running
    policy = Policy.from_checkpoint(...)
    ...
    ray_manager.release("inference")   # shuts down only when refcount == 0
"""

from __future__ import annotations

import logging
import threading

from tensortrade.ray_config import build_ray_init_kwargs, resolve_ray_address

logger = logging.getLogger(__name__)


class RayManager:
    """Thread-safe, ref-counted Ray lifecycle manager.

    Each consumer identifies itself with a *label* (e.g. ``"inference"``,
    ``"live_trading"``).  Ray is initialised on the first ``acquire`` and
    shut down only when the last consumer calls ``release``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consumers: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, label: str) -> None:
        """Register *label* as an active Ray consumer.

        Starts Ray if it is not already running.
        """
        with self._lock:
            self._consumers[label] = self._consumers.get(label, 0) + 1
            total = sum(self._consumers.values())
            logger.info(
                "Ray acquire(%s) — consumers: %s (total %d)",
                label,
                dict(self._consumers),
                total,
            )
            self._ensure_started()

    def release(self, label: str) -> None:
        """Un-register one reference for *label*.

        Shuts Ray down only when **all** consumers have released.
        """
        with self._lock:
            count = self._consumers.get(label, 0)
            if count <= 1:
                self._consumers.pop(label, None)
            else:
                self._consumers[label] = count - 1

            total = sum(self._consumers.values())
            logger.info(
                "Ray release(%s) — consumers: %s (total %d)",
                label,
                dict(self._consumers),
                total,
            )

            if total == 0:
                self._shutdown()

    def force_shutdown(self) -> None:
        """Unconditionally shut Ray down and clear all consumers.

        Used during server shutdown / cleanup.
        """
        with self._lock:
            self._consumers.clear()
            self._shutdown()

    @property
    def active_consumers(self) -> dict[str, int]:
        """Snapshot of current consumers (for status endpoints)."""
        with self._lock:
            return dict(self._consumers)

    @property
    def is_active(self) -> bool:
        """Whether Ray is currently running with active consumers."""
        with self._lock:
            return sum(self._consumers.values()) > 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Start Ray if not already initialised.  Must hold ``_lock``."""
        import ray

        if not ray.is_initialized():
            init_kwargs = build_ray_init_kwargs()
            address = resolve_ray_address()
            if address:
                logger.info("Connecting to shared Ray runtime at %s", address)
            else:
                logger.info("Starting local Ray runtime")
            ray.init(**init_kwargs)

    @staticmethod
    def _shutdown() -> None:
        """Shut Ray down if it is running."""
        try:
            import ray

            if ray.is_initialized():
                logger.info("Shutting down Ray runtime (no consumers)")
                ray.shutdown()
        except Exception:
            logger.debug("Error during Ray shutdown", exc_info=True)


# Module-level singleton — import this everywhere
ray_manager = RayManager()
