"""
Training bridge for connecting training processes to the dashboard.

Runs a background thread with an asyncio event loop that streams
data to the FastAPI server via WebSocket. Provides thread-safe
send() method and training control properties.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class TrainingBridge:
    """Connects a training process to the dashboard server.

    Runs a background thread with an asyncio event loop.
    The training process calls ``send()`` from any thread;
    messages are forwarded to the server via WebSocket.
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:8000/ws/training",
        max_queue_size: int = 1000,
    ) -> None:
        self.server_url = server_url
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=max_queue_size)
        self._should_stop = threading.Event()
        self._is_paused = threading.Event()
        self._connected = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the background WebSocket sender thread."""
        if self._thread and self._thread.is_alive():
            return
        self._should_stop.clear()
        self._is_paused.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="training-bridge")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and disconnect."""
        self._should_stop.set()
        self._queue.put(None)  # sentinel to unblock
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def send(self, message: dict) -> None:
        """Thread-safe: enqueue a message for sending to the server."""
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # Drop oldest message to make room
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                pass

    @property
    def should_stop(self) -> bool:
        """Check if the dashboard has requested training stop."""
        return self._should_stop.is_set()

    def wait_if_paused(self, timeout: float = 1.0) -> bool:
        """Block if training is paused. Returns True if resumed, False if stopped."""
        while self._is_paused.is_set() and not self._should_stop.is_set():
            self._is_paused.wait(timeout)
        return not self._should_stop.is_set()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def _run(self) -> None:
        """Background thread entry point."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_loop())
        except Exception:
            logger.exception("Training bridge error")
        finally:
            self._loop.close()

    async def _ws_loop(self) -> None:
        """Main WebSocket send loop with reconnection."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed; bridge disabled")
            return

        while not self._should_stop.is_set():
            try:
                async with websockets.connect(self.server_url) as ws:
                    self._connected.set()
                    logger.info("Training bridge connected to %s", self.server_url)

                    # Listen for control messages in background
                    recv_task = asyncio.create_task(self._recv_controls(ws))

                    try:
                        while not self._should_stop.is_set():
                            try:
                                msg = await asyncio.get_event_loop().run_in_executor(
                                    None, self._queue.get, True, 0.5
                                )
                            except queue.Empty:
                                continue

                            if msg is None:
                                break

                            await ws.send(json.dumps(msg))

                        # Drain remaining queued messages before closing
                        while not self._queue.empty():
                            try:
                                msg = self._queue.get_nowait()
                                if msg is None:
                                    break
                                await ws.send(json.dumps(msg))
                            except (queue.Empty, Exception):
                                break
                    finally:
                        recv_task.cancel()
                        self._connected.clear()

            except Exception:
                self._connected.clear()
                if not self._should_stop.is_set():
                    logger.debug("Bridge connection failed, retrying in 2s...")
                    await asyncio.sleep(2.0)

    async def _recv_controls(self, ws) -> None:  # type: ignore[no-untyped-def]
        """Listen for control messages from the server (stop/pause/resume)."""
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                cmd = msg.get("command")
                if cmd == "stop":
                    self._should_stop.set()
                elif cmd == "pause":
                    self._is_paused.set()
                elif cmd == "resume":
                    self._is_paused.clear()
        except Exception:
            pass
