"""Unit tests for trained-policy inference runner behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from tensortrade.api.inference_runner import InferenceRunner


class _FakeManager:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def broadcast_to_dashboards(self, message: dict) -> None:
        self.messages.append(message)


class _FakeBroker:
    """Mimics env.action_scheme.broker with an empty trades OrderedDict."""
    def __init__(self) -> None:
        from collections import OrderedDict
        self.trades: OrderedDict[str, list[object]] = OrderedDict()


class _FakeEnv:
    def __init__(self, sample_action: int = 2) -> None:
        self.window_size = 1
        self.portfolio = SimpleNamespace(net_worth=10000.0)
        self.action_space = MagicMock()
        self.action_space.sample.return_value = sample_action
        self.action_scheme = SimpleNamespace(broker=_FakeBroker())
        self.seen_actions: list[int] = []

    def reset(self):
        return [0.0], {}

    def step(self, action: int):
        self.seen_actions.append(int(action))
        return [0.0], 0.0, True, False, {}


def _make_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-01T00:00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
            },
        ],
    )


class _FakeBalance:
    def as_float(self) -> float:
        return 0.0


class _FakeWallet:
    def __init__(self) -> None:
        self.balance = _FakeBalance()


def _make_store() -> SimpleNamespace:
    exp = SimpleNamespace(
        config={"dataset_name": "Unit Test Dataset"},
        final_metrics={},
    )
    return SimpleNamespace(get_experiment=lambda _: exp)


def test_run_episode_errors_when_trained_policy_unavailable(monkeypatch):
    store = _make_store()
    manager = _FakeManager()
    runner = InferenceRunner(store=store, manager=manager, dataset_store=SimpleNamespace())

    monkeypatch.setattr(runner, "_load_trained_algo", lambda experiment: (_ for _ in ()).throw(ValueError("missing")))

    asyncio.run(runner.run_episode("exp-1"))

    error_msgs = [m for m in manager.messages if m.get("type") == "inference_status" and m.get("state") == "error"]
    assert len(error_msgs) == 1
    assert error_msgs[0]["error"] == "missing"


def test_run_episode_uses_trained_policy(monkeypatch):
    store = _make_store()
    manager = _FakeManager()
    runner = InferenceRunner(store=store, manager=manager, dataset_store=SimpleNamespace())
    env = _FakeEnv(sample_action=2)

    policy_algo = MagicMock()
    policy_algo.compute_single_action.return_value = 1
    monkeypatch.setattr(runner, "_create_env", lambda config, dataset_id=None, start_date=None, end_date=None: (env, _make_ohlcv(), _FakeWallet()))
    monkeypatch.setattr(runner, "_load_trained_algo", lambda experiment: (policy_algo, False))

    asyncio.run(runner.run_episode("exp-1"))

    assert env.action_space.sample.call_count == 0
    policy_algo.compute_single_action.assert_called_once()
    assert env.seen_actions == [1]
    policy_algo.stop.assert_called_once()
    error_msgs = [m for m in manager.messages if m.get("type") == "inference_status" and m.get("state") == "error"]
    assert error_msgs == []
