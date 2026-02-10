"""Tests for TradingEnv lifecycle: reset, step, render, save, close, _ensure_numpy, truncation."""

from unittest.mock import MagicMock

import numpy as np

import tensortrade.env.default as default
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
from tensortrade.env.generic.environment import TradingEnv
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import BTC, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    n_steps: int = 20,
    commission: float = 0.0,
    max_episode_steps: int | None = None,
    random_start_pct: float = 0.0,
    **env_kwargs,
) -> tuple[TradingEnv, Portfolio]:
    """Create a minimal trading environment for testing."""
    prices = [100.0 + i * 0.5 for i in range(n_steps)]
    price_stream = Stream.source(prices, dtype="float").rename("USD-BTC")

    exchange = Exchange(
        "exchange",
        service=execute_order,
        options=ExchangeOptions(commission=commission),
    )(price_stream)

    cash = Wallet(exchange, 10000.0 * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(prices, dtype="float").rename("close")]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = PBR(price=price_stream, commission=commission)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=1,
        max_allowed_loss=0.99,
        max_episode_steps=max_episode_steps,
        random_start_pct=random_start_pct,
        **env_kwargs,
    )
    return env, portfolio


# ---------------------------------------------------------------------------
# Step return format
# ---------------------------------------------------------------------------


class TestStepReturnFormat:
    """Step should return a 5-tuple per gymnasium convention."""

    def test_step_returns_five_tuple(self):
        env, _ = _make_env()
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset_returns_two_tuple(self):
        env, _ = _make_env()
        result = env.reset()
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# _ensure_numpy
# ---------------------------------------------------------------------------


class TestEnsureNumpy:
    """Tests for TradingEnv._ensure_numpy GPU compatibility method."""

    def test_numpy_array_passthrough(self):
        env, _ = _make_env()
        arr = np.array([1.0, 2.0, 3.0])
        result = env._ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_list_conversion(self):
        env, _ = _make_env()
        result = env._ensure_numpy([1.0, 2.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_scalar_conversion(self):
        env, _ = _make_env()
        result = env._ensure_numpy(42.0)
        assert isinstance(result, np.ndarray)

    def test_pytorch_tensor_mock(self):
        """Mock a PyTorch-like tensor with cpu().numpy()."""
        env, _ = _make_env()
        mock_tensor = MagicMock()
        expected = np.array([1.0, 2.0, 3.0])
        mock_tensor.cpu.return_value.numpy.return_value = expected
        result = env._ensure_numpy(mock_tensor)
        np.testing.assert_array_equal(result, expected)

    def test_tensorflow_tensor_mock(self):
        """Mock a TensorFlow-like tensor with .numpy()."""
        env, _ = _make_env()
        mock_tensor = MagicMock(spec=["numpy"])
        expected = np.array([4.0, 5.0])
        mock_tensor.numpy.return_value = expected
        # Ensure it doesn't have .cpu (would trigger PyTorch path)
        del mock_tensor.cpu
        result = env._ensure_numpy(mock_tensor)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Max episode steps (truncation)
# ---------------------------------------------------------------------------


class TestMaxEpisodeSteps:
    """Tests for max_episode_steps truncation behavior."""

    def test_truncation_when_max_steps_reached(self):
        env, _ = _make_env(n_steps=50, max_episode_steps=5)
        env.reset()

        for i in range(4):
            _, _, terminated, truncated, _ = env.step(0)
            assert not truncated, f"Truncated too early at step {i + 1}"

        # Step 5 should truncate
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated

    def test_no_truncation_when_no_max_steps(self):
        env, _ = _make_env(n_steps=10)
        env.reset()

        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(0)
            assert not truncated
            if terminated:
                break


# ---------------------------------------------------------------------------
# Render / Save / Close
# ---------------------------------------------------------------------------


class TestRenderSaveClose:
    """Tests for render, save, and close delegation to renderer."""

    def test_render_delegates_to_renderer(self):
        env, _ = _make_env()
        env.reset()
        mock_renderer = MagicMock()
        env.renderer = mock_renderer

        env.render()
        mock_renderer.render.assert_called_once_with(env)

    def test_save_delegates_to_renderer(self):
        env, _ = _make_env()
        mock_renderer = MagicMock()
        env.renderer = mock_renderer

        env.save()
        mock_renderer.save.assert_called_once()

    def test_close_delegates_to_renderer(self):
        env, _ = _make_env()
        mock_renderer = MagicMock()
        env.renderer = mock_renderer

        env.close()
        mock_renderer.close.assert_called_once()


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


class TestResetBehavior:
    """Tests for environment reset semantics."""

    def test_reset_generates_new_episode_id(self):
        env, _ = _make_env()
        env.reset()
        ep1 = env.episode_id
        env.reset()
        ep2 = env.episode_id
        assert ep1 != ep2

    def test_reset_resets_clock(self):
        env, _ = _make_env()
        env.reset()
        env.step(0)
        env.step(0)
        assert env.clock.step > 1

        env.reset()
        # Clock should be at step 1 after reset + initial increment
        assert env.clock.step == 1

    def test_observation_is_valid_after_reset(self):
        env, _ = _make_env()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] > 0
        assert not np.any(np.isnan(obs))


# ---------------------------------------------------------------------------
# Components property
# ---------------------------------------------------------------------------


class TestComponents:
    """Tests for env.components property."""

    def test_components_dict_has_all_keys(self):
        env, _ = _make_env()
        comps = env.components
        expected_keys = {
            "action_scheme",
            "reward_scheme",
            "observer",
            "stopper",
            "informer",
            "renderer",
        }
        assert set(comps.keys()) == expected_keys

    def test_components_are_not_none(self):
        env, _ = _make_env()
        for name, comp in env.components.items():
            assert comp is not None, f"Component {name} is None"


# ---------------------------------------------------------------------------
# Full episode run
# ---------------------------------------------------------------------------


class TestFullEpisode:
    """Tests for running a complete episode."""

    def test_episode_terminates(self):
        """An episode with limited data should eventually terminate."""
        env, _ = _make_env(n_steps=15)
        env.reset()

        terminated = False
        truncated = False
        steps = 0
        while not terminated and not truncated:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if steps > 100:
                break

        assert terminated or truncated

    def test_buy_hold_strategy_episode(self):
        """Buy once then hold for the rest of the episode."""
        env, portfolio = _make_env(n_steps=15)
        env.reset()

        env.step(1)  # buy

        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)  # hold
            done = terminated or truncated
            steps += 1
            if steps > 100:
                break

        assert steps > 0

    def test_random_agent_episode(self):
        """Random actions should not crash the environment."""
        env, _ = _make_env(n_steps=30)
        env.reset()

        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            assert isinstance(obs, np.ndarray)
            assert not np.any(np.isnan(obs))
            if steps > 100:
                break

    def test_observation_shape_consistent(self):
        """Observation shape should be consistent across steps."""
        env, _ = _make_env(n_steps=15)
        obs0, _ = env.reset()
        shape = obs0.shape

        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(0)
            assert obs.shape == shape
            if terminated or truncated:
                break
