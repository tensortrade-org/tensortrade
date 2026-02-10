"""
Composable callback factory for RLlib training.

Dynamically composes a callback class that adds trading-specific
episode metrics (P&L, net worth, trade count).

Important: Ray serialises callback classes to send them to remote
workers.  ``on_episode_start`` and ``on_episode_end`` run inside Ray
*workers* and must be fully picklable.  Driver-side logging
(TensorBoard, experiment store, dashboard bridge) is handled by the
``log_training_iteration`` helper in ``examples.training._common``
which runs explicitly in the training loop on the driver process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensortrade.api.training_bridge import TrainingBridge
    from tensortrade.training.experiment_store import ExperimentStore
    from tensortrade.training.tensorboard import TradingTensorBoardLogger


def make_training_callbacks(
    base_cls: type,
    tb_logger: TradingTensorBoardLogger | None = None,
    experiment_store: ExperimentStore | None = None,
    experiment_id: str | None = None,
    dashboard_bridge: TrainingBridge | None = None,
) -> type:
    """Create a callback class with trading episode metrics.

    Builds a subclass of ``base_cls`` (typically ``DefaultCallbacks``)
    that records P&L, net worth, and trade stats on episode boundaries.
    These metrics are collected by RLlib and reported in training results.

    Driver-side logging (TensorBoard, experiment store, dashboard) is
    intentionally **not** done here to keep the class fully picklable
    for Ray workers.  Use ``log_training_iteration()`` from
    ``_common.py`` in your training loop instead.

    Parameters
    ----------
    base_cls:
        RLlib callback base class (e.g. ``DefaultCallbacks``).
    tb_logger:
        Accepted for API compatibility but unused (logging moved to driver).
    experiment_store:
        Accepted for API compatibility but unused (logging moved to driver).
    experiment_id:
        Accepted for API compatibility but unused (logging moved to driver).
    dashboard_bridge:
        Accepted for API compatibility but unused (logging moved to driver).

    Returns
    -------
    type
        A new callback class with episode-level trading metrics.
    """

    class ComposedCallbacks(base_cls):  # type: ignore[misc]
        """Trading callbacks with episode-level P&L tracking.

        Fully picklable — no references to unpicklable objects.
        Safe to send to Ray remote workers.
        """

        def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
            super().on_episode_start(
                worker=worker,
                base_env=base_env,
                policies=policies,
                episode=episode,
                env_index=env_index,
                **kwargs,
            )
            env = base_env.get_sub_environments()[env_index]
            if hasattr(env, "portfolio"):
                episode.user_data["initial_net_worth"] = float(env.portfolio.net_worth)

        def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
            super().on_episode_end(
                worker=worker,
                base_env=base_env,
                policies=policies,
                episode=episode,
                env_index=env_index,
                **kwargs,
            )
            env = base_env.get_sub_environments()[env_index]
            if hasattr(env, "portfolio"):
                final = float(env.portfolio.net_worth)
                initial = episode.user_data.get("initial_net_worth", 10000)
                pnl = final - initial
                pnl_pct = (pnl / initial) * 100 if initial > 0 else 0.0

                episode.custom_metrics["pnl"] = pnl
                episode.custom_metrics["pnl_pct"] = pnl_pct
                episode.custom_metrics["final_net_worth"] = final
                episode.custom_metrics["initial_net_worth"] = initial

            if hasattr(env, "reward_scheme") and hasattr(env.reward_scheme, "get_stats"):
                stats = env.reward_scheme.get_stats()
                episode.custom_metrics["trade_count"] = stats.get("trade_count", 0)
                episode.custom_metrics["buy_count"] = stats.get("buy_count", 0)
                episode.custom_metrics["sell_count"] = stats.get("sell_count", 0)
                episode.custom_metrics["hold_count"] = stats.get("hold_count", 0)

    ComposedCallbacks.__name__ = f"ComposedCallbacks({base_cls.__name__})"
    ComposedCallbacks.__qualname__ = ComposedCallbacks.__name__
    return ComposedCallbacks
