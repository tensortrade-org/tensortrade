"""
TensorBoard integration for TensorTrade training.

Provides structured logging of trading metrics to TensorBoard
with grouped scalar layouts for Trading, Performance, and Behavior.
"""

from dataclasses import dataclass


@dataclass
class TensorBoardConfig:
    log_dir: str = "~/ray_results/tensortrade"
    flush_secs: int = 30


class TradingTensorBoardLogger:
    """Logs trading-specific metrics to TensorBoard.

    Groups metrics into Trading (pnl, net_worth), Performance
    (episode_return, loss), and Behavior (trade_count, hold_count).
    """

    def __init__(self, config: TensorBoardConfig | None = None) -> None:
        import os

        from torch.utils.tensorboard import SummaryWriter

        self.config = config or TensorBoardConfig()
        log_dir = os.path.expanduser(self.config.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=self.config.flush_secs,
        )
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Set up custom scalar layout for organized dashboards."""
        from torch.utils.tensorboard.summary import custom_scalars

        layout = {
            "Trading": {
                "PnL": ["Multiline", ["Trading/pnl", "Trading/pnl_pct"]],
                "Net Worth": ["Multiline", ["Trading/net_worth"]],
            },
            "Performance": {
                "Episode Return": ["Multiline", ["Performance/episode_return_mean"]],
                "Loss": ["Multiline", ["Performance/total_loss", "Performance/policy_loss", "Performance/vf_loss"]],
            },
            "Behavior": {
                "Actions": ["Multiline", ["Behavior/trade_count", "Behavior/hold_count"]],
            },
        }
        self._writer.file_writer.add_summary(custom_scalars(layout))

    def log_training_result(self, result: dict, iteration: int) -> None:
        """Extract and log metrics from an RLlib training result dict."""
        # Custom trading metrics
        custom = result.get("env_runners", {}).get("custom_metrics", {})

        trading_metrics = {
            "Trading/pnl": custom.get("pnl_mean", 0),
            "Trading/pnl_pct": custom.get("pnl_pct_mean", 0),
            "Trading/net_worth": custom.get("final_net_worth_mean", 0),
        }

        behavior_metrics = {
            "Behavior/trade_count": custom.get("trade_count_mean", 0),
            "Behavior/hold_count": custom.get("hold_count_mean", 0),
        }

        # Standard RLlib performance metrics
        env_runners = result.get("env_runners", {})
        perf_metrics = {
            "Performance/episode_return_mean": env_runners.get("episode_return_mean", 0),
        }

        # Learner losses
        learner = result.get("learners", {}).get("default_policy", {})
        if learner:
            perf_metrics["Performance/total_loss"] = learner.get("total_loss", 0)
            perf_metrics["Performance/policy_loss"] = learner.get("policy_loss", 0)
            perf_metrics["Performance/vf_loss"] = learner.get("vf_loss", 0)

        for tag, value in {**trading_metrics, **perf_metrics, **behavior_metrics}.items():
            if value is not None:
                self._writer.add_scalar(tag, float(value), iteration)

    def log_evaluation(self, metrics: dict, iteration: int, prefix: str = "Eval") -> None:
        """Log evaluation-phase metrics (validation or test)."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(f"{prefix}/{key}", float(value), iteration)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
