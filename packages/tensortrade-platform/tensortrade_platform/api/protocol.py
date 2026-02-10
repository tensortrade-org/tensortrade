"""
WebSocket and REST message protocol types.

Defines TypedDicts for all messages exchanged between
the training process, FastAPI server, and dashboard.
"""

from typing import TypedDict


class StepUpdateMessage(TypedDict, total=False):
    type: str  # "step_update"
    step: int
    price: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    net_worth: float
    action: int
    reward: float


class TradeEventMessage(TypedDict, total=False):
    type: str  # "trade"
    step: int
    side: str  # "buy" | "sell"
    price: float
    size: float
    commission: float


class TrainingUpdateMessage(TypedDict, total=False):
    type: str  # "training_update"
    iteration: int
    episode_return_mean: float
    pnl_mean: float
    pnl_pct_mean: float
    net_worth_mean: float
    trade_count_mean: float
    hold_count_mean: float
    buy_count_mean: float
    sell_count_mean: float
    hold_ratio_mean: float
    trade_ratio_mean: float
    pnl_per_trade_mean: float
    buy_sell_imbalance_mean: float


class EpisodeEventMessage(TypedDict, total=False):
    type: str  # "episode_start" | "episode_end"
    episode: int
    initial_net_worth: float
    final_net_worth: float
    pnl: float
    trade_count: int


class ExperimentSummary(TypedDict, total=False):
    id: str
    name: str
    script: str
    status: str
    started_at: str
    completed_at: str | None
    config: dict
    final_metrics: dict
    tags: list[str]


class LeaderboardEntry(TypedDict, total=False):
    experiment_id: str
    name: str
    script: str
    rank: int
    metric_name: str
    metric_value: float
    final_metrics: dict
    started_at: str
    tags: list[str]


class OptunaTrialMessage(TypedDict, total=False):
    type: str  # "optuna_trial"
    study_name: str
    trial_number: int
    params: dict
    value: float | None
    state: str
    duration_seconds: float | None


class AIInsightMessage(TypedDict, total=False):
    id: str
    experiment_ids: list[str]
    analysis_type: str
    summary: str
    findings: list[str]
    suggestions: list[str]
    confidence: str
    created_at: str


class TrainingStatusMessage(TypedDict, total=False):
    type: str  # "status"
    is_training: bool
    is_paused: bool
    experiment_id: str | None
    current_iteration: int


class EpisodeMetricsMessage(TypedDict, total=False):
    type: str  # "episode_metrics"
    episode: int
    reward_total: float
    pnl: float
    pnl_pct: float
    net_worth: float
    trade_count: int
    hold_count: int
    buy_count: int
    sell_count: int
    hold_ratio: float
    trade_ratio: float
    pnl_per_trade: float
    buy_sell_imbalance: float
    action_distribution: dict[str, int]


class TrainingProgressMessage(TypedDict, total=False):
    type: str  # "training_progress"
    experiment_id: str
    iteration: int
    total_iterations: int
    elapsed_seconds: float
    eta_seconds: float | None
