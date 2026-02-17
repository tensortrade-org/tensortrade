"""
Hyperparameter pack store backed by SQLite.
Stores training configurations (HP packs) for reuse and comparison.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class HyperparameterPack:
    id: str
    name: str
    description: str
    config: dict[str, object]
    created_at: str
    updated_at: str


DEFAULT_DB_DIR = os.path.expanduser("~/.tensortrade")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "experiments.db")


# Seed pack configurations extracted from training scripts
SEED_PACKS: list[dict[str, object]] = [
    {
        "name": "Simple PPO",
        "description": "Basic PPO config from train_trend.py, suitable for quick experiments.",
        "config": {
            "algorithm": "PPO",
            "learning_rate": 5e-5,
            "gamma": 0.99,
            "lambda_": 0.9,
            "clip_param": 0.1,
            "entropy_coeff": 0.1,
            "vf_loss_coeff": 0.5,
            "num_sgd_iter": 5,
            "sgd_minibatch_size": 128,
            "train_batch_size": 4000,
            "num_rollout_workers": 4,
            "rollout_fragment_length": 200,
            "model": {"fcnet_hiddens": [32, 32], "fcnet_activation": "tanh"},
            "action_scheme": "BSH",
            "reward_scheme": "PBR",
            "reward_params": {
                "trade_penalty_multiplier": 1.1,
                "churn_penalty_multiplier": 1.0,
                "churn_window": 6,
                "reward_clip": 200.0,
            },
            "commission": 0.0005,
            "initial_cash": 10000,
            "window_size": 10,
            "max_allowed_loss": 0.3,
            "max_episode_steps": None,
            "num_iterations": 100,
        },
    },
    {
        "name": "Best Known",
        "description": "Best Optuna-tuned config from train_best.py with rich features.",
        "config": {
            "algorithm": "PPO",
            "learning_rate": 3.29e-05,
            "gamma": 0.992,
            "lambda_": 0.9,
            "clip_param": 0.123,
            "entropy_coeff": 0.015,
            "vf_loss_coeff": 0.5,
            "num_sgd_iter": 7,
            "sgd_minibatch_size": 256,
            "train_batch_size": 2000,
            "num_rollout_workers": 4,
            "rollout_fragment_length": 200,
            "model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "tanh"},
            "action_scheme": "BSH",
            "reward_scheme": "PBR",
            "reward_params": {
                "trade_penalty_multiplier": 1.1,
                "churn_penalty_multiplier": 1.0,
                "churn_window": 6,
                "reward_clip": 200.0,
            },
            "commission": 0.003,
            "initial_cash": 10000,
            "window_size": 17,
            "max_allowed_loss": 0.32,
            "max_episode_steps": None,
            "num_iterations": 100,
        },
    },
    {
        "name": "Trend Following",
        "description": "Tiny network with high entropy, minimal trend features to prevent overfitting.",
        "config": {
            "algorithm": "PPO",
            "learning_rate": 5e-5,
            "gamma": 0.99,
            "lambda_": 0.9,
            "clip_param": 0.1,
            "entropy_coeff": 0.1,
            "vf_loss_coeff": 0.5,
            "num_sgd_iter": 5,
            "sgd_minibatch_size": 128,
            "train_batch_size": 4000,
            "num_rollout_workers": 4,
            "rollout_fragment_length": 200,
            "model": {"fcnet_hiddens": [32, 32], "fcnet_activation": "tanh"},
            "action_scheme": "BSH",
            "reward_scheme": "PBR",
            "reward_params": {
                "trade_penalty_multiplier": 1.1,
                "churn_penalty_multiplier": 1.0,
                "churn_window": 6,
                "reward_clip": 200.0,
            },
            "commission": 0.0005,
            "initial_cash": 10000,
            "window_size": 10,
            "max_allowed_loss": 0.3,
            "max_episode_steps": None,
            "num_iterations": 100,
        },
    },
    {
        "name": "Optuna Optimized",
        "description": "Search-space optimized config from Optuna study with TPE sampler.",
        "config": {
            "algorithm": "PPO",
            "learning_rate": 3.29e-05,
            "gamma": 0.992,
            "lambda_": 0.9,
            "clip_param": 0.123,
            "entropy_coeff": 0.015,
            "vf_loss_coeff": 0.5,
            "num_sgd_iter": 7,
            "sgd_minibatch_size": 256,
            "train_batch_size": 2000,
            "num_rollout_workers": 2,
            "rollout_fragment_length": 200,
            "model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "tanh"},
            "action_scheme": "BSH",
            "reward_scheme": "PBR",
            "reward_params": {
                "trade_penalty_multiplier": 1.1,
                "churn_penalty_multiplier": 1.0,
                "churn_window": 6,
                "reward_clip": 200.0,
            },
            "commission": 0.0005,
            "initial_cash": 10000,
            "window_size": 17,
            "max_allowed_loss": 0.32,
            "max_episode_steps": None,
            "num_iterations": 40,
        },
    },
    {
        "name": "Trend PBR",
        "description": (
            "TrendPBR reward with EMA(12,26) trend detection. "
            "Rewards being long in uptrends, penalizes being long in downtrends. "
            "Uses Best Known PPO params with wider network for trend learning."
        ),
        "config": {
            "algorithm": "PPO",
            "learning_rate": 3e-05,
            "gamma": 0.993,
            "lambda_": 0.92,
            "clip_param": 0.15,
            "entropy_coeff": 0.02,
            "vf_loss_coeff": 0.5,
            "num_sgd_iter": 8,
            "sgd_minibatch_size": 256,
            "train_batch_size": 4000,
            "num_rollout_workers": 4,
            "rollout_fragment_length": 200,
            "model": {"fcnet_hiddens": [128, 128, 64], "fcnet_activation": "tanh"},
            "action_scheme": "BSH",
            "reward_scheme": "TrendPBR",
            "reward_params": {
                "trade_penalty_multiplier": 1.2,
                "churn_penalty_multiplier": 1.0,
                "churn_window": 6,
                "reward_clip": 200.0,
            },
            "commission": 0.003,
            "initial_cash": 10000,
            "window_size": 30,
            "max_allowed_loss": 0.35,
            "max_episode_steps": None,
            "num_iterations": 100,
        },
    },
    {
        "name": "Trend PBR Cautious",
        "description": (
            "TrendPBR with zero commission, high entropy (0.10), and strong trend signal "
            "(weight=0.003, scale=40). Achieved positive test PnL (+$47) on a 30-day BTC "
            "downtrend by learning to stay in cash and selectively catch upswings. "
            "Uses compact [64,64] network for generalization."
        ),
        "config": {
            "algorithm": "PPO",
            "learning_rate": 3.29e-05,
            "gamma": 0.992,
            "lambda_": 0.9,
            "clip_param": 0.123,
            "entropy_coeff": 0.10,
            "vf_loss_coeff": 0.5,
            "num_sgd_iter": 7,
            "sgd_minibatch_size": 256,
            "train_batch_size": 2000,
            "num_rollout_workers": 2,
            "rollout_fragment_length": 200,
            "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
            "action_scheme": "BSH",
            "reward_scheme": "TrendPBR",
            "reward_params": {
                "trade_penalty_multiplier": 1.0,
                "churn_penalty_multiplier": 0.75,
                "churn_window": 4,
                "trend_weight": 0.003,
                "trend_scale": 40.0,
            },
            "commission": 0.0,
            "initial_cash": 10000,
            "window_size": 17,
            "max_allowed_loss": 0.32,
            "max_episode_steps": None,
            "num_iterations": 170,
        },
    },
]


class HyperparameterStore:
    """SQLite-backed persistent store for hyperparameter packs."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._seed_defaults()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS hyperparameter_packs (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                config TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def _seed_defaults(self) -> None:
        """Seed default packs if table is empty."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM hyperparameter_packs"
        ).fetchone()[0]
        if count > 0:
            return

        now = datetime.now(UTC).isoformat()
        for seed in SEED_PACKS:
            pack_id = str(uuid.uuid4())
            self._conn.execute(
                """INSERT INTO hyperparameter_packs
                   (id, name, description, config, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    pack_id,
                    seed["name"],
                    seed["description"],
                    json.dumps(seed["config"]),
                    now,
                    now,
                ),
            )
        self._conn.commit()

    def create_pack(
        self,
        name: str,
        description: str = "",
        config: dict[str, object] | None = None,
    ) -> str:
        """Create a new hyperparameter pack. Returns the new pack ID."""
        pack_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT INTO hyperparameter_packs
               (id, name, description, config, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pack_id, name, description, json.dumps(config or {}), now, now),
        )
        self._conn.commit()
        return pack_id

    def get_pack(self, pack_id: str) -> HyperparameterPack | None:
        """Retrieve a single pack by ID."""
        row = self._conn.execute(
            "SELECT * FROM hyperparameter_packs WHERE id = ?", (pack_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_pack(row)

    def get_pack_by_name(self, name: str) -> HyperparameterPack | None:
        """Retrieve a pack by name."""
        row = self._conn.execute(
            "SELECT * FROM hyperparameter_packs WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_pack(row)

    def list_packs(self) -> list[HyperparameterPack]:
        """List all hyperparameter packs ordered by name."""
        rows = self._conn.execute(
            "SELECT * FROM hyperparameter_packs ORDER BY name"
        ).fetchall()
        return [self._row_to_pack(r) for r in rows]

    def update_pack(
        self,
        pack_id: str,
        name: str | None = None,
        description: str | None = None,
        config: dict[str, object] | None = None,
    ) -> bool:
        """Update an existing pack. Returns True if updated, False if not found."""
        existing = self.get_pack(pack_id)
        if existing is None:
            return False

        now = datetime.now(UTC).isoformat()
        new_name = name if name is not None else existing.name
        new_desc = description if description is not None else existing.description
        new_config = config if config is not None else existing.config

        self._conn.execute(
            """UPDATE hyperparameter_packs
               SET name = ?, description = ?, config = ?, updated_at = ?
               WHERE id = ?""",
            (new_name, new_desc, json.dumps(new_config), now, pack_id),
        )
        self._conn.commit()
        return True

    def delete_pack(self, pack_id: str) -> bool:
        """Delete a pack. Returns True if deleted, False if not found."""
        cursor = self._conn.execute(
            "DELETE FROM hyperparameter_packs WHERE id = ?", (pack_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def duplicate_pack(self, pack_id: str, new_name: str | None = None) -> str:
        """Duplicate a pack with a new name. Returns the new pack ID.

        Raises ValueError if source pack not found.
        """
        source = self.get_pack(pack_id)
        if source is None:
            raise ValueError(f"Pack not found: {pack_id}")

        name = new_name or f"{source.name} (copy)"
        # Ensure unique name
        counter = 1
        base_name = name
        while self.get_pack_by_name(name) is not None:
            counter += 1
            name = f"{base_name} ({counter})"

        return self.create_pack(
            name=name,
            description=source.description,
            config=source.config,
        )

    @staticmethod
    def _row_to_pack(row: sqlite3.Row) -> HyperparameterPack:
        return HyperparameterPack(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            config=json.loads(row["config"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
