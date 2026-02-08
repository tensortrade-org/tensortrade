"""
Dataset configuration store backed by SQLite.
Stores dataset configs for training (source, features, splits).
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class DatasetConfig:
    id: str
    name: str
    description: str
    source_type: str  # csv_upload, crypto_download, synthetic
    source_config: dict[str, object]
    features: list[dict[str, object]]
    split_config: dict[str, object]
    created_at: str
    updated_at: str


DEFAULT_DB_DIR = os.path.expanduser("~/.tensortrade")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "experiments.db")


# Seed dataset configurations
SEED_DATASETS: list[dict[str, object]] = [
    {
        "name": "BTC/USD Hourly (Bitfinex)",
        "description": "Hourly BTC/USD data from Bitfinex with full feature set.",
        "source_type": "crypto_download",
        "source_config": {
            "exchange": "Bitfinex",
            "base": "USD",
            "quote": "BTC",
            "timeframe": "1h",
        },
        "features": [
            {"type": "returns", "periods": [1, 4, 12, 24, 48], "normalize": "tanh"},
            {"type": "rsi", "period": 14, "normalize": True},
            {"type": "sma_trend", "fast": 20, "slow": 50, "normalize": "tanh"},
            {"type": "trend_strength", "fast": 20, "slow": 50},
            {"type": "volatility", "period": 24, "rolling_norm_period": 72},
            {"type": "volume_ratio", "period": 20},
            {"type": "bollinger_position", "period": 20, "std_dev": 2},
        ],
        "split_config": {"train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15},
    },
    {
        "name": "BTC/USD Trend Features",
        "description": "Minimal 5-feature trend-following dataset.",
        "source_type": "crypto_download",
        "source_config": {
            "exchange": "Bitfinex",
            "base": "USD",
            "quote": "BTC",
            "timeframe": "1h",
        },
        "features": [
            {"type": "sma_trend", "fast": 10, "slow": 50, "normalize": "tanh"},
            {"type": "returns", "periods": [24], "normalize": "tanh"},
            {"type": "rsi", "period": 14, "normalize": True},
            {"type": "volatility", "period": 24, "rolling_norm_period": 72},
            {"type": "trend_strength", "fast": 10, "slow": 30},
        ],
        "split_config": {"train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15},
    },
    {
        "name": "Synthetic GBM",
        "description": "Synthetic Geometric Brownian Motion data for testing.",
        "source_type": "synthetic",
        "source_config": {
            "base_price": 50000,
            "base_volume": 1000,
            "num_candles": 5000,
            "timeframe": "1h",
            "volatility": 0.02,
            "drift": 0.0001,
        },
        "features": [
            {"type": "returns", "periods": [1, 4, 12], "normalize": "tanh"},
            {"type": "rsi", "period": 14, "normalize": True},
            {"type": "volatility", "period": 24, "rolling_norm_period": 72},
        ],
        "split_config": {"train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15},
    },
]


class DatasetStore:
    """SQLite-backed persistent store for dataset configurations."""

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
            CREATE TABLE IF NOT EXISTS dataset_configs (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL,
                source_config TEXT NOT NULL DEFAULT '{}',
                features TEXT NOT NULL DEFAULT '[]',
                split_config TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def _seed_defaults(self) -> None:
        """Seed default datasets if table is empty."""
        count = self._conn.execute("SELECT COUNT(*) FROM dataset_configs").fetchone()[0]
        if count > 0:
            return

        now = datetime.now(UTC).isoformat()
        for seed in SEED_DATASETS:
            ds_id = str(uuid.uuid4())
            self._conn.execute(
                """INSERT INTO dataset_configs
                   (id, name, description, source_type, source_config,
                    features, split_config, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ds_id,
                    seed["name"],
                    seed["description"],
                    seed["source_type"],
                    json.dumps(seed["source_config"]),
                    json.dumps(seed["features"]),
                    json.dumps(seed["split_config"]),
                    now,
                    now,
                ),
            )
        self._conn.commit()

    def create_config(
        self,
        name: str,
        description: str = "",
        source_type: str = "csv_upload",
        source_config: dict[str, object] | None = None,
        features: list[dict[str, object]] | None = None,
        split_config: dict[str, object] | None = None,
    ) -> str:
        """Create a new dataset configuration. Returns the new config ID."""
        ds_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        sc = source_config or {}
        feats = features or []
        split = split_config or {"train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15}

        self._conn.execute(
            """INSERT INTO dataset_configs
               (id, name, description, source_type, source_config,
                features, split_config, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ds_id,
                name,
                description,
                source_type,
                json.dumps(sc),
                json.dumps(feats),
                json.dumps(split),
                now,
                now,
            ),
        )
        self._conn.commit()
        return ds_id

    def get_config(self, config_id: str) -> DatasetConfig | None:
        """Retrieve a single dataset config by ID."""
        row = self._conn.execute("SELECT * FROM dataset_configs WHERE id = ?", (config_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_dataset(row)

    def get_config_by_name(self, name: str) -> DatasetConfig | None:
        """Retrieve a dataset config by name."""
        row = self._conn.execute("SELECT * FROM dataset_configs WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return self._row_to_dataset(row)

    def list_configs(self) -> list[DatasetConfig]:
        """List all dataset configurations ordered by name."""
        rows = self._conn.execute("SELECT * FROM dataset_configs ORDER BY name").fetchall()
        return [self._row_to_dataset(r) for r in rows]

    def update_config(
        self,
        config_id: str,
        name: str | None = None,
        description: str | None = None,
        source_type: str | None = None,
        source_config: dict[str, object] | None = None,
        features: list[dict[str, object]] | None = None,
        split_config: dict[str, object] | None = None,
    ) -> bool:
        """Update a dataset config. Returns True if updated, False if not found."""
        existing = self.get_config(config_id)
        if existing is None:
            return False

        now = datetime.now(UTC).isoformat()
        new_name = name if name is not None else existing.name
        new_desc = description if description is not None else existing.description
        new_src_type = source_type if source_type is not None else existing.source_type
        new_src_cfg = source_config if source_config is not None else existing.source_config
        new_feats = features if features is not None else existing.features
        new_split = split_config if split_config is not None else existing.split_config

        self._conn.execute(
            """UPDATE dataset_configs
               SET name = ?, description = ?, source_type = ?,
                   source_config = ?, features = ?, split_config = ?,
                   updated_at = ?
               WHERE id = ?""",
            (
                new_name,
                new_desc,
                new_src_type,
                json.dumps(new_src_cfg),
                json.dumps(new_feats),
                json.dumps(new_split),
                now,
                config_id,
            ),
        )
        self._conn.commit()
        return True

    def delete_config(self, config_id: str) -> bool:
        """Delete a dataset config. Returns True if deleted."""
        cursor = self._conn.execute("DELETE FROM dataset_configs WHERE id = ?", (config_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def preview_dataset(self, config_id: str) -> dict[str, object]:
        """Preview a dataset config with row count, columns, date range, and sample stats.

        Returns a dict with rows, columns, date_range, and sample statistics.
        Raises ValueError if config not found.
        """
        config = self.get_config(config_id)
        if config is None:
            raise ValueError(f"Dataset config not found: {config_id}")

        import numpy as np

        from tensortrade.training.feature_engine import FeatureEngine

        # Generate or load data based on source type
        df = self._load_source_data(config)

        # Compute features
        engine = FeatureEngine()
        df = engine.compute(df, config.features)

        # Build preview
        date_range: dict[str, str | None] = {"start": None, "end": None}
        if "date" in df.columns:
            date_range["start"] = str(df["date"].iloc[0])
            date_range["end"] = str(df["date"].iloc[-1])

        col_stats: dict[str, dict[str, float]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) > 0:
                col_stats[str(col)] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        return {
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": date_range,
            "stats": col_stats,
        }

    @staticmethod
    def _load_source_data(config: DatasetConfig) -> "pd.DataFrame":
        """Load or generate source data based on config source_type."""
        import numpy as np
        import pandas as pd

        if config.source_type == "synthetic":
            sc = config.source_config
            n = int(sc.get("num_candles", 5000))  # type: ignore[arg-type]
            base_price = float(sc.get("base_price", 50000))  # type: ignore[arg-type]
            base_volume = float(sc.get("base_volume", 1000))  # type: ignore[arg-type]
            vol = float(sc.get("volatility", 0.02))  # type: ignore[arg-type]
            drift = float(sc.get("drift", 0.0001))  # type: ignore[arg-type]

            rng = np.random.default_rng(42)
            returns = rng.normal(drift, vol, n)
            prices = base_price * np.exp(np.cumsum(returns))
            df = pd.DataFrame(
                {
                    "open": prices * (1 + rng.normal(0, 0.001, n)),
                    "high": prices * (1 + np.abs(rng.normal(0, vol, n))),
                    "low": prices * (1 - np.abs(rng.normal(0, vol, n))),
                    "close": prices,
                    "volume": base_volume * (1 + rng.normal(0, 0.3, n)),
                }
            )
            return df

        # For csv_upload and crypto_download, return an empty OHLCV frame
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    @staticmethod
    def _row_to_dataset(row: sqlite3.Row) -> DatasetConfig:
        return DatasetConfig(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            source_type=row["source_type"],
            source_config=json.loads(row["source_config"]),
            features=json.loads(row["features"]),
            split_config=json.loads(row["split_config"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # --- Aliases for server.py compatibility ---

    def get_dataset(self, dataset_id: str) -> DatasetConfig | None:
        """Alias for get_config()."""
        return self.get_config(dataset_id)

    def list_datasets(self) -> list[DatasetConfig]:
        """Alias for list_configs()."""
        return self.list_configs()

    def create_dataset(
        self,
        name: str,
        description: str = "",
        source_type: str = "csv_upload",
        source_config: dict[str, object] | None = None,
        features: list[dict[str, object]] | None = None,
        split_config: dict[str, object] | None = None,
    ) -> DatasetConfig:
        """Create a dataset and return the full DatasetConfig object."""
        ds_id = self.create_config(
            name=name,
            description=description,
            source_type=source_type,
            source_config=source_config,
            features=features,
            split_config=split_config,
        )
        config = self.get_config(ds_id)
        assert config is not None
        return config

    def update_dataset(
        self,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None,
        source_type: str | None = None,
        source_config: dict[str, object] | None = None,
        features: list[dict[str, object]] | None = None,
        split_config: dict[str, object] | None = None,
    ) -> DatasetConfig | None:
        """Update a dataset and return the updated DatasetConfig, or None if not found."""
        updated = self.update_config(
            dataset_id,
            name=name,
            description=description,
            source_type=source_type,
            source_config=source_config,
            features=features,
            split_config=split_config,
        )
        if not updated:
            return None
        return self.get_config(dataset_id)

    def delete_dataset(self, dataset_id: str) -> bool:
        """Alias for delete_config()."""
        return self.delete_config(dataset_id)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
