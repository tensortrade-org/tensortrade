"""
Persistent experiment store backed by SQLite.

Stores all training runs, iteration metrics, trades, Optuna trials,
and AI insights for the TensorTrade training intelligence platform.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass
class Experiment:
    id: str
    name: str
    script: str
    status: str  # running, completed, failed, pruned
    started_at: str
    completed_at: str | None = None
    config: dict = field(default_factory=dict)
    final_metrics: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class IterationRecord:
    id: int | None
    experiment_id: str
    iteration: int
    metrics: dict
    timestamp: str


@dataclass
class TradeRecord:
    id: int | None
    experiment_id: str
    episode: int
    step: int
    side: str  # buy, sell
    price: float
    size: float
    commission: float


@dataclass
class OptunaTrialRecord:
    id: int | None
    study_name: str
    trial_number: int
    experiment_id: str | None
    params: dict
    value: float | None
    state: str  # complete, pruned, fail
    duration_seconds: float | None


@dataclass
class LeaderboardEntry:
    experiment_id: str
    name: str
    script: str
    rank: int
    metric_name: str
    metric_value: float
    final_metrics: dict
    started_at: str
    tags: list[str]


DEFAULT_DB_DIR = os.path.expanduser("~/.tensortrade")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "experiments.db")


class ExperimentStore:
    """SQLite-backed persistent store for training experiments."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                script TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT NOT NULL,
                completed_at TEXT,
                config TEXT NOT NULL DEFAULT '{}',
                final_metrics TEXT NOT NULL DEFAULT '{}',
                tags TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                metrics TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_iterations_experiment
                ON iterations(experiment_id, iteration);

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                episode INTEGER NOT NULL,
                step INTEGER NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                commission REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_trades_experiment
                ON trades(experiment_id);

            CREATE TABLE IF NOT EXISTS optuna_trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_name TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                experiment_id TEXT,
                params TEXT NOT NULL DEFAULT '{}',
                value REAL,
                state TEXT NOT NULL DEFAULT 'complete',
                duration_seconds REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_optuna_study
                ON optuna_trials(study_name, trial_number);

            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                experiment_ids TEXT NOT NULL DEFAULT '[]',
                analysis_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                findings TEXT NOT NULL DEFAULT '[]',
                suggestions TEXT NOT NULL DEFAULT '[]',
                confidence TEXT NOT NULL DEFAULT 'medium',
                raw_response TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # --- Experiment CRUD ---

    def create_experiment(
        self,
        name: str,
        script: str,
        config: dict | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Create a new experiment and return its ID."""
        exp_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO experiments (id, name, script, status, started_at, config, tags)
               VALUES (?, ?, ?, 'running', ?, ?, ?)""",
            (exp_id, name, script, now, json.dumps(config or {}), json.dumps(tags or [])),
        )
        self._conn.commit()
        return exp_id

    def complete_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
        final_metrics: dict | None = None,
    ) -> None:
        """Mark an experiment as completed/failed/pruned."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE experiments
               SET status = ?, completed_at = ?, final_metrics = ?
               WHERE id = ?""",
            (status, now, json.dumps(final_metrics or {}), experiment_id),
        )
        self._conn.commit()

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Retrieve a single experiment by ID."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experiment(row)

    def list_experiments(
        self,
        script: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Experiment]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params: list[str | int] = []
        if script:
            query += " AND script = ?"
            params.append(script)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    # --- Iteration Logging ---

    def log_iteration(
        self,
        experiment_id: str,
        iteration: int,
        metrics: dict,
    ) -> None:
        """Log metrics for a training iteration."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO iterations (experiment_id, iteration, metrics, timestamp)
               VALUES (?, ?, ?, ?)""",
            (experiment_id, iteration, json.dumps(metrics), now),
        )
        self._conn.commit()

    def get_iterations(
        self, experiment_id: str
    ) -> list[IterationRecord]:
        """Get all iterations for an experiment."""
        rows = self._conn.execute(
            """SELECT * FROM iterations WHERE experiment_id = ?
               ORDER BY iteration""",
            (experiment_id,),
        ).fetchall()
        return [
            IterationRecord(
                id=r["id"],
                experiment_id=r["experiment_id"],
                iteration=r["iteration"],
                metrics=json.loads(r["metrics"]),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # --- Trade Logging ---

    def log_trade(
        self,
        experiment_id: str,
        episode: int,
        step: int,
        side: str,
        price: float,
        size: float,
        commission: float = 0.0,
    ) -> None:
        """Log a single trade."""
        self._conn.execute(
            """INSERT INTO trades (experiment_id, episode, step, side, price, size, commission)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, episode, step, side, price, size, commission),
        )
        self._conn.commit()

    def log_trades_batch(
        self,
        trades: Sequence[TradeRecord],
    ) -> None:
        """Log multiple trades in a single transaction."""
        self._conn.executemany(
            """INSERT INTO trades (experiment_id, episode, step, side, price, size, commission)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (t.experiment_id, t.episode, t.step, t.side, t.price, t.size, t.commission)
                for t in trades
            ],
        )
        self._conn.commit()

    def get_trades(
        self,
        experiment_id: str,
        episode: int | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[TradeRecord]:
        """Get trades for an experiment, optionally filtered by episode."""
        query = "SELECT * FROM trades WHERE experiment_id = ?"
        params: list[str | int] = [experiment_id]
        if episode is not None:
            query += " AND episode = ?"
            params.append(episode)
        query += " ORDER BY episode, step LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [
            TradeRecord(
                id=r["id"],
                experiment_id=r["experiment_id"],
                episode=r["episode"],
                step=r["step"],
                side=r["side"],
                price=r["price"],
                size=r["size"],
                commission=r["commission"],
            )
            for r in rows
        ]

    def get_all_trades(
        self,
        limit: int = 1000,
        offset: int = 0,
        experiment_id: str | None = None,
        side: str | None = None,
    ) -> list[dict]:
        """Get trades across all experiments with experiment metadata."""
        query = """
            SELECT t.*, e.name as experiment_name, e.script
            FROM trades t
            JOIN experiments e ON t.experiment_id = e.id
            WHERE 1=1
        """
        params: list[str | int] = []
        if experiment_id:
            query += " AND t.experiment_id = ?"
            params.append(experiment_id)
        if side:
            query += " AND t.side = ?"
            params.append(side)
        query += " ORDER BY t.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # --- Optuna Trials ---

    def log_optuna_trial(
        self,
        study_name: str,
        trial_number: int,
        params: dict,
        value: float | None = None,
        state: str = "complete",
        duration_seconds: float | None = None,
        experiment_id: str | None = None,
    ) -> None:
        """Log an Optuna trial result."""
        self._conn.execute(
            """INSERT INTO optuna_trials
               (study_name, trial_number, experiment_id, params, value, state, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                study_name,
                trial_number,
                experiment_id,
                json.dumps(params),
                value,
                state,
                duration_seconds,
            ),
        )
        self._conn.commit()

    def get_optuna_trials(
        self, study_name: str
    ) -> list[OptunaTrialRecord]:
        """Get all trials for a study."""
        rows = self._conn.execute(
            """SELECT * FROM optuna_trials WHERE study_name = ?
               ORDER BY trial_number""",
            (study_name,),
        ).fetchall()
        return [
            OptunaTrialRecord(
                id=r["id"],
                study_name=r["study_name"],
                trial_number=r["trial_number"],
                experiment_id=r["experiment_id"],
                params=json.loads(r["params"]),
                value=r["value"],
                state=r["state"],
                duration_seconds=r["duration_seconds"],
            )
            for r in rows
        ]

    def get_optuna_studies(self) -> list[dict]:
        """List all Optuna studies with summary stats."""
        rows = self._conn.execute("""
            SELECT
                study_name,
                COUNT(*) as total_trials,
                SUM(CASE WHEN state = 'complete' THEN 1 ELSE 0 END) as completed_trials,
                SUM(CASE WHEN state = 'pruned' THEN 1 ELSE 0 END) as pruned_trials,
                MAX(value) as best_value,
                MIN(value) as worst_value,
                AVG(value) as avg_value
            FROM optuna_trials
            GROUP BY study_name
            ORDER BY study_name
        """).fetchall()
        return [dict(r) for r in rows]

    def get_study_trial_curves(self, study_name: str) -> list[dict]:
        """Get all trials for a study with their per-iteration training curves.

        Joins optuna_trials → experiments → iterations to return
        per-trial training curves for visualization.
        """
        trials = self.get_optuna_trials(study_name)
        results: list[dict] = []
        for trial in trials:
            trial_data: dict = {
                "trial_number": trial.trial_number,
                "state": trial.state,
                "params": trial.params,
                "value": trial.value,
                "duration_seconds": trial.duration_seconds,
                "iterations": [],
            }
            if trial.experiment_id:
                iterations = self.get_iterations(trial.experiment_id)
                trial_data["iterations"] = [
                    {"iteration": it.iteration, "metrics": it.metrics}
                    for it in iterations
                ]
            results.append(trial_data)
        return results

    # --- Leaderboard ---

    # Metric aliases: when a metric isn't found, try these fallbacks
    _METRIC_ALIASES: dict[str, list[str]] = {
        "pnl": ["pnl", "pnl_mean", "test_pnl", "best_val_pnl", "objective_value"],
        "net_worth": ["net_worth", "net_worth_mean", "final_net_worth"],
        "test_pnl": ["test_pnl", "pnl"],
        "objective_value": ["objective_value", "best_val_pnl", "pnl"],
    }

    def get_leaderboard(
        self,
        metric: str = "pnl",
        script: str | None = None,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> list[LeaderboardEntry]:
        """Get ranked experiments by a metric from final_metrics JSON.

        Tries metric aliases if the exact metric yields no results.
        """
        candidates = self._METRIC_ALIASES.get(metric, [metric])
        for candidate in candidates:
            entries = self._get_leaderboard_for_metric(
                candidate, script, limit, tags
            )
            if entries:
                return entries
        return []

    def _get_leaderboard_for_metric(
        self,
        metric: str,
        script: str | None,
        limit: int,
        tags: list[str] | None,
    ) -> list[LeaderboardEntry]:
        """Query leaderboard for a specific metric key."""
        query = """
            SELECT id, name, script, final_metrics, started_at, tags
            FROM experiments
            WHERE status = 'completed'
              AND json_extract(final_metrics, ?) IS NOT NULL
        """
        params: list[str | int] = [f"$.{metric}"]
        if script:
            query += " AND script = ?"
            params.append(script)
        query += f" ORDER BY json_extract(final_metrics, ?) DESC LIMIT ?"
        params.extend([f"$.{metric}", limit])

        rows = self._conn.execute(query, params).fetchall()
        entries: list[LeaderboardEntry] = []
        for rank, row in enumerate(rows, 1):
            fm = json.loads(row["final_metrics"])
            exp_tags = json.loads(row["tags"])
            if tags and not any(t in exp_tags for t in tags):
                continue
            entries.append(
                LeaderboardEntry(
                    experiment_id=row["id"],
                    name=row["name"],
                    script=row["script"],
                    rank=rank,
                    metric_name=metric,
                    metric_value=fm.get(metric, 0),
                    final_metrics=fm,
                    started_at=row["started_at"],
                    tags=exp_tags,
                )
            )
        return entries

    # --- Comparison ---

    def compare_experiments(
        self, experiment_ids: list[str]
    ) -> list[dict]:
        """Compare multiple experiments side-by-side."""
        placeholders = ",".join("?" for _ in experiment_ids)
        rows = self._conn.execute(
            f"SELECT * FROM experiments WHERE id IN ({placeholders})",
            experiment_ids,
        ).fetchall()
        results = []
        for row in rows:
            exp = self._row_to_experiment(row)
            iterations = self.get_iterations(exp.id)
            results.append({
                "experiment": asdict(exp),
                "iteration_count": len(iterations),
                "iterations": [asdict(it) for it in iterations],
            })
        return results

    # --- Insights ---

    def store_insight(
        self,
        insight_id: str,
        experiment_ids: list[str],
        analysis_type: str,
        summary: str,
        findings: list[str],
        suggestions: list[str],
        confidence: str,
        raw_response: str,
    ) -> None:
        """Store an AI-generated insight."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO insights
               (id, experiment_ids, analysis_type, summary, findings,
                suggestions, confidence, raw_response, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                insight_id,
                json.dumps(experiment_ids),
                analysis_type,
                summary,
                json.dumps(findings),
                json.dumps(suggestions),
                confidence,
                raw_response,
                now,
            ),
        )
        self._conn.commit()

    def get_insight(self, insight_id: str) -> dict | None:
        """Retrieve a stored insight."""
        row = self._conn.execute(
            "SELECT * FROM insights WHERE id = ?", (insight_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "experiment_ids": json.loads(row["experiment_ids"]),
            "analysis_type": row["analysis_type"],
            "summary": row["summary"],
            "findings": json.loads(row["findings"]),
            "suggestions": json.loads(row["suggestions"]),
            "confidence": row["confidence"],
            "raw_response": row["raw_response"],
            "created_at": row["created_at"],
        }

    def get_latest_insight_for_study(self, study_name: str) -> dict | None:
        """Get the most recent insight for a given study."""
        tag = json.dumps([f"study:{study_name}"])
        row = self._conn.execute(
            """SELECT * FROM insights
               WHERE experiment_ids = ? AND analysis_type = 'campaign_analysis'
               ORDER BY created_at DESC LIMIT 1""",
            (tag,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "experiment_ids": json.loads(row["experiment_ids"]),
            "analysis_type": row["analysis_type"],
            "summary": row["summary"],
            "findings": json.loads(row["findings"]),
            "suggestions": json.loads(row["suggestions"]),
            "confidence": row["confidence"],
            "raw_response": row["raw_response"],
            "created_at": row["created_at"],
        }

    def list_insights(self, limit: int = 50) -> list[dict]:
        """List all stored insights."""
        rows = self._conn.execute(
            "SELECT * FROM insights ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "experiment_ids": json.loads(r["experiment_ids"]),
                "analysis_type": r["analysis_type"],
                "summary": r["summary"],
                "findings": json.loads(r["findings"]),
                "suggestions": json.loads(r["suggestions"]),
                "confidence": r["confidence"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # --- Helpers ---

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        return Experiment(
            id=row["id"],
            name=row["name"],
            script=row["script"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            config=json.loads(row["config"]),
            final_metrics=json.loads(row["final_metrics"]),
            tags=json.loads(row["tags"]),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
