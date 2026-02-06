"""Tests for the ExperimentStore SQLite backend."""

import os
import tempfile
import pytest

from tensortrade.training.experiment_store import (
    ExperimentStore,
    Experiment,
    IterationRecord,
    TradeRecord,
    OptunaTrialRecord,
    LeaderboardEntry,
)


@pytest.fixture
def store(tmp_path):
    """Create a fresh ExperimentStore with a temporary DB."""
    db_path = str(tmp_path / "test_experiments.db")
    s = ExperimentStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def experiment_id(store):
    """Create a test experiment and return its ID."""
    return store.create_experiment(
        name="test_run",
        script="train_test",
        config={"lr": 0.001, "batch_size": 32},
        tags=["test", "unit"],
    )


class TestExperimentCRUD:
    def test_create_experiment(self, store):
        exp_id = store.create_experiment(
            name="my_run", script="train_advanced", config={"lr": 0.001}
        )
        assert isinstance(exp_id, str)
        assert len(exp_id) == 36  # UUID format

    def test_get_experiment(self, store, experiment_id):
        exp = store.get_experiment(experiment_id)
        assert exp is not None
        assert isinstance(exp, Experiment)
        assert exp.name == "test_run"
        assert exp.script == "train_test"
        assert exp.status == "running"
        assert exp.config == {"lr": 0.001, "batch_size": 32}
        assert exp.tags == ["test", "unit"]

    def test_get_experiment_not_found(self, store):
        result = store.get_experiment("nonexistent-id")
        assert result is None

    def test_complete_experiment(self, store, experiment_id):
        store.complete_experiment(
            experiment_id,
            status="completed",
            final_metrics={"pnl": 500.0, "net_worth": 10500.0},
        )
        exp = store.get_experiment(experiment_id)
        assert exp is not None
        assert exp.status == "completed"
        assert exp.completed_at is not None
        assert exp.final_metrics == {"pnl": 500.0, "net_worth": 10500.0}

    def test_list_experiments(self, store):
        store.create_experiment(name="run1", script="train_a")
        store.create_experiment(name="run2", script="train_b")
        store.create_experiment(name="run3", script="train_a")

        all_exps = store.list_experiments()
        assert len(all_exps) == 3

        filtered = store.list_experiments(script="train_a")
        assert len(filtered) == 2
        assert all(e.script == "train_a" for e in filtered)

    def test_list_experiments_by_status(self, store):
        eid1 = store.create_experiment(name="run1", script="train_a")
        store.create_experiment(name="run2", script="train_a")
        store.complete_experiment(eid1, status="completed")

        running = store.list_experiments(status="running")
        assert len(running) == 1

        completed = store.list_experiments(status="completed")
        assert len(completed) == 1

    def test_list_experiments_limit_offset(self, store):
        for i in range(10):
            store.create_experiment(name=f"run{i}", script="test")

        page1 = store.list_experiments(limit=3, offset=0)
        assert len(page1) == 3

        page2 = store.list_experiments(limit=3, offset=3)
        assert len(page2) == 3

        assert page1[0].name != page2[0].name


class TestIterationLogging:
    def test_log_and_get_iterations(self, store, experiment_id):
        store.log_iteration(experiment_id, 1, {"episode_return": 100.0, "pnl": 50.0})
        store.log_iteration(experiment_id, 2, {"episode_return": 120.0, "pnl": 75.0})
        store.log_iteration(experiment_id, 3, {"episode_return": 90.0, "pnl": 30.0})

        iters = store.get_iterations(experiment_id)
        assert len(iters) == 3
        assert all(isinstance(it, IterationRecord) for it in iters)
        assert iters[0].iteration == 1
        assert iters[0].metrics["pnl"] == 50.0
        assert iters[2].iteration == 3

    def test_get_iterations_empty(self, store, experiment_id):
        iters = store.get_iterations(experiment_id)
        assert iters == []


class TestTradeLogging:
    def test_log_single_trade(self, store, experiment_id):
        store.log_trade(experiment_id, episode=0, step=10, side="buy", price=100.0, size=0.5)
        trades = store.get_trades(experiment_id)
        assert len(trades) == 1
        assert trades[0].side == "buy"
        assert trades[0].price == 100.0

    def test_log_trades_batch(self, store, experiment_id):
        trades = [
            TradeRecord(None, experiment_id, 0, 10, "buy", 100.0, 0.5, 0.1),
            TradeRecord(None, experiment_id, 0, 20, "sell", 110.0, 0.5, 0.1),
            TradeRecord(None, experiment_id, 1, 5, "buy", 95.0, 1.0, 0.2),
        ]
        store.log_trades_batch(trades)

        result = store.get_trades(experiment_id)
        assert len(result) == 3
        assert result[0].side == "buy"
        assert result[1].side == "sell"

    def test_get_trades_by_episode(self, store, experiment_id):
        store.log_trade(experiment_id, episode=0, step=10, side="buy", price=100.0, size=0.5)
        store.log_trade(experiment_id, episode=1, step=5, side="sell", price=110.0, size=0.5)

        ep0 = store.get_trades(experiment_id, episode=0)
        assert len(ep0) == 1
        assert ep0[0].episode == 0

    def test_get_all_trades_with_metadata(self, store, experiment_id):
        store.log_trade(experiment_id, episode=0, step=10, side="buy", price=100.0, size=0.5)
        trades = store.get_all_trades()
        assert len(trades) == 1
        assert trades[0]["experiment_name"] == "test_run"
        assert trades[0]["script"] == "train_test"

    def test_get_all_trades_filter_side(self, store, experiment_id):
        store.log_trade(experiment_id, episode=0, step=10, side="buy", price=100.0, size=0.5)
        store.log_trade(experiment_id, episode=0, step=20, side="sell", price=110.0, size=0.5)

        buys = store.get_all_trades(side="buy")
        assert len(buys) == 1
        assert buys[0]["side"] == "buy"


class TestOptunaTrials:
    def test_log_and_get_trial(self, store):
        store.log_optuna_trial(
            study_name="my_study",
            trial_number=0,
            params={"lr": 0.001, "hidden_size": 64},
            value=0.85,
            state="complete",
            duration_seconds=120.5,
        )
        trials = store.get_optuna_trials("my_study")
        assert len(trials) == 1
        assert isinstance(trials[0], OptunaTrialRecord)
        assert trials[0].params == {"lr": 0.001, "hidden_size": 64}
        assert trials[0].value == 0.85

    def test_get_optuna_studies(self, store):
        store.log_optuna_trial("study_a", 0, {"lr": 0.001}, 0.8, "complete", 60.0)
        store.log_optuna_trial("study_a", 1, {"lr": 0.01}, 0.7, "pruned", 30.0)
        store.log_optuna_trial("study_b", 0, {"lr": 0.005}, 0.9, "complete", 90.0)

        studies = store.get_optuna_studies()
        assert len(studies) == 2
        study_a = next(s for s in studies if s["study_name"] == "study_a")
        assert study_a["total_trials"] == 2
        assert study_a["completed_trials"] == 1
        assert study_a["pruned_trials"] == 1

    def test_optuna_trial_with_experiment_link(self, store, experiment_id):
        store.log_optuna_trial(
            "my_study", 0, {"lr": 0.001}, 0.85, "complete", 120.0,
            experiment_id=experiment_id,
        )
        trials = store.get_optuna_trials("my_study")
        assert trials[0].experiment_id == experiment_id


class TestLeaderboard:
    def test_leaderboard_ranking(self, store):
        for i, pnl in enumerate([100, 500, 250, -50]):
            eid = store.create_experiment(name=f"run{i}", script="test")
            store.complete_experiment(eid, final_metrics={"pnl": pnl, "net_worth": 10000 + pnl})

        board = store.get_leaderboard(metric="pnl")
        assert len(board) == 4
        assert board[0].metric_value == 500
        assert board[0].rank == 1
        assert board[-1].metric_value == -50

    def test_leaderboard_filter_by_script(self, store):
        for script, pnl in [("a", 100), ("b", 200), ("a", 150)]:
            eid = store.create_experiment(name=f"run_{script}", script=script)
            store.complete_experiment(eid, final_metrics={"pnl": pnl})

        board = store.get_leaderboard(metric="pnl", script="a")
        assert len(board) == 2
        assert all(e.script == "a" for e in board)

    def test_leaderboard_different_metrics(self, store):
        eid1 = store.create_experiment(name="high_pnl", script="test")
        store.complete_experiment(eid1, final_metrics={"pnl": 1000, "net_worth": 10500})

        eid2 = store.create_experiment(name="high_nw", script="test")
        store.complete_experiment(eid2, final_metrics={"pnl": 100, "net_worth": 15000})

        pnl_board = store.get_leaderboard(metric="pnl")
        assert pnl_board[0].name == "high_pnl"

        nw_board = store.get_leaderboard(metric="net_worth")
        assert nw_board[0].name == "high_nw"


class TestComparison:
    def test_compare_experiments(self, store):
        ids = []
        for i in range(3):
            eid = store.create_experiment(name=f"run{i}", script="test")
            store.log_iteration(eid, 1, {"pnl": i * 100})
            ids.append(eid)

        comparison = store.compare_experiments(ids)
        assert len(comparison) == 3
        assert all("experiment" in c for c in comparison)
        assert all("iterations" in c for c in comparison)


class TestInsights:
    def test_store_and_get_insight(self, store, experiment_id):
        store.store_insight(
            insight_id="test-insight-1",
            experiment_ids=[experiment_id],
            analysis_type="experiment",
            summary="The experiment showed positive results.",
            findings=["PnL was positive", "Trade count was reasonable"],
            suggestions=["Try higher learning rate", "Increase batch size"],
            confidence="high",
            raw_response="Full response text...",
        )

        insight = store.get_insight("test-insight-1")
        assert insight is not None
        assert insight["summary"] == "The experiment showed positive results."
        assert len(insight["findings"]) == 2
        assert len(insight["suggestions"]) == 2
        assert insight["confidence"] == "high"

    def test_list_insights(self, store, experiment_id):
        for i in range(5):
            store.store_insight(
                insight_id=f"insight-{i}",
                experiment_ids=[experiment_id],
                analysis_type="experiment",
                summary=f"Summary {i}",
                findings=[],
                suggestions=[],
                confidence="medium",
                raw_response="",
            )

        insights = store.list_insights(limit=3)
        assert len(insights) == 3

    def test_get_insight_not_found(self, store):
        assert store.get_insight("nonexistent") is None


class TestDBLifecycle:
    def test_store_creates_db_file(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "test.db")
        s = ExperimentStore(db_path=db_path)
        assert os.path.exists(db_path)
        s.close()

    def test_data_persists_across_connections(self, tmp_path):
        db_path = str(tmp_path / "persist_test.db")
        s1 = ExperimentStore(db_path=db_path)
        eid = s1.create_experiment(name="persistent", script="test")
        s1.close()

        s2 = ExperimentStore(db_path=db_path)
        exp = s2.get_experiment(eid)
        assert exp is not None
        assert exp.name == "persistent"
        s2.close()
