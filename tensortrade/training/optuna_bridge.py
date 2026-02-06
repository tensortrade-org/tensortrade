"""
Bridge between Optuna studies and the experiment store.

Links each Optuna trial to an experiment record and provides
summary/importance analysis utilities.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    from tensortrade.training.experiment_store import ExperimentStore


class OptunaExperimentBridge:
    """Bridges Optuna trials with ExperimentStore records.

    On each trial completion, creates an experiment record and links
    the trial params + result to it. Also provides summary and
    parameter importance analysis.
    """

    def __init__(
        self,
        store: ExperimentStore,
        study_name: str,
        script: str = "train_optuna",
    ) -> None:
        self.store = store
        self.study_name = study_name
        self.script = script
        self._trial_start_times: dict[int, float] = {}
        self._trial_experiment_ids: dict[int, str] = {}

    def on_trial_begin(self, trial: optuna.Trial) -> str:
        """Call at the start of each trial to create an experiment record.

        Returns the experiment_id for use during training.
        """
        self._trial_start_times[trial.number] = time.time()
        exp_id = self.store.create_experiment(
            name=f"{self.study_name}/trial_{trial.number}",
            script=self.script,
            config=dict(trial.params) if trial.params else {},
            tags=["optuna", self.study_name],
        )
        self._trial_experiment_ids[trial.number] = exp_id
        return exp_id

    def on_trial_complete(
        self,
        trial: optuna.Trial,
        value: float | None = None,
    ) -> None:
        """Call when a trial finishes (complete or pruned)."""
        trial_number = trial.number
        duration = time.time() - self._trial_start_times.get(trial_number, time.time())
        exp_id = self._trial_experiment_ids.get(trial_number)

        state_map = {
            optuna.trial.TrialState.COMPLETE: "complete",
            optuna.trial.TrialState.PRUNED: "pruned",
            optuna.trial.TrialState.FAIL: "fail",
        }
        state = state_map.get(trial.state, "complete")

        # Log to optuna_trials table
        self.store.log_optuna_trial(
            study_name=self.study_name,
            trial_number=trial_number,
            params=dict(trial.params),
            value=value if value is not None else trial.value,
            state=state,
            duration_seconds=duration,
            experiment_id=exp_id,
        )

        # Complete the experiment record
        if exp_id:
            exp_status = "completed" if state == "complete" else state
            final_metrics = {"objective_value": value or trial.value}
            final_metrics.update(trial.params)
            self.store.complete_experiment(exp_id, exp_status, final_metrics)

    def make_optuna_callback(self):
        """Return a callback function suitable for study.optimize()."""
        bridge = self

        def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            bridge.on_trial_complete(trial, value=trial.value)

        return callback

    def get_experiment_id(self, trial_number: int) -> str | None:
        """Get the experiment ID for a given trial number."""
        return self._trial_experiment_ids.get(trial_number)

    def get_study_summary(self, study: optuna.Study) -> dict:
        """Get a summary of the study results."""
        trials = study.trials
        complete_trials = [
            t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            t for t in trials if t.state == optuna.trial.TrialState.PRUNED
        ]

        values = [t.value for t in complete_trials if t.value is not None]

        return {
            "study_name": self.study_name,
            "total_trials": len(trials),
            "completed_trials": len(complete_trials),
            "pruned_trials": len(pruned_trials),
            "best_value": study.best_value if complete_trials else None,
            "best_params": study.best_params if complete_trials else {},
            "best_trial_number": study.best_trial.number if complete_trials else None,
            "value_mean": sum(values) / len(values) if values else None,
            "value_std": _std(values) if len(values) > 1 else None,
        }

    @staticmethod
    def get_param_importance(study: optuna.Study) -> dict[str, float]:
        """Get hyperparameter importance scores for the study."""
        try:
            return optuna.importance.get_param_importances(study)
        except Exception:
            return {}

    @staticmethod
    def get_optimization_history(study: optuna.Study) -> list[dict]:
        """Get the optimization history as a list of dicts."""
        history: list[dict] = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": dict(trial.params),
                    "duration": trial.duration.total_seconds() if trial.duration else None,
                })
        return history


def _std(values: list[float]) -> float:
    """Compute standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5
