"""
AI Insights Engine powered by Anthropic Claude API.

Analyzes training results, compares experiments, suggests
hyperparameter strategies, and provides trade pattern analysis.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensortrade.training.experiment_store import ExperimentStore


@dataclass
class InsightReport:
    id: str
    experiment_ids: list[str]
    analysis_type: str  # "experiment" | "comparison" | "strategy" | "trades"
    summary: str
    findings: list[str]
    suggestions: list[str]
    confidence: str  # "high" | "medium" | "low"
    raw_response: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class InsightsEngine:
    """Uses Anthropic Claude API to analyze training results."""

    def __init__(
        self,
        store: ExperimentStore,
        api_key: str | None = None,
    ) -> None:
        import anthropic

        self.store = store
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    async def analyze_experiment(
        self, experiment_id: str, *, custom_prompt: str | None = None
    ) -> InsightReport:
        """Analyze a single experiment's performance."""
        exp = self.store.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        iterations = self.store.get_iterations(experiment_id)
        trades = self.store.get_trades(experiment_id, limit=500)

        prompt = self._build_experiment_prompt(exp, iterations, trades)
        return await self._query_claude(
            prompt=prompt,
            experiment_ids=[experiment_id],
            analysis_type="experiment",
            custom_prompt=custom_prompt,
        )

    async def compare_experiments(
        self, experiment_ids: list[str], *, custom_prompt: str | None = None
    ) -> InsightReport:
        """Compare multiple experiments and identify what works."""
        experiments = []
        for eid in experiment_ids:
            exp = self.store.get_experiment(eid)
            if exp:
                iters = self.store.get_iterations(eid)
                experiments.append({"experiment": exp, "iterations": iters})

        if len(experiments) < 2:
            raise ValueError("Need at least 2 experiments to compare")

        prompt = self._build_comparison_prompt(experiments)
        return await self._query_claude(
            prompt=prompt,
            experiment_ids=experiment_ids,
            analysis_type="comparison",
            custom_prompt=custom_prompt,
        )

    async def suggest_next_strategy(
        self, study_name: str, *, custom_prompt: str | None = None
    ) -> InsightReport:
        """Analyze Optuna study and suggest next hyperparameter regions."""
        trials = self.store.get_optuna_trials(study_name)
        if not trials:
            raise ValueError(f"No trials found for study {study_name}")

        prompt = self._build_strategy_prompt(study_name, trials)
        return await self._query_claude(
            prompt=prompt,
            experiment_ids=[],
            analysis_type="strategy",
            custom_prompt=custom_prompt,
        )

    async def analyze_trades(
        self, experiment_id: str, *, custom_prompt: str | None = None
    ) -> InsightReport:
        """Deep analysis of trade patterns."""
        exp = self.store.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        trades = self.store.get_trades(experiment_id, limit=1000)
        if not trades:
            raise ValueError(f"No trades found for experiment {experiment_id}")

        prompt = self._build_trades_prompt(exp, trades)
        return await self._query_claude(
            prompt=prompt,
            experiment_ids=[experiment_id],
            analysis_type="trades",
            custom_prompt=custom_prompt,
        )

    def _build_experiment_prompt(self, exp, iterations, trades) -> str:
        iter_summary = []
        for it in iterations[-20:]:  # Last 20 iterations
            iter_summary.append(
                f"  Iter {it.iteration}: {json.dumps(it.metrics)}"
            )

        trade_summary = {"total": len(trades), "buys": 0, "sells": 0}
        for t in trades:
            if t.side == "buy":
                trade_summary["buys"] += 1
            else:
                trade_summary["sells"] += 1

        return f"""Analyze this reinforcement learning trading experiment:

Experiment: {exp.name} (script: {exp.script})
Status: {exp.status}
Config: {json.dumps(exp.config, indent=2)}
Final Metrics: {json.dumps(exp.final_metrics, indent=2)}

Recent Training Iterations (last 20):
{chr(10).join(iter_summary)}

Trade Summary: {json.dumps(trade_summary)}
Total Trades: {len(trades)}

Provide:
1. A 2-3 sentence performance summary
2. Key findings (3-5 observations about what's working/not working)
3. Actionable suggestions (3-5 specific hyperparameter changes or strategy modifications)
4. Confidence level (high/medium/low) based on data quality

Format as JSON:
{{"summary": "...", "findings": ["..."], "suggestions": ["..."], "confidence": "..."}}"""

    def _build_comparison_prompt(self, experiments) -> str:
        exp_descriptions = []
        for item in experiments:
            exp = item["experiment"]
            iters = item["iterations"]
            last_metrics = iters[-1].metrics if iters else {}
            exp_descriptions.append(
                f"- {exp.name} (script: {exp.script})\n"
                f"  Config: {json.dumps(exp.config)}\n"
                f"  Final: {json.dumps(exp.final_metrics)}\n"
                f"  Last iteration metrics: {json.dumps(last_metrics)}\n"
                f"  Iterations: {len(iters)}"
            )

        return f"""Compare these reinforcement learning trading experiments:

{chr(10).join(exp_descriptions)}

Provide:
1. A 2-3 sentence overview comparing the experiments
2. Key findings: what config differences led to better/worse results (3-5 points)
3. Suggestions: optimal parameter ranges based on the comparison (3-5 points)
4. Confidence level

Format as JSON:
{{"summary": "...", "findings": ["..."], "suggestions": ["..."], "confidence": "..."}}"""

    def _build_strategy_prompt(self, study_name, trials) -> str:
        trial_summaries = []
        for t in trials:
            trial_summaries.append(
                f"  Trial {t.trial_number}: value={t.value}, state={t.state}, "
                f"params={json.dumps(t.params)}"
            )

        complete = [t for t in trials if t.state == "complete"]
        values = [t.value for t in complete if t.value is not None]

        return f"""Analyze this Optuna hyperparameter optimization study:

Study: {study_name}
Total Trials: {len(trials)}
Completed: {len(complete)}
Best Value: {max(values) if values else 'N/A'}
Worst Value: {min(values) if values else 'N/A'}

All Trials:
{chr(10).join(trial_summaries)}

Provide:
1. A 2-3 sentence summary of the optimization progress
2. Key findings about which parameters matter most (3-5 points)
3. Suggestions for next hyperparameter regions to explore (3-5 specific ranges)
4. Confidence level

Format as JSON:
{{"summary": "...", "findings": ["..."], "suggestions": ["..."], "confidence": "..."}}"""

    def _build_trades_prompt(self, exp, trades) -> str:
        trade_data = []
        for t in trades[:200]:  # Limit context
            trade_data.append({
                "episode": t.episode,
                "step": t.step,
                "side": t.side,
                "price": t.price,
                "size": t.size,
                "commission": t.commission,
            })

        return f"""Analyze the trade patterns from this RL trading experiment:

Experiment: {exp.name}
Config: {json.dumps(exp.config, indent=2)}
Final Metrics: {json.dumps(exp.final_metrics, indent=2)}
Total Trades: {len(trades)}

Sample Trades (first 200):
{json.dumps(trade_data, indent=2)}

Provide:
1. A 2-3 sentence overview of trading behavior
2. Key findings about entry/exit timing, position sizing, and patterns (3-5 points)
3. Suggestions for improving trading strategy (3-5 actionable improvements)
4. Confidence level

Format as JSON:
{{"summary": "...", "findings": ["..."], "suggestions": ["..."], "confidence": "..."}}"""

    async def _query_claude(
        self,
        prompt: str,
        experiment_ids: list[str],
        analysis_type: str,
        custom_prompt: str | None = None,
    ) -> InsightReport:
        """Send prompt to Claude and parse structured response."""
        import asyncio

        if custom_prompt:
            prompt += f"\n\nAdditional user question/context:\n{custom_prompt}"

        # Run synchronous API call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            ),
        )

        raw_text = response.content[0].text

        # Parse JSON from response
        parsed = self._parse_response(raw_text)

        report = InsightReport(
            id=str(uuid.uuid4()),
            experiment_ids=experiment_ids,
            analysis_type=analysis_type,
            summary=parsed.get("summary", "Analysis complete"),
            findings=parsed.get("findings", []),
            suggestions=parsed.get("suggestions", []),
            confidence=parsed.get("confidence", "medium"),
            raw_response=raw_text,
        )

        # Store the insight
        self.store.store_insight(
            insight_id=report.id,
            experiment_ids=report.experiment_ids,
            analysis_type=report.analysis_type,
            summary=report.summary,
            findings=report.findings,
            suggestions=report.suggestions,
            confidence=report.confidence,
            raw_response=report.raw_response,
        )

        return report

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Extract JSON from Claude's response text."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as summary
        return {
            "summary": text[:500],
            "findings": [],
            "suggestions": [],
            "confidence": "low",
        }
