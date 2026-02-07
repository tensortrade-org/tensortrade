"""
AI Insights Engine powered by Anthropic Claude API.

Analyzes training results, compares experiments, suggests
hyperparameter strategies, and provides trade pattern analysis.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncGenerator
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

    async def analyze_campaign(
        self, study_name: str, *, custom_prompt: str | None = None
    ) -> InsightReport:
        """Comprehensive written analysis of an Optuna campaign's results."""
        trials = self.store.get_optuna_trials(study_name)
        if not trials:
            raise ValueError(f"No trials found for study {study_name}")

        prompt = self._build_campaign_analysis_prompt(study_name, trials)
        return await self._query_claude(
            prompt=prompt,
            experiment_ids=[f"study:{study_name}"],
            analysis_type="campaign_analysis",
            custom_prompt=custom_prompt,
        )

    async def stream_campaign_analysis(
        self, study_name: str, *, custom_prompt: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream campaign analysis via SSE-formatted events.

        Yields SSE events:
          event: chunk   — text delta from the model
          event: complete — final InsightReport JSON
          event: error   — error message
        """
        import asyncio

        trials = self.store.get_optuna_trials(study_name)
        if not trials:
            yield f"event: error\ndata: {json.dumps({'error': f'No trials found for study {study_name}'})}\n\n"
            return

        prompt = self._build_campaign_analysis_prompt(study_name, trials)
        if custom_prompt:
            prompt += f"\n\nAdditional user question/context:\n{custom_prompt}"

        accumulated_text = ""
        try:
            loop = asyncio.get_event_loop()

            # Run the synchronous streaming call in an executor via a
            # thread that pushes chunks into an asyncio.Queue.
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _run_stream() -> None:
                try:
                    with self.client.messages.stream(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}],
                    ) as stream:
                        for text in stream.text_stream:
                            loop.call_soon_threadsafe(queue.put_nowait, text)
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, f"__ERROR__:{exc}"
                    )

            # Fire the stream in a background thread — don't await it,
            # so we can drain the queue concurrently as chunks arrive.
            stream_future = loop.run_in_executor(None, _run_stream)

            # Drain the queue, yielding SSE chunks as they arrive
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, str) and chunk.startswith("__ERROR__:"):
                    error_msg = chunk[len("__ERROR__:"):]
                    yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"
                    return
                accumulated_text += chunk
                yield f"event: chunk\ndata: {json.dumps({'text': chunk})}\n\n"

            # Ensure the background thread has finished
            await stream_future

            # Parse and persist the completed response
            parsed = self._parse_response(accumulated_text)
            report = InsightReport(
                id=str(uuid.uuid4()),
                experiment_ids=[f"study:{study_name}"],
                analysis_type="campaign_analysis",
                summary=parsed.get("summary", "Analysis complete"),
                findings=parsed.get("findings", []),
                suggestions=parsed.get("suggestions", []),
                confidence=parsed.get("confidence", "medium"),
                raw_response=accumulated_text,
            )

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

            yield f"event: complete\ndata: {json.dumps(asdict(report))}\n\n"

        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

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

    def _build_campaign_analysis_prompt(self, study_name: str, trials: list) -> str:
        complete = [t for t in trials if t.state == "complete"]
        pruned = [t for t in trials if t.state == "pruned"]
        failed = [t for t in trials if t.state == "fail"]
        values = [t.value for t in complete if t.value is not None]

        best_value = max(values) if values else None
        worst_value = min(values) if values else None
        avg_value = sum(values) / len(values) if values else None

        # Best trial details
        best_trial = max(complete, key=lambda t: t.value or float("-inf")) if complete else None
        best_params_str = json.dumps(best_trial.params, indent=2) if best_trial else "N/A"

        # Top-5 and bottom-5 trials
        sorted_complete = sorted(complete, key=lambda t: t.value or 0, reverse=True)
        top5 = sorted_complete[:5]
        bottom5 = sorted_complete[-5:] if len(sorted_complete) > 5 else []

        top5_lines = []
        for t in top5:
            top5_lines.append(
                f"  Trial {t.trial_number}: value={t.value}, params={json.dumps(t.params)}"
            )
        bottom5_lines = []
        for t in bottom5:
            bottom5_lines.append(
                f"  Trial {t.trial_number}: value={t.value}, params={json.dumps(t.params)}"
            )

        # Parameter importance via correlation
        importance_lines = self._compute_param_importance(complete)

        # Convergence info
        convergence_info = "N/A"
        if best_trial:
            convergence_info = (
                f"Best found at trial {best_trial.trial_number} of {len(trials)} total. "
            )
            if best_trial.trial_number < len(trials) * 0.3:
                convergence_info += "Early convergence — search may benefit from wider exploration."
            elif best_trial.trial_number > len(trials) * 0.8:
                convergence_info += "Late improvement — search may still be converging, consider more trials."
            else:
                convergence_info += "Mid-campaign best — reasonable convergence pattern."

        return f"""Analyze this Optuna hyperparameter search campaign.
Write like a human analyst presenting findings to a team.

Study: {study_name}
Total Trials: {len(trials)} (completed: {len(complete)}, pruned: {len(pruned)}, failed: {len(failed)})
Best Value: {best_value}
Worst Value: {worst_value}
Average Value: {f'{avg_value:.2f}' if avg_value is not None else 'N/A'}

Best Trial Parameters:
{best_params_str}

Top 5 Trials:
{chr(10).join(top5_lines)}

Bottom 5 Trials:
{chr(10).join(bottom5_lines) if bottom5_lines else '  (fewer than 10 completed trials)'}

Parameter Importance (correlation with objective):
{chr(10).join(importance_lines) if importance_lines else '  Insufficient data'}

Convergence: {convergence_info}

IMPORTANT — you must respond in exactly two parts:

PART 1: Write a flowing, readable analysis using markdown. Cover:
- Executive summary (2-3 sentences)
- Which parameters matter and their optimal ranges
- Convergence assessment
- Risks and data quality concerns
- Concrete next steps

PART 2: After your written analysis, output a fenced JSON block (```json ... ```) with this exact structure:
{{"summary": "3-5 sentence executive summary",
"findings": ["finding 1", "finding 2", "finding 3"],
"suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
"confidence": "high|medium|low"}}

Write Part 1 first (the human-readable analysis), then Part 2 (the JSON block) at the very end."""

    @staticmethod
    def _compute_param_importance(complete_trials: list) -> list[str]:
        """Compute simple correlation-based parameter importance."""
        if len(complete_trials) < 3:
            return []

        values = [t.value for t in complete_trials if t.value is not None]
        if not values:
            return []

        # Collect numeric params
        param_keys: set[str] = set()
        for t in complete_trials:
            for k, v in t.params.items():
                if isinstance(v, (int, float)):
                    param_keys.add(k)

        importance: dict[str, float] = {}
        for key in param_keys:
            param_vals = []
            obj_vals = []
            for t in complete_trials:
                if t.value is not None and key in t.params:
                    val = t.params[key]
                    if isinstance(val, (int, float)):
                        param_vals.append(float(val))
                        obj_vals.append(float(t.value))

            if len(param_vals) < 3:
                continue

            # Pearson correlation
            n = len(param_vals)
            mean_p = sum(param_vals) / n
            mean_o = sum(obj_vals) / n
            cov = sum((p - mean_p) * (o - mean_o) for p, o in zip(param_vals, obj_vals)) / n
            std_p = (sum((p - mean_p) ** 2 for p in param_vals) / n) ** 0.5
            std_o = (sum((o - mean_o) ** 2 for o in obj_vals) / n) ** 0.5
            if std_p > 0 and std_o > 0:
                importance[key] = abs(cov / (std_p * std_o))

        # Sort by importance descending
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return [f"  {k}: {v:.3f}" for k, v in sorted_imp]

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

        # Try to extract from ```json ... ``` fenced block
        fence_start = text.find("```json")
        if fence_start >= 0:
            json_start = text.find("{", fence_start)
            fence_end = text.find("```", fence_start + 7)
            if json_start >= 0 and fence_end > json_start:
                try:
                    return json.loads(text[json_start:fence_end].strip())
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
