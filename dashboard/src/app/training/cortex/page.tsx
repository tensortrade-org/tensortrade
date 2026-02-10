"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { BehavioralOrbit } from "@/components/cortex/BehavioralOrbit";
import { DecisionFlow } from "@/components/cortex/DecisionFlow";
import { LearningPulse } from "@/components/cortex/LearningPulse";
import { StrategyGenome } from "@/components/cortex/StrategyGenome";
import { StatusIndicator } from "@/components/training/StatusIndicator";
import { useTrainingStore } from "@/stores/trainingStore";

export default function AgentCortexPage() {
	const iterations = useTrainingStore((s) => s.iterations);
	const episodes = useTrainingStore((s) => s.episodes);
	const progress = useTrainingStore((s) => s.progress);

	return (
		<div className="space-y-4">
			{/* Header */}
			<div className="flex items-center justify-between">
				<div>
					<h1 className="text-xl font-semibold text-[var(--text-primary)]">Agent Cortex</h1>
					<p className="text-sm text-[var(--text-secondary)]">
						Deep visualization of training dynamics
					</p>
				</div>
				<StatusIndicator />
			</div>

			{/* Strategy Genome — full width */}
			<Card>
				<CardHeader title="Strategy Genome" />
				<div className="h-64">
					<StrategyGenome iterations={iterations} />
				</div>
			</Card>

			{/* Bottom row — 3 panels */}
			<div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
				<Card>
					<CardHeader title="Behavioral Orbit" />
					<div className="aspect-square">
						<BehavioralOrbit iterations={iterations} />
					</div>
				</Card>

				<Card>
					<CardHeader title="Learning Pulse" />
					<div className="aspect-square">
						<LearningPulse iterations={iterations} episodes={episodes} progress={progress} />
					</div>
				</Card>

				<Card>
					<CardHeader title="Decision Flow" />
					<div className="aspect-square">
						<DecisionFlow iterations={iterations} />
					</div>
				</Card>
			</div>
		</div>
	);
}
