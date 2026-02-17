"use client";

import { Card } from "@/components/common/Card";
import { DatasetStep } from "@/components/launch/DatasetStep";
import { EnvironmentStep } from "@/components/launch/EnvironmentStep";
import { HyperparamStep } from "@/components/launch/HyperparamStep";
import { NameStep } from "@/components/launch/NameStep";
import { ReviewStep } from "@/components/launch/ReviewStep";
import { StepIndicator } from "@/components/launch/StepIndicator";
import { type LaunchStep, STEP_ORDER, useLaunchStore } from "@/stores/launchStore";

const STEP_COMPONENTS: Record<LaunchStep, React.ComponentType> = {
	name: NameStep,
	dataset: DatasetStep,
	hyperparams: HyperparamStep,
	environment: EnvironmentStep,
	review: ReviewStep,
};

function canProceed(
	step: LaunchStep,
	state: { name: string; datasetId: string | null; hpPackId: string | null },
): boolean {
	switch (step) {
		case "name":
			return state.name.trim().length > 0;
		case "dataset":
			return state.datasetId !== null;
		case "hyperparams":
			return state.hpPackId !== null;
		case "environment":
			return true;
		case "review":
			return true;
	}
}

export function LaunchWizard() {
	const currentStep = useLaunchStore((s) => s.currentStep);
	const setStep = useLaunchStore((s) => s.setStep);
	const name = useLaunchStore((s) => s.name);
	const datasetId = useLaunchStore((s) => s.datasetId);
	const hpPackId = useLaunchStore((s) => s.hpPackId);
	const launchedExperimentId = useLaunchStore((s) => s.launchedExperimentId);

	const currentIndex = STEP_ORDER.indexOf(currentStep);
	const isFirst = currentIndex === 0;
	const isLast = currentIndex === STEP_ORDER.length - 1;
	const StepComponent = STEP_COMPONENTS[currentStep];

	const handleBack = () => {
		if (!isFirst) {
			setStep(STEP_ORDER[currentIndex - 1]);
		}
	};

	const handleNext = () => {
		if (!isLast && canProceed(currentStep, { name, datasetId, hpPackId })) {
			setStep(STEP_ORDER[currentIndex + 1]);
		}
	};

	const handleStepClick = (stepIndex: number) => {
		if (stepIndex < currentIndex) {
			setStep(STEP_ORDER[stepIndex]);
			return;
		}
		for (let i = 0; i < stepIndex; i++) {
			if (!canProceed(STEP_ORDER[i], { name, datasetId, hpPackId })) {
				return;
			}
		}
		setStep(STEP_ORDER[stepIndex]);
	};

	return (
		<Card className="overflow-hidden">
			<StepIndicator currentStep={currentIndex} onStepClick={handleStepClick} />

			<div className="border-t border-[var(--border-color)] p-6">
				<StepComponent />
			</div>

			{!launchedExperimentId && (
				<div className="flex items-center justify-between border-t border-[var(--border-color)] px-6 py-4">
					<button
						type="button"
						onClick={handleBack}
						disabled={isFirst}
						className="rounded-md border border-[var(--border-color)] px-4 py-2 text-sm font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-secondary)] disabled:cursor-not-allowed disabled:opacity-40"
					>
						Back
					</button>

					{!isLast && (
						<button
							type="button"
							onClick={handleNext}
							disabled={!canProceed(currentStep, { name, datasetId, hpPackId })}
							className="rounded-md bg-[var(--accent-blue)] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[var(--accent-blue)]/80 disabled:cursor-not-allowed disabled:opacity-40"
						>
							Next
						</button>
					)}
				</div>
			)}
		</Card>
	);
}
