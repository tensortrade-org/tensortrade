"use client";

interface StepIndicatorProps {
	currentStep: number;
	onStepClick: (step: number) => void;
}

const STEP_LABELS = [
	"Name & Strategy",
	"Dataset",
	"Hyperparameters",
	"Environment",
	"Review & Launch",
];

export function StepIndicator({ currentStep, onStepClick }: StepIndicatorProps) {
	return (
		<div className="flex items-center justify-between px-4 py-3">
			{STEP_LABELS.map((label, idx) => {
				const isCompleted = idx < currentStep;
				const isActive = idx === currentStep;

				return (
					<div key={label} className="flex flex-1 items-center">
						<button
							type="button"
							onClick={() => onStepClick(idx)}
							className="flex flex-col items-center gap-1"
						>
							<div
								className={`flex h-8 w-8 items-center justify-center rounded-full border-2 text-xs font-bold transition-colors ${
									isActive
										? "border-[var(--accent-blue)] bg-[var(--accent-blue)] text-white"
										: isCompleted
											? "border-[var(--accent-green)] bg-[var(--accent-green)] text-white"
											: "border-[var(--border-color)] bg-transparent text-[var(--text-secondary)]"
								}`}
							>
								{isCompleted ? "\u2713" : idx + 1}
							</div>
							<span
								className={`text-xs whitespace-nowrap ${
									isActive
										? "text-[var(--accent-blue)] font-medium"
										: isCompleted
											? "text-[var(--accent-green)]"
											: "text-[var(--text-secondary)]"
								}`}
							>
								{label}
							</span>
						</button>
						{idx < STEP_LABELS.length - 1 && (
							<div
								className={`mx-2 h-0.5 flex-1 ${
									idx < currentStep ? "bg-[var(--accent-green)]" : "bg-[var(--border-color)]"
								}`}
							/>
						)}
					</div>
				);
			})}
		</div>
	);
}
