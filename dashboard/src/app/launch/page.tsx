"use client";

import { LaunchWizard } from "@/components/launch/LaunchWizard";

export default function LaunchPage() {
	return (
		<div className="mx-auto max-w-4xl space-y-6 p-6">
			<div>
				<h1 className="text-2xl font-bold text-[var(--text-primary)]">Launch Training</h1>
				<p className="mt-1 text-sm text-[var(--text-secondary)]">
					Configure and start a new training experiment
				</p>
			</div>
			<LaunchWizard />
		</div>
	);
}
