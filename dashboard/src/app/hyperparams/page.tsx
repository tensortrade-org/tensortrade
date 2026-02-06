"use client";

import { PackCompare } from "@/components/hyperparams/PackCompare";
import { PackEditor } from "@/components/hyperparams/PackEditor";
import { PackList } from "@/components/hyperparams/PackList";
import { useState } from "react";

type TabId = "edit" | "compare";

interface Tab {
	id: TabId;
	label: string;
}

const tabs: Tab[] = [
	{ id: "edit", label: "Edit" },
	{ id: "compare", label: "Compare" },
];

export default function HyperparamsPage() {
	const [activeTab, setActiveTab] = useState<TabId>("edit");

	return (
		<div className="flex h-full flex-col">
			<div className="mb-4">
				<h1 className="text-xl font-semibold text-[var(--text-primary)]">Hyperparameter Studio</h1>
			</div>
			<div className="flex flex-1 overflow-hidden rounded-lg border border-[var(--border-color)]">
				{/* Left sidebar */}
				<div className="w-64 shrink-0 overflow-y-auto">
					<PackList />
				</div>

				{/* Right content area */}
				<div className="flex flex-1 flex-col overflow-hidden">
					{/* Tab bar */}
					<div className="flex border-b border-[var(--border-color)] bg-[var(--bg-secondary)]">
						{tabs.map((tab) => (
							<button
								key={tab.id}
								type="button"
								onClick={() => setActiveTab(tab.id)}
								className={`px-4 py-2 text-sm font-medium transition-colors ${
									activeTab === tab.id
										? "border-b-2 border-[var(--accent-blue)] text-[var(--accent-blue)]"
										: "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
								}`}
							>
								{tab.label}
							</button>
						))}
					</div>

					{/* Tab content */}
					<div className="flex-1 overflow-y-auto p-4">
						{activeTab === "edit" ? <PackEditor /> : <PackCompare />}
					</div>
				</div>
			</div>
		</div>
	);
}
