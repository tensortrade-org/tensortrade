interface BadgeProps {
	label: string;
	variant?: "default" | "success" | "danger" | "warning" | "info" | "purple";
}

const variantStyles: Record<NonNullable<BadgeProps["variant"]>, string> = {
	default: "bg-[var(--border-color)] text-[var(--text-secondary)]",
	success: "bg-[var(--accent-green)]/15 text-[var(--accent-green)]",
	danger: "bg-[var(--accent-red)]/15 text-[var(--accent-red)]",
	warning: "bg-[var(--accent-amber)]/15 text-[var(--accent-amber)]",
	info: "bg-[var(--accent-blue)]/15 text-[var(--accent-blue)]",
	purple: "bg-[var(--accent-purple)]/15 text-[var(--accent-purple)]",
};

export function Badge({ label, variant = "default" }: BadgeProps) {
	return (
		<span
			className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${variantStyles[variant]}`}
		>
			{label}
		</span>
	);
}

interface StatusBadgeProps {
	status: string;
}

const statusVariant: Record<string, BadgeProps["variant"]> = {
	running: "info",
	completed: "success",
	failed: "danger",
	pruned: "warning",
};

export function StatusBadge({ status }: StatusBadgeProps) {
	return <Badge label={status} variant={statusVariant[status] ?? "default"} />;
}
