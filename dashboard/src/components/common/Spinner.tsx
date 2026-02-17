interface SpinnerProps {
	size?: "sm" | "md" | "lg";
}

const sizeClasses: Record<NonNullable<SpinnerProps["size"]>, string> = {
	sm: "h-4 w-4 border-2",
	md: "h-6 w-6 border-2",
	lg: "h-10 w-10 border-3",
};

export function Spinner({ size = "md" }: SpinnerProps) {
	return (
		<div
			className={`animate-spin rounded-full border-[var(--border-color)] border-t-[var(--accent-blue)] ${sizeClasses[size]}`}
		/>
	);
}

export function LoadingState({ message = "Loading..." }: { message?: string }) {
	return (
		<div className="flex flex-col items-center justify-center gap-3 py-12">
			<Spinner size="lg" />
			<p className="text-sm text-[var(--text-secondary)]">{message}</p>
		</div>
	);
}
