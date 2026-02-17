interface CardProps {
	children: React.ReactNode;
	className?: string;
}

export function Card({ children, className = "" }: CardProps) {
	return (
		<div
			className={`rounded-lg border border-[var(--border-color)] bg-[var(--bg-card)] p-4 ${className}`}
		>
			{children}
		</div>
	);
}

interface CardHeaderProps {
	title: string;
	action?: React.ReactNode;
}

export function CardHeader({ title, action }: CardHeaderProps) {
	return (
		<div className="mb-3 flex items-center justify-between">
			<h3 className="text-sm font-medium text-[var(--text-secondary)]">{title}</h3>
			{action}
		</div>
	);
}
