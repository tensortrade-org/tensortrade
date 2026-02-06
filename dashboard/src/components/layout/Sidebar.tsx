"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

interface NavItem {
	label: string;
	href: string;
	icon: string;
}

const navItems: NavItem[] = [
	{ label: "Overview", href: "/", icon: "\u2302" },
	{ label: "Launch", href: "/launch", icon: "\u25B6" },
	{ label: "Alpha Search", href: "/campaign", icon: "\u2694" },
	{ label: "HP Studio", href: "/hyperparams", icon: "\u2692" },
	{ label: "Datasets", href: "/datasets", icon: "\u25A6" },
	{ label: "Inference", href: "/live", icon: "\u23E9" },
	{ label: "Experiments", href: "/experiments", icon: "\u2630" },
	{ label: "Leaderboard", href: "/leaderboard", icon: "\u2655" },
	{ label: "Optuna", href: "/optuna", icon: "\u2699" },
	{ label: "Insights", href: "/insights", icon: "\u2605" },
	{ label: "Trades", href: "/trades", icon: "\u21C4" },
];

export function Sidebar() {
	const pathname = usePathname();

	const isActive = (href: string) => {
		if (href === "/") return pathname === "/";
		return pathname.startsWith(href);
	};

	return (
		<aside className="flex h-screen w-56 flex-col border-r border-[var(--border-color)] bg-[var(--bg-secondary)]">
			<div className="flex h-14 items-center gap-2 border-b border-[var(--border-color)] px-4">
				<span className="text-lg font-bold text-[var(--accent-blue)]">TT</span>
				<span className="text-sm font-semibold text-[var(--text-primary)]">TensorTrade</span>
			</div>
			<nav className="flex-1 overflow-y-auto px-2 py-3">
				{navItems.map((item) => (
					<Link
						key={item.href}
						href={item.href}
						className={`mb-0.5 flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${
							isActive(item.href)
								? "bg-[var(--accent-blue)]/10 text-[var(--accent-blue)]"
								: "text-[var(--text-secondary)] hover:bg-[var(--bg-primary)] hover:text-[var(--text-primary)]"
						}`}
					>
						<span className="w-5 text-center">{item.icon}</span>
						{item.label}
					</Link>
				))}
			</nav>
			<div className="border-t border-[var(--border-color)] px-4 py-3 text-xs text-[var(--text-secondary)]">
				v1.0.4
			</div>
		</aside>
	);
}
