import { DashboardShell } from "@/components/layout/DashboardShell";
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
	title: "TensorTrade Dashboard",
	description: "Real-time training intelligence platform",
};

export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="en">
			<body>
				<DashboardShell>{children}</DashboardShell>
			</body>
		</html>
	);
}
