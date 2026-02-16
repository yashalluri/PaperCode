import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Paper Replicator",
  description: "Autonomous ML paper reproduction powered by Claude Code",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0a0a0f] text-gray-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
