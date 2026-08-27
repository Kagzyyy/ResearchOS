import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchOS — AI paper search and citation explainer",
  description:
    "Ask questions about AI research papers and de-reference citations, grounded in retrieved passages.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-black text-slate-200 antialiased">{children}</body>
    </html>
  );
}
