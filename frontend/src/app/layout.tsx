import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI-Powered Urban Crime Risk & Hotspot Prediction System",
  description: "Chicago Spatial Analytics, DBSCAN Hotspot Detection & Crime Risk Intelligence Dashboard (Milestone 2)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}
