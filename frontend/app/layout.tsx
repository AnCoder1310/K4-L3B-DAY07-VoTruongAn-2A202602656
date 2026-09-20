import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "K4-L3B · CP6 RAG Evaluation Studio",
  description: "Lab 7 CP6: Embedding, Vector Store, Chunking, and Benchmark Evaluation for Shopee Policies",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi">
      <body className="bg-brand-bg text-brand-text antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}
