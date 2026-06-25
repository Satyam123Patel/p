import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Camera Mouse — Gesture Control",
  description:
    "Control your cursor with hand gestures using your webcam. Powered by MediaPipe AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body style={{ margin: 0, padding: 0, overflow: "hidden" }}>
        {children}
      </body>
    </html>
  );
}
