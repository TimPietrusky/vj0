import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";

export const metadata: Metadata = {
  title: "vj0 — live audio-reactive visuals",
  description: "Real-time audio visualization for live visual artists",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // suppressHydrationWarning on BOTH <html> and <body>: browser extensions
    // inject attributes on these top-level elements before React hydrates,
    // and there's no way to make the SSR output match. Known offenders:
    //   <html>: Google Analytics Opt-out (data-google-analytics-opt-out),
    //           Dark Reader (data-darkreader-*)
    //   <body>: ColorZilla (cz-shortcut-listen),
    //           Grammarly (data-gr-*, data-new-gr-c-s-*),
    //           various password managers
    // The flag is one-level-deep — only this element's own attributes are
    // exempted, children are still hydration-checked normally.
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${GeistSans.variable} ${GeistMono.variable} antialiased`}
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
