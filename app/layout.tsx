import type { Metadata } from 'next';
import { GeistMono } from 'geist/font/mono';
import { GeistSans } from 'geist/font/sans';
import './globals.css';

const siteUrl = process.env.SITE_URL ?? 'https://docs.humanfia.ai/KDA-wishlist';

export const metadata: Metadata = {
  metadataBase: new URL(`${siteUrl}/`),
  title: 'KDA: Kernel Design Agents',
  description:
    'Submit reproducible GPU kernel definitions and workloads for Kernel Design Agents to optimize, validate, and return.',
  keywords: [
    'Kernel Design Agents',
    'GPU kernel optimization',
    'FlashInfer Trace',
    'KDA Wishlist',
  ],
  openGraph: {
    type: 'website',
    url: `${siteUrl}/`,
    title: 'KDA Wishlist — Community wants, KDA provides!',
    description:
      'Turn a reproducible kernel definition into an optimized, validated implementation.',
    images: [{
      url: `${siteUrl}/og.png`,
      width: 1200,
      height: 630,
      alt: 'KDA Wishlist workflow',
    }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'KDA Wishlist — Community wants, KDA provides!',
    description:
      'Turn a reproducible kernel definition into an optimized, validated implementation.',
    images: [
      `${siteUrl}/og.png`,
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${GeistSans.variable} ${GeistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
