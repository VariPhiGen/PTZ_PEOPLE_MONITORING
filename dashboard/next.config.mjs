// @ts-check

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:18000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "http",  hostname: "**" },
      { protocol: "https", hostname: "**" },
    ],
  },

  // Proxy /api/* → FastAPI backend.
  // This makes relative "/api/..." calls work when the dashboard is accessed
  // directly on port 3020 as well as through the nginx proxy on port 7080.
  async rewrites() {
    return [
      {
        source:      "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
