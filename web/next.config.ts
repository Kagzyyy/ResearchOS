import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // No `output: "standalone"` here: that mode is for self-hosting (Docker) and
  // breaks Vercel's build, which manages its own serverless output. If you ever
  // containerise this front end, add it back for that build only.
};

export default nextConfig;
