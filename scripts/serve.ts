/**
 * Serve the demo page, with the bundle in dist/ and the weights, from the repository root.
 * Run with: bun scripts/serve.ts (after `bun run build`)
 */

import { join } from "node:path";

const root = join(import.meta.dir, "..");

const server = Bun.serve({
  async fetch(req) {
    const { pathname } = new URL(req.url);
    const file = Bun.file(join(root, pathname === "/" ? "index.html" : pathname));
    return (await file.exists()) ? new Response(file) : new Response("Not found", { status: 404 });
  },
});

console.log(`Serving the demo page at ${server.url}`);
