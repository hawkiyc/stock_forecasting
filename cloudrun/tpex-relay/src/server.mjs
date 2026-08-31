import http from "node:http";

import { handleRequest } from "./relay.mjs";

const port = Number.parseInt(process.env.PORT ?? "8080", 10);
const sharedSecret = process.env.TPEX_PROXY_SHARED_SECRET ?? "";

if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new Error("PORT must be an integer from 1 through 65535");
}
if (sharedSecret.length < 32 || sharedSecret.length > 512) {
  throw new Error("TPEX_PROXY_SHARED_SECRET must contain 32 through 512 characters");
}

const relayEnvironment = Object.freeze({
  K_REVISION: process.env.K_REVISION ?? "local",
  TPEX_PROXY_SHARED_SECRET: sharedSecret,
  TPEX_RELAY_REGION: process.env.TPEX_RELAY_REGION ?? "unknown",
});

function webHeaders(rawHeaders) {
  const headers = new Headers();
  for (const [name, rawValue] of Object.entries(rawHeaders)) {
    if (Array.isArray(rawValue)) {
      for (const value of rawValue) {
        headers.append(name, value);
      }
    } else if (typeof rawValue === "string") {
      headers.set(name, rawValue);
    }
  }
  return headers;
}

const server = http.createServer(async (incoming, outgoing) => {
  try {
    const requestUrl = new URL(incoming.url ?? "/", "https://relay.invalid");
    const request = new Request(requestUrl, {
      method: incoming.method ?? "GET",
      headers: webHeaders(incoming.headers),
    });
    const response = await handleRequest(request, relayEnvironment);
    outgoing.writeHead(response.status, Object.fromEntries(response.headers.entries()));
    outgoing.end(Buffer.from(await response.arrayBuffer()));
  } catch (_error) {
    outgoing.writeHead(500, {
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
    });
    outgoing.end(JSON.stringify({ error: "relay_internal_error" }));
  }
});

server.headersTimeout = 10_000;
server.requestTimeout = 60_000;
server.keepAliveTimeout = 5_000;
server.listen(port);

process.on("SIGTERM", () => {
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 9_000).unref();
});
