import assert from "node:assert/strict";
import test from "node:test";

import { handleRequest, relayContract } from "../src/relay.mjs";

const TOKEN = "fixture-token-that-is-longer-than-thirty-two-characters";
const ENVIRONMENT = Object.freeze({
  K_REVISION: "fixture-revision-00001",
  TPEX_PROXY_SHARED_SECRET: TOKEN,
  TPEX_RELAY_REGION: "asia-east1",
});
const ACTION_URL =
  "https://relay.example/www/zh-tw/bulletin/exDailyQ" +
  "?startDate=2026%2F01%2F01&endDate=2026%2F01%2F31&response=json";

function request(url = ACTION_URL, token = TOKEN) {
  return new Request(url, { headers: { "X-TPEX-Relay-Token": token } });
}

test("authenticated warmup starts the service without contacting TPEx", async () => {
  let fetched = false;
  const response = await handleRequest(
    request("https://relay.example/_internal/warmup"),
    ENVIRONMENT,
    async () => {
      fetched = true;
      return new Response("{}");
    },
  );

  assert.equal(response.status, 200);
  assert.equal(fetched, false);
  assert.deepEqual(await response.json(), {
    ready: true,
    version: "1",
    region: "asia-east1",
    revision: "fixture-revision-00001",
  });
  assert.equal(response.headers.get("X-TPEX-Relay-Version"), "1");
  assert.equal(response.headers.get("X-TPEX-Relay-Region"), "asia-east1");
});

test("forwards only the fixed TPEx origin and returns original JSON bytes", async () => {
  let capturedUrl;
  let capturedOptions;
  const originalBody = '{"tables":[],"spacing": "preserved"}\n';
  const response = await handleRequest(request(), ENVIRONMENT, async (url, options) => {
    capturedUrl = url;
    capturedOptions = options;
    return new Response(originalBody, {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  });

  assert.equal(response.status, 200);
  assert.equal(await response.text(), originalBody);
  assert.equal(capturedUrl.origin, "https://www.tpex.org.tw");
  assert.equal(capturedUrl.pathname, "/www/zh-tw/bulletin/exDailyQ");
  assert.equal(capturedUrl.searchParams.get("startDate"), "2026/01/01");
  assert.equal(capturedOptions.redirect, "manual");
  assert.equal(capturedOptions.cache, "no-store");
  assert.equal(capturedOptions.headers.get("Cookie"), null);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.equal(response.headers.get("X-TPEX-Upstream-Redirects"), "0");
});

test("follows a bounded same-origin redirect with its session cookie", async () => {
  const fetchedCookies = [];
  const response = await handleRequest(request(), ENVIRONMENT, async (url, options) => {
    fetchedCookies.push(options.headers.get("Cookie"));
    if (fetchedCookies.length === 1) {
      return new Response(null, {
        status: 302,
        headers: {
          Location: url.toString(),
          "Set-Cookie": "JSESSIONID=fixture-session; Path=/; Secure; HttpOnly",
        },
      });
    }
    return new Response(JSON.stringify({ tables: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });

  assert.equal(response.status, 200);
  assert.deepEqual(fetchedCookies, [null, "JSESSIONID=fixture-session"]);
  assert.equal(response.headers.get("X-TPEX-Upstream-Redirects"), "1");
  assert.equal(response.headers.get("X-TPEX-Upstream-Redirect-Cookies"), "1");
});

test("rejects cross-origin redirects and no-progress redirect loops", async () => {
  const crossOrigin = await handleRequest(request(), ENVIRONMENT, async () =>
    new Response(null, { status: 302, headers: { Location: "https://example.com/" } }),
  );
  assert.equal(crossOrigin.status, 502);
  assert.deepEqual(await crossOrigin.json(), {
    error: "tpex_upstream_redirect_rejected",
  });

  let loopCalls = 0;
  const loop = await handleRequest(request(), ENVIRONMENT, async (url) => {
    loopCalls += 1;
    return new Response(null, { status: 302, headers: { Location: url.toString() } });
  });
  assert.equal(loop.status, 502);
  assert.equal(loopCalls, 1);
  assert.deepEqual(await loop.json(), { error: "tpex_upstream_redirect_loop" });
});

test("rejects missing authentication, unapproved requests, and invalid success JSON", async () => {
  let fetched = false;
  const unauthorized = await handleRequest(
    request(ACTION_URL, "wrong-token-that-is-also-longer-than-thirty-two"),
    ENVIRONMENT,
    async () => {
      fetched = true;
      return new Response("{}");
    },
  );
  assert.equal(unauthorized.status, 401);
  assert.equal(fetched, false);

  const candidates = [
    request("https://relay.example/arbitrary?response=json"),
    request(`${ACTION_URL}&target=https%3A%2F%2Fevil.example`),
    request(`${ACTION_URL}&response=json`),
    request(
      "https://relay.example/www/zh-tw/bulletin/exDailyQ" +
        "?startDate=2026%2F02%2F31&endDate=2026%2F03%2F01&response=json",
    ),
    request("https://relay.example/_internal/warmup?probe=upstream"),
    new Request(ACTION_URL, {
      method: "POST",
      headers: { "X-TPEX-Relay-Token": TOKEN },
    }),
  ];
  for (const candidate of candidates) {
    const response = await handleRequest(candidate, ENVIRONMENT);
    assert.ok([400, 405].includes(response.status));
  }

  const invalidJson = await handleRequest(request(), ENVIRONMENT, async () =>
    new Response("<html>not json</html>", {
      status: 200,
      headers: { "Content-Type": "text/html" },
    }),
  );
  assert.equal(invalidJson.status, 502);
  assert.deepEqual(await invalidJson.json(), { error: "tpex_upstream_invalid_json" });
});

test("preserves bounded upstream error responses without exposing cookies", async () => {
  const response = await handleRequest(request(), ENVIRONMENT, async () =>
    new Response("access denied", {
      status: 403,
      headers: {
        "Content-Type": "text/plain",
        "Set-Cookie": "must-not-leave-relay=1",
      },
    }),
  );

  assert.equal(response.status, 403);
  assert.equal(await response.text(), "access denied");
  assert.equal(response.headers.get("Set-Cookie"), null);
  assert.equal(relayContract.upstreamOrigin, "https://www.tpex.org.tw");
  assert.equal(relayContract.upstreamTimeoutMilliseconds, 30_000);
  assert.equal(relayContract.maxUpstreamBodyBytes, 16 * 1024 * 1024);
});

test("rejects an upstream body declared above the fixed response limit", async () => {
  const response = await handleRequest(request(), ENVIRONMENT, async () =>
    new Response("{}", {
      status: 200,
      headers: {
        "Content-Length": String(relayContract.maxUpstreamBodyBytes + 1),
        "Content-Type": "application/json",
      },
    }),
  );

  assert.equal(response.status, 502);
  assert.deepEqual(await response.json(), { error: "tpex_upstream_body_too_large" });
});
